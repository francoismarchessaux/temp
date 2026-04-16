import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from dataclasses import dataclass
from typing import Optional

from utils import COV_METHODS, compute_covariance, eigen_decomposition, excel_to_date
from pyJade.pearl import pearl_service


@dataclass
class PCAResult:
    eigvecs: pd.DataFrame
    eigvals: np.ndarray
    evr: np.ndarray
    cum_evr: np.ndarray
    loadings: pd.DataFrame
    n_components: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    stability_log: dict = None


class StaticPCA:
    """ Single PCA fitted once on a fixed historical window """
    def __init__(
        self,
        n_components: int,
        cov_method: str,
        ewma_halflife: int = None,
        ewma_lambda: float = None,
    ):
        self.n_components = n_components
        self.cov_method = cov_method
        self.ewma_halflife = ewma_halflife
        self.ewma_lambda = ewma_lambda
        self.result_: Optional[PCAResult] = None

    def fit(self, Xc: pd.DataFrame) -> None:
        cov = compute_covariance(
            df=Xc,
            method=self.cov_method,
            ewma_halflife=self.ewma_halflife,
            ewma_lambda=self.ewma_lambda
        )
        eigvecs, eigvals, evr, cum_evr, loadings, n = eigen_decomposition(
            df=Xc,
            cov=cov,
            n_components=self.n_components,
            target_var=None,
            auto_select=False
        )

        self.result_ = PCAResult(
            eigvecs=eigvecs,
            eigvals=eigvals,
            evr=evr,
            cum_evr=cum_evr,
            loadings=loadings,
            n_components=n,
            window_start=Xc.index[0],
            window_end=Xc.index[-1]
        )


class RollingPCA:
    def __init__(
        self,
        window: str = "6M",
        step: str = "1D",
        n_components: int = 8,
        cov_method: str = "ewma",
        ewma_halflife: int = 11,
        ewma_lambda: float = None,
    ):
        self.window = window
        self.step = step
        self.n_components = n_components
        self.cov_method = cov_method
        self.ewma_halflife = ewma_halflife
        self.ewma_lambda = ewma_lambda

        self.windows_: list[PCAResult] = []

    @staticmethod
    def _align_to_reference(
        new_loadings: np.ndarray,
        ref_loadings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Align new PCs to a reference set via two steps:

        1. Permutation — match each new PC to the reference PC it tracks most
           closely, using Hungarian assignment on the absolute cosine similarity
           matrix. This handles PC order swaps that occur when two eigenvalues
           become numerically close (e.g., PC2 and PC3 during a regime shift).

        2. Sign — flip each permuted PC whose dot product with its reference is
           negative, so the orientation is preserved across windows.

        Loadings coming from `numpy.linalg.eigh` are orthonormal, so the raw
        matrix product (new.T @ ref) is directly a cosine similarity matrix.

        Returns the aligned loadings, the permutation array (perm[j] = index in
        new_loadings that now occupies output position j), the sign vector, and
        a diagnostics log.
        """
        k = new_loadings.shape[1]

        # Cosine similarity matrix: rows = new PCs, cols = reference PCs
        sim = new_loadings.T @ ref_loadings

        # Hungarian on -|sim| → maximise absolute similarity (sign-agnostic match).
        # linear_sum_assignment returns row_ind = [0..k-1] and col_ind such that
        # new PC i is assigned to ref PC col_ind[i]. We want the inverse: which
        # new PC should sit in reference position j → perm[col_ind[i]] = i.
        row_ind, col_ind = linear_sum_assignment(-np.abs(sim))
        perm = np.empty(k, dtype=int)
        perm[col_ind] = row_ind

        # Apply permutation, then resolve sign per column
        permuted = new_loadings[:, perm]
        signs = np.where(np.sum(permuted * ref_loadings, axis=0) < 0, -1.0, 1.0)
        aligned = permuted * signs

        # Diagnostics
        cos_sims = np.sum(aligned * ref_loadings, axis=0)
        log = {
            "n_swaps": int(np.sum(perm != np.arange(k))),
            "n_flips": int(np.sum(signs < 0)),
            "cosine_similarity": cos_sims,
            "permutation": perm.tolist(),
        }

        return aligned, perm, signs, log

    @property
    def evr_history(self) -> Optional[pd.DataFrame]:
        if not self.windows_:
            return None

        pc_history = {}
        dates = [res.window_start for res in self.windows_]
        for pc in range(self.n_components):
            pc_history[f"PC{pc+1}"] = [res.evr[pc] for res in self.windows_]

        history = pd.DataFrame(pc_history, index=dates)
        history["All"] = history.sum(axis=1)

        return history * 100

    @property
    def stability_history(self) -> Optional[pd.DataFrame]:
        """Per-window PC-to-reference cosine similarity (1.0 = perfect tracking)."""
        if not self.windows_ or self.windows_[0].stability_log is None and len(self.windows_) == 1:
            return None

        rows, dates = [], []
        for res in self.windows_:
            if res.stability_log is None:
                continue
            rows.append(res.stability_log["cosine_similarity"])
            dates.append(res.window_start)

        if not rows:
            return None

        return pd.DataFrame(
            rows,
            index=dates,
            columns=[f"PC{i+1}" for i in range(self.n_components)],
        )

    def fit(self, X: pd.DataFrame) -> None:
        # Reset if model has already been fitted
        self.windows_.clear()

        static_model = StaticPCA(
            n_components=self.n_components,
            cov_method=self.cov_method,
            ewma_halflife=self.ewma_halflife,
            ewma_lambda=self.ewma_lambda,
        )

        # Generate rolling windows
        start_date = X.index[0]
        while True:
            end_date = excel_to_date(pearl_service().ARM_ADDPERIOD(start_date, self.window))
            if end_date > X.index[-1]:
                break

            # Slice data for current window
            X_win = X[start_date:end_date]

            # Fit model
            static_model.fit(X_win)
            res = static_model.result_

            # Cross-window stabilisation against the previous window's loadings.
            # Applied only from the second window onwards; the first window sets
            # the reference orientation for the whole history.
            if self.windows_:
                ref_loadings = self.windows_[-1].loadings.values
                aligned, perm, signs, log = self._align_to_reference(
                    new_loadings=res.loadings.values,
                    ref_loadings=ref_loadings,
                )

                # Overwrite loadings with the aligned matrix
                res.loadings = pd.DataFrame(
                    aligned,
                    index=res.loadings.index,
                    columns=res.loadings.columns,
                )

                # Permute the first n_components of eigvals/evr so they stay
                # positionally consistent with the aligned loadings. Signs do
                # not affect eigenvalues (they are scalars tied to a direction,
                # not the direction's orientation). Values beyond n_components
                # retain their original descending-magnitude order.
                k = self.n_components
                res.eigvals = np.concatenate([res.eigvals[:k][perm], res.eigvals[k:]])
                res.evr = np.concatenate([res.evr[:k][perm], res.evr[k:]])
                res.cum_evr = np.cumsum(res.evr)
                res.stability_log = log

            self.windows_.append(res)

            start_date = excel_to_date(pearl_service().ARM_ADDPERIOD(start_date, self.step))
