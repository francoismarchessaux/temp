import numpy as np
import pandas as pd

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
        # FIX: removed auto_select_n and target_variance — n_components is now always fixed
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
        # FIX: auto_select=False — number of components is fixed, target_var is unused
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
        avg_windows: int = 4,
    ):
        # FIX: removed auto_select_n and target_variance — n_components is now always fixed
        self.window = window
        self.step = step
        self.n_components = n_components
        self.cov_method = cov_method
        self.ewma_halflife = ewma_halflife
        self.ewma_lambda = ewma_lambda
        self.avg_windows = avg_windows

        self.windows_: list[PCAResult] = []
        self.smoothed_loadings_: list[pd.DataFrame] = []  # avg_windows-averaged loadings per step

    @staticmethod
    def _align_signs(new_loadings: np.ndarray, ref_loadings: np.ndarray) -> np.ndarray:
        """
        Flip each PC column of new_loadings to match the sign orientation of ref_loadings.
        For each PC i, if dot(new_i, ref_i) < 0 the vectors point in opposite directions
        and the column is flipped. This prevents PC sign flipping between consecutive
        windows, which is a direct source of PnL error in rolling PCA.
        """
        aligned = new_loadings.copy()
        for i in range(new_loadings.shape[1]):
            if np.dot(new_loadings[:, i], ref_loadings[:, i]) < 0:
                aligned[:, i] *= -1
        return aligned

    @property
    def evr_history(self) -> Optional[pd.DataFrame]:
        if not self.windows_:
            return None

        # FIX: n_components is fixed so res.evr[pc] is always valid — no more IndexError
        pc_history = {}
        dates = [res.window_start for res in self.windows_]
        for pc in range(self.n_components):
            pc_history[f"PC{pc+1}"] = [res.evr[pc] for res in self.windows_]

        history = pd.DataFrame(pc_history, index=dates)
        history["All"] = history.sum(axis=1)

        return history * 100

    def fit(self, X: pd.DataFrame) -> None:
        # Reset if model has already been fitted
        self.windows_.clear()
        self.smoothed_loadings_.clear()

        # FIX: removed hardcoded auto_select_n=True and target_variance=0.95
        # that were silently overriding the user's parameters
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

            # FIX: cross-window sign alignment — align each PC column of the new window
            # to the previous window's orientation before storing. Without this, PC signs
            # can arbitrarily flip between windows, producing large spurious PnL errors.
            if self.windows_:
                ref_loadings = self.windows_[-1].loadings.values
                aligned = self._align_signs(res.loadings.values, ref_loadings)
                res.loadings = pd.DataFrame(
                    aligned, index=res.loadings.index, columns=res.loadings.columns
                )

            self.windows_.append(res)

            # FIX: implement avg_windows — average the last avg_windows aligned loadings
            # matrices to smooth out window-to-window noise in PC estimates.
            # smoothed_loadings_[i] corresponds to windows_[i].
            recent = self.windows_[-self.avg_windows:]
            avg_loadings = np.mean([r.loadings.values for r in recent], axis=0)
            self.smoothed_loadings_.append(
                pd.DataFrame(
                    avg_loadings, index=res.loadings.index, columns=res.loadings.columns
                )
            )

            start_date = excel_to_date(pearl_service().ARM_ADDPERIOD(start_date, self.step))
