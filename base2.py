from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Optional plotting (you can remove if you want pure core)
import matplotlib.pyplot as plt


# =========================
# Helpers: parsing nodes and building a surface grid
# =========================

_TENOR_RE = re.compile(r"^(\d+)([DWMY])$", re.IGNORECASE)
_NODE_RE = re.compile(r"^(\d+[DWMY])(\d+[DWMY])$", re.IGNORECASE)


def _tenor_to_months(s: str) -> float:
    """
    Convert a tenor string like '2W', '1M', '5Y' to a numeric measure (months).
    Used for sorting expiry/tenor axes in plots.
    """
    s = str(s).upper()
    m = _TENOR_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse tenor '{s}'. Expected formats like 1W, 3M, 2Y.")
    n = int(m.group(1))
    u = m.group(2)
    if u == "D":
        return n / 30.0
    if u == "W":
        return (7.0 * n) / 30.0
    if u == "M":
        return float(n)
    if u == "Y":
        return 12.0 * n
    raise ValueError(f"Unknown unit '{u}' in tenor '{s}'.")


def node_to_expiry_tenor(node: str) -> Tuple[str, str]:
    """
    Parse a node label like '1M1Y' into ('1M', '1Y').
    """
    s = str(node).upper().replace(" ", "")
    m = _NODE_RE.match(s)
    if not m:
        raise ValueError(
            f"Cannot parse node '{node}'. Expected formats like '1M1Y', '3M10Y', '1Y30Y'."
        )
    return m.group(1), m.group(2)


def pivot_nodes_to_surface(
    values: Union[pd.Series, np.ndarray],
    nodes: Sequence[str],
) -> pd.DataFrame:
    """
    Convert a vector indexed by node labels into a 2D DataFrame (expiry x tenor).
    This enables surface heatmaps for moves/loadings/residuals.
    """
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or len(values) != len(nodes):
            raise ValueError("values must be a 1D array aligned with nodes.")
        s = pd.Series(values, index=pd.Index(nodes, name="node"))
    else:
        s = values.copy()
        s.index = pd.Index(nodes, name="node")

    exp_ten = [node_to_expiry_tenor(n) for n in s.index]
    expiry = [e for e, _ in exp_ten]
    tenor = [t for _, t in exp_ten]

    df = pd.DataFrame({"expiry": expiry, "tenor": tenor, "value": s.values})
    # Sort axes numerically
    expiry_order = sorted(df["expiry"].unique(), key=_tenor_to_months)
    tenor_order = sorted(df["tenor"].unique(), key=_tenor_to_months)

    surf = (
        df.pivot(index="expiry", columns="tenor", values="value")
        .reindex(index=expiry_order, columns=tenor_order)
    )
    return surf


def _safe_datetime_index(cols: pd.Index) -> pd.DatetimeIndex:
    """
    Convert dataframe columns to DatetimeIndex if possible.
    """
    if isinstance(cols, pd.DatetimeIndex):
        return cols
    try:
        return pd.to_datetime(cols)
    except Exception as e:
        raise ValueError("Columns must be parseable as dates (DatetimeIndex or date-like strings).") from e


# =========================
# PCA Results + Metrics
# =========================

@dataclass
class ReconstructionMetrics:
    rmse_global: float
    mae_global: float
    frob_rel: float
    r2: float
    rmse_by_node: pd.Series
    rmse_by_date: pd.Series

    def top_dates(self, n: int = 10) -> pd.Series:
        return self.rmse_by_date.sort_values(ascending=False).head(n)

    def top_nodes(self, n: int = 10) -> pd.Series:
        return self.rmse_by_node.sort_values(ascending=False).head(n)


@dataclass
class PCAResult:
    # Core PCA artifacts
    cov_matrix: Optional[pd.DataFrame] = None
    eigvals: Optional[np.ndarray] = None
    eigvecs: Optional[pd.DataFrame] = None  # full eigenvectors (N x N), columns in PC order
    explained_var_ratios: Optional[np.ndarray] = None

    loadings: Optional[pd.DataFrame] = None  # (N x k)
    scores: Optional[pd.DataFrame] = None    # (T x k)

    # Data metadata
    nodes: Optional[pd.Index] = None
    dates: Optional[pd.DatetimeIndex] = None
    mean_: Optional[pd.Series] = None  # mean removed (per node)
    standardized: bool = False
    std_: Optional[pd.Series] = None   # std used if standardized

    # Reconstruction / diagnostics
    reconstruction_k: Optional[int] = None
    X_hat: Optional[pd.DataFrame] = None     # (T x N)
    residuals: Optional[pd.DataFrame] = None # (T x N)
    metrics: Optional[ReconstructionMetrics] = None

    # Rolling context metadata (optional)
    window_start: Optional[pd.Timestamp] = None
    window_end: Optional[pd.Timestamp] = None


# =========================
# Static PCA Engine
# =========================

SignRule = Literal["sum_positive", "max_abs_positive", "anchor_positive", "align_previous"]


class StaticPCA:
    """
    PCA for swaption vol surface moves (or any panel of node moves).

    Expected input format:
      - df_nodes_by_dates: rows = nodes (e.g., '1M1Y'), columns = dates, values = daily changes (demeaned).
    Internally uses X = df.T: rows = dates, cols = nodes.
    """

    def __init__(
        self,
        n_components: int = 3,
        cov_estimator: Optional[Any] = None,
        standardize: bool = False,
        sign_rule: SignRule = "sum_positive",
        sign_anchor_node: Optional[str] = None,
    ):
        self.n_components = int(n_components)
        if self.n_components <= 0:
            raise ValueError("n_components must be >= 1")
        self.cov_estimator = cov_estimator
        self.standardize = bool(standardize)
        self.sign_rule = sign_rule
        self.sign_anchor_node = sign_anchor_node

        self.result_: Optional[PCAResult] = None

    # ---------- Data adapters ----------

    @staticmethod
    def _extract_df(data: Union[pd.DataFrame, Any]) -> pd.DataFrame:
        """
        Best-effort adapter:
          - if data is DataFrame: use it
          - else try common MarketData access patterns
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()

        # Try common patterns: data.df, data.data, data.get_df(), data.to_df(), data.surface_df, etc.
        for attr in ["df", "data", "surface", "surface_df", "vol_df", "panel"]:
            if hasattr(data, attr):
                obj = getattr(data, attr)
                if isinstance(obj, pd.DataFrame):
                    return obj.copy()

        for method in ["get_df", "to_df", "get_surface_df", "get_panel", "get_data"]:
            if hasattr(data, method):
                obj = getattr(data, method)()
                if isinstance(obj, pd.DataFrame):
                    return obj.copy()

        raise TypeError(
            "Unsupported data input. Provide a DataFrame (nodes x dates) or a MarketData-like object exposing it."
        )

    # ---------- Covariance estimation ----------

    def _compute_cov(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        X: (T x N) dates x nodes
        Returns covariance (N x N) as DataFrame
        """
        if self.cov_estimator is None:
            # sample covariance
            cov = np.cov(X.values, rowvar=False, ddof=1)
            return pd.DataFrame(cov, index=X.columns, columns=X.columns)

        # Try common estimator APIs
        if hasattr(self.cov_estimator, "compute_cov"):
            cov = self.cov_estimator.compute_cov(X)
        elif callable(self.cov_estimator):
            cov = self.cov_estimator(X)
        else:
            raise TypeError("cov_estimator must be None, callable, or expose compute_cov(X).")

        if isinstance(cov, np.ndarray):
            cov = pd.DataFrame(cov, index=X.columns, columns=X.columns)
        if not isinstance(cov, pd.DataFrame):
            raise TypeError("Covariance estimator must return a pandas DataFrame or numpy array.")

        # Ensure symmetry and alignment
        cov = cov.reindex(index=X.columns, columns=X.columns)
        cov = 0.5 * (cov + cov.T)
        return cov

    # ---------- PCA core ----------

    @staticmethod
    def _eigh_sorted(cov: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Symmetric eigen-decomposition + descending sort.
        Returns (eigvals_desc, eigvecs_desc)
        """
        w, v = np.linalg.eigh(cov.values)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:, idx]
        return w, v

    def _normalize_signs(
        self,
        loadings: pd.DataFrame,
        scores: pd.DataFrame,
        prev_loadings: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Enforce deterministic sign conventions for interpretability and stability.
        For rolling PCA, you typically pass prev_loadings to align signs to the previous window.
        """
        L = loadings.copy()
        S = scores.copy()

        for j, col in enumerate(L.columns):
            vec = L[col].values

            flip = False
            if self.sign_rule == "sum_positive":
                flip = (np.nansum(vec) < 0.0)

            elif self.sign_rule == "max_abs_positive":
                i = int(np.nanargmax(np.abs(vec)))
                flip = (vec[i] < 0.0)

            elif self.sign_rule == "anchor_positive":
                if self.sign_anchor_node is None:
                    raise ValueError("sign_anchor_node must be provided when sign_rule='anchor_positive'.")
                if self.sign_anchor_node not in L.index:
                    raise ValueError(f"Anchor node '{self.sign_anchor_node}' not found in loadings index.")
                flip = (float(L.loc[self.sign_anchor_node, col]) < 0.0)

            elif self.sign_rule == "align_previous":
                if prev_loadings is None:
                    # fallback deterministic rule
                    flip = (np.nansum(vec) < 0.0)
                else:
                    prev_vec = prev_loadings[col].values
                    dot = float(np.dot(vec, prev_vec))
                    flip = (dot < 0.0)

            else:
                raise ValueError(f"Unknown sign_rule: {self.sign_rule}")

            if flip:
                L[col] = -L[col]
                S[col] = -S[col]

        return L, S

    @staticmethod
    def _standardize_if_needed(X: pd.DataFrame, standardize: bool) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Returns:
          X_centered_or_standardized, mean, std (std None if not standardized)
        """
        mean_ = X.mean(axis=0)
        Xc = X - mean_
        if not standardize:
            return Xc, mean_, None

        std_ = Xc.std(axis=0, ddof=1).replace(0.0, np.nan)
        Xz = Xc.div(std_, axis=1).fillna(0.0)
        return Xz, mean_, std_

    def fit(
        self,
        data: Union[pd.DataFrame, Any],
        n_components: Optional[int] = None,
        prev_loadings_for_sign: Optional[pd.DataFrame] = None,
    ) -> PCAResult:
        """
        Fit static PCA on the full dataset (or provided slice).
        """
        df = self._extract_df(data)

        # df is nodes x dates (per your spec)
        nodes = df.index
        dates = _safe_datetime_index(df.columns)
        df.columns = dates

        # X = dates x nodes
        X = df.T
        if X.isna().any().any():
            raise ValueError("Input contains NaNs. You said data is clean; please ensure no missing values.")

        # Center / standardize as configured (even if your MarketData already demeaned, this is safe)
        X_proc, mean_, std_ = self._standardize_if_needed(X, self.standardize)

        cov = self._compute_cov(X_proc)
        eigvals, eigvecs = self._eigh_sorted(cov)

        total = float(np.sum(eigvals))
        if total <= 0.0:
            raise ValueError("Sum of eigenvalues is non-positive. Check covariance estimation / input scaling.")

        evr = eigvals / total

        k = int(n_components) if n_components is not None else self.n_components
        k = min(k, X_proc.shape[1])

        V_k = eigvecs[:, :k]  # N x k
        # Loadings as eigenvectors (orthonormal in feature space)
        loadings = pd.DataFrame(V_k, index=cov.index, columns=[f"PC{i+1}" for i in range(k)])

        # Scores: T x k
        scores = pd.DataFrame(X_proc.values @ V_k, index=X_proc.index, columns=loadings.columns)

        # Sign normalization
        loadings, scores = self._normalize_signs(loadings, scores, prev_loadings=prev_loadings_for_sign)

        result = PCAResult(
            cov_matrix=cov,
            eigvals=eigvals,
            eigvecs=pd.DataFrame(eigvecs, index=cov.index, columns=[f"PC{i+1}" for i in range(eigvecs.shape[1])]),
            explained_var_ratios=evr,
            loadings=loadings,
            scores=scores,
            nodes=nodes,
            dates=dates,
            mean_=mean_,
            standardized=self.standardize,
            std_=std_,
        )
        self.result_ = result
        return result

    # ---------- Reconstruction / evaluation ----------

    def reconstruct(
        self,
        data: Union[pd.DataFrame, Any],
        k: Optional[int] = None,
        store: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reconstruct X using first k PCs.
        Returns (X_hat, residuals), both as DataFrames in shape (T x N).
        """
        if self.result_ is None or self.result_.loadings is None or self.result_.scores is None:
            raise RuntimeError("Fit the model before reconstruct().")

        df = self._extract_df(data)
        dates = _safe_datetime_index(df.columns)
        df.columns = dates
        X = df.T

        # Apply same preprocessing
        mean_ = self.result_.mean_
        if mean_ is None:
            mean_ = X.mean(axis=0)
        Xc = X - mean_

        if self.result_.standardized:
            std_ = self.result_.std_
            if std_ is None:
                std_ = Xc.std(axis=0, ddof=1).replace(0.0, np.nan)
            Xp = Xc.div(std_, axis=1).fillna(0.0)
        else:
            Xp = Xc

        k_use = int(k) if k is not None else self.result_.loadings.shape[1]
        k_use = min(k_use, self.result_.loadings.shape[1])

        L = self.result_.loadings.iloc[:, :k_use].values  # N x k
        # Project using loadings (eigenvectors) -> scores
        Z = Xp.values @ L                                  # T x k
        X_hat = Z @ L.T                                    # T x N

        X_hat_df = pd.DataFrame(X_hat, index=Xp.index, columns=Xp.columns)
        resid_df = Xp - X_hat_df

        if store:
            self.result_.reconstruction_k = k_use
            self.result_.X_hat = X_hat_df
            self.result_.residuals = resid_df

        return X_hat_df, resid_df

    @staticmethod
    def reconstruction_metrics(X: pd.DataFrame, X_hat: pd.DataFrame) -> ReconstructionMetrics:
        """
        Compute global + node/date RMSE/MAE, relative Frobenius error, and R^2.
        X, X_hat: (T x N) aligned
        """
        X, X_hat = X.align(X_hat, join="inner", axis=0)
        X, X_hat = X.align(X_hat, join="inner", axis=1)

        resid = (X - X_hat)
        mse_global = float(np.mean(np.square(resid.values)))
        rmse_global = float(np.sqrt(mse_global))
        mae_global = float(np.mean(np.abs(resid.values)))

        frob_res = float(np.linalg.norm(resid.values, ord="fro"))
        frob_x = float(np.linalg.norm(X.values, ord="fro"))
        frob_rel = frob_res / frob_x if frob_x > 0 else np.nan

        # R^2 relative to columnwise mean baseline
        X_mean = X.mean(axis=0)
        sse = float(np.sum(np.square(resid.values)))
        sst = float(np.sum(np.square((X - X_mean).values)))
        r2 = 1.0 - (sse / sst) if sst > 0 else np.nan

        rmse_by_node = pd.Series(
            np.sqrt(np.mean(np.square(resid.values), axis=0)),
            index=X.columns,
            name="rmse_by_node",
        )
        rmse_by_date = pd.Series(
            np.sqrt(np.mean(np.square(resid.values), axis=1)),
            index=X.index,
            name="rmse_by_date",
        )

        return ReconstructionMetrics(
            rmse_global=rmse_global,
            mae_global=mae_global,
            frob_rel=frob_rel,
            r2=r2,
            rmse_by_node=rmse_by_node,
            rmse_by_date=rmse_by_date,
        )

    def evaluate(
        self,
        data: Union[pd.DataFrame, Any],
        k: Optional[int] = None,
        store: bool = True,
    ) -> ReconstructionMetrics:
        """
        Reconstruct and compute reconstruction metrics.
        """
        df = self._extract_df(data)
        dates = _safe_datetime_index(df.columns)
        df.columns = dates
        X = df.T

        X_hat, resid = self.reconstruct(data, k=k, store=store)

        # IMPORTANT: metrics should be computed in the same transformed space used by reconstruct()
        mean_ = self.result_.mean_
        Xc = X - mean_
        if self.result_.standardized:
            Xp = Xc.div(self.result_.std_, axis=1).fillna(0.0)
        else:
            Xp = Xc

        metrics = self.reconstruction_metrics(Xp, X_hat)

        if store and self.result_ is not None:
            self.result_.metrics = metrics

        return metrics

    # ---------- Plotting helpers ----------

    def plot_surface_move_comparison(
        self,
        data: Union[pd.DataFrame, Any],
        date: Union[str, pd.Timestamp],
        k: Optional[int] = None,
        title_prefix: str = "",
        figsize: Tuple[int, int] = (15, 4),
    ) -> None:
        """
        Heatmaps: actual move vs reconstructed move vs residual on a specific date.
        Assumes node labels are like '1M1Y' etc.
        """
        if self.result_ is None:
            raise RuntimeError("Fit the model first.")

        df = self._extract_df(data)
        df.columns = _safe_datetime_index(df.columns)
        dt = pd.Timestamp(date)

        if dt not in df.columns:
            raise ValueError(f"Date {dt} not found in data columns.")

        # Ensure reconstruction exists for the desired k
        self.reconstruct(data, k=k, store=True)

        if self.result_.X_hat is None or self.result_.residuals is None:
            raise RuntimeError("Reconstruction failed unexpectedly.")

        # Build vectors for this date (work in transformed space: demeaned/standardized)
        X = df.T
        Xc = X - self.result_.mean_
        if self.result_.standardized:
            Xp = Xc.div(self.result_.std_, axis=1).fillna(0.0)
        else:
            Xp = Xc

        actual_vec = Xp.loc[dt]
        recon_vec = self.result_.X_hat.loc[dt]
        resid_vec = self.result_.residuals.loc[dt]

        nodes = list(actual_vec.index)

        actual_surf = pivot_nodes_to_surface(actual_vec, nodes)
        recon_surf = pivot_nodes_to_surface(recon_vec, nodes)
        resid_surf = pivot_nodes_to_surface(resid_vec, nodes)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, surf, ttl in zip(
            axes,
            [actual_surf, recon_surf, resid_surf],
            ["Actual move", "Reconstructed move", "Residual (Actual - Recon)"],
        ):
            im = ax.imshow(surf.values, aspect="auto")
            ax.set_title((title_prefix + " " + ttl).strip())
            ax.set_yticks(range(len(surf.index)))
            ax.set_yticklabels(list(surf.index))
            ax.set_xticks(range(len(surf.columns)))
            ax.set_xticklabels(list(surf.columns), rotation=45, ha="right")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def plot_explained_variance(
        self,
        max_pcs: int = 10,
        figsize: Tuple[int, int] = (10, 4),
    ) -> None:
        if self.result_ is None or self.result_.explained_var_ratios is None:
            raise RuntimeError("Fit first.")
        evr = self.result_.explained_var_ratios
        m = min(max_pcs, len(evr))
        xs = np.arange(1, m + 1)

        plt.figure(figsize=figsize)
        plt.plot(xs, evr[:m], marker="o")
        plt.title("Explained variance ratios (scree)")
        plt.xlabel("Principal component")
        plt.ylabel("Explained variance ratio")
        plt.xticks(xs)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=figsize)
        plt.plot(xs, np.cumsum(evr[:m]), marker="o")
        plt.title("Cumulative explained variance")
        plt.xlabel("Principal component")
        plt.ylabel("Cumulative explained variance")
        plt.xticks(xs)
        plt.grid(True)
        plt.show()


# =========================
# Rolling PCA Engine + Alignment + Stability Metrics
# =========================

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def _principal_angles_degrees(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Principal angles between subspaces spanned by columns of U and V.
    U, V should be (N x k) with orthonormal columns.
    Returns angles in degrees (size = k).
    """
    # Orthonormalize (safe)
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    M = Qu.T @ Qv
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.arccos(s) * (180.0 / np.pi)
    return angles


def _best_permutation_by_similarity(sim: np.ndarray) -> List[int]:
    """
    Find permutation p maximizing sum(sim[i, p[i]]).
    Brute-force for small k; greedy fallback for larger.
    """
    k = sim.shape[0]
    if k <= 8:
        best_score = -np.inf
        best_p = list(range(k))
        for p in itertools.permutations(range(k)):
            score = sum(sim[i, p[i]] for i in range(k))
            if score > best_score:
                best_score = score
                best_p = list(p)
        return best_p

    # Greedy fallback
    used = set()
    p = [-1] * k
    for i in range(k):
        j = int(np.argmax(sim[i, :]))
        while j in used:
            sim[i, j] = -np.inf
            j = int(np.argmax(sim[i, :]))
        p[i] = j
        used.add(j)
    return p


@dataclass
class RollingStability:
    explained_var: pd.DataFrame                    # index = window_end, columns = PC1..PCk
    cosine_similarity: pd.DataFrame                # index = window_end (from 2nd window), columns = PC1..PCk
    turnover: pd.DataFrame                         # 1 - abs(cos) after alignment
    subspace_angles_deg: pd.DataFrame              # index = window_end (from 2nd window), columns = angle1..anglek
    hedge_node_loading_stability: Optional[pd.DataFrame] = None  # optional


@dataclass
class RollingPCAResult:
    windows: List[PCAResult] = field(default_factory=list)
    stability: Optional[RollingStability] = None


class RollingPCA:
    """
    Rolling window PCA with alignment and stability metrics.

    Input df format: nodes x dates (like your MarketData output).
    """

    def __init__(
        self,
        n_components: int = 3,
        window_size: int = 252,
        step: int = 1,
        cov_estimator: Optional[Any] = None,
        standardize: bool = False,
        # alignment choices
        align_components: bool = True,
        sign_rule_for_rolling: SignRule = "align_previous",
        sign_anchor_node: Optional[str] = None,
    ):
        self.n_components = int(n_components)
        self.window_size = int(window_size)
        self.step = int(step)
        if self.window_size <= 2:
            raise ValueError("window_size must be > 2")
        if self.step <= 0:
            raise ValueError("step must be >= 1")

        self.cov_estimator = cov_estimator
        self.standardize = bool(standardize)

        self.align_components = bool(align_components)
        self.sign_rule_for_rolling = sign_rule_for_rolling
        self.sign_anchor_node = sign_anchor_node

        self.result_: Optional[RollingPCAResult] = None

    @staticmethod
    def _extract_df(data: Union[pd.DataFrame, Any]) -> pd.DataFrame:
        return StaticPCA._extract_df(data)

    def fit(
        self,
        data: Union[pd.DataFrame, Any],
        hedge_nodes: Optional[Sequence[str]] = None,
    ) -> RollingPCAResult:
        df = self._extract_df(data)
        df.columns = _safe_datetime_index(df.columns)
        df = df.sort_index(axis=1)  # sort dates

        dates = df.columns
        n_dates = len(dates)
        if n_dates < self.window_size:
            raise ValueError(f"Not enough dates ({n_dates}) for window_size={self.window_size}.")

        # rolling windows endpoints
        window_ends = list(range(self.window_size - 1, n_dates, self.step))

        windows: List[PCAResult] = []

        prev_loadings: Optional[pd.DataFrame] = None

        for end_idx in window_ends:
            start_idx = end_idx - self.window_size + 1
            cols = dates[start_idx : end_idx + 1]
            df_w = df.loc[:, cols]

            pca = StaticPCA(
                n_components=self.n_components,
                cov_estimator=self.cov_estimator,
                standardize=self.standardize,
                sign_rule=self.sign_rule_for_rolling,
                sign_anchor_node=self.sign_anchor_node,
            )

            res = pca.fit(df_w, prev_loadings_for_sign=prev_loadings)
            res.window_start = pd.Timestamp(cols[0])
            res.window_end = pd.Timestamp(cols[-1])

            windows.append(res)
            prev_loadings = res.loadings

        # Component alignment (handle swapping + sign consistently)
        if self.align_components:
            windows = self._align_windows(windows)

        # Compute stability metrics
        stability = self._compute_stability(windows, hedge_nodes=hedge_nodes)

        out = RollingPCAResult(windows=windows, stability=stability)
        self.result_ = out
        return out

    def _align_windows(self, windows: List[PCAResult]) -> List[PCAResult]:
        """
        Align components across windows (permute + sign).
        Alignment is done sequentially: each window aligns to previous window.
        """
        if not windows:
            return windows

        aligned = [windows[0]]
        for i in range(1, len(windows)):
            prev = aligned[-1]
            curr = windows[i]

            if prev.loadings is None or curr.loadings is None or curr.scores is None:
                aligned.append(curr)
                continue

            L_prev = prev.loadings.values  # N x k
            L_curr = curr.loadings.values  # N x k

            k = L_prev.shape[1]
            # Similarity matrix based on absolute cosine similarity
            sim = np.zeros((k, k), dtype=float)
            for a in range(k):
                for b in range(k):
                    sim[a, b] = abs(_cosine_similarity(L_prev[:, a], L_curr[:, b]))

            perm = _best_permutation_by_similarity(sim)

            # Permute curr components into prev order
            cols = list(prev.loadings.columns)
            curr_load = curr.loadings.copy()
            curr_score = curr.scores.copy()

            curr_load = curr_load.iloc[:, perm]
            curr_load.columns = cols

            curr_score = curr_score.iloc[:, perm]
            curr_score.columns = cols

            # Align sign to previous
            for c in cols:
                dot = float(np.dot(curr_load[c].values, prev.loadings[c].values))
                if dot < 0:
                    curr_load[c] = -curr_load[c]
                    curr_score[c] = -curr_score[c]

            # Update eigenvalues / evr ordering consistently if present
            if curr.eigvals is not None and curr.explained_var_ratios is not None:
                # eigvals and evr are full length; we only reliably permute first k comps
                # Build a new order for first k and keep remaining as-is
                full_n = len(curr.eigvals)
                first_k = perm + list(range(k, full_n))
                curr.eigvals = curr.eigvals[first_k]
                curr.explained_var_ratios = curr.explained_var_ratios[first_k]

            if curr.eigvecs is not None:
                # eigvecs is N x N with PC columns; permute first k consistently
                full_n = curr.eigvecs.shape[1]
                first_k = perm + list(range(k, full_n))
                curr.eigvecs = curr.eigvecs.iloc[:, first_k]
                curr.eigvecs.columns = [f"PC{i+1}" for i in range(full_n)]

            curr.loadings = curr_load
            curr.scores = curr_score

            aligned.append(curr)

        return aligned

    def _compute_stability(
        self,
        windows: List[PCAResult],
        hedge_nodes: Optional[Sequence[str]] = None,
    ) -> RollingStability:
        if not windows or windows[0].loadings is None:
            raise ValueError("No rolling PCA windows to analyze.")

        k = windows[0].loadings.shape[1]
        pcs = [f"PC{i+1}" for i in range(k)]
        idx_all = [w.window_end for w in windows]

        # Explained variance ratios for first k
        ev_mat = []
        for w in windows:
            if w.explained_var_ratios is None:
                raise ValueError("Missing explained_var_ratios in a window.")
            ev_mat.append(w.explained_var_ratios[:k])
        explained_var = pd.DataFrame(ev_mat, index=pd.Index(idx_all, name="window_end"), columns=pcs)

        # Cosine similarity (component-wise) + turnover between consecutive windows
        cos_rows = []
        turn_rows = []
        angle_rows = []
        idx_cons = []

        for i in range(1, len(windows)):
            w_prev = windows[i - 1]
            w_curr = windows[i]
            if w_prev.loadings is None or w_curr.loadings is None:
                continue

            idx_cons.append(w_curr.window_end)

            cos_vals = []
            turn_vals = []
            for pc in pcs:
                c = _cosine_similarity(w_prev.loadings[pc].values, w_curr.loadings[pc].values)
                cos_vals.append(c)
                turn_vals.append(1.0 - abs(c) if not np.isnan(c) else np.nan)

            cos_rows.append(cos_vals)
            turn_rows.append(turn_vals)

            # Subspace angles between spans of first k PCs
            U = w_prev.loadings.values[:, :k]
            V = w_curr.loadings.values[:, :k]
            ang = _principal_angles_degrees(U, V)
            angle_rows.append(list(ang))

        cosine_similarity = pd.DataFrame(cos_rows, index=pd.Index(idx_cons, name="window_end"), columns=pcs)
        turnover = pd.DataFrame(turn_rows, index=pd.Index(idx_cons, name="window_end"), columns=pcs)
        subspace_angles_deg = pd.DataFrame(
            angle_rows,
            index=pd.Index(idx_cons, name="window_end"),
            columns=[f"angle{i+1}" for i in range(k)],
        )

        hedge_stab = None
        if hedge_nodes is not None:
            hedge_nodes = list(hedge_nodes)
            # For each hedge node + PC, track loading across windows and summarize volatility
            rows = []
            for pc in pcs:
                series_by_node = {}
                for hn in hedge_nodes:
                    vals = []
                    for w in windows:
                        if w.loadings is None:
                            vals.append(np.nan)
                        else:
                            vals.append(float(w.loadings.loc[hn, pc]) if hn in w.loadings.index else np.nan)
                    series_by_node[hn] = pd.Series(vals, index=pd.Index(idx_all, name="window_end"))
                # Summaries per node
                for hn, ser in series_by_node.items():
                    rows.append(
                        {
                            "pc": pc,
                            "hedge_node": hn,
                            "mean_loading": float(np.nanmean(ser.values)),
                            "std_loading": float(np.nanstd(ser.values, ddof=1)),
                            "mean_abs_loading": float(np.nanmean(np.abs(ser.values))),
                            "cv_abs": float(np.nanstd(np.abs(ser.values), ddof=1) / np.nanmean(np.abs(ser.values)))
                            if np.nanmean(np.abs(ser.values)) > 0 else np.nan,
                        }
                    )
            hedge_stab = pd.DataFrame(rows)

        return RollingStability(
            explained_var=explained_var,
            cosine_similarity=cosine_similarity,
            turnover=turnover,
            subspace_angles_deg=subspace_angles_deg,
            hedge_node_loading_stability=hedge_stab,
        )

    # ---------- Convenience plots ----------

    def plot_explained_variance_stability(self, figsize: Tuple[int, int] = (10, 4)) -> None:
        if self.result_ is None or self.result_.stability is None:
            raise RuntimeError("Fit first.")
        ev = self.result_.stability.explained_var
        plt.figure(figsize=figsize)
        for c in ev.columns:
            plt.plot(ev.index, ev[c], marker="o", linewidth=1)
        plt.title("Rolling explained variance ratios")
        plt.xlabel("Window end")
        plt.ylabel("Explained variance ratio")
        plt.grid(True)
        plt.legend(ev.columns)
        plt.tight_layout()
        plt.show()

    def plot_loading_cosine_similarity(self, figsize: Tuple[int, int] = (10, 4)) -> None:
        if self.result_ is None or self.result_.stability is None:
            raise RuntimeError("Fit first.")
        cs = self.result_.stability.cosine_similarity
        plt.figure(figsize=figsize)
        for c in cs.columns:
            plt.plot(cs.index, cs[c], marker="o", linewidth=1)
        plt.title("Rolling loading cosine similarity (consecutive windows)")
        plt.xlabel("Window end")
        plt.ylabel("Cosine similarity")
        plt.grid(True)
        plt.legend(cs.columns)
        plt.tight_layout()
        plt.show()

    def plot_subspace_angles(self, figsize: Tuple[int, int] = (10, 4)) -> None:
        if self.result_ is None or self.result_.stability is None:
            raise RuntimeError("Fit first.")
        ang = self.result_.stability.subspace_angles_deg
        plt.figure(figsize=figsize)
        for c in ang.columns:
            plt.plot(ang.index, ang[c], marker="o", linewidth=1)
        plt.title("Rolling subspace principal angles (degrees)")
        plt.xlabel("Window end")
        plt.ylabel("Degrees")
        plt.grid(True)
        plt.legend(ang.columns)
        plt.tight_layout()
        plt.show()


# =========================
# Minimal usage example (adapt to your codebase)
# =========================
#
# df = market_data.df  # nodes x dates of demeaned daily changes
#
# # Static PCA
# pca = StaticPCA(n_components=3, cov_estimator=LedoitWolfEstimator(), sign_rule="sum_positive")
# res = pca.fit(df)
# metrics = pca.evaluate(df, k=3)
# print(metrics.rmse_global, metrics.r2)
# pca.plot_explained_variance(max_pcs=6)
# pca.plot_surface_move_comparison(df, date="2024-03-08", k=3)
#
# # Rolling PCA
# rpca = RollingPCA(n_components=3, window_size=252, step=5, cov_estimator=LedoitWolfEstimator())
# rres = rpca.fit(df, hedge_nodes=["1M2Y", "3M5Y", "1Y10Y"])
# rpca.plot_explained_variance_stability()
# rpca.plot_loading_cosine_similarity()
# rpca.plot_subspace_angles()
#
