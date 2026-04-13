from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Any

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf, OAS


CovEstimator = Literal["sample", "ledoit_wolf", "oas", "ewma"]
CenterMethod = Literal["mean", "ewma_mean", "zero"]


# ============================================================
# Utility functions
# ============================================================

def _ensure_dataframe(
    x: pd.DataFrame | np.ndarray,
    index: Optional[pd.Index] = None,
    columns: Optional[pd.Index] = None,
    name: str = "x",
) -> pd.DataFrame:
    """
    Convert input to DataFrame while preserving index/columns if possible.
    Assumes rows = time, columns = features.
    """
    if isinstance(x, pd.DataFrame):
        return x.copy()

    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a pd.DataFrame or np.ndarray, got {type(x)}")

    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {x.shape}")

    if index is None:
        index = pd.RangeIndex(x.shape[0], name="row")
    if columns is None:
        columns = pd.RangeIndex(x.shape[1], name="feature")

    return pd.DataFrame(x, index=index, columns=columns)


def _ensure_series(
    x: pd.Series | np.ndarray,
    index: Optional[pd.Index] = None,
    name: str = "x",
) -> pd.Series:
    """
    Convert input to Series while preserving index if possible.
    """
    if isinstance(x, pd.Series):
        return x.copy()

    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a pd.Series or np.ndarray, got {type(x)}")

    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")

    if index is None:
        index = pd.RangeIndex(len(x), name="feature")

    return pd.Series(x, index=index, name=name)


def _validate_no_missing(df: pd.DataFrame, name: str) -> None:
    if df.isnull().values.any():
        missing = int(df.isnull().sum().sum())
        raise ValueError(f"{name} contains {missing} missing values. Please clean/fill before fitting.")


def _validate_same_columns(a: pd.DataFrame, b: pd.DataFrame | pd.Series, a_name: str, b_name: str) -> None:
    if not a.columns.equals(b.index if isinstance(b, pd.Series) else b.columns):
        raise ValueError(
            f"{a_name} and {b_name} do not have matching feature indices.\n"
            f"{a_name}.columns[:5] = {list(a.columns[:5])}\n"
            f"{b_name}.index/columns[:5] = {list((b.index if isinstance(b, pd.Series) else b.columns)[:5])}"
        )


def _ewma_weights(n_obs: int, half_life: float) -> np.ndarray:
    """
    Returns normalized EWMA weights from oldest to newest.
    """
    if half_life <= 0:
        raise ValueError("half_life must be > 0")

    lam = np.exp(np.log(0.5) / half_life)
    # oldest to newest
    exponents = np.arange(n_obs - 1, -1, -1)
    w = (1.0 - lam) * (lam ** exponents)
    w = w / w.sum()
    return w


def _weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sum(X * w[:, None], axis=0)


def _weighted_cov(X: np.ndarray, w: np.ndarray, mean: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Weighted covariance with normalized weights.
    Rows = observations, cols = features.
    """
    if mean is None:
        mean = _weighted_mean(X, w)

    Xc = X - mean
    cov = (Xc * w[:, None]).T @ Xc
    return cov


def _make_psd(mat: np.ndarray, min_eig: float = 1e-12) -> np.ndarray:
    """
    Force symmetric PSD matrix by clipping eigenvalues.
    """
    mat = 0.5 * (mat + mat.T)
    evals, evecs = np.linalg.eigh(mat)
    evals = np.clip(evals, min_eig, None)
    return evecs @ np.diag(evals) @ evecs.T


def _diag_weight_matrix(weights: Optional[pd.Series], columns: pd.Index) -> np.ndarray:
    """
    Build diagonal metric matrix W in feature space.
    If weights is None, returns identity.
    """
    n = len(columns)
    if weights is None:
        return np.eye(n)

    weights = weights.reindex(columns)
    if weights.isnull().any():
        raise ValueError("weights has missing values after reindexing to feature columns.")
    if (weights <= 0).any():
        raise ValueError("weights must be strictly positive.")

    return np.diag(weights.values.astype(float))


def _sort_eigensystem(evals: np.ndarray, evecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


def _sign_align_loadings(loadings: pd.DataFrame, ref_loadings: pd.DataFrame) -> pd.DataFrame:
    """
    Align factor signs to a reference loading matrix.
    Both matrices must have same shape/index/columns.
    """
    aligned = loadings.copy()
    for col in aligned.columns:
        dot = np.dot(aligned[col].values, ref_loadings[col].values)
        if dot < 0:
            aligned[col] = -aligned[col]
    return aligned


# ============================================================
# Static PCA model
# ============================================================

@dataclass
class PCAConfig:
    n_components: int = 5
    cov_estimator: CovEstimator = "ledoit_wolf"
    center_method: CenterMethod = "mean"
    ewma_half_life: float = 20.0
    standardize: bool = False
    feature_weights: Optional[pd.Series] = None
    min_eigenvalue: float = 1e-10


@dataclass
class PCAResult:
    mean_: pd.Series
    scale_: pd.Series
    eigenvalues_: pd.Series
    explained_variance_ratio_: pd.Series
    loadings_: pd.DataFrame
    columns_: pd.Index
    config_: PCAConfig


class SurfacePCA:
    """
    PCA on market shocks X where:
      - rows = dates / observations
      - columns = surface buckets / features

    Supports:
      - sample covariance
      - Ledoit-Wolf
      - OAS
      - EWMA covariance

    Important:
    This implementation uses JOINT projection on the selected factor basis:
        c = (B' W B)^(-1) B' W x
    rather than projecting PC-by-PC independently.
    """

    def __init__(self, config: PCAConfig):
        self.config = config
        self.result_: Optional[PCAResult] = None

    # --------------------------------------------------------
    # Fit helpers
    # --------------------------------------------------------

    def _compute_center(self, X: pd.DataFrame) -> pd.Series:
        if self.config.center_method == "mean":
            return X.mean(axis=0)
        if self.config.center_method == "ewma_mean":
            w = _ewma_weights(len(X), self.config.ewma_half_life)
            return pd.Series(_weighted_mean(X.values, w), index=X.columns, name="mean")
        if self.config.center_method == "zero":
            return pd.Series(0.0, index=X.columns, name="mean")
        raise ValueError(f"Unsupported center_method={self.config.center_method}")

    def _compute_scale(self, X_centered: pd.DataFrame) -> pd.Series:
        if not self.config.standardize:
            return pd.Series(1.0, index=X_centered.columns, name="scale")

        # use standard deviation
        std = X_centered.std(axis=0, ddof=1)
        std = std.replace(0.0, 1.0)
        return std.rename("scale")

    def _estimate_covariance(self, X_scaled: pd.DataFrame) -> np.ndarray:
        Xv = X_scaled.values

        if self.config.cov_estimator == "sample":
            cov = np.cov(Xv, rowvar=False, ddof=1)

        elif self.config.cov_estimator == "ledoit_wolf":
            cov = LedoitWolf().fit(Xv).covariance_

        elif self.config.cov_estimator == "oas":
            cov = OAS().fit(Xv).covariance_

        elif self.config.cov_estimator == "ewma":
            w = _ewma_weights(len(X_scaled), self.config.ewma_half_life)
            cov = _weighted_cov(Xv, w=w, mean=np.zeros(Xv.shape[1]))

        else:
            raise ValueError(f"Unsupported cov_estimator={self.config.cov_estimator}")

        cov = _make_psd(cov, min_eig=self.config.min_eigenvalue)
        return cov

    # --------------------------------------------------------
    # Main API
    # --------------------------------------------------------

    def fit(self, X: pd.DataFrame | np.ndarray) -> "SurfacePCA":
        X = _ensure_dataframe(X, name="X")
        _validate_no_missing(X, "X")

        if self.config.n_components <= 0:
            raise ValueError("n_components must be > 0")
        if self.config.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed number of features")

        mean_ = self._compute_center(X)
        X_centered = X - mean_

        scale_ = self._compute_scale(X_centered)
        X_scaled = X_centered.divide(scale_, axis=1)

        cov = self._estimate_covariance(X_scaled)
        evals, evecs = np.linalg.eigh(cov)
        evals, evecs = _sort_eigensystem(evals, evecs)

        k = self.config.n_components
        evals_k = evals[:k]
        evecs_k = evecs[:, :k]

        # Loadings in scaled feature space
        pc_names = [f"PC{i+1}" for i in range(k)]
        loadings_scaled = pd.DataFrame(evecs_k, index=X.columns, columns=pc_names)

        # Convert loadings back to original feature scale for reconstruction / explain
        # x_scaled = D^{-1}(x - mean), so x ≈ mean + D B c
        loadings_original = loadings_scaled.multiply(scale_, axis=0)

        explained_ratio = evals_k / np.sum(evals)

        self.result_ = PCAResult(
            mean_=mean_.rename("mean"),
            scale_=scale_.rename("scale"),
            eigenvalues_=pd.Series(evals_k, index=pc_names, name="eigenvalue"),
            explained_variance_ratio_=pd.Series(explained_ratio, index=pc_names, name="explained_variance_ratio"),
            loadings_=loadings_original,
            columns_=X.columns,
            config_=self.config,
        )
        return self

    def _check_is_fitted(self) -> PCAResult:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self.result_

    def loadings(self) -> pd.DataFrame:
        return self._check_is_fitted().loadings_.copy()

    def summary(self) -> pd.DataFrame:
        res = self._check_is_fitted()
        out = pd.concat(
            [res.eigenvalues_, res.explained_variance_ratio_],
            axis=1,
        )
        out["cumulative_explained_variance_ratio"] = out["explained_variance_ratio"].cumsum()
        return out

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        metric_weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute factor scores by JOINT projection:
            c = (B' W B)^(-1) B' W (x - mean)

        Returns a DataFrame with rows = observations, cols = PCs
        """
        res = self._check_is_fitted()
        X = _ensure_dataframe(X, columns=res.columns_, name="X")
        _validate_no_missing(X, "X")

        if not X.columns.equals(res.columns_):
            X = X.reindex(columns=res.columns_)
            if X.isnull().values.any():
                raise ValueError("X could not be reindexed cleanly to fitted columns.")

        B = res.loadings_.values  # in original feature space
        W = _diag_weight_matrix(metric_weights, res.columns_)

        gram = B.T @ W @ B
        gram = _make_psd(gram, min_eig=1e-12)
        gram_inv = np.linalg.pinv(gram)

        Xc = (X - res.mean_).values
        C = (gram_inv @ B.T @ W @ Xc.T).T

        return pd.DataFrame(C, index=X.index, columns=res.loadings_.columns)

    def reconstruct(
        self,
        X: pd.DataFrame | np.ndarray,
        metric_weights: Optional[pd.Series] = None,
        add_mean: bool = True,
    ) -> pd.DataFrame:
        """
        Reconstruct observations from selected PCs:
            x_hat = mean + B c
        """
        res = self._check_is_fitted()
        scores = self.transform(X, metric_weights=metric_weights)

        B = res.loadings_.values
        Xhat_centered = scores.values @ B.T
        Xhat = Xhat_centered + (res.mean_.values if add_mean else 0.0)

        return pd.DataFrame(Xhat, index=scores.index, columns=res.columns_)

    def residual(
        self,
        X: pd.DataFrame | np.ndarray,
        metric_weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        X_df = _ensure_dataframe(X, name="X")
        Xhat = self.reconstruct(X_df, metric_weights=metric_weights, add_mean=True)
        return X_df.reindex(columns=Xhat.columns) - Xhat

    def reconstruction_error(
        self,
        X: pd.DataFrame | np.ndarray,
        metric_weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Returns per-date reconstruction diagnostics:
          - abs_error_l2
          - rel_error_l2
          - weighted_abs_error
          - weighted_rel_error
        """
        res = self._check_is_fitted()
        X = _ensure_dataframe(X, columns=res.columns_, name="X")
        Xhat = self.reconstruct(X, metric_weights=metric_weights, add_mean=True)

        err = X - Xhat
        W = _diag_weight_matrix(metric_weights, res.columns_)

        abs_l2 = np.sqrt((err.values ** 2).sum(axis=1))
        x_l2 = np.sqrt((X.values ** 2).sum(axis=1))
        rel_l2 = abs_l2 / np.maximum(x_l2, 1e-12)

        weighted_abs = np.sqrt(np.einsum("ij,jk,ik->i", err.values, W, err.values))
        weighted_x = np.sqrt(np.einsum("ij,jk,ik->i", X.values, W, X.values))
        weighted_rel = weighted_abs / np.maximum(weighted_x, 1e-12)

        return pd.DataFrame(
            {
                "abs_error_l2": abs_l2,
                "rel_error_l2": rel_l2,
                "weighted_abs_error": weighted_abs,
                "weighted_rel_error": weighted_rel,
            },
            index=X.index,
        )

    def explain_pnl(
        self,
        X: pd.DataFrame | np.ndarray,
        vega: pd.DataFrame | np.ndarray,
        metric_weights: Optional[pd.Series] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        PCA PnL explain for time series of market shocks and vegas.

        Inputs:
          X    : rows = dates, cols = features, market shock vectors
          vega : rows = dates, cols = features, vega vectors on same grid

        Returns:
          {
            "scores": factor scores,
            "factor_stock": factor stocks g' b_k,
            "factor_pnl": factor contributions,
            "summary": total real / pca / residual pnl
          }
        """
        res = self._check_is_fitted()

        X = _ensure_dataframe(X, columns=res.columns_, name="X")
        vega = _ensure_dataframe(vega, index=X.index, columns=res.columns_, name="vega")

        if not X.columns.equals(res.columns_):
            X = X.reindex(columns=res.columns_)
        if not vega.columns.equals(res.columns_):
            vega = vega.reindex(columns=res.columns_)

        if X.isnull().values.any():
            raise ValueError("X contains missing values after reindexing.")
        if vega.isnull().values.any():
            raise ValueError("vega contains missing values after reindexing.")

        scores = self.transform(X, metric_weights=metric_weights)
        B = res.loadings_

        # factor stock per date: g_t' b_k
        factor_stock = pd.DataFrame(
            vega.values @ B.values,
            index=X.index,
            columns=B.columns,
        )

        factor_pnl = scores * factor_stock
        pca_pnl = factor_pnl.sum(axis=1)

        real_pnl = pd.Series((vega.values * X.values).sum(axis=1), index=X.index, name="real_pnl")
        residual_pnl = real_pnl - pca_pnl

        summary = pd.DataFrame(
            {
                "real_pnl": real_pnl,
                "pca_pnl": pca_pnl,
                "residual_pnl": residual_pnl,
                "abs_residual_pnl": residual_pnl.abs(),
            }
        )

        return {
            "scores": scores,
            "factor_stock": factor_stock,
            "factor_pnl": factor_pnl,
            "summary": summary,
        }


# ============================================================
# Rolling PCA model
# ============================================================

@dataclass
class RollingPCAConfig(PCAConfig):
    window: int = 126
    step: int = 5
    align_signs: bool = True


class RollingSurfacePCA:
    """
    Rolling PCA:
      - fit a static PCA on each rolling window
      - optionally align signs to previous window
      - use latest available fitted window for each date in transform/reconstruct/explain

    This is a practical production-friendly version.
    """

    def __init__(self, config: RollingPCAConfig):
        self.config = config
        self.models_: Dict[pd.Timestamp, SurfacePCA] = {}
        self.window_end_dates_: List[pd.Timestamp] = []

    def fit(self, X: pd.DataFrame | np.ndarray) -> "RollingSurfacePCA":
        X = _ensure_dataframe(X, name="X")
        _validate_no_missing(X, "X")

        if self.config.window <= self.config.n_components:
            raise ValueError("window must be larger than n_components")
        if self.config.window > len(X):
            raise ValueError("window is larger than available number of observations")
        if self.config.step <= 0:
            raise ValueError("step must be > 0")

        prev_loadings: Optional[pd.DataFrame] = None

        for end_idx in range(self.config.window, len(X) + 1, self.config.step):
            window_df = X.iloc[end_idx - self.config.window : end_idx]
            end_date = pd.Timestamp(window_df.index[-1])

            static_cfg = PCAConfig(
                n_components=self.config.n_components,
                cov_estimator=self.config.cov_estimator,
                center_method=self.config.center_method,
                ewma_half_life=self.config.ewma_half_life,
                standardize=self.config.standardize,
                feature_weights=self.config.feature_weights,
                min_eigenvalue=self.config.min_eigenvalue,
            )

            model = SurfacePCA(static_cfg).fit(window_df)

            if self.config.align_signs and prev_loadings is not None:
                aligned = _sign_align_loadings(model.result_.loadings_, prev_loadings)
                model.result_.loadings_ = aligned

            prev_loadings = model.result_.loadings_.copy()
            self.models_[end_date] = model
            self.window_end_dates_.append(end_date)

        return self

    def _check_is_fitted(self) -> None:
        if not self.models_:
            raise RuntimeError("Rolling model is not fitted. Call fit() first.")

    def _latest_model_for_date(self, dt: pd.Timestamp) -> Tuple[pd.Timestamp, SurfacePCA]:
        """
        For a given date dt, use the latest model whose window end <= dt.
        """
        self._check_is_fitted()
        valid_dates = [d for d in self.window_end_dates_ if d <= dt]
        if not valid_dates:
            raise ValueError(f"No fitted rolling PCA window available for date {dt}.")
        chosen = max(valid_dates)
        return chosen, self.models_[chosen]

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        metric_weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        X = _ensure_dataframe(X, name="X")
        rows = []

        for dt, row in X.iterrows():
            _, model = self._latest_model_for_date(pd.Timestamp(dt))
            scores = model.transform(row.to_frame().T, metric_weights=metric_weights)
            rows.append(scores.iloc[0])

        return pd.DataFrame(rows, index=X.index)

    def reconstruct(
        self,
        X: pd.DataFrame | np.ndarray,
        metric_weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        X = _ensure_dataframe(X, name="X")
        rows = []

        for dt, row in X.iterrows():
            _, model = self._latest_model_for_date(pd.Timestamp(dt))
            xhat = model.reconstruct(row.to_frame().T, metric_weights=metric_weights)
            rows.append(xhat.iloc[0])

        return pd.DataFrame(rows, index=X.index, columns=X.columns)

    def explain_pnl(
        self,
        X: pd.DataFrame | np.ndarray,
        vega: pd.DataFrame | np.ndarray,
        metric_weights: Optional[pd.Series] = None,
    ) -> Dict[str, pd.DataFrame]:
        X = _ensure_dataframe(X, name="X")
        vega = _ensure_dataframe(vega, index=X.index, columns=X.columns, name="vega")

        score_rows = []
        stock_rows = []
        pnl_rows = []
        summary_rows = []

        for dt in X.index:
            _, model = self._latest_model_for_date(pd.Timestamp(dt))
            one_x = X.loc[[dt]]
            one_g = vega.loc[[dt]]

            out = model.explain_pnl(one_x, one_g, metric_weights=metric_weights)

            score_rows.append(out["scores"].iloc[0])
            stock_rows.append(out["factor_stock"].iloc[0])
            pnl_rows.append(out["factor_pnl"].iloc[0])
            summary_rows.append(out["summary"].iloc[0])

        return {
            "scores": pd.DataFrame(score_rows, index=X.index),
            "factor_stock": pd.DataFrame(stock_rows, index=X.index),
            "factor_pnl": pd.DataFrame(pnl_rows, index=X.index),
            "summary": pd.DataFrame(summary_rows, index=X.index),
        }


# ============================================================
# Evaluation helpers
# ============================================================

def pnl_metrics(summary: pd.DataFrame) -> pd.Series:
    """
    Compute useful headline metrics from explain_pnl()['summary'].
    """
    real = summary["real_pnl"]
    pred = summary["pca_pnl"]
    err = summary["residual_pnl"]

    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    corr = real.corr(pred) if len(summary) > 1 else np.nan
    sign_acc = np.mean(np.sign(real) == np.sign(pred))
    r2_like = 1.0 - np.sum(err ** 2) / np.maximum(np.sum((real - real.mean()) ** 2), 1e-12)

    return pd.Series(
        {
            "rmse": rmse,
            "mae": mae,
            "correlation": corr,
            "sign_accuracy": sign_acc,
            "r2_like": r2_like,
            "mean_abs_real_pnl": np.mean(np.abs(real)),
            "mean_abs_residual_pnl": np.mean(np.abs(err)),
        }
    )


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Example assumptions:
    #   vol_changes : DataFrame, rows=dates, cols=surface buckets
    #   vegas       : DataFrame, rows=dates, cols=surface buckets
    #
    # Replace these two lines with your actual data loading.
    # --------------------------------------------------------
    n_dates = 300
    n_features = 80

    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    columns = pd.Index([f"bucket_{i}" for i in range(n_features)], name="bucket")

    vol_changes = pd.DataFrame(rng.normal(size=(n_dates, n_features)), index=dates, columns=columns)
    vegas = pd.DataFrame(rng.normal(size=(n_dates, n_features)), index=dates, columns=columns)

    # Optional feature weights:
    # Example 1: inverse bucket noise
    bucket_std = vol_changes.std(axis=0).replace(0.0, 1.0)
    metric_weights = (1.0 / (bucket_std ** 2)).rename("metric_weight")

    # ----------------------------
    # Static PCA
    # ----------------------------
    static_cfg = PCAConfig(
        n_components=7,
        cov_estimator="ledoit_wolf",
        center_method="mean",
        ewma_half_life=20.0,
        standardize=False,
    )

    static_model = SurfacePCA(static_cfg).fit(vol_changes)

    print("\n=== STATIC PCA SUMMARY ===")
    print(static_model.summary())

    static_explain = static_model.explain_pnl(
        X=vol_changes,
        vega=vegas,
        metric_weights=metric_weights,
    )

    print("\n=== STATIC PCA PNL METRICS ===")
    print(pnl_metrics(static_explain["summary"]))

    print("\n=== STATIC PCA LAST 5 DAYS SUMMARY ===")
    print(static_explain["summary"].tail())

    # ----------------------------
    # Rolling PCA
    # ----------------------------
    rolling_cfg = RollingPCAConfig(
        n_components=7,
        cov_estimator="ewma",
        center_method="mean",
        ewma_half_life=20.0,
        standardize=False,
        window=126,   # ~6 months of business days
        step=5,       # weekly update
        align_signs=True,
    )

    rolling_model = RollingSurfacePCA(rolling_cfg).fit(vol_changes)

    rolling_explain = rolling_model.explain_pnl(
        X=vol_changes.iloc[126:],
        vega=vegas.iloc[126:],
        metric_weights=metric_weights,
    )

    print("\n=== ROLLING PCA PNL METRICS ===")
    print(pnl_metrics(rolling_explain["summary"]))

    print("\n=== ROLLING PCA LAST 5 DAYS SUMMARY ===")
    print(rolling_explain["summary"].tail())
