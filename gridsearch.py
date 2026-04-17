"""
Grid search over PCA hyperparameters, evaluated via out-of-sample reconstruction
error on daily vol-surface changes.

Parameters searched (all modifiable in the `param_grid` dict passed to PCAGridSearch):
    cov_method            : sample / ledoit_wolf / oas / ewma / ewma_lw
    ewma_halflife         : only used when cov_method in {"ewma", "ewma_lw"}
    n_components          : fixed number of PCs
    window                : rolling PCA window (e.g. "3M", "6M", "12M")
    step                  : rolling step (e.g. "1D", "1W")
    avg_windows           : number of recent stabilised windows averaged for smoothing
    history_length_years  : slice the last N years of data before fitting; None = all data

Evaluation:
    For each consecutive window pair (i, i+1), use window i's (smoothed) loadings to
    reconstruct each daily change dv_t for t in (window_i.end, window_{i+1}.end].
    Residuals pooled across all windows → metrics (MSE, RMSE, median/max L2 norm).
"""
from itertools import product
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from PCAEngines import RollingPCA


# Default search grid. Override by passing your own dict to PCAGridSearch.
DEFAULT_PARAM_GRID = {
    "cov_method":            ["sample", "ledoit_wolf", "oas", "ewma", "ewma_lw"],
    "ewma_halflife":         [5, 11, 22, 45],           # only used for ewma / ewma_lw
    "n_components":          [3, 5, 8, 10, 12],
    "window":                ["3M", "6M", "9M", "12M"],
    "step":                  ["1D"],
    "avg_windows":           [1, 2, 4, 8],
    "history_length_years":  [None, 3, 5, 7],           # None = use all available history
}


@dataclass
class BacktestResult:
    params: dict
    metrics: dict
    n_evals: int
    error: Optional[str] = None


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def compute_oos_reconstruction_errors(
    X: pd.DataFrame, model: RollingPCA, use_smoothed: bool = True
) -> np.ndarray:
    """
    Out-of-sample reconstruction residuals, pooled across all windows.

    For each consecutive window pair (i, i+1):
        L = smoothed_loadings_[i]  (or windows_[i].loadings if use_smoothed=False)
        For each date t in (window_i.end, window_{i+1}.end]:
            dv_t      = X_t - X_{t-1}
            residual  = dv_t - L L^T dv_t
            contribute  ||residual||^2

    Returns a flat ndarray of squared residual norms, one per evaluation date.
    Since np.linalg.eigh returns orthonormal eigenvectors, L has orthonormal columns,
    so  ||residual||^2 = ||dv||^2 - ||L^T dv||^2  — used below for speed.
    """
    if len(model.windows_) < 2:
        return np.array([])

    X_arr = X.values
    dates = X.index
    all_errors: list[np.ndarray] = []

    for i in range(len(model.windows_) - 1):
        L = (model.smoothed_loadings_[i].values if use_smoothed
             else model.windows_[i].loadings.values)

        t0 = model.windows_[i].window_end
        t1 = model.windows_[i + 1].window_end
        mask = (dates > t0) & (dates <= t1)
        idx = np.where(mask)[0]
        idx = idx[idx > 0]  # need a previous row to compute dv
        if len(idx) == 0:
            continue

        dV = X_arr[idx] - X_arr[idx - 1]                          # (n_eval, n_features)
        proj = dV @ L                                             # (n_eval, n_components)
        total_sq = np.einsum("ij,ij->i", dV, dV)
        proj_sq = np.einsum("ij,ij->i", proj, proj)
        all_errors.append(np.maximum(total_sq - proj_sq, 0.0))    # guard tiny negatives

    return np.concatenate(all_errors) if all_errors else np.array([])


def default_metric_fn(errors_sq: np.ndarray) -> dict:
    """Aggregate pool of squared residual norms into scalar metrics."""
    if len(errors_sq) == 0:
        return {"mse": np.nan, "rmse": np.nan,
                "mean_norm": np.nan, "median_norm": np.nan, "max_norm": np.nan}
    mse = float(errors_sq.mean())
    return {
        "mse":         mse,
        "rmse":        float(np.sqrt(mse)),
        "mean_norm":   float(np.sqrt(errors_sq).mean()),
        "median_norm": float(np.sqrt(np.median(errors_sq))),
        "max_norm":    float(np.sqrt(errors_sq.max())),
    }


def evaluate_config(
    X: pd.DataFrame,
    params: dict,
    use_smoothed: bool = True,
    metric_fn: Callable[[np.ndarray], dict] = default_metric_fn,
) -> BacktestResult:
    """Fit one RollingPCA configuration and compute pooled OOS metrics."""
    try:
        # Slice training history if requested
        hist_years = params.get("history_length_years")
        if hist_years is None:
            X_eval = X
        else:
            cutoff = X.index[-1] - pd.DateOffset(years=hist_years)
            X_eval = X[X.index >= cutoff]

        model = RollingPCA(
            window=params["window"],
            step=params["step"],
            n_components=params["n_components"],
            cov_method=params["cov_method"],
            ewma_halflife=params.get("ewma_halflife"),
            avg_windows=params["avg_windows"],
        )
        model.fit(X_eval)

        errors_sq = compute_oos_reconstruction_errors(X_eval, model, use_smoothed=use_smoothed)
        return BacktestResult(params=params, metrics=metric_fn(errors_sq), n_evals=len(errors_sq))

    except Exception as e:
        return BacktestResult(params=params, metrics={}, n_evals=0, error=f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# Grid search driver
# ----------------------------------------------------------------------
class PCAGridSearch:
    """
    Exhaustive grid search over PCA hyperparameters.

    Parameters
    ----------
    X            : (T x N) DataFrame of vol surfaces
    param_grid   : dict of lists. See DEFAULT_PARAM_GRID for the schema.
    use_smoothed : evaluate using RollingPCA.smoothed_loadings_ (True)
                   or raw windows_[i].loadings (False)
    metric_fn    : callable(errors_sq: ndarray) -> dict of scalar metrics.
                   Swap in your own PnL function here (e.g. vega-weighted).
    n_jobs       : joblib parallelism. -1 = all cores. 1 = sequential.
    verbose      : print progress and summary
    """
    def __init__(
        self,
        X: pd.DataFrame,
        param_grid: Optional[dict] = None,
        use_smoothed: bool = True,
        metric_fn: Callable[[np.ndarray], dict] = default_metric_fn,
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        self.X = X
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.use_smoothed = use_smoothed
        self.metric_fn = metric_fn
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.results_: list[BacktestResult] = []

    def _iter_combinations(self):
        """
        Enumerate all valid parameter combos. `ewma_halflife` is only iterated for
        ewma-family covariance methods; for other methods it is collapsed to a single
        None value so the grid doesn't explode with redundant configurations.
        """
        g = self.param_grid
        cov_methods          = g.get("cov_method", ["sample"])
        halflives            = g.get("ewma_halflife", [None])
        n_components_list    = g.get("n_components", [5])
        windows              = g.get("window", ["6M"])
        steps                = g.get("step", ["1D"])
        avg_windows_list     = g.get("avg_windows", [1])
        history_years_list   = g.get("history_length_years", [None])

        for cov in cov_methods:
            hl_options = halflives if cov in ("ewma", "ewma_lw") else [None]
            for hl, n, w, s, aw, hy in product(
                hl_options, n_components_list, windows, steps, avg_windows_list, history_years_list
            ):
                yield {
                    "cov_method":           cov,
                    "ewma_halflife":        hl,
                    "n_components":         n,
                    "window":               w,
                    "step":                 s,
                    "avg_windows":          aw,
                    "history_length_years": hy,
                }

    def run(self) -> pd.DataFrame:
        combos = list(self._iter_combinations())
        if self.verbose:
            print(f"[PCAGridSearch] evaluating {len(combos)} configurations"
                  f"{' in parallel' if _HAS_JOBLIB and self.n_jobs != 1 else ' sequentially'}")

        def _run_one(params):
            return evaluate_config(self.X, params, self.use_smoothed, self.metric_fn)

        if _HAS_JOBLIB and self.n_jobs != 1:
            self.results_ = Parallel(n_jobs=self.n_jobs, verbose=10 if self.verbose else 0)(
                delayed(_run_one)(p) for p in combos
            )
        else:
            iterator = tqdm(combos, desc="Grid search") if (_HAS_TQDM and self.verbose) else combos
            self.results_ = [_run_one(p) for p in iterator]

        if self.verbose:
            n_failed = sum(1 for r in self.results_ if r.error is not None)
            if n_failed:
                print(f"[PCAGridSearch] {n_failed} configuration(s) raised errors — see `error` column")

        return self.results_df()

    def results_df(self, sort_by: str = "mse") -> pd.DataFrame:
        """All results as a DataFrame, sorted ascending by `sort_by` (lower is better)."""
        rows = []
        for r in self.results_:
            row = {**r.params, **r.metrics, "n_evals": r.n_evals}
            if r.error is not None:
                row["error"] = r.error
            rows.append(row)
        df = pd.DataFrame(rows)
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=True, na_position="last")
        return df.reset_index(drop=True)

    def best_params(self, metric: str = "mse") -> dict:
        """Return the full parameter dict of the lowest-error configuration."""
        df = self.results_df(sort_by=metric)
        if df.empty or df[metric].isna().all():
            return {}
        param_keys = list(next(iter(self._iter_combinations())).keys())
        return df.iloc[0][param_keys].to_dict()

    def summary(self, top_n: int = 10, metric: str = "mse") -> pd.DataFrame:
        """Top-N configurations for quick inspection."""
        return self.results_df(sort_by=metric).head(top_n)
