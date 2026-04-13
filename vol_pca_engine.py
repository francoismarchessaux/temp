"""
vol_pca_engine.py
=================
Vol Surface PCA Engine — USD ATM Normal Vols (Swaptions + Cap Floors)
Supports: Static PCA · Rolling PCA · Iterated (regime-switching) PCA
Covariance methods: sample · ledoit_wolf · oas · ewma · ewma_lw

Usage
-----
    from vol_pca_engine import (
        SurfaceConfig, GridSpec, SurfacePreprocessor,
        StaticPCA, RollingPCA, IteratedPCA,
        PnLAttributor, PCAModelComparison,
    )

All three PCA classes share the same interface:
    model.fit(X, dates)
    result = model.get_loadings(date=None)
    report = attributor.attribute(result, vega_swp, vol_change_swp)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CovMethod = Literal["sample", "ledoit_wolf", "oas", "ewma", "ewma_lw"]


# =============================================================================
# 1.  DATA INTERFACE
# =============================================================================

class VolDataLoader(ABC):
    """
    Abstract base class. Implement the three methods for your data source
    (Bloomberg, internal REST API, pickle files, …).
    """

    @abstractmethod
    def get_vol_changes(
        self,
        start_date: str,
        end_date:   str,
        surface:    Literal["swaptions", "capfloor"],
    ) -> pd.DataFrame:
        """
        Daily vol changes for each pillar over the requested period.

        Returns
        -------
        pd.DataFrame
            index   : DatetimeIndex (business days)
            columns : pd.MultiIndex(expiry, tenor)
            values  : ATM normal vol changes in bps  (NOT divided by 1000)
        """

    @abstractmethod
    def get_vega(
        self,
        as_of:   str,
        surface: Literal["swaptions", "capfloor"],
    ) -> pd.DataFrame:
        """
        Vega sensitivity surface as of a given date.

        Returns
        -------
        pd.DataFrame
            index   : expiries
            columns : tenors
            values  : vega in USD (keep units consistent across surfaces)
        """

    @abstractmethod
    def get_spot_vol_change(
        self,
        as_of:   str,
        surface: Literal["swaptions", "capfloor"],
    ) -> pd.DataFrame:
        """
        Today's vol change surface (same shape as get_vega).
        """


# =============================================================================
# 2.  SURFACE PREPROCESSOR
# =============================================================================

@dataclass
class GridSpec:
    expiries: List[str]
    tenors:   List[str]


@dataclass
class SurfaceConfig:
    swaption_grid:       GridSpec
    capfloor_grid:       GridSpec
    cap_injection_tenor: str   = "1Y"    # swaption pillar that receives cap vega
    separate_pca:        bool  = True    # recommended: separate cap / swaption PCA
    unit_divisor:        float = 1000.0  # bps → kbps for numerical stability


class SurfacePreprocessor:
    """
    Reindexes raw surfaces onto the model grid and flattens them for PCA.
    """

    def __init__(self, config: SurfaceConfig):
        self.cfg = config

    def reindex(
        self,
        raw:     pd.DataFrame,
        surface: Literal["swaptions", "capfloor"],
    ) -> pd.DataFrame:
        """Align a raw surface to the model grid, fill gaps with 0, scale."""
        grid = (
            self.cfg.swaption_grid
            if surface == "swaptions"
            else self.cfg.capfloor_grid
        )
        return (
            raw.reindex(index=grid.expiries, columns=grid.tenors)
               .fillna(0.0)
               / self.cfg.unit_divisor
        )

    def flatten(self, surface_df: pd.DataFrame) -> np.ndarray:
        """Row-major flatten of a (expiries × tenors) surface."""
        return surface_df.values.flatten().astype(float)

    def flatten_history(
        self,
        raw_history: pd.DataFrame,
        surface:     Literal["swaptions", "capfloor"],
    ) -> np.ndarray:
        """
        Convert a (T × MultiIndex) history DataFrame into a (T × n_features) array.

        raw_history columns must be pd.MultiIndex(expiry, tenor) as returned
        by VolDataLoader.get_vol_changes().
        """
        grid = (
            self.cfg.swaption_grid
            if surface == "swaptions"
            else self.cfg.capfloor_grid
        )
        records = []
        for _, row in raw_history.iterrows():
            if isinstance(row.index, pd.MultiIndex):
                day_df = row.unstack(level="tenor")
            else:
                # Fallback: single-level columns assumed to be tenors
                day_df = pd.DataFrame(
                    row.values.reshape(len(grid.expiries), len(grid.tenors)),
                    index=grid.expiries,
                    columns=grid.tenors,
                )
            aligned = (
                day_df.reindex(index=grid.expiries, columns=grid.tenors)
                      .fillna(0.0)
                / self.cfg.unit_divisor
            )
            records.append(aligned.values.flatten())
        return np.array(records, dtype=float)


# =============================================================================
# 3.  COVARIANCE ESTIMATORS
# =============================================================================

def compute_covariance(
    X:             np.ndarray,
    method:        CovMethod = "ledoit_wolf",
    ewma_lambda:   float     = 0.94,
) -> np.ndarray:
    """
    Compute a (n × n) covariance matrix from a (T × n) data matrix.

    Parameters
    ----------
    X            : (T × n) — rows are observations, must already be mean-centred
    method       : estimator name
    ewma_lambda  : EWMA decay (0 < λ < 1; higher = slower, more stable)

    Returns
    -------
    np.ndarray  (n × n) positive semi-definite covariance matrix
    """
    T, n = X.shape

    if method == "sample":
        return np.cov(X, rowvar=False, ddof=1)

    elif method == "ledoit_wolf":
        return LedoitWolf(store_precision=False).fit(X).covariance_

    elif method == "oas":
        return OAS(store_precision=False).fit(X).covariance_

    elif method == "ewma":
        # Weights: most recent observation has highest weight
        weights = np.array(
            [(1 - ewma_lambda) * ewma_lambda ** i for i in range(T - 1, -1, -1)]
        )
        weights /= weights.sum()
        Xw = X - (weights @ X)                  # weighted mean-centre
        return (Xw * weights[:, None]).T @ Xw

    elif method == "ewma_lw":
        # EWMA structure + Ledoit-Wolf shrinkage intensity
        ewma_cov = compute_covariance(X, method="ewma", ewma_lambda=ewma_lambda)
        alpha    = LedoitWolf(store_precision=False).fit(X).shrinkage_
        target   = np.diag(np.diag(ewma_cov))   # diagonal shrinkage target
        return (1 - alpha) * ewma_cov + alpha * target

    else:
        raise ValueError(f"Unknown covariance method: {method!r}")


# =============================================================================
# 4.  CORE RESULT CONTAINER
# =============================================================================

@dataclass
class PCAResult:
    """Stores fitted PCA output and exposes transform / P&L attribution."""

    loadings:                 np.ndarray   # (n_components × n_features)
    explained_variance:       np.ndarray   # (n_components,)  eigenvalues
    explained_variance_ratio: np.ndarray   # (n_components,)
    cumulative_variance_ratio: np.ndarray  # (n_components,)
    mean:                     np.ndarray   # (n_features,)  training mean
    n_components:             int
    cov_method:               str
    fit_date:                 Optional[str] = None
    window_start:             Optional[str] = None
    window_end:               Optional[str] = None

    # ------------------------------------------------------------------ #

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project a flattened surface onto PCs → (n_components,) scores."""
        return self.loadings @ (x - self.mean)

    def reconstruct(
        self, x: np.ndarray, n_pcs: Optional[int] = None
    ) -> np.ndarray:
        """Reconstruct x using the first n_pcs components."""
        n = n_pcs or self.n_components
        L = self.loadings[:n]
        scores = L @ (x - self.mean)
        return self.mean + L.T @ scores

    def pnl_attribution(
        self,
        vega_flat:       np.ndarray,
        vol_change_flat: np.ndarray,
    ) -> pd.Series:
        """
        Decompose P&L by principal component.

        Formula:  PnL_k = (vega · L_k) × (Δvol · L_k)

        The outer product vega·L_k captures the vega exposure to PC k.
        Δvol·L_k is the vol move projected onto PC k.
        Their product is the P&L driven by that component.

        Parameters
        ----------
        vega_flat       : (n_features,) flattened vega surface
        vol_change_flat : (n_features,) flattened vol-change surface

        Returns
        -------
        pd.Series  indexed ['PC1', 'PC2', …]
        """
        pnl = {}
        for k in range(self.n_components):
            lk  = self.loadings[k]
            pnl[f"PC{k + 1}"] = float(lk @ vega_flat) * float(lk @ vol_change_flat)
        return pd.Series(pnl)


# =============================================================================
# 5.  SHARED EIGENDECOMPOSITION HELPER
# =============================================================================

def _eigen_pca(
    Xc:          np.ndarray,
    cov:         np.ndarray,
    n_components: int,
    target_var:  float,
    auto_select: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract PCA loadings from a precomputed covariance matrix.

    Returns  (loadings, eigenvalues, explained_variance_ratio)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)   # ascending order
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = np.maximum(eigenvalues[idx], 0.0)   # numerical floor
    eigenvectors = eigenvectors[:, idx]

    total_var = max(eigenvalues.sum(), 1e-12)
    evr       = eigenvalues / total_var
    cumevr    = np.cumsum(evr)

    # Determine how many PCs to keep
    if auto_select:
        n = int(np.searchsorted(cumevr, target_var) + 1)
        n = max(n, n_components)               # never go below user minimum
    else:
        n = n_components

    n = min(n, len(eigenvalues), Xc.shape[0] - 1, Xc.shape[1])

    loadings = eigenvectors[:, :n].T           # (n × p)

    # Sign convention: largest absolute weight positive
    for i in range(n):
        if loadings[i, np.abs(loadings[i]).argmax()] < 0:
            loadings[i] *= -1

    return loadings, eigenvalues[:n], evr[:n]


# =============================================================================
# 6.  STATIC PCA
# =============================================================================

class StaticPCA:
    """
    Single PCA fitted once on a fixed historical window.
    Loadings are constant until you call fit() again.

    Parameters
    ----------
    n_components    : minimum number of PCs to retain
    cov_method      : covariance estimator
    ewma_lambda     : EWMA decay (only used when cov_method in {'ewma','ewma_lw'})
    auto_select_n   : if True, raise n_components until target_variance is reached
    target_variance : variance threshold for auto_select_n
    """

    def __init__(
        self,
        n_components:    int       = 8,
        cov_method:      CovMethod = "ledoit_wolf",
        ewma_lambda:     float     = 0.96,
        auto_select_n:   bool      = True,
        target_variance: float     = 0.95,
    ):
        self.n_components    = n_components
        self.cov_method      = cov_method
        self.ewma_lambda     = ewma_lambda
        self.auto_select_n   = auto_select_n
        self.target_variance = target_variance
        self.result_: Optional[PCAResult] = None

    def fit(
        self,
        X:         np.ndarray,
        fit_date:  Optional[str] = None,
    ) -> "StaticPCA":
        """
        Fit PCA on (T × n_features) matrix X.

        X should already be the flattened history (use SurfacePreprocessor).
        Zero rows (non-trading days) are automatically dropped.
        """
        X   = X[np.abs(X).sum(axis=1) > 1e-10].astype(float)
        mu  = X.mean(axis=0)
        Xc  = X - mu

        cov = compute_covariance(Xc, method=self.cov_method,
                                 ewma_lambda=self.ewma_lambda)

        loadings, evals, evr = _eigen_pca(
            Xc, cov,
            n_components = self.n_components,
            target_var   = self.target_variance,
            auto_select  = self.auto_select_n,
        )

        cumevr = np.cumsum(evr)
        n = len(evals)

        self.result_ = PCAResult(
            loadings                  = loadings,
            explained_variance        = evals,
            explained_variance_ratio  = evr,
            cumulative_variance_ratio = cumevr,
            mean                      = mu,
            n_components              = n,
            cov_method                = self.cov_method,
            fit_date                  = fit_date,
        )
        return self

    def get_loadings(self, date: Optional[str] = None) -> PCAResult:
        """Return the fitted result (date argument is ignored for static PCA)."""
        if self.result_ is None:
            raise RuntimeError("Call fit() first.")
        return self.result_


# =============================================================================
# 7.  ROLLING PCA
# =============================================================================

class RollingPCA:
    """
    Re-estimates PCA loadings over a sliding window.

    Parameters
    ----------
    window          : number of trading days per window
    step            : number of days between window re-estimates
    avg_windows     : number of consecutive windows to average (smoothing)
    (other params same as StaticPCA)
    """

    def __init__(
        self,
        window:          int       = 125,
        step:            int       = 5,
        n_components:    int       = 8,
        cov_method:      CovMethod = "ledoit_wolf",
        ewma_lambda:     float     = 0.96,
        avg_windows:     int       = 4,
        auto_select_n:   bool      = True,
        target_variance: float     = 0.95,
    ):
        self.window          = window
        self.step            = step
        self.n_components    = n_components
        self.cov_method      = cov_method
        self.ewma_lambda     = ewma_lambda
        self.avg_windows     = avg_windows
        self.auto_select_n   = auto_select_n
        self.target_variance = target_variance

        self.windows_: List[PCAResult]   = []
        self.dates_:   pd.DatetimeIndex  = pd.DatetimeIndex([])

    def fit(
        self,
        X:     np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> "RollingPCA":
        """Fit rolling windows across the full history."""
        X = X[np.abs(X).sum(axis=1) > 1e-10].astype(float)
        T = X.shape[0]

        # Synchronise dates and data lengths
        dates = dates[:T]

        _static = StaticPCA(
            n_components    = self.n_components,
            cov_method      = self.cov_method,
            ewma_lambda     = self.ewma_lambda,
            auto_select_n   = self.auto_select_n,
            target_variance = self.target_variance,
        )

        end_positions = list(range(self.window, T, self.step))
        if not end_positions or end_positions[-1] != T:
            end_positions.append(T)

        self.windows_ = []
        result_dates  = []

        for end in end_positions:
            start  = max(0, end - self.window)
            X_win  = X[start:end]
            if X_win.shape[0] < max(20, self.n_components + 5):
                continue
            _static.fit(X_win, fit_date=str(dates[end - 1]))
            res              = _static.result_
            res.window_start = str(dates[start])
            res.window_end   = str(dates[end - 1])
            self.windows_.append(res)
            result_dates.append(dates[end - 1])

        self.dates_ = pd.DatetimeIndex(result_dates)
        return self

    def get_loadings(self, date: Optional[str] = None) -> PCAResult:
        """
        Return the effective loadings for a given date.

        If avg_windows > 1, the last `avg_windows` window results are
        sign-aligned and averaged, then re-orthogonalised via QR.
        """
        if not self.windows_:
            raise RuntimeError("Call fit() first.")

        if date is None:
            idx = len(self.windows_)
        else:
            dt  = pd.Timestamp(date)
            idx = int(np.searchsorted(self.dates_, dt, side="right"))
            idx = max(1, min(idx, len(self.windows_)))

        slice_start = max(0, idx - self.avg_windows)
        candidates  = self.windows_[slice_start:idx]

        if len(candidates) == 1 or self.avg_windows <= 1:
            return candidates[-1]

        return self._average_windows(candidates)

    def _average_windows(self, candidates: List[PCAResult]) -> PCAResult:
        ref    = candidates[-1]
        n_pcs  = ref.loadings.shape[0]
        accum  = np.zeros_like(ref.loadings)

        for res in candidates:
            n_common = min(n_pcs, res.loadings.shape[0])
            for k in range(n_common):
                sign = np.sign(np.dot(ref.loadings[k], res.loadings[k]) + 1e-12)
                accum[k] += sign * res.loadings[k]
            # PCs beyond n_common stay zero

        accum /= len(candidates)

        # Re-orthogonalise via QR decomposition
        Q, _      = np.linalg.qr(accum.T)
        avg_load  = Q.T[:n_pcs]

        return PCAResult(
            loadings                  = avg_load,
            explained_variance        = ref.explained_variance,
            explained_variance_ratio  = ref.explained_variance_ratio,
            cumulative_variance_ratio = ref.cumulative_variance_ratio,
            mean                      = np.mean([c.mean for c in candidates], axis=0),
            n_components              = n_pcs,
            cov_method                = ref.cov_method,
            fit_date                  = ref.fit_date,
            window_start              = candidates[0].window_start,
            window_end                = ref.window_end,
        )


# =============================================================================
# 8.  ITERATED (REGIME-SWITCHING) PCA
# =============================================================================

class IteratedPCA:
    """
    Fits a separate PCA per vol regime detected via Gaussian Mixture Model.

    On each day, a soft-weighted blend of regime loadings is returned based
    on the posterior regime probability.

    Parameters
    ----------
    n_regimes        : number of market regimes (typically 2–3)
    regime_features  : how to build features for the GMM regime detector:
                         "vol_level"  → rolling mean absolute vol change (default)
                         "pca_scores" → scores on global first-3 PCs
                         "raw"        → full flattened surface (high-dim)
    blend            : if True (default), soft-blend loadings by regime proba;
                       if False, use hard assignment to most likely regime
    """

    def __init__(
        self,
        n_regimes:       int       = 2,
        n_components:    int       = 8,
        cov_method:      CovMethod = "ledoit_wolf",
        ewma_lambda:     float     = 0.96,
        regime_features: Literal["vol_level", "pca_scores", "raw"] = "vol_level",
        blend:           bool      = True,
        auto_select_n:   bool      = True,
        target_variance: float     = 0.95,
    ):
        self.n_regimes       = n_regimes
        self.n_components    = n_components
        self.cov_method      = cov_method
        self.ewma_lambda     = ewma_lambda
        self.regime_features = regime_features
        self.blend           = blend
        self.auto_select_n   = auto_select_n
        self.target_variance = target_variance

        self.regime_results_: Dict[int, PCAResult] = {}
        self.gmm_:            Optional[GaussianMixture] = None
        self.regime_labels_:  np.ndarray = np.array([])
        self._X_history:      np.ndarray = np.array([])

    # ------------------------------------------------------------------ #

    def _build_regime_features(self, X: np.ndarray) -> np.ndarray:
        if self.regime_features == "vol_level":
            vol_series = pd.Series(np.abs(X).mean(axis=1))
            vol_roll   = (
                vol_series.rolling(21, min_periods=5)
                           .mean()
                           .bfill()
                           .values
            )
            return vol_roll.reshape(-1, 1)

        elif self.regime_features == "pca_scores":
            _s = StaticPCA(n_components=5, cov_method="sample",
                           auto_select_n=False)
            _s.fit(X)
            return (X - _s.result_.mean) @ _s.result_.loadings.T

        else:  # raw — use PCA to compress first to avoid GMM curse of dimensionality
            _s = StaticPCA(n_components=10, cov_method="ledoit_wolf",
                           auto_select_n=False)
            _s.fit(X)
            return (X - _s.result_.mean) @ _s.result_.loadings.T

    def fit(
        self,
        X:     np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> "IteratedPCA":
        X = X[np.abs(X).sum(axis=1) > 1e-10].astype(float)
        self._X_history = X

        # --- Step 1: detect regimes ---
        feat = self._build_regime_features(X)
        self.gmm_ = GaussianMixture(
            n_components    = self.n_regimes,
            covariance_type = "full",
            n_init          = 20,
            random_state    = 42,
            max_iter        = 200,
        ).fit(feat)
        self.regime_labels_ = self.gmm_.predict(feat)
        self.regime_probs_  = self.gmm_.predict_proba(feat)
        self._feat_history  = feat

        # --- Step 2: fit PCA per regime ---
        _static = StaticPCA(
            n_components    = self.n_components,
            cov_method      = self.cov_method,
            ewma_lambda     = self.ewma_lambda,
            auto_select_n   = self.auto_select_n,
            target_variance = self.target_variance,
        )

        print(f"  Regime distribution:")
        for r in range(self.n_regimes):
            mask   = self.regime_labels_ == r
            n_days = mask.sum()
            X_reg  = X[mask]

            if n_days < max(20, self.n_components + 5):
                print(f"    Regime {r}: {n_days} days — too few, skipped.")
                continue

            _static.fit(X_reg)
            self.regime_results_[r] = _static.result_
            cumvar = _static.result_.cumulative_variance_ratio[-1] * 100
            print(f"    Regime {r}: {n_days} days ({n_days/len(X)*100:.0f}%) | "
                  f"{_static.result_.n_components} PCs → {cumvar:.1f}% var")

        return self

    def get_loadings(
        self,
        current_obs: np.ndarray,
    ) -> PCAResult:
        """
        Return effective loadings for a new observation.

        current_obs : (n_features,) today's flattened vol-change surface.
        """
        if not self.regime_results_:
            raise RuntimeError("Call fit() first.")

        feat  = self._build_regime_features(current_obs.reshape(1, -1))
        probs = self.gmm_.predict_proba(feat)[0]       # (n_regimes,)

        if not self.blend:
            best = int(np.argmax(probs))
            return self.regime_results_.get(
                best,
                next(iter(self.regime_results_.values()))
            )

        # ---- Soft blend ----
        ref_r = max(self.regime_results_, key=lambda r: (self.regime_labels_ == r).sum())
        ref   = self.regime_results_[ref_r]
        n_pcs = ref.loadings.shape[0]

        accum_load = np.zeros_like(ref.loadings)
        accum_mean = np.zeros_like(ref.mean)
        total_w    = 0.0

        for r, res in self.regime_results_.items():
            w = float(probs[r])
            if w < 1e-6:
                continue
            n_common = min(n_pcs, res.loadings.shape[0])
            for k in range(n_common):
                sign = np.sign(np.dot(ref.loadings[k], res.loadings[k]) + 1e-12)
                accum_load[k] += w * sign * res.loadings[k]
            accum_mean += w * res.mean
            total_w    += w

        accum_load /= max(total_w, 1e-12)
        accum_mean /= max(total_w, 1e-12)

        # Re-orthogonalise via QR
        Q, _       = np.linalg.qr(accum_load.T)
        avg_load   = Q.T[:n_pcs]

        label_str = " | ".join(
            f"R{r}={probs[r]:.2f}" for r in range(self.n_regimes)
        )

        return PCAResult(
            loadings                  = avg_load,
            explained_variance        = ref.explained_variance,
            explained_variance_ratio  = ref.explained_variance_ratio,
            cumulative_variance_ratio = ref.cumulative_variance_ratio,
            mean                      = accum_mean,
            n_components              = n_pcs,
            cov_method                = f"regime-blend [{label_str}]",
            fit_date                  = None,
        )

    @property
    def regime_summary(self) -> pd.DataFrame:
        rows = []
        for r in range(self.n_regimes):
            n_days = int((self.regime_labels_ == r).sum())
            res    = self.regime_results_.get(r)
            rows.append({
                "n_days":      n_days,
                "pct_time":    f"{n_days / len(self.regime_labels_) * 100:.0f}%",
                "n_pcs":       res.n_components if res else "—",
                "var_exp":     f"{res.cumulative_variance_ratio[-1]*100:.1f}%"
                               if res else "—",
            })
        return pd.DataFrame(rows, index=[f"Regime {r}" for r in range(self.n_regimes)])


# =============================================================================
# 9.  P&L ATTRIBUTOR
# =============================================================================

@dataclass
class PnLReport:
    pc_pnl:     pd.Series    # P&L contribution per PC
    real_pnl:   float        # full vega × vol_change
    pca_pnl:    float        # sum of PC P&Ls
    error:      float        # real − pca
    error_pct:  float        # |error| / |real|
    model_type: str
    result:     PCAResult

    def pc_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "PnL":        self.pc_pnl,
            "Cumulative": self.pc_pnl.cumsum(),
        })

    def headline(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Real": [self.real_pnl], "PCA": [self.pca_pnl],
             "Error": [self.error], "Error %": [f"{self.error_pct*100:.1f}%"]},
            index=["Final"],
        )


class PnLAttributor:
    """
    Computes P&L attribution for any PCAResult.
    Works identically for Static, Rolling, and Iterated outputs.
    """

    def __init__(self, preprocessor: SurfacePreprocessor):
        self.prep = preprocessor

    def attribute(
        self,
        pca_result:     PCAResult,
        vega_swp:       pd.DataFrame,
        vol_change_swp: pd.DataFrame,
        model_type:     str = "static",
    ) -> PnLReport:
        """
        Parameters
        ----------
        pca_result     : output of any model's get_loadings()
        vega_swp       : swaption vega surface on the model grid
        vol_change_swp : today's swaption vol-change surface on the model grid
        model_type     : label for the report
        """
        vega_flat = self.prep.flatten(vega_swp)
        dv_flat   = self.prep.flatten(vol_change_swp)

        real_pnl = float(np.dot(vega_flat, dv_flat))
        pc_pnl   = pca_result.pnl_attribution(vega_flat, dv_flat)
        pca_pnl  = float(pc_pnl.sum())
        error    = real_pnl - pca_pnl

        return PnLReport(
            pc_pnl    = pc_pnl,
            real_pnl  = real_pnl,
            pca_pnl   = pca_pnl,
            error     = error,
            error_pct = abs(error) / (abs(real_pnl) + 1e-12),
            model_type= model_type,
            result    = pca_result,
        )


# =============================================================================
# 10. MODEL COMPARISON RUNNER
# =============================================================================

class PCAModelComparison:
    """
    Convenience class: fits all three PCA variants and compares them
    on a given vega / vol-change observation.

    Parameters
    ----------
    (same as individual model parameters)
    """

    def __init__(
        self,
        preprocessor:    SurfacePreprocessor,
        n_components:    int       = 8,
        target_variance: float     = 0.95,
        cov_method:      CovMethod = "ledoit_wolf",
        ewma_lambda:     float     = 0.96,
        rolling_window:  int       = 125,
        rolling_step:    int       = 5,
        rolling_avg:     int       = 4,
        n_regimes:       int       = 2,
    ):
        self.prep = preprocessor

        self.static_   = StaticPCA(
            n_components=n_components, cov_method=cov_method,
            ewma_lambda=ewma_lambda, auto_select_n=True,
            target_variance=target_variance,
        )
        self.rolling_  = RollingPCA(
            window=rolling_window, step=rolling_step,
            n_components=n_components, cov_method=cov_method,
            ewma_lambda=ewma_lambda, avg_windows=rolling_avg,
            auto_select_n=True, target_variance=target_variance,
        )
        self.iterated_ = IteratedPCA(
            n_regimes=n_regimes, n_components=n_components,
            cov_method=cov_method, ewma_lambda=ewma_lambda,
            auto_select_n=True, target_variance=target_variance,
        )
        self.attributor_ = PnLAttributor(preprocessor)
        self._fitted     = False

    def fit(
        self,
        X:     np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> "PCAModelComparison":
        print("── Static PCA ──────────────────────────────────")
        self.static_.fit(X)
        res = self.static_.result_
        print(f"  {res.n_components} PCs → "
              f"{res.cumulative_variance_ratio[-1]*100:.1f}% variance explained")

        print("\n── Rolling PCA ─────────────────────────────────")
        self.rolling_.fit(X, dates)
        n_wins = len(self.rolling_.windows_)
        print(f"  {n_wins} windows fitted")

        print("\n── Iterated PCA ────────────────────────────────")
        self.iterated_.fit(X, dates)

        self._fitted = True
        return self

    def compare(
        self,
        vega_swp:       pd.DataFrame,
        vol_change_swp: pd.DataFrame,
        today:          Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return a side-by-side comparison of all three models for today.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        today_flat = self.prep.flatten(vol_change_swp)
        rows = {}

        for name, model in [
            ("Static",   self.static_),
            ("Rolling",  self.rolling_),
            ("Iterated", self.iterated_),
        ]:
            if name == "Iterated":
                res = model.get_loadings(today_flat)
            elif name == "Rolling":
                res = model.get_loadings(today)
            else:
                res = model.get_loadings()

            report = self.attributor_.attribute(
                pca_result     = res,
                vega_swp       = vega_swp,
                vol_change_swp = vol_change_swp,
                model_type     = name,
            )
            rows[name] = {
                "Real P&L":   round(report.real_pnl, 3),
                "PCA P&L":    round(report.pca_pnl, 3),
                "Error":      round(report.error, 3),
                "Error %":    f"{report.error_pct * 100:.1f}%",
                "N PCs":      res.n_components,
                "Cov method": res.cov_method,
            }

        return pd.DataFrame(rows).T
