"""
volsurface.py — US Rates Options vol-surface PCA library.

All model and analysis code. No plotting (see volsurface_plots.py).

Locked-in decisions from the research:
  bp diffs; drop 2D; weekday_rolling calendar cleaning (theta once ARM expiry
  dates are wired); per-window static sigma (EWMA rejected); window 500 on
  factor-identification grounds; rank factors by BOOK PnL variance, not surface
  variance; sigma is a first-class object.

Central identity:  PnL_t = vega' dv_t = u' z_t,  u = sigma * vega.

Conventions:
  changes : DataFrame, index = dates ASCENDING, columns = MultiIndex(expiry, tenor)
  vega    : Series over the same MultiIndex, $ per +1bp
  loadings: unit-norm columns in z-space
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence, Optional, Union, Dict, List, Tuple

import numpy as np
import pandas as pd

# =====================================================================
# PART 0 — CONFIG
# =====================================================================

REDUCED_EXPIRIES = ["1M", "3M", "6M", "1Y", "18M", "2Y", "3Y", "4Y", "5Y",
                    "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]
REDUCED_TENORS = ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y",
                  "15Y", "20Y", "25Y", "30Y"]

CALENDAR_AFFECTED = ("2D", "1W", "2W", "1M", "3M")


@dataclass
class PCAConfig:
    """Every knob in one object so a run is reproducible."""
    n_components: int = 6
    n_report: int = 8
    window: int = 500
    refit_every: int = 21
    demean: bool = True
    standardize: bool = True
    calendar_method: str = "weekday_rolling"
    calendar_window: int = 250
    calendar_rows: Sequence[str] = CALENDAR_AFFECTED
    drop_expiries: Sequence[str] = ("2D",)
    min_obs: int = 60
    sign_anchor: str = "positive_sum"

    def copy(self, **kw) -> "PCAConfig":
        return PCAConfig(**{**self.__dict__, **kw})

    def __repr__(self) -> str:
        return (f"PCAConfig(k={self.n_components}, window={self.window}, "
                f"calendar='{self.calendar_method}')")


@dataclass
class Regime:
    name: str
    start: str
    end: str
    kind: str = "stress"     # "stress" | "calm" | "transition"


# Adjust dates to the desk's own view before quoting these.
REGIMES: List[Regime] = [
    Regime("Covid 2020",     "2020-02-20", "2020-05-29", "stress"),
    Regime("Calm 2021",      "2021-01-04", "2021-09-30", "calm"),
    Regime("Hiking 2022",    "2022-03-01", "2022-11-30", "stress"),
    Regime("SVB 2023",       "2023-03-01", "2023-05-31", "stress"),
    Regime("Autumn 2023",    "2023-08-01", "2023-11-30", "stress"),
    Regime("Calm 2024",      "2024-04-01", "2024-12-31", "calm"),
    Regime("Tariffs 2025",   "2025-03-15", "2025-06-30", "stress"),
]


# =====================================================================
# PART 1 — DATA ADAPTERS (DataManager interop)
# =====================================================================
# Three facts about DataManager objects that must be respected:
#   1. get_vol / get_vega return HistoricalSurfaces, not DataFrames/Series.
#   2. HistoricalSurfaces is ordered MOST-RECENT-FIRST. Diffing it unflipped
#      silently reverses the sign of every daily change.
#   3. Vega sits on trade-level pillars, not vol pillars -> must be REBUCKETED
#      (a plain reindex zeroes most of the book).

_UNIT_DAYS = {"D": 1.0, "W": 7.0, "M": 30.0, "Y": 365.0}
_UNIT_YEARS = {"D": 1.0 / 365.0, "W": 7.0 / 365.0, "M": 1.0 / 12.0, "Y": 1.0}


def maturity_to_days(x) -> float:
    """Sort key for tenor/expiry labels. Unknown labels sort last."""
    try:
        s = str(x).strip().upper()
        return float(s[:-1]) * _UNIT_DAYS[s[-1]]
    except Exception:
        return float("inf")


def tenor_to_years(label) -> float:
    s = str(label).strip().upper()
    if not s or s[-1] not in _UNIT_YEARS:
        raise ValueError(f"cannot parse tenor label {label!r}")
    return float(s[:-1]) * _UNIT_YEARS[s[-1]]


def _sort_grid(cols: pd.MultiIndex) -> pd.MultiIndex:
    df = cols.to_frame(index=False)
    df["_e"] = df["expiry"].map(maturity_to_days)
    df["_t"] = df["tenor"].map(maturity_to_days)
    df = df.sort_values(["_e", "_t"])
    return pd.MultiIndex.from_arrays([df["expiry"].values, df["tenor"].values],
                                     names=["expiry", "tenor"])


def to_panel(obj) -> pd.DataFrame:
    """HistoricalSurfaces (or DataFrame) -> chronological ASCENDING DataFrame."""
    df = getattr(obj, "data", obj)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"cannot read a panel out of {type(obj)}")
    if not isinstance(df.columns, pd.MultiIndex):
        raise TypeError("expected MultiIndex(expiry, tenor) columns")
    df = df.copy()
    df.columns.names = ["expiry", "tenor"]
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df.sort_index()


def bucket_matrix(source: Sequence[str], target: Sequence[str]) -> pd.DataFrame:
    """Linear-in-time redistribution matrix, rows sum to 1 (vega conserving).
    Same construction as DataManager.bucket_matrix, replicated so this module
    imports without a live Jade/Pearl session."""
    src = np.array([tenor_to_years(x) for x in source])
    tgt = np.array([tenor_to_years(x) for x in target])
    if np.any(np.diff(tgt) <= 0):
        raise ValueError("target grid must be strictly increasing in years")
    W = np.zeros((len(src), len(tgt)))
    for i, t in enumerate(src):
        if t <= tgt[0]:
            W[i, 0] = 1.0
        elif t >= tgt[-1]:
            W[i, -1] = 1.0
        else:
            j = int(np.searchsorted(tgt, t)) - 1
            wl = (tgt[j + 1] - t) / (tgt[j + 1] - tgt[j])
            W[i, j], W[i, j + 1] = wl, 1.0 - wl
    return pd.DataFrame(W, index=list(source), columns=list(target))


def _project_2d(surf: pd.DataFrame, expiries, tenors) -> pd.Series:
    We = bucket_matrix(list(surf.index), list(expiries))
    Wt = bucket_matrix(list(surf.columns), list(tenors))
    V = surf.reindex(index=We.index, columns=Wt.index).astype(float).fillna(0.0).to_numpy()
    out = pd.DataFrame(We.to_numpy().T @ V @ Wt.to_numpy(), index=expiries, columns=tenors)
    out.index.name, out.columns.name = "expiry", "tenor"
    return out.stack()


def prepare_grid(surfaces, expiries=None, tenors=None,
                 drop_expiries=("2D",), verbose=True) -> pd.DataFrame:
    """Restrict a vol panel to the modelling grid. Reports missing pillars."""
    panel = to_panel(surfaces)
    expiries = list(REDUCED_EXPIRIES if expiries is None else expiries)
    tenors = list(REDUCED_TENORS if tenors is None else tenors)
    expiries = [e for e in expiries if e not in set(drop_expiries)]

    have_e = set(panel.columns.get_level_values("expiry"))
    have_t = set(panel.columns.get_level_values("tenor"))
    miss_e = [e for e in expiries if e not in have_e]
    miss_t = [t for t in tenors if t not in have_t]
    if verbose and (miss_e or miss_t):
        print(f"[grid] missing expiries: {miss_e or 'none'} | missing tenors: {miss_t or 'none'}")

    expiries = [e for e in expiries if e in have_e]
    tenors = [t for t in tenors if t in have_t]
    keep = [c for c in panel.columns if c[0] in set(expiries) and c[1] in set(tenors)]
    out = panel.loc[:, keep]
    out = out.loc[:, _sort_grid(out.columns)]

    if verbose:
        T, p = out.shape
        verdict = ("unusable" if T < p else "weak" if T < 3 * p
                   else "fair" if T < 10 * p else "good")
        print(f"[grid] {len(expiries)}x{len(tenors)} = {p} points, T={T}, "
              f"T/p={T/p:.1f} ({verdict})")
    return out


def to_changes(levels: pd.DataFrame, method: str = "diff") -> pd.DataFrame:
    """Daily changes. 'diff' = bp differences (the production choice)."""
    lv = to_panel(levels)
    if method == "diff":
        ch = lv.diff()
    elif method == "rel":
        ch = lv.pct_change()
    elif method == "log":
        if (lv <= 0).any().any():
            warnings.warn("non-positive vols — log changes unsafe")
        ch = np.log(lv).diff()
    else:
        raise ValueError(f"unknown method {method!r}")
    return ch.dropna(how="all")


def align_vega(vega, grid: pd.MultiIndex, date=None, aggregate=None,
               halflife: float = 63.0, check: bool = True, verbose: bool = True):
    """Any vega object -> Series (single book) or DataFrame (dates x grid).

    Accepts HistoricalSurfaces, VegaSurface, 2D DataFrame, MultiIndex Series,
    or an already-panelled DataFrame. Rebuckets onto `grid`, conserving vega.
    aggregate: None | "last" | "mean" | "ewma".
    """
    expiries = sorted(set(grid.get_level_values("expiry")), key=maturity_to_days)
    tenors = sorted(set(grid.get_level_values("tenor")), key=maturity_to_days)
    df = getattr(vega, "data", vega)

    if isinstance(df, pd.Series):
        if not isinstance(df.index, pd.MultiIndex):
            raise TypeError("Series vega needs MultiIndex(expiry, tenor)")
        return _finish(_project_2d(df.unstack("tenor"), expiries, tenors),
                       grid, float(df.sum()), check, verbose)

    if isinstance(df, pd.DataFrame) and not isinstance(df.columns, pd.MultiIndex):
        return _finish(_project_2d(df, expiries, tenors), grid,
                       float(np.nansum(df.values)), check, verbose)

    panel = to_panel(vega)
    if date is not None:
        row = panel.loc[pd.Timestamp(date)]
        return _finish(_project_2d(row.unstack("tenor"), expiries, tenors),
                       grid, float(row.sum()), check, verbose)

    src_e = sorted(set(panel.columns.get_level_values("expiry")), key=maturity_to_days)
    src_t = sorted(set(panel.columns.get_level_values("tenor")), key=maturity_to_days)
    We = bucket_matrix(src_e, expiries).to_numpy()
    Wt = bucket_matrix(src_t, tenors).to_numpy()
    full = panel.reindex(columns=pd.MultiIndex.from_product(
        [src_e, src_t], names=["expiry", "tenor"])).fillna(0.0)
    A = full.to_numpy().reshape(len(full), len(src_e), len(src_t))
    P = np.einsum("ij,tjk,kl->til", We.T, A, Wt).reshape(len(full), len(expiries) * len(tenors))
    out = pd.DataFrame(P, index=full.index, columns=pd.MultiIndex.from_product(
        [expiries, tenors], names=["expiry", "tenor"])).reindex(columns=grid)

    if check:
        err = float(np.abs(out.sum(axis=1).to_numpy() - panel.sum(axis=1).to_numpy()).max())
        if err > 1e-6 * max(float(np.abs(panel.sum(axis=1)).max()), 1.0):
            warnings.warn(f"rebucketing lost vega: max daily discrepancy {err:.3e}")
    if verbose:
        print(f"[vega] {panel.shape[1]} trade pillars -> {out.shape[1]} model points; "
              f"{out.index[0].date()} -> {out.index[-1].date()} ({len(out)} days)")

    if aggregate is None:
        return out
    if aggregate == "last":
        return out.iloc[-1]
    if aggregate == "mean":
        return out.mean()
    if aggregate == "ewma":
        return out.ewm(halflife=halflife).mean().iloc[-1]
    raise ValueError(f"unknown aggregate {aggregate!r}")


def _finish(ser, grid, total_in, check, verbose):
    ser.index.names = ["expiry", "tenor"]
    out = ser.reindex(grid).fillna(0.0)
    if check and abs(float(out.sum()) - float(total_in)) > 1e-6 * max(abs(float(total_in)), 1.0):
        warnings.warn(f"rebucketing lost vega: {total_in:.6f} -> {float(out.sum()):.6f}")
    if verbose:
        print(f"[vega] -> {len(out)} points, net {out.sum():,.1f}, gross {out.abs().sum():,.1f}")
    return out


# =====================================================================
# PART 2 — SURFACE / DATA-QUALITY DIAGNOSTICS
# =====================================================================

def point_stats(changes: pd.DataFrame) -> pd.DataFrame:
    """Per-node daily-change stats + staleness. Ranked by zero_share."""
    ch = changes
    out = pd.DataFrame({
        "mean": ch.mean(), "std": ch.std(ddof=1),
        "skew": ch.skew(), "kurtosis": ch.kurt(),
        "min": ch.min(), "max": ch.max(),
        "zero_share": (ch.abs() < 1e-10).mean(),
        "lag1_ac": ch.apply(lambda c: c.autocorr(1)),
    })
    out.index.names = ["expiry", "tenor"]
    return out


def staleness_map(changes: pd.DataFrame) -> pd.DataFrame:
    """Share of days with an exactly-zero change, as an expiry x tenor grid.
    High values = the node is not being re-marked daily."""
    z = (changes.abs() < 1e-10).mean()
    z.index.names = ["expiry", "tenor"]
    m = z.unstack("tenor")
    return m.reindex(index=sorted(m.index, key=maturity_to_days),
                     columns=sorted(m.columns, key=maturity_to_days))


def marking_noise_diagnostics(changes: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """Per-node signatures of marking noise rather than market moves.

    neg_ac1     : negative lag-1 autocorrelation = bid/ask bounce or stale marks
    reversal    : share of days where the change reverses the previous day
    zero_share  : staleness
    """
    ch = changes
    ac1 = ch.apply(lambda c: c.autocorr(1))
    rev = ((ch * ch.shift(1)) < 0).mean()
    out = pd.DataFrame({
        "lag1_ac": ac1,
        "reversal_share": rev,
        "zero_share": (ch.abs() < 1e-10).mean(),
        "std": ch.std(ddof=1),
    })
    out["suspect"] = (out["lag1_ac"] < -0.10) | (out["zero_share"] > 0.20)
    out.index.names = ["expiry", "tenor"]
    return out


def rolling_dispersion(changes: pd.DataFrame, window: int = 21) -> pd.Series:
    """Rolling mean of the cross-sectional std of daily changes — a vol gauge."""
    return changes.std(axis=1).rolling(window).mean()


# =====================================================================
# PART 3 — CALENDAR / EXPIRY-ROLL EFFECT
# =====================================================================
# The artefact: consecutive business days share an expiry date, so short-expiry
# vol grinds one way then jumps at the roll. Deterministic sawtooth in the
# changes -> a spurious factor.
# 'project' and 'overlap' were tested on real data and FAILED (project destroyed
# ~20% of variance and raised PC1 concentration; overlap induced negative lag-5
# autocorrelation). They are not implemented.

def weekday_profile(changes: pd.DataFrame, rows: Sequence[str] = CALENDAR_AFFECTED) -> pd.DataFrame:
    """Mean daily change by weekday, per expiry row. The raw evidence of the effect."""
    names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    wd = changes.index.dayofweek
    out = {}
    for e in sorted(set(changes.columns.get_level_values("expiry")), key=maturity_to_days):
        sub = changes.loc[:, [c for c in changes.columns if c[0] == e]].mean(axis=1)
        out[e] = {names[d]: float(sub[wd == d].mean()) for d in range(5)}
        out[e]["spread"] = max(out[e].values()) - min(out[e].values())
        out[e]["flagged"] = e in set(rows)
    return pd.DataFrame(out).T


def _weekday_rolling_adjust(s: pd.Series, window: int) -> pd.Series:
    """Subtract a TRAILING per-weekday mean (causal — a full-sample mean peeks)."""
    wd = s.index.dayofweek
    out = s.copy().astype(float)
    for d in range(5):
        mask = wd == d
        if mask.sum() < 5:
            continue
        sub = s[mask]
        trail = sub.rolling(window=max(4, window // 5), min_periods=4).mean().shift(1)
        fallback = sub.expanding(min_periods=2).mean().shift(1)
        out.loc[sub.index] = sub - trail.fillna(fallback).fillna(0.0)
    return out


def clean_calendar(changes: pd.DataFrame, method: str = "weekday_rolling",
                   rows: Sequence[str] = CALENDAR_AFFECTED, window: int = 250,
                   expiry_dates: Optional[pd.DataFrame] = None,
                   levels: Optional[pd.DataFrame] = None,
                   verbose: bool = True) -> pd.DataFrame:
    """'none' | 'weekday_rolling' (default) | 'theta' (needs true ARM expiry dates)."""
    if method in ("none", None):
        return changes.copy()
    ch = changes.copy()
    target = [c for c in ch.columns if c[0] in set(rows)]
    if not target:
        if verbose:
            print("[calendar] no affected rows on this grid — passthrough")
        return ch

    if method == "weekday_rolling":
        if not isinstance(ch.index, pd.DatetimeIndex):
            raise TypeError("weekday_rolling needs a DatetimeIndex")
        for c in target:
            ch[c] = _weekday_rolling_adjust(ch[c], window)
    elif method == "theta":
        if expiry_dates is None or levels is None:
            raise ValueError("theta needs `expiry_dates` (dates x expiry pillar, true "
                             "option expiry from ARM) and `levels`")
        ch = _theta_adjust(ch, to_panel(levels), expiry_dates, rows)
    else:
        raise ValueError(f"unknown/rejected calendar method {method!r} "
                         "('project' and 'overlap' failed on real data)")
    if verbose:
        print(f"[calendar] {method} applied to {len(target)} columns / "
              f"{len(set(c[0] for c in target))} expiry rows")
    return ch


def _theta_adjust(ch, lv, expiry_dates, rows):
    """Remove roll-down implied by the local term-structure slope."""
    expiries = sorted(set(c[0] for c in ch.columns), key=maturity_to_days)
    for tnr in sorted(set(c[1] for c in ch.columns), key=maturity_to_days):
        col = {e: (e, tnr) for e in expiries if (e, tnr) in ch.columns}
        for e in [x for x in expiries if x in set(rows) and x in col]:
            if e not in expiry_dates.columns:
                continue
            tau = ((pd.to_datetime(expiry_dates[e]).reindex(ch.index)
                    - pd.Series(ch.index, index=ch.index)).dt.days / 365.0)
            dtau = tau.diff()
            i = expiries.index(e)
            nb = [x for x in expiries[max(0, i - 1): i + 2] if x in col and x != e]
            x0 = maturity_to_days(e) / 365.0
            slopes = []
            for n in nb:
                x1 = maturity_to_days(n) / 365.0
                if abs(x1 - x0) > 1e-9:
                    slopes.append((lv[col[n]] - lv[col[e]]) / (x1 - x0))
            if slopes:
                slope = pd.concat(slopes, axis=1).mean(axis=1).reindex(ch.index)
                ch[col[e]] = ch[col[e]] - (slope * dtau).fillna(0.0)
    return ch


def calendar_acceptance(scores: pd.DataFrame, model=None, max_pc: int = 8,
                        min_var_share: float = 0.005) -> pd.DataFrame:
    """corr(PC score, weekday dummy). THE acceptance test for calendar cleaning.

    Verdict is attrs['worst_signal'], computed only over PCs holding at least
    `min_var_share` of variance. A naive max over all PCs is dominated by
    degenerate noise PCs and reports that a successful cleaning made things worse.
    """
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("scores need a DatetimeIndex")
    names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    wd = scores.index.dayofweek
    rows = {}
    for k in scores.columns[:max_pc]:
        v = scores[k].values.astype(float)
        if np.std(v) < 1e-12:
            rows[k] = {names[d]: 0.0 for d in range(5)}
            continue
        rows[k] = {names[d]: float(np.corrcoef(v, (wd == d).astype(float))[0, 1])
                   for d in range(5)}
    out = pd.DataFrame(rows).T
    out["max_abs"] = out.abs().max(axis=1)
    if model is not None:
        share = model.explained.reindex(out.index)
        out["surf_var_%"] = 100 * share
        out["signal"] = share >= min_var_share
    else:
        out["surf_var_%"], out["signal"] = np.nan, True
    sig = out.loc[out["signal"], "max_abs"]
    out.attrs["worst_signal"] = float(sig.max()) if len(sig) else np.nan
    out.attrs["worst_all"] = float(out["max_abs"].max())
    out.attrs["n_signal_pcs"] = int(out["signal"].sum())
    return out


def calendar_scorecard(changes: pd.DataFrame, config: PCAConfig,
                       methods: Sequence[str] = ("none", "weekday_rolling"),
                       **kw) -> pd.DataFrame:
    """Compare cleaning methods on: weekday leakage, variance retained, PC1 share,
    effective dimensionality, lag-5 autocorrelation of the short rows."""
    base_var = float(changes.var().sum())
    rows = {}
    for m in methods:
        cl = clean_calendar(changes, method=m, rows=config.calendar_rows,
                            window=config.calendar_window, verbose=False, **kw).dropna(how="any")
        mod = VolPCA(config=config).fit(cl)
        ca = calendar_acceptance(mod.transform(cl, k=config.n_report), model=mod)
        short = [c for c in cl.columns if c[0] in set(config.calendar_rows)]
        ac5 = float(np.mean([cl[c].autocorr(5) for c in short])) if short else np.nan
        rows[m] = {
            "weekday_leak_signal": ca.attrs["worst_signal"],
            "weekday_leak_naive": ca.attrs["worst_all"],
            "var_retained_%": 100 * float(cl.var().sum()) / base_var,
            "PC1_share_%": 100 * mod.explained.iloc[0],
            "eff_dim": effective_dimensionality(cl.corr()),
            "short_lag5_ac": ac5,
            "n_signal_PCs": mod.n_signal,
        }
    return pd.DataFrame(rows).T.round(4)


# =====================================================================
# PART 4 — CORRELATION STRUCTURE & DIMENSIONALITY
# =====================================================================

def marchenko_pastur_bounds(n_obs: int, n_vars: int, sigma2: float = 1.0):
    """Noise-bulk edges for standardised data."""
    q = n_vars / n_obs
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def effective_dimensionality(corr: pd.DataFrame) -> float:
    """(sum lam)^2 / sum(lam^2). 1 = one-factor, p = independent."""
    lam = np.clip(np.linalg.eigvalsh(np.asarray(corr, dtype=float)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def correlation_summary(changes: pd.DataFrame) -> Dict:
    C = changes.corr()
    off = C.values[~np.eye(len(C), dtype=bool)]
    return {"mean_corr": float(off.mean()), "median_corr": float(np.median(off)),
            "min_corr": float(off.min()), "eff_dim": effective_dimensionality(C),
            "n_points": len(C)}


def neighbour_correlation(changes: pd.DataFrame, tenor: str, max_sep: int = 6) -> pd.Series:
    """Mean corr between expiries N pillars apart, at one tenor.

    Flat (cumulative) cap vols show near-1 correlation decaying very slowly,
    because adjacent expiries share most of their optionality by construction.
    Compare an IRG curve against SWOPT: if IRG sits far above and decays slower,
    the feed is cumulative and MUST be stripped to caplet vols first.
    """
    cols = [c for c in changes.columns if c[1] == tenor]
    cols = sorted(cols, key=lambda c: maturity_to_days(c[0]))
    C = changes[cols].corr().values
    out = {}
    for sep in range(1, min(max_sep, len(cols)) + 1):
        d = [C[i, i + sep] for i in range(len(cols) - sep)]
        out[sep] = float(np.mean(d)) if d else np.nan
    return pd.Series(out, name=f"mean_corr_{tenor}")


def seam_test(changes_a: pd.DataFrame, changes_b: pd.DataFrame,
              label_a: str = "SWOPT", label_b: str = "IRG") -> pd.DataFrame:
    """Correlation of shared pillars across two feeds. Near-1 with near-zero
    tracking error at a shared pillar means a duplicated/interpolated column."""
    shared = [c for c in changes_a.columns if c in set(changes_b.columns)]
    rows = {}
    for c in shared:
        a, b = changes_a[c].align(changes_b[c], join="inner")
        if len(a) < 20:
            continue
        rows[f"{c[0]}x{c[1]}"] = {
            "corr": float(a.corr(b)),
            "tracking_err": float((a - b).std(ddof=1)),
            f"std_{label_a}": float(a.std(ddof=1)),
            f"std_{label_b}": float(b.std(ddof=1)),
        }
    out = pd.DataFrame(rows).T
    if len(out):
        out["suspect_duplicate"] = (out["corr"] > 0.995) & (
            out["tracking_err"] < 0.05 * out[f"std_{label_a}"])
    return out.round(4)


# =====================================================================
# PART 5 — TRANSFORMS & NORMALITY
# =====================================================================

def transform_comparison(levels: pd.DataFrame, config: PCAConfig) -> pd.DataFrame:
    """Which transform gets closest to multivariate normal?

    Compares bp diffs / relative / log, each raw vs z-scored vs EWMA-scaled, on
    marginal kurtosis, skew, and the joint chi-2 QQ slope (1.0 = MV normal).
    EWMA lost on the joint fit even though it improves marginals — dividing by
    lagged trailing vol manufactures outliers when vol gaps up.
    """
    out = {}
    for meth in ("diff", "rel", "log"):
        try:
            ch = to_changes(levels, meth).dropna(how="any")
        except Exception:
            continue
        for scale in ("raw", "zscore", "ewma"):
            x = ch.copy()
            if scale == "zscore":
                x = (x - x.mean()) / x.std(ddof=1)
            elif scale == "ewma":
                sd = x.pow(2).ewm(halflife=63).mean().pow(0.5).shift(1)
                x = (x / sd).dropna(how="any")
            x = x.replace([np.inf, -np.inf], np.nan).dropna(how="any")
            if len(x) < 50:
                continue
            out[f"{meth}/{scale}"] = {
                "mean_kurtosis": float(x.kurt().mean()),
                "mean_abs_skew": float(x.skew().abs().mean()),
                "chi2_slope": _chi2_qq_slope(x),
                "max_abs_z": float(np.abs((x - x.mean()) / x.std(ddof=1)).values.max()),
            }
    res = pd.DataFrame(out).T
    res["chi2_dist_from_1"] = (res["chi2_slope"] - 1.0).abs()
    return res.round(4)


def _chi2_qq_slope(x: pd.DataFrame, max_dim: int = 40) -> float:
    """Slope of squared Mahalanobis distance vs chi-2 quantiles. 1.0 = MV normal."""
    from scipy import stats
    X = x.values
    if X.shape[1] > max_dim:                       # project to keep S invertible
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        X = Xc @ Vt[:max_dim].T
    Xc = X - X.mean(0)
    S = np.cov(Xc, rowvar=False)
    try:
        Si = np.linalg.inv(S + 1e-10 * np.eye(S.shape[0]))
    except np.linalg.LinAlgError:
        return np.nan
    d2 = np.einsum("ij,jk,ik->i", Xc, Si, Xc)
    d2 = np.sort(d2)
    q = stats.chi2.ppf((np.arange(len(d2)) + 0.5) / len(d2), df=X.shape[1])
    return float(np.polyfit(q, d2, 1)[0])


def normality_by_node(changes: pd.DataFrame) -> pd.DataFrame:
    """Per-node Jarque-Bera and tail counts."""
    from scipy import stats
    z = (changes - changes.mean()) / changes.std(ddof=1)
    jb = z.apply(lambda c: stats.jarque_bera(c.dropna())[0])
    out = pd.DataFrame({
        "kurtosis": changes.kurt(), "skew": changes.skew(), "jarque_bera": jb,
        "share_gt_3sd": (z.abs() > 3).mean(), "share_gt_5sd": (z.abs() > 5).mean(),
    })
    out.index.names = ["expiry", "tenor"]
    return out


# =====================================================================
# PART 6 — REGIMES
# =====================================================================

def regime_slice(changes: pd.DataFrame, r: Regime) -> pd.DataFrame:
    return changes.loc[str(r.start):str(r.end)]


def regime_stats(changes: pd.DataFrame, regimes: Sequence[Regime] = None) -> pd.DataFrame:
    """Per-regime vol level, dispersion and effective dimensionality."""
    regimes = regimes or REGIMES
    base = float(changes.std(ddof=1).mean())
    rows = {}
    for r in regimes:
        sub = regime_slice(changes, r).dropna(how="any")
        if len(sub) < 20:
            continue
        rows[r.name] = {
            "kind": r.kind, "n_days": len(sub),
            "mean_std_bp": float(sub.std(ddof=1).mean()),
            "vol_ratio_vs_full": float(sub.std(ddof=1).mean()) / base,
            "mean_corr": float(sub.corr().values[~np.eye(sub.shape[1], dtype=bool)].mean()),
            "eff_dim": effective_dimensionality(sub.corr()),
        }
    return pd.DataFrame(rows).T


def regime_comparison(changes: pd.DataFrame, config: PCAConfig,
                      regimes: Sequence[Regime] = None, k: int = 6,
                      n_draws: int = 25) -> pd.DataFrame:
    """Pairwise subspace overlap between regimes, WITH a split-half noise floor.

    A raw overlap means nothing until you know what two random halves of the
    same regime score. Overlaps inside the floor are NOT evidence of a regime
    change in factor structure.
    """
    regimes = regimes or REGIMES
    fits, floors = {}, {}
    cfg = config.copy(window=0, n_components=k, n_report=k)
    for r in regimes:
        sub = regime_slice(changes, r).dropna(how="any")
        if len(sub) < max(config.min_obs, 3 * k):
            continue
        try:
            fits[r.name] = VolPCA(config=cfg).fit(sub)
            floors[r.name] = split_half_overlap(sub, cfg, k=k, n_draws=n_draws)["mean"]
        except ValueError:
            continue
    names = list(fits)
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                M.loc[a, b] = M.loc[b, a] = subspace_overlap(
                    fits[a].loadings, fits[b].loadings, k=k)
    M.attrs["noise_floor"] = (pd.Series(floors).dropna()
                              if floors else pd.Series(dtype=float))
    M.attrs["fits"] = fits
    return M.round(4)


# =====================================================================
# PART 7 — SCALE MODEL & CORE PCA
# =====================================================================

@dataclass
class ScaleModel:
    """mu and the effective scale s. The bridge between z-space and money.

    Fitted on the estimation window, frozen, applied out of sample. Re-estimated
    each refit so s tracks regimes (max/min ~7.4x across regimes) while staying
    static within a window.

    `weights` gives importance-weighted PCA: s = sigma / w. Larger w -> smaller s
    -> bigger z -> more pull on the factors. The PnL identity is untouched, so
    every downstream routine works on a weighted model unchanged.
    """
    mu: pd.Series
    sigma: pd.Series
    demean: bool = True
    standardize: bool = True
    weights: Optional[pd.Series] = None

    @classmethod
    def fit(cls, changes, demean=True, standardize=True,
            floor_quantile=0.01, weights=None) -> "ScaleModel":
        mu = changes.mean() if demean else pd.Series(0.0, index=changes.columns)
        sd = changes.std(ddof=1)
        if standardize:
            sd = sd.clip(lower=max(sd.quantile(floor_quantile), 1e-8))
        else:
            sd = pd.Series(1.0, index=changes.columns)
        if weights is not None:
            w = weights.reindex(changes.columns)
            if w.isna().any():
                raise ValueError(f"weights missing for {int(w.isna().sum())} points")
            sd = sd / w.clip(lower=1e-6)
        return cls(mu=mu, sigma=sd, demean=demean, standardize=standardize,
                   weights=None if weights is None else weights.reindex(changes.columns))

    def to_z(self, changes: pd.DataFrame) -> pd.DataFrame:
        return (changes[self.mu.index] - self.mu) / self.sigma

    def to_bp(self, z: pd.DataFrame, add_mean: bool = True) -> pd.DataFrame:
        out = z[self.sigma.index] * self.sigma
        return out + self.mu if add_mean else out

    def book_direction(self, vega: pd.Series) -> pd.Series:
        """u = sigma * vega — the only direction in z-space the book cares about."""
        v = vega.reindex(self.sigma.index)
        if v.isna().any():
            warnings.warn(f"vega missing on {int(v.isna().sum())} points, filled 0")
            v = v.fillna(0.0)
        return self.sigma * v


@dataclass
class VolPCA:
    config: PCAConfig = field(default_factory=PCAConfig)
    loadings: Optional[pd.DataFrame] = None
    eigenvalues: Optional[pd.Series] = None
    explained: Optional[pd.Series] = None
    scale: Optional[ScaleModel] = None
    n_signal: Optional[int] = None
    fit_index: Optional[pd.Index] = None
    grid: Optional[pd.MultiIndex] = None
    label: str = "PCA"

    def fit(self, changes, reference: Optional["VolPCA"] = None,
            weights: Optional[pd.Series] = None) -> "VolPCA":
        """Fit on the trailing `window` observations.

        Pass `reference` (the previous refit) in any rolling application, or PC
        labels are not comparable across refits and a 'PC3 hedge' can silently
        become a different hedge overnight.
        """
        cfg = self.config
        ch = changes.sort_index().dropna(how="all")
        if cfg.window and len(ch) > cfg.window:
            ch = ch.iloc[-cfg.window:]
        ch = ch.dropna(axis=0, how="any")
        if len(ch) < cfg.min_obs:
            raise ValueError(f"only {len(ch)} usable rows, need >= {cfg.min_obs}")
        T, p = ch.shape
        if T < p:
            warnings.warn(f"T={T} < p={p}: covariance rank-deficient, trailing PCs are noise")

        self.scale = ScaleModel.fit(ch, demean=cfg.demean,
                                    standardize=cfg.standardize, weights=weights)
        Z = self.scale.to_z(ch)
        lam, V = np.linalg.eigh(np.cov(Z.values, rowvar=False))
        order = np.argsort(lam)[::-1]
        lam, V = np.clip(lam[order], 0.0, None), V[:, order]

        names = [f"PC{i+1}" for i in range(len(lam))]
        self.eigenvalues = pd.Series(lam, index=names)
        self.explained = self.eigenvalues / self.eigenvalues.sum()
        self.loadings = pd.DataFrame(V, index=ch.columns, columns=names)
        self.grid, self.fit_index = ch.columns, ch.index
        self.n_signal = int((lam > marchenko_pastur_bounds(T, p)[1]).sum())

        self._anchor_signs()
        if reference is not None:
            self._align_to(reference)
        return self

    def _anchor_signs(self):
        """Fix the arbitrary eigenvector sign so PC1 is 'vol up'."""
        flip = (self.loadings.sum(axis=0) < 0 if self.config.sign_anchor == "positive_sum"
                else self.loadings.iloc[0] < 0)
        self.loadings.loc[:, flip[flip].index] *= -1.0

    def _align_to(self, reference: "VolPCA"):
        perm, signs = align_pcs(reference.loadings, self.loadings,
                                k=min(self.config.n_report, self.loadings.shape[1]))
        cols = list(self.loadings.columns)
        order = perm + [c for c in cols if c not in perm]
        L = self.loadings[order].copy(); L.columns = cols
        for i, s in enumerate(signs):
            L.iloc[:, i] *= s
        ev = self.eigenvalues[order].copy(); ev.index = cols
        self.loadings, self.eigenvalues = L, ev
        self.explained = self.eigenvalues / self.eigenvalues.sum()

    def transform(self, changes, k: Optional[int] = None) -> pd.DataFrame:
        k = k or self.config.n_components
        Z = self.scale.to_z(changes[self.grid])
        return pd.DataFrame(Z.values @ self.loadings.values[:, :k],
                            index=changes.index, columns=self.loadings.columns[:k])

    def inverse_transform(self, scores, add_mean: bool = False) -> pd.DataFrame:
        k = scores.shape[1]
        Zh = pd.DataFrame(scores.values @ self.loadings.values[:, :k].T,
                          index=scores.index, columns=self.grid)
        return self.scale.to_bp(Zh, add_mean=add_mean)

    def reconstruct(self, changes, k=None) -> pd.DataFrame:
        return self.inverse_transform(self.transform(changes, k=k))

    def residual(self, changes, k=None) -> pd.DataFrame:
        centred = changes[self.grid] - (self.scale.mu if self.config.demean else 0.0)
        return centred - self.reconstruct(changes, k=k)

    def summary(self, k=None) -> pd.DataFrame:
        k = k or self.config.n_report
        out = pd.DataFrame({"eigenvalue": self.eigenvalues.iloc[:k],
                            "surface_var_%": 100 * self.explained.iloc[:k],
                            "cum_%": 100 * self.explained.iloc[:k].cumsum()})
        _, hi = marchenko_pastur_bounds(len(self.fit_index), len(self.grid))
        out["above_MP"] = out["eigenvalue"] > hi
        return out.round(3)

    def __repr__(self):
        if self.loadings is None:
            return "VolPCA(unfitted)"
        return (f"VolPCA({self.label}, p={len(self.grid)}, T={len(self.fit_index)}, "
                f"signal={self.n_signal}, PC1={100*self.explained.iloc[0]:.1f}%)")


# =====================================================================
# PART 8 — IDENTIFICATION ACROSS REFITS
# =====================================================================

def align_pcs(reference: pd.DataFrame, candidate: pd.DataFrame, k: int = 6):
    """Match candidate PCs to reference PCs by absolute overlap (Hungarian).
    Returns (permutation, signs)."""
    k = int(min(k, reference.shape[1], candidate.shape[1]))
    R, Cd = reference.iloc[:, :k].values, candidate.iloc[:, :k].values
    M = R.T @ Cd
    A = np.abs(M)
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-A)
    except Exception:
        cols, used = [], set()
        for i in range(k):
            j = int(np.argmax([A[i, j] if j not in used else -1 for j in range(k)]))
            cols.append(j); used.add(j)
        rows = list(range(k))
    return ([candidate.columns[j] for j in cols],
            [float(np.sign(M[i, j]) or 1.0) for i, j in zip(rows, cols)])


def principal_angles(L1, L2, k: int = 6) -> np.ndarray:
    """Principal angles (deg) between two top-k subspaces. Invariant to ordering
    and to rotation inside the subspace."""
    k = int(min(k, L1.shape[1], L2.shape[1]))
    A = np.linalg.qr(L1.iloc[:, :k].values)[0]
    B = np.linalg.qr(L2.iloc[:, :k].values)[0]
    sv = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))


def subspace_overlap(L1, L2, k: int = 6) -> float:
    """Mean cos^2 of principal angles. Always read against split_half_overlap."""
    return float(np.mean(np.cos(np.radians(principal_angles(L1, L2, k=k))) ** 2))


def split_half_overlap(changes, config: PCAConfig, k: int = 6,
                       n_draws: int = 30, seed: int = 0) -> Dict:
    """Noise floor for subspace overlap: two random halves of the SAME sample.
    Anything inside this distribution is not a finding."""
    ch = changes.dropna(how="any")
    rng = np.random.default_rng(seed)
    n = len(ch)
    # each half is n/2 rows, so the per-fit min_obs must scale down or every draw
    # is rejected and the floor comes back empty
    cfg = config.copy(window=0, n_components=k, n_report=k,
                      min_obs=max(3 * k, min(config.min_obs, n // 2)))
    vals = []
    for _ in range(n_draws):
        idx = rng.permutation(n)
        try:
            a = VolPCA(config=cfg).fit(ch.iloc[np.sort(idx[: n // 2])])
            b = VolPCA(config=cfg).fit(ch.iloc[np.sort(idx[n // 2:])])
            vals.append(subspace_overlap(a.loadings, b.loadings, k=k))
        except ValueError:
            continue
    v = np.array(vals)
    if v.size == 0:
        warnings.warn(f"split-half floor undefined: {n} rows is too few for k={k}")
        return {"mean": np.nan, "p05": np.nan, "p95": np.nan, "draws": v}
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)), "draws": v}


def pc_stability_across_refits(changes, config: PCAConfig, k: int = 6,
                               step: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
    """PER-PC swap rate and overlap across sequential refits.

    Reported per PC, never aggregated: an aggregate is dominated by degenerate
    trailing PCs and gives a backwards answer. swap_rate > 0.10 or
    mean_overlap < 0.90 means you cannot put a NAMED hedge on that factor.
    """
    step = step or config.refit_every
    ch = changes.sort_index().dropna(how="any")
    W = config.window
    if len(ch) < W + step:
        raise ValueError(f"need >= {W+step} rows, have {len(ch)}")
    prev, recs = None, []
    starts = list(range(0, len(ch) - W + 1, step))
    for s in starts:
        m = VolPCA(config=config).fit(ch.iloc[s:s + W])
        if prev is not None:
            perm, signs = align_pcs(prev.loadings, m.loadings, k=k)
            ov = np.abs(np.diag(prev.loadings.iloc[:, :k].values.T @ m.loadings[perm].values))
            for i in range(k):
                recs.append({"pc": f"PC{i+1}", "swapped": perm[i] != f"PC{i+1}",
                             "overlap": float(ov[i]), "sign_flip": signs[i] < 0})
        prev = m
    if not recs:
        raise ValueError("not enough refits to compare")
    df = pd.DataFrame(recs)
    out = df.groupby("pc").agg(swap_rate=("swapped", "mean"),
                               mean_overlap=("overlap", "mean"),
                               min_overlap=("overlap", "min"),
                               sign_flip_rate=("sign_flip", "mean"))
    out = out.reindex([f"PC{i+1}" for i in range(k)])
    out["identified"] = (out["swap_rate"] < 0.10) & (out["mean_overlap"] > 0.90)
    out.attrs["n_refits"] = len(starts)
    if verbose:
        bad = out.index[~out["identified"]].tolist()
        print(f"[identification] {len(starts)} refits, W={W}: "
              f"NOT identified = {bad or 'none'}")
    return out.round(3)


def rolling_factor_share(changes, config: PCAConfig, k: int = 5, step: int = 5) -> pd.DataFrame:
    """Time series of each PC's surface-variance share. A rising PC1 share is the
    classic stress signature."""
    ch = changes.dropna(how="any")
    W = config.window
    cfg = config.copy(n_components=k, n_report=k)
    rows, idx, prev = [], [], None
    for s in range(0, len(ch) - W + 1, step):
        try:
            m = VolPCA(config=cfg).fit(ch.iloc[s:s + W], reference=prev)
        except ValueError:
            continue
        rows.append(100 * m.explained.iloc[:k].values)
        idx.append(ch.index[s + W - 1]); prev = m
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx),
                        columns=[f"PC{i+1}" for i in range(k)])


def score_diagnostics(scores: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """Are PC scores well-behaved enough for the risk numbers to mean anything?
    Negative lag1_ac = stale marks / bid-ask bounce. abs_ac1 = vol clustering."""
    out = {}
    for c in scores.columns:
        v = scores[c].dropna().values
        if len(v) < 20 or np.std(v) < 1e-12:
            continue
        v = (v - v.mean()) / v.std()
        ac = [float(np.corrcoef(v[:-l], v[l:])[0, 1]) for l in range(1, max_lag + 1)]
        av = np.abs(v)
        out[c] = {"lag1_ac": ac[0], "lag5_ac": ac[4] if len(ac) > 4 else np.nan,
                  "max_abs_ac_2to10": float(np.max(np.abs(ac[1:]))),
                  "abs_ac1": float(np.corrcoef(av[:-1], av[1:])[0, 1]),
                  "skew": float((v ** 3).mean()), "kurtosis": float((v ** 4).mean() - 3.0),
                  "ac_band_95": 1.96 / np.sqrt(len(v))}
    return pd.DataFrame(out).T.round(4)


# =====================================================================
# PART 9 — RISK ATTRIBUTION (book PnL, not surface variance)
# =====================================================================

def pc_risk_exposure(model: VolPCA, vega: pd.Series, k: Optional[int] = None) -> pd.DataFrame:
    """Rank factors by share of BOOK PnL variance, alongside surface variance.
    Lead with this table, not the scree plot — the two rankings can be opposite.

    exposure_$ : u'L_k, dollar PnL for a +1 sd move of PC k. This is what you hedge.
    """
    k = k or model.config.n_report
    u = model.scale.book_direction(vega)
    expo = pd.Series(model.loadings.values.T @ u.values, index=model.loadings.columns)
    var = expo ** 2 * model.eigenvalues
    total = float(var.sum())
    out = pd.DataFrame({
        "exposure_$": expo.iloc[:k], "pnl_var_$^2": var.iloc[:k],
        "pnl_var_%": 100 * var.iloc[:k] / total if total > 0 else np.nan,
        "surface_var_%": 100 * model.explained.iloc[:k]})
    out["pnl_cum_%"] = out["pnl_var_%"].cumsum()
    out["risk_rank"] = out["pnl_var_%"].rank(ascending=False).astype(int)
    out["surface_rank"] = out["surface_var_%"].rank(ascending=False).astype(int)
    out.attrs["total_pnl_var"] = total
    out.attrs["daily_pnl_sd"] = float(np.sqrt(total))
    return out.round(4)


def book_direction_alignment(model: VolPCA, vega: pd.Series, k: Optional[int] = None) -> pd.DataFrame:
    """Where u sits in the PC basis — the analytic version of the backtest.

    PnL = u'z, so a k-factor model explains PnL iff u lies in span(top-k).
    pnl_r2_implied should reproduce the walk-forward R2 curve; if it does not,
    one of the two is wrong.
    """
    k = k or model.config.n_report
    u = model.scale.book_direction(vega)
    proj = pd.Series(model.loadings.values.T @ u.values, index=model.loadings.columns)
    un2 = float(u.values @ u.values)
    var = proj ** 2 * model.eigenvalues
    total = float(var.sum())
    out = pd.DataFrame({"cos2": (proj.iloc[:k] ** 2) / un2,
                        "cos2_cum": ((proj ** 2) / un2).cumsum().iloc[:k],
                        "pnl_var_%": 100 * var.iloc[:k] / total})
    out["pnl_r2_implied"] = (var.cumsum() / total).iloc[:k]
    out.attrs["u_norm"] = float(np.sqrt(un2))
    out.attrs["gross_vega"] = float(vega.abs().sum())
    out.attrs["net_vega"] = float(vega.sum())
    return out.round(4)


def horizon_comparison(changes: pd.DataFrame, vega: pd.Series, config: PCAConfig,
                       horizons: Sequence[int] = (1, 5, 10, 21),
                       k_list: Sequence[int] = (3, 6)) -> pd.DataFrame:
    """Does the factor model explain PnL better over longer holding periods?
    If R2 rises with horizon, the 1-day residual is marking noise that washes out."""
    ch = changes.dropna(how="any")
    rows = []
    for h in horizons:
        agg = ch.rolling(h).sum().dropna(how="any").iloc[::h]
        if len(agg) < config.min_obs:
            continue
        m = VolPCA(config=config.copy(window=0, n_components=max(k_list),
                                      n_report=max(k_list))).fit(agg)
        al = book_direction_alignment(m, vega, k=max(k_list))
        r = {"horizon_days": h, "n_obs": len(agg)}
        for k in k_list:
            r[f"pnl_r2_k{k}"] = float(al["pnl_r2_implied"].iloc[k - 1])
        r["PC1_surface_%"] = 100 * m.explained.iloc[0]
        rows.append(r)
    return pd.DataFrame(rows).set_index("horizon_days").round(4)


def scope_check(vega: pd.Series, grid: pd.MultiIndex) -> pd.DataFrame:
    """How much of the book's vega actually sits on the modelled grid?
    Anything off-grid is risk the model cannot see."""
    on = vega.reindex(grid).fillna(0.0)
    return pd.DataFrame({
        "gross_total": [float(vega.abs().sum())],
        "gross_on_grid": [float(on.abs().sum())],
        "share_on_grid": [float(on.abs().sum() / max(vega.abs().sum(), 1e-12))],
        "net_total": [float(vega.sum())], "net_on_grid": [float(on.sum())],
    }).round(4)


# =====================================================================
# PART 10 — HEDGE CONSTRUCTION
# =====================================================================

def unit_vega_instruments(grid: pd.MultiIndex, subset=None) -> pd.DataFrame:
    """Candidate hedges as unit-vega vectors, one per node. Replace with the
    desk's real instrument vega profiles when hedging with actual straddles."""
    subset = list(grid) if subset is None else [tuple(s) for s in subset]
    subset = [s for s in subset if s in set(grid)]
    M = pd.DataFrame(0.0, index=grid, columns=pd.MultiIndex.from_tuples(
        subset, names=["expiry", "tenor"]))
    for s in subset:
        M.loc[s, s] = 1.0
    return M


def hedge_exposure_matrix(model: VolPCA, instruments: pd.DataFrame, k=None) -> pd.DataFrame:
    """A[k,j] = factor-k exposure of one unit of instrument j. A = L' diag(sigma) W.
    Note the sigma: long-expiry instruments carry little factor exposure per unit
    of vega, so the solver keeps wanting short-expiry ones (which are collinear)."""
    k = k or model.config.n_components
    W = instruments.reindex(model.grid).fillna(0.0)
    SW = W.mul(model.scale.sigma, axis=0)
    return pd.DataFrame(model.loadings.values[:, :k].T @ SW.values,
                        index=model.loadings.columns[:k], columns=instruments.columns)


def select_hedge_instruments(model: VolPCA, candidates: pd.DataFrame, k=None,
                             n_select=None, verbose=True) -> Dict:
    """Greedily pick instruments that make A well-conditioned.
    Watch min_singular: if tiny, the candidate set has no distinct direction for
    one factor and the solver answers with huge offsetting notionals."""
    k = k or model.config.n_components
    n_select = n_select or k
    A_all = hedge_exposure_matrix(model, candidates, k=k)
    cols = list(A_all.columns)
    if n_select > len(cols):
        raise ValueError(f"asked for {n_select} from {len(cols)} candidates")
    chosen = []
    for _ in range(n_select):
        best, best_score = None, np.inf
        for c in cols:
            if c in chosen:
                continue
            sv = np.linalg.svd(A_all[chosen + [c]].values, compute_uv=False)
            score = -sv[-1] if len(chosen) + 1 < k else sv[0] / max(sv[-1], 1e-12)
            if score < best_score:
                best, best_score = c, score
        chosen.append(best)
    A = A_all[chosen]
    sv = np.linalg.svd(A.values, compute_uv=False)
    sv_n = np.linalg.svd(A_all[cols[:n_select]].values, compute_uv=False)
    out = {"instruments": chosen, "A": A,
           "kappa": float(sv[0] / max(sv[-1], 1e-12)),
           "kappa_naive": float(sv_n[0] / max(sv_n[-1], 1e-12)),
           "singular_values": sv, "min_singular": float(sv[-1])}
    if verbose:
        print(f"[hedge] kappa(selected) = {out['kappa']:,.1f}  "
              f"vs first-{n_select} = {out['kappa_naive']:,.1f}")
        if sv[-1] < 0.10 * sv[0]:
            print(f"[hedge] WARNING min singular {sv[-1]:.3f} vs max {sv[0]:.3f} — widen candidates")
    return out


def solve_hedge(model: VolPCA, vega: pd.Series, instruments: pd.DataFrame,
                k=None, ridge: float = 0.0, verbose: bool = True) -> Dict:
    """Solve A h = -b for hedge notionals. Reports the honest costs: residual
    exposure, gross notional ratio, and the variance BEYOND PC k that the hedge
    cannot touch."""
    k = k or model.config.n_components
    u = model.scale.book_direction(vega)
    b = pd.Series(model.loadings.values[:, :k].T @ u.values,
                  index=model.loadings.columns[:k])
    A = hedge_exposure_matrix(model, instruments, k=k)
    if ridge > 0:
        h = -A.values.T @ np.linalg.solve(A.values @ A.values.T + ridge * np.eye(k), b.values)
    else:
        h, *_ = np.linalg.lstsq(A.values, -b.values, rcond=None)
    h = pd.Series(h, index=A.columns)
    resid = b + pd.Series(A.values @ h.values, index=b.index)
    hv = float((resid ** 2 * model.eigenvalues.iloc[:k]).sum())
    bv = float((b ** 2 * model.eigenvalues.iloc[:k]).sum())
    proj = model.loadings.values.T @ u.values
    var_all = proj ** 2 * model.eigenvalues.values
    out = {"notionals": h.round(4), "book_exposure": b.round(4),
           "residual_exposure": resid.round(6),
           "hedge_vega": instruments.reindex(model.grid).fillna(0.0).mul(h, axis=1).sum(axis=1),
           "gross_notional": float(h.abs().sum()),
           "gross_ratio": float(h.abs().sum() / max(vega.abs().sum(), 1e-12)),
           "factor_var_removed_%": 100 * (1 - hv / bv) if bv > 0 else np.nan,
           "unhedged_var_%": 100 * float(var_all[k:].sum() / max(var_all.sum(), 1e-12))}
    if verbose:
        print(f"[hedge] removes {out['factor_var_removed_%']:.1f}% of top-{k} factor PnL var; "
              f"{out['unhedged_var_%']:.1f}% sits beyond PC{k} and is untouched; "
              f"gross {out['gross_ratio']:.2f}x book")
    return out


# =====================================================================
# PART 11 — ROLLING REFIT & WALK-FORWARD PnL EXPLAIN
# =====================================================================

def rolling_refit(changes, config: PCAConfig, verbose=True) -> List[VolPCA]:
    """Sequential refits, each aligned to the previous so 'PC3' means one thing."""
    ch = changes.sort_index().dropna(how="any")
    W, step = config.window, config.refit_every
    if len(ch) < W + 1:
        raise ValueError(f"need > {W} rows, have {len(ch)}")
    models, prev = [], None
    for s in range(0, len(ch) - W + 1, step):
        prev = VolPCA(config=config).fit(ch.iloc[s:s + W], reference=prev)
        models.append(prev)
    if verbose:
        print(f"[rolling] {len(models)} refits, W={W}, step={step}")
    return models


def pnl_explain_backtest(changes, vega, config: PCAConfig,
                         k_list: Sequence[int] = (1, 2, 3, 5, 6, 8), verbose=True) -> Dict:
    """Walk-forward: fit on trailing window, explain the NEXT refit_every days.
    Strictly out of sample. `vega` may be a Series (static book) or a DataFrame
    indexed by date (the book as it actually was — the honest version)."""
    ch = changes.sort_index().dropna(how="any")
    W, step, kmax = config.window, config.refit_every, max(k_list)
    cfg = config.copy(n_components=kmax, n_report=kmax)
    static = isinstance(vega, pd.Series)
    recs, prev = [], None
    for s in range(0, len(ch) - W - step + 1, step):
        train, test = ch.iloc[s:s + W], ch.iloc[s + W: s + W + step]
        if test.empty:
            break
        prev = m = VolPCA(config=cfg).fit(train, reference=prev)
        for t in test.index:
            if static:
                v = vega
            else:
                if t not in vega.index:
                    continue
                v = vega.reindex(columns=m.grid).loc[t]
            v = v.reindex(m.grid).fillna(0.0)
            dv = test.loc[t, m.grid]
            u = m.scale.book_direction(v)
            z = ((dv - m.scale.mu) / m.scale.sigma).values
            sc = m.loadings.values.T @ z
            ex = m.loadings.values.T @ u.values
            row = {"date": t, "actual": float(v.values @ dv.values),
                   "refit_date": m.fit_index[-1]}
            for k in k_list:
                row[f"pred_{k}"] = float(ex[:k] @ sc[:k])
            recs.append(row)
    if not recs:
        msg = (f"no test observations. changes {ch.index[0].date()}..{ch.index[-1].date()} "
               f"({len(ch)} rows), window={W}")
        if not static:
            ov = ch.index.intersection(vega.index)
            msg += (f"; vega {vega.index[0].date()}..{vega.index[-1].date()} "
                    f"({len(vega)} rows), overlap={len(ov)} days. "
                    f"Need overlap > window+refit_every = {W+step}. "
                    f"Use a static book (aggregate='ewma') or shorten the window.")
        raise ValueError(msg)
    df = pd.DataFrame(recs).set_index("date")
    ss = float(((df["actual"] - df["actual"].mean()) ** 2).sum())
    r2 = {k: 1 - float(((df["actual"] - df[f"pred_{k}"]) ** 2).sum()) / ss if ss > 0 else np.nan
          for k in k_list}
    r2s = pd.Series(r2, name="pnl_r2"); r2s.index.name = "n_factors"
    df["unexplained"] = df["actual"] - df[f"pred_{kmax}"]
    df["days_since_refit"] = (df.index - df["refit_date"]).dt.days
    if verbose:
        for k, v in r2.items():
            print(f"[backtest] {k:>2} factors: R2 = {v:6.3f}"
                  f"{'   <-- worse than predicting zero' if v < 0 else ''}")
    return {"r2": r2s, "daily": df, "n_obs": len(df)}


# =====================================================================
# PART 12 — WINDOW ROBUSTNESS & COVARIANCE FORECASTING
# =====================================================================

def factor_covariance(model: VolPCA, k: int, include_residual: bool = True) -> np.ndarray:
    """bp-space covariance implied by a k-factor model. This is what competes
    with the raw sample covariance as a FORECAST."""
    L = model.loadings.values[:, :k]
    Cz = (L * model.eigenvalues.values[:k]) @ L.T
    if include_residual:
        r = max(float(model.eigenvalues.values[k:].sum()), 0.0) / max(len(model.grid) - k, 1)
        Cz = Cz + np.eye(len(model.grid)) * r
    D = model.scale.sigma.values
    return (Cz * D) * D[:, None]


def _mvp_weights(S: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    p = S.shape[0]
    try:
        x = np.linalg.solve(S + ridge * np.trace(S) / p * np.eye(p), np.ones(p))
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(S) @ np.ones(p)
    return x / x.sum()


def covariance_forecast_test(changes, windows: Sequence[int],
                             k_list: Sequence[int] = (3, 5, 6, 10),
                             horizon: int = 21, step: int = 21,
                             n_portfolios: int = 200, seed: int = 0,
                             verbose: bool = True) -> pd.DataFrame:
    """Does a window of length W forecast FUTURE covariance?

    Walk forward: estimate on trailing W days, score on the next `horizon` days.
    mvp_oos_vol : realized OOS vol of the min-variance portfolio. Lower = better.
                  The most demanding test — MVP weights load on the smallest,
                  worst-estimated eigen-directions.
    calib_slope : realized-on-predicted variance slope across random portfolios.
    estimator "sample" is the raw sample covariance — the benchmark to beat.
    """
    ch = changes.dropna(how="any").sort_index()
    rng = np.random.default_rng(seed)
    p = ch.shape[1]
    Wr = rng.standard_normal((p, n_portfolios))
    Wr /= np.abs(Wr).sum(axis=0, keepdims=True)
    recs = []
    for W in windows:
        if len(ch) < W + horizon:
            if verbose:
                print(f"[cov] window {W} skipped — not enough data")
            continue
        cfg = PCAConfig(window=W, n_components=max(k_list), n_report=max(k_list))
        for s in range(0, len(ch) - W - horizon + 1, step):
            train, test = ch.iloc[s:s + W], ch.iloc[s + W: s + W + horizon]
            if len(test) < max(5, horizon // 2):
                continue
            try:
                m = VolPCA(config=cfg).fit(train)
            except ValueError:
                continue
            Xt = test.values - train.values.mean(axis=0, keepdims=True)
            S_real = (Xt.T @ Xt) / max(len(Xt) - 1, 1)
            real = np.einsum("ij,jk,ik->i", Wr.T, S_real, Wr.T)
            cands = {f"k={k}": factor_covariance(m, k) for k in k_list}
            cands["sample"] = np.cov(train.values, rowvar=False)
            for name, S in cands.items():
                w = _mvp_weights(S)
                pred = np.einsum("ij,jk,ik->i", Wr.T, S, Wr.T)
                ok = (pred > 0) & np.isfinite(real)
                recs.append({"window": W, "estimator": name,
                             "mvp_oos_vol": float(np.std(test.values @ w, ddof=1)),
                             "mvp_pred_vol": float(np.sqrt(max(w @ S @ w, 0))),
                             "calib_slope": float(np.polyfit(pred[ok], real[ok], 1)[0])
                             if ok.sum() > 10 else np.nan,
                             "frob_rel_err": float(np.linalg.norm(S - S_real) /
                                                   max(np.linalg.norm(S_real), 1e-12))})
    if not recs:
        raise ValueError("no evaluation windows — sample too short")
    out = pd.DataFrame(recs).groupby(["window", "estimator"]).agg(
        mvp_oos_vol=("mvp_oos_vol", "mean"), mvp_pred_vol=("mvp_pred_vol", "mean"),
        calib_slope=("calib_slope", "median"), frob_rel_err=("frob_rel_err", "median"),
        n_evals=("mvp_oos_vol", "size")).reset_index()
    out["vol_underestimate_x"] = out["mvp_oos_vol"] / out["mvp_pred_vol"]
    if verbose:
        best = out.loc[out.groupby("estimator")["mvp_oos_vol"].idxmin()]
        for _, r in best.iterrows():
            print(f"[cov] {r['estimator']:>8}: best W={int(r['window'])}, "
                  f"oos_vol={r['mvp_oos_vol']:.4f}, calib={r['calib_slope']:.2f}")
    return out.round(4)


def window_stability_curve(changes, windows: Sequence[int], k: int = 6,
                           step: int = 21, verbose: bool = True) -> pd.DataFrame:
    """Subspace overlap between CONSECUTIVE refits, vs window length.
    The estimation-noise half of the bias/variance picture; pair with
    covariance_forecast_test, which carries the bias half."""
    ch = changes.dropna(how="any")
    recs = []
    for W in windows:
        if len(ch) < W + step:
            continue
        cfg = PCAConfig(window=W, n_components=k, n_report=k)
        prev, ovs, angs, evs = None, [], [], []
        for s in range(0, len(ch) - W + 1, step):
            try:
                m = VolPCA(config=cfg).fit(ch.iloc[s:s + W])
            except ValueError:
                continue
            if prev is not None:
                ovs.append(subspace_overlap(prev.loadings, m.loadings, k=k))
                angs.append(principal_angles(prev.loadings, m.loadings, k=k).max())
            evs.append(m.explained.iloc[:k].values)
            prev = m
        if ovs:
            recs.append({"window": W, "n_refits": len(ovs) + 1,
                         "subspace_overlap": float(np.mean(ovs)),
                         "overlap_min": float(np.min(ovs)),
                         "max_principal_angle_deg": float(np.mean(angs)),
                         "pc1_share": float(np.mean([e[0] for e in evs])),
                         "topk_share": float(np.mean([e.sum() for e in evs]))})
    return pd.DataFrame(recs).round(4)


def eigen_decay_table(changes, windows: Sequence[int], k: int = 12) -> pd.DataFrame:
    """Explained-variance profile, MP edge and effective dimension per window.
    n_above_MP rises with the window because the noise bulk narrows as T grows —
    that is a sample-size statement, not the market simplifying."""
    ch = changes.dropna(how="any")
    rows = {}
    for W in windows:
        if len(ch) < W:
            continue
        m = VolPCA(config=PCAConfig(window=W, n_components=k, n_report=k)).fit(ch)
        T, p = len(m.fit_index), len(m.grid)
        rows[W] = {**{f"PC{i+1}": 100 * m.explained.iloc[i]
                      for i in range(min(k, len(m.explained)))},
                   "n_above_MP": m.n_signal,
                   "MP_edge": marchenko_pastur_bounds(T, p)[1],
                   "T/p": T / p,
                   "eff_dim": effective_dimensionality(m.scale.to_z(ch.iloc[-W:]).corr())}
    return pd.DataFrame(rows).T.round(3)


# =====================================================================
# PART 13 — TWO-STAGE SWOPT -> IRG HIERARCHICAL MODEL
# =====================================================================

def two_stage_model(ch_swopt: pd.DataFrame, ch_irg: pd.DataFrame,
                    config: PCAConfig, k_basis: int = 2, verbose: bool = True) -> Dict:
    """SWOPT factors first, then basis factors from the IRG residual.

    Rationale: a naive blend lets the (larger, noisier) IRG block distort the
    SWOPT factors, and the shared-pillar seam creates near-collinear columns.
    Two-stage keeps SWOPT factors clean and carries IRG's incremental risk
    explicitly as a basis factor.
    """
    common = ch_swopt.index.intersection(ch_irg.index)
    sw, ir = ch_swopt.loc[common].dropna(how="any"), ch_irg.loc[common]
    ir = ir.loc[sw.index].dropna(how="any")
    sw = sw.loc[ir.index]

    m_sw = VolPCA(config=config).fit(sw)
    scores = m_sw.transform(sw, k=config.n_components)

    X = np.column_stack([np.ones(len(scores)), scores.values])
    beta, *_ = np.linalg.lstsq(X, ir.values, rcond=None)
    fitted = pd.DataFrame(X @ beta, index=ir.index, columns=ir.columns)
    resid = ir - fitted

    m_basis = VolPCA(config=config.copy(n_components=k_basis, n_report=k_basis)).fit(resid)
    explained = 1 - float(resid.var().sum()) / float(ir.var().sum())
    if verbose:
        print(f"[two-stage] SWOPT factors explain {100*explained:.1f}% of IRG variance; "
              f"{k_basis} basis factors capture "
              f"{100*m_basis.explained.iloc[:k_basis].sum():.1f}% of the residual")
    return {"swopt_model": m_sw, "basis_model": m_basis, "betas": pd.DataFrame(
        beta[1:], index=scores.columns, columns=ir.columns),
        "residual": resid, "irg_explained_by_swopt": explained}


def two_stage_risk_split(res: Dict, vega_swopt: pd.Series, vega_irg: pd.Series) -> pd.DataFrame:
    """Split book PnL variance between SWOPT factors and the IRG basis factors."""
    m_sw, m_b = res["swopt_model"], res["basis_model"]
    a = pc_risk_exposure(m_sw, vega_swopt, k=m_sw.config.n_components)
    b = pc_risk_exposure(m_b, vega_irg.reindex(m_b.grid).fillna(0.0),
                         k=m_b.config.n_components)
    tot = a.attrs["total_pnl_var"] + b.attrs["total_pnl_var"]
    return pd.DataFrame({
        "block": ["SWOPT factors", "IRG basis"],
        "pnl_var": [a.attrs["total_pnl_var"], b.attrs["total_pnl_var"]],
        "share_%": [100 * a.attrs["total_pnl_var"] / tot, 100 * b.attrs["total_pnl_var"] / tot],
        "daily_sd": [a.attrs["daily_pnl_sd"], b.attrs["daily_pnl_sd"]]}).round(3)


# =====================================================================
# PART 14 — WEIGHTED (TRADER) MODEL: beta + time-weighted vega
# =====================================================================
# An importance-weighted PCA is just a PCA with a different scale vector:
#   standard  z = (dv - mu)/sigma        weighted  z = (dv - mu)/(sigma/w)
# So it is a different ScaleModel and every routine above works on it unchanged.

def rolling_beta(changes, benchmark: Tuple[str, str] = ("1Y", "10Y"), window: int = 90,
                 asof=None, verbose: bool = True) -> pd.Series:
    """Beta of every node to a benchmark node over a trailing window.
    Slope in bp-per-bp: when the benchmark moves 1bp, this node moves beta bp."""
    ch = changes.sort_index()
    if asof is not None:
        ch = ch.loc[:pd.Timestamp(asof)]
    ch = ch.iloc[-window:].dropna(how="any")
    if benchmark not in ch.columns:
        raise KeyError(f"benchmark {benchmark} not on the grid")
    if len(ch) < max(20, window // 3):
        raise ValueError(f"only {len(ch)} observations for a {window}d beta")
    b = ch[benchmark]
    var_b = float(b.var(ddof=1))
    beta = ch.apply(lambda c: float(c.cov(b)) / var_b)
    if verbose:
        print(f"[beta] {len(ch)}d to {benchmark[0]}x{benchmark[1]}: "
              f"{beta.min():.2f}–{beta.max():.2f}, median {beta.median():.2f}")
    return beta.rename("beta")


def time_weighted_vega(vega, grid: Optional[pd.MultiIndex] = None, mode: str = "ewma",
                       halflife: float = 63.0, expiry_power: float = 0.5,
                       verbose: bool = True) -> pd.Series:
    """AMBIGUOUS TERM — both readings implemented, confirm which the trader means.

    mode="ewma"   : time-decayed average of HISTORICAL vega (time = recency).
                    Tilts toward where the book has been positioned.
    mode="expiry" : vega * (expiry years)**expiry_power, the sqrt(T) convention
                    (time = time to expiry). Tilts long regardless of position.
    mode="both"   : ewma then expiry scaling.
    """
    if grid is not None:
        v = align_vega(vega, grid, aggregate=("ewma" if mode in ("ewma", "both") else "last"),
                       halflife=halflife, verbose=False)
        if isinstance(v, pd.DataFrame):
            v = (v.ewm(halflife=halflife).mean().iloc[-1]
                 if mode in ("ewma", "both") else v.iloc[-1])
    else:
        v = vega if isinstance(vega, pd.Series) else to_panel(vega).ewm(
            halflife=halflife).mean().iloc[-1]
    if mode in ("expiry", "both"):
        yrs = pd.Series([max(tenor_to_years(e), 1e-6) for e, _ in v.index], index=v.index)
        v = v * yrs ** expiry_power
    if verbose:
        print(f"[tw-vega] mode={mode}, gross {v.abs().sum():,.1f}")
    return v.rename("tw_vega")


def trader_weights(changes, vega, benchmark: Tuple[str, str] = ("1Y", "10Y"),
                   beta_window: int = 90, lam: float = 0.5, tw_mode: str = "ewma",
                   halflife: float = 63.0, expiry_power: float = 0.5,
                   floor: float = 0.05, cap: Optional[float] = 10.0,
                   asof=None, verbose: bool = True) -> pd.DataFrame:
    """w = lam * |beta|_norm + (1-lam) * |tw_vega|_norm.

    lam=1 pure beta (book-independent), lam=0 pure time-weighted vega.
    Each component is normalised to mean 1 before blending, or lam would not
    actually control the mix. Absolute values: a weight must be positive or the
    scale s = sigma/w flips sign and corrupts the PnL identity.
    floor/cap stop a concentrated book from collapsing the PCA onto one node.
    """
    beta = rolling_beta(changes, benchmark=benchmark, window=beta_window,
                        asof=asof, verbose=verbose)
    twv = time_weighted_vega(vega, grid=changes.columns, mode=tw_mode,
                             halflife=halflife, expiry_power=expiry_power, verbose=verbose)

    def _n(x):
        a = x.abs(); m = float(a.mean())
        return a / m if m > 1e-12 else pd.Series(1.0, index=x.index)

    bn, vn = _n(beta), _n(twv.reindex(beta.index).fillna(0.0))
    w = (lam * bn + (1.0 - lam) * vn)
    w = (w / w.mean()).clip(lower=floor, upper=cap)
    w = w / w.mean()
    if verbose:
        print(f"[weights] lam={lam}: range {w.min():.2f}–{w.max():.2f}, "
              f"top: {[f'{a}x{b}' for a, b in w.nlargest(3).index]}")
    return pd.DataFrame({"beta": beta, "beta_norm": bn,
                         "tw_vega": twv.reindex(beta.index), "tw_vega_norm": vn,
                         "weight": w})


def fit_weighted_model(changes, weights: pd.Series, config: PCAConfig,
                       base: str = "sigma", reference=None, label="weighted") -> VolPCA:
    """base='sigma' -> s = sigma/w (keeps the correlation PCA, adds importance).
    base='unit'  -> s = 1/w (raw bp variance dominates again)."""
    cfg = config if base == "sigma" else config.copy(standardize=False)
    m = VolPCA(config=cfg).fit(changes, reference=reference, weights=weights)
    m.label = label
    return m


def compare_models(models: Dict[str, VolPCA], vega: pd.Series,
                   k_list: Sequence[int] = (1, 3, 6)) -> pd.DataFrame:
    """Head-to-head, computed by the SAME code for every model."""
    rows = {}
    for name, m in models.items():
        al = book_direction_alignment(m, vega, k=max(k_list) + 2)
        r = {"PC1_surface_%": 100 * m.explained.iloc[0],
             "top6_surface_%": 100 * m.explained.iloc[:6].sum(),
             "n_signal_PCs": m.n_signal}
        for k in k_list:
            r[f"PnL_R2_k{k}"] = float(al["pnl_r2_implied"].iloc[k - 1])
        rows[name] = r
    return pd.DataFrame(rows).T.round(4)
