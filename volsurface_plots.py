"""
volsurface_plots.py — every chart for the vol-surface PCA project.

All functions take objects produced by volsurface.py and return a matplotlib
Figure. No computation beyond what a chart needs.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import volsurface as vs

BLUE, RED, GREEN, ORANGE, PURPLE, GREY = (
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#94A3B8")

__all__ = [
    "set_style", "plot_surface_snapshot", "plot_node_timeseries", "plot_dispersion",
    "plot_staleness", "plot_point_stats", "plot_weekday_profile",
    "plot_calendar_acceptance", "plot_calendar_scorecard",
    "plot_correlation", "plot_neighbour_correlation", "plot_transform_comparison",
    "plot_normality", "plot_regime_stats", "plot_regime_overlap",
    "plot_scree", "plot_loadings", "plot_eigen_decay", "plot_score_diagnostics",
    "plot_identification", "plot_rolling_share", "plot_loading_drift",
    "plot_risk_ranking", "plot_book_alignment", "plot_horizon",
    "plot_hedge_report", "plot_hedge_churn",
    "plot_backtest", "plot_window_robustness", "plot_covariance_forecast",
    "plot_subspace_matrix", "plot_weight_diagnostics", "plot_lambda_sweep",
    "plot_model_comparison",
]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "figure.figsize": (14, 5),
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    })


def _grid_2d(series: pd.Series) -> pd.DataFrame:
    m = series.unstack("tenor")
    return m.reindex(index=sorted(m.index, key=vs.maturity_to_days),
                     columns=sorted(m.columns, key=vs.maturity_to_days))


def _heat(ax, M: pd.DataFrame, title, cmap="RdBu_r", diverging=True, fs=6):
    if diverging:
        v = float(np.nanmax(np.abs(M.values)))
        im = ax.imshow(M.values, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    else:
        im = ax.imshow(M.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns, rotation=90, fontsize=fs)
    ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index, fontsize=fs)
    ax.set_title(title); ax.set_xlabel("tenor"); ax.grid(False)
    return im


# ---------------------------------------------------------------- 1. surface
def plot_surface_snapshot(levels: pd.DataFrame, date=None):
    d = levels.index[-1] if date is None else pd.Timestamp(date)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    im = _heat(ax[0], _grid_2d(levels.loc[d]), f"Vol surface {pd.Timestamp(d).date()} (bp/annum)",
               cmap="viridis", diverging=False)
    ax[0].set_ylabel("expiry"); fig.colorbar(im, ax=ax[0], fraction=0.046)
    for tn in sorted(set(levels.columns.get_level_values("tenor")), key=vs.maturity_to_days)[::3]:
        sub = levels.loc[d, [c for c in levels.columns if c[1] == tn]]
        x = [vs.maturity_to_days(e) / 365.0 for e, _ in sub.index]
        ax[1].semilogx(x, sub.values, "o-", ms=3, label=f"{tn} tenor")
    ax[1].set_xlabel("expiry (years, log)"); ax[1].set_ylabel("bp/annum")
    ax[1].set_title("Term structure by tenor"); ax[1].legend()
    fig.tight_layout(); return fig


def plot_node_timeseries(levels, nodes: Sequence[Tuple[str, str]], changes=None):
    n = 2 if changes is not None else 1
    fig, ax = plt.subplots(1, n, figsize=(7 * n, 4), squeeze=False)
    for nd in nodes:
        if nd in levels.columns:
            ax[0, 0].plot(levels.index, levels[nd], lw=1, label=f"{nd[0]}x{nd[1]}")
    ax[0, 0].legend(); ax[0, 0].set_ylabel("bp/annum"); ax[0, 0].set_title("Levels")
    if changes is not None:
        for nd in nodes:
            if nd in changes.columns:
                ax[0, 1].plot(changes.index, changes[nd], lw=0.6, alpha=0.8,
                              label=f"{nd[0]}x{nd[1]}")
        ax[0, 1].legend(); ax[0, 1].set_ylabel("bp"); ax[0, 1].set_title("Daily changes")
    fig.tight_layout(); return fig


def plot_dispersion(disp: pd.Series, regimes: Optional[Sequence] = None):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(disp.index, disp.values, color=RED, lw=1.2)
    ax.set_ylabel("bp"); ax.set_title("Cross-sectional dispersion of daily changes (rolling)")
    if regimes:
        for r in regimes:
            ax.axvspan(pd.Timestamp(r.start), pd.Timestamp(r.end),
                       alpha=0.13, color=RED if r.kind == "stress" else GREEN)
            ax.text(pd.Timestamp(r.start), ax.get_ylim()[1] * 0.96, r.name,
                    fontsize=6, rotation=90, va="top")
    fig.tight_layout(); return fig


def plot_staleness(stale: pd.DataFrame, vega: Optional[pd.Series] = None):
    n = 2 if vega is not None else 1
    fig, ax = plt.subplots(1, n, figsize=(6.5 * n, 4.5), squeeze=False)
    im = _heat(ax[0, 0], stale, "Share of days with zero change (staleness)",
               cmap="Reds", diverging=False)
    ax[0, 0].set_ylabel("expiry"); fig.colorbar(im, ax=ax[0, 0], fraction=0.046)
    if vega is not None:
        im2 = _heat(ax[0, 1], _grid_2d(vega.abs()), "|book vega| — does it overlap the stale nodes?",
                    cmap="Blues", diverging=False)
        fig.colorbar(im2, ax=ax[0, 1], fraction=0.046)
    fig.tight_layout(); return fig


def plot_point_stats(stats: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    im = _heat(ax[0], _grid_2d(stats["std"]), "Daily change std (bp)", cmap="viridis", diverging=False)
    ax[0].set_ylabel("expiry"); fig.colorbar(im, ax=ax[0], fraction=0.046)
    im = _heat(ax[1], _grid_2d(stats["kurtosis"]), "Excess kurtosis", cmap="magma", diverging=False)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    im = _heat(ax[2], _grid_2d(stats["lag1_ac"]), "Lag-1 autocorrelation")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- 2. calendar
def plot_weekday_profile(prof: pd.DataFrame):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    idx = sorted(prof.index, key=vs.maturity_to_days)
    P = prof.loc[idx]
    im = ax[0].imshow(P[days].values.astype(float), cmap="RdBu_r", aspect="auto",
                      vmin=-np.nanmax(np.abs(P[days].values.astype(float))),
                      vmax=np.nanmax(np.abs(P[days].values.astype(float))))
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(days)
    ax[0].set_yticks(range(len(idx))); ax[0].set_yticklabels(idx, fontsize=7)
    ax[0].set_title("Mean daily change by weekday, per expiry"); ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    cols = [RED if f else GREY for f in P["flagged"]]
    ax[1].barh(range(len(idx)), P["spread"].values.astype(float), color=cols)
    ax[1].set_yticks(range(len(idx))); ax[1].set_yticklabels(idx, fontsize=7)
    ax[1].set_xlabel("max-min weekday mean (bp)")
    ax[1].set_title("Weekday spread — red = flagged as calendar-affected")
    fig.tight_layout(); return fig


def plot_calendar_acceptance(cal_raw: pd.DataFrame, cal_clean: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    idx = cal_clean.index
    x = np.arange(len(idx))
    ax.bar(x - 0.2, cal_raw["max_abs"].reindex(idx), 0.4, label="raw", color=GREY)
    ax.bar(x + 0.2, cal_clean["max_abs"], 0.4, label="cleaned", color=RED)
    for i, sig in enumerate(cal_clean["signal"]):
        if not sig:
            ax.axvspan(i - 0.5, i + 0.5, color="grey", alpha=0.18)
    ax.set_xticks(x); ax.set_xticklabels(idx); ax.legend()
    ax.set_ylabel("max |corr| with a weekday dummy")
    ax.set_title(f"Calendar leakage: signal-PC worst {cal_raw.attrs['worst_signal']:.3f}"
                 f" -> {cal_clean.attrs['worst_signal']:.3f}   (grey = noise PCs, ignore)")
    fig.tight_layout(); return fig


def plot_calendar_scorecard(score: pd.DataFrame):
    cols = ["weekday_leak_signal", "var_retained_%", "PC1_share_%", "eff_dim"]
    fig, ax = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.8))
    for a, c in zip(ax, cols):
        a.bar(range(len(score)), score[c].values, color=BLUE)
        a.set_xticks(range(len(score))); a.set_xticklabels(score.index, rotation=20, ha="right", fontsize=7)
        a.set_title(c, fontsize=9)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- 3. structure
def plot_correlation(changes: pd.DataFrame, summary: Dict):
    C = changes.corr()
    off = C.values[~np.eye(len(C), dtype=bool)]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    im = ax[0].imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].grid(False)
    ax[0].set_title(f"Correlation of daily changes ({len(C)} nodes)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].hist(off, bins=60, color=BLUE, alpha=0.85)
    ax[1].axvline(off.mean(), color=RED, lw=1.5, label=f"mean {off.mean():.3f}")
    ax[1].legend(); ax[1].set_xlabel("pairwise correlation")
    ax[1].set_title(f"Effective dimensionality {summary['eff_dim']:.2f} of {summary['n_points']}")
    fig.tight_layout(); return fig


def plot_neighbour_correlation(curves: Dict[str, pd.Series]):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, s in curves.items():
        ax.plot(s.index, s.values, "o-", label=name)
    ax.set_xlabel("expiry separation (pillars)"); ax.set_ylabel("mean corr of daily changes")
    ax.set_ylim(0, 1.02); ax.legend()
    ax.set_title("Flat-vol signature: slow decay near 1.0 = cumulative (flat cap) quotes")
    fig.tight_layout(); return fig


def plot_transform_comparison(tc: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    idx = list(tc.index)
    for a, c, ttl in [(ax[0], "mean_kurtosis", "Mean marginal excess kurtosis"),
                      (ax[1], "chi2_slope", "Joint chi² QQ slope (1.0 = MV normal)"),
                      (ax[2], "max_abs_z", "Largest |z| observed")]:
        a.barh(range(len(idx)), tc[c].values, color=BLUE)
        a.set_yticks(range(len(idx))); a.set_yticklabels(idx, fontsize=7)
        a.set_title(ttl, fontsize=9)
    ax[1].axvline(1.0, color=RED, ls="--", lw=1.2)
    fig.tight_layout(); return fig


def plot_normality(changes: pd.DataFrame, nodes: Sequence[Tuple[str, str]]):
    from scipy import stats
    nodes = [n for n in nodes if n in changes.columns][:4]
    fig, ax = plt.subplots(2, len(nodes), figsize=(3.6 * len(nodes), 6.4), squeeze=False)
    for i, nd in enumerate(nodes):
        v = changes[nd].dropna().values
        v = (v - v.mean()) / v.std()
        ax[0, i].hist(v, bins=60, density=True, color=BLUE, alpha=0.8)
        xs = np.linspace(-5, 5, 200)
        ax[0, i].plot(xs, stats.norm.pdf(xs), color=RED, lw=1.2)
        ax[0, i].set_title(f"{nd[0]}x{nd[1]}  kurt {stats.kurtosis(v):.1f}", fontsize=9)
        stats.probplot(v, dist="norm", plot=ax[1, i])
        ax[1, i].get_lines()[0].set_markersize(2.5)
        ax[1, i].set_title("")
    fig.suptitle("Marginal distributions vs normal (top) and QQ (bottom)")
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- 4. regimes
def plot_regime_stats(rs: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cols = [RED if k == "stress" else GREEN for k in rs["kind"]]
    for a, c, ttl in [(ax[0], "vol_ratio_vs_full", "Vol vs full sample (1.0 = typical)"),
                      (ax[1], "mean_corr", "Mean pairwise correlation"),
                      (ax[2], "eff_dim", "Effective dimensionality")]:
        a.bar(range(len(rs)), rs[c].values.astype(float), color=cols)
        a.set_xticks(range(len(rs))); a.set_xticklabels(rs.index, rotation=30, ha="right", fontsize=7)
        a.set_title(ttl, fontsize=9)
    ax[0].axhline(1.0, color="k", ls=":", lw=1)
    fig.tight_layout(); return fig


def plot_regime_overlap(M: pd.DataFrame):
    floor = M.attrs.get("noise_floor")
    fig, ax = plt.subplots(figsize=(1.05 * len(M) + 3.4, 0.95 * len(M) + 2.8))
    im = ax.imshow(M.values, cmap="viridis", vmin=max(0.0, M.values.min() - 0.02), vmax=1.0)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(M.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=8)
    ax.grid(False)
    for i in range(len(M)):
        for j in range(len(M)):
            below = floor is not None and i != j and M.values[i, j] < floor.min()
            ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="w" if M.values[i, j] < 0.75 else "k",
                    fontweight="bold" if below else "normal")
    ttl = "Pairwise regime subspace overlap"
    if floor is not None:
        ttl += f"\nsplit-half noise floor {floor.min():.3f}–{floor.max():.3f} (bold = genuinely below)"
    ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- 5. PCA core
def plot_scree(model, k: int = 15):
    T, p = len(model.fit_index), len(model.grid)
    hi = vs.marchenko_pastur_bounds(T, p)[1]
    lam = model.eigenvalues.iloc[:k]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4))
    ax[0].bar(range(1, len(lam) + 1), lam.values, color=BLUE)
    ax[0].axhline(hi, color=RED, ls="--", lw=1.2,
                  label=f"MP edge {hi:.2f} ({model.n_signal} PCs above)")
    ax[0].set_xlabel("component"); ax[0].set_ylabel("eigenvalue"); ax[0].legend()
    ax[0].set_title(f"Scree vs noise floor (T={T}, p={p})")
    cum = 100 * model.explained.iloc[:k].cumsum()
    ax[1].plot(range(1, len(cum) + 1), cum.values, "o-", color=BLUE)
    ax[1].axhline(90, color="grey", ls=":", lw=1); ax[1].set_ylim(0, 101)
    ax[1].set_xlabel("components"); ax[1].set_ylabel("cumulative %")
    ax[1].set_title("Cumulative SURFACE variance (not the metric that matters)")
    fig.tight_layout(); return fig


def plot_loadings(model, k: int = 4):
    fig, ax = plt.subplots(1, k, figsize=(3.9 * k, 3.8), squeeze=False)
    vmax = float(np.abs(model.loadings.iloc[:, :k].values).max())
    for i in range(k):
        pc = model.loadings.columns[i]
        M = _grid_2d(model.loadings[pc])
        im = ax[0, i].imshow(M.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax[0, i].set_xticks(range(len(M.columns)))
        ax[0, i].set_xticklabels(M.columns, rotation=90, fontsize=6)
        ax[0, i].set_yticks(range(len(M.index)))
        ax[0, i].set_yticklabels(M.index, fontsize=6)
        ax[0, i].set_title(f"{pc} — {100*model.explained[pc]:.1f}% surface var", fontsize=9)
        ax[0, i].grid(False); ax[0, i].set_xlabel("tenor")
        fig.colorbar(im, ax=ax[0, i], fraction=0.046)
    ax[0, 0].set_ylabel("expiry")
    fig.tight_layout(); return fig


def plot_eigen_decay(decay: pd.DataFrame, k: int = 12):
    pcs = [c for c in decay.columns if c.startswith("PC")][:k]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cmap = plt.get_cmap("viridis")
    for i, (W, row) in enumerate(decay.iterrows()):
        col = cmap(i / max(len(decay) - 1, 1))
        y = row[pcs].values.astype(float)
        ax[0].plot(range(1, len(pcs) + 1), y, "o-", color=col, ms=4, label=f"W={int(W)}")
        ax[1].semilogy(range(1, len(pcs) + 1), np.clip(y, 1e-4, None), "o-", color=col, ms=4)
        ax[2].plot(range(1, len(pcs) + 1), np.cumsum(y), "o-", color=col, ms=4)
    ax[0].set_ylabel("% surface variance"); ax[0].set_title("Eigenvalue decay"); ax[0].legend(fontsize=7)
    ax[1].set_title("Log scale — a straight line means geometric decay")
    ax[2].axhline(90, color="grey", ls=":", lw=1); ax[2].set_ylim(0, 101); ax[2].set_title("Cumulative")
    for a in ax:
        a.set_xlabel("component")
    fig.tight_layout(); return fig


def plot_score_diagnostics(scores: pd.DataFrame, diag: pd.DataFrame, k: int = 4):
    from scipy import stats
    cols = [c for c in scores.columns[:k] if c in diag.index]
    fig, ax = plt.subplots(2, len(cols), figsize=(3.6 * len(cols), 6.4), squeeze=False)
    for i, c in enumerate(cols):
        v = scores[c].dropna().values
        v = (v - v.mean()) / v.std()
        stats.probplot(v, dist="norm", plot=ax[0, i])
        ax[0, i].get_lines()[0].set_markersize(2.5)
        ax[0, i].set_title(f"{c} QQ (exc.kurt {diag.loc[c,'kurtosis']:.2f})", fontsize=9)
        lags = list(range(1, 11))
        acs = [np.corrcoef(v[:-l], v[l:])[0, 1] for l in lags]
        band = diag.loc[c, "ac_band_95"]
        ax[1, i].bar(lags, acs, color=BLUE)
        ax[1, i].axhline(band, color=RED, ls="--", lw=1)
        ax[1, i].axhline(-band, color=RED, ls="--", lw=1)
        ax[1, i].set_ylim(-0.35, 0.35); ax[1, i].set_xlabel("lag")
        ax[1, i].set_title(f"{c} autocorrelation", fontsize=9)
    fig.suptitle("Score diagnostics — normality (top), independence (bottom)")
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- 6. stability
def plot_identification(stab_pc: pd.DataFrame):
    sw = stab_pc.pivot(index="pc", columns="window", values="swap_rate")
    ov = stab_pc.pivot(index="pc", columns="window", values="mean_overlap")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(sw)); w = 0.8 / max(len(sw.columns), 1)
    for i, c in enumerate(sw.columns):
        ax[0].bar(x + i * w - 0.4, sw[c], w, label=f"W={c}")
        ax[1].bar(x + i * w - 0.4, ov[c], w, label=f"W={c}")
    ax[0].axhline(0.10, color=RED, ls="--", lw=1.2, label="10% tolerance")
    ax[0].set_xticks(x); ax[0].set_xticklabels(sw.index); ax[0].set_ylabel("swap rate")
    ax[0].set_title("How often does this PC change places?")
    ax[1].axhline(0.90, color=RED, ls="--", lw=1.2, label="0.90 tolerance")
    ax[1].set_xticks(x); ax[1].set_xticklabels(ov.index); ax[1].set_ylim(0, 1.02)
    ax[1].set_ylabel("mean |overlap| with prior refit")
    ax[1].set_title("Does this PC point the same way after a refit?")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout(); return fig


def plot_rolling_share(share: pd.DataFrame, regimes: Optional[Sequence] = None):
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(share.index, share["PC1"], color=RED, lw=1.5)
    ax[0].set_ylabel("PC1 share (%)")
    ax[0].set_title("PC1 concentration through time — rises in stress")
    ax[1].stackplot(share.index, *[share[c].values for c in share.columns],
                    labels=list(share.columns), alpha=0.85)
    ax[1].set_ylabel("cumulative share (%)"); ax[1].legend(loc="lower left", ncol=5, fontsize=7)
    ax[1].set_title("Full factor composition")
    if regimes:
        for a in ax:
            for r in regimes:
                a.axvspan(pd.Timestamp(r.start), pd.Timestamp(r.end), alpha=0.13,
                          color=RED if r.kind == "stress" else GREEN)
    fig.tight_layout(); return fig


def plot_loading_drift(snaps: Dict[str, "vs.VolPCA"], pcs: Sequence[str] = ("PC1", "PC2", "PC3")):
    fig, ax = plt.subplots(1, len(pcs), figsize=(5 * len(pcs), 4))
    cmap = plt.get_cmap("plasma")
    for j, pc in enumerate(pcs):
        for i, (d, m) in enumerate(snaps.items()):
            prof = m.loadings[pc].groupby(level="expiry").mean()
            prof = prof.reindex(sorted(prof.index, key=vs.maturity_to_days))
            ax[j].plot(range(len(prof)), prof.values, "o-", ms=3, lw=1.2,
                       color=cmap(i / max(len(snaps) - 1, 1)), label=d)
        ax[j].set_xticks(range(len(prof))); ax[j].set_xticklabels(prof.index, rotation=90, fontsize=6)
        ax[j].axhline(0, color="k", lw=0.8)
        ax[j].set_title(f"{pc} loading by expiry, across refits", fontsize=10)
    ax[0].legend(fontsize=6); ax[0].set_ylabel("loading (tenor-averaged)")
    fig.tight_layout(); return fig


def plot_subspace_matrix(models: Dict[str, "vs.VolPCA"], k: int = 6, title: str = ""):
    names = list(models)
    M = np.ones((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                M[i, j] = M[j, i] = vs.subspace_overlap(models[a].loadings, models[b].loadings, k=k)
    fig, ax = plt.subplots(figsize=(1.05 * len(names) + 3.2, 0.95 * len(names) + 2.6))
    im = ax.imshow(M, cmap="viridis", vmin=max(0.0, M.min() - 0.02), vmax=1.0)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="w" if M[i, j] < 0.75 else "k")
    ax.set_title(title or f"Pairwise subspace overlap (k={k})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- 7. book risk
def plot_risk_ranking(risk: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(risk))
    ax[0].bar(x - 0.2, risk["surface_var_%"], 0.4, label="surface var %", color=GREY)
    ax[0].bar(x + 0.2, risk["pnl_var_%"], 0.4, label="book PnL var %", color=RED)
    ax[0].set_xticks(x); ax[0].set_xticklabels(risk.index); ax[0].legend(); ax[0].set_ylabel("%")
    ax[0].set_title("What the surface does vs what the book feels")
    ax[1].bar(x, risk["exposure_$"], color=[BLUE if v > 0 else RED for v in risk["exposure_$"]])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(risk.index)
    ax[1].set_ylabel("$ PnL per +1 sd move")
    ax[1].set_title("Book exposure per factor (this is what you hedge)")
    fig.tight_layout(); return fig


def plot_book_alignment(align: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4))
    x = np.arange(len(align))
    ax[0].bar(x, align["cos2"], color=BLUE)
    ax[0].set_xticks(x); ax[0].set_xticklabels(align.index)
    ax[0].set_ylabel("cos² of u on PC")
    ax[0].set_title("Fraction of the book DIRECTION in each factor")
    ax[1].plot(x + 1, align["pnl_r2_implied"], "o-", color=RED, label="implied PnL R²")
    ax[1].plot(x + 1, align["cos2_cum"], "s--", color=GREY, label="cumulative cos²")
    ax[1].set_xlabel("factors retained"); ax[1].set_ylim(0, 1.02); ax[1].legend()
    ax[1].set_title("How much PnL a k-factor model can explain")
    fig.tight_layout(); return fig


def plot_horizon(h: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in [c for c in h.columns if c.startswith("pnl_r2")]:
        ax.plot(h.index, h[c], "o-", label=c)
    ax.set_xlabel("holding horizon (business days)"); ax.set_ylabel("implied PnL R²")
    ax.legend(); ax.set_title("Does the model explain more over longer horizons?")
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- 8. hedging
def plot_hedge_report(sel: Dict, sol: Dict):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    h = sol["notionals"]
    lbl = [f"{a}x{b}" for a, b in h.index]
    ax[0].barh(range(len(h)), h.values, color=BLUE)
    ax[0].set_yticks(range(len(h))); ax[0].set_yticklabels(lbl, fontsize=8)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_title(f"Hedge notionals (gross {sol['gross_ratio']:.2f}x book)")
    b, r = sol["book_exposure"], sol["residual_exposure"]
    x = np.arange(len(b))
    ax[1].bar(x - 0.2, b.values, 0.4, label="before", color=GREY)
    ax[1].bar(x + 0.2, r.values, 0.4, label="after", color=RED)
    ax[1].set_xticks(x); ax[1].set_xticklabels(b.index); ax[1].legend()
    ax[1].set_title("Factor exposure ($ per sd)")
    sv = sel["singular_values"]
    ax[2].bar(range(1, len(sv) + 1), sv, color="#334155")
    ax[2].set_xlabel("index"); ax[2].set_title(f"Singular values of A (κ = {sel['kappa']:,.1f})")
    fig.tight_layout(); return fig


def plot_hedge_churn(H: pd.DataFrame, kappas: Sequence[float]):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for c in H.columns:
        ax[0].plot(H.index, H[c], lw=1.1, label=f"{c[0]}x{c[1]}")
    ax[0].legend(fontsize=6); ax[0].set_ylabel("notional"); ax[0].set_title("Hedge notionals through time")
    turn = H.diff().abs().sum(axis=1) / H.abs().sum(axis=1).replace(0, np.nan)
    ax[1].plot(turn.index, turn.values, color=RED, lw=1.2)
    ax[1].set_ylabel("Σ|Δh| / Σ|h|"); ax[1].set_title(f"Rebalance turnover — median {turn.median():.1%}")
    ax[2].plot(H.index, kappas, color="#334155", lw=1.2)
    ax[2].set_ylabel("κ(A)"); ax[2].set_title("Hedge conditioning through time")
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- 9. backtest
def plot_backtest(bt: Dict, regimes: Optional[Sequence] = None):
    r2, df = bt["r2"], bt["daily"]
    fig = plt.figure(figsize=(13, 6.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.1])
    a0 = fig.add_subplot(gs[0, 0])
    a0.plot(r2.index, r2.values, "o-", color=RED)
    a0.axhline(0, color="k", lw=0.8); a0.set_xlabel("factors"); a0.set_ylabel("OOS PnL R²")
    a0.set_title("Walk-forward PnL explain")
    a1 = fig.add_subplot(gs[0, 1])
    a1.scatter(df["days_since_refit"], df["unexplained"].abs(), s=8, alpha=0.5, color=BLUE)
    a1.set_xlabel("business days since refit"); a1.set_ylabel("|unexplained PnL|")
    a1.set_title("Does error grow as the factors go stale?")
    a2 = fig.add_subplot(gs[1, :])
    a2.plot(df.index, df["unexplained"], lw=0.8, color="#334155")
    s = df["unexplained"].std()
    a2.axhline(3 * s, color=RED, ls="--", lw=1, label="±3σ"); a2.axhline(-3 * s, color=RED, ls="--", lw=1)
    if regimes:
        for r in regimes:
            a2.axvspan(pd.Timestamp(r.start), pd.Timestamp(r.end), alpha=0.12,
                       color=RED if r.kind == "stress" else GREEN)
    a2.legend(); a2.set_title("Unexplained daily PnL — look for clustering near refits and regime turns")
    fig.tight_layout(); return fig


def plot_window_robustness(stab: pd.DataFrame, cov: pd.DataFrame,
                           floor: Optional[Dict] = None, k: int = 6):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(stab["window"], stab["subspace_overlap"], "o-", color=BLUE, label="mean")
    ax[0].fill_between(stab["window"], stab["overlap_min"], stab["subspace_overlap"],
                       alpha=0.18, color=BLUE, label="min-to-mean")
    if floor:
        ax[0].axhline(floor["mean"], color=RED, ls="--", lw=1.2,
                      label=f"split-half floor {floor['mean']:.3f}")
        ax[0].axhspan(floor["p05"], floor["p95"], color=RED, alpha=0.10)
    ax[0].set_xlabel("window (days)"); ax[0].set_ylabel(f"subspace overlap (k={k})")
    ax[0].set_title("Stability: do consecutive refits span the same space?", fontsize=9)
    ax[0].legend(fontsize=7)
    for est, g in cov.groupby("estimator"):
        g = g.sort_values("window")
        st = "s--" if est == "sample" else "o-"
        ax[1].plot(g["window"], g["mvp_oos_vol"], st, label=est, lw=1.4, ms=4)
        ax[2].semilogy(g["window"], g["vol_underestimate_x"], st, label=est, lw=1.4, ms=4)
    ax[1].set_xlabel("window (days)"); ax[1].set_ylabel("realized OOS min-var vol")
    ax[1].set_title("Forecast quality: lower = better future covariance", fontsize=9)
    ax[2].axhline(1.0, color="k", lw=1, ls=":")
    ax[2].set_xlabel("window (days)"); ax[2].set_ylabel("realized / predicted vol (log)")
    ax[2].set_title("Calibration: >1 means risk was under-promised", fontsize=9)
    for a in ax[1:]:
        a.legend(fontsize=7)
    fig.tight_layout(); return fig


def plot_covariance_forecast(cov: pd.DataFrame):
    piv = {m: cov.pivot(index="window", columns="estimator", values=m)
           for m in ["mvp_oos_vol", "calib_slope", "vol_underestimate_x", "frob_rel_err"]}
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    spec = [("mvp_oos_vol", ax[0, 0], "Forecast quality — LOWER IS BETTER", False),
            ("calib_slope", ax[0, 1], "Calibration (1.0 = perfect)", False),
            ("vol_underestimate_x", ax[1, 0], "Realized / predicted vol — >1 is dangerous", True),
            ("frob_rel_err", ax[1, 1], "Relative Frobenius error (ranking only)", False)]
    for m, a, ttl, logy in spec:
        for c in piv[m].columns:
            st = "s--" if c == "sample" else "o-"
            lw = 2.0 if c == "sample" else 1.3
            (a.semilogy if logy else a.plot)(piv[m].index, piv[m][c], st, label=c, lw=lw, ms=5)
        a.set_title(ttl, fontsize=10); a.set_xlabel("window (days)"); a.legend(fontsize=7)
    ax[0, 1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1, 0].axhline(1.0, color="k", ls=":", lw=1)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- 10. weighted
def plot_weight_diagnostics(wdf: pd.DataFrame, benchmark=("1Y", "10Y")):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col, ttl, cmap, div in [
            (ax[0], "beta", f"beta to {benchmark[0]}x{benchmark[1]}", "RdBu_r", True),
            (ax[1], "tw_vega", "time-weighted vega", "RdBu_r", True),
            (ax[2], "weight", "blended PCA weight", "viridis", False)]:
        im = _heat(a, _grid_2d(wdf[col]), ttl, cmap=cmap, diverging=div)
        fig.colorbar(im, ax=a, fraction=0.046)
    ax[0].set_ylabel("expiry")
    fig.tight_layout(); return fig


def plot_lambda_sweep(sweep: pd.DataFrame, baseline: Optional[Dict] = None):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4))
    for c in [c for c in sweep.columns if c.startswith("PnL_R2")]:
        ax[0].plot(sweep.index, sweep[c], "o-", ms=4, label=c.replace("PnL_R2_", ""))
    if baseline:
        for c, v in baseline.items():
            ax[0].axhline(v, ls="--", lw=1, alpha=0.7, color=GREY)
        ax[0].text(sweep.index[0], list(baseline.values())[0], " unweighted", fontsize=7, va="bottom")
    ax[0].set_xlabel("lambda  (0 = pure time-weighted vega, 1 = pure beta)")
    ax[0].set_ylabel("implied book PnL R²"); ax[0].legend()
    ax[0].set_title("Does weighting buy PnL explanatory power?")
    ax[1].plot(sweep.index, sweep["PC1_surface_%"], "o-", color=RED, label="PC1 %")
    ax[1].plot(sweep.index, sweep["top6_surface_%"], "s-", color=BLUE, label="top-6 %")
    ax[1].set_xlabel("lambda"); ax[1].set_ylabel("% surface variance"); ax[1].legend()
    ax[1].set_title("What the weighting costs in variance capture")
    fig.tight_layout(); return fig


def plot_model_comparison(cmp: pd.DataFrame, ks: Sequence[int] = (1, 3, 6)):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(cmp)); w = 0.8 / len(ks)
    for i, k in enumerate(ks):
        ax[0].bar(x + i * w - 0.4, cmp[f"PnL_R2_k{k}"], w, label=f"k={k}")
    ax[0].set_xticks(x); ax[0].set_xticklabels(cmp.index, rotation=20, ha="right", fontsize=7)
    ax[0].set_ylabel("implied book PnL R²"); ax[0].legend()
    ax[0].set_title("PnL explained — the point of weighting")
    ax[1].bar(x, cmp["PC1_surface_%"], color=GREY)
    ax[1].set_xticks(x); ax[1].set_xticklabels(cmp.index, rotation=20, ha="right", fontsize=7)
    ax[1].set_ylabel("% surface variance in PC1")
    ax[1].set_title("PC1 concentration — the cost")
    fig.tight_layout(); return fig
