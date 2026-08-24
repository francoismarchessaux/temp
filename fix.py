# ============================================================
# BLOCK A — k = 4 vs 5, and where the PnL explain actually sits
# ============================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import volsurface as vs, volsurface_plots as vp

K = 4                                    # <-- 4 or 5
CFG_K = CFG.copy(n_components=K, n_report=8)

bt = vs.pnl_explain_backtest(clean_ch, vega, CFG_K, k_list=(1,2,3,4,5,6,8))
daily = bt["daily"]

print("\n--- k decision ---")
mp_edge = vs.marchenko_pastur_bounds(CFG.window, clean_ch.shape[1])[1]
summ = model.summary(k=8)
print(summ.to_string())
print(f"\nMP edge {mp_edge:.2f} -> {int(summ['above_MP'].sum())} factors clear the noise floor")
print(f"walk-forward PnL R2 at k={K}: {bt['r2'][K]:.3f}")

# ============================================================
# BLOCK B — horizon sweep, done correctly
# Loadings are ALWAYS fitted on daily data (a 21-day fit would have
# 500/21 = 24 observations vs p=130 and be singular). Only the SCORING
# horizon changes: aggregate the frozen daily predictions and the actual
# PnL over non-overlapping h-day blocks.
# ============================================================
HORIZONS = {"daily":1, "weekly":5, "2 weeks":10, "3 weeks":15, "monthly":21}
rows = []
for name, h in HORIZONS.items():
    g = np.arange(len(daily)) // h
    agg = daily.groupby(g).agg({"actual":"sum", **{f"pred_{k}":"sum" for k in (1,2,3,4,5,6,8)}})
    agg = agg.iloc[:-1] if len(agg) > 1 else agg          # drop the ragged last block
    ss_tot = float(((agg["actual"] - agg["actual"].mean())**2).sum())
    r = {"horizon": name, "days": h, "n_blocks": len(agg),
         "pnl_sd": float(agg["actual"].std(ddof=1))}
    for k in (3, 4, 5, 6):
        ss_res = float(((agg["actual"] - agg[f"pred_{k}"])**2).sum())
        r[f"R2_k{k}"] = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rows.append(r)
hz = pd.DataFrame(rows).set_index("horizon")
print("\n--- PnL explain by holding horizon (loadings fitted daily) ---")
print(hz.round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for k in (3, 4, 5, 6):
    ax[0].plot(hz["days"], hz[f"R2_k{k}"], "o-", label=f"k={k}")
ax[0].set_xlabel("holding horizon (business days)"); ax[0].set_ylabel("out-of-sample PnL R²")
ax[0].set_title("Does the model explain more over longer horizons?"); ax[0].legend()
ax[1].plot(hz["days"], hz["pnl_sd"], "o-", color=vp.RED)
ax[1].plot(hz["days"], hz["pnl_sd"].iloc[0]*np.sqrt(hz["days"]), "k--", lw=1, label="√t scaling")
ax[1].set_xlabel("horizon (days)"); ax[1].set_ylabel("PnL sd")
ax[1].set_title("PnL sd vs √t — above the line = trending, not i.i.d."); ax[1].legend()
plt.tight_layout(); plt.show()

# ============================================================
# BLOCK C — hedge on the trader's actual instruments
# ============================================================
TRADER_SET = [("1Y","1Y"), ("2Y","1Y"), ("1Y","10Y"), ("1Y","5Y")]
missing = [t for t in TRADER_SET if t not in set(model.grid)]
print(f"\n--- hedge on the trader's set ---\nmissing from grid: {missing or 'none'}")
TRADER_SET = [t for t in TRADER_SET if t in set(model.grid)]

cands_t = vs.unit_vega_instruments(model.grid, subset=TRADER_SET)
A = vs.hedge_exposure_matrix(model, cands_t, k=K)
sv = np.linalg.svd(A.values, compute_uv=False)
print(f"kappa = {sv[0]/max(sv[-1],1e-12):,.1f}   singular values: {np.round(sv,3)}")
if len(TRADER_SET) < K:
    print(f"WARNING only {len(TRADER_SET)} instruments for k={K} — the solve is "
          "under-determined and some factor exposure CANNOT be removed.")

sol_t = vs.solve_hedge(model, vega, cands_t, k=K)
print("\nnotionals ($ vega per bp):")
_n = sol_t["notionals"].copy(); _n.index = [f"{a}x{b}" for a, b in _n.index]
print(_n.round(1).to_string())
print("\nfactor exposure before -> after:")
print(pd.DataFrame({"before": sol_t["book_exposure"],
                    "after":  sol_t["residual_exposure"]}).round(3).to_string())

# compare against the unconstrained greedy pick
sel_g = vs.select_hedge_instruments(model, vs.unit_vega_instruments(model.grid), k=K, verbose=False)
sol_g = vs.solve_hedge(model, vega, vs.unit_vega_instruments(model.grid)[sel_g["instruments"]],
                       k=K, verbose=False)
print(f"\n{'':22s}{'trader set':>14s}{'greedy pick':>14s}")
for lbl, key in [("kappa","kappa"), ("factor var removed %","factor_var_removed_%"),
                 ("unhedged beyond PC k %","unhedged_var_%"), ("gross / book vega","gross_ratio")]:
    a = sv[0]/max(sv[-1],1e-12) if key=="kappa" else sol_t[key]
    b = sel_g["kappa"] if key=="kappa" else sol_g[key]
    print(f"{lbl:22s}{a:14.2f}{b:14.2f}")

# ============================================================
# BLOCK D — PnL explain diagnostics (the core deliverable)
# ============================================================
daily["pred"] = daily[f"pred_{K}"]
daily["err"]  = daily["actual"] - daily["pred"]
print(f"\n--- PnL explain, k={K} ---")
print(f"actual sd {daily['actual'].std():,.1f} | unexplained sd {daily['err'].std():,.1f} "
      f"| ratio {daily['err'].std()/daily['actual'].std():.3f}")
print(f"mean |error| {daily['err'].abs().mean():,.1f} | worst {daily['err'].abs().max():,.1f}")
print("\n10 worst days:")
print(daily.reindex(daily["err"].abs().nlargest(10).index)[["actual","pred","err"]].round(1).to_string())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(daily["pred"], daily["actual"], s=6, alpha=0.4, color=vp.BLUE)
lim = [daily[["pred","actual"]].min().min(), daily[["pred","actual"]].max().max()]
ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("PCA-predicted PnL"); ax[0].set_ylabel("actual PnL")
ax[0].set_title(f"k={K}: predicted vs actual (R²={bt['r2'][K]:.3f})")
ax[1].hist(daily["err"], bins=80, color=vp.RED, alpha=0.85)
ax[1].set_xlabel("unexplained PnL"); ax[1].set_title("Residual distribution")
ax[2].plot(daily.index, daily["err"].abs().rolling(63).mean(), color="#334155")
ax[2].set_ylabel("63d mean |error|"); ax[2].set_title("Is the error stable through time?")
plt.tight_layout(); plt.show()

# which grid points drive the residual?
resid_bp = model.residual(clean_ch, k=K)
contrib = (resid_bp * vega).abs().mean().sort_values(ascending=False)
print("\ntop 10 nodes by mean |unexplained PnL contribution|:")
_c = contrib.head(10).copy(); _c.index = [f"{a}x{b}" for a, b in _c.index]
print(_c.round(2).to_string())
