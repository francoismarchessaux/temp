import numpy as np
import pandas as pd

CAL_ROWS = ("2D", "1W", "2W", "1M", "3M")


def _as_series(x, grid, what):
    """Series(MultiIndex) | 2D DataFrame(expiry x tenor) | 1-row panel -> Series on grid."""
    obj = getattr(x, "data", x)
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.columns, pd.MultiIndex):
            if len(obj) != 1:
                raise ValueError(f"{what}: panel has {len(obj)} rows, pass one date")
            obj = obj.iloc[0]
        else:
            obj = obj.stack()
    if not isinstance(obj, pd.Series):
        raise TypeError(f"{what}: cannot read a surface out of {type(x)}")
    obj.index.names = ["expiry", "tenor"]
    out = obj.reindex(grid).astype(float)
    if out.isna().any():
        miss = [f"{e}x{t}" for e, t in out.index[out.isna()]][:6]
        raise ValueError(f"{what}: {int(out.isna().sum())} grid nodes missing, e.g. {miss}")
    return out


def _today_calendar_adj(hist_changes, grid, weekday, rows=CAL_ROWS, window=250):
    """The same trailing per-weekday mean clean_calendar would subtract, for a new day."""
    adj = pd.Series(0.0, index=grid)
    if hist_changes is None:
        return adj
    n = max(4, window // 5)
    for c in [c for c in grid if c[0] in set(rows)]:
        if c not in hist_changes.columns:
            continue
        sub = hist_changes[c]
        sub = sub[sub.index.dayofweek == weekday]
        if len(sub) >= 4:
            adj[c] = float(sub.iloc[-n:].mean())
    return adj


def pnl_snapshot(model, vol_now, vol_prev, vega, k=None, real_pnl=None,
                 hist_changes=None, n_nodes=12, label=None, verbose=True):
    """Real vs PCA vega PnL for one day, with the error split into its two parts.

    E1 reconciliation = real_pnl - vega'dv        (vega/marks/off-grid/smile/IRG)
    E2 truncation     = u'z - sum_{j<=k} c_j      (the cost of using k factors)

    Only E2 is a PCA problem. If E1 dominates, no amount of factor work helps.
    Pass hist_changes (the calendar-cleaned history the model was fitted on) so
    today's 1M/3M rows get the same roll adjustment the fit assumed.
    """
    k = k or model.config.n_components
    grid = model.grid
    v_now = _as_series(vol_now, grid, "vol_now")
    v_prev = _as_series(vol_prev, grid, "vol_prev")
    vega = _as_series(vega, grid, "vega")

    dv_raw = v_now - v_prev
    wd = getattr(vol_now, "name", None)
    wd = pd.Timestamp(wd).dayofweek if wd is not None else pd.Timestamp.today().dayofweek
    adj = _today_calendar_adj(hist_changes, grid, wd,
                              rows=model.config.calendar_rows,
                              window=model.config.calendar_window)
    dv = dv_raw - adj

    sc = model.scale
    z = (dv - sc.mu) / sc.sigma
    u = sc.book_direction(vega)
    L, lam = model.loadings, model.eigenvalues

    s = pd.Series(L.values.T @ z.values, index=L.columns)
    expo = pd.Series(L.values.T @ u.values, index=L.columns)
    contrib = expo * s

    pnl_raw = float(vega @ dv_raw)
    pnl_clean = float(vega @ dv)
    drift = float(vega @ sc.mu)
    pnl_centered = float(u @ z)
    pnl_k = float(contrib.iloc[:k].sum())
    e2 = pnl_centered - pnl_k
    e1 = None if real_pnl is None else float(real_pnl) - pnl_raw

    zk = pd.Series(L.iloc[:, :k].values @ s.iloc[:k].values, index=grid)
    node_resid = (u * (z - zk)).sort_values(key=np.abs, ascending=False)

    tbl = pd.DataFrame({
        "exposure_$": expo.iloc[:max(k, 8)],
        "score_sd": s.iloc[:max(k, 8)] / np.sqrt(lam.iloc[:max(k, 8)]),
        "contrib_$": contrib.iloc[:max(k, 8)],
    })
    tbl["cum_$"] = tbl["contrib_$"].cumsum()
    tbl["cum_%_of_u'z"] = 100 * tbl["cum_$"] / pnl_centered if pnl_centered else np.nan

    if verbose:
        def pct(x, base):
            return f"{100*x/base:+6.1f}%" if base else "     n/a"
        print(f"=== PnL snapshot {label or ''} ".ljust(64, "=") )
        print(f"  grid {len(grid)} nodes | k={k} | fit "
              f"{model.fit_index[0].date()} -> {model.fit_index[-1].date()}")
        print(f"  gross vega {float(vega.abs().sum()):,.0f} | net {float(vega.sum()):,.0f} "
              f"| calendar adj on {int((adj != 0).sum())} nodes")
        print()
        if real_pnl is not None:
            print(f"  real vega PnL (sheet)      {float(real_pnl):>15,.0f}")
        print(f"  vega' dv   (full rank, raw){pnl_raw:>15,.0f}")
        if real_pnl is not None:
            print(f"  E1 reconciliation          {e1:>15,.0f}   {pct(e1, real_pnl)} of real")
        print(f"  vega' dv   (calendar-clean){pnl_clean:>15,.0f}")
        print(f"  less drift vega'mu         {-drift:>15,.0f}")
        print(f"  u'z        (all {len(L.columns)} factors){pnl_centered:>13,.0f}")
        print(f"  PCA k={k}                    {pnl_k:>15,.0f}")
        print(f"  E2 truncation              {e2:>15,.0f}   {pct(e2, pnl_centered)} of u'z")
        if real_pnl is not None:
            tot = float(real_pnl) - pnl_k
            print(f"  TOTAL error real - PCA     {tot:>15,.0f}   {pct(tot, real_pnl)} of real")
        print("\n  per-factor contribution")
        print(tbl.round(1).to_string())
        print(f"\n  top {n_nodes} nodes in the k={k} residual (bp of vol -> $)")
        print(pd.DataFrame({
            "resid_$": node_resid.iloc[:n_nodes].round(0),
            "dv_bp": dv.reindex(node_resid.index[:n_nodes]).round(2),
            "vega": vega.reindex(node_resid.index[:n_nodes]).round(0),
        }).to_string())

    return {"real": real_pnl, "pnl_raw": pnl_raw, "pnl_clean": pnl_clean,
            "drift": drift, "pnl_centered": pnl_centered, "pnl_k": pnl_k,
            "E1_reconciliation": e1, "E2_truncation": e2,
            "factors": tbl, "node_residual": node_resid, "dv": dv, "z": z}


# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    exp = ["1M", "3M", "6M", "1Y", "18M", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "15Y", "20Y"]
    ten = ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
    grid = pd.MultiIndex.from_product([exp, ten], names=["expiry", "tenor"])
    p = len(grid)

    class _S:
        mu = pd.Series(rng.normal(0, .02, p), index=grid)
        sigma = pd.Series(rng.uniform(.4, 5.4, p), index=grid)
        def book_direction(self, v): return self.sigma * v.reindex(self.sigma.index)

    class _Cfg:
        n_components = 6; calendar_rows = ("1M", "3M"); calendar_window = 250

    class _M:
        config = _Cfg(); scale = _S(); grid = grid
        fit_index = pd.bdate_range("2024-01-01", periods=500)
        Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        lam = np.sort(rng.uniform(.01, 1, p))[::-1]; lam = lam / lam.sum() * p
        loadings = pd.DataFrame(Q, index=grid, columns=[f"PC{i+1}" for i in range(p)])
        eigenvalues = pd.Series(lam, index=loadings.columns)

    m = _M()
    v_prev = pd.Series(rng.uniform(60, 120, p), index=grid)
    v_now = v_prev + pd.Series(rng.normal(0, 3, p), index=grid)
    v_now.name = pd.Timestamp("2026-08-31")
    vega = pd.Series(rng.normal(0, 40_000, p), index=grid)

    r = pnl_snapshot(m, v_now, v_prev, vega, k=6, real_pnl=None, verbose=False)

    # the identity that must close: u'z == sum of ALL factor contributions
    allc = float((pd.Series(m.loadings.values.T @ ((v_now - v_prev - m.scale.mu) / m.scale.sigma).values,
                            index=m.loadings.columns)
                  * pd.Series(m.loadings.values.T @ (m.scale.sigma * vega).values,
                              index=m.loadings.columns)).sum())
    print(f"u'z                    = {r['pnl_centered']:,.2f}")
    print(f"sum of all p contribs  = {allc:,.2f}")
    print(f"identity closes        : {abs(allc - r['pnl_centered']) < 1e-6 * abs(allc)}")
    print()
    print(f"vega'dv                = {r['pnl_raw']:,.2f}")
    print(f"u'z + vega'mu          = {r['pnl_centered'] + r['drift']:,.2f}")
    print(f"drift identity closes  : {abs(r['pnl_raw'] - r['pnl_centered'] - r['drift']) < 1e-6 * abs(r['pnl_raw'])}")
    print()
    r = pnl_snapshot(m, v_now, v_prev, vega, k=6, real_pnl=r["pnl_raw"] * 1.18,
                     label="2026-08-31 vs 2026-08-28", n_nodes=4)
