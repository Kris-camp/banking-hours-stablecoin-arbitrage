"""Robustness of the Turkey--Mexico contrast (Table emp_mxn) to timing, rollover, thin trading
and the side of the band.

DESIGN
======
Construction: identical to code/empirics_mxn.py (Binance USDT/local minute VWAP against the
Dukascopy USD/local mid, five-minute means, weekdays, |b| <= 1000 bp, bands profiled by month x
local banking state, month-by-state effects, beta from y ~ q + up + dn, day-clustered SEs with the
finite-sample correction).  The speed of arbitrage is the coefficient on d_out in
R = -s (dy - beta q) on d_out, d_in (pooled sides), or on up / dn in y ~ q + up + dn (by side).
Day clusters are UTC calendar days; when the two markets are stacked in one regression the same
UTC day is one cluster across both markets, which gives the cross-market covariance.  Every
cross-market difference is reported with this joint ("stacked") SE and with the
independent-samples SE that Table emp_mxn uses.

(a) Common UTC windows.  W1 = UTC 06:00-14:00 (Turkey 09:00-17:00 local, banks open;
    Mexico 00:00-08:00 local, banks closed; London session).  W2 = UTC 15:00-20:30 (Turkey
    18:00-23:30 local, banks closed; Mexico 09:00-14:30 local, banks open; New York session;
    ends where the rollover exclusion begins).  Stacked regression of R on d_out and d_in
    interacted with market x window, market x month x window fixed effects, sample restricted
    to W1 and W2.  Bands: the baseline month x local-state bands (W1 and W2 map one-to-one into
    banking states in both markets).  Headline statistic:
        DiD = log(k_TR,W1 / k_TR,W2) - log(k_MX,W1 / k_MX,W2)
    by the delta method with the stacked covariance.  The global-session alternative (speed set
    by the UTC session, not by banks) implies DiD = 0; banks mattering more in Turkey implies
    DiD > 0.  With log speeds additive in market, session and own-bank effects, the DiD equals
    delta_TR + delta_MX (the two own-bank effects), so it cannot by itself separate a Turkey-
    specific rail effect from a generic local business-hours effect present in both markets;
    the within-market ratios are reported for that reason.  Expected pattern if banks matter:
    Turkey fast in W1 and slow in W2 (log ratio > 0); Mexico not slow in W1 (log ratio near 0).
    Level speeds and level differences are reported too, but levels are not comparable across
    markets because the price impact of arbitrage flow (lambda) differs.
    Variants of (a): (a2) bands re-profiled by month x UTC window;
    (a3) the thin-trading filters of (c); (a4) the demand-only OFI control of (e); (a5) by side.
    Exploratory (a6): a third window W0 = UTC 23:00-06:00 [see r1] in which both markets' banks are
    closed identifies the two own-bank effects separately,
        delta_TR = log(k_TR,W1/k_TR,W0) - log(k_MX,W1/k_MX,W0),
        delta_MX = log(k_MX,W2/k_MX,W0) - log(k_TR,W2/k_TR,W0);
    reported only if all six speeds are positive, and flagged as unreliable if any speed is
    within two s.e. of zero (the delta method fails there).

(b) Rollover.  The 20:30-22:30 UTC exclusion falls in Mexico's business hours (14:30-16:30 local)
    and in Turkey's closed hours (23:30-01:30 local).  (b1) no rollover exclusion in either
    market; (b2) aligned local-time exclusion: both markets drop 20:30-22:30 UTC AND the local-time
    image of the other market's loss, i.e. local 14:30-16:30 and 23:30-01:30 in both markets.
    Bands and beta re-estimated in each variant.  Statistic: the Table emp_mxn difference in log
    open/closed ratios, pooled sides.

(c) Thin USDT/MXN trading (MXN filtered, Turkey at baseline; the same filter applied to both
    markets is also reported).  Baseline bands and beta are kept; only the regression sample
    changes.  (c1) at least 10 trades in both interval t-1 and interval t; (c2) USDT volume of
    both t-1 and t above the median of its month x banking-state cell; (c3) errors-in-variables
    check, both markets: the side s and the distances d_out, d_in are computed from b_{t-2}
    instead of b_{t-1} (three consecutive intervals required), so iid measurement error in the
    five-minute premium cannot generate reversion.  Statistic: log open/closed ratio by market
    and their difference, local-time design and common-UTC-window design.

(d) Sides.  y ~ q + up + dn with business-hours interactions (as the 'above band only' row of
    Table emp_mxn), month-by-state effects: closed-hours speed, business-hours increment (level),
    open speed, open/closed ratio and log ratio, separately above and below the band, and pooled
    (R-based, as in the table).  Cross-market differences of increments (levels) and of log
    ratios.  Same in the common-UTC-window design (W1 vs W2).

(e) Demand-only OFI control (optional).  USDT/MXN aggTrades from data.binance.vision
    (code/rob_mexico_utc_aggtrades.py, same size classes as Turkey).  q_dem = small + medium
    (< 10,000 USDT) signed flow standardized within month; beta_dem from y ~ q_dem + up + dn;
    R_dem = -s(dy - beta_dem q_dem).  Both markets, local-time and UTC-window designs.

Classification stored in the JSON ("verdicts"): the contrast "survives" a variant if the cross-market statistic
(difference of log open/closed ratios in the local-time design; the DiD in the UTC-window design)
is positive and significant at 5% (two-sided) with the stacked SE.  A positive but insignificant
point estimate is reported as "direction preserved, not significant"; a sign reversal as
"does not survive".

NOTES AND ADDITIONAL VARIANTS
(r1) W0 starts at 23:00 UTC, not at the end of the rollover exclusion (22:30), because 22:30-23:00 UTC
     (16:30-17:00 Mexico City) is Mexican business hours; in W0 both markets' banks are closed.
(r2) A variant of (b) with no rollover exclusion for MXN only, Turkey at baseline
     ("b_rollover.none_mxn_only").
(r3) R-based by-side estimates (R on d_out above, d_out below, d_in, with interactions), the
     same dependent variable as the pooled row, because the by-side rows of (d) use the
     y ~ q + up + dn specification of Table emp_mxn and so are not directly comparable to the
     pooled R-based row ("d_sides_local_R", "a7_window_sides_R").

Inputs: data_raw/binance (USDTTRY, USDTMXN), data_raw/dukascopy (USDTRY, USDMXN), data_raw/ofi_size and
data_raw/ofi_size_mxn (aggtrades_ofi.py, rob_mexico_utc_aggtrades.py).

Outputs: tables/rob_mexico_utc.json, tables/rob_mexico_utc.tex, figures/rob_mexico_utc.pdf.
"""
import os, glob, json, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import _ssr_grid
from build_panel import load_binance, load_fx

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
W0_, W1_ = "2024-11-01", "2026-08-31"
UTC_OFF = {"TRY": 3, "MXN": -6}
MK = ["TRY", "MXN"]


def log(*s):
    print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)


# ------------------------------------------------------------------ construction (as empirics_mxn.py)
def profile(d, trim=0.05):
    y, x, q = d.y.values, d.x.values, d.q.values
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + 1.0, 1.0)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]


RAWP = {}


def raw_minutes(local):
    if local not in RAWP:
        u = load_binance(f"USDT{local}").loc[W0_:W1_]
        fx = load_fx(f"USD{local}")
        p = pd.DataFrame({"pl": u.vwap, "ofi": 2 * u.tbv - u.v, "vol": u.v, "ntr": u.n}).join(
            fx.reindex(u.index, method="ffill", limit=2))
        p["b0"] = 1e4 * np.log(p.pl / p.fx_mid)
        RAWP[local] = p
    return RAWP[local].copy()


def excl_mask(idx, local, excl):
    hr = idx.hour + idx.minute / 60
    if excl == "none":
        return np.zeros(len(idx), bool)
    m = (hr >= 20.5) & (hr < 22.5)
    if excl == "aligned":
        lh = (hr + UTC_OFF[local]) % 24
        m = m | ((lh >= 14.5) & (lh < 16.5)) | (lh >= 23.5) | (lh < 1.5)
    return np.asarray(m)


def build(local, excl="base"):
    p = raw_minutes(local)
    p["b"] = p.b0
    p.loc[excl_mask(p.index, local, excl), "b"] = np.nan
    p.loc[p.index.dayofweek >= 5, "b"] = np.nan
    p.loc[p.b.abs() > 1000, "b"] = np.nan
    g = p.resample("5min")
    a = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum(), "vol": g.vol.sum(), "ntr": g.ntr.sum(),
                      "pl": np.log(g.pl.mean()), "pg": np.log(g.fx_mid.mean())})
    a = a[a.nb >= 2]
    a["month"] = a.index.tz_localize(None).to_period("M").astype(str)
    a["q"] = a.ofi / a.ofi.groupby(a.month).transform("std")
    dt_ = a.index.to_series().diff()
    consec = dt_ == pd.Timedelta("5min")
    consec2 = consec & (dt_.shift(1) == pd.Timedelta("5min"))
    a["y"] = a.b.diff(); a["x"] = a.b.shift(1); a["x2"] = a.b.shift(2)
    a["ntr_l"] = a.ntr.shift(1); a["vol_l"] = a.vol.shift(1)
    a["c2ok"] = consec2.values
    a = a[consec & a.y.notna() & a.x.notna() & a.q.notna()].copy()
    lh = (a.index.hour + UTC_OFF[local]) % 24
    a["lh"] = lh
    a["uh"] = a.index.hour + a.index.minute / 60
    a["bh"] = ((lh >= 9) & (lh < 17)).astype(float)
    a["state"] = np.where(a.bh == 1, "open", "closed")
    a["ms"] = a.month + "_" + a.state
    for key, d in a.groupby("ms"):
        if len(d) < 300: continue
        l, u_ = profile(d)
        a.loc[d.index, "l"], a.loc[d.index, "u"] = l, u_
    a = a[a.l.notna()].copy()
    a["day"] = a.index.date
    a["mkt"] = local
    # UTC windows
    a["win"] = np.where((a.uh >= 6) & (a.uh < 14), "W1", np.where((a.uh >= 15) & (a.uh < 20.5), "W2",
                        np.where((a.uh >= 23) | (a.uh < 6), "W0", "")))
    geom(a)
    return a


def geom(a, xcol="x"):
    """side, distances and one-sided terms from the band and the lagged premium column xcol."""
    x = a[xcol]
    m = (a.l + a.u) / 2
    a["s"] = np.where(x > a.u, 1.0, np.where(x < a.l, -1.0, np.sign(x - m)))
    a["d_out"] = np.maximum(x - a.u, 0) + np.maximum(a.l - x, 0)
    a["d_in"] = np.where(a.d_out == 0, np.abs(x - m), 0.0)
    a["up"] = -np.maximum(x - a.u, 0); a["dn"] = np.maximum(a.l - x, 0)
    a["d_upo"] = np.maximum(x - a.u, 0); a["d_dno"] = np.maximum(a.l - x, 0)
    return a


# ------------------------------------------------------------------ estimation
def ols(d, ycol, cols, fe="ms"):
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    if fe:
        gm = d.groupby(fe)
        Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), n=n, V=V, cols=list(cols),
                bv=b, ndays=int(ng))


def dm(res, fun):
    """value and delta-method s.e. of fun(coef dict) using res['V'] (central differences)."""
    b0 = res["bv"].copy(); cols = res["cols"]
    f0 = fun(dict(zip(cols, b0)))
    g = np.zeros(len(b0))
    for i in range(len(b0)):
        h = 1e-6 * max(abs(b0[i]), 1e-4)
        bp, bm = b0.copy(), b0.copy(); bp[i] += h; bm[i] -= h
        g[i] = (fun(dict(zip(cols, bp))) - fun(dict(zip(cols, bm)))) / (2 * h)
    return float(f0), float(np.sqrt(g @ res["V"] @ g))


def stack(A, cols, fe, ycol):
    """Stacked two-market regression: every regressor interacted with market; FE within market."""
    D = []
    for k in MK:
        d = A[k][[ycol, fe, "day"] + cols].copy()
        for c in cols:
            for k2 in MK:
                d[f"{c}__{k2}"] = d[c] if k2 == k else 0.0
        d["fe2"] = k + "_" + d[fe].astype(str)
        D.append(d.drop(columns=cols))
    D = pd.concat(D)
    cc = [f"{c}__{k}" for k in MK for c in cols]
    return ols(D, ycol, cc, fe="fe2")


def indep_se(res, f_tr, f_mx):
    """SE of f_tr - f_mx treating the markets as independent (diagonal blocks of the stacked V)."""
    cols = res["cols"]
    V = res["V"].copy()
    itr = [i for i, c in enumerate(cols) if c.endswith("__TRY")]; imx = [i for i, c in enumerate(cols) if c.endswith("__MXN")]
    V[np.ix_(itr, imx)] = 0; V[np.ix_(imx, itr)] = 0
    r2 = dict(res); r2["V"] = V
    return dm(r2, lambda b: f_tr(b) - f_mx(b))[1]


def prep_R(a, qcol="q", beta=None):
    if beta is None:
        beta = ols(a, "y", [qcol, "up", "dn"])["b"][qcol]
    a["R"] = -a.s * (a.y - beta * a[qcol])
    a["beta_used"] = beta
    return beta


def add_bh(a, cols=("d_out", "d_in", "up", "dn", "q", "q_dem")):
    for c in cols:
        if c in a: a[c + "_bh"] = a[c] * a.bh


def pair(v):
    return [float(v[0]), float(v[1])]


# --- local-time design (Table emp_mxn) -------------------------------------------------
def local_design(A, ycol="R"):
    """Pooled sides, R on d_out, d_out_bh, d_in, d_in_bh, month x state FE, stacked."""
    for k in MK: add_bh(A[k])
    r = stack(A, ["d_out", "d_out_bh", "d_in", "d_in_bh"], "ms", ycol)
    out = {}
    lr = {}
    for k in MK:
        c0, c1 = f"d_out__{k}", f"d_out_bh__{k}"
        f = (lambda c0, c1: (lambda b: np.log((b[c0] + b[c1]) / b[c0])))(c0, c1)
        lr[k] = f
        out[k] = dict(closed=[r["b"][c0], r["se"][c0]], incr=[r["b"][c1], r["se"][c1]],
                      open=list(dm(r, lambda b, c0=c0, c1=c1: b[c0] + b[c1])),
                      ratio=float((r["b"][c0] + r["b"][c1]) / r["b"][c0]),
                      log_ratio=list(dm(r, f)) if (r["b"][c0] > 0 and r["b"][c0] + r["b"][c1] > 0) else [np.nan, np.nan],
                      d_in=[r["b"][f"d_in__{k}"], r["se"][f"d_in__{k}"]])
    if all(np.isfinite(out[k]["log_ratio"][0]) for k in MK):
        v, s = dm(r, lambda b: lr["TRY"](b) - lr["MXN"](b))
        out["diff_log_ratio"] = dict(est=v, se_stacked=s, se_indep=indep_se(r, lr["TRY"], lr["MXN"]))
    else:
        out["diff_log_ratio"] = dict(est=np.nan, se_stacked=np.nan, se_indep=np.nan)
    v, s = dm(r, lambda b: b["d_out_bh__TRY"] - b["d_out_bh__MXN"])
    out["diff_incr"] = dict(est=v, se_stacked=s, se_indep=float(np.hypot(r["se"]["d_out_bh__TRY"], r["se"]["d_out_bh__MXN"])))
    out["n"] = {k: int((A[k][ycol].notna()).sum()) for k in MK}
    return out


def local_sides(A):
    """y on q, up, dn and BH interactions, month x state FE, stacked (as the asym row of Table emp_mxn)."""
    for k in MK: add_bh(A[k])
    r = stack(A, ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"], "ms", "y")
    out = {}
    for side in ["up", "dn"]:
        o = {}
        f = {}
        for k in MK:
            c0, c1 = f"{side}__{k}", f"{side}_bh__{k}"
            f[k] = (lambda c0, c1: (lambda b: np.log((b[c0] + b[c1]) / b[c0])))(c0, c1)
            ok = r["b"][c0] > 0 and r["b"][c0] + r["b"][c1] > 0
            o[k] = dict(closed=[r["b"][c0], r["se"][c0]], incr=[r["b"][c1], r["se"][c1]],
                        open=list(dm(r, lambda b, c0=c0, c1=c1: b[c0] + b[c1])),
                        ratio=float((r["b"][c0] + r["b"][c1]) / r["b"][c0]),
                        log_ratio=list(dm(r, f[k])) if ok else [np.nan, np.nan])
        v, s = dm(r, lambda b: f["TRY"](b) - f["MXN"](b))
        o["diff_log_ratio"] = dict(est=v, se_stacked=s, se_indep=indep_se(r, f["TRY"], f["MXN"]))
        v, s = dm(r, lambda b, side=side: b[f"{side}_bh__TRY"] - b[f"{side}_bh__MXN"])
        o["diff_incr"] = dict(est=v, se_stacked=s,
                              se_indep=float(np.hypot(r["se"][f"{side}_bh__TRY"], r["se"][f"{side}_bh__MXN"])))
        out["above" if side == "up" else "below"] = o
    return out


def local_sides_R(A):
    """[r3] R on d_upo, d_dno, d_in and BH interactions, month x state FE, stacked."""
    for k in MK:
        for c in ["d_upo", "d_dno", "d_in"]:
            A[k][c + "_bh"] = A[k][c] * A[k].bh
    r = stack(A, ["d_upo", "d_upo_bh", "d_dno", "d_dno_bh", "d_in", "d_in_bh"], "ms", "R")
    out = {}
    for side, nm in [("d_upo", "above"), ("d_dno", "below")]:
        o, f = {}, {}
        for k in MK:
            c0, c1 = f"{side}__{k}", f"{side}_bh__{k}"
            f[k] = (lambda c0, c1: (lambda b: np.log((b[c0] + b[c1]) / b[c0])))(c0, c1)
            ok = r["b"][c0] > 0 and r["b"][c0] + r["b"][c1] > 0
            o[k] = dict(closed=[r["b"][c0], r["se"][c0]], incr=[r["b"][c1], r["se"][c1]],
                        open=list(dm(r, lambda b, c0=c0, c1=c1: b[c0] + b[c1])),
                        ratio=float((r["b"][c0] + r["b"][c1]) / r["b"][c0]),
                        log_ratio=list(dm(r, f[k])) if ok else [np.nan, np.nan])
        if all(np.isfinite(o[k]["log_ratio"][0]) for k in MK):
            v, se_ = dm(r, lambda b: f["TRY"](b) - f["MXN"](b))
            o["diff_log_ratio"] = dict(est=v, se_stacked=se_, se_indep=indep_se(r, f["TRY"], f["MXN"]))
        else:
            o["diff_log_ratio"] = dict(est=np.nan, se_stacked=np.nan, se_indep=np.nan)
        v, se_ = dm(r, lambda b, side=side: b[f"{side}_bh__TRY"] - b[f"{side}_bh__MXN"])
        o["diff_incr"] = dict(est=v, se_stacked=se_, se_indep=float(np.hypot(r["se"][f"{side}_bh__TRY"], r["se"][f"{side}_bh__MXN"])))
        out[nm] = o
    return out


# --- common-UTC-window design -----------------------------------------------------------
def window_design(A, ycol="R", wins=("W1", "W2"), regs=("d_out", "d_in"), fe_base="month", sides=False):
    B = {}
    for k in MK:
        d = A[k][A[k].win.isin(wins) & A[k][ycol].notna()].copy()
        for w in wins:
            ind = (d.win == w).astype(float)
            for c in regs:
                d[f"{c}_{w}"] = d[c] * ind
            if sides:  # y-spec: q by window too
                d[f"q_{w}"] = d.q * ind
        d["fe"] = d[fe_base] + "_" + d.win
        B[k] = d
    cols = [f"{c}_{w}" for c in regs for w in wins]
    if sides: cols = cols + [f"q_{w}" for w in wins]
    r = stack(B, cols, "fe", ycol)
    return r, B


def window_summary(r, reg="d_out", sign=1.0):
    """speeds by market x window, within-market log ratio W1/W2, DiD, level contrasts."""
    out = {}
    k_ = lambda k, w: f"{reg}_{w}__{k}"
    for k in MK:
        out[k] = {w: [sign * r["b"][k_(k, w)], r["se"][k_(k, w)]] for w in ["W1", "W2"]}
        if all(sign * r["b"][k_(k, w)] > 0 for w in ["W1", "W2"]):
            out[k]["log_W1_over_W2"] = list(dm(r, lambda b, k=k: np.log(b[k_(k, "W1")] / b[k_(k, "W2")])))
        else:
            out[k]["log_W1_over_W2"] = [np.nan, np.nan]
        out[k]["level_W1_minus_W2"] = list(dm(r, lambda b, k=k: sign * (b[k_(k, "W1")] - b[k_(k, "W2")])))
        out[k]["ratio_W1_over_W2"] = float(r["b"][k_(k, "W1")] / r["b"][k_(k, "W2")])
    fT = lambda b: np.log(b[k_("TRY", "W1")] / b[k_("TRY", "W2")])
    fM = lambda b: np.log(b[k_("MXN", "W1")] / b[k_("MXN", "W2")])
    if np.isfinite(out["TRY"]["log_W1_over_W2"][0]) and np.isfinite(out["MXN"]["log_W1_over_W2"][0]):
        v, s = dm(r, lambda b: fT(b) - fM(b))
        out["DiD_log"] = dict(est=v, se_stacked=s, se_indep=indep_se(r, fT, fM))
        # difference of open/closed log ratios within the common windows (TR: W1/W2; MX: W2/W1)
        v, s = dm(r, lambda b: fT(b) + fM(b))
        out["diff_open_closed_log_ratio"] = dict(est=v, se_stacked=s)
    else:
        out["DiD_log"] = dict(est=np.nan, se_stacked=np.nan, se_indep=np.nan)
        out["diff_open_closed_log_ratio"] = dict(est=np.nan, se_stacked=np.nan)
    v, s = dm(r, lambda b: sign * ((b[k_("TRY", "W1")] - b[k_("TRY", "W2")]) - (b[k_("MXN", "W1")] - b[k_("MXN", "W2")])))
    out["DiD_level"] = dict(est=v, se_stacked=s)
    out["n"] = int(r["n"])
    return out


def verdict(est, se):
    if not np.isfinite(est): return "undefined (a speed is not positive)"
    z = est / se
    if est > 0 and z > 1.96: return "survives (positive, significant at 5%)"
    if est > 0: return f"direction preserved, not significant (z={z:.2f})"
    return f"does not survive (sign reversed, z={z:.2f})"


# The JSON stores the module docstring up to "Outputs:" as prespec; keep that marker in the docstring.
OUT = dict(prespec=__doc__.split("Outputs:")[0])

# =================================================================== baseline build
log("building panels")
BASE = {k: build(k, "base") for k in MK}
BETA = {k: prep_R(BASE[k]) for k in MK}
OUT["beta"] = BETA
OUT["n_base"] = {k: int(len(BASE[k])) for k in MK}
log("built", OUT["n_base"], BETA)

# replicate Table emp_mxn (sanity check against tables/emp_mxn.json)
rep = local_design(BASE)
OUT["local_baseline"] = rep
log("replication: log ratios", rep["TRY"]["log_ratio"], rep["MXN"]["log_ratio"], rep["diff_log_ratio"])

# =================================================================== (a) common UTC windows
r_a, _ = window_design(BASE)
OUT["a_window"] = window_summary(r_a)
log("(a)", json.dumps(OUT["a_window"], default=float))

# (a2) bands re-profiled by month x UTC window
A2 = {}
for k in MK:
    d = BASE[k][BASE[k].win.isin(["W1", "W2"])].copy()
    d["mw"] = d.month + "_" + d.win
    for key, g in d.groupby("mw"):
        if len(g) < 300:
            d.loc[g.index, ["l", "u"]] = np.nan; continue
        l, u_ = profile(g)
        d.loc[g.index, "l"], d.loc[g.index, "u"] = l, u_
    d = d[d.l.notna()].copy()
    geom(d)
    prep_R(d, beta=BETA[k])
    A2[k] = d
r_a2, _ = window_design(A2)
OUT["a2_window_bands"] = window_summary(r_a2)
OUT["a2_window_bands"]["band_width"] = {k: A2[k].groupby(["month", "win"]).apply(lambda g: g.u.iloc[0] - g.l.iloc[0]).groupby(level=1).mean().to_dict() for k in MK}
log("(a2)", json.dumps(OUT["a2_window_bands"], default=float))

# (a5) by side within the windows: y on q, up, dn by window
for k in MK:
    pass
r_a5, _ = window_design(BASE, ycol="y", regs=("up", "dn"), sides=True)
OUT["a5_window_sides"] = {"above": window_summary(r_a5, "up"), "below": window_summary(r_a5, "dn")}
log("(a5)", json.dumps(OUT["a5_window_sides"], default=float))

# (a6) exploratory three-window decomposition
r_a6, _ = window_design(BASE, wins=("W0", "W1", "W2"))
kk = lambda k, w: f"d_out_{w}__{k}"
sp = {k: {w: [r_a6["b"][kk(k, w)], r_a6["se"][kk(k, w)]] for w in ["W0", "W1", "W2"]} for k in MK}
a6 = dict(speeds=sp)
if all(sp[k][w][0] > 0 for k in MK for w in sp[k]):
    a6["delta_TR"] = list(dm(r_a6, lambda b: np.log(b[kk("TRY", "W1")] / b[kk("TRY", "W0")]) - np.log(b[kk("MXN", "W1")] / b[kk("MXN", "W0")])))
    a6["delta_MX"] = list(dm(r_a6, lambda b: np.log(b[kk("MXN", "W2")] / b[kk("MXN", "W0")]) - np.log(b[kk("TRY", "W2")] / b[kk("TRY", "W0")])))
    a6["delta_TR_minus_delta_MX"] = list(dm(r_a6, lambda b: np.log(b[kk("TRY", "W1")] / b[kk("TRY", "W0")]) - np.log(b[kk("MXN", "W1")] / b[kk("MXN", "W0")])
                                            - np.log(b[kk("MXN", "W2")] / b[kk("MXN", "W0")]) + np.log(b[kk("TRY", "W2")] / b[kk("TRY", "W0")])))
    a6["reliable"] = bool(all(sp[k][w][0] > 2 * sp[k][w][1] for k in MK for w in sp[k]))
else:
    a6["note"] = "a speed is not positive; logs undefined"
OUT["a6_three_windows"] = a6
log("(a6)", json.dumps(a6, default=float))

# (a7) [r3] R-based by-side speeds within the windows
r_a7, _ = window_design(BASE, regs=("d_upo", "d_dno", "d_in"))
OUT["a7_window_sides_R"] = {"above": window_summary(r_a7, "d_upo"), "below": window_summary(r_a7, "d_dno")}
log("(a7)", json.dumps(OUT["a7_window_sides_R"], default=float)[:1500])

# =================================================================== (b) rollover exclusion
OUT["b_rollover"] = {}
for ex in ["none", "aligned"]:
    Ab = {k: build(k, ex) for k in MK}
    for k in MK: prep_R(Ab[k])
    OUT["b_rollover"][ex] = dict(local=local_design(Ab), sides=local_sides(Ab),
                                 n={k: int(len(Ab[k])) for k in MK})
    if ex == "none":
        # [r2] MXN without the exclusion, Turkey at baseline
        Am = {"TRY": BASE["TRY"].copy(), "MXN": Ab["MXN"].copy()}
        OUT["b_rollover"]["none_mxn_only"] = dict(local=local_design(Am), sides=local_sides(Am))
        log("(b) none_mxn_only", OUT["b_rollover"]["none_mxn_only"]["local"]["diff_log_ratio"])
        # with no exclusion Mexico's full 09:00-17:00 business day is observed: UTC 15:00-23:00
        for k in MK:
            Ab[k]["win"] = np.where((Ab[k].uh >= 6) & (Ab[k].uh < 14), "W1", np.where((Ab[k].uh >= 15) & (Ab[k].uh < 23), "W2", ""))
        rb, _ = window_design(Ab)
        OUT["b_rollover"][ex]["window_W2_15_23"] = window_summary(rb)
    log("(b)", ex, json.dumps(OUT["b_rollover"][ex]["local"]["diff_log_ratio"], default=float))
    del Ab

# =================================================================== (c) thin trading
OUT["c_thin"] = {}
med = {}
for k in MK:
    a = BASE[k]
    a["vmed"] = a.groupby("ms").vol.transform("median")
    a["f_c1"] = (a.ntr >= 10) & (a.ntr_l >= 10)
    a["f_c2"] = (a.vol > a.vmed) & (a.vol_l > a.vmed)
OUT["c_thin"]["share_kept"] = {f: {k: {st: float(BASE[k].loc[BASE[k].state == st, f].mean()) for st in ["open", "closed"]}
                                   for k in MK} for f in ["f_c1", "f_c2"]}
OUT["c_thin"]["share_kept_window"] = {f: {k: {w: float(BASE[k].loc[BASE[k].win == w, f].mean()) for w in ["W1", "W2"]}
                                          for k in MK} for f in ["f_c1", "f_c2"]}
OUT["c_thin"]["median_trades_5min"] = {k: {w: float(BASE[k].loc[BASE[k].win == w, "ntr"].median()) for w in ["W0", "W1", "W2"]} for k in MK}
for f in ["f_c1", "f_c2"]:
    for scope in ["mxn_only", "both"]:
        Ac = {}
        for k in MK:
            if scope == "both" or k == "MXN":
                Ac[k] = BASE[k][BASE[k][f]].copy()
            else:
                Ac[k] = BASE[k].copy()
        rw, _ = window_design(Ac)
        OUT["c_thin"][f"{f}_{scope}"] = dict(local=local_design(Ac), window=window_summary(rw))
        log("(c)", f, scope, OUT["c_thin"][f"{f}_{scope}"]["local"]["diff_log_ratio"], OUT["c_thin"][f"{f}_{scope}"]["window"]["DiD_log"])
# (c3) distance from b_{t-2}
Ac = {}
for k in MK:
    d = BASE[k][BASE[k].c2ok & BASE[k].x2.notna()].copy()
    geom(d, "x2")
    prep_R(d, beta=BETA[k])
    Ac[k] = d
rw, _ = window_design(Ac)
OUT["c_thin"]["c3_lag2"] = dict(local=local_design(Ac), window=window_summary(rw), sides=local_sides(Ac))
log("(c3)", OUT["c_thin"]["c3_lag2"]["local"]["diff_log_ratio"], OUT["c_thin"]["c3_lag2"]["window"]["DiD_log"])
# same-sample lag-1 benchmark for c3 (three consecutive intervals required)
Ac1 = {k: BASE[k][BASE[k].c2ok & BASE[k].x2.notna()].copy() for k in MK}
rw1, _ = window_design(Ac1)
OUT["c_thin"]["c3_lag1_same_sample"] = dict(local=local_design(Ac1), window=window_summary(rw1))
del Ac, Ac1

# =================================================================== (d) sides, local-time design
OUT["d_sides_local"] = local_sides(BASE)
OUT["d_sides_local"]["pooled"] = {k: OUT["local_baseline"][k] for k in MK}
OUT["d_sides_local"]["pooled"]["diff_log_ratio"] = OUT["local_baseline"]["diff_log_ratio"]
OUT["d_sides_local"]["pooled"]["diff_incr"] = OUT["local_baseline"]["diff_incr"]
log("(d)", json.dumps(OUT["d_sides_local"], default=float)[:2000])
OUT["d_sides_local_R"] = local_sides_R(BASE)
log("(d R)", json.dumps(OUT["d_sides_local_R"], default=float)[:2000])

# =================================================================== (e) demand-only OFI
def size_flows(k):
    d = "ofi_size" if k == "TRY" else "ofi_size_mxn"
    z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, d, "*.parquet")))])
    return z[~z.index.duplicated()]


Ae = {}
OUT["e_demand"] = {"check_corr_total_ofi": {}, "large_share_of_volume": {}}
for k in MK:
    z = size_flows(k)
    d = BASE[k].join(z, how="left")
    OUT["e_demand"]["check_corr_total_ofi"][k] = float(np.corrcoef(d.ofi.fillna(0), (d.ofi_small + d.ofi_mid + d.ofi_large).fillna(0))[0, 1])
    OUT["e_demand"]["large_share_of_volume"][k] = float(d.vol_large.sum() / (d.vol_small + d.vol_mid + d.vol_large).sum())
    d = d[d.ofi_small.notna()].copy()
    dem = d.ofi_small + d.ofi_mid
    d["q_dem"] = dem / dem.groupby(d.month).transform("std")
    b_ = prep_R(d, "q_dem")
    OUT["e_demand"].setdefault("beta_dem", {})[k] = float(b_)
    Ae[k] = d
rw, _ = window_design(Ae)
OUT["e_demand"]["local"] = local_design(Ae)
OUT["e_demand"]["window"] = window_summary(rw)
log("(e)", OUT["e_demand"]["local"]["diff_log_ratio"], OUT["e_demand"]["window"]["DiD_log"])

# =================================================================== UTC cycle for the figure
cyc = {}
for k in MK:
    rows = []
    for h0 in range(0, 24, 2):
        d = BASE[k][(BASE[k].uh >= h0) & (BASE[k].uh < h0 + 2)]
        if (d.d_out > 0).sum() < 200: continue
        r = ols(d, "R", ["d_out", "d_in"], fe="month")
        rows.append(dict(h=h0 + 1, k=float(r["b"]["d_out"]), se=float(r["se"]["d_out"]), n=int(r["n"])))
    cyc[k] = rows
OUT["utc_cycle"] = cyc

# =================================================================== verdicts
V = {}
V["a_window_DiD"] = verdict(OUT["a_window"]["DiD_log"]["est"], OUT["a_window"]["DiD_log"]["se_stacked"])
V["a2_window_bands_DiD"] = verdict(OUT["a2_window_bands"]["DiD_log"]["est"], OUT["a2_window_bands"]["DiD_log"]["se_stacked"])
V["a5_above_DiD"] = verdict(OUT["a5_window_sides"]["above"]["DiD_log"]["est"], OUT["a5_window_sides"]["above"]["DiD_log"]["se_stacked"])
V["a5_below_DiD"] = verdict(OUT["a5_window_sides"]["below"]["DiD_log"]["est"], OUT["a5_window_sides"]["below"]["DiD_log"]["se_stacked"])
V["local_baseline"] = verdict(OUT["local_baseline"]["diff_log_ratio"]["est"], OUT["local_baseline"]["diff_log_ratio"]["se_stacked"])
for ex in ["none", "aligned"]:
    x = OUT["b_rollover"][ex]["local"]["diff_log_ratio"]; V[f"b_{ex}"] = verdict(x["est"], x["se_stacked"])
for key in ["f_c1_mxn_only", "f_c1_both", "f_c2_mxn_only", "f_c2_both", "c3_lag2"]:
    x = OUT["c_thin"][key]["local"]["diff_log_ratio"]; V[f"c_{key}_local"] = verdict(x["est"], x["se_stacked"])
    x = OUT["c_thin"][key]["window"]["DiD_log"]; V[f"c_{key}_window"] = verdict(x["est"], x["se_stacked"])
for side in ["above", "below"]:
    x = OUT["d_sides_local"][side]["diff_log_ratio"]; V[f"d_{side}_local"] = verdict(x["est"], x["se_stacked"])
for side in ["above", "below"]:
    x = OUT["d_sides_local_R"][side]["diff_log_ratio"]; V[f"d_R_{side}_local"] = verdict(x["est"], x["se_stacked"])
    x = OUT["a7_window_sides_R"][side]["DiD_log"]; V[f"a7_R_{side}_window"] = verdict(x["est"], x["se_stacked"])
x = OUT["b_rollover"]["none_mxn_only"]["local"]["diff_log_ratio"]; V["b_none_mxn_only"] = verdict(x["est"], x["se_stacked"])
x = OUT["e_demand"]["local"]["diff_log_ratio"]; V["e_demand_local"] = verdict(x["est"], x["se_stacked"])
x = OUT["e_demand"]["window"]["DiD_log"]; V["e_demand_window"] = verdict(x["est"], x["se_stacked"])
OUT["verdicts"] = V
json.dump(OUT, open(os.path.join(TAB, "rob_mexico_utc.json"), "w"), indent=1, default=float)
log("verdicts", json.dumps(V, indent=1))


# =================================================================== table
def f4(v): return "--" if not np.isfinite(v) else f"{v:.4f}"
def f2(v): return "--" if not np.isfinite(v) else f"{v:.2f}"
def st(c, s):
    if not (np.isfinite(c) and np.isfinite(s)) or s == 0: return ""
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""
def cs(v, fmt=f2): return f"${fmt(v[0])}{st(v[0], v[1])}$ $({fmt(v[1])})$"
def dd(x, fmt=f2): return f"${fmt(x['est'])}{st(x['est'], x['se_stacked'])}$ $({fmt(x['se_stacked'])})$"


L = []
L.append("\\begin{tabular}{lcccccc}\n\\toprule\n")
L.append("\\multicolumn{7}{l}{\\emph{Panel A. Common UTC windows: W1 = 06:00--14:00 UTC (Turkey open, Mexico closed); W2 = 15:00--20:30 UTC (Turkey closed, Mexico open)}} \\\\\n")
L.append(" & \\multicolumn{2}{c}{Turkey} & \\multicolumn{2}{c}{Mexico} & & \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n")
L.append(" & $\\hat\\kappa_{W1}$ & $\\hat\\kappa_{W2}$ & $\\hat\\kappa_{W1}$ & $\\hat\\kappa_{W2}$ & & \\\\\n")
L.append(" & \\multicolumn{2}{c}{$\\log(\\kappa_{W1}/\\kappa_{W2})$} & \\multicolumn{2}{c}{$\\log(\\kappa_{W1}/\\kappa_{W2})$} & DiD (log) & $N$ \\\\\n\\midrule\n")
rowsA = [("Baseline bands, pooled sides", OUT["a_window"]),
         ("Bands by month $\\times$ UTC window", OUT["a2_window_bands"]),
         ("Above the band only", OUT["a5_window_sides"]["above"]),
         ("Below the band only", OUT["a5_window_sides"]["below"]),
         ("Above the band only, $R$-based", OUT["a7_window_sides_R"]["above"]),
         ("Below the band only, $R$-based", OUT["a7_window_sides_R"]["below"]),
         ("MXN: $\\geq$10 trades in $t-1$ and $t$", OUT["c_thin"]["f_c1_mxn_only"]["window"]),
         ("MXN: volume above cell median in $t-1$ and $t$", OUT["c_thin"]["f_c2_mxn_only"]["window"]),
         ("Distance measured from $b_{t-2}$ (both)", OUT["c_thin"]["c3_lag2"]["window"]),
         ("Demand-only OFI control (both)", OUT["e_demand"]["window"])]
for lab, w in rowsA:
    L.append(f"{lab} & ${f4(w['TRY']['W1'][0])}{st(*w['TRY']['W1'])}$ & ${f4(w['TRY']['W2'][0])}{st(*w['TRY']['W2'])}$ & "
             f"${f4(w['MXN']['W1'][0])}{st(*w['MXN']['W1'])}$ & ${f4(w['MXN']['W2'][0])}{st(*w['MXN']['W2'])}$ & & \\\\\n")
    L.append(f" & $({f4(w['TRY']['W1'][1])})$ & $({f4(w['TRY']['W2'][1])})$ & $({f4(w['MXN']['W1'][1])})$ & $({f4(w['MXN']['W2'][1])})$ & & \\\\\n")
    L.append(f" & \\multicolumn{{2}}{{c}}{{{cs(w['TRY']['log_W1_over_W2'])}}} & \\multicolumn{{2}}{{c}}{{{cs(w['MXN']['log_W1_over_W2'])}}} & {dd(w['DiD_log'])} & {w['n']:,} \\\\[2pt]\n")
L.append("\\midrule\n")
L.append("\\multicolumn{7}{l}{\\emph{Panel B. Local business hours (Table~\\ref{tab:emp_mxn} design): open/closed ratio of the speed outside the band}} \\\\\n")
L.append(" & \\multicolumn{2}{c}{Turkey} & \\multicolumn{2}{c}{Mexico} & & \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n")
L.append(" & ratio & $\\log$ ratio & ratio & $\\log$ ratio & Difference & $N$ (MXN) \\\\\n\\midrule\n")
rowsB = [("Baseline (Table~\\ref{tab:emp_mxn})", OUT["local_baseline"]),
         ("No rollover exclusion", OUT["b_rollover"]["none"]["local"]),
         ("No rollover exclusion, MXN only", OUT["b_rollover"]["none_mxn_only"]["local"]),
         ("Aligned local-time exclusion", OUT["b_rollover"]["aligned"]["local"]),
         ("MXN: $\\geq$10 trades in $t-1$ and $t$", OUT["c_thin"]["f_c1_mxn_only"]["local"]),
         ("MXN: volume above cell median in $t-1$ and $t$", OUT["c_thin"]["f_c2_mxn_only"]["local"]),
         ("Both: volume above cell median", OUT["c_thin"]["f_c2_both"]["local"]),
         ("Distance measured from $b_{t-2}$ (both)", OUT["c_thin"]["c3_lag2"]["local"]),
         ("Demand-only OFI control (both)", OUT["e_demand"]["local"])]
for lab, w in rowsB:
    L.append(f"{lab} & {w['TRY']['ratio']:.1f} & {cs(w['TRY']['log_ratio'])} & {w['MXN']['ratio']:.1f} & {cs(w['MXN']['log_ratio'])} & {dd(w['diff_log_ratio'])} & {w['n']['MXN']:,} \\\\\n")
L.append("\\midrule\n")
L.append("\\multicolumn{7}{l}{\\emph{Panel C. By side of the band, local business hours: closed speed, business-hours increment (level), open/closed}} \\\\\n")
L.append(" & \\multicolumn{2}{c}{Turkey} & \\multicolumn{2}{c}{Mexico} & \\multicolumn{2}{c}{Difference} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n")
L.append(" & closed & increment & closed & increment & increment & $\\log$ ratio \\\\\n\\midrule\n")
S = OUT["d_sides_local"]
for lab, w in [("Above the band ($\\kappa^+$)", S["above"]), ("Below the band ($\\kappa^-$)", S["below"]),
               ("Above the band, $R$-based", OUT["d_sides_local_R"]["above"]), ("Below the band, $R$-based", OUT["d_sides_local_R"]["below"]),
               ("Pooled sides ($\\kappa$, $R$-based)", S["pooled"])]:
    L.append(f"{lab} & {cs(w['TRY']['closed'], f4)} & {cs(w['TRY']['incr'], f4)} & {cs(w['MXN']['closed'], f4)} & {cs(w['MXN']['incr'], f4)} & {dd(w['diff_incr'], f4)} & {dd(w['diff_log_ratio'])} \\\\\n")
    L.append(f"\\quad open/closed ratio & \\multicolumn{{2}}{{c}}{{{w['TRY']['ratio']:.1f}}} & \\multicolumn{{2}}{{c}}{{{w['MXN']['ratio']:.1f}}} & & \\\\\n")
L.append("\\bottomrule\n\\end{tabular}\n")
open(os.path.join(TAB, "rob_mexico_utc.tex"), "w").write("".join(L))

# =================================================================== figure
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
for i, (k, col, mk, lab) in enumerate([("TRY", C1, "o", "Turkey"), ("MXN", C2, "s", "Mexico")]):
    c = pd.DataFrame(cyc[k])
    ax[i].errorbar(c.h, c.k, yerr=1.96 * c.se, fmt=mk + "-", ms=3, color=col, ecolor=col, lw=0.9)
    o0, o1 = (9 - UTC_OFF[k]) % 24, (17 - UTC_OFF[k]) % 24
    ax[i].axvspan(o0, o1, color=col, alpha=0.10, lw=0)
    ax[i].axvspan(20.5, 22.5, color=C3, alpha=0.25, lw=0, hatch="///", fill=False)
    for w0, w1, nm in [(6, 14, "W1"), (15, 20.5, "W2")]:
        ax[i].annotate("", xy=(w0, 1.0), xytext=(w1, 1.0), xycoords=("data", "axes fraction"),
                       arrowprops=dict(arrowstyle="<->", lw=0.6, color="k"))
        ax[i].text((w0 + w1) / 2, 1.02, nm, transform=ax[i].get_xaxis_transform(), ha="center", fontsize=7)
    ax[i].axhline(0, color="k", lw=0.4)
    ax[i].set_xticks(range(0, 25, 6)); ax[i].set_xlim(0, 24)
    ax[i].set_xlabel("UTC hour")
    ax[i].set_title(f"({'ab'[i]}) {lab}: banks open shaded", fontsize=9, pad=12)
ax[0].set_ylabel(r"speed outside the band, $\hat\kappa$")
w = OUT["a_window"]
for j, (k, col, mk) in enumerate([("TRY", C1, "o"), ("MXN", C2, "s")]):
    v, s = w[k]["log_W1_over_W2"]
    ax[2].errorbar([j], [v], yerr=[1.96 * s], fmt=mk, color=col, ms=4, capsize=2)
v, s = w["DiD_log"]["est"], w["DiD_log"]["se_stacked"]
ax[2].errorbar([2], [v], yerr=[1.96 * s], fmt="D", color="k", ms=4, capsize=2)
ax[2].axhline(0, color="k", lw=0.4)
ax[2].set_xticks([0, 1, 2]); ax[2].set_xticklabels(["Turkey", "Mexico", "DiD"], fontsize=8)
ax[2].set_xlim(-0.6, 2.6)
ax[2].set_ylabel(r"$\log(\hat\kappa_{W1}/\hat\kappa_{W2})$")
ax[2].set_title("(c) W1 vs W2", fontsize=9, pad=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "rob_mexico_utc.pdf")); plt.close(fig)
log("done")
