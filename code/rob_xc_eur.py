"""Cross-country extension of Table emp_mxn: the euro (Binance EUR/USDT against Dukascopy EUR/USD).

DESIGN
======
Market and prices.  Binance quotes EUR/USDT (USDT per EUR); Dukascopy quotes EUR/USD (USD per EUR).
The local currency is the euro, so the local price of USDT is 1/EURUSDT and the local price of the
dollar is 1/EURUSD.  The premium of USDT in euros is
    b = 1e4 * (log EURUSD_mid - log EURUSDT_vwap)   [= 1e4 log(P_EUR(USDT) / P_EUR(USD))],
the same object as log(USDT/TRY VWAP / USD/TRY mid) in Turkey: b > 0 means USDT is expensive in
euros.  EURUSDT_vwap is the minute quote volume over base volume.  Order flow is oriented as USDT
purchases with euros, i.e. taker SELLS of EUR/USDT, measured in USDT:
    OFI = QV - 2 * TBQV   (quote volume minus twice taker-buy quote volume),
so that, as in Turkey, positive OFI pushes b up.  Volume is quote volume (USDT).

Construction: identical to empirics_mxn.build(): minute premium against the Dukascopy mid (bid and
ask candles; mid = bid x (1 + half the median spread) on any day without an ask file), FX
forward-filled at most two minutes, weekdays only, 20:30-22:30 UTC rollover excluded, |b| < 1000 bp,
five-minute means with at least two valid minutes, OFI summed and standardised by its within-month
s.d., consecutive intervals only, bands profiled (1 bp grid, 5% trimming, model._ssr_grid) by UTC
month x local banking state (cells with at least 300 intervals), month-by-state fixed effects,
standard errors clustered by UTC date with the finite-sample correction.  Local business hours are
09:00-17:00 Frankfurt time (Europe/Berlin via zoneinfo, so CET in winter and CEST in summer).
Window: January 2024 - August 2026 (all months with Binance EUR/USDT klines).

Tests (pooled sides, Table emp_mxn specification):
 (1) beta from y ~ q + up + dn (month x state FE); R = -s (dy - beta q); R on d_out, d_out x BH,
     d_in, d_in x BH.  Closed-hours speed kappa_c = coef(d_out); business-hours increment
     = coef(d_out x BH); open/closed ratio (kappa_c + incr)/kappa_c; its log with delta-method s.e.
     The log ratio is reported only if both the closed and the open speed are positive.
 (2) Above-band and below-band versions: R on (x-u)+, (l-x)+, d_in and their BH interactions
     (R-based), and the asymmetric y ~ q + up + dn (+ x BH) regression of Table emp_mxn (y-based).
 (3) Daily cycle: R on d_out, d_in by two-hour block of Frankfurt time, month FE (blocks with fewer
     than 200 out-of-band intervals are skipped), as Figure emp_mxn_cycle.  Night = blocks 00-08,
     day = blocks 10-16; also the pooled night and day speeds.
 (4) Stacked comparison with Turkey on the common months (Jan 2024 - Aug 2026), Turkey built exactly
     as in empirics_mxn.build() (UTC+3) over the same months: both markets in one regression,
     regressors and FE interacted with market, clustered by UTC date (the same date is one cluster
     in both markets), so the difference in log ratios log r_TRY - log r_EUR has a joint s.e.;
     the independent-samples s.e. of Table emp_mxn is reported alongside.
 (5) Direct comparability with Table emp_mxn: the same EUR estimates on its window
     (Nov 2024 - Aug 2026), stacked with Turkey and Mexico built as in empirics_mxn.py (this
     re-creates the 4.2 and 1.6 ratios as a check).
 Robustness: (r1) drop TARGET2 closing days (New Year, Good Friday, Easter Monday,
 1 May, 25-26 December) and re-profile the bands; (r2) keep intervals with active interbank quoting
 (Dukascopy tick volume over the interval at least the 25th percentile of its business-hours value,
 as in empirics_extra.py), baseline bands; (r3) the bid-based mid on every day, as for Mexico, whose
 Dukascopy files are bid only (relevant only if ask files do not cover every day).
 Data quality (reported, not used to select anything): share of weekday minutes with trades; median
 minute USDT volume and trade count (overall, by month, by Frankfurt hour); Dukascopy activity by
 Frankfurt hour (share of minutes with non-zero tick volume, with a change in the bid close, median
 tick volume, median bid-ask spread) and the same Dukascopy statistics for USD/TRY by Istanbul hour;
 share of five-minute intervals in which the FX mid moves.

EXPLORATORY (the leg decomposition in `robustness` shows the EUR/USD leg moving away from the band
in business hours): how closely the Binance
price tracks the interbank rate (dpl on dpg and lagged dpg with the asymmetric regressors), the
variance of the FX leg relative to the premium change by banking state, and the pooled ratio with
the lagged FX move, -s*dpg_{t-1}, as an added control (and its business-hours interaction).

Measurement details: (i) the interval tick volume `fxv` counts Dukascopy tick volume only in the minute
it was recorded (it is not forward-filled with the mid, which would double-count stale minutes; this
matters for robustness r2 only); (ii) the Dukascopy activity profiles drop the minutes after the Friday
close and the rollover window.
DIAGNOSTICS, all with bands and beta
fixed at their baseline values unless stated: r1 decomposed into dropping the TARGET days with the
baseline bands vs. re-profiling the bands; the single most influential UTC day (first-order influence
on the log ratio, outcome-selected and labelled as such); dropping the 3 or 7 days with the largest
premium relative to its month (max |x - month median| / month s.d., chosen without the estimates);
the same diagnostics for Turkey and Mexico; subperiods 2024 and 2025-01..2026-08; stacked differences
after dropping each market's 7 most extreme days.

What the rails predict: euro-area instant credit transfers
(SCT Inst through TIPS and RT1) settle around the clock, while the large-value RTGS (T2) runs on
TARGET business days only.  If the 24/7 instant rail carries the euro leg of USDT arbitrage, the
open/closed ratio should look like Mexico's (about 1.6); if large-value settlement binds, like
Turkey's (about 4.2).  The dollar leg is common to all three markets.
Inputs: data_raw/binance (EURUSDT, USDTTRY, USDTMXN), data_raw/dukascopy (EURUSD and USDTRY bid/ask, USDMXN bid).
Outputs: tables/rob_xc_eur.json, tables/rob_xc_eur.tex, figures/rob_xc_eur.pdf.
"""
import os, json, time, glob, lzma, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import _ssr_grid
from build_panel import load_binance, load_fx, COLS

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4 = "#1f4e79", "#c0504d", "#7f7f7f", "#2e7d32"
W0, W1 = "2024-01-01", "2026-08-31"      # EUR window = common window with Turkey
WM0 = "2024-11-01"                        # start of the Table emp_mxn window
UTC_OFF = {"TRY": 3, "MXN": -6}
BERLIN = ZoneInfo("Europe/Berlin")
TARGET_CLOSED = ["2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-12-25", "2024-12-26",
                 "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-12-25", "2025-12-26",
                 "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-01"]


def log(*s):
    print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)


# ------------------------------------------------------------------ data
def load_klines(sym):
    """Binance one-minute klines with quote and taker-buy quote volume (same timestamp logic as load_binance)."""
    dfs = []
    for f in sorted(glob.glob(os.path.join(RAW, "binance", f"{sym}-1m-*.csv"))):
        d = pd.read_csv(f, header=None, names=COLS)
        t = d.open_time.astype("int64")
        ts = np.where(t > 1e14, t // 1000, t)
        d.index = pd.to_datetime(ts, unit="ms", utc=True)
        dfs.append(d)
    d = pd.concat(dfs)
    return d[~d.index.duplicated()].sort_index()


_KL, _FX, _BIN = {}, {}, {}


def kl_eur():
    if "EUR" not in _KL: _KL["EUR"] = load_klines("EURUSDT")
    return _KL["EUR"]


def fx(pair):
    if pair not in _FX: _FX[pair] = load_fx(pair)
    return _FX[pair]


def binance(sym):
    if sym not in _BIN: _BIN[sym] = load_binance(sym)
    return _BIN[sym]


def fx_bid_only(pair):
    """load_fx output rebuilt from the bid candles alone: mid = bid x (1 + half the median spread) on every day."""
    key = pair + "_bidonly"
    if key not in _FX:
        b = duk_raw(pair, "BID")
        b = b[~b.index.duplicated()]
        b = b[b.fxv > 0]
        half = 0.5 * fx(pair).fx_spread.median()
        _FX[key] = pd.DataFrame({"fx_mid": b.c * (1 + half), "fxv": b.fxv})
    return _FX[key]


def raw_minutes(mkt, w0, w1, bid_only=False):
    if mkt == "EUR":
        k = kl_eur().loc[w0:w1]
        vw = np.where(k.v > 0, k.qv / k.v.replace(0, np.nan), np.nan)
        p = pd.DataFrame({"pl": 1.0 / vw, "ofi": k.qv - 2 * k.tbqv, "vol": k.qv, "ntr": k.n}, index=k.index)
        src = fx_bid_only("EURUSD") if bid_only else fx("EURUSD")
        f = src.reindex(k.index, method="ffill", limit=2)
        p["fx_mid"] = 1.0 / f.fx_mid          # EUR per USD
        # tick volume only in the minute Dukascopy recorded it (a forward fill of the mid would copy the last
        # tick volume into up to two stale minutes, overstating quote activity)
        p["fxv"] = src.fxv.reindex(k.index).fillna(0.0)
    else:
        u = binance(f"USDT{mkt}").loc[w0:w1]
        src = fx(f"USD{mkt}")
        p = pd.DataFrame({"pl": u.vwap, "ofi": 2 * u.tbv - u.v, "vol": u.v, "ntr": u.n}).join(
            src[["fx_mid"]].reindex(u.index, method="ffill", limit=2))
        p["fxv"] = src.fxv.reindex(u.index).fillna(0.0)
    return p


# ------------------------------------------------------------------ construction (as empirics_mxn.build)
def profile(d, trim=0.05):
    y, x, q = d.y.values, d.x.values, d.q.values
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + 1.0, 1.0)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]


def local_hour(idx, mkt):
    if mkt == "EUR":
        return np.asarray(idx.tz_convert(BERLIN).hour)
    return np.asarray((idx.hour + UTC_OFF[mkt]) % 24)


def build(mkt, w0=W0, w1=W1, drop_days=None, bid_only=False):
    p = raw_minutes(mkt, w0, w1, bid_only=bid_only)
    p["b"] = 1e4 * np.log(p.pl / p.fx_mid)
    hr = p.index.hour + p.index.minute / 60
    p.loc[(hr >= 20.5) & (hr < 22.5), "b"] = np.nan
    p.loc[p.index.dayofweek >= 5, "b"] = np.nan
    p.loc[p.b.abs() > 1000, "b"] = np.nan
    if drop_days is not None:
        p.loc[pd.Index(p.index.date).isin(drop_days), "b"] = np.nan
    g = p.resample("5min")
    a = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum(), "vol": g.vol.sum(), "ntr": g.ntr.sum(),
                      "fxv": g.fxv.sum(), "pl": np.log(g.pl.mean()), "pg": np.log(g.fx_mid.mean())})
    a = a[a.nb >= 2]
    a["month"] = a.index.tz_localize(None).to_period("M").astype(str)
    a["q"] = a.ofi / a.ofi.groupby(a.month).transform("std")
    a["dpl"] = 1e4 * a.pl.diff(); a["dpg"] = 1e4 * a.pg.diff()
    consec = a.index.to_series().diff() == pd.Timedelta("5min")
    a["y"] = a.b.diff(); a["x"] = a.b.shift(1)
    a = a[consec & a.y.notna() & a.x.notna() & a.q.notna()].copy()
    lh = local_hour(a.index, mkt)
    a["lh"] = lh
    a["bh"] = ((lh >= 9) & (lh < 17)).astype(float)
    a["state"] = np.where(a.bh == 1, "open", "closed")
    a["ms"] = a.month + "_" + a.state
    for key, d in a.groupby("ms"):
        if len(d) < 300: continue
        l, u_ = profile(d)
        a.loc[d.index, "l"], a.loc[d.index, "u"] = l, u_
    a = a[a.l.notna()].copy()
    m = (a.l + a.u) / 2
    a["s"] = np.where(a.x > a.u, 1.0, np.where(a.x < a.l, -1.0, np.sign(a.x - m)))
    a["d_out"] = np.maximum(a.x - a.u, 0) + np.maximum(a.l - a.x, 0)
    a["d_in"] = np.where(a.d_out == 0, np.abs(a.x - m), 0.0)
    a["up"] = -np.maximum(a.x - a.u, 0); a["dn"] = np.maximum(a.l - a.x, 0)
    a["d_upo"] = np.maximum(a.x - a.u, 0); a["d_dno"] = np.maximum(a.l - a.x, 0)
    a["day"] = a.index.date
    a["mkt"] = mkt
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
    return dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), n=n, V=V, cols=list(cols), bv=b,
                ndays=int(ng))


def dm(res, fun):
    """value and delta-method s.e. of fun(coef dict) (central differences)."""
    b0 = res["bv"].copy(); cols = res["cols"]
    f0 = fun(dict(zip(cols, b0)))
    g = np.zeros(len(b0))
    for i in range(len(b0)):
        h = 1e-6 * max(abs(b0[i]), 1e-4)
        bp, bm = b0.copy(), b0.copy(); bp[i] += h; bm[i] -= h
        g[i] = (fun(dict(zip(cols, bp))) - fun(dict(zip(cols, bm)))) / (2 * h)
    return float(f0), float(np.sqrt(g @ res["V"] @ g))


def prep(a):
    beta = ols(a, "y", ["q", "up", "dn"])["b"]["q"]
    a["R"] = -a.s * (a.y - beta * a.q)
    for c in ["d_out", "d_in", "up", "dn", "q", "d_upo", "d_dno"]:
        a[c + "_bh"] = a[c] * a.bh
    return beta


def ratio_block(r, c0, c1):
    k0, k1 = r["b"][c0], r["b"][c1]
    ok = (k0 > 0) and (k0 + k1 > 0)
    f = lambda b: np.log((b[c0] + b[c1]) / b[c0])
    lr = dm(r, f) if ok else (np.nan, np.nan)
    return dict(closed=[float(k0), float(r["se"][c0])], incr=[float(k1), float(r["se"][c1])],
                open=list(dm(r, lambda b: b[c0] + b[c1])), ratio=float((k0 + k1) / k0) if k0 != 0 else np.nan,
                logratio=[float(lr[0]), float(lr[1])])


def estimate(a):
    beta = prep(a)
    sym = ols(a, "R", ["d_out", "d_out_bh", "d_in", "d_in_bh"])
    asym = ols(a, "y", ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
    sideR = ols(a, "R", ["d_upo", "d_upo_bh", "d_dno", "d_dno_bh", "d_in", "d_in_bh"])
    w = a.groupby(["month", "state"]).agg(l=("l", "first"), u=("u", "first")).reset_index(); w["w"] = w.u - w.l
    out = dict(n=int(len(a)), days=int(a.day.nunique()), months=sorted(a.month.unique().tolist()), beta=float(beta),
               pooled=ratio_block(sym, "d_out", "d_out_bh"),
               d_in=[float(sym["b"]["d_in"]), float(sym["se"]["d_in"])],
               d_in_bh=[float(sym["b"]["d_in_bh"]), float(sym["se"]["d_in_bh"])],
               above_R=ratio_block(sideR, "d_upo", "d_upo_bh"), below_R=ratio_block(sideR, "d_dno", "d_dno_bh"),
               above_y=ratio_block(asym, "up", "up_bh"), below_y=ratio_block(asym, "dn", "dn_bh"),
               asym={c: [float(asym["b"][c]), float(asym["se"][c])] for c in asym["cols"]},
               sd_prem=float(a.x.std()), mean_prem=float(a.x.mean()),
               sd_prem_within_month=float(a.groupby("month").x.std().median()),
               share_out=float((a.d_out > 0).mean()), share_above=float((a.d_upo > 0).mean()),
               share_below=float((a.d_dno > 0).mean()),
               width_open=float(w[w.state == "open"].w.mean()), width_closed=float(w[w.state == "closed"].w.mean()),
               vol_per_min=float(a.vol.mean() / 5))
    return out, sym


def tracking(a):
    """Exploratory: Binance-to-interbank tracking and the ratio net of lagged FX moves."""
    d = a.dropna(subset=["dpg", "dpl"]).copy()
    ok = d.index.to_series().diff() == pd.Timedelta("5min")
    d["dpg_l"] = d.dpg.shift(1).where(ok)
    d = d.dropna(subset=["dpg_l"])
    d["dpg_bh"] = d.dpg * d.bh; d["dpg_l_bh"] = d.dpg_l * d.bh
    r = ols(d, "dpl", ["dpg", "dpg_l", "dpg_bh", "dpg_l_bh", "q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
    d["sdpg_l"] = -d.s * d.dpg_l; d["sdpg_l_bh"] = d.sdpg_l * d.bh
    r1 = ols(d, "R", ["d_out", "d_out_bh", "d_in", "d_in_bh", "sdpg_l", "sdpg_l_bh"])
    var = {("open" if st else "closed"): dict(var_dpg_over_var_db=float(d[d.bh == st].dpg.var() / d[d.bh == st].y.var()),
                                              corr_db_minus_dpg=float(np.corrcoef(d[d.bh == st].y, -d[d.bh == st].dpg)[0, 1]),
                                              sd_db=float(d[d.bh == st].y.std()), sd_dpg=float(d[d.bh == st].dpg.std()))
           for st in (0, 1)}
    return dict(dpl_on_dpg={c: [float(r["b"][c]), float(r["se"][c])] for c in r["cols"]},
                ratio_with_lagged_fx_control=dict(ratio_block(r1, "d_out", "d_out_bh"),
                                                  sdpg_l=[float(r1["b"]["sdpg_l"]), float(r1["se"]["sdpg_l"])],
                                                  sdpg_l_bh=[float(r1["b"]["sdpg_l_bh"]), float(r1["se"]["sdpg_l_bh"])]),
                variance_by_state=var)


def cycle(a):
    rows = []
    for h0 in range(0, 24, 2):
        d = a[(a.lh >= h0) & (a.lh < h0 + 2)]
        if (d.d_out > 0).sum() < 200: continue
        r = ols(d, "R", ["d_out", "d_in"], fe="month")
        rows.append(dict(h=h0 + 1, k=float(r["b"]["d_out"]), se=float(r["se"]["d_out"]), n=int(r["n"]),
                         n_out=int((d.d_out > 0).sum())))
    c = pd.DataFrame(rows)
    night_b = c[c.h.isin([1, 3, 5, 7])].k.mean(); day_b = c[c.h.isin([11, 13, 15])].k.mean()
    res = dict(blocks=rows, night_over_day_blocks=float(night_b / day_b))
    for nm, msk in [("night_00_08", a.lh < 8), ("day_10_16", (a.lh >= 10) & (a.lh < 16))]:
        r = ols(a[msk], "R", ["d_out", "d_in"], fe="month")
        res[nm] = [float(r["b"]["d_out"]), float(r["se"]["d_out"]), int(r["n"])]
    return res


def stack(A, cols, fe, ycol):
    MK = list(A)
    D = []
    for k in MK:
        d = A[k][[ycol, fe, "day"] + cols].copy()
        for c in cols:
            for k2 in MK:
                d[f"{c}__{k2}"] = d[c] if k2 == k else 0.0
        d["fe2"] = k + "_" + d[fe].astype(str)
        D.append(d.drop(columns=cols))
    D = pd.concat(D)
    return ols(D, ycol, [f"{c}__{k}" for k in MK for c in cols], fe="fe2")


def indep_se(res, f, m1, m2):
    cols = res["cols"]; V = res["V"].copy()
    i1 = [i for i, c in enumerate(cols) if c.endswith("__" + m1)]; i2 = [i for i, c in enumerate(cols) if c.endswith("__" + m2)]
    V[np.ix_(i1, i2)] = 0; V[np.ix_(i2, i1)] = 0
    r2 = dict(res); r2["V"] = V
    return dm(r2, f)[1]


def stacked_diff(A, m1, m2, regs=("d_out", "d_out_bh", "d_in", "d_in_bh"), c0="d_out", c1="d_out_bh"):
    """log ratio of m1 minus log ratio of m2 from the stacked regression."""
    B = {m1: A[m1], m2: A[m2]}
    r = stack(B, list(regs), "ms", "R")
    lr = lambda k: (lambda b: np.log((b[f"{c1}__{k}"] + b[f"{c0}__{k}"]) / b[f"{c0}__{k}"]))
    ok = all(r["b"][f"{c0}__{k}"] > 0 and r["b"][f"{c0}__{k}"] + r["b"][f"{c1}__{k}"] > 0 for k in B)
    out = {k: dict(closed=[float(r["b"][f"{c0}__{k}"]), float(r["se"][f"{c0}__{k}"])],
                   incr=[float(r["b"][f"{c1}__{k}"]), float(r["se"][f"{c1}__{k}"])],
                   ratio=float((r["b"][f"{c0}__{k}"] + r["b"][f"{c1}__{k}"]) / r["b"][f"{c0}__{k}"]),
                   logratio=list(dm(r, lr(k))) if ok else [np.nan, np.nan]) for k in B}
    if ok:
        f = lambda b: lr(m1)(b) - lr(m2)(b)
        v, s = dm(r, f)
        out["diff_logratio"] = dict(est=v, se_stacked=s, se_indep=indep_se(r, f, m1, m2),
                                    z=v / s, label=f"log ratio {m1} - log ratio {m2}")
    else:
        out["diff_logratio"] = dict(est=np.nan, se_stacked=np.nan, se_indep=np.nan, z=np.nan)
    v, s = dm(r, lambda b: b[f"{c1}__{m1}"] - b[f"{c1}__{m2}"])
    out["diff_incr"] = dict(est=v, se_stacked=s)
    out["n"] = {k: int(len(B[k])) for k in B}; out["days_common"] = int(len(set(B[m1].day) & set(B[m2].day)))
    out["ndays_clusters"] = r["ndays"]
    return out


# ------------------------------------------------------------------ diagnostics
SYM = ["d_out", "d_out_bh", "d_in", "d_in_bh"]


def band_stats(a):
    g = a.groupby("ms").agg(l=("l", "first"), u=("u", "first"), med=("x", "median"), sd=("x", "std"))
    return dict(cells=int(len(g)), degenerate_cells_width_le_1bp=int(((g.u - g.l) <= 1).sum()),
                cells_with_median_outside_band=int(((g.med < g.l) | (g.med > g.u)).sum()),
                median_width_over_within_cell_sd=float(((g.u - g.l) / g.sd).median()))


def day_influence(a):
    """First-order change in the pooled log ratio from dropping each UTC day (bands and beta fixed)."""
    Y = a.R.values; X = a[SYM].values
    gm = a.groupby("ms"); Yd = Y - gm.R.transform("mean").values; Xd = X - gm[SYM].transform("mean").values
    XtXi = np.linalg.inv(Xd.T @ Xd); b = XtXi @ Xd.T @ Yd; e = Yd - Xd @ b
    codes, uniq = pd.factorize(a.day); G = np.zeros((len(uniq), 4)); np.add.at(G, codes, Xd * e[:, None])
    dfb = (XtXi @ G.T).T
    lr = lambda bb: np.log((bb[0] + bb[1]) / bb[0]) if bb[0] > 0 and bb[0] + bb[1] > 0 else np.nan
    return pd.Series([lr(b) - lr(b - d) for d in dfb], index=uniq)


def extreme_days(a, n):
    """The n UTC days with the largest premium relative to its month: max |x - month median| / month s.d.
    Chosen from the premium alone, not from the estimates."""
    z = (a.x - a.groupby("month").x.transform("median")).abs() / a.groupby("month").x.transform("std")
    return list(z.groupby(a.day).max().sort_values(ascending=False).index[:n])


def pooled(a):
    r = ols(a, "R", SYM)
    return dict(ratio_block(r, "d_out", "d_out_bh"), n=int(r["n"]))


def audit_market(a):
    I = day_influence(a)
    top = I.abs().idxmax()
    ex3, ex7 = extreme_days(a, 3), extreme_days(a, 7)
    return dict(baseline=pooled(a), bands=band_stats(a),
                most_influential_day=str(top), its_first_order_effect_on_logratio=float(I[top]),
                without_most_influential_day=pooled(a[a.day != top]),
                extreme_days_7=[str(d) for d in ex7],
                drop_3_extreme_days=pooled(a[~a.day.isin(ex3)]), drop_7_extreme_days=pooled(a[~a.day.isin(ex7)]))


# ------------------------------------------------------------------ data quality
def duk_raw(pair, side="BID"):
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "dukascopy", f"{pair}_{side}_*.bi5"))):
        if os.path.getsize(f) == 0: continue
        try:
            raw = lzma.decompress(open(f, "rb").read())
        except Exception:
            continue
        day = dt.datetime.strptime(os.path.basename(f).split("_")[2][:8], "%Y%m%d").replace(tzinfo=dt.timezone.utc)
        a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6)
        vol = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5]
        idx = pd.Timestamp(day) + pd.to_timedelta(a[:, 0].astype("int64"), unit="s")
        c = a[:, 2].astype(float)
        rows.append(pd.DataFrame({"c": c / 1e5, "chg": np.r_[False, c[1:] != c[:-1]], "rng": a[:, 3] != a[:, 4],
                                  "fxv": vol}, index=idx))
    return pd.concat(rows) if rows else pd.DataFrame()


def fx_open_minutes(idx):
    """Weekday minutes outside the 20:30-22:30 UTC rollover window and before the Friday 20:30 UTC close,
    i.e. the minutes the panel can use. Excluding the Friday-evening minutes after the FX market closes keeps
    them from understating quote activity at 23-02 local time."""
    hr = idx.hour + idx.minute / 60
    dow = idx.dayofweek
    return (dow < 5) & ~((hr >= 20.5) & (hr < 22.5)) & ~((dow == 4) & (hr >= 20.5))


def duk_profile(pair, mkt):
    b = duk_raw(pair, "BID")
    b = b[fx_open_minutes(b.index)].copy()
    b["lh"] = local_hour(b.index, mkt)
    out = b.groupby("lh").agg(share_active=("fxv", lambda s: float((s > 0).mean())), share_bid_change=("chg", "mean"),
                              share_range=("rng", "mean"),
                              median_tickvol_active=("fxv", lambda s: float(s[s > 0].median())))
    ask = duk_raw(pair, "ASK")
    if len(ask):
        ask = ask[~ask.index.duplicated()]
        j = b[["c", "fxv", "lh"]].join(ask.c.rename("a"), how="inner")
        j = j[j.fxv > 0]
        j["spr"] = 1e4 * (j.a - j.c) / ((j.a + j.c) / 2)
        out["median_spread_bp"] = j.groupby("lh").spr.median()
    out = out.reset_index()
    return out, b


def main():
    OUT = {}
    # ---------------- Binance EUR/USDT coverage and liquidity
    k = kl_eur().loc[W0:W1]
    months_bin = sorted(k.index.tz_localize(None).to_period("M").astype(str).unique().tolist())
    wk = k[k.index.dayofweek < 5].copy()
    wk["lh"] = local_hour(wk.index, "EUR")
    wk["month"] = wk.index.tz_localize(None).to_period("M").astype(str)
    bq = dict(months_available=months_bin, n_weekday_minutes=int(len(wk)),
              share_minutes_with_trades=float((wk.n > 0).mean()),
              median_minute_volume_usdt=float(wk.qv.median()), mean_minute_volume_usdt=float(wk.qv.mean()),
              median_minute_trades=float(wk.n.median()),
              by_month=wk.groupby("month").agg(share_traded=("n", lambda s: float((s > 0).mean())),
                                               median_qv=("qv", "median"), median_n=("n", "median")).round(4).reset_index().to_dict("records"),
              by_local_hour=wk.groupby("lh").agg(share_traded=("n", lambda s: float((s > 0).mean())),
                                                 median_qv=("qv", "median"), median_n=("n", "median")).round(4).reset_index().to_dict("records"))
    # Turkey for reference (same statistics, weekdays, common window)
    t = binance("USDTTRY").loc[W0:W1]; t = t[t.index.dayofweek < 5]
    bq["TRY_reference"] = dict(share_minutes_with_trades=float((t.n > 0).mean()), median_minute_volume_usdt=float(t.v.median()),
                               median_minute_trades=float(t.n.median()))
    # price discreteness: tick size in bp of the price (smallest positive change of the minute close)
    def tick_bp(c):
        dc = np.abs(np.diff(c.dropna().values)); tk = dc[dc > 1e-12].min()
        return float(tk), float(1e4 * tk / c.median())
    bq["tick_EURUSDT"] = tick_bp(k.c); bq["tick_USDTTRY"] = tick_bp(t.close)
    log("binance done", bq["median_minute_volume_usdt"], bq["median_minute_trades"])

    # ---------------- Dukascopy coverage and activity
    days_req = pd.bdate_range(W0, W1)
    fb = {os.path.basename(f).split("_")[2][:8]: os.path.getsize(f) for f in glob.glob(os.path.join(RAW, "dukascopy", "EURUSD_BID_*.bi5"))}
    fa = {os.path.basename(f).split("_")[2][:8]: os.path.getsize(f) for f in glob.glob(os.path.join(RAW, "dukascopy", "EURUSD_ASK_*.bi5"))}
    dk = days_req.strftime("%Y%m%d")
    dq = dict(weekdays_requested=int(len(dk)), bid_files=int(sum(d in fb for d in dk)),
              bid_nonempty=int(sum(fb.get(d, 0) > 0 for d in dk)), ask_files=int(sum(d in fa for d in dk)),
              ask_nonempty=int(sum(fa.get(d, 0) > 0 for d in dk)),
              bid_days_by_month=pd.Series([d[:6] for d in dk if fb.get(d, 0) > 0]).value_counts().sort_index().to_dict())
    prof_eur, braw = duk_profile("EURUSD", "EUR")
    prof_try, _ = duk_profile("USDTRY", "TRY")
    # scale check: daily median EUR/USD level at a few dates
    lv = braw[braw.fxv > 0].c.resample("D").median().dropna()
    dq["scale_check_daily_median_bid"] = {str(d.date()): float(v) for d, v in lv.iloc[[0, len(lv) // 4, len(lv) // 2, 3 * len(lv) // 4, -1]].items()}
    dq["EUR_by_frankfurt_hour"] = prof_eur.round(4).to_dict("records")
    dq["TRY_by_istanbul_hour_reference"] = prof_try.round(4).to_dict("records")
    dq["TRY_ask_days"] = sorted(os.path.basename(f).split("_")[2][:8] for f in glob.glob(os.path.join(RAW, "dukascopy", "USDTRY_ASK_*.bi5"))
                                if os.path.getsize(f) > 0)
    dq["note"] = ("Activity profiles use weekday minutes outside 20:30-22:30 UTC and before the Friday 20:30 UTC close. "
                  "USD/TRY spreads rest on the ask files listed in TRY_ask_days only.")
    log("dukascopy done", dq["bid_nonempty"], dq["ask_nonempty"])

    # ---------------- baseline EUR, full window
    A_eur = build("EUR", W0, W1)
    est, sym = estimate(A_eur)
    cyc = cycle(A_eur)
    log("EUR baseline", est["pooled"])
    # panel-level quality: share of five-minute intervals in which the FX mid moves, by Frankfurt hour
    pq = A_eur.groupby("lh").agg(n=("y", "size"), share_fx_moves=("dpg", lambda s: float((s.abs() > 1e-9).mean())),
                                 mean_abs_dpg=("dpg", lambda s: float(s.abs().mean())),
                                 mean_abs_dpl=("dpl", lambda s: float(s.abs().mean())),
                                 share_out=("d_out", lambda s: float((s > 0).mean())),
                                 median_vol_5min=("vol", "median")).round(4).reset_index()
    all5 = pd.date_range(W0, pd.Timestamp(W1) + pd.Timedelta("23:55:00"), freq="5min", tz="UTC")
    all5 = all5[all5.dayofweek < 5]
    hr = all5.hour + all5.minute / 60
    all5 = all5[~((hr >= 20.5) & (hr < 22.5))]
    pq["share_of_eligible_intervals_kept"] = (pq.n / pd.Series(local_hour(all5, "EUR")).value_counts().reindex(pq.lh).values).round(4)

    # ---------------- Turkey on the common window, stacked
    A_try = build("TRY", W0, W1)
    est_try, _ = estimate(A_try)
    cyc_try = cycle(A_try)
    diff_try = stacked_diff({"TRY": A_try, "EUR": A_eur}, "TRY", "EUR")
    diff_try_above = stacked_diff({"TRY": A_try, "EUR": A_eur}, "TRY", "EUR",
                                  regs=("d_upo", "d_upo_bh", "d_dno", "d_dno_bh", "d_in", "d_in_bh"), c0="d_upo", c1="d_upo_bh")
    diff_try_below = stacked_diff({"TRY": A_try, "EUR": A_eur}, "TRY", "EUR",
                                  regs=("d_upo", "d_upo_bh", "d_dno", "d_dno_bh", "d_in", "d_in_bh"), c0="d_dno", c1="d_dno_bh")
    log("TRY common", est_try["pooled"], diff_try["diff_logratio"])

    # ---------------- Table emp_mxn window: EUR, TRY, MXN
    Am = {"TRY": build("TRY", WM0, W1), "MXN": build("MXN", WM0, W1), "EUR": build("EUR", WM0, W1)}
    em = {k: estimate(v)[0] for k, v in Am.items()}
    cyc_m = {k: cycle(v) for k, v in Am.items()}
    dm_try = stacked_diff(Am, "TRY", "EUR")
    dm_mxn = stacked_diff(Am, "EUR", "MXN")
    log("emp_mxn window", {k: (em[k]["pooled"]["ratio"], em[k]["pooled"]["logratio"]) for k in em})

    # ---------------- robustness
    rob = {}
    A_h = build("EUR", W0, W1, drop_days=pd.to_datetime(TARGET_CLOSED).date)
    rob["drop_TARGET_closing_days"] = estimate(A_h)[0]["pooled"]
    rob["bid_only_mid"] = estimate(build("EUR", W0, W1, bid_only=True))[0]["pooled"]
    thr = A_eur.loc[A_eur.bh == 1, "fxv"].quantile(0.25)
    act = A_eur[A_eur.fxv >= thr].copy()
    r_act = ols(act, "R", ["d_out", "d_out_bh", "d_in", "d_in_bh"])
    rob["active_fx_quotes"] = dict(ratio_block(r_act, "d_out", "d_out_bh"), threshold=float(thr), n=int(r_act["n"]),
                                   share_closed_kept=float((act.bh == 0).sum() / (A_eur.bh == 0).sum()),
                                   share_open_kept=float((act.bh == 1).sum() / (A_eur.bh == 1).sum()))
    # price legs: does the benchmark revert? (dpg and dpl on the asym regressors, as empirics_extra)
    dd = A_eur.dropna(subset=["dpg", "dpl"])
    for leg in ["dpg", "dpl"]:
        r = ols(dd, leg, ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
        rob[f"leg_{leg}"] = {c: [float(r["b"][c]), float(r["se"][c])] for c in r["cols"]}
    explo = dict(EUR=tracking(A_eur), TRY_common_window=tracking(A_try))
    log("robustness done", rob["drop_TARGET_closing_days"]["ratio"], rob["active_fx_quotes"]["ratio"])

    # ---------------- diagnostics
    aud = {}
    # (a) decompose r1: TARGET days dropped with the baseline bands, and the cells whose bands move
    aud["drop_TARGET_days_baseline_bands"] = pooled(A_eur[~A_eur.day.isin(set(pd.to_datetime(TARGET_CLOSED).date))])
    bA = A_eur.groupby("ms")[["l", "u"]].first(); bH = A_h.groupby("ms")[["l", "u"]].first()
    cb = bA.join(bH, rsuffix="_reprofiled"); cb = cb[(cb.l != cb.l_reprofiled) | (cb.u != cb.u_reprofiled)]
    aud["cells_whose_band_moves_when_TARGET_days_dropped"] = cb.reset_index().to_dict("records")
    B = A_eur.drop(columns=["l", "u"]).join(bH, on="ms")
    mB = (B.l + B.u) / 2
    B["s"] = np.where(B.x > B.u, 1.0, np.where(B.x < B.l, -1.0, np.sign(B.x - mB)))
    B["d_out"] = np.maximum(B.x - B.u, 0) + np.maximum(B.l - B.x, 0); B["d_in"] = np.where(B.d_out == 0, np.abs(B.x - mB), 0.0)
    B["up"] = -np.maximum(B.x - B.u, 0); B["dn"] = np.maximum(B.l - B.x, 0)
    B["d_upo"] = np.maximum(B.x - B.u, 0); B["d_dno"] = np.maximum(B.l - B.x, 0)
    prep(B)
    aud["all_days_with_bands_reprofiled_without_TARGET_days"] = pooled(B)
    # (b) single-day dependence, extreme-day exclusion and band shape, every market and window
    aud["by_market"] = {"EUR_2024_2026": audit_market(A_eur), "TRY_2024_2026": audit_market(A_try),
                        "EUR_emp_mxn_window": audit_market(Am["EUR"]), "TRY_emp_mxn_window": audit_market(Am["TRY"]),
                        "MXN_emp_mxn_window": audit_market(Am["MXN"])}
    # (c) subperiods with the baseline bands
    aud["subperiods"] = {f"{k}_{y0}_{y1}": pooled(a[(a.month >= y0) & (a.month <= y1)])
                         for k, a in [("EUR", A_eur), ("TRY", A_try)]
                         for y0, y1 in [("2024-01", "2024-12"), ("2025-01", "2026-08")]}
    # (d) stacked differences after dropping each market's 7 most extreme days (outcome-blind)
    X7 = {k: v[~v.day.isin(extreme_days(v, 7))] for k, v in Am.items()}
    aud["emp_mxn_window_drop_7_extreme_days"] = dict(
        TRY_minus_EUR=stacked_diff(X7, "TRY", "EUR")["diff_logratio"], EUR_minus_MXN=stacked_diff(X7, "EUR", "MXN")["diff_logratio"],
        TRY_minus_MXN=stacked_diff(X7, "TRY", "MXN")["diff_logratio"])
    X7c = {"TRY": A_try[~A_try.day.isin(extreme_days(A_try, 7))], "EUR": A_eur[~A_eur.day.isin(extreme_days(A_eur, 7))]}
    aud["common_window_drop_7_extreme_days_TRY_minus_EUR"] = stacked_diff(X7c, "TRY", "EUR")["diff_logratio"]
    log("audit done", aud["by_market"]["EUR_2024_2026"]["without_most_influential_day"]["ratio"])

    # ---------------- assemble
    P = est["pooled"]
    OUT = dict(
        market="EUR: Binance EUR/USDT vs Dukascopy EUR/USD; b = 1e4*(log EURUSD_mid - log EURUSDT_vwap), "
               "OFI = USDT bought with euros (taker sells of EUR/USDT), business hours 09:00-17:00 Europe/Berlin",
        window=[W0, W1], months=est["months"], n_intervals=est["n"], n_days=est["days"],
        median_volume_usdt=bq["median_minute_volume_usdt"], median_trades_per_minute=bq["median_minute_trades"],
        beta=est["beta"],
        kappa_closed=P["closed"][0], kappa_closed_se=P["closed"][1],
        increment=P["incr"][0], increment_se=P["incr"][1],
        kappa_open=P["open"][0], kappa_open_se=P["open"][1],
        ratio=P["ratio"], logratio=P["logratio"][0], logratio_se=P["logratio"][1],
        d_in=est["d_in"], d_in_bh=est["d_in_bh"],
        above_band=dict(R_based=est["above_R"], y_based=est["above_y"], share_intervals=est["share_above"]),
        below_band=dict(R_based=est["below_R"], y_based=est["below_y"], share_intervals=est["share_below"]),
        asym=est["asym"],
        premium=dict(mean=est["mean_prem"], sd=est["sd_prem"], sd_within_month_median=est["sd_prem_within_month"],
                     share_out=est["share_out"], width_open=est["width_open"], width_closed=est["width_closed"],
                     vol_per_min=est["vol_per_min"]),
        cycle=dict(EUR=cyc, TRY_common_window=cyc_try),
        diff_vs_TRY=dict(window=[W0, W1], TRY=est_try, pooled=diff_try, above_R=diff_try_above, below_R=diff_try_below),
        emp_mxn_window=dict(window=[WM0, W1], estimates=em, cycle=cyc_m, diff_TRY_minus_EUR=dm_try, diff_EUR_minus_MXN=dm_mxn),
        robustness=rob, exploratory=explo, audit=aud,
        quality=dict(binance=bq, dukascopy=dq, panel_by_frankfurt_hour=pq.to_dict("records")),
        runtime_s=float(time.time() - T0))
    json.dump(OUT, open(os.path.join(TAB, "rob_xc_eur.json"), "w"), indent=1, default=float)
    write_tex(OUT)
    figure(OUT, bq, prof_eur, prof_try, pq)
    log("written")
    return OUT


def star(c, s):
    if not np.isfinite(c) or not np.isfinite(s) or s == 0: return ""
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def write_tex(O):
    """Five columns: Turkey and euro on the common window (Jan 2024-Aug 2026); Turkey, euro and Mexico on
    the Table emp_mxn window (Nov 2024-Aug 2026). Differences use the stacked (joint) s.e."""
    T = O["diff_vs_TRY"]["TRY"]; E = {"pooled": {"closed": [O["kappa_closed"], O["kappa_closed_se"]],
                                                 "incr": [O["increment"], O["increment_se"]], "ratio": O["ratio"],
                                                 "logratio": [O["logratio"], O["logratio_se"]]},
                                      "above_R": O["above_band"]["R_based"], "below_R": O["below_band"]["R_based"],
                                      "sd_prem": O["premium"]["sd"], "width_open": O["premium"]["width_open"],
                                      "width_closed": O["premium"]["width_closed"], "vol_per_min": O["premium"]["vol_per_min"],
                                      "n": O["n_intervals"]}
    em = O["emp_mxn_window"]["estimates"]
    cols = [T, E, em["TRY"], em["EUR"], em["MXN"]]

    def f(v, nd=4):
        return f"${v[0]:.{nd}f}{star(v[0], v[1])}$" if np.isfinite(v[0]) else "--"

    def s(v, nd=4):
        return f"$({v[1]:.{nd}f})$" if np.isfinite(v[1]) else ""
    L = ["\\begin{tabular}{lccccc}", "\\toprule",
         " & \\multicolumn{2}{c}{Jan 2024 -- Aug 2026} & \\multicolumn{3}{c}{Nov 2024 -- Aug 2026} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}", " & Turkey & Euro & Turkey & Euro & Mexico \\\\", "\\midrule"]
    rows = [("Speed outside band, banks closed", lambda c: c["pooled"]["closed"]),
            ("$\\quad\\times$ business hours", lambda c: c["pooled"]["incr"]),
            ("Above band: banks closed", lambda c: c["above_R"]["closed"]),
            ("$\\quad\\times$ business hours", lambda c: c["above_R"]["incr"]),
            ("Below band: banks closed", lambda c: c["below_R"]["closed"]),
            ("$\\quad\\times$ business hours", lambda c: c["below_R"]["incr"])]
    for lab, g in rows:
        L.append(lab + " & " + " & ".join(f(g(c)) for c in cols) + " \\\\")
        L.append(" & " + " & ".join(s(g(c)) for c in cols) + " \\\\")
    L.append("\\midrule")
    L.append("Open/closed ratio, pooled sides & " + " & ".join(f"{c['pooled']['ratio']:.1f}" for c in cols) + " \\\\")
    L.append("$\\log$ ratio & " + " & ".join(f(c["pooled"]["logratio"], 2) for c in cols) + " \\\\")
    L.append(" & " + " & ".join(s(c["pooled"]["logratio"], 2) for c in cols) + " \\\\")
    d1 = O["diff_vs_TRY"]["pooled"]["diff_logratio"]; d2 = O["emp_mxn_window"]["diff_EUR_minus_MXN"]["diff_logratio"]
    d3 = O["emp_mxn_window"]["diff_TRY_minus_EUR"]["diff_logratio"]
    fmt = lambda d: (f"${d['est']:.2f}{star(d['est'], d['se_stacked'])}$", f"$({d['se_stacked']:.2f})$") if np.isfinite(d["est"]) else ("--", "")
    L.append(f"$\\log$ ratio, Turkey minus euro & \\multicolumn{{2}}{{c}}{{{fmt(d1)[0]}}} & \\multicolumn{{3}}{{c}}{{{fmt(d3)[0]}}} \\\\")
    L.append(f" & \\multicolumn{{2}}{{c}}{{{fmt(d1)[1]}}} & \\multicolumn{{3}}{{c}}{{{fmt(d3)[1]}}} \\\\")
    L.append(f"$\\log$ ratio, euro minus Mexico & & & & \\multicolumn{{2}}{{c}}{{{fmt(d2)[0]}}} \\\\")
    L.append(f" & & & & \\multicolumn{{2}}{{c}}{{{fmt(d2)[1]}}} \\\\")
    L.append("\\midrule")
    L.append("SD of premium (bp) & " + " & ".join(f"{c['sd_prem']:.1f}" for c in cols) + " \\\\")
    L.append("Mean band width, open / closed (bp) & " + " & ".join(f"{c['width_open']:.0f} / {c['width_closed']:.0f}" for c in cols) + " \\\\")
    L.append("USDT volume per minute (thousand) & " + " & ".join(f"{c['vol_per_min']/1e3:.1f}" for c in cols) + " \\\\")
    L.append("$N$ & " + " & ".join(f"{c['n']:,}" for c in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "rob_xc_eur.tex"), "w").write("\n".join(L) + "\n")


def figure(O, bq, prof_eur, prof_try, pq):
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.2))
    # (a) daily cycle
    a = ax[0, 0]
    for key, col, mk, lab, dx in [("EUR", C2, "s", "Euro (Frankfurt time)", 0.3), ("TRY_common_window", C1, "o", "Turkey (Istanbul time)", 0.0)]:
        c = pd.DataFrame(O["cycle"][key]["blocks"])
        a.errorbar(c.h + dx, c.k, yerr=1.96 * c.se, fmt=mk + "-", ms=3.5, color=col, ecolor=col, lw=0.9, alpha=0.9, label=lab)
    a.axvspan(9, 17, color=C3, alpha=0.12); a.axhline(0, color="k", lw=0.4)
    a.set_xlabel("local hour"); a.set_ylabel(r"speed outside the band, $\hat\kappa$")
    a.set_xticks(range(0, 25, 4)); a.legend(frameon=False, fontsize=8, loc="upper left")
    a.set_title("(a) Daily cycle, Jan 2024 -- Aug 2026", fontsize=9, loc="left")
    # (b) open/closed ratios with 95% log-delta intervals
    a = ax[0, 1]
    em = O["emp_mxn_window"]["estimates"]; T = O["diff_vs_TRY"]["TRY"]["pooled"]
    items = [("Turkey\n24-26", T, C1), ("Euro\n24-26", dict(ratio=O["ratio"], logratio=[O["logratio"], O["logratio_se"]]), C2),
             ("Turkey\n11/24-26", em["TRY"]["pooled"], C1), ("Euro\n11/24-26", em["EUR"]["pooled"], C2),
             ("Mexico\n11/24-26", em["MXN"]["pooled"], C4)]
    for i, (lab, d, col) in enumerate(items):
        lr, se = d["logratio"]
        if np.isfinite(lr):
            a.errorbar(i, np.exp(lr), yerr=[[np.exp(lr) - np.exp(lr - 1.96 * se)], [np.exp(lr + 1.96 * se) - np.exp(lr)]],
                       fmt="o", color=col, ms=5, capsize=3)
        else:
            a.plot(i, d["ratio"], "x", color=col)
    a.axhline(1, color="k", lw=0.4); a.set_yscale("log")
    a.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16]); a.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8", "16"])
    a.minorticks_off()
    a.set_xticks(range(len(items))); a.set_xticklabels([x[0] for x in items], fontsize=8)
    a.set_ylabel("open / closed speed ratio (log scale)")
    a.set_title("(b) Open/closed ratio, 95% interval", fontsize=9, loc="left")
    # (c) Binance liquidity by Frankfurt hour
    a = ax[1, 0]
    h = pd.DataFrame(bq["by_local_hour"])
    a.bar(h.lh, h.median_qv / 1e3, color=C2, alpha=0.7, width=0.8)
    a.set_xlabel("Frankfurt hour"); a.set_ylabel("median minute volume (thousand USDT)")
    a2 = a.twinx(); a2.plot(h.lh, h.median_n, "k-", lw=0.9); a2.set_ylabel("median trades per minute")
    a2.spines["top"].set_visible(False)
    a.axvspan(8.5, 16.5, color=C3, alpha=0.12)
    a.set_title("(c) Binance EUR/USDT activity", fontsize=9, loc="left")
    # (d) interbank quote activity by local hour
    a = ax[1, 1]
    a.plot(prof_eur.lh, prof_eur.share_bid_change, "s-", color=C2, ms=3, lw=0.9, label="EUR/USD, Frankfurt hour")
    a.plot(prof_try.lh, prof_try.share_bid_change, "o-", color=C1, ms=3, lw=0.9, label="USD/TRY, Istanbul hour")
    a.set_ylim(0, 1.02); a.axvspan(9, 17, color=C3, alpha=0.12)
    a.set_xlabel("local hour"); a.set_ylabel("share of minutes with a new bid")
    a.set_xticks(range(0, 25, 4)); a.legend(frameon=False, fontsize=8, loc="lower center")
    a.set_title("(d) Dukascopy quote activity", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "rob_xc_eur.pdf")); plt.close(fig)


if __name__ == "__main__":
    O = main()
    short = {k: O[k] for k in ["months", "n_intervals", "median_volume_usdt", "kappa_closed", "kappa_closed_se", "increment",
                               "increment_se", "ratio", "logratio", "logratio_se"]}
    print(json.dumps(short, indent=1, default=float))
