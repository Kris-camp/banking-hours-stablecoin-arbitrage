"""Cross-country contrast: Poland (USDT/PLN) against Turkey, in the design of Table emp_mxn.

DESIGN
======
Data.  Binance USDT/PLN one-minute klines (data.binance.vision; files exist for 2024-01 to
2026-06 and for the first twelve hours of 2026-07; there is no 2026-08 file) against Dukascopy
USD/PLN one-minute BID and ASK candles (prices / 1e5, 0-based months in the URL), downloaded
sequentially for the weekdays 2024-01-01 -- 2026-07-01.  Benchmark = (bid+ask)/2 if ASK files cover
at least 95% of the BID days (build_panel.load_fx, unchanged); otherwise, so that the benchmark is
not mixed across days, the bid plus half the median relative spread on every day, which is how the
USD/TRY and USD/MXN benchmarks of Table emp_mxn are built (BID files only).  In the authors' data
the USD/PLN ASK files cover one day per month, so the second construction applies.

Construction: identical to build() in code/empirics_mxn.py.  Binance minute VWAP vs the Dukascopy
mid forward-filled at most two minutes; weekdays (UTC) only; the 20:30--22:30 UTC rollover window
dropped; |b| <= 1000 bp; five-minute means with at least two valid minutes; OFI = 2 TBV - V summed
over the interval and standardized within month; consecutive five-minute intervals only.  Local
business hours = 09:00--17:00 Europe/Warsaw, with daylight saving time handled by zoneinfo
(CET/CEST, so the open window is 08:00--16:00 UTC in winter and 07:00--15:00 UTC in summer).
Bands profiled by month x local banking state (cells with fewer than 300 intervals dropped, as in
empirics_mxn.py), month-by-state fixed effects, standard errors clustered by UTC day with the
finite-sample correction.

Estimates (Poland, full PLN sample):
 (1) Pooled sides, as Table emp_mxn: beta from y ~ q + up + dn; R = -s (dy - beta q);
     R ~ d_out + d_out*BH + d_in + d_in*BH.  kappa_closed = coef on d_out, increment = coef on
     d_out*BH, ratio = (kappa_closed + increment) / kappa_closed, log ratio with delta-method s.e.
     (only when both the closed and the open speed are positive).
 (2) By side, two versions: (2a) y ~ q + up + dn + (q, up, dn) x BH, the "above band only" row of
     Table emp_mxn, and its below-band analogue; (2b) R-based, R ~ d_above + d_below + d_in, each
     interacted with BH (as rob_mexico_utc.py r3).  Closed speed, increment, ratio, log ratio.
 (3) Daily cycle: R ~ d_out + d_in with month fixed effects within each two-hour block of Warsaw
     time (blocks with fewer than 200 outside-band intervals skipped), as Figure emp_mxn_cycle.
     Night = blocks 00--08 local, day = blocks 10--16 local; night/day = mean of the night block
     speeds over the mean of the day block speeds (as empirics_mxn.py).  In addition, pooled night
     (00--08) and day (10--16) speeds, each from one regression with month fixed effects, with s.e.
 (4) Versus Turkey on the common sample: Turkey built exactly as in empirics_mxn.py (Istanbul
     UTC+3), restricted to the months in the Polish sample; beta re-estimated on that subsample.
     Stacked two-market regression (every regressor interacted with market, market x month x state
     fixed effects, one UTC-day cluster across both markets).  Statistic:
         D = log ratio(TRY) - log ratio(PLN)
     with the stacked delta-method s.e. and the independent-samples s.e.  Also by side (2a, 2b).
     Sanity check: Turkey on 2024-11 -- 2026-08 must reproduce tables/emp_mxn.json (ratio 4.16,
     log ratio 1.43, s.e. 0.44).

Expectations.  Poland's large-value rails (the NBP's RTGS system SORBNET3 and the
KIR Elixir clearing sessions) settle on business days only; an instant system (Express Elixir)
runs around the clock with a per-transfer cap set by the operator and banks.  If closed
large-value rails slow arbitrage, the Polish ratio should exceed one (log ratio > 0).  Because the
Polish instant rail is larger relative to typical USDT/PLN trade sizes than Turkey's FAST limit is
to USDT/TRY trades, the rails hypothesis does not by itself predict a Turkish-size slowdown; no sign
is predicted for D.  Unlike Mexico, Polish and Turkish business hours overlap in UTC (06:00--
14:00 vs 07:00/08:00--15:00/16:00), so the global trading session is close to common to the two.

Data quality:
 (q1) Binance: share of weekday minutes with at least one trade, median and mean one-minute USDT
      volume and trade count, by month and by Warsaw hour; price tick in bp.
 (q2) Dukascopy: by Warsaw hour, the share of minutes with positive tick volume, the share of
      minutes whose bid close differs from the previous minute's (staleness), median relative
      bid-ask spread; the same for USD/TRY by Istanbul hour over the same months.
 (q3) Share of potential five-minute intervals (weekdays, outside the rollover window) that enter
      the regression sample, by Warsaw hour.
 (q4) Thin-trading checks of (1), baseline bands and beta: (c1) at least 10 trades in both t-1 and
      t; (c2) USDT volume of both t-1 and t above the median of its month x state cell; (c3)
      errors-in-variables: side and distances from b_{t-2} instead of b_{t-1} (three consecutive
      intervals), so iid noise in the five-minute premium cannot generate reversion.
 The JSON classifies the Polish estimates as "informative" only if (i) the closed and the open
 speed in (1) are both more than two s.e. above zero, and (ii) under (c3) the closed speed stays
 positive and the log ratio keeps its sign; otherwise as uninformative about the rails hypothesis.

Inputs: data_raw/binance (USDTPLN, USDTTRY), data_raw/dukascopy (USDPLN bid and ask, USDTRY), tables/emp_mxn.json.
Outputs: tables/rob_xc_pln.json, figures/rob_xc_pln.pdf.

AUDIT NOTE: WIND-DOWN MONTH AND SENSITIVITIES
=============================================
1. Wind-down month.  USD/PLN trading on Binance ended 2026-07-01 11:52 UTC.  In the last two weeks
   of June 2026 the premium left its usual range (daily median +12 to +55 bp in 1-16 June, -19 to
   -73 bp in 17-26 June, -312 and -364 bp on 29-30 June, minimum -711 bp).  The 5%-trimmed band grid
   for the June 2026 cells therefore starts at -348 / -283 bp, and the profile puts both band edges
   there (l = u = -348 bp closed, -283 bp open; cell medians +18 / +16 bp).  As a result 100% of
   June 2026 intervals are classed "outside the band" with a median distance of 319 bp (4.6 bp in
   the other months).  The construction fails for that cell, and the delisting collapse is not an
   arbitrage regime.  The headline estimates therefore drop the wind-down month 2026-06 (EXIT_MONTH).
   Bands, q and the other months' samples are unchanged (all are by month); beta is re-estimated.
   The full-sample results including 2026-06 are also reported, and the json reports cutoff variants
   that end on 16 and 26 June (June bands re-profiled on the partial month).
2. Sensitivities: trimming on the regressor only (|x - cell median| > 300/200/150 bp), for PLN and
   TRY; dropping Polish weekday public holidays.
"""
import os, glob, json, lzma, time, datetime as dt
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
TZ = {"PLN": "Europe/Warsaw", "TRY": "Europe/Istanbul"}
WIN = {"PLN": ("2024-01-01", "2026-07-01 23:59"), "TRY_pub": ("2024-11-01", "2026-08-31"),
       "TRY": ("2024-01-01", "2026-07-01 23:59")}


def log(*s):
    print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)


def local_hour(idx, local):
    if local == "TRY":   # as empirics_mxn.py: Istanbul is UTC+3 all year
        return np.asarray((idx.hour + 3) % 24)
    return np.asarray(idx.tz_convert(TZ[local]).hour)


# ------------------------------------------------------------------ construction (as empirics_mxn.py)
def profile(d, trim=0.05):
    y, x, q = d.y.values, d.x.values, d.q.values
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + 1.0, 1.0)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]


RAWM = {}


def load_fx_cov(pair):
    """build_panel.load_fx, but the ask is used only if ASK files cover at least 95% of the BID days;
    otherwise the benchmark is the bid plus half the median relative spread on every day (BID-only,
    as for USD/TRY and USD/MXN in Table emp_mxn), so that it is not mixed across days."""
    nb = sum(os.path.getsize(f) > 0 for f in glob.glob(os.path.join(RAW, "dukascopy", f"{pair}_BID_*.bi5")))
    na = sum(os.path.getsize(f) > 0 for f in glob.glob(os.path.join(RAW, "dukascopy", f"{pair}_ASK_*.bi5")))
    FXSRC[pair] = dict(bid_days=int(nb), ask_days=int(na), mid="bid+ask" if na >= 0.95 * nb else "bid+half median spread")
    fx = load_fx(pair)
    if na >= 0.95 * nb or na == 0:
        return fx
    # mixed coverage: recompute the mid from the bid alone with the median spread of the ASK days
    half = 0.5 * float(fx.fx_spread.median())
    bid = duka_raw(pair, "BID", "00000000", "99999999")
    bid = bid[~bid.index.duplicated()]
    bid = bid[bid.fxv > 0]
    FXSRC[pair]["median_rel_spread"] = 2 * half
    return pd.DataFrame({"fx_mid": bid.close * (1 + half), "fx_spread": np.nan, "fxv": bid.fxv})


FXSRC = {}


def raw_minutes(local):
    if local not in RAWM:
        u = load_binance(f"USDT{local}")
        fx = load_fx_cov(f"USD{local}") if local == "PLN" else load_fx(f"USD{local}")
        p = pd.DataFrame({"pl": u.vwap, "ofi": 2 * u.tbv - u.v, "vol": u.v, "ntr": u.n}).join(
            fx.reindex(u.index, method="ffill", limit=2))
        RAWM[local] = p
    return RAWM[local]


def geom(a, xcol="x"):
    x = a[xcol]
    m = (a.l + a.u) / 2
    a["s"] = np.where(x > a.u, 1.0, np.where(x < a.l, -1.0, np.sign(x - m)))
    a["d_out"] = np.maximum(x - a.u, 0) + np.maximum(a.l - x, 0)
    a["d_in"] = np.where(a.d_out == 0, np.abs(x - m), 0.0)
    a["up"] = -np.maximum(x - a.u, 0); a["dn"] = np.maximum(a.l - x, 0)
    a["d_upo"] = np.maximum(x - a.u, 0); a["d_dno"] = np.maximum(a.l - x, 0)
    return a


def build(local, w0, w1):
    p = raw_minutes(local).loc[w0:w1].copy()
    p["b"] = 1e4 * np.log(p.pl / p.fx_mid)
    hr = p.index.hour + p.index.minute / 60
    p.loc[(hr >= 20.5) & (hr < 22.5), "b"] = np.nan
    p.loc[p.index.dayofweek >= 5, "b"] = np.nan
    p.loc[p.b.abs() > 1000, "b"] = np.nan
    g = p.resample("5min")
    a = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum(), "vol": g.vol.sum(), "ntr": g.ntr.sum(),
                      "pl": np.log(g.pl.mean()), "pg": np.log(g.fx_mid.mean())})
    a = a[a.nb >= 2]
    a["month"] = a.index.tz_localize(None).to_period("M").astype(str)
    a["q"] = a.ofi / a.ofi.groupby(a.month).transform("std")
    a["dpl"] = 1e4 * a.pl.diff(); a["dpg"] = 1e4 * a.pg.diff()
    dt_ = a.index.to_series().diff()
    consec = dt_ == pd.Timedelta("5min")
    a["c2ok"] = (consec & (dt_.shift(1) == pd.Timedelta("5min"))).values
    a["y"] = a.b.diff(); a["x"] = a.b.shift(1); a["x2"] = a.b.shift(2)
    a["ntr_l"] = a.ntr.shift(1); a["vol_l"] = a.vol.shift(1)
    a = a[consec & a.y.notna() & a.x.notna() & a.q.notna()].copy()
    a["lh"] = local_hour(a.index, local)
    a["bh"] = ((a.lh >= 9) & (a.lh < 17)).astype(float)
    a["state"] = np.where(a.bh == 1, "open", "closed")
    a["ms"] = a.month + "_" + a.state
    for key, d in a.groupby("ms"):
        if len(d) < 300: continue
        l, u_ = profile(d)
        a.loc[d.index, "l"], a.loc[d.index, "u"] = l, u_
    a = a[a.l.notna()].copy()
    geom(a)
    a["day"] = a.index.date
    a["mkt"] = local
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


def prep(a):
    """beta from y ~ q + up + dn (month x state FE) on this sample, R, and BH interactions."""
    beta = ols(a, "y", ["q", "up", "dn"])["b"]["q"]
    a["R"] = -a.s * (a.y - beta * a.q)
    for c in ["d_out", "d_in", "up", "dn", "q", "d_upo", "d_dno"]:
        a[c + "_bh"] = a[c] * a.bh
    return float(beta)


def ratio_block(r, c0, c1):
    k0, k1 = r["b"][c0], r["b"][c1]
    ok = k0 > 0 and k0 + k1 > 0
    return dict(closed=[float(k0), float(r["se"][c0])], incr=[float(k1), float(r["se"][c1])],
                open=list(dm(r, lambda b: b[c0] + b[c1])),
                ratio=list(dm(r, lambda b: (b[c0] + b[c1]) / b[c0])),
                log_ratio=list(dm(r, lambda b: np.log((b[c0] + b[c1]) / b[c0]))) if ok else [np.nan, np.nan])


def pooled(a, ycol="R"):
    r = ols(a, ycol, ["d_out", "d_out_bh", "d_in", "d_in_bh"])
    o = ratio_block(r, "d_out", "d_out_bh")
    o["d_in"] = [float(r["b"]["d_in"]), float(r["se"]["d_in"])]
    o["d_in_bh"] = [float(r["b"]["d_in_bh"]), float(r["se"]["d_in_bh"])]
    o["n"] = int(r["n"]); o["days"] = r["ndays"]
    return o


def sides(a):
    ry = ols(a, "y", ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
    rR = ols(a, "R", ["d_upo", "d_upo_bh", "d_dno", "d_dno_bh", "d_in", "d_in_bh"])
    out = {}
    for nm, cy, cR in [("above", "up", "d_upo"), ("below", "dn", "d_dno")]:
        out[nm] = dict(y_spec=ratio_block(ry, cy, cy + "_bh"), R_spec=ratio_block(rR, cR, cR + "_bh"),
                       n_outside=int((a[cR] > 0).sum()), n_outside_open=int(((a[cR] > 0) & (a.bh == 1)).sum()))
    out["beta_q_closed"] = [float(ry["b"]["q"]), float(ry["se"]["q"])]
    out["beta_q_bh"] = [float(ry["b"]["q_bh"]), float(ry["se"]["q_bh"])]
    return out


def cycle(a):
    rows = []
    for h0 in range(0, 24, 2):
        d = a[(a.lh >= h0) & (a.lh < h0 + 2)]
        if (d.d_out > 0).sum() < 200: continue
        r = ols(d, "R", ["d_out", "d_in"], fe="month")
        rows.append(dict(h=h0 + 1, k=float(r["b"]["d_out"]), se=float(r["se"]["d_out"]), n=int(r["n"]),
                         n_out=int((d.d_out > 0).sum())))
    c = pd.DataFrame(rows)
    night = c[c.h.isin([1, 3, 5, 7])].k.mean(); day = c[c.h.isin([11, 13, 15])].k.mean()
    out = dict(blocks=rows, night_over_day=float(night / day), night_mean_blocks=float(night), day_mean_blocks=float(day))
    for nm, lo, hi in [("night_pooled_00_08", 0, 8), ("day_pooled_10_16", 10, 16)]:
        d = a[(a.lh >= lo) & (a.lh < hi)]
        r = ols(d, "R", ["d_out", "d_in"], fe="month")
        out[nm] = [float(r["b"]["d_out"]), float(r["se"]["d_out"]), int(r["n"])]
    return out


def stack(A, cols, fe, ycol, mk):
    D = []
    for k in mk:
        d = A[k][[ycol, fe, "day"] + cols].copy()
        for c in cols:
            for k2 in mk:
                d[f"{c}__{k2}"] = d[c] if k2 == k else 0.0
        d["fe2"] = k + "_" + d[fe].astype(str)
        D.append(d.drop(columns=cols))
    D = pd.concat(D)
    cc = [f"{c}__{k}" for k in mk for c in cols]
    return ols(D, ycol, cc, fe="fe2")


def indep_se(res, fa, fb, ka, kb):
    cols = res["cols"]; V = res["V"].copy()
    ia = [i for i, c in enumerate(cols) if c.endswith("__" + ka)]; ib = [i for i, c in enumerate(cols) if c.endswith("__" + kb)]
    V[np.ix_(ia, ib)] = 0; V[np.ix_(ib, ia)] = 0
    r2 = dict(res); r2["V"] = V
    return dm(r2, lambda b: fa(b) - fb(b))[1]


def diff_vs(A, cols, ycol, pairs, mk=("TRY", "PLN")):
    """stacked regression; for each (closed, incr) pair the TRY-minus-PLN log-ratio and increment contrasts."""
    r = stack(A, list(cols), "ms", ycol, list(mk))
    out = {}
    for c0, c1 in pairs:
        o = {}
        f = {}
        for k in mk:
            a0, a1 = f"{c0}__{k}", f"{c1}__{k}"
            f[k] = (lambda a0, a1: (lambda b: np.log((b[a0] + b[a1]) / b[a0])))(a0, a1)
            o[k] = ratio_block(r, a0, a1)
        ka, kb = mk
        if all(np.isfinite(o[k]["log_ratio"][0]) for k in mk):
            v, s = dm(r, lambda b: f[ka](b) - f[kb](b))
            o["diff_log_ratio"] = dict(est=v, se_stacked=s, se_indep=indep_se(r, f[ka], f[kb], ka, kb))
        else:
            o["diff_log_ratio"] = dict(est=np.nan, se_stacked=np.nan, se_indep=np.nan)
        v, s = dm(r, lambda b: b[f"{c1}__{ka}"] - b[f"{c1}__{kb}"])
        o["diff_incr"] = dict(est=v, se_stacked=s)
        out[c0] = o
    out["n"] = {k: int((A[k][ycol].notna()).sum()) for k in mk}
    out["ndays_union"] = int(r["ndays"])
    return out


# ------------------------------------------------------------------ data-quality helpers
def duka_raw(pair, side, d0, d1):
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "dukascopy", f"{pair}_{side}_*.bi5"))):
        ds = os.path.basename(f).split("_")[2][:8]
        if not (d0 <= ds <= d1) or os.path.getsize(f) == 0: continue
        try:
            raw = lzma.decompress(open(f, "rb").read())
        except Exception:
            continue
        day = pd.Timestamp(dt.datetime.strptime(ds, "%Y%m%d"), tz="UTC")
        a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6)
        v = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5]
        idx = day + pd.to_timedelta(a[:, 0].astype("int64"), unit="s")
        rows.append(pd.DataFrame({"close": a[:, 2] / 1e5, "fxv": v}, index=idx))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["close", "fxv"])


def fx_activity(pair, local, d0, d1):
    bid = duka_raw(pair, "BID", d0, d1)
    bid = bid[~bid.index.duplicated()]
    bid = bid[bid.index.dayofweek < 5]
    hr = bid.index.hour + bid.index.minute / 60
    bid = bid[~((hr >= 20.5) & (hr < 22.5))]
    chg = bid.close.diff().ne(0) & (bid.index.to_series().diff() == pd.Timedelta("1min")).values
    df = pd.DataFrame({"lh": local_hour(bid.index, local), "active": (bid.fxv > 0).values, "chg": chg.values})
    ask = duka_raw(pair, "ASK", d0, d1)
    out = dict(n_days_bid=int(pd.Index(bid.index.date).nunique()), n_minutes=int(len(bid)),
               share_active=float(df.active.mean()), share_quote_change=float(df.chg.mean()))
    if len(ask):
        ask = ask[~ask.index.duplicated()]
        j = pd.DataFrame({"b": bid.close, "a": ask.close.reindex(bid.index), "v": bid.fxv}).dropna()
        j = j[j.v > 0]
        j["spr"] = 1e4 * (j.a - j.b) / ((j.a + j.b) / 2)
        df["spr"] = j.spr.reindex(bid.index).values
        out["n_days_ask"] = int(pd.Index(ask.index.date).nunique())
        out["median_spread_bp"] = float(j.spr.median())
    byh = df.groupby("lh").agg(share_active=("active", "mean"), share_quote_change=("chg", "mean"))
    if "spr" in df:
        byh["median_spread_bp"] = df.groupby("lh").spr.median()
    out["by_local_hour"] = {int(h): {k: float(v) for k, v in r.items()} for h, r in byh.iterrows()}
    return out, byh


# =================================================================== Step 1: Binance coverage
# The JSON stores parts of the module docstring: the text before "Outputs:" as prespec and the text after
# the upper-case word AUDIT as audit_note. Keep "Outputs:" and exactly one upper-case AUDIT in the docstring.
OUT = dict(market="PLN", prespec=__doc__.split("Outputs:")[0], audit_note=__doc__.split("AUDIT")[1])
log("loading Binance and Dukascopy")
ub = load_binance("USDTPLN")
fmon = sorted({os.path.basename(f)[11:18] for f in glob.glob(os.path.join(RAW, "binance", "USDTPLN-1m-*.csv"))})
OUT["binance_files"] = fmon
wk = ub.loc[WIN["PLN"][0]:WIN["PLN"][1]]
wk = wk[wk.index.dayofweek < 5]
hr = wk.index.hour + wk.index.minute / 60
wk = wk[~((hr >= 20.5) & (hr < 22.5))]
mon = wk.groupby(wk.index.tz_localize(None).to_period("M").astype(str))
OUT["binance_by_month"] = {m: dict(minutes=int(len(d)), share_traded=float((d.n > 0).mean()), median_vol=float(d.v.median()),
                                   mean_vol=float(d.v.mean()), median_trades=float(d.n.median())) for m, d in mon}
lhw = local_hour(wk.index, "PLN")
bh_byh = pd.DataFrame({"lh": lhw, "traded": (wk.n > 0).values, "v": wk.v.values, "n": wk.n.values}).groupby("lh")
OUT["binance_by_local_hour"] = {int(h): dict(share_traded=float(d.traded.mean()), median_vol=float(d.v.median()),
                                             mean_vol=float(d.v.mean()), median_trades=float(d.n.median())) for h, d in bh_byh}
px = wk.close[wk.close > 0]
ticks = np.diff(np.unique(np.round(px.values, 6)))
tick = float(ticks[ticks > 1e-9].min())
OUT["median_volume_usdt"] = float(wk.v.median())
OUT["median_trades_per_min"] = float(wk.n.median())
binq = dict(window="weekdays, outside 20:30-22:30 UTC, 2024-01-01 to 2026-07-01",
            share_minutes_traded=float((wk.n > 0).mean()), median_vol_per_min=float(wk.v.median()),
            mean_vol_per_min=float(wk.v.mean()), median_trades_per_min=float(wk.n.median()),
            median_vol_traded_minutes=float(wk.v[wk.n > 0].median()),
            price_tick_pln=tick, price_tick_bp=float(1e4 * tick / px.median()),
            last_minute_with_trade=str(ub[ub.n > 0].index.max()))
log("binance", binq)

# =================================================================== Step 2: price-scale check
fx_p = load_fx_cov("USDPLN")
OUT["fx_benchmark"] = FXSRC.get("USDPLN")
chk = []
for ts in ["2024-01-15 12:00", "2024-07-15 12:00", "2025-01-15 12:00", "2025-04-15 12:00", "2025-10-15 12:00", "2026-03-16 12:00", "2026-06-15 12:00"]:
    t = pd.Timestamp(ts, tz="UTC")
    f = fx_p.fx_mid.reindex([t], method="ffill").iloc[0]
    b_ = ub.loc[t - pd.Timedelta("30min"):t + pd.Timedelta("30min")]
    v = float((b_.vwap * b_.v).sum() / b_.v.sum()) if b_.v.sum() > 0 else np.nan
    chk.append(dict(utc=ts, dukascopy_mid=float(f), binance_vwap_pm30min=v, premium_bp=float(1e4 * np.log(v / f))))
OUT["scale_check"] = chk
log("scale check", chk)

# =================================================================== Step 3: panels
EXIT_MONTH = "2026-06"   # the Binance USDT/PLN wind-down month (delisting 2026-07-01); see the docstring
PL_HOL = ["2024-01-01", "2024-04-01", "2024-05-01", "2024-05-03", "2024-05-30", "2024-08-15", "2024-11-01",
          "2024-11-11", "2024-12-25", "2024-12-26", "2025-01-01", "2025-01-06", "2025-04-21", "2025-05-01",
          "2025-06-19", "2025-08-15", "2025-11-11", "2025-12-24", "2025-12-25", "2025-12-26", "2026-01-01",
          "2026-01-06", "2026-04-06", "2026-05-01", "2026-06-04"]   # Polish public holidays on weekdays


def thin_checks(a, beta):
    out = {}
    out["c1_10trades"] = pooled(a[(a.ntr >= 10) & (a.ntr_l >= 10)].copy())
    med = a.groupby("ms").vol.transform("median")
    out["c2_vol_above_cell_median"] = pooled(a[(a.vol > med) & (a.vol_l > med)].copy())
    s3 = a[a.c2ok & a.x2.notna()].copy()
    geom(s3, "x2")
    s3["R"] = -s3.s * (s3.y - beta * s3.q)
    for c in ["d_out", "d_in"]:
        s3[c + "_bh"] = s3[c] * s3.bh
    out["c3_eiv_lag2"] = pooled(s3)
    return out


def informative(base, c3):
    i = base["closed"][0] > 2 * base["closed"][1] and base["open"][0] > 2 * base["open"][1]
    ii = (c3["closed"][0] > 0 and np.isfinite(c3["log_ratio"][0]) and np.isfinite(base["log_ratio"][0])
          and np.sign(c3["log_ratio"][0]) == np.sign(base["log_ratio"][0]))
    return dict(rule_i_both_speeds_gt_2se=bool(i), rule_ii_eiv_keeps_sign=bool(ii), informative=bool(i and ii))


def versus(T, P):
    A = {"TRY": T, "PLN": P}
    dv = dict(pooled=diff_vs(A, ["d_out", "d_out_bh", "d_in", "d_in_bh"], "R", [("d_out", "d_out_bh")])["d_out"])
    sy = diff_vs(A, ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"], "y", [("up", "up_bh"), ("dn", "dn_bh")])
    sR = diff_vs(A, ["d_upo", "d_upo_bh", "d_dno", "d_dno_bh", "d_in", "d_in_bh"], "R", [("d_upo", "d_upo_bh"), ("d_dno", "d_dno_bh")])
    dv.update(above_y=sy["up"], below_y=sy["dn"], above_R=sR["d_upo"], below_R=sR["d_dno"],
              note="D = log ratio(TRY) - log ratio(PLN); stacked = one UTC-day cluster across both markets",
              common_months=sorted(P.month.unique()))
    dv["est"] = dv["pooled"]["diff_log_ratio"]["est"]
    dv["se_stacked"] = dv["pooled"]["diff_log_ratio"]["se_stacked"]
    dv["se_indep"] = dv["pooled"]["diff_log_ratio"]["se_indep"]
    dv["TRY_ratio_common"] = dv["pooled"]["TRY"]["ratio"][0]
    dv["TRY_logratio_common"] = dv["pooled"]["TRY"]["log_ratio"]
    return dv


def premium_stats(a):
    w = a.groupby(["month", "state"]).agg(l=("l", "first"), u=("u", "first")).reset_index(); w["w"] = w.u - w.l
    return w, dict(mean=float(a.x.mean()), sd=float(a.x.std()), share_out=float((a.d_out > 0).mean()),
                   share_out_open=float((a.d_out[a.bh == 1] > 0).mean()), share_out_closed=float((a.d_out[a.bh == 0] > 0).mean()),
                   median_d_out=float(a.d_out.median()),
                   width_open=float(w[w.state == "open"].w.mean()), width_closed=float(w[w.state == "closed"].w.mean()),
                   median_width_open=float(w[w.state == "open"].w.median()), median_width_closed=float(w[w.state == "closed"].w.median()),
                   n_degenerate_cells=int((w.w == 0).sum()), n_cells=int(len(w)),
                   n_open=int((a.bh == 1).sum()), n_closed=int((a.bh == 0).sum()),
                   vol_per_min_in_sample=float(a.vol.mean() / 5))


P_full = build("PLN", *WIN["PLN"])
T_all = build("TRY", *WIN["TRY"])
P = P_full[P_full.month != EXIT_MONTH].copy()
beta_p = prep(P)
months = sorted(P.month.unique())
OUT["exit_month_dropped"] = EXIT_MONTH
OUT["months"] = months
OUT["n_intervals"] = int(len(P))
OUT["days"] = int(P.day.nunique())
OUT["beta_q"] = beta_p
w, OUT["premium"] = premium_stats(P)
OUT["cells"] = {f"{r.month}_{r.state}": [float(r.l), float(r.u)] for r in w.itertuples()}
log("PLN panel", len(P), "months", len(months), "beta", beta_p)

# wind-down diagnostics
jn = P_full[P_full.month == EXIT_MONTH]
wd = jn.groupby(jn.day).x.agg(["median", "min", "size"])
OUT["wind_down"] = dict(
    cell_bands={s: [float(d.l.iloc[0]), float(d.u.iloc[0])] for s, d in jn.groupby("state")},
    cell_median_premium={s: float(d.x.median()) for s, d in jn.groupby("state")},
    share_outside_band=float((jn.d_out > 0).mean()), median_d_out=float(jn.d_out.median()),
    median_d_out_other_months=float(P.d_out.median()),
    daily={str(k): dict(median=float(r["median"]), min=float(r["min"]), n=int(r["size"])) for k, r in wd.iterrows()})
log("wind-down", {k: v for k, v in OUT["wind_down"].items() if k != "daily"})

# =================================================================== Step 4: estimates
base = pooled(P)
OUT["pooled"] = base
OUT["kappa_closed"], OUT["kappa_closed_se"] = base["closed"]
OUT["increment"], OUT["increment_se"] = base["incr"]
OUT["kappa_open"], OUT["kappa_open_se"] = base["open"]
OUT["ratio"], OUT["ratio_se"] = base["ratio"]
OUT["logratio"], OUT["logratio_se"] = base["log_ratio"]
log("pooled", json.dumps(base, default=float))
sd_ = sides(P)
OUT["above_band"], OUT["below_band"] = sd_["above"], sd_["below"]
OUT["beta_q_by_state"] = dict(closed=sd_["beta_q_closed"], bh_increment=sd_["beta_q_bh"])
log("sides", json.dumps(sd_, default=float))
cyc_p = cycle(P)
OUT["cycle"] = cyc_p
log("cycle", pd.DataFrame(cyc_p["blocks"]).round(4).to_string())

# Turkey: replication of the published window, then the common months
T_pub = build("TRY", *WIN["TRY_pub"])
prep(T_pub)
rep = pooled(T_pub)
pubj = json.load(open(os.path.join(TAB, "emp_mxn.json")))["TRY"]
OUT["TRY_replication"] = dict(ratio=rep["ratio"][0], log_ratio=rep["log_ratio"], n=rep["n"],
                              published=dict(ratio=pubj["ratio"], log_ratio=pubj["log_ratio"], se=pubj["se_log_ratio"], n=pubj["n"]))
log("TRY replication", OUT["TRY_replication"])
T = T_all[T_all.month.isin(months)].copy()
beta_t = prep(T)
OUT["TRY_common"] = dict(months=sorted(T.month.unique()), n=int(len(T)), beta_q=beta_t, pooled=pooled(T),
                         sides=sides(T), sd_prem=float(T.x.std()), vol_per_min=float(T.vol.mean() / 5))
cyc_t = cycle(T)
OUT["TRY_common"]["cycle"] = cyc_t
OUT["diff_vs_TRY"] = versus(T, P)
log("diff vs TRY", json.dumps(OUT["diff_vs_TRY"]["pooled"], default=float))

# =================================================================== Step 5: quality and thin-trading checks
q = dict(binance=binq)
d0, d1 = "20240101", "20260701"
fxa_p, byh_p = fx_activity("USDPLN", "PLN", d0, d1)
fxa_t, byh_t = fx_activity("USDTRY", "TRY", d0, d1)
q["dukascopy_USDPLN"] = fxa_p
q["dukascopy_USDTRY_same_months"] = fxa_t
# five-minute intervals kept, by Warsaw hour (potential = weekday, non-rollover five-minute slots)
slots = pd.date_range(P.index.min().floor("D"), P.index.max().ceil("D"), freq="5min", inclusive="left")
slots = slots[(slots.dayofweek < 5)]
sh = slots.hour + slots.minute / 60
slots = slots[~((sh >= 20.5) & (sh < 22.5))]
pot = pd.Series(local_hour(slots, "PLN")).value_counts().sort_index()
kept = pd.Series(P.lh.values).value_counts().sort_index()
q["share_intervals_kept_by_local_hour"] = {int(h): float(kept.get(h, 0) / pot[h]) for h in pot.index}
q["share_intervals_kept"] = float(len(P) / len(slots))
q["median_trades_per_interval"] = float(P.ntr.median())
q["median_vol_per_interval"] = float(P.vol.median())
thin = thin_checks(P, beta_p)
thinT = thin_checks(T, beta_t)
q["thin_trading_PLN"] = thin
q["thin_trading_TRY_common"] = thinT
q["informative"] = informative(base, thin["c3_eiv_lag2"])
OUT["quality"] = q
log("thin", json.dumps({k: (v["closed"], v["incr"], v["log_ratio"], v["n"]) for k, v in thin.items()}, default=float))
log("thin TRY", json.dumps({k: (v["closed"], v["incr"], v["log_ratio"], v["n"]) for k, v in thinT.items()}, default=float))
log("informative", q["informative"])

# =================================================================== full sample (incl. the wind-down month)
Pf = P_full.copy()
beta_pf = prep(Pf)
Tf = T_all[T_all.month.isin(sorted(Pf.month.unique()))].copy()
beta_tf = prep(Tf)
base_f = pooled(Pf)
thin_f = thin_checks(Pf, beta_pf)
cyc_f = cycle(Pf)
_, prem_f = premium_stats(Pf)
OUT["prespecified_full_sample"] = dict(
    months=sorted(Pf.month.unique()), n_intervals=int(len(Pf)), beta_q=beta_pf, premium=prem_f, pooled=base_f,
    sides={k: v for k, v in sides(Pf).items() if k in ("above", "below")},
    cycle={k: v for k, v in cyc_f.items() if k != "blocks"},
    TRY_common_pooled=pooled(Tf), diff_vs_TRY=versus(Tf, Pf), thin_trading_PLN=thin_f,
    informative=informative(base_f, thin_f["c3_eiv_lag2"]))
log("full sample incl. 2026-06", json.dumps(base_f, default=float), OUT["prespecified_full_sample"]["informative"])

# =================================================================== sensitivities
sens = {}
for lab, end in [("end_2026-06-16", "2026-06-16 23:59"), ("end_2026-06-26", "2026-06-26 23:59")]:
    Pv = build("PLN", WIN["PLN"][0], end)
    prep(Pv)
    jv = Pv[Pv.month == EXIT_MONTH]
    sens["cutoff_" + lab] = dict(pooled=pooled(Pv), n=int(len(Pv)),
                                 june_bands={s: [float(d.l.iloc[0]), float(d.u.iloc[0])] for s, d in jv.groupby("state")})
for nm, a in [("PLN", P), ("TRY_common", T)]:
    med = a.groupby("ms").x.transform("median")
    for cut in [300, 200, 150]:
        keep = (a.x - med).abs() <= cut
        sens[f"xtrim{cut}_{nm}"] = dict(dropped=int((~keep).sum()), pooled=pooled(a[keep].copy()))
hol = pd.Index(pd.to_datetime(PL_HOL).date)
sens["drop_PL_holidays"] = pooled(P[~P.day.isin(hol)].copy())
lomo = {}
for m in months:
    o = pooled(P[P.month != m].copy())
    lomo[m] = dict(closed=o["closed"], incr=o["incr"], log_ratio=o["log_ratio"])
sens["leave_one_month_out"] = lomo
OUT["sensitivity_post_hoc"] = sens
log("sensitivity", json.dumps({k: (v["pooled"] if "pooled" in v else v)["log_ratio"] for k, v in sens.items() if k != "leave_one_month_out"}, default=float))

# =================================================================== figure
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for k, cyc, col, mk, lab, off in [("TRY", cyc_t, C1, "o", "Turkey (same months)", 0.0), ("PLN", cyc_p, C2, "s", "Poland", 0.3)]:
    c = pd.DataFrame(cyc["blocks"])
    ax[0].errorbar(c.h + off, c.k, yerr=1.96 * c.se, fmt=mk + "-", ms=3.5, color=col, ecolor=col, lw=0.9, alpha=0.9, label=lab)
ax[0].axvspan(9, 17, color=C3, alpha=0.12); ax[0].axhline(0, color="k", lw=0.4)
ax[0].set_xlabel("local hour"); ax[0].set_ylabel(r"reversion speed outside the band, $\hat\kappa$")
ax[0].set_xticks(range(0, 25, 4)); ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[0].set_title(f"(a) Speed by two-hour block, {months[0]} to {months[-1]}", fontsize=9, loc="left")
hb = pd.DataFrame(OUT["binance_by_local_hour"]).T.sort_index()
ax[1].plot(hb.index, hb.share_traded, "s-", color=C2, ms=3, lw=0.9, label="USDT/PLN: minutes with a trade")
ax[1].plot(byh_p.index, byh_p.share_quote_change, "^--", color=C2, ms=3, lw=0.9, label="USD/PLN: minutes with a new quote")
ax[1].plot(byh_t.index, byh_t.share_quote_change, "o--", color=C1, ms=3, lw=0.9, label="USD/TRY: minutes with a new quote")
ax[1].axvspan(9, 17, color=C3, alpha=0.12)
ax[1].set_ylim(0, 1.05); ax[1].set_xticks(range(0, 25, 4)); ax[1].set_xlabel("local hour"); ax[1].set_ylabel("share of weekday minutes")
ax[1].legend(frameon=False, fontsize=7, loc="lower right")
ax[1].set_title("(b) Trading and quoting activity", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rob_xc_pln.pdf")); plt.close(fig)

json.dump(OUT, open(os.path.join(TAB, "rob_xc_pln.json"), "w"), indent=1, default=float)
log("done")
