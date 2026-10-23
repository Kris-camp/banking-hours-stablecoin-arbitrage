"""Cross-country contrast, South Africa: Binance USDT/ZAR against Dukascopy USD/ZAR, compared with
Turkey (Table emp_mxn: open/closed ratio of the speed outside the band 4.2) and Mexico (1.6).

Binance USDTZAR monthly archives: 32 months, 2024-01 to 2026-08, about 74% of minutes with at least one
trade, median one-minute volume about 55 USDT.

DESIGN
======
Construction: identical to code/empirics_mxn.py build(), applied to ZAR.
  Binance USDT/ZAR minute VWAP against the Dukascopy USD/ZAR mid (bid/ask one-minute candles; where an
  ask file is missing, load_fx uses bid x (1 + half the median relative spread)), forward-filled at
  most 2 minutes, minutes with zero Dukascopy volume dropped (load_fx), weekdays only, 20:30-22:30 UTC
  rollover excluded, |b| <= 1000 bp, five-minute means with at least two valid minutes, OFI summed and
  standardized within month, consecutive intervals only.  Local business hours 09:00-17:00
  Africa/Johannesburg (zoneinfo; SAST is UTC+2 with no DST in the sample, zoneinfo would handle it
  otherwise); Turkey 09:00-17:00 Europe/Istanbul (UTC+3), as in empirics_mxn.py.  Bands profiled by
  month x local banking state on a 1 bp grid with 5% trimming, cells with fewer than 300 intervals
  dropped; month-by-state fixed effects; standard errors clustered by UTC day with the finite-sample
  correction (ols() copied from empirics_mxn.py).  No holiday adjustment (as in empirics_mxn.py).

Windows.  Primary: all ZAR months, January 2024 - August 2026.  Secondary: November 2024 - August 2026,
  the window of Table emp_mxn, so the ZAR ratio can be read next to Turkey 4.2 and Mexico 1.6.  The
  secondary window uses the same month x state bands (bands are month-specific) and re-estimates beta.
  Turkey is rebuilt with the same code over the same months (restricted to the months that survive in
  the ZAR panel); on the secondary window this must reproduce Table emp_mxn's Turkish numbers
  (validation of the reimplementation).

Estimates (per market, same specification as Table emp_mxn):
  (1) beta from y ~ q + up + dn; R = -s (dy - beta q); R ~ d_out + d_out x BH + d_in + d_in x BH.
      kappa_closed = coef on d_out, increment = coef on d_out x BH, kappa_open = sum, ratio =
      open/closed, log ratio with delta-method s.e.  The log ratio is reported only if both speeds are
      positive.
  (2) Above / below band: (a) y ~ q + up + dn + q x BH + up x BH + dn x BH (the "above band only" row
      of Table emp_mxn), (b) R on d_out above, d_out below, d_in, each x BH (R-based, as in
      rob_mexico_utc.py [r3]).
  (3) Daily cycle (as Figure emp_mxn_cycle): R ~ d_out + d_in with month FE in each two-hour block of
      local time, blocks with at least 200 out-of-band intervals.  Night = blocks 00-08 local, day =
      blocks 10-16 local; night/day = mean of night-block speeds / mean of day-block speeds (as
      emp_mxn.json).  Also a pooled night (00-08) vs day (10-16) regression with separate slopes and
      month x period FE, giving log(day/night) with a delta-method s.e.
  (4) Stacked difference versus Turkey: both markets in one regression, every regressor interacted
      with market, market x month x state FE, the same UTC day one cluster across markets.  Headline:
      log ratio(TRY) - log ratio(ZAR), pooled sides, with the stacked s.e. and the independent-samples
      s.e.; also above and below band from spec (2a).  Versus Mexico (secondary window only): the
      emp_mxn.json Mexican log ratio, independent-samples s.e. (Mexico is not rebuilt here).

Classification stored in the JSON (primary window; reported for both):
  "Turkey-like" if the ZAR log ratio is positive and significant at 5% and the stacked TRY - ZAR
  difference is not significant at 5%; "Mexico-like" (smaller business-hours effect than Turkey) if
  TRY - ZAR > 0 at 5% with the stacked s.e.; "inconclusive" otherwise.

Data quality and liquidity (diagnostics; they do not change the baseline sample):
  (Q0) Binance: share of minutes with at least one trade, median one-minute USDT volume and trade count,
       by month and by local hour (weekdays); Turkey over the same window for scale.
  (Q1) Dukascopy USD/ZAR by local hour (weekdays): share of minutes with positive quote volume, share of
       minutes whose bid close differs from the previous minute's, median relative bid-ask spread;
       Dukascopy price scale (1e5, 0-based month in the URL) checked against the Binance level by month.
  (Q2) Panel coverage: share of possible weekday five-minute intervals that enter the regression, by
       local hour; month x state cells dropped for having fewer than 300 intervals.
  (Q3) Noise: first-order autocorrelation of the five-minute premium change by banking state, both
       markets (strongly negative values mean measurement noise in the five-minute premium).
  (Q4) Thin-trading filter: at least 10 trades in both t-1 and t; baseline bands and beta kept.
  (Q5) Errors in variables: side and distances computed from b_{t-2} (three consecutive intervals),
       so iid noise in the five-minute premium cannot generate reversion; baseline bands and beta.
  Reliability flags (reported, not used to change the sample): (i) kappa_closed not significantly
  positive at 5% (the log-ratio delta method is unreliable near zero); (ii) Q5 closed-hours speed
  below half its baseline value (reversion is mostly noise); (iii) fewer than half of night-time
  (00-08 local) weekday minutes have an active Dukascopy quote.

  (Q6) Benchmark construction.  load_fx uses the bid-ask mid on days with an ASK file and
       bid x (1 + half the median relative spread) otherwise, which is how the Turkish and Mexican
       panels are built (they have almost no ASK files).  Q6 re-runs the pooled-sides estimate with the
       bid-only construction on every day (bands re-profiled), the exact Turkey/Mexico construction.
       If ASK coverage is incomplete, the baseline is the load_fx mix and the JSON reports coverage.
       The duka_raw() reader below reproduces load_fx; the script checks that the two agree.

INFLUENTIAL DAYS (they do not change the baseline or the classification above)
  A1 day-jackknife s.e.; A2 estimates without "dislocation days" (five-minute |b - monthly median| above
  300/400/500 bp; all three reported); A3 the stacked TRY - ZAR difference in the b_{t-2} variant.
  Stored under the "audit" key of the JSON and in the bottom rows of the table.  See the comment block
  "INFLUENTIAL DAYS (AUDIT ADDITION)" above the code that computes them.

Inputs: data_raw/binance (USDTZAR, USDTTRY), data_raw/dukascopy (USDZAR bid and ask, USDTRY), tables/emp_mxn.json.
Outputs: tables/rob_xc_zar.json, tables/rob_xc_zar.tex, figures/rob_xc_zar.pdf.
"""
import os, glob, json, lzma, time, datetime as dt
from zoneinfo import ZoneInfo
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
W0, W1 = "2024-01-01", "2026-08-31"
M_MXN = "2024-11"  # first month of the Table emp_mxn window
TZ = {"TRY": ZoneInfo("Europe/Istanbul"), "ZAR": ZoneInfo("Africa/Johannesburg")}
MK = ["TRY", "ZAR"]
NAME = "rob_xc_zar"


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


def raw_minutes(local):
    u = load_binance(f"USDT{local}").loc[W0:W1]
    fx = load_fx(f"USD{local}")
    p = pd.DataFrame({"pl": u.vwap, "ofi": 2 * u.tbv - u.v, "vol": u.v, "ntr": u.n}).join(
        fx.reindex(u.index, method="ffill", limit=2))
    return p, u, fx


def duka_raw(pair):
    rows = []
    nb = na = nmiss = 0
    for side in ("BID", "ASK"):
        for f in sorted(glob.glob(os.path.join(RAW, "dukascopy", f"{pair}_{side}_*.bi5"))):
            if os.path.getsize(f) == 0:
                nmiss += 1; continue
            raw = lzma.decompress(open(f, "rb").read())
            day = pd.Timestamp(dt.datetime.strptime(os.path.basename(f).split("_")[2][:8], "%Y%m%d"), tz="UTC")
            a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6)
            v = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5]
            idx = day + pd.to_timedelta(a[:, 0].astype("int64"), unit="s")
            rows.append(pd.DataFrame({"side": side, "close": a[:, 2] / 1e5, "v": v}, index=idx))
            if side == "BID": nb += 1
            else: na += 1
    d = pd.concat(rows)
    bid = d[d.side == "BID"]; ask = d[d.side == "ASK"]
    bid = bid[~bid.index.duplicated()].sort_index(); ask = ask[~ask.index.duplicated()].sort_index()
    m = pd.DataFrame({"bid": bid.close, "v": bid.v}); m["ask"] = ask.close.reindex(m.index)
    return m, dict(bid_days=nb, ask_days=na, empty_files=nmiss)

def fx_from_raw(m, use_ask=True):
    """replicates build_panel.load_fx from duka_raw() output; use_ask=False gives the bid-only benchmark."""
    m = m[m.v > 0].copy()
    rel = ((m.ask - m.bid) / m.bid).dropna()
    half = 0.5 * (rel.median() if len(rel) else 1e-4)
    ask = m.ask if use_ask else pd.Series(np.nan, index=m.index)
    mid = np.where(ask.notna(), (m.bid + ask) / 2, m.bid * (1 + half))
    return pd.DataFrame({"fx_mid": mid, "fx_spread": (ask - m.bid) / mid, "fxv": m.v}, index=m.index)


def geom(a, xcol="x"):
    x = a[xcol]
    m = (a.l + a.u) / 2
    a["s"] = np.where(x > a.u, 1.0, np.where(x < a.l, -1.0, np.sign(x - m)))
    a["d_out"] = np.maximum(x - a.u, 0) + np.maximum(a.l - x, 0)
    a["d_in"] = np.where(a.d_out == 0, np.abs(x - m), 0.0)
    a["up"] = -np.maximum(x - a.u, 0); a["dn"] = np.maximum(a.l - x, 0)
    a["d_upo"] = np.maximum(x - a.u, 0); a["d_dno"] = np.maximum(a.l - x, 0)
    return a


def build(local, p):
    p = p.copy()
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
    dt_ = a.index.to_series().diff()
    consec = dt_ == pd.Timedelta("5min")
    consec2 = consec & (dt_.shift(1) == pd.Timedelta("5min"))
    a["y"] = a.b.diff(); a["x"] = a.b.shift(1); a["x2"] = a.b.shift(2)
    a["ntr_l"] = a.ntr.shift(1)
    a["c2ok"] = consec2.values
    a = a[consec & a.y.notna() & a.x.notna() & a.q.notna()].copy()
    a["lh"] = a.index.tz_convert(TZ[local]).hour
    a["bh"] = ((a.lh >= 9) & (a.lh < 17)).astype(float)
    a["state"] = np.where(a.bh == 1, "open", "closed")
    a["ms"] = a.month + "_" + a.state
    dropped = []
    for key, d in a.groupby("ms"):
        if len(d) < 300:
            dropped.append([key, int(len(d))]); continue
        l, u_ = profile(d)
        a.loc[d.index, "l"], a.loc[d.index, "u"] = l, u_
    a = a[a.l.notna()].copy()
    geom(a)
    a["day"] = a.index.date
    a["mkt"] = local
    return a, dropped


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


def P(v):
    return [float(v[0]), float(v[1])]


def prep_R(a, beta=None):
    if beta is None:
        beta = ols(a, "y", ["q", "up", "dn"])["b"]["q"]
    a["R"] = -a.s * (a.y - beta * a.q)
    for c in ["d_out", "d_in", "up", "dn", "q", "d_upo", "d_dno"]:
        a[c + "_bh"] = a[c] * a.bh
    return float(beta)


def two_state(r, c0, c1):
    """closed speed, increment, open speed, ratio and log ratio from coefficients c0 (closed) and c1 (BH x)."""
    b0, b1 = r["b"][c0], r["b"][c1]
    ok = b0 > 0 and b0 + b1 > 0
    return dict(closed=[float(b0), float(r["se"][c0])], incr=[float(b1), float(r["se"][c1])],
                open=P(dm(r, lambda b: b[c0] + b[c1])),
                ratio=float((b0 + b1) / b0) if b0 != 0 else float("nan"),
                log_ratio=P(dm(r, lambda b: np.log((b[c0] + b[c1]) / b[c0]))) if ok else [float("nan")] * 2)


def market_estimates(a):
    r = ols(a, "R", ["d_out", "d_out_bh", "d_in", "d_in_bh"])
    o = two_state(r, "d_out", "d_out_bh")
    o["d_in"] = [float(r["b"]["d_in"]), float(r["se"]["d_in"])]
    o["d_in_bh"] = [float(r["b"]["d_in_bh"]), float(r["se"]["d_in_bh"])]
    o["n"], o["days"] = int(r["n"]), int(r["ndays"])
    ra = ols(a, "y", ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
    rr = ols(a, "R", ["d_upo", "d_upo_bh", "d_dno", "d_dno_bh", "d_in", "d_in_bh"])
    o["above_band"] = dict(spec_y=two_state(ra, "up", "up_bh"), spec_R=two_state(rr, "d_upo", "d_upo_bh"))
    o["below_band"] = dict(spec_y=two_state(ra, "dn", "dn_bh"), spec_R=two_state(rr, "d_dno", "d_dno_bh"))
    o["q"] = [float(ra["b"]["q"]), float(ra["se"]["q"])]; o["q_bh"] = [float(ra["b"]["q_bh"]), float(ra["se"]["q_bh"])]
    w = a.groupby(["month", "state"]).agg(l=("l", "first"), u=("u", "first")).reset_index()
    w["w"] = w.u - w.l
    o["sd_prem"] = float(a.x.std()); o["mean_prem"] = float(a.x.mean()); o["share_out"] = float((a.d_out > 0).mean())
    o["width_open"] = float(w[w.state == "open"].w.mean()); o["width_closed"] = float(w[w.state == "closed"].w.mean())
    o["vol_per_min"] = float(a.vol.mean() / 5)
    return o


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
    # pooled night (00-08) vs day (10-16), separate slopes, month x period FE
    d = a[(a.lh < 8) | ((a.lh >= 10) & (a.lh < 16))].copy()
    nt = (d.lh < 8).astype(float)
    for c_ in ["d_out", "d_in"]:
        d[c_ + "_n"] = d[c_] * nt; d[c_ + "_d"] = d[c_] * (1 - nt)
    d["fe"] = d.month + "_" + np.where(nt == 1, "n", "d")
    r = ols(d, "R", ["d_out_n", "d_out_d", "d_in_n", "d_in_d"], fe="fe")
    kn, kd = r["b"]["d_out_n"], r["b"]["d_out_d"]
    pooled = dict(night=[float(kn), float(r["se"]["d_out_n"])], day=[float(kd), float(r["se"]["d_out_d"])],
                  diff_day_minus_night=P(dm(r, lambda b: b["d_out_d"] - b["d_out_n"])),
                  log_day_over_night=P(dm(r, lambda b: np.log(b["d_out_d"] / b["d_out_n"]))) if (kn > 0 and kd > 0) else [float("nan")] * 2)
    return dict(blocks=rows, night_over_day=float(night / day), pooled=pooled)


def stack(A, cols, fe, ycol):
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


def indep_se(res, fun):
    cols = res["cols"]; V = res["V"].copy()
    i0 = [i for i, c in enumerate(cols) if c.endswith("__TRY")]; i1 = [i for i, c in enumerate(cols) if c.endswith("__ZAR")]
    V[np.ix_(i0, i1)] = 0; V[np.ix_(i1, i0)] = 0
    r2 = dict(res); r2["V"] = V
    return dm(r2, fun)[1]


def diff_vs_try(A):
    out = {}
    specs = [("pooled", ["d_out", "d_out_bh", "d_in", "d_in_bh"], "R", "d_out", "d_out_bh"),
             ("above_band", ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"], "y", "up", "up_bh"),
             ("below_band", ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"], "y", "dn", "dn_bh")]
    for nm, cols, ycol, c0, c1 in specs:
        r = stack(A, cols, "ms", ycol)
        o = {k: two_state(r, f"{c0}__{k}", f"{c1}__{k}") for k in MK}
        lr = lambda k: (lambda b: np.log((b[f"{c0}__{k}"] + b[f"{c1}__{k}"]) / b[f"{c0}__{k}"]))
        if all(np.isfinite(o[k]["log_ratio"][0]) for k in MK):
            f = lambda b: lr("TRY")(b) - lr("ZAR")(b)
            v, s = dm(r, f)
            o["diff_log_ratio_TRY_minus_ZAR"] = dict(est=v, se_stacked=s, se_indep=indep_se(r, f))
        else:
            o["diff_log_ratio_TRY_minus_ZAR"] = dict(est=float("nan"), se_stacked=float("nan"), se_indep=float("nan"))
        f = lambda b: b[f"{c1}__TRY"] - b[f"{c1}__ZAR"]
        v, s = dm(r, f)
        o["diff_increment_level_TRY_minus_ZAR"] = dict(est=v, se_stacked=s, se_indep=indep_se(r, f),
                                                       note="levels not comparable across markets (lambda differs)")
        out[nm] = o
    out["months"] = sorted(A["ZAR"].month.unique().tolist())
    out["n"] = {k: int(len(A[k])) for k in MK}
    return out


def variant(a, beta, mode):
    """Q4 thin-trading filter / Q5 errors-in-variables, baseline bands and beta."""
    d = a.copy()
    if mode == "thin10":
        d = d[(d.ntr >= 10) & (d.ntr_l >= 10)].copy()
    elif mode == "eiv":
        d = d[d.c2ok & d.x2.notna()].copy()
        geom(d, "x2")
    prep_R(d, beta)
    r = ols(d, "R", ["d_out", "d_out_bh", "d_in", "d_in_bh"])
    o = two_state(r, "d_out", "d_out_bh"); o["n"] = int(r["n"])
    return o


def ac1(a):
    d = a[a.c2ok & a.x2.notna()]
    yl = d.x - d.x2
    return {st: float(np.corrcoef(d.y[d.state == st], yl[d.state == st])[0, 1]) for st in ["open", "closed"]}


def run_window(A, tag):
    log("window", tag, {k: (len(A[k]), A[k].month.min(), A[k].month.max()) for k in MK})
    R = {}
    for k in MK:
        beta = prep_R(A[k])
        R[k] = market_estimates(A[k]); R[k]["beta"] = beta
        R[k]["cycle"] = cycle(A[k])
        R[k]["noise_ac1_dy"] = ac1(A[k])
        log(tag, k, "closed", R[k]["closed"], "incr", R[k]["incr"], "ratio", round(R[k]["ratio"], 3), "logratio", R[k]["log_ratio"])
    R["diff_vs_TRY"] = diff_vs_try(A)
    log(tag, "diff log ratio TRY-ZAR", R["diff_vs_TRY"]["pooled"]["diff_log_ratio_TRY_minus_ZAR"])
    for k in MK:
        R[k]["q4_thin10"] = variant(A[k], R[k]["beta"], "thin10")
        R[k]["q5_eiv_bt2"] = variant(A[k], R[k]["beta"], "eiv")
    return R


# ------------------------------------------------------------------ data
OUT = dict(market="ZAR", pair_local="USDTZAR (Binance)", pair_fx="USDZAR (Dukascopy SWFX)",
           local_tz="Africa/Johannesburg", business_hours="09:00-17:00 local (07:00-15:00 UTC)")
PRAW, U, FX, A, DROP = {}, {}, {}, {}, {}
for k in MK:
    PRAW[k], U[k], FX[k] = raw_minutes(k)
    A[k], DROP[k] = build(k, PRAW[k])
    log("built", k, len(A[k]), "dropped cells", DROP[k])
DZRAW, nfiles = duka_raw("USDZAR")
_chk = fx_from_raw(DZRAW, True).fx_mid.reindex(FX["ZAR"].index)
FXCHECK = float(np.nanmax(np.abs(_chk.values / FX["ZAR"].fx_mid.values - 1)))
log("fx_from_raw vs load_fx max rel diff", FXCHECK)
assert FXCHECK < 1e-9
_fxb = fx_from_raw(DZRAW, False)
_u = U["ZAR"]
PB = pd.DataFrame({"pl": _u.vwap, "ofi": 2 * _u.tbv - _u.v, "vol": _u.v, "ntr": _u.n}).join(
    _fxb.reindex(_u.index, method="ffill", limit=2))
A_BID, DROP_BID = build("ZAR", PB)
ask_days = sorted({os.path.basename(f).split("_")[2][:8] for f in glob.glob(os.path.join(RAW, "dukascopy", "USDZAR_ASK_*.bi5"))
                   if os.path.getsize(f) > 0})
# restrict Turkey to the months present in the ZAR panel
zm = sorted(A["ZAR"].month.unique())
A["TRY"] = A["TRY"][A["TRY"].month.isin(zm)].copy()

# ------------------------------------------------------------------ quality (Q0-Q2)
Q = {}
u = U["ZAR"].copy()
u["m"] = u.index.tz_localize(None).to_period("M").astype(str)
u["lh"] = u.index.tz_convert(TZ["ZAR"]).hour
wk = u.index.dayofweek < 5
bym = u.groupby("m").agg(share_minutes_with_trades=("n", lambda s: float((s > 0).mean())),
                         median_volume_usdt=("v", "median"), mean_volume_usdt=("v", "mean"), median_trades=("n", "median"),
                         binance_median_price=("close", "median"))
fxm = FX["ZAR"].copy(); fxm["m"] = fxm.index.tz_localize(None).to_period("M").astype(str)
bym["dukascopy_median_mid"] = fxm.groupby("m").fx_mid.median()
pm = PRAW["ZAR"].copy(); pm["b"] = 1e4 * np.log(pm.pl / pm.fx_mid); pm["m"] = pm.index.tz_localize(None).to_period("M").astype(str)
bym["median_minute_premium_bp"] = pm.groupby("m").b.median()
Q["binance_by_month"] = bym.reset_index().to_dict("records")
ut = U["TRY"]
Q["binance_overall"] = dict(
    ZAR=dict(share_minutes_with_trades=float((u.n > 0).mean()), median_volume_usdt=float(u.v.median()),
             median_volume_usdt_weekdays=float(u.v[wk].median()), mean_volume_usdt=float(u.v.mean()),
             median_volume_usdt_minutes_with_trades=float(u.v[u.n > 0].median()), median_trades=float(u.n.median()),
             mean_trades=float(u.n.mean())),
    TRY_same_window=dict(share_minutes_with_trades=float((ut.n > 0).mean()), median_volume_usdt=float(ut.v.median()),
                         mean_volume_usdt=float(ut.v.mean()), median_trades=float(ut.n.median())))
uw = u[wk]
Q["binance_by_local_hour_weekdays"] = uw.groupby("lh").agg(
    share_minutes_with_trades=("n", lambda s: float((s > 0).mean())), median_volume_usdt=("v", "median"),
    mean_volume_usdt=("v", "mean"), median_trades=("n", "median")).reset_index().to_dict("records")


dz = DZRAW[DZRAW.index.dayofweek < 5].copy()
dz["lh"] = dz.index.tz_convert(TZ["ZAR"]).hour
dz["active"] = dz.v > 0
dz["chg"] = dz.bid.diff().ne(0) & (dz.index.to_series().diff() == pd.Timedelta("1min")).values
dz["spr_bp"] = np.where(dz.active & dz.ask.notna(), 1e4 * (dz.ask - dz.bid) / dz.bid, np.nan)
Q["dukascopy_files"] = nfiles
Q["dukascopy_by_local_hour_weekdays"] = dz.groupby("lh").agg(
    share_minutes_active=("active", "mean"), share_minutes_bid_changed=("chg", "mean"),
    median_spread_bp=("spr_bp", "median"), mean_quote_volume=("v", "mean")).reset_index().to_dict("records")
Q["dukascopy_overall"] = dict(share_minutes_active=float(dz.active.mean()), share_minutes_bid_changed=float(dz.chg.mean()),
                              median_spread_bp=float(np.nanmedian(dz.spr_bp)))
night_active = float(dz[dz.lh < 8].active.mean())
Q["dukascopy_night_share_active"] = night_active
Q["scale_check"] = ("Dukascopy prices are int32 / 1e5 (build_panel.load_fx); the URL month is 0-based "
                    "(download_dukascopy.fetch uses month-1). Monthly median Dukascopy mid matches the Binance USDT/ZAR "
                    "price level month by month (binance_by_month), so neither the scale nor the month offset is off.")
# panel coverage by local hour: intervals in the regression sample / possible weekday intervals (excl. rollover)
az = A["ZAR"]
alli = pd.date_range(W0, pd.Timestamp(W1) + pd.Timedelta("1D"), freq="5min", tz="UTC", inclusive="left")
alli = alli[(alli.dayofweek < 5)]
hr = alli.hour + alli.minute / 60
alli = alli[~((hr >= 20.5) & (hr < 22.5))]
poss = pd.Series(1, index=alli).groupby(alli.tz_convert(TZ["ZAR"]).hour).sum()
got = az.groupby("lh").size()
Q["panel_coverage_by_local_hour"] = [dict(lh=int(h), share_intervals_in_sample=float(got.get(h, 0) / poss[h])) for h in poss.index]
Q["panel_coverage_overall"] = float(len(az) / len(alli))
Q["cells_dropped_lt300"] = {k: DROP[k] for k in MK}
Q["months_in_panel"] = zm

# ------------------------------------------------------------------ estimation: primary and emp_mxn windows
RES = run_window(A, "primary")
A2 = {k: A[k][A[k].month >= M_MXN].copy() for k in MK}
RES2 = run_window(A2, "emp_mxn_window")

# Q6: bid-only benchmark on every day (the Turkey/Mexico construction), both windows
Q6 = {}
for tag, m0 in [("primary", "0000"), ("emp_mxn_window", M_MXN)]:
    zb = A_BID[A_BID.month >= m0]
    mb = sorted(zb.month.unique())
    AB = {"TRY": A["TRY"][A["TRY"].month.isin(mb) & (A["TRY"].month >= m0)].copy(), "ZAR": zb.copy()}
    for k in MK: prep_R(AB[k])
    e = market_estimates(AB["ZAR"])
    Q6[tag] = dict(ZAR={kk: e[kk] for kk in ["closed", "incr", "open", "ratio", "log_ratio", "n", "above_band", "below_band"]},
                   diff_vs_TRY=diff_vs_try(AB)["pooled"]["diff_log_ratio_TRY_minus_ZAR"], cells_dropped=DROP_BID)
    log("Q6", tag, Q6[tag]["ZAR"]["ratio"], Q6[tag]["ZAR"]["log_ratio"], Q6[tag]["diff_vs_TRY"])
Q["fx_construction"] = dict(bid_days=nfiles["bid_days"], ask_days=nfiles["ask_days"], empty_files=nfiles["empty_files"],
                            ask_first=ask_days[0] if ask_days else None, ask_last=ask_days[-1] if ask_days else None,
                            load_fx_replication_max_rel_diff=FXCHECK,
                            note="baseline: bid-ask mid on days with an ASK file, bid x (1+half median spread) otherwise (load_fx)")

# versus Mexico (secondary window, independent samples, from tables/emp_mxn.json)
emx = json.load(open(os.path.join(TAB, "emp_mxn.json")))
mx = dict(log_ratio=[emx["MXN"]["log_ratio"], emx["MXN"]["se_log_ratio"]], ratio=emx["MXN"]["ratio"])
z2 = RES2["ZAR"]["log_ratio"]
RES2["diff_vs_MXN"] = dict(ZAR_minus_MXN=[float(z2[0] - mx["log_ratio"][0]), float(np.hypot(z2[1], mx["log_ratio"][1]))],
                           MXN=mx, note="Mexico from tables/emp_mxn.json; independent-samples s.e.")
RES2["TRY_validation_vs_emp_mxn"] = dict(
    this_script=dict(ratio=RES2["TRY"]["ratio"], log_ratio=RES2["TRY"]["log_ratio"], closed=RES2["TRY"]["closed"], incr=RES2["TRY"]["incr"], n=RES2["TRY"]["n"]),
    emp_mxn=dict(ratio=emx["TRY"]["ratio"], log_ratio=[emx["TRY"]["log_ratio"], emx["TRY"]["se_log_ratio"]],
                 closed=emx["TRY"]["sym"]["d_out"], incr=emx["TRY"]["sym"]["d_out_bh"], n=emx["TRY"]["n"]))


def verdict(R):
    z = R["ZAR"]["log_ratio"]; d = R["diff_vs_TRY"]["pooled"]["diff_log_ratio_TRY_minus_ZAR"]
    if not np.isfinite(z[0]):
        return "inconclusive (ZAR log ratio undefined)"
    zsig = z[0] / z[1] > 1.96
    dsig = abs(d["est"] / d["se_stacked"]) > 1.96
    if d["est"] > 0 and dsig:
        return "Mexico-like: business-hours effect significantly smaller than Turkey's"
    if zsig and not dsig:
        return "Turkey-like: significant business-hours effect, not significantly different from Turkey"
    return "inconclusive"


def flags(R):
    z = R["ZAR"]
    return dict(closed_not_sig_positive=bool(z["closed"][0] / z["closed"][1] < 1.96),
                eiv_closed_below_half=bool(z["q5_eiv_bt2"]["closed"][0] < 0.5 * z["closed"][0]),
                fx_night_active_below_half=bool(night_active < 0.5))


z = RES["ZAR"]
OUT.update(dict(
    months=zm, n_months=len(zm), n_intervals=int(z["n"]), n_days=int(z["days"]),
    median_volume_usdt=Q["binance_overall"]["ZAR"]["median_volume_usdt"],
    median_trades_per_minute=Q["binance_overall"]["ZAR"]["median_trades"],
    share_minutes_with_trades=Q["binance_overall"]["ZAR"]["share_minutes_with_trades"],
    kappa_closed=z["closed"][0], kappa_closed_se=z["closed"][1], increment=z["incr"][0], increment_se=z["incr"][1],
    kappa_open=z["open"][0], kappa_open_se=z["open"][1],
    ratio=z["ratio"], logratio=z["log_ratio"][0], logratio_se=z["log_ratio"][1],
    above_band=z["above_band"], below_band=z["below_band"], cycle=z["cycle"],
    diff_vs_TRY=RES["diff_vs_TRY"], quality=Q,
    beta=z["beta"], q=z["q"], q_bh=z["q_bh"], d_in=z["d_in"], d_in_bh=z["d_in_bh"],
    sd_prem=z["sd_prem"], mean_prem=z["mean_prem"], share_out=z["share_out"],
    width_open=z["width_open"], width_closed=z["width_closed"], vol_per_min=z["vol_per_min"],
    noise_ac1_dy=z["noise_ac1_dy"], q4_thin10=z["q4_thin10"], q5_eiv_bt2=z["q5_eiv_bt2"], q6_bid_only=Q6,
    TRY_common_sample={k: v for k, v in RES["TRY"].items()},
    verdict=verdict(RES), flags=flags(RES),
    window_emp_mxn=dict(months=[M_MXN, "2026-08"], ZAR=RES2["ZAR"], TRY=RES2["TRY"], diff_vs_TRY=RES2["diff_vs_TRY"],
                        diff_vs_MXN=RES2["diff_vs_MXN"], TRY_validation_vs_emp_mxn=RES2["TRY_validation_vs_emp_mxn"],
                        verdict=verdict(RES2), flags=flags(RES2)),
    runtime_s=float(time.time() - T0)))
json.dump(OUT, open(os.path.join(TAB, f"{NAME}.json"), "w"), indent=1, default=float)
log("json written")


# ------------------------------------------------------------------ INFLUENTIAL DAYS (AUDIT ADDITION)
# Nothing above is changed. The ZAR estimates rest on a handful of days on which Binance USDT/ZAR
# traded 5-13% below the interbank rate for an hour or more (e.g. 2024-09-19, 2025-07-01, 2025-09-30), with
# the FX mid flat. These are genuine prints, not data errors, but they dominate the out-of-band
# regressions. Reported here: (A1) the exact delete-one-day jackknife s.e. (bands and beta held fixed) for
# the log ratios and the stacked TRY - ZAR difference; (A2) the estimates after dropping "dislocation days",
# meaning days with a five-minute mean |b - monthly median minute b| above 300, 400 or 500 bp (all three
# thresholds are reported, none is preferred), for ZAR only and for both markets, with bands re-profiled;
# (A3) the stacked TRY - ZAR difference in the b_{t-2} errors-in-variables variant (Q5).
def jackknife(d, ycol, cols, fe, funs):
    """exact delete-one-UTC-day jackknife of the within(fe) OLS; bands, beta and R held fixed."""
    X = d[cols].values.astype(float); Y = d[ycol].values.astype(float)
    cell = pd.factorize(d[fe])[0]; day = pd.factorize(d.day)[0]
    nc, nd, k = cell.max() + 1, day.max() + 1, len(cols)
    Z = np.column_stack([X, Y])
    n_c = np.bincount(cell, minlength=nc).astype(float)
    S_c = np.zeros((nc, k + 1)); np.add.at(S_c, cell, Z)
    ZZ = Z.T @ Z
    ZZd = np.zeros((nd, k + 1, k + 1))
    for j in range(k + 1):
        np.add.at(ZZd[:, j, :], day, Z * Z[:, [j]])
    key = day.astype(np.int64) * nc + cell
    uk, inv = np.unique(key, return_inverse=True)
    Sdc = np.zeros((len(uk), k + 1)); np.add.at(Sdc, inv, Z); ndc = np.bincount(inv).astype(float)
    kd, kc = uk // nc, uk % nc
    st, en = np.searchsorted(kd, np.arange(nd)), np.searchsorted(kd, np.arange(nd), side="right")

    def solve(ZZt, n, S):
        ok = n > 0
        M = ZZt - (S[ok].T / n[ok]) @ S[ok]
        return np.linalg.solve(M[:k, :k], M[:k, k])
    b0 = solve(ZZ, n_c, S_c)
    B = np.zeros((nd, k))
    for g_ in range(nd):
        n2, S2 = n_c.copy(), S_c.copy(); sl = slice(st[g_], en[g_])
        n2[kc[sl]] -= ndc[sl]; S2[kc[sl]] -= Sdc[sl]
        B[g_] = solve(ZZ - ZZd[g_], n2, S2)
    out = {}
    for nm, f in funs.items():
        th = np.array([f(dict(zip(cols, b))) for b in B])
        out[nm] = dict(est=float(f(dict(zip(cols, b0)))),
                       se_jackknife=float(np.sqrt((nd - 1) / nd * np.nansum((th - np.nanmean(th)) ** 2))),
                       n_undefined=int(np.isnan(th).sum()))
    return out


def _lr(c0, c1):
    return lambda b: np.log((b[c0] + b[c1]) / b[c0]) if (b[c0] > 0 and b[c0] + b[c1] > 0) else np.nan


def stacked_frame(Aw, cols, ycol):
    D = []
    for k in MK:
        d = Aw[k][[ycol, "ms", "day"] + cols].copy()
        for c in cols:
            for k2 in MK:
                d[f"{c}__{k2}"] = d[c] if k2 == k else 0.0
        d["fe2"] = k + "_" + d["ms"].astype(str)
        D.append(d.drop(columns=cols))
    return pd.concat(D)


def audit_jackknife(Aw):
    cols = ["d_out", "d_out_bh", "d_in", "d_in_bh"]
    D = stacked_frame(Aw, cols, "R")
    cc = [f"{c}__{k}" for k in MK for c in cols]
    f_t, f_z = _lr("d_out__TRY", "d_out_bh__TRY"), _lr("d_out__ZAR", "d_out_bh__ZAR")
    funs = {"closed_ZAR": lambda b: b["d_out__ZAR"], "incr_ZAR": lambda b: b["d_out_bh__ZAR"],
            "log_ratio_TRY": f_t, "log_ratio_ZAR": f_z, "diff_log_ratio_TRY_minus_ZAR": lambda b: f_t(b) - f_z(b)}
    o = jackknife(D, "R", cc, "fe2", funs)
    d = o["diff_log_ratio_TRY_minus_ZAR"]
    o["t_diff_jackknife"] = float(d["est"] / d["se_jackknife"])
    return o


def flag_days(p, thr):
    b = 1e4 * np.log(p.pl / p.fx_mid); b = b[p.index.dayofweek < 5]
    m = b.index.tz_localize(None).to_period("M")
    d5 = (b - b.groupby(m).transform("median")).abs().resample("5min").mean()
    return sorted(set(d5[d5 > thr].index.date))


def drop_days(p, days):
    p = p.copy(); p.loc[pd.Index(p.index.date).isin(days), "pl"] = np.nan
    return p


def short(e):
    return dict(closed=e["closed"], incr=e["incr"], ratio=e["ratio"], log_ratio=e["log_ratio"], n=e["n"],
                above_band_y=dict(closed=e["above_band"]["spec_y"]["closed"], log_ratio=e["above_band"]["spec_y"]["log_ratio"]),
                below_band_y=dict(closed=e["below_band"]["spec_y"]["closed"], log_ratio=e["below_band"]["spec_y"]["log_ratio"]))


def audit_window(Aw):
    o = {}
    for k in MK:
        prep_R(Aw[k]); o[k] = short(market_estimates(Aw[k]))
        o[k]["pooled_night_day"] = cycle(Aw[k])["pooled"]
    dv = diff_vs_try(Aw)
    o["diff_vs_TRY"] = {s: dv[s]["diff_log_ratio_TRY_minus_ZAR"] for s in ["pooled", "above_band", "below_band"]}
    o["verdict_rule"] = verdict(dict(ZAR=o["ZAR"], diff_vs_TRY=dv))
    return o


AUD = dict(note="Influential-day diagnostics A1-A3; see the comment block above the code that computes them. The baseline and the "
                "verdict above are unchanged.")
log("audit: jackknife")
AUD["A1_jackknife_day"] = dict(primary=audit_jackknife(A), emp_mxn_window=audit_jackknife(A2))
log("audit: jackknife", {w: round(v["t_diff_jackknife"], 2) for w, v in AUD["A1_jackknife_day"].items()})
A2x = {}
for thr in [500, 400, 300]:
    dz, dt_ = flag_days(PRAW["ZAR"], thr), flag_days(PRAW["TRY"], thr)
    Az = build("ZAR", drop_days(PRAW["ZAR"], dz))[0]
    At = build("TRY", drop_days(PRAW["TRY"], dt_))[0]; At = At[At.month.isin(zm)].copy()
    ent = dict(days_ZAR=[str(x) for x in dz], days_TRY=[str(x) for x in dt_])
    for tag, m0 in [("primary", "0000"), ("emp_mxn_window", M_MXN)]:
        ent[tag] = dict(ZAR_only=audit_window({"TRY": A["TRY"][A["TRY"].month >= m0].copy(), "ZAR": Az[Az.month >= m0].copy()}),
                        both=audit_window({"TRY": At[At.month >= m0].copy(), "ZAR": Az[Az.month >= m0].copy()}))
    A2x[f"thr{thr}bp"] = ent
    log("audit: drop", thr, len(dz), "ZAR days;", {t: (round(ent[t]["ZAR_only"]["ZAR"]["ratio"], 2), [round(v, 2) for v in ent[t]["ZAR_only"]["ZAR"]["log_ratio"]],
                                                       round(ent[t]["ZAR_only"]["diff_vs_TRY"]["pooled"]["est"], 2), round(ent[t]["ZAR_only"]["diff_vs_TRY"]["pooled"]["se_stacked"], 2))
                                                   for t in ["primary", "emp_mxn_window"]})
AUD["A2_drop_dislocation_days"] = A2x
A3 = {}
for tag, Aw, Rw in [("primary", A, RES), ("emp_mxn_window", A2, RES2)]:
    E = {}
    for k in MK:
        d = Aw[k][Aw[k].c2ok & Aw[k].x2.notna()].copy(); geom(d, "x2"); prep_R(d, Rw[k]["beta"]); E[k] = d
    r = stack(E, ["d_out", "d_out_bh", "d_in", "d_in_bh"], "ms", "R")
    f = lambda b: _lr("d_out__TRY", "d_out_bh__TRY")(b) - _lr("d_out__ZAR", "d_out_bh__ZAR")(b)
    v, s = dm(r, f)
    A3[tag] = dict(TRY_log_ratio=Rw["TRY"]["q5_eiv_bt2"]["log_ratio"], ZAR_log_ratio=Rw["ZAR"]["q5_eiv_bt2"]["log_ratio"],
                   ZAR_ratio=Rw["ZAR"]["q5_eiv_bt2"]["ratio"], diff_log_ratio_TRY_minus_ZAR=dict(est=v, se_stacked=s, se_indep=indep_se(r, f)))
    log("audit: EIV diff", tag, round(v, 3), round(s, 3))
AUD["A3_eiv_bt2_diff_vs_TRY"] = A3
OUT["audit"] = AUD
json.dump(OUT, open(os.path.join(TAB, f"{NAME}.json"), "w"), indent=1, default=float)
log("json re-written with audit")


# ------------------------------------------------------------------ table
def star(c, s):
    t = abs(c / s) if s > 0 else 0
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def cell(v, d=4):
    return f"${v[0]:.{d}f}{star(v[0], v[1])}$" if np.isfinite(v[0]) else "--"


def secell(v, d=4):
    return f"$({v[1]:.{d}f})$" if np.isfinite(v[1]) else ""


with open(os.path.join(TAB, f"{NAME}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Jan 2024 -- Aug 2026} & \\multicolumn{2}{c}{Nov 2024 -- Aug 2026} \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & Turkey & South Africa & Turkey & South Africa \\\\\n\\midrule\n")
    rows = [("Speed outside band, banks closed", "closed", 4), ("$\\quad\\times$ business hours", "incr", 4),
            ("In-band placebo, banks closed", "d_in", 4), ("$\\quad\\times$ business hours", "d_in_bh", 4),
            ("$\\log$ open/closed ratio", "log_ratio", 2)]
    for lab, key, d in rows:
        vals = [RES["TRY"][key], RES["ZAR"][key], RES2["TRY"][key], RES2["ZAR"][key]]
        fh.write(lab + " & " + " & ".join(cell(v, d) for v in vals) + " \\\\\n")
        fh.write(" & " + " & ".join(secell(v, d) for v in vals) + " \\\\\n")
    fh.write("Open/closed ratio & " + " & ".join(f"{R_[k]['ratio']:.1f}" for R_ in (RES, RES2) for k in MK) + " \\\\\n")
    d1 = RES["diff_vs_TRY"]["pooled"]["diff_log_ratio_TRY_minus_ZAR"]; d2 = RES2["diff_vs_TRY"]["pooled"]["diff_log_ratio_TRY_minus_ZAR"]
    fh.write(f"Turkey $-$ South Africa, $\\log$ ratio & & {cell([d1['est'], d1['se_stacked']], 2)} & & {cell([d2['est'], d2['se_stacked']], 2)} \\\\\n")
    fh.write(f" & & {secell([d1['est'], d1['se_stacked']], 2)} & & {secell([d2['est'], d2['se_stacked']], 2)} \\\\\n")
    fh.write("Night / day speed & " + " & ".join(f"{R_[k]['cycle']['night_over_day']:.2f}" for R_ in (RES, RES2) for k in MK) + " \\\\\n")
    fh.write("SD of premium (bp) & " + " & ".join(f"{R_[k]['sd_prem']:.1f}" for R_ in (RES, RES2) for k in MK) + " \\\\\n")
    fh.write("USDT volume per minute (thousand) & " + " & ".join(f"{R_[k]['vol_per_min']/1e3:.2f}" for R_ in (RES, RES2) for k in MK) + " \\\\\n")
    fh.write("$N$ & " + " & ".join(f"{R_[k]['n']:,}" for R_ in (RES, RES2) for k in MK) + " \\\\\n\\midrule\n")
    # influential-day rows: jackknife s.e. of the difference and the log ratio without dislocation days
    j1, j2 = AUD["A1_jackknife_day"]["primary"], AUD["A1_jackknife_day"]["emp_mxn_window"]
    fh.write("Audit: Turkey $-$ South Africa, day-jackknife s.e. & & " + secell([0, j1["diff_log_ratio_TRY_minus_ZAR"]["se_jackknife"]], 2)
             + " & & " + secell([0, j2["diff_log_ratio_TRY_minus_ZAR"]["se_jackknife"]], 2) + " \\\\\n")
    for thr in [500, 300]:
        e = A2x[f"thr{thr}bp"]
        vals = [e[t]["ZAR_only"][k]["log_ratio"] for t in ("primary", "emp_mxn_window") for k in MK]
        fh.write(f"Audit: $\\log$ ratio, ZAR days with $|b|$ dislocation $>{thr}$ bp dropped ({len(e['days_ZAR'])}) & " + " & ".join(cell(v, 2) for v in vals) + " \\\\\n")
        fh.write(" & " + " & ".join(secell(v, 2) for v in vals) + " \\\\\n")
        dd_ = [e[t]["ZAR_only"]["diff_vs_TRY"]["pooled"] for t in ("primary", "emp_mxn_window")]
        fh.write(f"\\quad Turkey $-$ South Africa & & {cell([dd_[0]['est'], dd_[0]['se_stacked']], 2)} & & {cell([dd_[1]['est'], dd_[1]['se_stacked']], 2)} \\\\\n")
        fh.write(f" & & {secell([dd_[0]['est'], dd_[0]['se_stacked']], 2)} & & {secell([dd_[1]['est'], dd_[1]['se_stacked']], 2)} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
for k, col, mk, lab, off in [("TRY", C1, "o", "Turkey (same months)", 0.0), ("ZAR", C2, "s", "South Africa", 0.3)]:
    c = pd.DataFrame(RES[k]["cycle"]["blocks"])
    ax[0].errorbar(c.h + off, c.k, yerr=1.96 * c.se, fmt=mk + "-", ms=3.5, color=col, ecolor=col, lw=0.9, alpha=0.9, label=lab)
ax[0].axvspan(9, 17, color=C3, alpha=0.12); ax[0].axhline(0, color="k", lw=0.4)
ax[0].set_xlabel("local hour"); ax[0].set_ylabel(r"reversion speed outside the band, $\hat\kappa$")
ax[0].set_xticks(range(0, 25, 4)); ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[0].set_title("(a) Speed by two-hour block", fontsize=9, loc="left")
bh_ = pd.DataFrame(Q["binance_by_local_hour_weekdays"]); dh_ = pd.DataFrame(Q["dukascopy_by_local_hour_weekdays"])
ax[1].plot(bh_.lh + 0.5, bh_.share_minutes_with_trades, "s-", ms=3, color=C2, lw=1.0, label="Binance: minute has a trade")
ax[1].plot(dh_.lh + 0.5, dh_.share_minutes_bid_changed, "o-", ms=3, color=C1, lw=1.0, label="Dukascopy: bid changes")
ax[1].axvspan(9, 17, color=C3, alpha=0.12); ax[1].set_ylim(0, 1.02)
ax[1].set_xlabel("local hour"); ax[1].set_ylabel("share of weekday minutes"); ax[1].set_xticks(range(0, 25, 4))
ax[1].legend(frameon=False, fontsize=7.5, loc="lower center")
ax[1].set_title("(b) USDT/ZAR and USD/ZAR activity", fontsize=9, loc="left")
ax[2].plot(dh_.lh + 0.5, dh_.median_spread_bp, "o-", ms=3, color=C1, lw=1.0)
ax[2].axvspan(9, 17, color=C3, alpha=0.12); ax[2].set_ylim(0, None)
ax[2].set_xlabel("local hour"); ax[2].set_ylabel("median bid-ask spread (bp)"); ax[2].set_xticks(range(0, 25, 4))
ax[2].set_title("(c) Dukascopy USD/ZAR spread", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{NAME}.pdf")); plt.close(fig)
log("done")

# ------------------------------------------------------------------ console summary
for tag, R_ in [("primary", RES), ("emp_mxn window", RES2)]:
    print("==", tag)
    for k in MK:
        r = R_[k]
        print(k, "closed", np.round(r["closed"], 4), "incr", np.round(r["incr"], 4), "ratio", round(r["ratio"], 2),
              "logratio", np.round(r["log_ratio"], 3), "n", r["n"], "night/day", round(r["cycle"]["night_over_day"], 3))
        print("   above", {s: np.round(r["above_band"][s]["log_ratio"], 3).tolist() for s in ["spec_y", "spec_R"]},
              "below", {s: np.round(r["below_band"][s]["log_ratio"], 3).tolist() for s in ["spec_y", "spec_R"]})
        print("   thin10", np.round(r["q4_thin10"]["closed"], 4), np.round(r["q4_thin10"]["log_ratio"], 3),
              "eiv", np.round(r["q5_eiv_bt2"]["closed"], 4), np.round(r["q5_eiv_bt2"]["log_ratio"], 3), "ac1", r["noise_ac1_dy"])
    print("diff TRY-ZAR", R_["diff_vs_TRY"]["pooled"]["diff_log_ratio_TRY_minus_ZAR"])
    print("verdict", verdict(R_), flags(R_))
print("Q6", json.dumps(Q6, default=float)[:1500]); print("vs MXN", RES2["diff_vs_MXN"]); print("validation", RES2["TRY_validation_vs_emp_mxn"])
