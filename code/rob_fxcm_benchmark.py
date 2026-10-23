"""Robustness: a second, independent dollar benchmark (FXCM USD/TRY one-minute candles).

The main dollar benchmark is Dukascopy SWFX, an aggregator. FXCM publishes
one-minute bid/ask candles for USD/TRY (data_raw/fxcm/, from code/download_fxcm.py). Time stamps
are UTC bar-start times (weeks open Sunday 22:00 UTC in winter and 21:00 UTC in summer, i.e. 17:00
New York, and close Friday 21:59 / 20:59 UTC). The FXCM mid is (BidClose + AskClose) / 2, the
analogue of the close-based Dukascopy mid in build_panel.load_fx.

TESTS
 (0) Time-stamp check: correlation of one-minute log changes of the two mids at leads/lags of
     -3..+3 minutes (weekdays, rollover window excluded). Expect the peak at lag 0.
 (a) Benchmark comparison, on five-minute intervals in which both aligned mids are available on
     >= 2 minutes (same filters as below): correlation of five-minute log changes of the two mids
     (five-minute means of the mid); SD in bp of the Dukascopy-minus-FXCM log difference (five-
     minute means), raw and net of month means, and SD of its five-minute change; correlation of
     the two five-minute premium changes. All overall and by banking state (Istanbul 09-17 open
     vs closed). Also the FXCM relative bid-ask spread by state.
 (b) Replace the Dukascopy mid by the FXCM mid in code/empirics_main.py with every filter and
     step unchanged: ffill <= 2 minutes onto Binance minutes, weekdays only, 20:30-22:30 UTC
     excluded, |b| < 1000 bp, five-minute means with >= 2 valid minutes, consecutive intervals,
     OFI standardized within month, monthly bands profiled on a 1 bp grid with 5% trimming,
     months with >= 500 intervals. Re-estimate
       - Table emp_main columns (2)-(4): threshold model, in-band reversion term, BH interactions;
       - Table emp_state column (1): bands by month x banking state (cells >= 300 intervals),
         month x state fixed effects, slopes interacted with the banks-open indicator;
       - Table emp_legs: local leg (dlog USDT/TRY) and global leg (dlog FXCM USD/TRY), plus the
         BH-interacted legs of Table emp_quality columns (3)-(4).
     Headline quantities: the BH increment in kappa+ and the open/closed ratio (delta method),
     the in-band coefficient, and the leg coefficients with omega.
 (c) The same estimates on the common sample of five-minute intervals that are valid under both
     benchmarks, for both benchmarks (bands re-profiled on the common sample), so that the
     comparison isolates the benchmark from the sample.
 The Dukascopy pipeline is re-run here and checked against tables/emp_numbers.json.
 Coverage decomposition (stored separately in the JSON): Dukascopy re-estimated
 on the calendar days covered by FXCM, and interval counts by Istanbul two-hour block, to split the
 own-sample vs common-sample difference into calendar and intraday coverage.
 Ratio rows in the .tex: stars test log(open/closed) = 0 (delta method); s.e. is that of the ratio.
No sup-LR bootstrap here; the summed sup-LR statistic is reported.
Inputs: data_raw/panel_TRY.parquet, data_raw/dukascopy, data_raw/fxcm, tables/emp_numbers.json.
Outputs: tables/rob_fxcm_benchmark.json, tables/rob_fxcm_benchmark.tex.
Run: python -P code/rob_fxcm_benchmark.py   (-P: no script/cwd path injection; data_raw/fxcm holds downloaded files)
"""
import os, sys, glob, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from model import _ssr_grid
from build_panel import load_fx
warnings.filterwarnings("ignore")

RAW, TAB = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables"))
FREQ, START, END = "5min", "2024-01-01", "2026-08-31"
OUT = dict(notes="FXCM USD/TRY one-minute bid/ask candles as an alternative benchmark; see docstring of code/rob_fxcm_benchmark.py")


# ------------------------------------------------------------------ data
def load_fxcm(pair="USDTRY"):
    dfs = []
    for f in sorted(glob.glob(os.path.join(RAW, "fxcm", f"{pair}_*.csv.gz"))):
        if os.path.getsize(f) == 0: continue
        d = pd.read_csv(f, compression="gzip")
        d.columns = [c.strip() for c in d.columns]
        if len(d) == 0: continue
        d.index = pd.to_datetime(d.DateTime.str.strip(), format="%m/%d/%Y %H:%M:%S.%f", utc=True)
        dfs.append(d.drop(columns="DateTime").astype(float))
    d = pd.concat(dfs).sort_index()
    d = d[~d.index.duplicated()]
    d = d[(d.BidClose > 0) & (d.AskClose >= d.BidClose)]
    m = pd.DataFrame(index=d.index)
    m["fx_mid"] = (d.BidClose + d.AskClose) / 2
    m["fx_spread"] = (d.AskClose - d.BidClose) / m.fx_mid
    m["fxv"] = 1.0           # FXCM has no volume: count of quoted minutes
    return m


P0 = pd.read_parquet(os.path.join(RAW, "panel_TRY.parquet")).drop(columns=["fx_mid", "fx_spread", "fxv"], errors="ignore")
P0 = P0.loc[START:END]
FX = {"duk": load_fx("USDTRY"), "fxcm": load_fxcm("USDTRY")}
OUT["fxcm_raw"] = dict(first=str(FX["fxcm"].index.min()), last=str(FX["fxcm"].index.max()),
                       n_minutes=int(len(FX["fxcm"])),
                       files=len(glob.glob(os.path.join(RAW, "fxcm", "USDTRY_*.csv.gz"))),
                       files_empty_or_404=int(sum(os.path.getsize(f) < 200 for f in glob.glob(os.path.join(RAW, "fxcm", "USDTRY_*.csv.gz")))))


def minute_panel(fx):
    """Exactly the minute-level steps of empirics_main.py."""
    p = P0.join(fx.reindex(P0.index, method="ffill", limit=2))
    p["b"] = 1e4 * np.log(p.p_usdt / p.fx_mid)
    hr = p.index.hour + p.index.minute / 60
    p.loc[(hr >= 20.5) & (hr < 22.5), "b"] = np.nan
    p.loc[p.index.dayofweek >= 5, "b"] = np.nan
    p.loc[p.b.abs() > 1000, "b"] = np.nan
    return p


def five_min(p):
    g = p.resample(FREQ)
    a = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum(), "vol": g.vol.sum(), "fxv": g.fxv.sum(),
                      "pl": np.log(g.p_usdt.mean()), "pg": np.log(g.fx_mid.mean()), "spr": g.fx_spread.mean()})
    a = a[a.nb >= 2]
    naive = a.index.tz_localize(None)
    a["month"] = naive.to_period("M").astype(str)
    a["q"] = a.ofi / a.ofi.groupby(a.month).transform("std")
    a["dpl"] = 1e4 * a.pl.diff(); a["dpg"] = 1e4 * a.pg.diff()
    consec = a.index.to_series().diff() == pd.Timedelta(FREQ)
    a["y"] = a.b.diff(); a["x"] = a.b.shift(1)
    a = a[consec & a.y.notna() & a.x.notna() & a.q.notna()].copy()
    a["day"] = a.index.date
    ih = (a.index.hour + 3) % 24
    a["bh"] = ((ih >= 9) & (ih < 17)).astype(float)
    return a


# ------------------------------------------------------------------ estimators (copied conventions)
def profile_ul(y, x, q, trim=0.05, step=1.0):
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[i], grid[j], S.min()


def ols_fe(d, ycol, cols, fe="month"):
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    gm = d.groupby(fe)
    Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(zip(cols, b)), dict(zip(cols, np.sqrt(np.diag(V)))), n, V, cols


def ratio(r, k, kb):
    """open/closed ratio (k+kb)/k and log ratio, delta-method s.e. from the joint covariance."""
    i, j = r[4].index(k), r[4].index(kb)
    V = r[3][np.ix_([i, j], [i, j])]
    c, d = r[0][k], r[0][kb]
    R = (c + d) / c; gR = np.array([-d / c ** 2, 1 / c])
    L = np.log((c + d) / c) if c > 0 and c + d > 0 else np.nan
    gL = np.array([1 / (c + d) - 1 / c, 1 / (c + d)])
    return dict(ratio=float(R), ratio_se=float(np.sqrt(gR @ V @ gR)), log_ratio=float(L), log_ratio_se=float(np.sqrt(gL @ V @ gL)))


def pk(r):
    return {k: [float(r[0][k]), float(r[1][k])] for k in r[0]} | {"n": int(r[2])}


def monthly_bands(a):
    bands, slr = {}, 0.0
    for per, d in a.groupby("month"):
        if len(d) < 500: continue
        y, x, q = d.y.values, d.x.values, d.q.values
        u, l, smin = profile_ul(y, x, q)
        X = np.column_stack([np.ones(len(y)), q, -x]); c = np.linalg.lstsq(X, y, rcond=None)[0]
        s0 = np.sum((y - X @ c) ** 2)
        lr = len(y) * np.log(s0 / smin); slr += lr
        bands[per] = (float(l), float(u))
    a = a.copy()
    a["l"] = a.month.map(lambda k: bands.get(k, (np.nan, np.nan))[0])
    a["u"] = a.month.map(lambda k: bands.get(k, (np.nan, np.nan))[1])
    a = a[a.l.notna()].copy()
    a["up"] = -np.maximum(a.x - a.u, 0); a["dn"] = np.maximum(a.l - a.x, 0)
    a["exc"] = np.maximum(a.x - a.u, 0) + np.maximum(a.l - a.x, 0)
    a["inband"] = (a.exc == 0).astype(float)
    return a, bands, slr


def estimate(a0, label):
    """emp_main (2)-(4), emp_legs, emp_quality (3)-(4), emp_state (1) on a five-minute panel."""
    a, bands, slr = monthly_bands(a0)
    R = dict(n=int(len(a)), n_days=int(a.day.nunique()), n_months=len(bands), sum_supLR=float(slr),
             share_inband=float(a.inband.mean()),
             band_width_median=float(np.median([u - l for l, u in bands.values()])),
             bands={k: list(v) for k, v in bands.items()})
    thr = ols_fe(a, "y", ["q", "up", "dn"])
    a["negx_in"] = -(a.x - (a.l + a.u) / 2) * a.inband
    thr_in = ols_fe(a, "y", ["q", "up", "dn", "negx_in"])
    for c in ["q", "up", "dn"]:
        a[c + "_bh"] = a[c] * a.bh
    cols = ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"]
    bh = ols_fe(a, "y", cols)
    R["thr"], R["thr_in"], R["bh"] = pk(thr), pk(thr_in), pk(bh)
    R["bh_ratio_kp"] = ratio(bh, "up", "up_bh"); R["bh_ratio_km"] = ratio(bh, "dn", "dn_bh")
    R["inband_ci95_upper"] = float(thr_in[0]["negx_in"] + 1.96 * thr_in[1]["negx_in"])
    # legs
    dd = a.dropna(subset=["dpl", "dpg"])
    dl, dg = ols_fe(dd, "dpl", ["q", "up", "dn"]), ols_fe(dd, "dpg", ["q", "up", "dn"])
    R["legs"] = {"local": pk(dl), "global": pk(dg)}
    kl_p, kg_p, kl_m, kg_m = dl[0]["up"], -dg[0]["up"], dl[0]["dn"], -dg[0]["dn"]
    R["omega_p"], R["omega_m"] = float(kg_p / (kl_p + kg_p)), float(kg_m / (kl_m + kg_m))
    R["legs_bh"] = {"local": pk(ols_fe(dd, "dpl", cols)), "global": pk(ols_fe(dd, "dpg", cols))}
    # state bands (emp_state column 1)
    a["state"] = np.where(a.bh == 1, "open", "closed")
    sb = {}
    for (m, st), d in a.groupby(["month", "state"]):
        if len(d) < 300: continue
        u, l, _ = profile_ul(d.y.values, d.x.values, d.q.values)
        sb[(m, st)] = (l, u)
    a["l_s"] = [sb.get((m, s), (np.nan, np.nan))[0] for m, s in zip(a.month, a.state)]
    a["u_s"] = [sb.get((m, s), (np.nan, np.nan))[1] for m, s in zip(a.month, a.state)]
    s = a[a.l_s.notna()].copy()
    s["up_s"] = -np.maximum(s.x - s.u_s, 0); s["dn_s"] = np.maximum(s.l_s - s.x, 0)
    for c in ["q", "up_s", "dn_s"]:
        s[c + "_bh"] = s[c] * s.bh
    s["ms"] = s.month + "_" + s.state
    rs = ols_fe(s, "y", ["q", "up_s", "dn_s", "q_bh", "up_s_bh", "dn_s_bh"], fe="ms")
    B = pd.DataFrame([(m, st, l, u) for (m, st), (l, u) in sb.items()], columns=["month", "state", "l", "u"])
    B["w"] = B.u - B.l
    wide = B.pivot(index="month", columns="state", values="w").dropna()
    diff = wide.closed - wide.open
    R["state"] = dict(coef=pk(rs), ratio_kp=ratio(rs, "up_s", "up_s_bh"),
                      width_open_mean=float(wide.open.mean()), width_closed_mean=float(wide.closed.mean()),
                      width_open_median=float(wide.open.median()), width_closed_median=float(wide.closed.median()),
                      diff_mean=float(diff.mean()), diff_se=float(diff.std() / np.sqrt(len(diff))), n_months=int(len(diff)))
    print(f"[{label}] n={R['n']} thr up={thr[0]['up']:.4f}({thr[1]['up']:.4f}) dn={thr[0]['dn']:.4f}  "
          f"BH: up={bh[0]['up']:.4f}({bh[1]['up']:.4f}) up_bh={bh[0]['up_bh']:.4f}({bh[1]['up_bh']:.4f}) "
          f"ratio={R['bh_ratio_kp']['ratio']:.2f}  in-band={thr_in[0]['negx_in']:.4f}({thr_in[1]['negx_in']:.4f})  "
          f"state up_s_bh={rs[0]['up_s_bh']:.4f}({rs[1]['up_s_bh']:.4f})  global up={dg[0]['up']:.4f}({dg[1]['up']:.4f})", flush=True)
    return R


# ------------------------------------------------------------------ build both panels
MP = {k: minute_panel(v) for k, v in FX.items()}
A = {k: five_min(v) for k, v in MP.items()}

# (0) time-stamp check on the raw one-minute mids
grid = pd.date_range(P0.index.min(), P0.index.max(), freq="1min")
lm = {k: np.log(v.fx_mid.reindex(grid, method="ffill", limit=2)) for k, v in FX.items()}
hr = grid.hour + grid.minute / 60
ok = (grid.dayofweek < 5) & ~((hr >= 20.5) & (hr < 22.5))
d1 = {k: v.diff().where(ok) for k, v in lm.items()}
OUT["lead_lag_corr_1min"] = {str(L): float(d1["duk"].corr(d1["fxcm"].shift(-L))) for L in range(-3, 4)}
print("lead/lag corr (fxcm shifted by L minutes ahead):", OUT["lead_lag_corr_1min"], flush=True)

# (a) benchmark comparison on five-minute intervals, both mids available on >= 2 valid minutes
m1 = pd.DataFrame({"duk": MP["duk"].fx_mid, "fxcm": MP["fxcm"].fx_mid, "spr_fxcm": MP["fxcm"].fx_spread,
                   "spr_duk": MP["duk"].fx_spread,
                   "bd": MP["duk"].b, "bf": MP["fxcm"].b})
m1 = m1[m1.bd.notna() & m1.bf.notna()]                     # both premia valid under the common filters
g = m1.resample(FREQ)
c5 = pd.DataFrame({"n": g.duk.count(), "ld": np.log(g.duk.mean()), "lf": np.log(g.fxcm.mean()),
                   "bd": g.bd.mean(), "bf": g.bf.mean(), "spr": g.spr_fxcm.mean()})
c5 = c5[c5.n >= 2]
cons = c5.index.to_series().diff() == pd.Timedelta(FREQ)
c5["dld"] = 1e4 * c5.ld.diff(); c5["dlf"] = 1e4 * c5.lf.diff()
c5["dbd"] = c5.bd.diff(); c5["dbf"] = c5.bf.diff()
c5["gap"] = 1e4 * (c5.ld - c5.lf)
c5["month"] = c5.index.tz_localize(None).to_period("M").astype(str)
c5["gap_dm"] = c5.gap - c5.gap.groupby(c5.month).transform("mean")
c5["dgap"] = c5.gap.diff()
c5.loc[~cons, ["dld", "dlf", "dbd", "dbf", "dgap"]] = np.nan
ihh = (c5.index.hour + 3) % 24
c5["state"] = np.where((ihh >= 9) & (ihh < 17), "open", "closed")


def comp(d):
    dc = d.dropna(subset=["dld", "dlf"])
    return dict(n=int(len(d)), n_changes=int(len(dc)), corr_dlog=float(dc.dld.corr(dc.dlf)),
                corr_dprem=float(dc.dbd.corr(dc.dbf)),
                sd_dlog_duk=float(dc.dld.std()), sd_dlog_fxcm=float(dc.dlf.std()),
                mean_gap_bp=float(d.gap.mean()), sd_gap_bp=float(d.gap.std()), sd_gap_net_month_bp=float(d.gap_dm.std()),
                sd_dgap_bp=float(dc.dgap.std()), fxcm_spread_bp_median=float(1e4 * d.spr.median()),
                corr_level_prem=float(d.bd.corr(d.bf)))


OUT["comparison"] = {"all": comp(c5), "open": comp(c5[c5.state == "open"]), "closed": comp(c5[c5.state == "closed"])}
OUT["comparison"]["by_year"] = {str(y): comp(c5[c5.index.year == y]) for y in (2024, 2025, 2026)}
OUT["coverage"] = {k: dict(n_5min=int(len(v)), months=int(v.month.nunique())) for k, v in A.items()}
OUT["coverage"]["fxcm_quoted_minutes_share_of_duk"] = float(len(FX["fxcm"].loc[START:END]) / len(FX["duk"].loc[START:END]))
print(json.dumps(OUT["comparison"], indent=1), flush=True)

# (b) own samples; (c) common sample
OUT["duk"] = estimate(A["duk"], "Dukascopy, replication")
OUT["fxcm"] = estimate(A["fxcm"], "FXCM, own sample")
common = A["duk"].index.intersection(A["fxcm"].index)
OUT["duk_common"] = estimate(A["duk"].loc[common], "Dukascopy, common sample")
OUT["fxcm_common"] = estimate(A["fxcm"].loc[common], "FXCM, common sample")

# ------------------------------------------------------------------ coverage decomposition
# The Dukascopy open/closed ratio falls from 4.2 on its own sample to 2.7 on the common sample.
# Split that sample change into (i) calendar coverage (FXCM weeks missing: Dukascopy restricted to
# days on which FXCM has data) and (ii) intraday coverage (intervals without FXCM quotes).
fx_days = set(A["fxcm"].day)
OUT["posthoc"] = {"duk_fxcm_days": estimate(A["duk"][A["duk"].day.isin(fx_days)], "Dukascopy, FXCM days only")}
blk = lambda ix: pd.Series(1, ix).groupby(((ix.hour + 3) % 24) // 2 * 2).sum()
cov = pd.DataFrame({"duk": blk(A["duk"].index), "duk_fxcm_days": blk(A["duk"][A["duk"].day.isin(fx_days)].index),
                    "common": blk(common)})
cov["share_kept_common_of_fxcm_days"] = cov.common / cov.duk_fxcm_days
OUT["posthoc"]["intervals_by_istanbul_2h_block"] = {str(k): v for k, v in cov.round(4).T.to_dict().items()}
print(cov.to_string(), flush=True)

# replication check against the published numbers
pub = json.load(open(os.path.join(TAB, "emp_numbers.json")))
OUT["replication_check"] = dict(n_pub=pub["n_obs"], n_rerun=OUT["duk"]["n"],
                                up_bh_pub=pub["reg"]["bh"]["coef"]["up_bh"], up_bh_rerun=OUT["duk"]["bh"]["up_bh"][0],
                                negx_in_pub=pub["reg"]["thr_in"]["coef"]["negx_in"], negx_in_rerun=OUT["duk"]["thr_in"]["negx_in"][0])
json.dump(OUT, open(os.path.join(TAB, "rob_fxcm_benchmark.json"), "w"), indent=1, default=float)


# ------------------------------------------------------------------ LaTeX table
def star(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def cell(v, nd=4):
    st = star(v[0], v[1]) if len(v) == 2 else star(v[2], 1.0)     # ratios: test of log ratio = 0
    return f"${v[0]:.{nd}f}{st}$", f"$({v[1]:.{nd}f})$"


specs = [("duk", "Dukascopy"), ("fxcm", "FXCM"), ("duk_common", "Dukascopy"), ("fxcm_common", "FXCM")]
rows = [("\\emph{Monthly bands, Table~\\ref{tab:emp_main}}", None, None),
        ("$\\kappa^+$ (col.\\ 2)", lambda R: R["thr"]["up"], 4),
        ("$\\kappa^-$ (col.\\ 2)", lambda R: R["thr"]["dn"], 4),
        ("In-band reversion (col.\\ 3)", lambda R: R["thr_in"]["negx_in"], 4),
        ("$\\kappa^+$, banks closed (col.\\ 4)", lambda R: R["bh"]["up"], 4),
        ("$\\quad\\times$ banks open", lambda R: R["bh"]["up_bh"], 4),
        ("Open/closed ratio, $\\kappa^+$", lambda R: [R["bh_ratio_kp"]["ratio"], R["bh_ratio_kp"]["ratio_se"], R["bh_ratio_kp"]["log_ratio"] / R["bh_ratio_kp"]["log_ratio_se"]], 2),
        ("OFI$_t$ (col.\\ 4)", lambda R: R["bh"]["q"], 3),
        ("\\emph{State bands, Table~\\ref{tab:emp_state} col.\\ (1)}", None, None),
        ("$\\kappa^+$, banks closed", lambda R: R["state"]["coef"]["up_s"], 4),
        ("$\\quad\\times$ banks open", lambda R: R["state"]["coef"]["up_s_bh"], 4),
        ("Open/closed ratio, $\\kappa^+$", lambda R: [R["state"]["ratio_kp"]["ratio"], R["state"]["ratio_kp"]["ratio_se"], R["state"]["ratio_kp"]["log_ratio"] / R["state"]["ratio_kp"]["log_ratio_se"]], 2),
        ("\\emph{Legs, Table~\\ref{tab:emp_legs}}", None, None),
        ("Local leg, $-(b_{t-1}-u_m)_+$", lambda R: R["legs"]["local"]["up"], 4),
        ("Global leg, $-(b_{t-1}-u_m)_+$", lambda R: R["legs"]["global"]["up"], 4),
        ("Local leg, $(\\ell_m-b_{t-1})_+$", lambda R: R["legs"]["local"]["dn"], 4),
        ("Global leg, $(\\ell_m-b_{t-1})_+$", lambda R: R["legs"]["global"]["dn"], 4),
        ("Global leg, OFI$_t$", lambda R: R["legs"]["global"]["q"], 3)]
with open(os.path.join(TAB, "rob_fxcm_benchmark.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Own sample} & \\multicolumn{2}{c}{Common sample} \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & " + " & ".join(n for _, n in specs) + " \\\\\n\\midrule\n")
    for lab, f, nd in rows:
        if f is None:
            fh.write(f"{lab} & & & & \\\\\n"); continue
        cs = [cell(f(OUT[k]), nd) for k, _ in specs]
        fh.write(lab + " & " + " & ".join(c[0] for c in cs) + " \\\\\n & " + " & ".join(c[1] for c in cs) + " \\\\\n")
    fh.write("\\midrule\n$\\omega$, premium side & " + " & ".join(f"{OUT[k]['omega_p']:.2f}" for k, _ in specs) + " \\\\\n")
    fh.write("Median band width (bp) & " + " & ".join(f"{OUT[k]['band_width_median']:.0f}" for k, _ in specs) + " \\\\\n")
    fh.write("$N$ & " + " & ".join(f"{OUT[k]['n']:,}" for k, _ in specs) + " \\\\\n\\bottomrule\n\\end{tabular}\n")
print(json.dumps({k: OUT[k] for k in ("replication_check", "coverage", "lead_lag_corr_1min")}, indent=1, default=float))
