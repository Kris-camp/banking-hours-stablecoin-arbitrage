"""rob_size_dependence: is the business-hours increase in the speed of arbitrage state-dependent, and did the
13 Nov 2025 venue change remove it?

HYPOTHESIS (from the capacity model). Pre-positioned balances and capped FAST transfers can close small dislocations
when EFT is shut; large dislocations need EFT. Then the business-hours (BH) increase in the speed of arbitrage above
the band should be larger for larger dislocations and in periods of larger dislocations. A competing explanation of
the small 2026 increment is the venue change of 13 Nov 2025, when Binance restricted its lira (TRY) spot books to
verified Binance TR users and a designated lira liquidity provider; remaining open orders of other users were
cancelled after 27 Nov 2025.

CONTEXT. Above the band after 13 Nov 2025 the sample has 10,970 intervals at 0-10 bp, 5,283 at 10-30, 250 at 30-60
and none beyond 60. rob_by_period.py finds the 2026 rise in the closed-hours speed concentrated in 00-09 Istanbul.

DESIGN
------
Common. data_raw/a5.parquet (monthly bands l, u; 162,624 five-minute intervals). y = change in the premium (bp),
q = OFI standardized within month, BH = bh (09:00-17:00 Istanbul clock). Month fixed effects, standard errors
clustered by UTC day with the finite-sample correction of ols_fe, conditional on the bands. e = (x-u)+ is the excess
above the band, f = (l-x)+ the excess below it; up = -e, dn = f. P = 1{t >= 2025-11-13 00:00 UTC} (T0, a Thursday).
"Pre-venue" = P = 0. Baseline (emp_main col 4): y on [q, q*BH, up, up*BH, dn, dn*BH], month FE.

(1) Piecewise speeds by distance from the band edge. Piecewise-linear spline in the excess with knots at 10, 30 and
    60 bp: s1 = min(e,10), s2 = min((e-10)+,20), s3 = min((e-30)+,30), s4 = (e-60)+. Regressors -s_k and -s_k*BH, so
    each coefficient is the marginal speed per bp within segment k (closed) and its BH increment; below the band the
    same basis g_k of f enters with a plus sign. q and q*BH as in the baseline. Report closed_k, open_k = closed_k +
    incr_k and incr_k for k = 1..4 on each side.
    Tests (each side): (i) Wald test that the four increments are equal (3 df); (ii) linear trend L = sum_k w_k incr_k,
    w = (-1.5,-0.5,0.5,1.5)/5 (the OLS slope of the increment on segment rank), H1: L > 0, one-sided 5%;
    (iii) incr_4 - incr_1. Also the implied average speed A(e) = sum_k kappa_k len_k(e)/e at e = 5, 20, 45 and 90 bp
    (the share of a dislocation of size e closed per interval), closed, open and increment.
    Variants: (1a) full sample [primary]; (1b) pre-venue sample; (1c) bin-average speeds: regressors -e*1{e in bin k}
    and their BH interactions, bins (0,10], (10,30], (30,60], >60 (proportional reversion of dislocations in each
    bin), full sample; (1d) state bands (l_s, u_s by month x banking state, month x state FE, emp_state col 1).
    Classification (1), stored in the JSON: above the band the increment "rises with distance" if in (1a) L > 0 is significant at 5%
    one-sided AND incr_4 > incr_1; it "falls with distance" if L < 0 is significant at 5% one-sided; otherwise "no
    evidence of a rise". (1b)-(1d) are checks; the JSON reports in how many of them L > 0 is significant at 5%
    one-sided.

(2) Predetermined state variables (no selection on outcomes).
    Z_a = within-month SD of the five-minute premium b in the previous calendar month (January 2024 has no previous
          month in the panel and is dropped from regressions using Z_a);
    Z_b = realized volatility of the interbank rate on the previous trading day (the previous UTC date in the panel):
          sqrt(mean(dpg^2)) over that day's intervals, dpg = five-minute change of the log interbank mid (bp);
    Z_c = maximum distance outside the monthly band (exc) over the previous trading day's intervals.
    Each is standardized (z-score) over the full-sample intervals where it is available; the same standardization is
    used in every subsample.
    Spec: baseline + [Z, up*Z, up*BH*Z, dn*Z, dn*BH*Z] (Z itself omitted for Z_a, which month FE absorb). Key
    coefficient: up*BH*Z, the change in the BH increment of kappa+ per one SD of Z (H1: > 0, one-sided). Also up*Z
    (closed speed), up*Z + up*BH*Z (open speed) and the dn analogues.
    Variants: (2i) full sample [primary]; (2ii) pre-venue sample; (2iii) full sample with venue controls (all six
    baseline slopes also x P, FE month x P); (2iv) the three Z jointly on their common sample; (2v) logs (log Z_a,
    log Z_b, log(1+Z_c)), then standardized, full sample; (2vi) pooled price side of emp_flow col (1):
    R = -s(y - beta*q) on [d_out, d_out*bh_eff, d_in, d_in*bh_eff] + [Z, d_out*Z, d_out*bh_eff*Z], state bands,
    month FE, beta from emp_numbers.json; key d_out*bh_eff*Z; (2vii) spline control: (2i) with up/dn replaced by the
    spline of (1) and its BH interactions, Z interacted with the linear up, up*BH, dn, dn*BH (state dependence net of
    the nonlinearity in the current distance); (2viii) binary Z_a: previous-month SD above vs below its median across
    the pre-venue months (Feb 2024 - Nov 2025), all baseline slopes interacted with high and low, pre-venue sample;
    report the BH increment of kappa+ in each group and their difference.
    Classification (2): state dependence is "supported" if up*BH*Z > 0 is significant at 5% one-sided for at least two
    of Z_a, Z_b, Z_c in BOTH (2i) and (2ii); "mixed" if at least one is positive and significant at 5% one-sided in
    (2i) or (2ii) but the first condition fails; "not supported" otherwise.

(3) The venue change of 13 Nov 2025.
    (3a) Weekly event study: event week k = floor((t - T0)/7 days), k = -8..7 (18 Sep 2025 - 7 Jan 2026). y on
         [q, q*BH, dn, dn*BH] + up x 1{week k} x (1-BH) + up x 1{week k} x BH for every k; FE month x P (Nov 2025 split
         at T0); monthly bands. Week x state cells with fewer than 20 above-band intervals are pooled into one
         residual column and their speed is not reported. Summary: equal-weighted mean of the closed (open) weekly
         speeds after minus before, with s.e.
    (3b) Pre/post, 2025-07-01 to 2026-03-31: the baseline with every slope also x P; FE month x P. Report kappa+
         closed, open and increment before and after, and the changes; kappa- the same.
         (3b-seg) Like-for-like by distance: spline above the band with knots 10 and 30 (segments 0-10, 10-30, >30;
         merged above 30 because only 250 post-T0 above-band intervals exceed 30 bp), x BH x P, dn and dn*BH x P
         linear. Changes in the closed speed and increment of the 0-10 and 10-30 segments and their equal-weight
         average. Same on the full sample (pre = Jan 2024 - 12 Nov 2025): (3b-seg-full).
         (3b-legs) (3b) with the five-minute change in the log local price (dpl) and in the log interbank rate (dpg)
         as dependent variables: kappa_L = coefficient on up in dpl, kappa_G = minus that in dpg.
         (3b-band) (3b) with the November 2025 band re-profiled separately on 1-12 Nov and 13-30 Nov (profile of
         empirics_main.py: 1 bp grid, 5% trimming, control q).
    (3c) Large-trade response, emp_flow col (2): A_L = -s z_large (x100) on [d_out, d_out*bh_eff, d_in, d_in*bh_eff],
         each also x P; state bands; FE month x P; 2025-07-01 to 2026-03-31. Report theta closed and open before and
         after and the changes; the price-side R analogue on the same sample; secondary: d_out split into above and
         below the band.
    (3d) Abruptness. Candidate break dates D = every Thursday 00:00 UTC with [D-56d, D+56d) inside the sample. For
         each D, the (3b) specification on [D-56d, D+56d) with P_D and FE month x P_D. Statistics: change in the closed
         kappa+ (up x P_D), change in the increment (up*BH x P_D), their t-statistics. Placebo set: |D - T0| >= 56
         days. One-sided randomization p-values: p_closed = (1 + #{placebo change in closed >= actual})/(1 + N),
         p_incr = (1 + #{placebo change in increment <= actual})/(1 + N), and the same for t-statistics. Local profile:
         D = T0 + k weeks, k = -8..8; the break is "located at T0" if the largest t-statistic of the change in the
         closed speed over this profile is at |k| <= 1.
    Classification (3): the venue-change explanation is "supported" if in (3b) the closed speed rises and the increment
    falls, both significant at 5% two-sided, AND the 0-30 bp like-for-like increment change (3b-seg) is negative and
    significant at 5%, AND p_closed <= 0.10 AND the break is located at T0. "Consistent with a shift, not
    distinguishable from drift" if the (3b) and (3b-seg) conditions hold but the placebo or location condition fails.
    "Not supported" if the (3b) increment change is not significantly negative.

(4) Overall. State dependence is "supported" if (1) rises AND (2) is supported; "partial" if exactly one holds;
    "not supported" otherwise. The fall of the increment after T0 is "not explained by smaller dislocations" if the
    like-for-like 0-30 bp increment change in (3b-seg-full) is negative and significant at 5%.

(5) Additional descriptive checks A1-A4 (see the comments at the end of the code).

Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, data_raw/ofi_size/*.parquet, tables/emp_numbers.json.
Outputs: tables/rob_size_dependence.json, tables/rob_size_dependence.tex, figures/rob_size_dependence.pdf.
Run: python -P code/rob_size_dependence.py
"""
import os, sys, glob, json, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import _ssr_grid
warnings.filterwarnings("ignore")

RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
NAME = "size_dependence"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
T_START = time.time()
T0 = pd.Timestamp("2025-11-13", tz="UTC")
W0, W1 = pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")
OUT = dict(name=NAME, T0=str(T0), window_prepost=["2025-07-01", "2026-03-31"])

# ------------------------------------------------------------------ data
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
a = a.join(pd.read_parquet(os.path.join(RAW, "state_bands.parquet")), how="inner")
assert len(a) == 162624
a["day"] = a.index.date
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
a["hol"] = np.isin(a.day, pd.to_datetime(HOL).date).astype(float)
a["bh_eff"] = a.bh * (1 - a.hol)
a["P"] = (a.index >= T0).astype(float)
a["e"] = np.maximum(a.x - a.u, 0.0)
a["f"] = np.maximum(a.l - a.x, 0.0)
a["state"] = np.where(a.bh == 1, "o", "c")
a["ms"] = a.month + "_" + a.state
a["mP"] = a.month + "_" + a.P.astype(int).astype(str)
a["z_large"] = a.ofi_large / a.ofi_large.groupby(a.month).transform("std")
BETA = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
# state-band variables (emp_flow / emp_state conventions)
mid_s = (a.l_s + a.u_s) / 2
a["s_s"] = np.where(a.x > a.u_s, 1.0, np.where(a.x < a.l_s, -1.0, np.sign(a.x - mid_s)))
out_s = (a.x > a.u_s) | (a.x < a.l_s)
a["d_out"] = np.maximum(a.x - a.u_s, 0) + np.maximum(a.l_s - a.x, 0)
a["d_in"] = np.where(out_s, 0.0, np.abs(a.x - mid_s))
a["d_up"] = np.maximum(a.x - a.u_s, 0); a["d_dn"] = np.maximum(a.l_s - a.x, 0)
a["R"] = -a.s_s * (a.y - BETA * a.q)
a["A_L"] = -a.s_s * a.z_large
a["e_s"] = np.maximum(a.x - a.u_s, 0.0); a["f_s"] = np.maximum(a.l_s - a.x, 0.0)
for c in ["q", "up", "dn"]:
    a[c + "_bh"] = a[c] * a.bh
for c in ["d_out", "d_in", "d_up", "d_dn"]:
    a[c + "_bh"] = a[c] * a.bh_eff
BASE = ["q", "q_bh", "up", "up_bh", "dn", "dn_bh"]


# ------------------------------------------------------------------ estimation helpers
def fit(d, ycol, cols, fe="month"):
    """OLS with one-way FE (within) and day-clustered s.e. with the finite-sample correction of ols_fe."""
    Y = d[ycol].to_numpy(float); X = d[cols].to_numpy(float)
    if fe:
        g, _ = pd.factorize(d[fe]); cnt = np.bincount(g).astype(float)
        Y = Y - (np.bincount(g, Y) / cnt)[g]
        X = X - np.column_stack([np.bincount(g, X[:, j]) / cnt for j in range(X.shape[1])])[g]
    XtX = X.T @ X
    if np.linalg.cond(XtX) > 1e12:
        raise np.linalg.LinAlgError("near-singular design")
    XtXi = np.linalg.inv(XtX); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]; ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    G = np.zeros((ng, k)); np.add.at(G, codes, X * e[:, None])
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=b, V=V, cols=list(cols), idx={c: i for i, c in enumerate(cols)}, n=int(n), ng=int(ng))


def lc(r, w):
    g = np.zeros(len(r["b"]))
    for k, v in w.items():
        g[r["idx"][k]] += v
    return float(g @ r["b"]), float(np.sqrt(g @ r["V"] @ g))


def wald(r, ws):
    R = np.zeros((len(ws), len(r["b"])))
    for i, w in enumerate(ws):
        for k, v in w.items():
            R[i, r["idx"][k]] += v
    rb = R @ r["b"]; chi2 = float(rb @ np.linalg.solve(R @ r["V"] @ R.T, rb))
    return dict(chi2=chi2, df=len(ws), p=float(st.chi2.sf(chi2, len(ws))))


def es(est_se):
    e_, s_ = est_se
    return dict(est=e_, se=s_, z=e_ / s_ if s_ > 0 else np.nan, p2=float(2 * st.norm.sf(abs(e_ / s_))) if s_ > 0 else np.nan,
                p1_pos=float(st.norm.sf(e_ / s_)) if s_ > 0 else np.nan)


def seg_basis(d, var, knots, name, sign):
    edges = [0.0] + list(knots) + [np.inf]
    cols = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        c = f"{name}{k + 1}"
        d[c] = sign * np.clip(d[var].to_numpy() - lo, 0, hi - lo)
        d[c + "_bh"] = d[c] * d.bh
        cols.append(c)
    return cols


def seg_len(evalue, knots):
    edges = [0.0] + list(knots) + [np.inf]
    return np.array([np.clip(evalue - lo, 0, hi - lo) for lo, hi in zip(edges[:-1], edges[1:])])


SEGLAB = ["0-10", "10-30", "30-60", ">60"]
KN = [10, 30, 60]
WTR = np.array([-1.5, -0.5, 0.5, 1.5]) / 5


# ------------------------------------------------------------------ (1) piecewise speeds
def piecewise(d, evar="e", fvar="f", fe="month"):
    d = d.copy()
    us = seg_basis(d, evar, KN, "us", -1.0)
    ds = seg_basis(d, fvar, KN, "ds", 1.0)
    cols = ["q", "q_bh"] + us + [c + "_bh" for c in us] + ds + [c + "_bh" for c in ds]
    r = fit(d, "y", cols, fe)
    res = {"n": r["n"], "n_days": r["ng"]}
    for side, sc, var in [("above", us, evar), ("below", ds, fvar)]:
        cnt = {SEGLAB[k]: dict(closed=int(((d[sc[k]] != 0) & (d.bh == 0)).sum()), open=int(((d[sc[k]] != 0) & (d.bh == 1)).sum()))
               for k in range(4)}
        segs = {}
        for k, c in enumerate(sc):
            segs[SEGLAB[k]] = dict(closed=lc(r, {c: 1}), open=lc(r, {c: 1, c + "_bh": 1}), incr=lc(r, {c + "_bh": 1}),
                                   n_intervals_reaching=cnt[SEGLAB[k]])
        tr = es(lc(r, {c + "_bh": w for c, w in zip(sc, WTR)}))
        d41 = es(lc(r, {sc[3] + "_bh": 1, sc[0] + "_bh": -1}))
        wd = wald(r, [{sc[k] + "_bh": 1, sc[0] + "_bh": -1} for k in (1, 2, 3)])
        avg = {}
        for ev in [5, 20, 45, 90]:
            L = seg_len(ev, KN) / ev
            avg[str(ev)] = dict(closed=lc(r, {c: L[k] for k, c in enumerate(sc)}),
                                open=lc(r, dict([(c, L[k]) for k, c in enumerate(sc)] + [(c + "_bh", L[k]) for k, c in enumerate(sc)])),
                                incr=lc(r, {c + "_bh": L[k] for k, c in enumerate(sc)}))
        res[side] = dict(segments=segs, trend=tr, incr4_minus_incr1=d41, wald_equal_incr=wd, avg_speed=avg)
    return res


def binavg(d, fe="month"):
    d = d.copy()
    edges = [0, 10, 30, 60, np.inf]
    cols = ["q", "q_bh"]
    for side, var, sign in [("ub", "e", -1.0), ("db", "f", 1.0)]:
        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            c = f"{side}{k + 1}"
            d[c] = sign * d[var] * ((d[var] > lo) & (d[var] <= hi))
            d[c + "_bh"] = d[c] * d.bh
            cols += [c, c + "_bh"]
    r = fit(d, "y", cols, fe)
    res = {}
    for side, nm in [("above", "ub"), ("below", "db")]:
        segs = {SEGLAB[k]: dict(closed=lc(r, {f"{nm}{k+1}": 1}), open=lc(r, {f"{nm}{k+1}": 1, f"{nm}{k+1}_bh": 1}),
                                incr=lc(r, {f"{nm}{k+1}_bh": 1})) for k in range(4)}
        tr = es(lc(r, {f"{nm}{k+1}_bh": WTR[k] for k in range(4)}))
        d41 = es(lc(r, {f"{nm}4_bh": 1, f"{nm}1_bh": -1}))
        wd = wald(r, [{f"{nm}{k+1}_bh": 1, f"{nm}1_bh": -1} for k in (1, 2, 3)])
        res[side] = dict(segments=segs, trend=tr, incr4_minus_incr1=d41, wald_equal_incr=wd)
    res["n"] = r["n"]
    return res


print("(1) piecewise speeds ...", flush=True)
P1 = {}
P1["1a_full"] = piecewise(a)
P1["1b_prevenue"] = piecewise(a[a.P == 0])
P1["1c_binavg_full"] = binavg(a)
P1["1d_statebands_full"] = piecewise(a, "e_s", "f_s", fe="ms")
OUT["piecewise"] = P1
tr = P1["1a_full"]["above"]["trend"]; d41 = P1["1a_full"]["above"]["incr4_minus_incr1"]["est"]
if tr["est"] > 0 and tr["p1_pos"] < 0.05 and d41 > 0:
    rule1 = "rises with distance"
elif tr["est"] < 0 and (1 - tr["p1_pos"]) < 0.05:
    rule1 = "falls with distance"
else:
    rule1 = "no evidence of a rise"
n_checks = sum(int(P1[k]["above"]["trend"]["est"] > 0 and P1[k]["above"]["trend"]["p1_pos"] < 0.05)
               for k in ["1b_prevenue", "1c_binavg_full", "1d_statebands_full"])
rule1_below = ("rises with distance" if (P1["1a_full"]["below"]["trend"]["p1_pos"] < 0.05 and
                                          P1["1a_full"]["below"]["incr4_minus_incr1"]["est"] > 0) else
               "falls with distance" if (1 - P1["1a_full"]["below"]["trend"]["p1_pos"]) < 0.05 else "no evidence of a rise")
OUT["reading_rule_1"] = dict(above=rule1, below=rule1_below, n_checks_trend_pos_sig_of_3=n_checks)
print(" rule 1:", OUT["reading_rule_1"], flush=True)

# ------------------------------------------------------------------ (2) state variables
print("(2) state variables ...", flush=True)
# Z_a: previous calendar month's within-month SD of the five-minute premium
sd_m = a.groupby("month").b.std()
prev_m = {m: str(pd.Period(m, "M") - 1) for m in sd_m.index}
a["Za_raw"] = a.month.map(lambda m: sd_m.get(prev_m[m], np.nan))
# Z_b, Z_c: previous trading day (previous UTC date in the panel)
days = np.array(sorted(a.day.unique()))
rv_d = a.groupby("day").dpg.apply(lambda s: np.sqrt(np.mean(s.to_numpy() ** 2)))
mx_d = a.groupby("day").exc.max()
prev_day = dict(zip(days[1:], days[:-1]))
a["Zb_raw"] = a.day.map(lambda dd: rv_d.get(prev_day.get(dd), np.nan))
a["Zc_raw"] = a.day.map(lambda dd: mx_d.get(prev_day.get(dd), np.nan))
a["Za_log"] = np.log(a.Za_raw); a["Zb_log"] = np.log(a.Zb_raw); a["Zc_log"] = np.log1p(a.Zc_raw)
ZINFO = {}
for v in ["Za", "Zb", "Zc"]:
    for t in ["raw", "log"]:
        s = a[f"{v}_{t}"]
        a[f"{v}_{t}_z"] = (s - s.mean()) / s.std()
        ZINFO[f"{v}_{t}"] = dict(mean=float(s.mean()), sd=float(s.std()), n_nonmissing=int(s.notna().sum()),
                                 p50=float(s.median()), p99=float(s.quantile(0.99)), max=float(s.max()))
OUT["state_var_info"] = ZINFO
OUT["state_var_corr"] = a[["Za_raw_z", "Zb_raw_z", "Zc_raw_z"]].corr().round(3).to_dict()
DAILY = {"Za": False, "Zb": True, "Zc": True}


def add_z_inter(d, zc, terms):
    cols = []
    for t in terms:
        c = f"{t}__{zc}"
        d[c] = d[t] * d[zc]; cols.append(c)
    return cols


def state_reg(d, zlist, venue=False, spline=False, fe="month"):
    d = d.dropna(subset=zlist).copy()
    cols = list(BASE)
    if spline:
        us = seg_basis(d, "e", KN, "us", -1.0); ds = seg_basis(d, "f", KN, "ds", 1.0)
        cols = ["q", "q_bh"] + us + [c + "_bh" for c in us] + ds + [c + "_bh" for c in ds]
    if venue:
        for c in BASE:
            d[c + "_P"] = d[c] * d.P
        cols += [c + "_P" for c in BASE]
        fe = "mP"
    for zc in zlist:
        if DAILY[zc[:2]]:
            cols.append(zc)
        cols += add_z_inter(d, zc, ["up", "up_bh", "dn", "dn_bh"])
    r = fit(d, "y", cols, fe)
    res = {"n": r["n"], "n_days": r["ng"]}
    for zc in zlist:
        res[zc] = dict(incr_kp=es(lc(r, {f"up_bh__{zc}": 1})), closed_kp=es(lc(r, {f"up__{zc}": 1})),
                       open_kp=es(lc(r, {f"up__{zc}": 1, f"up_bh__{zc}": 1})),
                       incr_km=es(lc(r, {f"dn_bh__{zc}": 1})), closed_km=es(lc(r, {f"dn__{zc}": 1})))
    if not spline:
        res["base_incr_kp_at_Z0"] = lc(r, {"up_bh": 1}); res["base_closed_kp_at_Z0"] = lc(r, {"up": 1})
    return res


def state_reg_R(d, zc):
    d = d.dropna(subset=[zc]).copy()
    cols = ["d_out", "d_out_bh", "d_in", "d_in_bh"] + ([zc] if DAILY[zc[:2]] else [])
    d[f"d_out__{zc}"] = d.d_out * d[zc]; d[f"d_out_bh__{zc}"] = d.d_out_bh * d[zc]
    cols += [f"d_out__{zc}", f"d_out_bh__{zc}"]
    r = fit(d, "R", cols)
    return dict(incr=es(lc(r, {f"d_out_bh__{zc}": 1})), closed=es(lc(r, {f"d_out__{zc}": 1})),
                base_incr_at_Z0=lc(r, {"d_out_bh": 1}), n=r["n"])


P2 = {}
for v in ["Za", "Zb", "Zc"]:
    zc = f"{v}_raw_z"
    P2[v] = {"2i_full": state_reg(a, [zc])[zc], "2ii_prevenue": state_reg(a[a.P == 0], [zc])[zc],
             "2iii_venue_controls": state_reg(a, [zc], venue=True)[zc],
             "2v_log_full": state_reg(a, [f"{v}_log_z"])[f"{v}_log_z"],
             "2vi_priceside_R": state_reg_R(a, zc),
             "2vii_spline_control": state_reg(a, [zc], spline=True)[zc]}
    print(" ", v, {k: round(P2[v][k]["incr_kp"]["est"] if "incr_kp" in P2[v][k] else P2[v][k]["incr"]["est"], 5)
                   for k in P2[v]}, flush=True)
jt = state_reg(a, ["Za_raw_z", "Zb_raw_z", "Zc_raw_z"])
P2["2iv_joint"] = jt
jt2 = state_reg(a[a.P == 0], ["Za_raw_z", "Zb_raw_z", "Zc_raw_z"])
P2["2iv_joint_prevenue"] = jt2
# n for the single-variable regressions
P2["n"] = {v: dict(full=state_reg(a, [f"{v}_raw_z"])["n"], prevenue=state_reg(a[a.P == 0], [f"{v}_raw_z"])["n"])
           for v in ["Za", "Zb", "Zc"]}
# (2viii) binary previous-month SD, pre-venue sample
pre = a[(a.P == 0)].dropna(subset=["Za_raw"]).copy()
mlev = pre.groupby("month").Za_raw.first()
med = float(mlev.median())
pre["H"] = (pre.Za_raw > med).astype(float)
cols = []
for c in BASE:
    pre[c + "_H"] = pre[c] * pre.H; pre[c + "_L"] = pre[c] * (1 - pre.H); cols += [c + "_H", c + "_L"]
rb = fit(pre, "y", cols)
P2["2viii_binary_prevmonth_sd"] = dict(median_prev_sd=med, n_high_months=int((mlev > med).sum()), n_low_months=int((mlev <= med).sum()),
                                       incr_high=lc(rb, {"up_bh_H": 1}), incr_low=lc(rb, {"up_bh_L": 1}),
                                       closed_high=lc(rb, {"up_H": 1}), closed_low=lc(rb, {"up_L": 1}),
                                       diff_high_minus_low=es(lc(rb, {"up_bh_H": 1, "up_bh_L": -1})), n=rb["n"])
OUT["state"] = P2


def sig1(x):
    return x["est"] > 0 and x["p1_pos"] < 0.05


n_full = sum(sig1(P2[v]["2i_full"]["incr_kp"]) for v in ["Za", "Zb", "Zc"])
n_pre = sum(sig1(P2[v]["2ii_prevenue"]["incr_kp"]) for v in ["Za", "Zb", "Zc"])
rule2 = "supported" if (n_full >= 2 and n_pre >= 2) else ("mixed" if (n_full >= 1 or n_pre >= 1) else "not supported")
OUT["reading_rule_2"] = dict(n_sig_pos_full=int(n_full), n_sig_pos_prevenue=int(n_pre), verdict=rule2)
print(" rule 2:", OUT["reading_rule_2"], flush=True)

# ------------------------------------------------------------------ (3) venue change
print("(3) venue change ...", flush=True)


def prepost(d, ycol="y", fe="mP", pcol="P"):
    d = d.copy()
    for c in BASE:
        d[c + "_P"] = d[c] * d[pcol]
    r = fit(d, ycol, BASE + [c + "_P" for c in BASE], fe)
    o = {"n": r["n"], "n_days": r["ng"]}
    for nm, c in [("kp", "up"), ("km", "dn")]:
        o[nm] = dict(closed_pre=lc(r, {c: 1}), open_pre=lc(r, {c: 1, c + "_bh": 1}), incr_pre=lc(r, {c + "_bh": 1}),
                     closed_post=lc(r, {c: 1, c + "_P": 1}), open_post=lc(r, {c: 1, c + "_bh": 1, c + "_P": 1, c + "_bh_P": 1}),
                     incr_post=lc(r, {c + "_bh": 1, c + "_bh_P": 1}),
                     d_closed=es(lc(r, {c + "_P": 1})), d_open=es(lc(r, {c + "_P": 1, c + "_bh_P": 1})),
                     d_incr=es(lc(r, {c + "_bh_P": 1})))
    o["beta"] = dict(closed_pre=lc(r, {"q": 1}), closed_post=lc(r, {"q": 1, "q_P": 1}))
    up = d.e > 0
    o["n_above"] = {f"{s}_{p}": int((up & (d.bh == (s == "open")) & (d[pcol] == (p == "post"))).sum())
                    for s in ["closed", "open"] for p in ["pre", "post"]}
    return o, r


win = a[(a.index >= W0) & (a.index < W1)].copy()
P3 = {}
P3["3b_prepost"], _ = prepost(win)
print("  3b kp:", {k: (round(v["est"], 5), round(v["se"], 5)) for k, v in P3["3b_prepost"]["kp"].items() if k.startswith("d_")}, flush=True)
P3["3b_prepost_fullsample"], _ = prepost(a)


# like-for-like by distance
def seg_prepost(d, fe="mP"):
    d = d.copy()
    us = seg_basis(d, "e", [10, 30], "us", -1.0)
    base = ["q", "q_bh"] + us + [c + "_bh" for c in us] + ["dn", "dn_bh"]
    for c in base:
        d[c + "_P"] = d[c] * d.P
    r = fit(d, "y", base + [c + "_P" for c in base], fe)
    lab = ["0-10", "10-30", ">30"]
    o = {"n": r["n"]}
    for k, c in enumerate(us):
        o[lab[k]] = dict(closed_pre=lc(r, {c: 1}), closed_post=lc(r, {c: 1, c + "_P": 1}),
                         incr_pre=lc(r, {c + "_bh": 1}), incr_post=lc(r, {c + "_bh": 1, c + "_bh_P": 1}),
                         open_pre=lc(r, {c: 1, c + "_bh": 1}), open_post=lc(r, {c: 1, c + "_bh": 1, c + "_P": 1, c + "_bh_P": 1}),
                         d_closed=es(lc(r, {c + "_P": 1})), d_incr=es(lc(r, {c + "_bh_P": 1})),
                         n_reaching={f"{s}_{p}": int(((d[c] != 0) & (d.bh == (s == "open")) & (d.P == (p == "post"))).sum())
                                     for s in ["closed", "open"] for p in ["pre", "post"]})
    o["avg_0_30"] = dict(incr_pre=es(lc(r, {us[0] + "_bh": .5, us[1] + "_bh": .5})),
                         incr_post=es(lc(r, {us[0] + "_bh": .5, us[1] + "_bh": .5, us[0] + "_bh_P": .5, us[1] + "_bh_P": .5})),
                         d_incr=es(lc(r, {us[0] + "_bh_P": .5, us[1] + "_bh_P": .5})),
                         closed_pre=es(lc(r, {us[0]: .5, us[1]: .5})),
                         closed_post=es(lc(r, {us[0]: .5, us[1]: .5, us[0] + "_P": .5, us[1] + "_P": .5})),
                         d_closed=es(lc(r, {us[0] + "_P": .5, us[1] + "_P": .5})))
    return o


P3["3b_seg"] = seg_prepost(win)
P3["3b_seg_full"] = seg_prepost(a)
print("  3b-seg avg0-30:", {k: (round(v["est"], 5), round(v["se"], 5)) for k, v in P3["3b_seg"]["avg_0_30"].items()}, flush=True)
print("  3b-seg-full avg0-30:", {k: (round(v["est"], 5), round(v["se"], 5)) for k, v in P3["3b_seg_full"]["avg_0_30"].items()}, flush=True)

# legs
legs = {}
for nm, yc, sgn in [("local", "dpl", 1.0), ("interbank", "dpg", -1.0)]:
    o, r = prepost(win.dropna(subset=[yc]), ycol=yc)
    legs[nm] = dict(closed_pre=tuple(np.array(lc(r, {"up": 1})) * [sgn, 1]),
                    closed_post=tuple(np.array(lc(r, {"up": 1, "up_P": 1})) * [sgn, 1]),
                    open_pre=tuple(np.array(lc(r, {"up": 1, "up_bh": 1})) * [sgn, 1]),
                    open_post=tuple(np.array(lc(r, {"up": 1, "up_bh": 1, "up_P": 1, "up_bh_P": 1})) * [sgn, 1]),
                    d_closed=es(tuple(np.array(lc(r, {"up_P": 1})) * [sgn, 1])),
                    d_incr=es(tuple(np.array(lc(r, {"up_bh_P": 1})) * [sgn, 1])))
legs["note"] = "kappa_L = coefficient on up in dpl; kappa_G = minus the coefficient on up in dpg"
P3["3b_legs"] = legs


# re-profiled November 2025 band
def profile(y, x, q, step=1.0, TRIM=0.05):
    lo_, hi_ = np.percentile(x, [100 * TRIM, 100 * (1 - TRIM)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return float(grid[i]), float(grid[j])


wb = win.copy()
nov = wb.month == "2025-11"
bands_nov = {}
for p_ in [0.0, 1.0]:
    m = nov & (wb.P == p_)
    dd = wb[m]
    u_, l_ = profile(dd.y.to_numpy(float), dd.x.to_numpy(float), dd.q.to_numpy(float))
    bands_nov["pre" if p_ == 0 else "post"] = dict(l=l_, u=u_, n=int(m.sum()))
    wb.loc[m, "u"] = u_; wb.loc[m, "l"] = l_
bands_nov["published"] = dict(l=float(win.loc[nov, "l"].iloc[0]), u=float(win.loc[nov, "u"].iloc[0]))
wb["e"] = np.maximum(wb.x - wb.u, 0.0); wb["up"] = -wb.e; wb["dn"] = np.maximum(wb.l - wb.x, 0.0)
wb["up_bh"] = wb.up * wb.bh; wb["dn_bh"] = wb.dn * wb.bh
P3["3b_band_reprofiled"], _ = prepost(wb)
P3["3b_band_reprofiled"]["bands_nov2025"] = bands_nov


# flow side
def flow_prepost(d, ycol, regs):
    d = d.dropna(subset=[ycol]).copy()
    cols = []
    for c in regs:
        d[c + "_P"] = d[c] * d.P
    cols = regs + [c + "_P" for c in regs]
    r = fit(d, ycol, cols, "mP")
    o = {"n": r["n"]}
    for c in [c for c in regs if not c.endswith("_bh")]:
        o[c] = dict(closed_pre=lc(r, {c: 1}), open_pre=lc(r, {c: 1, c + "_bh": 1}), incr_pre=lc(r, {c + "_bh": 1}),
                    closed_post=lc(r, {c: 1, c + "_P": 1}), open_post=lc(r, {c: 1, c + "_bh": 1, c + "_P": 1, c + "_bh_P": 1}),
                    incr_post=lc(r, {c + "_bh": 1, c + "_bh_P": 1}),
                    d_closed=es(lc(r, {c + "_P": 1})), d_open=es(lc(r, {c + "_P": 1, c + "_bh_P": 1})),
                    d_incr=es(lc(r, {c + "_bh_P": 1})))
    return o


win["A_L100"] = win.A_L * 100
P3["3c_largetrade"] = flow_prepost(win, "A_L100", ["d_out", "d_out_bh", "d_in", "d_in_bh"])
P3["3c_priceside_R"] = flow_prepost(win, "R", ["d_out", "d_out_bh", "d_in", "d_in_bh"])
P3["3c_largetrade_bysides"] = flow_prepost(win, "A_L100", ["d_up", "d_up_bh", "d_dn", "d_dn_bh", "d_in", "d_in_bh"])
print("  3c theta d_out:", {k: tuple(np.round(v if isinstance(v, tuple) else (v["est"], v["se"]), 4))
                             for k, v in P3["3c_largetrade"]["d_out"].items()}, flush=True)

# (3a) weekly event study
ev = a[(a.index >= T0 - pd.Timedelta(days=56)) & (a.index < T0 + pd.Timedelta(days=56))].copy()
ev["wk"] = np.floor((ev.index - T0) / pd.Timedelta(days=7)).astype(int)
cols = ["q", "q_bh", "dn", "dn_bh"]
ev["up_rest"] = 0.0
cells = {}
for k in range(-8, 8):
    for s, sv in [("closed", 0.0), ("open", 1.0)]:
        m = (ev.wk == k) & (ev.bh == sv)
        nab = int((m & (ev.e > 0)).sum())
        c = f"upw_{k}_{s}"
        if nab >= 20:
            ev[c] = ev.up * m; cols.append(c); cells[(k, s)] = nab
        else:
            ev.loc[m, "up_rest"] = ev.loc[m, "up"]; cells[(k, s)] = -nab
if (ev.up_rest != 0).any():
    cols.append("up_rest")
rev = fit(ev, "y", cols, "mP")
EVS = []
for k in range(-8, 8):
    row = dict(week=k, start=str((T0 + pd.Timedelta(days=7 * k)).date()))
    for s in ["closed", "open"]:
        c = f"upw_{k}_{s}"
        row[s] = lc(rev, {c: 1}) if c in rev["idx"] else None
        row["n_above_" + s] = abs(cells[(k, s)])
    if row["closed"] is not None and row["open"] is not None:
        row["incr"] = lc(rev, {f"upw_{k}_open": 1, f"upw_{k}_closed": -1})
    EVS.append(row)


def mean_change(state):
    pre_ = [f"upw_{k}_{state}" for k in range(-8, 0) if f"upw_{k}_{state}" in rev["idx"]]
    post_ = [f"upw_{k}_{state}" for k in range(0, 8) if f"upw_{k}_{state}" in rev["idx"]]
    w = {c: -1 / len(pre_) for c in pre_}; w.update({c: 1 / len(post_) for c in post_})
    return dict(pre=lc(rev, {c: 1 / len(pre_) for c in pre_}), post=lc(rev, {c: 1 / len(post_) for c in post_}),
                change=es(lc(rev, w)), n_weeks_pre=len(pre_), n_weeks_post=len(post_))


P3["3a_event_weekly"] = dict(weeks=EVS, closed=mean_change("closed"), open=mean_change("open"), n=rev["n"],
                             n_cells_pooled=int(sum(1 for v in cells.values() if v <= 0)))
print("  3a closed:", P3["3a_event_weekly"]["closed"]["change"], " open:", P3["3a_event_weekly"]["open"]["change"], flush=True)

# (3d) abruptness: placebo break dates
print("  (3d) placebo breaks ...", flush=True)
first = pd.Timestamp("2024-01-02", tz="UTC"); last = pd.Timestamp("2026-09-01", tz="UTC")
D = pd.Timestamp("2024-01-04", tz="UTC")
cands = []
while D + pd.Timedelta(days=56) <= last:
    if D - pd.Timedelta(days=56) >= first:
        cands.append(D)
    D += pd.Timedelta(days=7)
PB = []
idx_all = a.index
for D in cands:
    m = (idx_all >= D - pd.Timedelta(days=56)) & (idx_all < D + pd.Timedelta(days=56))
    d = a[m].copy()
    d["PD"] = (d.index >= D).astype(float)
    d["mPD"] = d.month + "_" + d.PD.astype(int).astype(str)
    cnt = {f"{s}_{p}": int(((d.e > 0) & (d.bh == sv) & (d.PD == pv)).sum()) for s, sv in [("c", 0), ("o", 1)] for p, pv in [("pre", 0), ("post", 1)]}
    rec = dict(date=str(D.date()), weeks_from_T0=int(round((D - T0).days / 7)), min_cell=int(min(cnt.values())))
    if min(cnt.values()) < 20:
        rec["skipped"] = "fewer than 20 above-band intervals in a state x period cell"; PB.append(rec); continue
    try:
        o, _ = prepost(d, fe="mPD", pcol="PD")
    except np.linalg.LinAlgError:
        rec["skipped"] = "singular"; PB.append(rec); continue
    k = o["kp"]
    rec.update(d_closed=k["d_closed"]["est"], se_closed=k["d_closed"]["se"], t_closed=k["d_closed"]["z"],
               d_incr=k["d_incr"]["est"], se_incr=k["d_incr"]["se"], t_incr=k["d_incr"]["z"],
               d_open=k["d_open"]["est"], se_open=k["d_open"]["se"])
    PB.append(rec)
pbd = pd.DataFrame(PB)
ok = pbd[pbd.get("skipped").isna()] if "skipped" in pbd else pbd
act = ok[ok.weeks_from_T0 == 0].iloc[0]
plc = ok[ok.weeks_from_T0.abs() >= 8]
NP = len(plc)
pv = dict(
    p_closed=float((1 + (plc.d_closed >= act.d_closed).sum()) / (1 + NP)),
    p_closed_t=float((1 + (plc.t_closed >= act.t_closed).sum()) / (1 + NP)),
    p_incr=float((1 + (plc.d_incr <= act.d_incr).sum()) / (1 + NP)),
    p_incr_t=float((1 + (plc.t_incr <= act.t_incr).sum()) / (1 + NP)),
    p_closed_abs=float((1 + (plc.d_closed.abs() >= abs(act.d_closed)).sum()) / (1 + NP)),
    p_incr_abs=float((1 + (plc.d_incr.abs() >= abs(act.d_incr)).sum()) / (1 + NP)))
loc_ = ok[ok.weeks_from_T0.abs() <= 8].sort_values("weeks_from_T0")
k_max_closed = int(loc_.loc[loc_.t_closed.idxmax(), "weeks_from_T0"])
k_min_incr = int(loc_.loc[loc_.t_incr.idxmin(), "weeks_from_T0"])
P3["3d_abrupt"] = dict(actual=dict(d_closed=[float(act.d_closed), float(act.se_closed)], d_incr=[float(act.d_incr), float(act.se_incr)],
                                   d_open=[float(act.d_open), float(act.se_open)], t_closed=float(act.t_closed), t_incr=float(act.t_incr)),
                       n_candidates=len(pbd), n_estimated=len(ok), n_placebo=int(NP), pvalues=pv,
                       placebo_quantiles_d_closed={q_: float(plc.d_closed.quantile(q_)) for q_ in [0.05, 0.5, 0.9, 0.95]},
                       placebo_quantiles_d_incr={q_: float(plc.d_incr.quantile(q_)) for q_ in [0.05, 0.1, 0.5, 0.95]},
                       local_profile=loc_[["weeks_from_T0", "date", "d_closed", "t_closed", "d_incr", "t_incr"]].to_dict("records"),
                       local_argmax_t_closed_week=k_max_closed, local_argmin_t_incr_week=k_min_incr,
                       located_at_T0=bool(abs(k_max_closed) <= 1),
                       top5_placebo_d_closed=plc.nlargest(5, "d_closed")[["date", "d_closed", "t_closed"]].to_dict("records"),
                       all=pbd.to_dict("records"))
print("  3d:", pv, "argmax k", k_max_closed, flush=True)
OUT["venue"] = P3

kp = P3["3b_prepost"]["kp"]
c1 = kp["d_closed"]["est"] > 0 and kp["d_closed"]["p2"] < 0.05
c2 = kp["d_incr"]["est"] < 0 and kp["d_incr"]["p2"] < 0.05
c3 = P3["3b_seg"]["avg_0_30"]["d_incr"]["est"] < 0 and P3["3b_seg"]["avg_0_30"]["d_incr"]["p2"] < 0.05
c4 = pv["p_closed"] <= 0.10
c5 = P3["3d_abrupt"]["located_at_T0"]
if c1 and c2 and c3 and c4 and c5:
    rule3 = "supported"
elif c1 and c2 and c3:
    rule3 = "consistent with a shift, not distinguishable from drift"
elif not c2:
    rule3 = "not supported"
else:
    rule3 = "partial: increment falls but other conditions fail"
OUT["reading_rule_3"] = dict(closed_rises_sig=bool(c1), incr_falls_sig=bool(c2), like_for_like_0_30_falls_sig=bool(c3),
                             placebo_p_closed_le_010=bool(c4), located_at_T0=bool(c5), verdict=rule3)
sd_ok = rule1 == "rises with distance"; st_ok = rule2 == "supported"
rule4 = "supported" if (sd_ok and st_ok) else ("partial" if (sd_ok or st_ok) else "not supported")
lf = P3["3b_seg_full"]["avg_0_30"]["d_incr"]
OUT["reading_rule_4"] = dict(state_dependence=rule4,
                             post_T0_fall_not_explained_by_smaller_dislocations=bool(lf["est"] < 0 and lf["p2"] < 0.05))
print("rules:", OUT["reading_rule_3"], OUT["reading_rule_4"], flush=True)


# ------------------------------------------------------------------ table
def f4(v, nd=4):
    return "" if v is None else f"${v[0]:.{nd}f}$"


def s4(v, nd=4):
    return "" if v is None else f"$({v[1]:.{nd}f})$"


def tup(x):
    return (x["est"], x["se"]) if isinstance(x, dict) else x


def star(v):
    v = tup(v); t = abs(v[0] / v[1])
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def cellpair(v, nd=4):
    v = tup(v)
    return f"${v[0]:.{nd}f}{star(v)}$", f"$({v[1]:.{nd}f})$"


L = []
L.append("\\begin{tabular}{lcccccc}\n\\toprule\n")
L.append("\\multicolumn{7}{l}{\\emph{Panel A. Marginal speed by distance outside the band (piecewise-linear spline), full sample}} \\\\\n")
L.append(" & \\multicolumn{3}{c}{Above the band} & \\multicolumn{3}{c}{Below the band} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
L.append("Distance (bp) & Closed & Open & Increment & Closed & Open & Increment \\\\\n\\midrule\n")
pa = P1["1a_full"]
for sg in SEGLAB:
    top, bot = [], []
    for side in ["above", "below"]:
        for k in ["closed", "open", "incr"]:
            t_, b_ = cellpair(pa[side]["segments"][sg][k]); top.append(t_); bot.append(b_)
    sgl = {"0-10": "0--10", "10-30": "10--30", "30-60": "30--60", ">60": "$>$60"}[sg]
    L.append(f"{sgl} & " + " & ".join(top) + " \\\\\n & " + " & ".join(bot) + " \\\\\n")
top, bot = [], []
for side in ["above", "below"]:
    t_, b_ = cellpair(pa[side]["trend"]); top += ["", "", t_]; bot += ["", "", b_]
L.append("Trend in increment (per segment) & " + " & ".join(top) + " \\\\\n & " + " & ".join(bot) + " \\\\\n")
top, bot = [], []
for side in ["above", "below"]:
    t_, b_ = cellpair(pa[side]["incr4_minus_incr1"]); top += ["", "", t_]; bot += ["", "", b_]
L.append("Increment $>$60 minus 0--10 & " + " & ".join(top) + " \\\\\n & " + " & ".join(bot) + " \\\\\n")
L.append("Equal increments, Wald $p$ & & & " + f"{pa['above']['wald_equal_incr']['p']:.3f}" + " & & & " + f"{pa['below']['wald_equal_incr']['p']:.3f}" + " \\\\\n")
L.append("\\midrule\n\\multicolumn{7}{l}{\\emph{Panel B. Increment above the band by segment: checks}} \\\\\n")
L.append(" & 0--10 & 10--30 & 30--60 & $>$60 & Trend & \\\\\n\\midrule\n")
for lab, key in [("Before 13 Nov 2025", "1b_prevenue"), ("Bin averages (proportional)", "1c_binavg_full"), ("State bands", "1d_statebands_full")]:
    pp = P1[key]["above"]
    tb = [cellpair(pp["segments"][sg]["incr"]) for sg in SEGLAB] + [cellpair(pp["trend"])]
    L.append(lab + " & " + " & ".join(t_ for t_, _ in tb) + " & \\\\\n & " + " & ".join(b_ for _, b_ in tb) + " & \\\\\n")
L.append("\\midrule\n\\multicolumn{7}{l}{\\emph{Panel C. Change in the business-hours increment of $\\kappa^+$ per s.d.\\ of a predetermined state variable}} \\\\\n")
L.append(" & Full & Before & Venue & Logs & Price side & Spline \\\\\n & sample & 13 Nov 2025 & controls & & (pooled) & control \\\\\n\\midrule\n")
for lab, v in [("Previous month's SD of premium", "Za"), ("Previous day's interbank RV", "Zb"), ("Previous day's max.\\ distance", "Zc")]:
    ks = ["2i_full", "2ii_prevenue", "2iii_venue_controls", "2v_log_full", "2vi_priceside_R", "2vii_spline_control"]
    tb = [cellpair(P2[v][k]["incr_kp"] if "incr_kp" in P2[v][k] else P2[v][k]["incr"]) for k in ks]
    L.append(lab + " & " + " & ".join(t_ for t_, _ in tb) + " \\\\\n & " + " & ".join(b_ for _, b_ in tb) + " \\\\\n")
L.append("\\midrule\n\\multicolumn{7}{l}{\\emph{Panel D. The 13 November 2025 venue change, July 2025 -- March 2026}} \\\\\n")
L.append(" & Before & After & Change & & & \\\\\n\\midrule\n")
pp = P3["3b_prepost"]["kp"]
for lab, k0 in [("$\\kappa^+$ closed", "closed"), ("$\\kappa^+$ open", "open"), ("$\\kappa^+$ increment", "incr")]:
    tb = [cellpair(pp[f"{k0}_pre"]), cellpair(pp[f"{k0}_post"]), cellpair(pp[f"d_{k0}"])]
    L.append(lab + " & " + " & ".join(t_ for t_, _ in tb) + " & & & \\\\\n & " + " & ".join(b_ for _, b_ in tb) + " & & & \\\\\n")
pf = P3["3b_prepost_fullsample"]["kp"]
for lab, k0 in [("$\\kappa^+$ closed, Jan 2024 -- Aug 2026", "closed"), ("$\\kappa^+$ increment, Jan 2024 -- Aug 2026", "incr")]:
    tb = [cellpair(pf[f"{k0}_pre"]), cellpair(pf[f"{k0}_post"]), cellpair(pf[f"d_{k0}"])]
    L.append(lab + " & " + " & ".join(t_ for t_, _ in tb) + " & & & \\\\\n & " + " & ".join(b_ for _, b_ in tb) + " & & & \\\\\n")
sg = P3["3b_seg"]
for lab, k0 in [("Increment, 0--10 bp", "0-10"), ("Increment, 10--30 bp", "10-30")]:
    tb = [cellpair(sg[k0]["incr_pre"]), cellpair(sg[k0]["incr_post"]), cellpair(sg[k0]["d_incr"])]
    L.append(lab + " & " + " & ".join(t_ for t_, _ in tb) + " & & & \\\\\n & " + " & ".join(b_ for _, b_ in tb) + " & & & \\\\\n")
sgf = P3["3b_seg_full"]["avg_0_30"]
tb = [cellpair(sgf["incr_pre"]), cellpair(sgf["incr_post"]), cellpair(sgf["d_incr"])]
L.append("Increment, 0--30 bp, Jan 2024 -- Aug 2026 & " + " & ".join(t_ for t_, _ in tb) + " & & & \\\\\n & " + " & ".join(b_ for _, b_ in tb) + " & & & \\\\\n")
ft = P3["3c_largetrade"]["d_out"]
for lab, k0 in [("Large-trade $\\theta$ closed ($\\times100$)", "closed"), ("Large-trade $\\theta$ open ($\\times100$)", "open"), ("Large-trade $\\theta$ increment ($\\times100$)", "incr")]:
    tb = [cellpair(ft[f"{k0}_pre"], 3), cellpair(ft[f"{k0}_post"], 3), cellpair(ft[f"d_{k0}"], 3)]
    L.append(lab + " & " + " & ".join(t_ for t_, _ in tb) + " & & & \\\\\n & " + " & ".join(b_ for _, b_ in tb) + " & & & \\\\\n")
ab = P3["3d_abrupt"]
L.append(f"Placebo break dates: $p$, rise in closed $\\kappa^+$ & \\multicolumn{{3}}{{l}}{{{ab['pvalues']['p_closed']:.2f} ({ab['n_placebo']} dates)}} & & & \\\\\n")
L.append(f"Placebo break dates: $p$, fall in increment & \\multicolumn{{3}}{{l}}{{{ab['pvalues']['p_incr']:.2f}}} & & & \\\\\n")
L.append(f"Week of largest $t$ for the rise in closed $\\kappa^+$ ($-8$ to $8$) & \\multicolumn{{3}}{{l}}{{{ab['local_argmax_t_closed_week']}}} & & & \\\\\n")
L.append("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(TAB, f"rob_{NAME}.tex"), "w") as fh:
    fh.write("".join(L))

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.8))
xs = np.arange(4)
for side_off, (col, mk, lab, key) in enumerate([(C2, "s", "banks closed", "closed"), (C1, "o", "banks open", "open")]):
    v = np.array([pa["above"]["segments"][s][key] for s in SEGLAB])
    ax[0, 0].errorbar(xs + (side_off - 0.5) * 0.15, v[:, 0], yerr=1.96 * v[:, 1], fmt=mk, ms=5, color=col, ecolor=col, lw=1, label=lab)
ax[0, 0].axhline(0, color="k", lw=0.4)
ax[0, 0].set_xticks(xs); ax[0, 0].set_xticklabels(SEGLAB)
ax[0, 0].set_xlabel("distance above the band (bp)"); ax[0, 0].set_ylabel("marginal speed per 5 min")
ax[0, 0].set_title("(a) Speed by distance above the band", fontsize=9.5)
ax[0, 0].legend(frameon=False, fontsize=7.5)
labs_z = ["prev. month SD", "prev. day RV", "prev. day max dist."]
for j, (k, col, mk, lab) in enumerate([("2i_full", C1, "o", "full sample"), ("2ii_prevenue", C2, "s", "before 13 Nov 2025"),
                                        ("2iii_venue_controls", C3, "^", "venue controls")]):
    v = np.array([tup(P2[z_][k]["incr_kp"]) for z_ in ["Za", "Zb", "Zc"]])
    yy = np.arange(3) + (j - 1) * 0.22
    ax[0, 1].errorbar(v[:, 0], yy, xerr=1.96 * v[:, 1], fmt=mk, ms=5, color=col, ecolor=col, lw=1, label=lab)
ax[0, 1].axvline(0, color="k", lw=0.4)
ax[0, 1].set_yticks(range(3)); ax[0, 1].set_yticklabels(labs_z); ax[0, 1].invert_yaxis()
ax[0, 1].set_xlabel("change in BH increment of $\\kappa^+$ per s.d.")
ax[0, 1].set_title("(b) Predetermined state variables", fontsize=9.5)
ax[0, 1].legend(frameon=False, fontsize=7, loc="lower left")
ax[0, 1].xaxis.set_major_locator(plt.MaxNLocator(4))
E = P3["3a_event_weekly"]["weeks"]
for off, (s, col, mk, lab) in enumerate([("closed", C2, "s", "banks closed"), ("open", C1, "o", "banks open")]):
    w_ = [r_["week"] for r_ in E if r_[s] is not None]
    v = np.array([r_[s] for r_ in E if r_[s] is not None])
    ax[1, 0].errorbar(np.array(w_) + 0.5 + (off - 0.5) * 0.25, v[:, 0], yerr=1.96 * v[:, 1], fmt=mk, ms=4, color=col, ecolor=col,
                      lw=0.8, alpha=0.9, label=lab)
ax[1, 0].axvline(0, color="k", lw=0.7, ls="--"); ax[1, 0].axhline(0, color="k", lw=0.4)
ax[1, 0].set_xlabel("weeks relative to 13 Nov 2025"); ax[1, 0].set_ylabel("$\\hat\\kappa^+$ per 5 min")
ax[1, 0].set_title("(c) Weekly speeds around the venue change", fontsize=9.5)
ax[1, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
okd = ok.copy(); okd["dt"] = pd.to_datetime(okd.date)
ax[1, 1].fill_between(okd.dt, okd.d_closed - 1.96 * okd.se_closed, okd.d_closed + 1.96 * okd.se_closed, color=C3, alpha=0.25, lw=0)
ax[1, 1].plot(okd.dt, okd.d_closed, color=C3, lw=0.9, label="placebo break dates")
ax[1, 1].errorbar([pd.Timestamp(T0.date())], [act.d_closed], yerr=[1.96 * act.se_closed], fmt="o", ms=5, color=C2, ecolor=C2, label="13 Nov 2025")
ax[1, 1].axhline(0, color="k", lw=0.4)
ax[1, 1].set_ylabel("8 weeks after minus 8 weeks before")
ax[1, 1].set_title("(d) Change in closed $\\kappa^+$ at break dates", fontsize=9.5)
ax[1, 1].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[1, 1].tick_params(axis="x", rotation=30)
import matplotlib.dates as mdates
ax[1, 1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rob_{NAME}.pdf")); plt.close(fig)

# ------------------------------------------------------------------ ADDITIONAL DESCRIPTIVE CHECKS A1-A4
# They change no number above. (A1) speeds above the band by sub-period (Jan 2024-Jun 2025, 1 Jul-12 Nov 2025,
# 13 Nov 2025-Mar 2026, Apr-Aug 2026), baseline slopes x period, FE month x period, on all intervals and dropping
# intervals more than 30 (10) bp above the band; (A2) size composition of above-band intervals before/after T0 and the
# share of sum(e^2) from e > 30 / > 60 bp before T0 (leverage of large dislocations in the linear kappa+);
# (A3) pre/post-T0 linear speeds on intervals with e <= 30 (10) bp, full sample, FE month x P; (A4) Z_c rebuilt with the
# previous month's band (strictly predetermined; Z_c in (2) uses the current month's band).
AUD = {}
CUTS = [("A", "2024-01-01", "2025-07-01"), ("B", "2025-07-01", "2025-11-13"), ("C", "2025-11-13", "2026-04-01"),
        ("D", "2026-04-01", "2026-09-01")]


def subperiods(d0):
    d = d0.copy(); cols_ = []; d["per"] = ""
    for k, s0, s1 in CUTS:
        m = ((d.index >= pd.Timestamp(s0, tz="UTC")) & (d.index < pd.Timestamp(s1, tz="UTC"))).astype(float)
        d.loc[m == 1, "per"] = k
        for c in BASE:
            d[f"{c}_{k}"] = d[c] * m; cols_.append(f"{c}_{k}")
    d["mper"] = d.month + "_" + d.per
    r = fit(d, "y", cols_, "mper")
    o = {k: dict(period=[s0, s1], closed=lc(r, {f"up_{k}": 1}), open=lc(r, {f"up_{k}": 1, f"up_bh_{k}": 1}),
                 incr=lc(r, {f"up_bh_{k}": 1}), n_above=int(((d.per == k) & (d.e > 0)).sum()))
         for k, s0, s1 in CUTS}
    o["D_minus_C"] = dict(incr=es(lc(r, {"up_bh_D": 1, "up_bh_C": -1})), closed=es(lc(r, {"up_D": 1, "up_C": -1})),
                          open=es(lc(r, {"up_D": 1, "up_bh_D": 1, "up_C": -1, "up_bh_C": -1})))
    o["C_minus_B"] = dict(incr=es(lc(r, {"up_bh_C": 1, "up_bh_B": -1})))
    o["B_minus_A"] = dict(incr=es(lc(r, {"up_bh_B": 1, "up_bh_A": -1})))
    return o


AUD["A1_subperiods"] = {"all": subperiods(a), "e_le_30": subperiods(a[a.e <= 30]), "e_le_10": subperiods(a[a.e <= 10])}
comp = {}
for p_, lab_ in [(0, "pre"), (1, "post")]:
    ee = a.e[(a.P == p_) & (a.e > 0)]
    comp[lab_] = dict(n=int(len(ee)), share_0_10=float((ee <= 10).mean()), share_10_30=float(((ee > 10) & (ee <= 30)).mean()),
                      share_30_60=float(((ee > 30) & (ee <= 60)).mean()), share_gt60=float((ee > 60).mean()))
ee = a.e[(a.P == 0) & (a.e > 0)]
comp["pre_share_sum_e2_from_e_gt30"] = float((ee[ee > 30] ** 2).sum() / (ee ** 2).sum())
comp["pre_share_sum_e2_from_e_gt60"] = float((ee[ee > 60] ** 2).sum() / (ee ** 2).sum())
comp["gt60_by_year"] = {str(k): int(v) for k, v in a[a.e > 60].index.year.value_counts().sort_index().items()}
AUD["A2_composition"] = comp
A3 = {}
for lab_, msk in [("e_le_30", a.e <= 30), ("e_le_10", a.e <= 10)]:
    o_, _ = prepost(a[msk])
    A3[lab_] = o_["kp"]
AUD["A3_prepost_small_dislocations"] = A3
bnd = a.groupby("month")[["l", "u"]].first()
pm = {m: str(pd.Period(m, "M") - 1) for m in bnd.index}
a["exc_pm"] = (np.maximum(a.x - a.month.map(lambda m: bnd.u.get(pm[m], np.nan)), 0)
               + np.maximum(a.month.map(lambda m: bnd.l.get(pm[m], np.nan)) - a.x, 0))
mx_pm = a.groupby("day").exc_pm.max()
a["Zc_pm"] = a.day.map(lambda dd: mx_pm.get(prev_day.get(dd), np.nan))
a["Zc_pm_z"] = (a.Zc_pm - a.Zc_pm.mean()) / a.Zc_pm.std()
DAILY["Zc"] = True
AUD["A4_Zc_prev_month_band"] = {"full": state_reg(a, ["Zc_pm_z"])["Zc_pm_z"], "prevenue": state_reg(a[a.P == 0], ["Zc_pm_z"])["Zc_pm_z"]}
OUT["audit_posthoc"] = AUD
print("audit A1 all:", {k: (round(v["incr"][0], 4), round(v["incr"][1], 4)) for k, v in AUD["A1_subperiods"]["all"].items() if k in "ABCD"},
      " D-C:", round(AUD["A1_subperiods"]["all"]["D_minus_C"]["incr"]["est"], 4), flush=True)

OUT["runtime_s"] = time.time() - T_START


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(OUT), open(os.path.join(TAB, f"rob_{NAME}.json"), "w"), indent=1)
print("done in", round(OUT["runtime_s"], 1), "s")
