"""rob_jackknife_month: do single months drive the business-hours increase in kappa+ and the >60 bp size contrast?

April 2024 supplies a large share of the open-hours intervals far above the band (paper Appendix D, "Large-trade
flow by size of dislocation"). This script checks whether any single month drives (i) the baseline business-hours (BH) increase in kappa+ of Table emp_main, column (4), and (ii) the size
contrast of tables/rob_size_dependence.json: open- versus closed-hours reversion more than 60 bp above the band.

REFERENCE VALUES. The published full-sample estimates (emp_main col 4:
0.0079 (0.0014); rob_size_dependence 1c bin average >60 bp above: closed 0.0007 (0.0012), open 0.0103 (0.0015);
1a spline marginal >60 bp: closed -0.0016 (0.0030), open 0.0134 (0.0023), increment 0.0150 (0.0023)). Counts of
intervals more than 60 bp above the monthly band by month and banking state (bh): 3,799 in total (2,509 closed,
1,290 open); 2024-03 supplies 358 and 2024-04 376 of the 1,290 open-hours intervals (29% for April 2024), Jan-Apr 2024
together 737 (57%); none after Oct 2025. With bands by month x banking state, both edges pooled and banks open = bh,
April 2024 supplies 1,377 of 3,209 open-hours intervals beyond 60 bp (43%), 1,318 of them below the band (part (d) of
the addendum in the code recounts under rob_flow_spline's coding, banks open = bh_eff).

DESIGN
------
Data. data_raw/a5.parquet (162,624 five-minute intervals, 32 months Jan 2024 - Aug 2026). The monthly bands l, u are
kept at their published values in every sample: dropping a month removes its intervals (and its month fixed effect)
and leaves every other month's band unchanged. y = change in the premium (bp), q = OFI standardized within month,
BH = bh (09:00-17:00 Istanbul clock). Month fixed effects; standard errors clustered by UTC day with the finite-sample
correction of ols_fe (empirics_main.py), conditional on the bands.
Estimators (each identical to the published one on the full sample; the script asserts the reproduction).
  (B) Baseline, emp_main column (4): y on [q, q*BH, up, up*BH, dn, dn*BH]. Statistic: incr = coef on up*BH (the BH
      increase in kappa+); also closed = up, open = up + up*BH.
  (C) Bin averages, rob_size_dependence (1c): regressors -e*1{e in bin k} and +f*1{f in bin k} (e = (x-u)+,
      f = (l-x)+, bins (0,10], (10,30], (30,60], >60) and their BH interactions, plus q, q*BH. Statistics: the >60 bp
      above-band closed, open and increment (open minus closed) proportional speeds.
  (S) Piecewise-linear spline, rob_size_dependence (1a): knots 10, 30, 60 on both sides, each segment and its BH
      interaction, plus q, q*BH. Statistics: the marginal speed beyond 60 bp above the band, closed, open and increment.
      Secondary (reported): the implied average speed 90 bp above the band, closed, open and increment.
Samples. Full sample; 32 leave-one-month-out (LOO) samples; drop April 2024; drop January-April 2024 together.
Reported. For each increment (B, C, S): the full LOO distribution (estimate and s.e. for every dropped month), its
  minimum and the month that produces it, maximum, median, the number of LOO samples in which the increment is
  positive and significant at 5% (two-sided, z > 1.96; one-sided z > 1.645 also reported), the delete-one-month
  jackknife standard error sqrt((G-1)/G * sum_g (theta_(-g) - mean)^2) with G = 32 (a month-clustered check on the
  day-clustered s.e.), and the estimates when dropping April 2024 and January-April 2024.
Classification stored in the JSON.
  R1 (primary). The BH increase in kappa+ (B) is ROBUST to single months if in every one of the 32 LOO samples it is
     positive and significant at 5% two-sided (z > 1.96); otherwise NOT ROBUST, with the months that fail.
  R2. The same rule applied separately to the >60 bp open-minus-closed contrast in bin averages (C) and in the spline
     marginal speed (S).
  R3 (descriptive). Whether each of the three increments stays positive and significant at 5% two-sided when April
     2024 alone and January-April 2024 together are dropped; and in how many LOO samples the closed-hours >60 bp speed
     (C and S) is positive and significant at 5% two-sided (the paper states that beyond 60 bp closed-hours reversion
     is nil).

DESCRIPTIVE. Each month's share of the sum of squared distances beyond 60 bp above the band, sum of ((x-u)-60)+^2,
  by banking state: the variation that identifies the spline segment beyond 60 bp. An addendum in the code (a)-(d)
  reports the closed-hours speed and open/closed ratio in every sample and the concentration of the >60 bp variation.

Outputs: tables/rob_jackknife_month.json, tables/rob_jackknife_month.tex, figures/rob_jackknife_month.pdf.
Run: python -P code/rob_jackknife_month.py
"""
import os, sys, json, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
NAME = "jackknife_month"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
T_START = time.time()
Z2, Z1 = st.norm.ppf(0.975), st.norm.ppf(0.95)

# ------------------------------------------------------------------ data
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
assert len(a) == 162624
a["day"] = a.index.date
a["e"] = np.maximum(a.x - a.u, 0.0)
a["f"] = np.maximum(a.l - a.x, 0.0)
for c in ["q", "up", "dn"]:
    a[c + "_bh"] = a[c] * a.bh
MONTHS = sorted(a.month.unique())
assert len(MONTHS) == 32
BASE = ["q", "q_bh", "up", "up_bh", "dn", "dn_bh"]

# bin-average and spline regressors (rob_size_dependence 1c and 1a)
EDGES = [0, 10, 30, 60, np.inf]
BIN = ["q", "q_bh"]
for side, var, sign in [("ub", "e", -1.0), ("db", "f", 1.0)]:
    for k, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        c = f"{side}{k + 1}"
        a[c] = sign * a[var] * ((a[var] > lo) & (a[var] <= hi))
        a[c + "_bh"] = a[c] * a.bh
        BIN += [c, c + "_bh"]
KN = [10, 30, 60]
SPL = ["q", "q_bh"]
for side, var, sign in [("us", "e", -1.0), ("ds", "f", 1.0)]:
    cs_ = []
    for k, (lo, hi) in enumerate(zip([0.0] + KN, KN + [np.inf])):
        c = f"{side}{k + 1}"
        a[c] = sign * np.clip(a[var].to_numpy() - lo, 0, hi - lo)
        a[c + "_bh"] = a[c] * a.bh
        cs_.append(c)
    SPL += cs_ + [c + "_bh" for c in cs_]
L90 = np.array([np.clip(90.0 - lo, 0, hi - lo) for lo, hi in zip([0.0] + KN, KN + [np.inf])]) / 90.0


# ------------------------------------------------------------------ estimation helpers (copied conventions)
def fit(d, ycol, cols, fe="month"):
    """OLS with month FE (within) and day-clustered s.e. with the finite-sample correction of ols_fe."""
    Y = d[ycol].to_numpy(float); X = d[cols].to_numpy(float)
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
    return dict(b=b, V=V, idx={c: i for i, c in enumerate(cols)}, n=int(n), ng=int(ng))


def lc(r, w):
    g = np.zeros(len(r["b"]))
    for k, v in w.items():
        g[r["idx"][k]] += v
    return [float(g @ r["b"]), float(np.sqrt(g @ r["V"] @ g))]


def estimate(d):
    rb = fit(d, "y", BASE)
    rc = fit(d, "y", BIN)
    rs = fit(d, "y", SPL)
    gt60 = d.e > 60
    return dict(
        n=rb["n"], n_days=rb["ng"],
        n_gt60_above=dict(closed=int((gt60 & (d.bh == 0)).sum()), open=int((gt60 & (d.bh == 1)).sum())),
        base=dict(closed=lc(rb, {"up": 1}), open=lc(rb, {"up": 1, "up_bh": 1}), incr=lc(rb, {"up_bh": 1}),
                  dn_incr=lc(rb, {"dn_bh": 1})),
        bin60=dict(closed=lc(rc, {"ub4": 1}), open=lc(rc, {"ub4": 1, "ub4_bh": 1}), incr=lc(rc, {"ub4_bh": 1})),
        spl60=dict(closed=lc(rs, {"us4": 1}), open=lc(rs, {"us4": 1, "us4_bh": 1}), incr=lc(rs, {"us4_bh": 1})),
        avg90=dict(closed=lc(rs, {f"us{k + 1}": L90[k] for k in range(4)}),
                   open=lc(rs, dict([(f"us{k + 1}", L90[k]) for k in range(4)] + [(f"us{k + 1}_bh", L90[k]) for k in range(4)])),
                   incr=lc(rs, {f"us{k + 1}_bh": L90[k] for k in range(4)})),
    )


# ------------------------------------------------------------------ full sample and reproduction checks
print("full sample ...", flush=True)
FULL = estimate(a)
ref = json.load(open(os.path.join(TAB, "rob_size_dependence.json")))["piecewise"]
num = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["bh"]
chk = dict(
    base_incr=(FULL["base"]["incr"], [num["coef"]["up_bh"], num["se"]["up_bh"]]),
    bin60_closed=(FULL["bin60"]["closed"], ref["1c_binavg_full"]["above"]["segments"][">60"]["closed"]),
    bin60_open=(FULL["bin60"]["open"], ref["1c_binavg_full"]["above"]["segments"][">60"]["open"]),
    bin60_incr=(FULL["bin60"]["incr"], ref["1c_binavg_full"]["above"]["segments"][">60"]["incr"]),
    spl60_closed=(FULL["spl60"]["closed"], ref["1a_full"]["above"]["segments"][">60"]["closed"]),
    spl60_open=(FULL["spl60"]["open"], ref["1a_full"]["above"]["segments"][">60"]["open"]),
    spl60_incr=(FULL["spl60"]["incr"], ref["1a_full"]["above"]["segments"][">60"]["incr"]),
    avg90_open=(FULL["avg90"]["open"], ref["1a_full"]["above"]["avg_speed"]["90"]["open"]),
    avg90_closed=(FULL["avg90"]["closed"], ref["1a_full"]["above"]["avg_speed"]["90"]["closed"]),
)
REPRO = {}
for k, (mine, pub) in chk.items():
    dif = max(abs(mine[0] - pub[0]), abs(mine[1] - pub[1]))
    REPRO[k] = dict(here=mine, published=pub, max_abs_diff=dif)
    assert dif < 1e-9, (k, mine, pub)
assert f"{FULL['base']['incr'][0]:.4f}" == "0.0079" and f"{FULL['base']['incr'][1]:.4f}" == "0.0014"
print(" reproduced: base incr %.4f (%.4f)" % tuple(FULL["base"]["incr"]), flush=True)

# ------------------------------------------------------------------ leave-one-month-out and block drops
LOO = {}
for m in MONTHS:
    LOO[m] = estimate(a[a.month != m])
    LOO[m]["n_dropped"] = int((a.month == m).sum())
    print(f" drop {m}: base {LOO[m]['base']['incr'][0]:.4f} ({LOO[m]['base']['incr'][1]:.4f})  "
          f"bin60 {LOO[m]['bin60']['incr'][0]:.4f} ({LOO[m]['bin60']['incr'][1]:.4f})  "
          f"spl60 {LOO[m]['spl60']['incr'][0]:.4f} ({LOO[m]['spl60']['incr'][1]:.4f})", flush=True)
BLOCK = {"drop_2024-04": ["2024-04"], "drop_2024-01_to_2024-04": ["2024-01", "2024-02", "2024-03", "2024-04"]}
DROPS = {}
for k, ms in BLOCK.items():
    DROPS[k] = estimate(a[~a.month.isin(ms)])
    DROPS[k]["months_dropped"] = ms
    DROPS[k]["n_dropped"] = int(a.month.isin(ms).sum())


def summarize(key):
    est = np.array([LOO[m][key]["incr"][0] for m in MONTHS]); se = np.array([LOO[m][key]["incr"][1] for m in MONTHS])
    z = est / se
    G = len(est)
    jk = float(np.sqrt((G - 1) / G * np.sum((est - est.mean()) ** 2)))
    imin, imax = int(np.argmin(est)), int(np.argmax(est))
    izmin = int(np.argmin(z))
    fail2 = [MONTHS[i] for i in range(G) if not (est[i] > 0 and z[i] > Z2)]
    fail1 = [MONTHS[i] for i in range(G) if not (est[i] > 0 and z[i] > Z1)]
    return dict(full=FULL[key]["incr"],
                loo_min=dict(est=float(est[imin]), se=float(se[imin]), month=MONTHS[imin]),
                loo_max=dict(est=float(est[imax]), se=float(se[imax]), month=MONTHS[imax]),
                loo_median=float(np.median(est)), loo_mean=float(est.mean()),
                loo_min_z=dict(z=float(z[izmin]), month=MONTHS[izmin], est=float(est[izmin]), se=float(se[izmin])),
                n_pos_sig5_two_sided=int(G - len(fail2)), n_pos_sig5_one_sided=int(G - len(fail1)), n_loo=G,
                months_failing_two_sided=fail2, months_failing_one_sided=fail1,
                jackknife_se_month=jk, jackknife_se_over_day_se=float(jk / FULL[key]["incr"][1]),
                drop_2024_04=DROPS["drop_2024-04"][key]["incr"],
                drop_2024_01_04=DROPS["drop_2024-01_to_2024-04"][key]["incr"])


SUM = {k: summarize(k) for k in ["base", "bin60", "spl60", "avg90"]}


def pos_sig(v, zc=Z2):
    return bool(v[0] > 0 and v[0] / v[1] > zc)


R1 = "ROBUST" if SUM["base"]["n_pos_sig5_two_sided"] == 32 else "NOT ROBUST"
R2 = {k: ("ROBUST" if SUM[k]["n_pos_sig5_two_sided"] == 32 else "NOT ROBUST") for k in ["bin60", "spl60"]}
R3 = {"drops": {dk: {k: dict(est=DROPS[dk][k]["incr"], pos_sig5_two_sided=pos_sig(DROPS[dk][k]["incr"]),
                             pos_sig5_one_sided=pos_sig(DROPS[dk][k]["incr"], Z1)) for k in ["base", "bin60", "spl60"]}
                for dk in DROPS},
      "n_loo_closed_gt60_pos_sig5": {k: int(sum(pos_sig(LOO[m][k]["closed"]) for m in MONTHS)) for k in ["bin60", "spl60"]},
      "closed_gt60_pos_sig5_in_drops": {dk: {k: pos_sig(DROPS[dk][k]["closed"]) for k in ["bin60", "spl60"]} for dk in DROPS}}
OUT = dict(name=NAME, design="leave-one-month-out with the other months' bands unchanged; month FE; day clusters",
           reproduction=REPRO, full=FULL, loo=LOO, drops=DROPS, summary=SUM,
           reading=dict(R1_base=R1, R2_bin60=R2["bin60"], R2_spl60=R2["spl60"], R3=R3),
           counts_gt60_above_by_month={m: dict(closed=int(((a.month == m) & (a.e > 60) & (a.bh == 0)).sum()),
                    open=int(((a.month == m) & (a.e > 60) & (a.bh == 1)).sum())) for m in MONTHS})
# descriptive: each month's share of the identifying
# variation of the >60 bp spline segment, sum of (e-60)^2 over intervals more than 60 bp above the band, by state
s4 = np.maximum(a.e - 60, 0) ** 2
lev = s4.groupby([a.month, a.bh]).sum().unstack(fill_value=0.0)
lev = lev / lev.sum()
OUT["posthoc_share_sumsq_beyond60_by_month"] = {m: dict(closed=float(lev.loc[m, 0.0]), open=float(lev.loc[m, 1.0]))
                                                 for m in MONTHS if lev.loc[m].sum() > 0}
# ADDENDUM (descriptive; does not change any number above).
# (a) The closed-hours kappa+ and the open/closed ratio of emp_main col (4) in every LOO
# sample and in the two drops (log-ratio s.e. by the delta method with the day-clustered covariance), with the
# delete-one-month jackknife s.e. of the closed speed and of the log ratio. (b) Concentration of the variation beyond
# 60 bp in single days of April 2024 (Ramadan holiday 10-12 April, banks reopen Monday 15 April). (c) Open-hours
# intervals beyond 60 bp coded open under bh that fall on HOL weekday public holidays. (d) The count behind the
# "half of the open-hours intervals beyond 60 bp" note of rob_flow_spline, under that script's own coding (state bands,
# both edges pooled, banks open = bh_eff); the docstring's 1,377 of 3,209 uses bh instead of bh_eff.
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]   # copied from empirics_mech.py
a["hol"] = np.isin(a.day, pd.to_datetime(HOL).date).astype(float)


def ratio_stats(d):
    r = fit(d, "y", BASE); iu, ib = r["idx"]["up"], r["idx"]["up_bh"]
    c, o = r["b"][iu], r["b"][iu] + r["b"][ib]
    g = np.zeros(len(r["b"])); g[iu] = 1 / o - 1 / c; g[ib] = 1 / o
    lse = float(np.sqrt(g @ r["V"] @ g)) if (c > 0 and o > 0) else float("nan")
    return dict(closed=[float(c), float(np.sqrt(r["V"][iu, iu]))], open=float(o), ratio=float(o / c), log_ratio_se=lse)


RAT = {"full": ratio_stats(a)}
for m in MONTHS:
    RAT["drop_" + m] = ratio_stats(a[a.month != m])
for k, ms in BLOCK.items():
    RAT[k] = ratio_stats(a[~a.month.isin(ms)])
lr = np.array([np.log(RAT["drop_" + m]["ratio"]) for m in MONTHS]); cl = np.array([RAT["drop_" + m]["closed"][0] for m in MONTHS])
G_ = len(MONTHS)
jk_ = lambda v: float(np.sqrt((G_ - 1) / G_ * np.sum((v - v.mean()) ** 2)))
rr = np.exp(lr)
ss60 = np.maximum(a.e - 60, 0) ** 2
dayshare = {}
for dd in ["2024-04-10", "2024-04-11", "2024-04-12", "2024-04-15", "2024-04-16"]:
    dmask = a.day == pd.Timestamp(dd).date()
    dayshare[dd] = {s: float(ss60[dmask & (a.bh == v)].sum() / ss60[a.bh == v].sum()) for s, v in [("closed", 0), ("open", 1)]}
sb = a.join(pd.read_parquet(os.path.join(RAW, "state_bands.parquet")), how="inner")
dout_s = np.maximum(sb.x - sb.u_s, 0) + np.maximum(sb.l_s - sb.x, 0)
be = sb.bh * (1 - sb.hol)
op60 = (dout_s > 60) & (be == 1)
OUT["audit_addendum"] = dict(
    note="descriptive; changes no estimate above: (a) closed-hours speed and open/closed ratio in every sample, (b)-(d) concentration and counts of the variation beyond 60 bp",
    kappa_plus_ratio=dict(by_sample=RAT, loo_ratio_min=dict(ratio=float(rr.min()), month=MONTHS[int(rr.argmin())]),
                          loo_ratio_max=dict(ratio=float(rr.max()), month=MONTHS[int(rr.argmax())]),
                          loo_ratio_median=float(np.median(rr)),
                          loo_closed_max=dict(est=float(cl.max()), month=MONTHS[int(cl.argmax())]),
                          jackknife_se_log_ratio=jk_(lr), jackknife_se_closed=jk_(cl),
                          delta_se_log_ratio_full=RAT["full"]["log_ratio_se"], day_se_closed_full=RAT["full"]["closed"][1]),
    share_sumsq_beyond60_by_day_apr2024=dayshare,
    gt60_above_on_holidays=dict(open_coded_bh1=int(((a.e > 60) & (a.bh == 1) & (a.hol == 1)).sum()),
                                open_total=int(((a.e > 60) & (a.bh == 1)).sum()),
                                apr2024_open_on_holidays=int(((a.e > 60) & (a.bh == 1) & (a.hol == 1) & (a.month == "2024-04")).sum()),
                                apr2024_open_total=int(((a.e > 60) & (a.bh == 1) & (a.month == "2024-04")).sum())),
    flow_spline_count_bh_eff=dict(open_beyond60_total=int(op60.sum()),
                                  apr2024=int((op60 & (sb.month == "2024-04")).sum()),
                                  apr2024_below_band=int((op60 & (sb.month == "2024-04") & (sb.l_s - sb.x > 60)).sum())))
print("audit addendum:", json.dumps({k: v for k, v in OUT["audit_addendum"]["kappa_plus_ratio"].items() if k != "by_sample"}, default=float),
      json.dumps(OUT["audit_addendum"]["flow_spline_count_bh_eff"]), flush=True)
OUT["runtime_s"] = float(time.time() - T_START)
json.dump(OUT, open(os.path.join(TAB, f"rob_{NAME}.json"), "w"), indent=1, default=float)
print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in SUM.items()}, indent=1, default=float))
print("reading:", json.dumps(OUT["reading"], indent=1, default=float))


# ------------------------------------------------------------------ table
def f4(v, nd=4):
    return f"${v[0]:.{nd}f}$ $({v[1]:.{nd}f})$"


def star(v):
    t = abs(v[0] / v[1])
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def fs(v):
    return f"${v[0]:.4f}{star(v)}$ $({v[1]:.4f})$"


def mlab(m):
    return pd.Period(m, "M").strftime("%b %Y")


KEYS = ["base", "bin60", "spl60", "avg90"]
L = []
L.append("\\begin{tabular}{lcccc}\n\\toprule\n & BH increase in $\\kappa^+$ & \\multicolumn{3}{c}{Open minus closed, above the band} \\\\\n")
L.append("\\cmidrule(lr){3-5}\n & (Table~\\ref{tab:emp_main}, col.\\ 4) & Bin average, $>60$ bp & Spline, marginal $>60$ bp & Spline, average at 90 bp \\\\\n\\midrule\n")
L.append("\\multicolumn{5}{l}{\\emph{Panel A. Leave one month out (32 samples)}} \\\\\n")
L.append("Full sample & " + " & ".join(fs(SUM[k]["full"]) for k in KEYS) + " \\\\\n")
L.append("Minimum & " + " & ".join(fs([SUM[k]["loo_min"]["est"], SUM[k]["loo_min"]["se"]]) for k in KEYS) + " \\\\\n")
L.append("\\quad month dropped & " + " & ".join(mlab(SUM[k]["loo_min"]["month"]) for k in KEYS) + " \\\\\n")
L.append("Maximum & " + " & ".join(fs([SUM[k]["loo_max"]["est"], SUM[k]["loo_max"]["se"]]) for k in KEYS) + " \\\\\n")
L.append("\\quad month dropped & " + " & ".join(mlab(SUM[k]["loo_max"]["month"]) for k in KEYS) + " \\\\\n")
L.append("Median & " + " & ".join(f"${SUM[k]['loo_median']:.4f}$" for k in KEYS) + " \\\\\n")
L.append("Smallest $t$-statistic (month dropped) & " + " & ".join(
    f"{SUM[k]['loo_min_z']['z']:.1f} ({mlab(SUM[k]['loo_min_z']['month'])})" for k in KEYS) + " \\\\\n")
L.append("Positive at 5\\% (two-sided), of 32 & " + " & ".join(str(SUM[k]["n_pos_sig5_two_sided"]) for k in KEYS) + " \\\\\n")
L.append("Delete-one-month jackknife s.e. & " + " & ".join(f"${SUM[k]['jackknife_se_month']:.4f}$" for k in KEYS) + " \\\\\n")
L.append("\\midrule\n\\multicolumn{5}{l}{\\emph{Panel B. Dropping the early-2024 months}} \\\\\n")
for dk, lab in [("drop_2024-04", "Without April 2024"), ("drop_2024-01_to_2024-04", "Without January--April 2024")]:
    L.append(f"{lab}: increase & " + " & ".join(fs(DROPS[dk][k]["incr"]) for k in KEYS) + " \\\\\n")
    L.append("\\quad open & " + " & ".join(fs(DROPS[dk][k]["open"]) for k in KEYS) + " \\\\\n")
    L.append("\\quad closed & " + " & ".join(fs(DROPS[dk][k]["closed"]) for k in KEYS) + " \\\\\n")
    n60 = DROPS[dk]["n_gt60_above"]
    L.append(f"\\quad intervals $>60$ bp above (open / closed) & & \\multicolumn{{3}}{{c}}{{{n60['open']:,} / {n60['closed']:,}}} \\\\\n")
n60 = FULL["n_gt60_above"]
L.append(f"Full sample: intervals $>60$ bp above (open / closed) & & \\multicolumn{{3}}{{c}}{{{n60['open']:,} / {n60['closed']:,}}} \\\\\n")
L.append("\\midrule\n\\multicolumn{5}{l}{\\emph{Panel C. Each month dropped in turn: increase (month dropped, its intervals $>60$ bp above open/closed)}} \\\\\n")
for m in MONTHS:
    c = OUT["counts_gt60_above_by_month"][m]
    L.append(f"{mlab(m)} ({c['open']}/{c['closed']}) & " + " & ".join(fs(LOO[m][k]["incr"]) for k in KEYS) + " \\\\\n")
L.append("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(TAB, f"rob_{NAME}.tex"), "w") as fh:
    fh.write("".join(L))

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.8))
xs = np.arange(len(MONTHS))
xd = np.array([len(MONTHS) + 1.5, len(MONTHS) + 4.0])
ticks = [i for i, m in enumerate(MONTHS) if m[5:] in ("01", "07")]
for axx, key, title in [(ax[0, 0], "base", "(a) BH increase in $\\kappa^+$, baseline"),
                        (ax[0, 1], "bin60", "(b) Open minus closed, $>60$ bp: bin average"),
                        (ax[1, 0], "spl60", "(c) Open minus closed, $>60$ bp: spline, marginal")]:
    v = np.array([LOO[m][key]["incr"] for m in MONTHS])
    axx.axhspan(FULL[key]["incr"][0] - Z2 * FULL[key]["incr"][1], FULL[key]["incr"][0] + Z2 * FULL[key]["incr"][1],
                color=C3, alpha=0.15, lw=0)
    axx.axhline(FULL[key]["incr"][0], color=C3, lw=0.8, ls="--", label="full sample (95% CI shaded)")
    axx.errorbar(xs, v[:, 0], yerr=Z2 * v[:, 1], fmt="o", ms=3.5, color=C1, ecolor=C1, elinewidth=0.6, alpha=0.9, label="one month dropped")
    vd = np.array([DROPS[dk][key]["incr"] for dk in ["drop_2024-04", "drop_2024-01_to_2024-04"]])
    axx.errorbar(xd, vd[:, 0], yerr=Z2 * vd[:, 1], fmt="s", ms=4, color=C2, ecolor=C2, lw=0.8, label="Apr 2024 / Jan-Apr 2024 dropped")
    axx.axhline(0, color="k", lw=0.4)
    axx.set_xticks(ticks + list(xd))
    axx.set_xticklabels([MONTHS[i] for i in ticks] + ["no Apr 24", "no Jan-Apr 24"], rotation=45, ha="right", fontsize=7)
    axx.set_ylabel("per 5 min")
    axx.set_title(title, fontsize=9.5)
ax[0, 0].legend(frameon=False, fontsize=6.8, loc="lower left")
cnt = OUT["counts_gt60_above_by_month"]
ax[1, 1].bar(xs - 0.2, [cnt[m]["open"] for m in MONTHS], width=0.4, color=C1, label="banks open")
ax[1, 1].bar(xs + 0.2, [cnt[m]["closed"] for m in MONTHS], width=0.4, color=C2, label="banks closed")
ax[1, 1].set_xticks(ticks); ax[1, 1].set_xticklabels([MONTHS[i] for i in ticks], rotation=45, ha="right", fontsize=7)
ax[1, 1].set_ylabel("five-minute intervals")
ax[1, 1].set_title("(d) Intervals $>60$ bp above the band", fontsize=9.5)
ax[1, 1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rob_{NAME}.pdf")); plt.close(fig)
print(f"done in {time.time() - T_START:.0f}s")
