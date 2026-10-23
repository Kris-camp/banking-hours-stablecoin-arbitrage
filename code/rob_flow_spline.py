"""rob_flow_spline: does the large-trade (arbitrage) flow response show the same size pattern as prices?

MOTIVATION. tables/rob_size_dependence.json (Panel A) shows that above the band the closed-hours price reversion is
fast within 10 bp of the band and adds nothing beyond 60 bp (marginal closed speed -0.0016, s.e. 0.0030), while the
open-hours marginal speed beyond 60 bp is 0.0134. The settlement-capacity interpretation: when EFT is shut, pre-positioned
balances and capped FAST transfers can close small dislocations but not large ones, which need EFT. If so, the
FLOW of arbitrage trades should show the same pattern: closed-hours large-trade flow toward the band responds to the
distance near the band but not beyond 60 bp, while open-hours flow keeps rising with distance.

DESIGN
------
Data. data_raw/a5.parquet joined to data_raw/ofi_size/*.parquet and data_raw/state_bands.parquet (162,624
five-minute intervals). Construction exactly as Table emp_flow, column (2) (code/empirics_mech.py):
  z_large = large-trade (>=10k USDT) signed flow / its within-month SD; z_dem = small+medium (<10k) signed flow / its
  within-month SD. m = band midpoint; s = +1 if x > u, -1 if x < l, sign(x - m) inside the band (x = b_{t-1});
  d_out = (x-u)+ + (l-x)+, d_in = |x - m| for in-band premia (0 outside).
  A_L = -s * z_large and A_D = -s * z_dem: flow over the next five minutes (the interval following b_{t-1}),
  oriented toward the band. Banks open = bh_eff = bh * (1 - holiday), HOL list of empirics_mech.py (UTC date).
  Coefficients are reported per 100 bp (x100), in standard deviations of flow.
Spline. The linear d_out term (and d_out x banks-open) is replaced by a piecewise-linear spline in d_out with knots
  at 10, 30 and 60 bp: g1 = min(d,10), g2 = min((d-10)+,20), g3 = min((d-30)+,30), g4 = (d-60)+, each entered alone
  and x bh_eff. theta_k(closed) = coef on g_k; incr_k = coef on g_k x bh_eff; theta_k(open) = sum. Each theta is the
  marginal flow response per 100 bp of additional distance within segment k. d_in and d_in x bh_eff stay as in
  column (2). Standard errors clustered by UTC day with the finite-sample correction of ols/ols_fe.
Primary specification. State bands (l_s, u_s by month x banking state), full sample, month-by-state fixed effects
  (state = bh, the clock state on which the state bands are defined, as in emp_state), pooled over both edges.
  The published column (2) uses month fixed effects (empirics_mech.ols default); the month-FE version is reported
  as variant V, and the JSON states whether the classification below differs under it.
Checks (each changes one element of the primary).
  C1 before 13 Nov 2025: sample t < 2025-11-13 00:00 UTC.
  C2 above the band only: full sample, d_out split into d_up = (x-u_s)+ and d_dn = (l_s-x)+, each with its own
     spline and bh_eff interactions in one regression (plus d_in, d_in x bh_eff); P1-P3 evaluated on the above-band
     coefficients (the below-band ones are reported).
  C3 monthly bands: l, u from a5.parquet instead of the state bands (s, d_out, d_in recomputed); FE month x state.
Reference. The linear column (2) specification under each of the above (V with linear terms reproduces the
  published column (2): 0.030 and 0.108 per 100 bp).
Predictions for large-trade flow A_L under the capacity interpretation.
  P1: closed-hours marginal response beyond 60 bp (theta_4 closed) <= 0 or insignificant. HOLDS unless theta_4
      closed > 0 with one-sided p < 0.05.
  P2: open-minus-closed marginal response beyond 60 bp (incr_4) > 0, one-sided 5%. HOLDS if z > 1.645.
  P3: closed-hours response within 0-10 bp (theta_1 closed) > 0, one-sided 5%. HOLDS if z > 1.645.
Classification stored in the JSON: the capacity interpretation is SUPPORTED on the flow side if P1 and P2 hold in
  the primary specification AND P2 holds in at least one of C1, C2, C3. Otherwise NOT SUPPORTED, with the condition
  that fails. P3 is reported separately and does not enter the classification.
Small-and-medium-trade flow A_D. The model treats it as end-user demand, which flows WITH the dislocation (negative
  oriented response). The same segment estimates and the same three statistics are reported descriptively; no
  classification is applied to them.
Secondary (descriptive). Linear trend of the four increments across segment rank (weights (-1.5,-0.5,0.5,1.5)/5),
  Wald test of equal increments (3 df), the implied average response A(e) = sum_k theta_k len_k(e)/e at e = 5, 20,
  45, 90 bp, and the number of intervals and days reaching each segment by banking state.

Outputs: tables/rob_flow_spline.json, tables/rob_flow_spline.tex, figures/rob_flow_spline.pdf.
Run: python -P code/rob_flow_spline.py
"""
import os, sys, glob, json, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
NAME = "flow_spline"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
T_START = time.time()
T0 = pd.Timestamp("2025-11-13", tz="UTC")
OUT = dict(name=NAME, T0=str(T0), units="standard deviations of five-minute flow per 100 bp")

# ------------------------------------------------------------------ data (empirics_mech.py conventions)
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
a = a.join(pd.read_parquet(os.path.join(RAW, "state_bands.parquet")), how="inner")
a["day"] = a.index.date
for c in ["small", "mid", "large"]:
    a[f"z_{c}"] = a[f"ofi_{c}"] / a[f"ofi_{c}"].groupby(a.month).transform("std")
a["z_dem"] = (a.ofi_small + a.ofi_mid) / (a.ofi_small + a.ofi_mid).groupby(a.month).transform("std")
a = a.dropna(subset=["z_large", "z_dem"]).copy()
assert len(a) == 162624
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
a["hol"] = np.isin(a.day, pd.to_datetime(HOL).date).astype(float)
a["bh_eff"] = a.bh * (1 - a.hol)
a["state"] = np.where(a.bh == 1, "open", "closed")
a["ms"] = a.month + "_" + a.state
OUT["n"] = int(len(a)); OUT["n_days"] = int(a.day.nunique())

KN = [10, 30, 60]
EDGES = [0.0] + KN + [np.inf]
SEGLAB = ["0-10", "10-30", "30-60", ">60"]
WTR = np.array([-1.5, -0.5, 0.5, 1.5]) / 5
FLOWS = {"large": "A_L", "smallmed": "A_D"}


def band_vars(d, lcol, ucol):
    """Orientation and distances for a given band (empirics_mech.py)."""
    d = d.copy()
    l, u = d[lcol], d[ucol]
    m = (l + u) / 2
    d["s"] = np.where(d.x > u, 1.0, np.where(d.x < l, -1.0, np.sign(d.x - m)))
    out = (d.x > u) | (d.x < l)
    d["d_out"] = np.maximum(d.x - u, 0) + np.maximum(l - d.x, 0)
    d["d_in"] = np.where(out, 0.0, np.abs(d.x - m))
    d["d_up"] = np.maximum(d.x - u, 0); d["d_dn"] = np.maximum(l - d.x, 0)
    d["A_L"] = -d.s * d.z_large
    d["A_D"] = -d.s * d.z_dem
    for c in ["d_out", "d_in", "d_up", "d_dn"]:
        d[c + "_be"] = d[c] * d.bh_eff
    return d


def basis(d, var, name):
    cols = []
    v = d[var].to_numpy()
    for k, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        c = f"{name}{k + 1}"
        d[c] = np.clip(v - lo, 0, hi - lo)
        d[c + "_be"] = d[c] * d.bh_eff
        cols.append(c)
    return cols


def seg_len(e):
    return np.array([np.clip(e - lo, 0, hi - lo) for lo, hi in zip(EDGES[:-1], EDGES[1:])])


# ------------------------------------------------------------------ estimator (ols of empirics_mech with full V)
def fit(d, ycol, cols, fe):
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


def lc(r, w, scale=100.0):
    gv = np.zeros(len(r["b"]))
    for k, v in w.items():
        gv[r["idx"][k]] += v
    est, se = float(gv @ r["b"]) * scale, float(np.sqrt(gv @ r["V"] @ gv)) * scale
    return dict(est=est, se=se, z=est / se, p1_pos=float(st.norm.sf(est / se)), p2=float(2 * st.norm.sf(abs(est / se))))


def wald(r, ws):
    R = np.zeros((len(ws), len(r["b"])))
    for i, w in enumerate(ws):
        for k, v in w.items():
            R[i, r["idx"][k]] += v
    rb = R @ r["b"]; chi2 = float(rb @ np.linalg.solve(R @ r["V"] @ R.T, rb))
    return dict(chi2=chi2, df=len(ws), p=float(st.chi2.sf(chi2, len(ws))))


def counts(d, var):
    res = {}
    for k, lo in enumerate(EDGES[:-1]):
        reach = d[var].to_numpy() > lo
        res[SEGLAB[k]] = dict(closed=int((reach & (d.bh_eff == 0)).sum()), open=int((reach & (d.bh_eff == 1)).sum()),
                              days_closed=int(d.loc[reach & (d.bh_eff == 0), "day"].nunique()),
                              days_open=int(d.loc[reach & (d.bh_eff == 1), "day"].nunique()))
    return res


def summarize(r, sc):
    segs = {SEGLAB[k]: dict(closed=lc(r, {c: 1}), open=lc(r, {c: 1, c + "_be": 1}), incr=lc(r, {c + "_be": 1}))
            for k, c in enumerate(sc)}
    avg = {}
    for ev in [5, 20, 45, 90]:
        L = seg_len(ev) / ev
        avg[str(ev)] = dict(closed=lc(r, {c: L[k] for k, c in enumerate(sc)}),
                            open=lc(r, {**{c: L[k] for k, c in enumerate(sc)}, **{c + "_be": L[k] for k, c in enumerate(sc)}}),
                            incr=lc(r, {c + "_be": L[k] for k, c in enumerate(sc)}))
    c4, c1 = segs[">60"]["closed"], segs["0-10"]["closed"]; i4 = segs[">60"]["incr"]
    tests = dict(P1_closed_gt60_not_sig_pos=bool(not (c4["est"] > 0 and c4["p1_pos"] < 0.05)),
                 P2_incr_gt60_pos=bool(i4["z"] > 1.645),
                 P3_closed_0_10_pos=bool(c1["z"] > 1.645))
    return dict(segments=segs, trend=lc(r, {c + "_be": w for c, w in zip(sc, WTR)}),
                incr4_minus_incr1=lc(r, {sc[3] + "_be": 1, sc[0] + "_be": -1}),
                wald_equal_incr=wald(r, [{sc[k] + "_be": 1, sc[0] + "_be": -1} for k in (1, 2, 3)]),
                avg_response=avg, tests=tests)


def run_spec(d, fe, split):
    """Spline and linear versions of emp_flow col (2) for both flows."""
    d = d.copy()
    res = {"n": int(len(d)), "n_days": int(d.day.nunique()), "fe": fe, "split_above_below": split}
    if split:
        su, sd_ = basis(d, "d_up", "gu"), basis(d, "d_dn", "gd")
        cols = su + [c + "_be" for c in su] + sd_ + [c + "_be" for c in sd_] + ["d_in", "d_in_be"]
        lin = ["d_up", "d_up_be", "d_dn", "d_dn_be", "d_in", "d_in_be"]
        res["counts"] = dict(above=counts(d, "d_up"), below=counts(d, "d_dn"))
    else:
        so = basis(d, "d_out", "g")
        cols = so + [c + "_be" for c in so] + ["d_in", "d_in_be"]
        lin = ["d_out", "d_out_be", "d_in", "d_in_be"]
        res["counts"] = dict(pooled=counts(d, "d_out"))
    for fk, ycol in FLOWS.items():
        r = fit(d, ycol, cols, fe)
        rl = fit(d, ycol, lin, fe)
        out = {}
        if split:
            out["above"] = summarize(r, su); out["below"] = summarize(r, sd_)
            out["linear"] = {c: lc(rl, {c: 1}) for c in lin}
            out["linear"]["d_up_open"] = lc(rl, {"d_up": 1, "d_up_be": 1})
        else:
            out["pooled"] = summarize(r, so)
            out["linear"] = {c: lc(rl, {c: 1}) for c in lin}
            out["linear"]["d_out_open"] = lc(rl, {"d_out": 1, "d_out_be": 1})
        out["d_in"] = dict(closed=lc(r, {"d_in": 1}), incr=lc(r, {"d_in_be": 1}))
        res[fk] = out
    return res


dS = band_vars(a, "l_s", "u_s")
dM = band_vars(a, "l", "u")
SPECS = {
    "primary": dict(d=dS, fe="ms", split=False, label="Primary: state bands, month x state FE"),
    "C1_pre": dict(d=dS[dS.index < T0], fe="ms", split=False, label="Before 13 Nov 2025"),
    "C2_above": dict(d=dS, fe="ms", split=True, label="Above the band only"),
    "C3_monthly": dict(d=dM, fe="ms", split=False, label="Monthly bands"),
    "V_monthFE": dict(d=dS, fe="month", split=False, label="Month FE (as published col. 2)"),
}
R = {}
for k, sp in SPECS.items():
    R[k] = run_spec(sp["d"], sp["fe"], sp["split"])
    R[k]["label"] = sp["label"]
    print(k, f"{time.time() - T_START:.0f}s", flush=True)
OUT["specs"] = R
# sanity: V linear reproduces published column (2)
pub = json.load(open(os.path.join(TAB, "emp_mech.json")))["flow"]["A_L"]
OUT["check_reproduces_pub_col2"] = dict(
    d_out=[R["V_monthFE"]["large"]["linear"]["d_out"]["est"], 100 * pub["d_out"][0]],
    d_out_be=[R["V_monthFE"]["large"]["linear"]["d_out_be"]["est"], 100 * pub["d_out_bh"][0]])
assert abs(OUT["check_reproduces_pub_col2"]["d_out"][0] - OUT["check_reproduces_pub_col2"]["d_out"][1]) < 1e-6
assert abs(OUT["check_reproduces_pub_col2"]["d_out_be"][0] - OUT["check_reproduces_pub_col2"]["d_out_be"][1]) < 1e-6


def key_block(spec):
    return R[spec]["large"]["above" if R[spec]["split_above_below"] else "pooled"]


# ------------------------------------------------------------------ classification (see docstring)
P = key_block("primary")["tests"]
checks_P2 = {k: key_block(k)["tests"]["P2_incr_gt60_pos"] for k in ["C1_pre", "C2_above", "C3_monthly"]}
supported = P["P1_closed_gt60_not_sig_pos"] and P["P2_incr_gt60_pos"] and any(checks_P2.values())
fails = []
if not P["P1_closed_gt60_not_sig_pos"]:
    fails.append("P1 fails in primary")
if not P["P2_incr_gt60_pos"]:
    fails.append("P2 fails in primary")
if not any(checks_P2.values()):
    fails.append("P2 fails in all checks")
V = key_block("V_monthFE")["tests"]
OUT["reading"] = dict(
    supported=bool(supported), verdict="SUPPORTED" if supported else "NOT SUPPORTED", failed_conditions=fails,
    primary_tests=P, P2_in_checks=checks_P2,
    P_by_spec={k: key_block(k)["tests"] for k in SPECS},
    verdict_under_month_FE_primary=("SUPPORTED" if (V["P1_closed_gt60_not_sig_pos"] and V["P2_incr_gt60_pos"]
                                                     and any(checks_P2.values())) else "NOT SUPPORTED"),
    smallmed_descriptive={k: R[k]["smallmed"]["above" if R[k]["split_above_below"] else "pooled"]["tests"] for k in SPECS})
OUT["runtime_s"] = round(time.time() - T_START, 1)


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


# ------------------------------------------------------------------ LaTeX table
def star(z):
    z = abs(z)
    return "^{***}" if z > 2.576 else "^{**}" if z > 1.96 else "^{*}" if z > 1.645 else ""


def cell(x):
    return f"${x['est']:.3f}{star(x['z'])}$", f"$({x['se']:.3f})$"


ORDER = ["primary", "C1_pre", "C2_above", "C3_monthly", "V_monthFE"]
HEAD = ["Primary", "Before 13 Nov 2025", "Above band only", "Monthly bands", "Month FE"]
with open(os.path.join(TAB, f"rob_{NAME}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{l" + "c" * len(ORDER) + "}\n\\toprule\n & " + " & ".join(HEAD) + " \\\\\n & " +
             " & ".join(f"({i + 1})" for i in range(len(ORDER))) + " \\\\\n")
    for fk, ptitle in [("large", "A. Large trades ($\\ge$10{,}000 USDT), flow toward the band"),
                       ("smallmed", "B. Small and medium trades ($<$10{,}000 USDT), flow toward the band")]:
        fh.write(f"\\midrule\n\\multicolumn{{{len(ORDER) + 1}}}{{l}}{{\\emph{{{ptitle}}}}} \\\\\n")
        for part, plab in [("closed", "Banks closed, marginal response"), ("incr", "Open $-$ closed")]:
            fh.write(f"\\multicolumn{{{len(ORDER) + 1}}}{{l}}{{{plab}}} \\\\\n")
            for sg in SEGLAB:
                lab = f"\\quad {sg.replace('-', '--').replace('>', '$>$')} bp"
                cs = [cell(R[k][fk]["above" if R[k]["split_above_below"] else "pooled"]["segments"][sg][part]) for k in ORDER]
                fh.write(lab + " & " + " & ".join(c[0] for c in cs) + " \\\\\n")
                fh.write(" & " + " & ".join(c[1] for c in cs) + " \\\\\n")
        cs = [cell(R[k][fk]["above" if R[k]["split_above_below"] else "pooled"]["segments"][">60"]["open"]) for k in ORDER]
        fh.write("Banks open, $>$60 bp & " + " & ".join(c[0] for c in cs) + " \\\\\n & " + " & ".join(c[1] for c in cs) + " \\\\\n")
        lk = lambda k, p: R[k][fk]["linear"][("d_up" if R[k]["split_above_below"] else "d_out") + p]
        cs = [cell(lk(k, "")) for k in ORDER]
        fh.write("Linear distance, closed & " + " & ".join(c[0] for c in cs) + " \\\\\n & " + " & ".join(c[1] for c in cs) + " \\\\\n")
        cs = [cell(lk(k, "_be")) for k in ORDER]
        fh.write("Linear distance, open $-$ closed & " + " & ".join(c[0] for c in cs) + " \\\\\n & " + " & ".join(c[1] for c in cs) + " \\\\\n")
        if fk == "large":
            yn = lambda b: "yes" if b else "no"
            for tk, tl in [("P1_closed_gt60_not_sig_pos", "P1: closed $>$60 bp not positive"),
                           ("P2_incr_gt60_pos", "P2: open $-$ closed $>$60 bp $>0$"),
                           ("P3_closed_0_10_pos", "P3: closed 0--10 bp $>0$")]:
                fh.write(tl + " & " + " & ".join(yn(key_block(k)["tests"][tk]) for k in ORDER) + " \\\\\n")
    fh.write("\\midrule\n")
    cnt = lambda k: R[k]["counts"]["above" if R[k]["split_above_below"] else "pooled"][">60"]
    fh.write("Intervals $>$60 bp out, closed / open & " + " & ".join(f"{cnt(k)['closed']:,} / {cnt(k)['open']:,}" for k in ORDER) + " \\\\\n")
    fh.write("Days with an interval $>$60 bp out, closed / open & " + " & ".join(f"{cnt(k)['days_closed']} / {cnt(k)['days_open']}" for k in ORDER) + " \\\\\n")
    fh.write("Fixed effects & " + " & ".join("month" if R[k]["fe"] == "month" else "month$\\times$state" for k in ORDER) + " \\\\\n")
    fh.write("$N$ & " + " & ".join(f"{R[k]['n']:,}" for k in ORDER) + " \\\\\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
xs = np.arange(4)
panels = [(0, 0, "primary", "large", "(a) Large trades, both edges"),
          (0, 1, "primary", "smallmed", "(b) Small and medium trades, both edges"),
          (1, 0, "C2_above", "large", "(c) Large trades, above the band"),
          (1, 1, "C2_above", "smallmed", "(d) Small and medium trades, above the band")]
for i, j, sk, fk, title in panels:
    blk = R[sk][fk]["above" if R[sk]["split_above_below"] else "pooled"]["segments"]
    for off, (st_, col, mk, lab) in enumerate([("closed", C2, "s", "banks closed"), ("open", C1, "o", "banks open")]):
        v = np.array([[blk[s][st_]["est"], blk[s][st_]["se"]] for s in SEGLAB])
        ax[i, j].errorbar(xs + (off - 0.5) * 0.18, v[:, 0], yerr=1.96 * v[:, 1], fmt=mk, ms=5, color=col, ecolor=col,
                          lw=1, capsize=0, label=lab)
    ax[i, j].axhline(0, color=C3, lw=0.6)
    ax[i, j].set_title(title, fontsize=9, loc="left")
    ax[i, j].set_xticks(xs); ax[i, j].set_xticklabels([s.replace("-", "–") for s in SEGLAB])
    if j == 0:
        ax[i, j].set_ylabel("Marginal response\n(s.d. of flow per 100 bp)")
    if i == 1:
        ax[i, j].set_xlabel("Distance outside the band (bp)")
ax[0, 0].legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rob_{NAME}.pdf")); plt.close(fig)
print(json.dumps(clean(OUT["reading"]), indent=1))
print(f"done in {time.time() - T_START:.0f}s")
