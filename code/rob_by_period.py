"""rob_by_period: does the business-hours increase in the speed of arbitrage hold in every part of the sample?

Estimated by year, the business-hours increment in kappa+ of Table emp_main col (4) (0.0079, s.e. 0.0014, full
sample) is 0.0077 (0.0016) in 2024 and 0.0094 (0.0033) in 2025 but 0.0010 (0.0014) in 2026, where the closed-hours
speed is 0.0095. This script estimates the increment by period under several specifications, tests whether 2026
differs, and describes what changed.

Periods (UTC calendar months of the a5 panel, column `month`):
  full 2024-01..2026-08; years 2024, 2025, 2026 (Jan-Aug); 2024-2025 pooled; half-years 2024H1, 2024H2, 2025H1,
  2025H2, 2026H1 (Jan-Jun), 2026H2 (Jul-Aug, two months only).
Every period estimate is a separate regression on the period's intervals (month and month-x-state FE nest within
periods), day-clustered with the finite-sample correction of ols/ols_fe, conditional on the bands.

Specifications (each re-estimated within every period)
  (a) baseline, Table emp_main col (4): y on [q, up, dn, q*BH, up*BH, dn*BH]; monthly bands (a5.parquet); month FE;
      BH = bh (09:00-17:00 Istanbul clock).
  (b) demand-only order-flow control (rob_demand_ofi.py): q replaced by z_dem = (ofi_small+ofi_mid)/within-month SD.
      (b1) published monthly bands; (b2) monthly bands re-profiled with z_dem as the control.
  (c) state bands, Table emp_state col (1): up_s, dn_s from the bands by month x banking state, month x state FE,
      BH = bh. Every band cell (a month, or a month x state cell) lies inside one period and its profile uses only
      the cell's own data, so re-profiling within a period must reproduce the published bands. The script verifies
      this by re-profiling every monthly cell and every month x state cell (control q) and reports the maximum
      absolute difference; if it is zero the two band treatments coincide and are reported once.
  (d) EFT-hours coding (rob_tcmb_institutions.py): open = 08:30-17:30 Istanbul on full business days, 08:30-13:00 on
      the six half days, 0 on the HOL weekday holidays (Istanbul date); otherwise as (a).
  (e) price side of Table emp_flow col (1): R = -s*(y - beta*q) on [d_out, d_out*bh_eff, d_in, d_in*bh_eff]; state
      bands; month FE; bh_eff = bh*(1-hol) with the HOL list of empirics_mech.py (UTC date, as there).
      (e1) beta fixed at the published full-sample value (emp_numbers.json, reg.thr.q);
      (e2) beta re-estimated within the period from the col (2) analogue (y on q, up, dn; monthly bands; month FE).
      (e2) is the primary by-period version because Table emp_year shows beta falling from 1.04 (2024) to 0.25 (2026);
      in the full sample (e1) = (e2). Supplement: d_out split into above/below (d_up, d_dn) to report the
      price-side kappa+ by state.
Reported for each spec x period: kappa+ closed, the BH increment, kappa+ open (s.e. from the joint covariance),
open/closed ratio and the s.e. of its log (when both speeds are positive); the same for kappa-; beta by state where
the spec has one; N; the number of above-band intervals by state.

(2) Test that the 2026 increment equals the 2024-2025 increment. Full sample; each spec with every slope (flow
    and reversion terms and their BH interactions) also interacted with D26 = 1{2026}; FE as in the spec (they nest
    in years, so D26 itself is absorbed); day-clustered. Reported: the 2024-25 increment, the D26 x up x BH
    coefficient (2026 minus 2024-25 increment; s.e., z, two-sided normal p), and the same 2026-minus-2024/25
    differences for the closed speed (D26 x up) and the open speed (sum). For (e2) R uses the calendar-year beta.
    Secondary: fully interacted three-year model; Wald test (chi2, 2 df) that the BH increment of kappa+ is the same
    in 2024, 2025 and 2026, and the same for the closed speed; pairwise 2026-2024 and 2026-2025 differences.
    Summary classifications stored in the JSON:
      - the 2026 increment "differs" from 2024-25 if the D26 test rejects at 5% in spec (a);
      - the business-hours effect "disappears in 2026" if the 2026 increment is insignificant at 5% in (a) AND in at
        least three of the five checks {b1, b2, c, d, e2};
      - "because the closed-hours speed rose" requires, in (a), that the 2026 closed speed exceeds its 2024-25 value
        (D26 x up > 0, significant at 5%) while the open speed does not differ significantly at 5%.

(3) Large-trade flow response, Table emp_flow col (2): A_L = -s*z_large on [d_out, d_out*bh_eff, d_in, d_in*bh_eff];
    state bands; month FE; coefficients x 100 (per 100 bp). By period, the same D26 test, and the above-band split
    (d_up, d_dn). Price-flow cross-equation test (difference of log open/closed ratios, joint day-clustered covariance
    of (e) and (3) on the same sample) by period when all four speeds are positive.
    Long-run impact beta/kappa+ by state and period from (a) and (c), delta-method s.e., closed/open ratio and the
    s.e. of its log; undefined when a speed is not positive. A D26 test on the 2026-minus-2024/25 change of the log
    closed/open long-run-impact ratio is not run (it is a nonlinear function of the D26 model; reported by period).

(4) Rolling windows: 6 consecutive calendar months, step one month (2024-01..06 to 2026-03..08, 27 windows).
    kappa+ closed and open with 95% CIs under (a) and (c), and the large-trade theta closed and open under (3).
    Figure figures/rob_by_period.pdf.

(5) What changed in 2026 (descriptive, by year and half-year): mean and SD of the five-minute premium, median
    within-month SD, SD of the premium change; mean monthly band width and state band widths; share of intervals
    above/below the monthly band, overall and by banking state, and the mean distance above the band when above;
    mean and median five-minute volume by state; large-trade (>=10k USDT) share of volume and large trades per
    interval by state; mean USD/TRY and the USD value of the FAST cap; mean five-minute volume and large-trade share
    by Istanbul hour and year; and the speed by block of the day: kappa+ (spec (a) with up and dn interacted with
    night 00-09, day 09-17 and evening 17-24 Istanbul) and the large-trade theta (spec (3) with d_out and d_in
    interacted with night, evening, banks-open day (bh_eff) and holiday day (bh_hol)), by year. Leave-one-month-out
    within 2026 for the (a) closed speed and increment.

(6) Diagnostics for the night block (00:00-09:00 Istanbul), where the interbank benchmark is least actively
    quoted and the full-sample speed is zero, and from which the 2026 rise in the closed-hours speed comes:
(6i) Which price moves: the block specification of (5) with the five-minute change in the log local price (dpl) and
     in the log interbank rate (dpg) as dependent variables, by year (kappa_L = coefficient on up in dpl;
     kappa_G = minus the coefficient on up in dpg).
(6ii) Interbank quoting activity (fxv, ticks per five minutes): median by block of the day and year.
(6iii) Noise-robust timing: spec (a) and the block specification with the reversion regressors built from b_{t-2}
     instead of b_{t-1} (dependent variable still b_t - b_{t-1}; intervals t-2, t-1, t consecutive). Independent
     measurement noise in the premium creates spurious reversion on b_{t-1} but not on b_{t-2}; the b_{t-2}
     coefficients are attenuated by one interval of decay but are not contaminated by such noise.

Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, data_raw/ofi_size/*.parquet, tables/emp_numbers.json,
        tables/emp_state.json, tables/emp_mech.json, tables/rob_demand_ofi.json, tables/rob_tcmb_institutions.json.
Outputs: tables/rob_by_period.json, tables/rob_by_period.tex, figures/rob_by_period.pdf.
"""
import os, glob, json, time
import numpy as np, pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from model import _ssr_grid

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
NAME = "by_period"
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
t0 = time.time()

# ------------------------------------------------------------------ data
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
a["day"] = a.index.date
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
a = a.join(sb, how="inner")
assert len(a) == 162624 and a[["ofi_small", "ofi_mid", "ofi_large", "l_s", "u_s"]].notna().all().all()
for c in ["small", "mid", "large"]:
    a[f"z_{c}"] = a[f"ofi_{c}"] / a[f"ofi_{c}"].groupby(a.month).transform("std")
a["z_dem"] = (a.ofi_small + a.ofi_mid) / (a.ofi_small + a.ofi_mid).groupby(a.month).transform("std")
a["state"] = np.where(a.bh == 1, "open", "closed")
a["ms"] = a.month + "_" + a.state
a["year"] = a.month.str[:4]

HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
HALF = ["2024-04-09", "2024-10-28", "2025-06-05", "2025-10-28", "2026-03-19", "2026-05-26"]
a["hol"] = np.isin(a.day, pd.to_datetime(HOL).date).astype(float)          # UTC date, as empirics_mech.py
a["bh_eff"] = a.bh * (1 - a.hol)
a["bh_hol"] = a.bh * a.hol
ti = a.index.tz_convert("Europe/Istanbul")
hm = np.asarray(ti.hour * 60 + ti.minute)
ih = np.asarray(ti.hour)
iday = pd.Series(ti.strftime("%Y-%m-%d"), index=a.index)
hol_i = iday.isin(HOL).values; half_pm = iday.isin(HALF).values & (hm >= 13 * 60)
a["eft"] = np.where(hol_i, 0, np.where(half_pm, 0, (hm >= 510) & (hm < 1050))).astype(float)
a["ih"] = ih
a["night"] = (ih < 9).astype(float); a["eve"] = (ih >= 17).astype(float)
assert np.allclose(a.bh.values, ((ih >= 9) & (ih < 17)).astype(float))
OUT = dict(name=NAME, n=int(len(a)), n_days=int(a.day.nunique()))


# ------------------------------------------------------------------ estimators (copied conventions)
def profile(y, x, f, trim=0.05, step=1.0):
    """Identical to empirics_main/empirics_state/rob_demand_ofi: returns (l, u)."""
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, f, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]


def ols(d, ycol, cols, fe="month"):
    """Within-FE OLS, day-clustered with the finite-sample correction; per-day scores for joint covariances."""
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    gm = d.groupby(fe)
    Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes, uniq = pd.factorize(d.day)
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    adj = ng / (ng - 1) * (n - 1) / (n - k)
    S = G @ XtXi.T
    V = S.T @ S * adj
    return dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), V=V, S=S, days=uniq, adj=adj,
                cols=list(cols), n=int(n), n_days=int(ng))


def lin(r, w):
    g = np.array([w.get(c, 0.0) for c in r["cols"]]); b = np.array([r["b"][c] for c in r["cols"]])
    return [float(g @ b), float(np.sqrt(g @ r["V"] @ g))]


def zp(v):
    z_ = v[0] / v[1]
    return dict(z=float(z_), p=float(2 * st.norm.sf(abs(z_))))


def speeds(r, c0, c1, scale=1.0):
    k0, k1 = r["b"][c0], r["b"][c1]
    out = dict(closed=lin(r, {c0: scale}), incr=lin(r, {c1: scale}), open=lin(r, {c0: scale, c1: scale}))
    out["incr"] += [zp(out["incr"])["p"]]
    if k0 > 0 and k0 + k1 > 0:
        ix = {c: i for i, c in enumerate(r["cols"])}
        g = np.zeros(len(r["cols"])); g[ix[c0]] = 1 / (k0 + k1) - 1 / k0; g[ix[c1]] = 1 / (k0 + k1)
        out["ratio"] = float((k0 + k1) / k0); out["se_logratio"] = float(np.sqrt(g @ r["V"] @ g))
    else:
        out["ratio"] = None; out["se_logratio"] = None
    return out


def lr_impact(r, f, fb, c0, c1):
    """Long-run impact beta/kappa by state (rob_demand_ofi.state_summary), delta method."""
    ix = {c: i for i, c in enumerate(r["cols"])}; k = len(ix); V = r["V"]
    b0, b1, k0, k1 = r["b"][f], r["b"][fb], r["b"][c0], r["b"][c1]
    if not (k0 > 0 and k0 + k1 > 0 and b0 > 0 and b0 + b1 > 0):
        return dict(closed=None, open=None, ratio_closed_open=None, se_logratio=None)
    gc = np.zeros(k); gc[ix[f]] = 1 / k0; gc[ix[c0]] = -b0 / k0 ** 2
    go = np.zeros(k); go[ix[f]] = go[ix[fb]] = 1 / (k0 + k1); go[ix[c0]] = go[ix[c1]] = -(b0 + b1) / (k0 + k1) ** 2
    gl = np.zeros(k); gl[ix[f]] = 1 / b0 - 1 / (b0 + b1); gl[ix[fb]] = -1 / (b0 + b1)
    gl[ix[c0]] = -1 / k0 + 1 / (k0 + k1); gl[ix[c1]] = 1 / (k0 + k1)
    lrc, lro = b0 / k0, (b0 + b1) / (k0 + k1)
    return dict(closed=[float(lrc), float(np.sqrt(gc @ V @ gc))], open=[float(lro), float(np.sqrt(go @ V @ go))],
                ratio_closed_open=float(lrc / lro), se_logratio=float(np.sqrt(gl @ V @ gl)))


def joint(regs):
    days = regs[0]["days"]
    for r in regs[1:]:
        assert (r["days"] == days).all()
    S = np.hstack([r["S"] for r in regs])
    b = np.concatenate([[r["b"][c] for c in r["cols"]] for r in regs])
    return b, S.T @ S * regs[0]["adj"]


def cross_test(pr, fw, c0="d_out", c1="d_out_be"):
    """Difference of log open/closed ratios, price (pr) minus flow (fw), joint day-clustered covariance."""
    b, V = joint([pr, fw]); k = len(pr["cols"])
    iC, iB = pr["cols"].index(c0), pr["cols"].index(c1)
    kC, kB, tC, tB = b[iC], b[iB], b[k + iC], b[k + iB]
    if min(kC, kC + kB, tC, tC + tB) <= 0:
        return dict(ratio_price=None, ratio_flow=None, logdiff=None, se_logdiff=None)
    g = np.zeros(2 * k)
    g[iC] = 1 / (kC + kB) - 1 / kC; g[iB] = 1 / (kC + kB)
    g[k + iC] = -(1 / (tC + tB) - 1 / tC); g[k + iB] = -1 / (tC + tB)
    dl = np.log((kC + kB) / kC) - np.log((tC + tB) / tC)
    return dict(ratio_price=float((kC + kB) / kC), ratio_flow=float((tC + tB) / tC), logdiff=float(dl),
                se_logdiff=float(np.sqrt(g @ V @ g)))


# ------------------------------------------------------------------ bands: re-profiling checks and (b2)
def reprofile(groups, f, min_n):
    l, u = np.full(len(a), np.nan), np.full(len(a), np.nan)
    for key, ix in a.groupby(groups).indices.items():
        if len(ix) < min_n: continue
        d = a.iloc[ix]
        l[ix], u[ix] = profile(d.y.values, d.x.values, d[f].values.astype(float))
    return l, u


lM_q, uM_q = reprofile("bandper", "q", 500)
lS_q, uS_q = reprofile(["month", "state"], "q", 300)
lM_D, uM_D = reprofile("bandper", "z_dem", 500)
OUT["check_reprofile_within_period"] = dict(
    monthly_bands_q_max_abs_diff=float(np.nanmax(np.abs(np.r_[lM_q - a.l.values, uM_q - a.u.values]))),
    state_bands_q_max_abs_diff=float(np.nanmax(np.abs(np.r_[lS_q - a.l_s.values, uS_q - a.u_s.values]))),
    n_month_cells=int(a.bandper.nunique()), n_month_state_cells=int(a.ms.nunique()),
    note="Every band cell lies inside one period, so re-profiling within any period reproduces these values.")
print("bands done", f"{time.time() - t0:.0f}s", OUT["check_reprofile_within_period"], flush=True)

# ------------------------------------------------------------------ design columns
a["upD"] = -np.maximum(a.x - uM_D, 0); a["dnD"] = np.maximum(lM_D - a.x, 0)
a["up_s"] = -np.maximum(a.x - a.u_s, 0); a["dn_s"] = np.maximum(a.l_s - a.x, 0)
for S_ in ["bh", "eft"]:
    for c in ["q", "z_dem", "up", "dn", "upD", "dnD", "up_s", "dn_s"]:
        a[f"{c}__{S_}"] = a[c] * a[S_]
ms_ = (a.l_s + a.u_s) / 2
a["s"] = np.where(a.x > a.u_s, 1.0, np.where(a.x < a.l_s, -1.0, np.sign(a.x - ms_)))
out_s = (a.x > a.u_s) | (a.x < a.l_s)
a["d_out"] = np.maximum(a.x - a.u_s, 0) + np.maximum(a.l_s - a.x, 0)
a["d_in"] = np.where(~out_s, np.abs(a.x - ms_), 0.0)
a["d_up"] = np.maximum(a.x - a.u_s, 0); a["d_dn"] = np.maximum(a.l_s - a.x, 0)
for v in ["d_out", "d_in", "d_up", "d_dn"]:
    a[v + "_be"] = a[v] * a.bh_eff
    a[v + "_hol"] = a[v] * a.bh_hol
    a[v + "_night"] = a[v] * a.night
    a[v + "_eve"] = a[v] * a.eve
for c in ["up", "dn"]:
    a[c + "_night"] = a[c] * a.night; a[c + "_eve"] = a[c] * a.eve
a["A_L"] = -a.s * a.z_large
BETA_PUB = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
a["R1"] = -a.s * (a.y - BETA_PUB * a.q)

SPECS = {   # y, flow control, up, dn, open indicator, FE
    "a": dict(f="q", up="up", dn="dn", S="bh", fe="month", label="Baseline (Table emp_main, col. 4)"),
    "b1": dict(f="z_dem", up="up", dn="dn", S="bh", fe="month", label="Small and medium OFI control"),
    "b2": dict(f="z_dem", up="upD", dn="dnD", S="bh", fe="month", label="Small and medium OFI, bands re-estimated"),
    "c": dict(f="q", up="up_s", dn="dn_s", S="bh", fe="ms", label="Bands by month and banking state"),
    "d": dict(f="q", up="up", dn="dn", S="eft", fe="month", label="EFT operating hours"),
}


def spec_cols(sp):
    S_ = sp["S"]
    return [sp["f"], sp["up"], sp["dn"], f"{sp['f']}__{S_}", f"{sp['up']}__{S_}", f"{sp['dn']}__{S_}"]


PCOLS = ["d_out", "d_out_be", "d_in", "d_in_be"]
SCOLS = ["d_up", "d_up_be", "d_dn", "d_dn_be", "d_in", "d_in_be"]


def beta_col2(d):
    return ols(d, "y", ["q", "up", "dn"], fe="month")["b"]["q"]


def run_period(d):
    """All specifications on one period's sample."""
    res = {}
    for k, sp in SPECS.items():
        cols = spec_cols(sp); S_ = sp["S"]
        r = ols(d, "y", cols, fe=sp["fe"])
        res[k] = dict(kp=speeds(r, sp["up"], f"{sp['up']}__{S_}"), km=speeds(r, sp["dn"], f"{sp['dn']}__{S_}"),
                      beta=dict(closed=lin(r, {sp["f"]: 1}), open=lin(r, {sp["f"]: 1, f"{sp['f']}__{S_}": 1})),
                      lr_kp=lr_impact(r, sp["f"], f"{sp['f']}__{S_}", sp["up"], f"{sp['up']}__{S_}"),
                      n=r["n"], n_days=r["n_days"],
                      n_above_closed=int(((d[sp["up"]] < 0) & (d[S_] == 0)).sum()),
                      n_above_open=int(((d[sp["up"]] < 0) & (d[S_] == 1)).sum()),
                      _r=r)
    bP = beta_col2(d)
    dd = d.copy(); dd["R2"] = -dd.s * (dd.y - bP * dd.q)
    fw = ols(dd, "A_L", PCOLS); fws = ols(dd, "A_L", SCOLS)
    for k, yc in [("e1", "R1"), ("e2", "R2")]:
        pr = ols(dd, yc, PCOLS); ps = ols(dd, yc, SCOLS)
        res[k] = dict(beta_used=float(BETA_PUB if k == "e1" else bP), pooled=speeds(pr, "d_out", "d_out_be"),
                      kp=speeds(ps, "d_up", "d_up_be"), km=speeds(ps, "d_dn", "d_dn_be"),
                      placebo_in=speeds(pr, "d_in", "d_in_be"), cross_vs_flow=cross_test(pr, fw), n=pr["n"], _r=pr)
    res["flow"] = dict(pooled=speeds(fw, "d_out", "d_out_be", 100), kp=speeds(fws, "d_up", "d_up_be", 100),
                       km=speeds(fws, "d_dn", "d_dn_be", 100), placebo_in=speeds(fw, "d_in", "d_in_be", 100),
                       n=fw["n"], _r=fw)
    res["beta_col2"] = float(bP)
    return res


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    return o


PERIODS = {"full": ("2024-01", "2026-08"), "2024": ("2024-01", "2024-12"), "2025": ("2025-01", "2025-12"),
           "2026": ("2026-01", "2026-08"), "2024-2025": ("2024-01", "2025-12"),
           "2024H1": ("2024-01", "2024-06"), "2024H2": ("2024-07", "2024-12"), "2025H1": ("2025-01", "2025-06"),
           "2025H2": ("2025-07", "2025-12"), "2026H1": ("2026-01", "2026-06"), "2026H2": ("2026-07", "2026-08")}
RES = {}
for p, (m0, m1) in PERIODS.items():
    RES[p] = run_period(a[(a.month >= m0) & (a.month <= m1)])
    print(p, "a incr", RES[p]["a"]["kp"]["incr"], "closed", RES[p]["a"]["kp"]["closed"], flush=True)
OUT["by_period"] = {p: strip(v) for p, v in RES.items()}

# ------------------------------------------------------------------ replication checks (full sample)
pub = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["bh"]["coef"]
pub_s = json.load(open(os.path.join(TAB, "emp_state.json")))["state_bands"]["coef"]
pub_m = json.load(open(os.path.join(TAB, "emp_mech.json")))
pub_d = json.load(open(os.path.join(TAB, "rob_demand_ofi.json")))["A_emp_main_col4"]
pub_t = json.load(open(os.path.join(TAB, "rob_tcmb_institutions.json")))["audit_ratio_check"]
F = RES["full"]
OUT["check_replication"] = dict(
    a_up_bh=[F["a"]["kp"]["incr"][0], pub["up_bh"]], a_up=[F["a"]["kp"]["closed"][0], pub["up"]],
    b1_incr=[F["b1"]["kp"]["incr"][0], pub_d["fixed_D"]["kp"]["incr"][0]],
    b2_incr=[F["b2"]["kp"]["incr"][0], pub_d["reprofiled_D"]["kp"]["incr"][0]],
    c_incr=[F["c"]["kp"]["incr"][0], pub_s["up_s_bh"][0]],
    d_incr=[F["d"]["kp"]["incr"][0], pub_t["EFT hours: 08:30-17:30, 13:00 on half days, 0 on holidays"]["diff"]],
    e1_incr=[F["e1"]["pooled"]["incr"][0], pub_m["price"]["d_out_bh"][0]],
    flow_incr_x100=[F["flow"]["pooled"]["incr"][0], 100 * pub_m["flow"]["A_L"]["d_out_bh"][0]],
    cross_logdiff=[F["e1"]["cross_vs_flow"]["logdiff"], pub_m["cross"]["logdiff"]])
print(json.dumps(OUT["check_replication"]), flush=True)

# ------------------------------------------------------------------ (2) D26 tests
yb = {y_: beta_col2(a[a.year == y_]) for y_ in ["2024", "2025", "2026"]}
a["R2y"] = -a.s * (a.y - a.year.map(yb).astype(float) * a.q)
for y_ in ["2025", "2026"]:
    a["D" + y_[2:]] = (a.year == y_).astype(float)


def test_2026(ycol, cols, fe, c0, c1, scale=1.0):
    """cols + cols x D26 (2026 vs 2024-25) and cols + cols x D25 + cols x D26 (three years)."""
    d = a.copy()
    for c in cols:
        d[c + "_26"] = d[c] * d.D26; d[c + "_25"] = d[c] * d.D25
    r = ols(d, ycol, cols + [c + "_26" for c in cols], fe=fe)
    o = dict(incr_2024_25=lin(r, {c1: scale}), incr_2026=lin(r, {c1: scale, c1 + "_26": scale}),
             d_incr=lin(r, {c1 + "_26": scale}), d_closed=lin(r, {c0 + "_26": scale}),
             d_open=lin(r, {c0 + "_26": scale, c1 + "_26": scale}),
             closed_2024_25=lin(r, {c0: scale}), closed_2026=lin(r, {c0: scale, c0 + "_26": scale}),
             open_2024_25=lin(r, {c0: scale, c1: scale}),
             open_2026=lin(r, {c0: scale, c1: scale, c0 + "_26": scale, c1 + "_26": scale}))
    for k in ["d_incr", "d_closed", "d_open"]:
        o[k + "_test"] = zp(o[k])
    r3 = ols(d, ycol, cols + [c + "_25" for c in cols] + [c + "_26" for c in cols], fe=fe)
    ix = {c: i for i, c in enumerate(r3["cols"])}; b3 = np.array([r3["b"][c] for c in r3["cols"]])
    for nm, cc in [("incr", c1), ("closed", c0)]:
        Rm = np.zeros((2, len(ix))); Rm[0, ix[cc + "_25"]] = 1; Rm[1, ix[cc + "_26"]] = 1
        rb = Rm @ b3; W = float(rb @ np.linalg.solve(Rm @ r3["V"] @ Rm.T, rb))
        o[f"wald_equal_{nm}_3yr"] = dict(chi2=W, df=2, p=float(st.chi2.sf(W, 2)))
    o["pair_2026_minus_2024_incr"] = lin(r3, {c1 + "_26": scale})
    o["pair_2026_minus_2025_incr"] = lin(r3, {c1 + "_26": scale, c1 + "_25": -scale})
    o["pair_2026_minus_2024_closed"] = lin(r3, {c0 + "_26": scale})
    o["pair_2026_minus_2025_closed"] = lin(r3, {c0 + "_26": scale, c0 + "_25": -scale})
    o["n"] = r["n"]
    return o


TESTS = {}
for k, sp in SPECS.items():
    TESTS[k] = test_2026("y", spec_cols(sp), sp["fe"], sp["up"], f"{sp['up']}__{sp['S']}")
TESTS["e1"] = test_2026("R1", PCOLS, "month", "d_out", "d_out_be")
TESTS["e2"] = test_2026("R2y", PCOLS, "month", "d_out", "d_out_be")
TESTS["e2_above"] = test_2026("R2y", SCOLS, "month", "d_up", "d_up_be")
TESTS["flow"] = test_2026("A_L", PCOLS, "month", "d_out", "d_out_be", 100)
TESTS["flow_above"] = test_2026("A_L", SCOLS, "month", "d_up", "d_up_be", 100)
OUT["test_2026"] = TESTS
OUT["beta_by_year_col2"] = yb
print("tests done", f"{time.time() - t0:.0f}s", {k: (v["d_incr"], v["d_closed"]) for k, v in TESTS.items()}, flush=True)

# summary classifications (see docstring, part (2))
y26 = RES["2026"]
sig5 = lambda v: abs(v[0] / v[1]) > 1.96
checks = ["b1", "b2", "c", "d", "e2"]
n_insig = sum(not sig5((y26[k]["kp"] if k != "e2" else y26[k]["pooled"])["incr"]) for k in checks)
Ta = TESTS["a"]
OUT["reading_rules"] = dict(
    differs_2026_vs_2024_25=bool(Ta["d_incr_test"]["p"] < 0.05),
    a_2026_incr_insignificant=bool(not sig5(y26["a"]["kp"]["incr"])),
    n_checks_2026_incr_insignificant=f"{n_insig} of {len(checks)}",
    disappears_in_2026=bool((not sig5(y26["a"]["kp"]["incr"])) and n_insig >= 3),
    closed_speed_rose=bool(Ta["d_closed"][0] > 0 and Ta["d_closed_test"]["p"] < 0.05),
    open_speed_unchanged=bool(Ta["d_open_test"]["p"] >= 0.05),
    because_closed_speed_rose=bool(Ta["d_closed"][0] > 0 and Ta["d_closed_test"]["p"] < 0.05 and Ta["d_open_test"]["p"] >= 0.05))
print(OUT["reading_rules"], flush=True)

# ------------------------------------------------------------------ (4) rolling 6-month windows
months = sorted(a.month.unique())
ROLL = []
for i in range(len(months) - 5):
    w = months[i:i + 6]
    d = a[a.month.isin(w)]
    ra = ols(d, "y", spec_cols(SPECS["a"]), fe="month")
    rc = ols(d, "y", spec_cols(SPECS["c"]), fe="ms")
    rf = ols(d, "A_L", PCOLS)
    ROLL.append(dict(start=w[0], end=w[-1], a=speeds(ra, "up", "up__bh"), c=speeds(rc, "up_s", "up_s__bh"),
                     flow=speeds(rf, "d_out", "d_out_be", 100), n=ra["n"]))
OUT["rolling_6m"] = ROLL
print("rolling done", f"{time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ (5) what changed
tot = a.vol_small + a.vol_mid + a.vol_large
a["above_m"] = (a.x > a.u).astype(float); a["below_m"] = (a.x < a.l).astype(float)
a["above_s"] = (a.x > a.u_s).astype(float); a["below_s"] = (a.x < a.l_s).astype(float)
a["exc_up"] = np.maximum(a.x - a.u, 0)
a["usdtry"] = np.exp(a.pg)
cap_tl = np.where(a.index < pd.Timestamp("2024-04-04", tz="UTC"), 50000.0,
                  np.where(a.index >= pd.Timestamp("2026-08-26", tz="UTC"), 300000.0, 100000.0))
a["fast_usd"] = cap_tl / a.usdtry


def describe(d):
    o = dict(n=int(len(d)), n_days=int(d.day.nunique()),
             prem_mean=float(d.x.mean()), prem_sd=float(d.x.std()),
             prem_sd_within_month_median=float(d.groupby("month").x.std().median()),
             dprem_sd=float(d.y.std()),
             band_width_monthly_mean=float(d.groupby("month").apply(lambda g: (g.u - g.l).iloc[0]).mean()),
             band_width_state_open_mean=float(d[d.bh == 1].groupby("month").apply(lambda g: (g.u_s - g.l_s).iloc[0]).mean()),
             band_width_state_closed_mean=float(d[d.bh == 0].groupby("month").apply(lambda g: (g.u_s - g.l_s).iloc[0]).mean()),
             usdtry_mean=float(d.usdtry.mean()), fast_cap_usd_mean=float(d.fast_usd.mean()))
    tt = d.vol_small + d.vol_mid + d.vol_large
    for st_, m in [("all", np.ones(len(d), bool)), ("open", (d.bh == 1).values), ("closed", (d.bh == 0).values)]:
        e = d[m]; te = tt[m]
        o[st_] = dict(share_above_monthly=float(e.above_m.mean()), share_below_monthly=float(e.below_m.mean()),
                      share_above_state=float(e.above_s.mean()), share_below_state=float(e.below_s.mean()),
                      mean_excess_above_given_above=float(e.exc_up[e.above_m == 1].mean()) if e.above_m.sum() else None,
                      vol_mean_k=float(e.vol.mean() / 1e3), vol_median_k=float(e.vol.median() / 1e3),
                      large_share_vol=float(e.vol_large.sum() / te.sum()), n_large_per_interval=float(e.n_large.mean()),
                      abs_ofi_large_mean_k=float(e.ofi_large.abs().mean() / 1e3),
                      share_of_period_volume=float(te.sum() / tt.sum()), n=int(len(e)))
    return o


DESC = {p: describe(a[(a.month >= m0) & (a.month <= m1)]) for p, (m0, m1) in PERIODS.items()}
HOUR = {}
for y_ in ["2024", "2025", "2026"]:
    d = a[a.year == y_]; tt = (d.vol_small + d.vol_mid + d.vol_large)
    g = pd.DataFrame({"vol_k": d.vol / 1e3, "vl": d.vol_large, "tt": tt, "ih": d.ih})
    h = g.groupby("ih").agg(vol_mean_k=("vol_k", "mean"), vl=("vl", "sum"), tt=("tt", "sum"), n=("vol_k", "size"))
    h["large_share"] = h.vl / h.tt
    HOUR[y_] = {int(k): dict(vol_mean_k=float(v.vol_mean_k), large_share=float(v.large_share), n=int(v.n))
                for k, v in h.iterrows()}
    cl = d[d.bh == 0]; op = d[d.bh == 1]
    HOUR[y_]["closed_vs_open_mean_vol_ratio"] = float(cl.vol.mean() / op.vol.mean())
    HOUR[y_]["closed_share_of_volume"] = float(cl.vol.sum() / d.vol.sum())

# speed by block of the day, by year
BLOCK = {}
bc_price = ["q", "q__bh", "up_night", "up__bh", "up_eve", "dn_night", "dn__bh", "dn_eve"]
bc_flow = ["d_out_night", "d_out_eve", "d_out_be", "d_out_hol", "d_in_night", "d_in_eve", "d_in_be", "d_in_hol"]
for p in ["full", "2024", "2025", "2026", "2024-2025"]:
    m0, m1 = PERIODS[p]; d = a[(a.month >= m0) & (a.month <= m1)]
    rp = ols(d, "y", bc_price); rf = ols(d, "A_L", bc_flow)
    BLOCK[p] = dict(kp_night=lin(rp, {"up_night": 1}), kp_day=lin(rp, {"up__bh": 1}), kp_eve=lin(rp, {"up_eve": 1}),
                    km_night=lin(rp, {"dn_night": 1}), km_day=lin(rp, {"dn__bh": 1}), km_eve=lin(rp, {"dn_eve": 1}),
                    theta_night=lin(rf, {"d_out_night": 100}), theta_eve=lin(rf, {"d_out_eve": 100}),
                    theta_day_open=lin(rf, {"d_out_be": 100}), theta_day_holiday=lin(rf, {"d_out_hol": 100}),
                    n_above_night=int(((d.up < 0) & (d.night == 1)).sum()),
                    n_above_eve=int(((d.up < 0) & (d.eve == 1)).sum()),
                    n_above_day=int(((d.up < 0) & (d.bh == 1)).sum()))
# leave-one-month-out within 2026, spec (a)
LOMO = {}
d26 = a[a.year == "2026"]
for mo in sorted(d26.month.unique()):
    r = ols(d26[d26.month != mo], "y", spec_cols(SPECS["a"]))
    LOMO[mo] = dict(closed=lin(r, {"up": 1}), incr=lin(r, {"up__bh": 1}), open=lin(r, {"up": 1, "up__bh": 1}))
# monthly closed/open kappa+ in 2026 is too noisy cell by cell; report counts of above-band intervals by month
cnt = a.groupby(["month", "state"]).above_m.sum().unstack()
OUT["what_changed"] = dict(describe=DESC, by_hour=HOUR, block_speeds=BLOCK, lomo_2026_a=LOMO,
                           above_band_counts_by_month={k: {s: int(v) for s, v in r_.items()} for k, r_ in cnt.iterrows()})

# ------------------------------------------------------------------ (6) night-block diagnostics (see docstring)
PH = {"note": "Diagnostics for the night block: which price moves, interbank quoting activity, and reversion regressors built from b_{t-2}."}
a["blk"] = np.where(a.night == 1, "night", np.where(a.bh == 1, "day", "evening"))
consec2 = (pd.Series(a.index, index=a.index).diff() == pd.Timedelta("5min")).values
a["x2"] = np.where(consec2, a.x.shift(1).values, np.nan)          # b_{t-2}, only when t-1 directly follows t-2
a["up2"] = -np.maximum(a.x2 - a.u, 0); a["dn2"] = np.maximum(a.l - a.x2, 0)
for c in ["up2", "dn2"]:
    a[c + "__bh"] = a[c] * a.bh; a[c + "_night"] = a[c] * a.night; a[c + "_eve"] = a[c] * a.eve
LEGS, FXV, LAG2 = {}, {}, {}
for p in ["full", "2024", "2025", "2026", "2024-2025"]:
    m0, m1 = PERIODS[p]; d = a[(a.month >= m0) & (a.month <= m1)]
    rl = ols(d, "dpl", bc_price); rg = ols(d, "dpg", bc_price)
    LEGS[p] = {f"{side}_{b_}": dict(local=lin(rl, {c: 1}), global_=lin(rg, {c: -1}))
               for side, cs_ in [("kp", ["up_night", "up__bh", "up_eve"]), ("km", ["dn_night", "dn__bh", "dn_eve"])]
               for b_, c in zip(["night", "day", "eve"], cs_)}
    FXV[p] = {b_: float(d.loc[d.blk == b_, "fxv"].median()) for b_ in ["night", "day", "evening"]}
    d2 = d[d.x2.notna()]
    r2 = ols(d2, "y", ["q", "up2", "dn2", "q__bh", "up2__bh", "dn2__bh"])
    r2b = ols(d2, "y", ["q", "q__bh", "up2_night", "up2__bh", "up2_eve", "dn2_night", "dn2__bh", "dn2_eve"])
    r1 = ols(d2, "y", spec_cols(SPECS["a"]))                       # b_{t-1} version on the same sample
    LAG2[p] = dict(lag2=speeds(r2, "up2", "up2__bh"), lag1_same_sample=speeds(r1, "up", "up__bh"),
                   lag2_blocks=dict(night=lin(r2b, {"up2_night": 1}), day=lin(r2b, {"up2__bh": 1}),
                                    eve=lin(r2b, {"up2_eve": 1})), n=r2["n"])
PH.update(legs_by_block=LEGS, fxv_median_by_block=FXV, lag2_regressor=LAG2)
OUT["posthoc"] = PH
print("night-block diagnostics done", f"{time.time() - t0:.0f}s", flush=True)
OUT["runtime_s"] = float(time.time() - t0)
json.dump(OUT, open(os.path.join(TAB, f"rob_{NAME}.json"), "w"), indent=1, default=float)
print("json written", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ LaTeX table
def star(v):
    t = abs(v[0] / v[1])
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def cell(v, nd=4, stars=True):
    if v is None: return "--", ""
    return f"${v[0]:.{nd}f}{star(v) if stars else ''}$", f"$({v[1]:.{nd}f})$"


YRS = ["full", "2024", "2025", "2026"]
ROWS = [("a", "(a) Baseline (Table~\\ref{tab:emp_main}, col.\\ 4)"), ("b1", "(b) Small and medium trades only"),
        ("b2", "\\quad bands re-estimated"), ("c", "(c) Bands by month and banking state"),
        ("d", "(d) EFT operating hours"), ("e2", "(e) Price side, both edges (Table~\\ref{tab:emp_flow}, col.\\ 1)")]


def get(p, k, what):
    r = RES[p][k]
    return (r["pooled"] if k in ("e1", "e2", "flow") else r["kp"])[what]


def tget(k, what):
    return TESTS[k]["d_" + what]


with open(os.path.join(TAB, f"rob_{NAME}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccccc}\n\\toprule\n & Full sample & 2024 & 2025 & 2026 & 2026 $-$ (2024--25) \\\\\n"
             " & (1) & (2) & (3) & (4) & (5) \\\\\n\\midrule\n")
    for panel, what in [("A. Increase in the speed of arbitrage above the band when banks are open", "incr"),
                        ("B. Speed above the band, banks closed", "closed"),
                        ("C. Speed above the band, banks open", "open")]:
        fh.write(f"\\multicolumn{{6}}{{l}}{{\\emph{{{panel}}}}} \\\\\n")
        for k, lab in ROWS:
            cc = [cell(get(p, k, what)) for p in YRS] + [cell(tget(k, what))]
            fh.write(lab + " & " + " & ".join(x[0] for x in cc) + " \\\\\n & " + " & ".join(x[1] for x in cc) + " \\\\\n")
    fh.write("\\multicolumn{6}{l}{\\emph{D. Large-trade flow toward the band (Table~\\ref{tab:emp_flow}, col.\\ 2), per 100 bp}} \\\\\n")
    for lab, what in [("Banks closed", "closed"), ("\\quad$\\times$ banks open", "incr")]:
        cc = [cell(get(p, "flow", what), 3) for p in YRS] + [cell(tget("flow", what), 3)]
        fh.write(lab + " & " + " & ".join(x[0] for x in cc) + " \\\\\n & " + " & ".join(x[1] for x in cc) + " \\\\\n")
    fh.write("\\multicolumn{6}{l}{\\emph{E. Long-run impact $\\beta/\\kappa^+$ (bp), specification (a)}} \\\\\n")
    for lab, what in [("Banks closed", "closed"), ("Banks open", "open")]:
        cc = [cell(RES[p]["a"]["lr_kp"][what], 0, stars=False) for p in YRS] + [("", "")]
        fh.write(lab + " & " + " & ".join(x[0] for x in cc) + " \\\\\n & " + " & ".join(x[1] for x in cc) + " \\\\\n")
    fh.write("\\multicolumn{6}{l}{\\emph{F. Speed above the band by time of day (Istanbul), specification (a)}} \\\\\n")
    for lab, k_ in [("00:00--09:00 (banks closed)", "kp_night"), ("09:00--17:00 (banks open)", "kp_day"),
                    ("17:00--24:00 (banks closed)", "kp_eve")]:
        cc = [cell(BLOCK[p][k_]) for p in YRS] + [("", "")]
        fh.write(lab + " & " + " & ".join(x[0] for x in cc) + " \\\\\n & " + " & ".join(x[1] for x in cc) + " \\\\\n")
    fh.write("\\midrule\n$\\beta$ (col.\\ 2 analogue) & " + " & ".join(f"{RES[p]['beta_col2']:.2f}" for p in YRS) + " & \\\\\n")
    fh.write("Share of intervals above the band, banks closed & " + " & ".join(
        f"{DESC[p]['closed']['share_above_monthly']:.2f}" for p in YRS) + " & \\\\\n")
    fh.write("Share of intervals above the band, banks open & " + " & ".join(
        f"{DESC[p]['open']['share_above_monthly']:.2f}" for p in YRS) + " & \\\\\n")
    fh.write("$N$ & " + " & ".join(f"{RES[p]['a']['n']:,}" for p in YRS) + " & \\\\\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ figure
R_ = pd.DataFrame([dict(mid=pd.Period(r["start"], "M").to_timestamp() + pd.Timedelta(days=91),
                        **{f"{s}_{w}": r[s][w][0] for s in ["a", "c", "flow"] for w in ["closed", "open"]},
                        **{f"{s}_{w}_se": r[s][w][1] for s in ["a", "c", "flow"] for w in ["closed", "open"]})
                   for r in ROLL])
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.8))
titles = {"a": "(a) Above the band, monthly bands", "c": "(b) Above the band, bands by bank state",
          "flow": "(c) Large trades toward the band, both edges"}
for j, s in enumerate(["a", "c", "flow"]):
    A = ax.flat[j]
    for w, col, mk, lab in [("open", C1, "o", "banks open"), ("closed", C2, "s", "banks closed")]:
        A.fill_between(R_.mid, R_[f"{s}_{w}"] - 1.96 * R_[f"{s}_{w}_se"], R_[f"{s}_{w}"] + 1.96 * R_[f"{s}_{w}_se"],
                       color=col, alpha=0.12, lw=0)
        A.plot(R_.mid, R_[f"{s}_{w}"], color=col, marker=mk, ms=3, lw=1.2, label=lab)
    A.axhline(0, color="k", lw=0.4)
    for yb_ in ["2025-01-01", "2026-01-01"]:
        A.axvline(pd.Timestamp(yb_), color=C3, lw=0.5, ls=":")
    A.set_title(titles[s], fontsize=9.5)
    A.set_ylabel(r"$\hat\kappa^+$ per 5 min" if s != "flow" else r"$\hat\theta$, s.d. per 100 bp")
    A.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); A.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    A.tick_params(axis="x", rotation=30, labelsize=8)
    A.set_xlabel("centre of six-month window", fontsize=8.5)
ax.flat[0].legend(frameon=False, fontsize=8, loc="upper left")
D = ax.flat[3]
for y_, col, mk in [("2024", "#bdbdbd", "^"), ("2025", "#737373", "s"), ("2026", "#252525", "o")]:
    hh = pd.Series({h: v["vol_mean_k"] for h, v in HOUR[y_].items() if isinstance(h, int)}).sort_index()
    D.plot(hh.index, hh.values, color=col, marker=mk, ms=3, lw=1.1, label=y_)
D.axvspan(9, 16.99, color=C1, alpha=0.08)
D.set_xticks(range(0, 24, 3)); D.set_xlabel("Istanbul hour"); D.set_ylabel("thousand USDT per 5 min")
D.set_title("(d) Volume by hour of the day", fontsize=9.5); D.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rob_{NAME}.pdf")); plt.close(fig)
print(f"total {time.time() - t0:.0f}s")
