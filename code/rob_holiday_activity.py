"""Robustness: is the holiday placebo a thin-trading artefact?

Weekday public holidays are also low-activity days. The finding that the
business-hours increment in the speed of arbitrage is zero on holidays (Table emp_flow, col. 1
with a holiday interaction; emp_mech.json["holiday"]) could then reflect thin trading rather than
closed banks.

DESIGN
------
Sample, bands, dependent variable and regressors exactly as in empirics_mech.py: a5.parquet joined
with ofi_size and the month x banking-state bands; R = -s (dy - beta_hat q); regressors
d_out, d_out*bh_eff, d_out*bh_hol, d_in, d_in*bh_eff, d_in*bh_hol; month fixed effects;
day-clustered s.e. with the finite-sample correction of ols() in empirics_mech.py.
The ordinary-day increment is the coefficient on d_out*bh_eff ("inc_ord"), the holiday increment
the coefficient on d_out*bh_hol ("inc_hol"); "gap" = inc_ord - inc_hol.

A. Activity profile. Groups (Istanbul time, UTC+3):
   HOL_BH  09:00-17:00 on the weekday holidays (bh=1, hol=1; identical to the regression's bh_hol);
   ORD_BH  09:00-17:00 on ordinary weekdays (bh_eff=1);
   ORD_NIGHT 00:00-09:00 and ORD_EVE 17:00-24:00 on ordinary Istanbul weekdays (Istanbul date not a
   holiday, Istanbul weekday Mon-Fri); the panel already excludes 23:30-01:30 Istanbul (rollover).
   Measures per five-minute interval: USDT volume; number of trades (Binance kline trade count);
   number of aggressive orders (aggTrades); large-trade (>=10k USDT) share of aggressive volume;
   |dy| and SD(dy) of the premium; share of intervals with the lagged premium outside the band;
   mean d_out; interbank tick activity fxv. Means by group; HOL_BH minus ORD_BH and HOL_BH minus
   ORD_NIGHT differences with month fixed effects and day-clustered s.e.

B. Re-running the placebo at matched activity. Activity = month-relative log volume
   lv_rel = log(vol) - month mean of log(vol) (removes the trend in market size across the
   holiday months). Deciles are deciles of lv_rel over the full panel.
   (i-a) PRIMARY: WLS where ORD_BH intervals are reweighted so that their distribution across lv_rel
         deciles equals that of HOL_BH intervals (weight = holiday share / ordinary share in the
         decile, normalised so ORD_BH weights sum to the ORD_BH count); all other intervals weight 1.
   (i-b) Coarsened exact matching: same, with cells = month x lv_rel decile (ORD_BH intervals in
         cells with no holiday interval get weight 0).
   (i-c) As (i-a) with deciles of raw log volume.
   (i-d) As (i-a) with deciles of month-relative log number of trades.
   (i-e) As (i-a) with twenty bins (ventiles) of lv_rel.
   (i-f) As (i-a) but closed-hours intervals are also reweighted to the HOL_BH decile distribution,
         so all three states are compared at the holiday activity profile.
   (ii-a) Unweighted, adding lv_rel, d_out*lv_rel and d_in*lv_rel.
   (ii-b) Unweighted, d_out and d_in interacted with lv_rel decile dummies (speed free to vary
          non-parametrically with activity), decile main effects.
   (ii-c) (ii-a) plus the same interactions for month-relative log trade count and month-relative
          log(1+fxv).
   (ii-d) Volume slopes specific to each state (closed / ordinary open / holiday open):
          d_out*lv_rel*{closed, bh_eff, bh_hol}; increments evaluated at the HOL_BH mean of lv_rel
          (linear combinations with full covariance).
   (iii) Descriptive heterogeneity for the figure: kappa outside the band by group
         (night, evening, ORD_BH, HOL_BH) x lv_rel quintile, one regression, month FE.

Each row records whether the holiday result "survives activity matching" (key "survives"): the
ordinary-day increment is positive and significant at 5% AND the gap inc_ord - inc_hol is positive and
significant at 5%; (i-a) and (ii-a) are the main rows.
Weights are treated as fixed (no adjustment of s.e. for estimated weights).

C. Predetermined activity measures (block P1-P2 in the code, stored separately in the JSON).
   Contemporaneous volume contains the arbitrage trades themselves, so conditioning on it may condition
   on the outcome; P1 and P2 address this (see the comments there).

Inputs: data_raw/a5.parquet, data_raw/ofi_size/*.parquet, data_raw/state_bands.parquet, data_raw/panel_TRY.parquet,
tables/emp_numbers.json.
Outputs: tables/rob_holiday_activity.json, tables/rob_holiday_activity.tex,
tables/rob_holiday_activity_profile.tex, figures/rob_holiday_activity.pdf.
"""
import os, glob, json, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
NAME = "rob_holiday_activity"
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4 = "#1f4e79", "#c0504d", "#7f7f7f", "#e3a33b"
OUT = {}


# ------------------------------------------------------------------ data, identical to empirics_mech.py
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
a["day"] = a.index.date
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
a = a.join(sb, how="inner")
a["l"], a["u"] = a.l_s, a.u_s
m = (a.l + a.u) / 2
a["s"] = np.where(a.x > a.u, 1.0, np.where(a.x < a.l, -1.0, np.sign(a.x - m)))
a["out"] = ((a.x > a.u) | (a.x < a.l)).astype(float)
a["d_out"] = np.maximum(a.x - a.u, 0) + np.maximum(a.l - a.x, 0)
a["d_in"] = np.where(a.out == 0, np.abs(a.x - m), 0.0)
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
hol = pd.to_datetime(HOL).date
a["hol"] = np.isin(a.day, hol).astype(float)
a["bh_eff"] = a.bh * (1 - a.hol)
a["bh_hol"] = a.bh * a.hol
beta = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
a["R"] = -a.s * (a.y - beta * a.q)
for c in ["d_out", "d_in"]:
    a[c + "_bh"] = a[c] * a.bh_eff
    a[c + "_hol"] = a[c] * a.bh_hol
HC = ["d_out", "d_out_bh", "d_out_hol", "d_in", "d_in_bh", "d_in_hol"]

# ------------------------------------------------------------------ activity measures
p = pd.read_parquet(os.path.join(RAW, "panel_TRY.parquet"), columns=["vol", "ntrades"])
p5 = p.resample("5min", label="left", closed="left").sum()
a["ntr"] = p5.ntrades.reindex(a.index).values          # Binance kline trade count, full 5 minutes
a["vol5"] = p5.vol.reindex(a.index).values
a["nagg"] = a.n_small + a.n_mid + a.n_large            # aggressive orders (aggTrades)
a["vagg"] = a.vol_small + a.vol_mid + a.vol_large
a["lshare"] = a.vol_large / a.vagg
a["absdy"] = a.y.abs()
a["lv"] = np.log(a.vol)
a["lv_rel"] = a.lv - a.groupby("month").lv.transform("mean")
a["ln_rel"] = np.log(a.ntr.clip(lower=1)); a["ln_rel"] -= a.groupby("month").ln_rel.transform("mean")
a["lf_rel"] = np.log1p(a.fxv.astype(float)); a["lf_rel"] -= a.groupby("month").lf_rel.transform("mean")
OUT["check_vol_a5_vs_klines_5min_corr"] = float(np.corrcoef(a.vol, a.vol5)[0, 1])
OUT["check_vol_a5_over_klines_median"] = float((a.vol / a.vol5).median())

ist = a.index + pd.Timedelta(hours=3)
a["ih"] = ist.hour
a["hol_I"] = np.isin(ist.date, hol).astype(float)
a["wkd_I"] = (ist.dayofweek < 5).astype(float)
grp = np.full(len(a), "", dtype=object)
grp[(a.bh == 1) & (a.hol == 1)] = "HOL_BH"
grp[(a.bh == 1) & (a.hol == 0)] = "ORD_BH"
grp[(a.ih < 9) & (a.hol_I == 0) & (a.wkd_I == 1)] = "ORD_NIGHT"
grp[(a.ih >= 17) & (a.hol_I == 0) & (a.wkd_I == 1)] = "ORD_EVE"
a["grp"] = grp
GR = ["HOL_BH", "ORD_BH", "ORD_NIGHT", "ORD_EVE"]
OUT["n_holidays_in_sample"] = int(a.loc[a.hol == 1, "day"].nunique())


# ------------------------------------------------------------------ estimator
def wols(d, ycol, cols, w=None, fe="month"):
    """(Weighted) OLS with fixed effects and day-clustered s.e.; finite-sample correction as in ols()."""
    if w is None:
        w = np.ones(len(d))
    w = np.asarray(w, float)
    keep = w > 0
    d = d[keep]; w = w[keep]
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    if fe:
        g = pd.factorize(d[fe])[0]
        sw = np.bincount(g, w)
        Y = Y - (np.bincount(g, w * Y) / sw)[g]
        X = X - np.column_stack([(np.bincount(g, w * X[:, j]) / sw)[g] for j in range(X.shape[1])])
    XtWXi = np.linalg.inv((X * w[:, None]).T @ X)
    b = XtWXi @ (X * w[:, None]).T @ Y
    e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * (w * e)[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtWXi @ (G.T @ G) @ XtWXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=pd.Series(b, cols), V=pd.DataFrame(V, cols, cols), n=int(n), ndays=int(ng))


def lincom(r, wts):
    c = pd.Series(0.0, r["b"].index)
    for k_, v_ in wts.items():
        c[k_] = v_
    return float(c @ r["b"]), float(np.sqrt(c @ r["V"] @ c))


def summ(r, extra=None):
    """ordinary increment, holiday increment, gap (with s.e.); closed speed."""
    eo = {"d_out_bh": 1} if extra is None else extra["ord"]
    eh = {"d_out_hol": 1} if extra is None else extra["hol"]
    gap = dict(eo); [gap.__setitem__(k_, gap.get(k_, 0) - v_) for k_, v_ in eh.items()]
    ec = {"d_out": 1} if extra is None else extra.get("closed", {"d_out": 1})
    o = dict(closed=lincom(r, ec), inc_ord=lincom(r, eo), inc_hol=lincom(r, eh), gap=lincom(r, gap), n=r["n"])
    o["survives"] = bool(o["inc_ord"][0] / o["inc_ord"][1] > 1.96 and o["gap"][0] / o["gap"][1] > 1.96)
    return o


# ------------------------------------------------------------------ A. activity profile
prof = {}
for g_ in GR:
    d = a[a.grp == g_]
    prof[g_] = dict(n=int(len(d)), days=int(d.day.nunique()),
                    vol_mean=float(d.vol.mean()), vol_median=float(d.vol.median()),
                    ntr_mean=float(d.ntr.mean()), nagg_mean=float(d.nagg.mean()),
                    lshare_volw=float(d.vol_large.sum() / d.vagg.sum()), lshare_mean=float(d.lshare.mean()),
                    absdy_mean=float(d.absdy.mean()), sd_dy=float(d.y.std()),
                    share_out=float(d.out.mean()), d_out_mean=float(d.d_out.mean()),
                    fxv_mean=float(d.fxv.mean()), fxv_median=float(d.fxv.median()),
                    lv_rel_mean=float(d.lv_rel.mean()))
OUT["profile"] = prof
# differences with month FE, day-clustered
diffs = {}
MEAS = ["vol", "ntr", "nagg", "lshare", "absdy", "out", "d_out", "fxv", "lv_rel"]
for base in ["ORD_BH", "ORD_NIGHT", "ORD_EVE"]:
    d = a[a.grp.isin(["HOL_BH", base])].copy()
    d["is_hol"] = (d.grp == "HOL_BH").astype(float)
    diffs[base] = {}
    for mm in MEAS:
        dd = d.dropna(subset=[mm])
        r = wols(dd, mm, ["is_hol"])
        diffs[base][mm] = [float(r["b"]["is_hol"]), float(np.sqrt(r["V"].loc["is_hol", "is_hol"]))]
OUT["diff_hol_minus"] = diffs

# where do holiday business-hour intervals sit in the activity distribution?
dec = pd.qcut(a.lv_rel, 10, labels=False)
a["dec"] = dec.values
OUT["hol_bh_decile_shares"] = a[a.grp == "HOL_BH"].dec.value_counts(normalize=True).sort_index().round(4).to_dict()
OUT["ord_bh_decile_shares"] = a[a.grp == "ORD_BH"].dec.value_counts(normalize=True).sort_index().round(4).to_dict()
OUT["night_decile_shares"] = a[a.grp == "ORD_NIGHT"].dec.value_counts(normalize=True).sort_index().round(4).to_dict()


# ------------------------------------------------------------------ B. placebo re-run
def match_w(cell, target_mask, reweight_mask):
    """weights for reweight_mask intervals so their distribution over `cell` matches target_mask's."""
    cell = pd.Series(np.asarray(cell), index=a.index)
    tgt = cell[target_mask].value_counts(normalize=True)
    src = cell[reweight_mask].value_counts(normalize=True)
    ratio = (tgt / src).reindex(src.index).fillna(0.0)
    w = np.ones(len(a))
    wr = cell[reweight_mask].map(ratio).fillna(0.0).values
    wr = wr * reweight_mask.sum() / wr.sum()
    w[reweight_mask.values] = wr
    return w


hb = (a.bh == 1) & (a.hol == 1)
ob = (a.bh == 1) & (a.hol == 0)
cl = a.bh == 0
res = {}
base = wols(a, "R", HC)
res["baseline"] = summ(base)
OUT["baseline_coefs"] = {c: [float(base["b"][c]), float(np.sqrt(base["V"].loc[c, c]))] for c in HC}
a["st"] = np.where(a.bh == 0, "c", np.where(a.hol == 1, "h", "o"))
a["ms"] = a.month + "_" + np.where(a.bh == 1, "open", "closed")
res["baseline_monthxstate_fe"] = summ(wols(a, "R", HC, fe="ms"))

res["i_a_lvrel_deciles"] = summ(wols(a, "R", HC, w=match_w(a.dec, hb, ob)))
a["mdec"] = a.month + "_" + a.dec.astype(str)
res["i_b_cem_month_x_decile"] = summ(wols(a, "R", HC, w=match_w(a.mdec, hb, ob)))
a["dec_raw"] = pd.qcut(a.lv, 10, labels=False).values
res["i_c_rawlv_deciles"] = summ(wols(a, "R", HC, w=match_w(a.dec_raw, hb, ob)))
a["dec_n"] = pd.qcut(a.ln_rel.rank(method="first"), 10, labels=False).values
res["i_d_ntrades_deciles"] = summ(wols(a, "R", HC, w=match_w(a.dec_n, hb, ob)))
a["vent"] = pd.qcut(a.lv_rel, 20, labels=False).values
res["i_e_lvrel_ventiles"] = summ(wols(a, "R", HC, w=match_w(a.vent, hb, ob)))
w_f = match_w(a.dec, hb, ob) * match_w(a.dec, hb, cl)
res["i_f_closed_also_reweighted"] = summ(wols(a, "R", HC, w=w_f))
# effective sample sizes of the reweighted ordinary business-hour block
OUT["ess_ord_bh"] = {}
for k_, cell in [("i_a", a.dec), ("i_b", a.mdec), ("i_e", a.vent)]:
    w = match_w(cell, hb, ob)[ob.values]
    OUT["ess_ord_bh"][k_] = float(w.sum() ** 2 / (w ** 2).sum())
OUT["n_ord_bh"] = int(ob.sum()); OUT["n_hol_bh"] = int(hb.sum())

# (ii) activity controls
a["lv_rel_out"] = a.d_out * a.lv_rel; a["lv_rel_in"] = a.d_in * a.lv_rel
res["ii_a_dout_x_logvol"] = summ(wols(a, "R", HC + ["lv_rel", "lv_rel_out", "lv_rel_in"]))
dcols = []
for k_ in range(1, 10):
    a[f"D{k_}"] = (a.dec == k_).astype(float)
    a[f"D{k_}_out"] = a.d_out * a[f"D{k_}"]; a[f"D{k_}_in"] = a.d_in * a[f"D{k_}"]
    dcols += [f"D{k_}", f"D{k_}_out", f"D{k_}_in"]
res["ii_b_dout_x_vol_decile_dummies"] = summ(wols(a, "R", HC + dcols))
for v_ in ["ln_rel", "lf_rel"]:
    a[v_ + "_out"] = a.d_out * a[v_]; a[v_ + "_in"] = a.d_in * a[v_]
res["ii_c_logvol_logtrades_logfxv"] = summ(wols(a, "R", HC + ["lv_rel", "lv_rel_out", "lv_rel_in", "ln_rel", "ln_rel_out",
                                                               "ln_rel_in", "lf_rel", "lf_rel_out", "lf_rel_in"]))
# (ii-d) state-specific volume slopes, increments at the holiday mean activity
a["lvo_c"] = a.lv_rel_out * (a.bh == 0); a["lvo_o"] = a.lv_rel_out * a.bh_eff; a["lvo_h"] = a.lv_rel_out * a.bh_hol
a["lvi_c"] = a.lv_rel_in * (a.bh == 0); a["lvi_o"] = a.lv_rel_in * a.bh_eff; a["lvi_h"] = a.lv_rel_in * a.bh_hol
a["lv_o"] = a.lv_rel * a.bh_eff; a["lv_h"] = a.lv_rel * a.bh_hol
rd = wols(a, "R", HC + ["lv_rel", "lv_o", "lv_h", "lvo_c", "lvo_o", "lvo_h", "lvi_c", "lvi_o", "lvi_h"])
L = float(a.loc[hb, "lv_rel"].mean())
res["ii_d_state_specific_slopes_at_hol_activity"] = summ(rd, extra=dict(
    ord={"d_out_bh": 1, "lvo_o": L, "lvo_c": -L}, hol={"d_out_hol": 1, "lvo_h": L, "lvo_c": -L},
    closed={"d_out": 1, "lvo_c": L}))
res["ii_d_state_specific_slopes_at_hol_activity"]["lv_rel_eval"] = L
res["ii_d_state_specific_slopes_at_hol_activity"]["vol_slopes"] = {c: [float(rd["b"][c]), float(np.sqrt(rd["V"].loc[c, c]))]
                                                                   for c in ["lvo_c", "lvo_o", "lvo_h"]}
OUT["placebo"] = res

# ------------------------------------------------------------------ (iii) speed by group x activity quintile
a["qn"] = pd.qcut(a.lv_rel, 5, labels=False).values
hcols, cells = [], []
for g_ in GR:
    for qq in range(5):
        msk = ((a.grp == g_) & (a.qn == qq)).astype(float)
        if (msk * a.out).sum() < 40: continue
        nm = f"{g_}_{qq}"
        a["o_" + nm] = a.d_out * msk; a["i_" + nm] = a.d_in * msk; a["c_" + nm] = msk
        hcols += ["o_" + nm, "i_" + nm]; cells.append((g_, qq, nm, int(msk.sum()), int((msk * a.out).sum())))
sub = a[a.grp.isin(GR)]
hcols_fe = hcols + ["c_" + c[2] for c in cells[1:]]
rh = wols(sub, "R", hcols_fe)
het = []
for g_, qq, nm, n_, no_ in cells:
    het.append(dict(grp=g_, q=qq, k=float(rh["b"]["o_" + nm]), se=float(np.sqrt(rh["V"].loc["o_" + nm, "o_" + nm])),
                    n=n_, n_out=no_, lv_rel=float(a.loc[(a.grp == g_) & (a.qn == qq), "lv_rel"].mean())))
OUT["speed_by_group_quintile"] = het
# pooled group speeds in the same specification (no activity split), for reference
gcols = []
for g_ in GR:
    msk = (a.grp == g_).astype(float)
    a["go_" + g_] = a.d_out * msk; a["gi_" + g_] = a.d_in * msk; a["gc_" + g_] = msk
    gcols += ["go_" + g_, "gi_" + g_]
sub = a[a.grp.isin(GR)]
rg = wols(sub, "R", gcols + ["gc_" + g_ for g_ in GR[1:]])
OUT["speed_by_group"] = {g_: [float(rg["b"]["go_" + g_]), float(np.sqrt(rg["V"].loc["go_" + g_, "go_" + g_]))] for g_ in GR}


# ==================================================================================================
# C. Predetermined activity measures. Contemporaneous volume contains the arbitrage trades themselves
# (large trades lean against the premium, Table emp_size), so conditioning on it may condition on the
# outcome. P1 checks whether volume responds to d_out, more so when banks are open. P2 repeats
# (i-a) and (ii-a), with the same criterion, using activity measures less contaminated by
# arbitrage: (a) log volume over the previous 60 minutes (predetermined), (b) contemporaneous
# small+medium (end-user) volume, (c) small+medium volume over the previous 60 minutes.
# ==================================================================================================
a = a.copy()
a["eu"] = a.vol_small + a.vol_mid
a["l_large"] = np.log1p(a.vol_large); a["l_eu"] = np.log1p(a.eu)
for c_ in ["l_large", "l_eu"]:
    a[c_] -= a.groupby("month")[c_].transform("mean")
assert a.index.is_monotonic_increasing
for src, nm in [("vol", "lag_lv"), ("eu", "lag_leu")]:
    v_ = a[src].rolling("60min", closed="left").mean()
    a[nm] = np.log(v_.where(v_ > 0))
    a[nm] -= a.groupby("month")[nm].transform("mean")
a["c_leu"] = a.l_eu
# P1: activity response to the dislocation, by banking state
HCD = HC + ["bh_eff", "bh_hol"]
OUT["posthoc_volume_response"] = {}
for dep in ["lv_rel", "l_large", "l_eu"]:
    r = wols(a, dep, HCD)
    OUT["posthoc_volume_response"][dep] = {c: [float(r["b"][c]), float(np.sqrt(r["V"].loc[c, c]))] for c in HCD}
# P2
ph = {}
for nm, lab in [("lag_lv", "lagged 60-min total volume"), ("c_leu", "contemporaneous small+medium volume"),
                ("lag_leu", "lagged 60-min small+medium volume")]:
    d = a.dropna(subset=[nm]).copy()
    d["pdec"] = pd.qcut(d[nm], 10, labels=False).values
    hb_ = (d.bh == 1) & (d.hol == 1); ob_ = (d.bh == 1) & (d.hol == 0)
    tg = d.loc[hb_, "pdec"].value_counts(normalize=True); sr = d.loc[ob_, "pdec"].value_counts(normalize=True)
    ratio = (tg / sr).reindex(sr.index).fillna(0.0)
    w = np.ones(len(d)); wr = d.loc[ob_, "pdec"].map(ratio).values; w[ob_.values] = wr * ob_.sum() / wr.sum()
    d["x_out"] = d.d_out * d[nm]; d["x_in"] = d.d_in * d[nm]
    ph[nm] = dict(label=lab, n=int(len(d)), hol_mean=float(d.loc[hb_, nm].mean()), ord_bh_mean=float(d.loc[ob_, nm].mean()),
                  night_mean=float(d.loc[d.grp == "ORD_NIGHT", nm].mean()), eve_mean=float(d.loc[d.grp == "ORD_EVE", nm].mean()),
                  hol_decile_shares=d.loc[hb_, "pdec"].value_counts(normalize=True).sort_index().round(4).to_dict(),
                  baseline_same_sample=summ(wols(d, "R", HC)),
                  reweight=summ(wols(d, "R", HC, w=w)),
                  interact=summ(wols(d, "R", HC + [nm, "x_out", "x_in"])))
OUT["posthoc_activity"] = ph


def het_by(col, nq=5):
    d = a.dropna(subset=[col]).copy()
    d["qq"] = pd.qcut(d[col], nq, labels=False).values
    cols_, cells_ = [], []
    newc = {}
    for g_ in GR:
        for qq in range(nq):
            msk = ((d.grp == g_) & (d.qq == qq)).astype(float)
            if (msk * d.out).sum() < 40: continue
            nm_ = f"{g_}_{qq}"
            newc["o_" + nm_] = d.d_out * msk; newc["i_" + nm_] = d.d_in * msk; newc["c_" + nm_] = msk
            cols_ += ["o_" + nm_, "i_" + nm_]; cells_.append((g_, qq, nm_, int(msk.sum()), int((msk * d.out).sum())))
    d = pd.concat([d.drop(columns=[c for c in newc if c in d.columns]), pd.DataFrame(newc)], axis=1)
    sb_ = d[d.grp.isin(GR)]
    r = wols(sb_, "R", cols_ + ["c_" + c[2] for c in cells_[1:]])
    return [dict(grp=g_, q=qq, k=float(r["b"]["o_" + nm_]), se=float(np.sqrt(r["V"].loc["o_" + nm_, "o_" + nm_])), n=n_, n_out=no_,
                 x=float(d.loc[(d.grp == g_) & (d.qq == qq), col].mean())) for g_, qq, nm_, n_, no_ in cells_]


OUT["posthoc_speed_by_group_quintile_lagged_enduser"] = het_by("lag_leu")

json.dump(OUT, open(os.path.join(TAB, NAME + ".json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ table
def cell(v):
    t = abs(v[0] / v[1])
    st = "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""
    return f"${v[0]:.4f}{st}$".replace("-0.0000", "0.0000")


labels = [("baseline", "Baseline (Table~\\ref{tab:emp_flow}, col.~1, holiday interaction)"),
          ("i_a_lvrel_deciles", "(i-a) Reweight to holiday volume deciles"),
          ("i_b_cem_month_x_decile", "(i-b) Exact match on month $\\times$ volume decile"),
          ("i_c_rawlv_deciles", "(i-c) Reweight, raw (not month-relative) volume deciles"),
          ("i_d_ntrades_deciles", "(i-d) Reweight, trade-count deciles"),
          ("i_e_lvrel_ventiles", "(i-e) Reweight, volume ventiles"),
          ("i_f_closed_also_reweighted", "(i-f) Closed hours also reweighted"),
          ("ii_a_dout_x_logvol", "(ii-a) $d^{out}\\times\\log$ volume"),
          ("ii_b_dout_x_vol_decile_dummies", "(ii-b) $d^{out}\\times$ volume-decile dummies"),
          ("ii_c_logvol_logtrades_logfxv", "(ii-c) $d^{out}\\times$ log volume, log trades, log interbank ticks"),
          ("ii_d_state_specific_slopes_at_hol_activity", "(ii-d) State-specific volume slopes, at holiday activity")]
PH = OUT["posthoc_activity"]
plab = {"lag_lv": "previous-hour volume", "c_leu": "small+medium volume", "lag_leu": "previous-hour small+medium volume"}
CC = ["closed", "inc_ord", "inc_hol", "gap"]
with open(os.path.join(TAB, NAME + ".tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n & Banks closed & \\multicolumn{2}{c}{Business-hours increment} & Difference \\\\\n"
             "\\cmidrule(lr){3-4}\n & $\\hat\\kappa$ & ordinary weekdays & weekday holidays & ordinary $-$ holiday \\\\\n\\midrule\n"
             "\\multicolumn{5}{l}{\\emph{A. Reweighting and interactions on current activity}} \\\\\n")
    for k_, lab in labels:
        r = res[k_]
        fh.write(lab + " & " + " & ".join(cell(r[c]) for c in CC) + " \\\\\n")
        fh.write(" & " + " & ".join(f"$({r[c][1]:.4f})$" for c in CC) + " \\\\\n")
    fh.write("\\midrule\n\\multicolumn{5}{l}{\\emph{B. Activity measures that exclude current arbitrage trades}} \\\\\n")
    for nm in ["lag_lv", "c_leu", "lag_leu"]:
        for kk, pre in [("reweight", "Reweight to holiday deciles of "), ("interact", "$d^{out}\\times$ ")]:
            r = PH[nm][kk]
            fh.write(pre + plab[nm] + " & " + " & ".join(cell(r[c]) for c in CC) + " \\\\\n")
            fh.write(" & " + " & ".join(f"$({r[c][1]:.4f})$" for c in CC) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(TAB, NAME + "_profile.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n & Holidays & \\multicolumn{3}{c}{Ordinary weekdays} \\\\\n\\cmidrule(lr){3-5}\n"
             " & 09--17 & 09--17 & 00--09 & 17--24 \\\\\n\\midrule\n")
    rows = [("Volume (thousand USDT)", "vol_mean", 1e-3, "{:.0f}"), ("Number of trades", "ntr_mean", 1, "{:.0f}"),
            ("Large-trade share of volume", "lshare_volw", 1, "{:.2f}"), ("Mean $|\\Delta b|$ (bp)", "absdy_mean", 1, "{:.2f}"),
            ("SD of $\\Delta b$ (bp)", "sd_dy", 1, "{:.2f}"), ("Share outside band", "share_out", 1, "{:.2f}"),
            ("Interbank ticks, mean", "fxv_mean", 1, "{:.0f}"), ("Interbank ticks, median", "fxv_median", 1, "{:.0f}"),
            ("Intervals", "n", 1, "{:,}"), ("Days", "days", 1, "{:,}")]
    for lab, k_, sc, fm in rows:
        fh.write(lab + " & " + " & ".join(fm.format(prof[g_][k_] * sc) for g_ in GR) + " \\\\\n")
    sp = OUT["speed_by_group"]
    fh.write("$\\hat\\kappa$ outside the band & " + " & ".join(f"{sp[g_][0]:.4f}" for g_ in GR) + " \\\\\n")
    fh.write(" & " + " & ".join(f"({sp[g_][1]:.4f})" for g_ in GR) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ figure
GSTY = [("ORD_NIGHT", C3, "^", "ordinary nights 00-09", -0.03), ("ORD_EVE", C4, "D", "ordinary evenings 17-24", -0.01),
        ("ORD_BH", C1, "o", "ordinary weekdays 09-17", 0.01), ("HOL_BH", C2, "s", "public holidays 09-17", 0.03)]
fig = plt.figure(figsize=(7.6, 6.4), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
ax0, ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
H = pd.DataFrame(het).rename(columns={"lv_rel": "x"}); H2 = pd.DataFrame(OUT["posthoc_speed_by_group_quintile_lagged_enduser"])
for A_, D_, xl, tt in [(ax0, H, "log volume", "(a) Speed by volume quintile"),
                       (ax1, H2, "lagged log end-user volume", "(b) By lagged end-user volume")]:
    for g_, col, mk, lab, off in GSTY:
        r = D_[(D_.grp == g_) & (D_.se < 0.012)]
        A_.errorbar(r.x + off, r.k, yerr=1.96 * r.se, fmt=mk + "-", ms=3.5, color=col, ecolor=col, lw=0.9, alpha=0.9, label=lab)
    A_.axhline(0, color="k", lw=0.4); A_.set_xlabel(xl + "\n(quintile means, month-relative)", fontsize=8)
    A_.set_title(tt, fontsize=9.5); A_.set_ylim(-0.02, 0.035)
ax0.set_ylabel(r"reversion speed outside the band, $\hat\kappa$", fontsize=8.5)
ax0.legend(frameon=False, fontsize=7, loc="upper left")
rows_ = [("baseline", res["baseline"]), ("(i-a) reweight, volume deciles", res["i_a_lvrel_deciles"]),
         ("(i-b) month x volume decile", res["i_b_cem_month_x_decile"]), ("(i-c) raw volume deciles", res["i_c_rawlv_deciles"]),
         ("(i-d) trade-count deciles", res["i_d_ntrades_deciles"]), ("(i-e) volume ventiles", res["i_e_lvrel_ventiles"]),
         ("(i-f) closed also reweighted", res["i_f_closed_also_reweighted"]), ("(ii-a) d_out x log volume", res["ii_a_dout_x_logvol"]),
         ("(ii-b) d_out x volume deciles", res["ii_b_dout_x_vol_decile_dummies"]),
         ("(ii-c) + trades, interbank ticks", res["ii_c_logvol_logtrades_logfxv"]),
         ("reweight, prev-hour volume", PH["lag_lv"]["reweight"]),
         ("reweight, small+medium volume", PH["c_leu"]["reweight"]),
         ("reweight, prev-hour small+medium", PH["lag_leu"]["reweight"])]
yy = np.arange(len(rows_))[::-1]
for j, (c, col, mk, lab) in enumerate([("inc_ord", C1, "o", "ordinary weekdays"), ("inc_hol", C2, "s", "weekday holidays")]):
    v = np.array([r[c][0] for _, r in rows_]); s_ = np.array([r[c][1] for _, r in rows_])
    ax2.errorbar(v, yy + (0.15 if j == 0 else -0.15), xerr=1.96 * s_, fmt=mk, ms=3.5, color=col, ecolor=col, lw=0.9, label=lab)
ax2.axvline(0, color="k", lw=0.4)
ax2.set_yticks(yy); ax2.set_yticklabels([n_ for n_, _ in rows_], fontsize=7.5)
ax2.set_xlabel(r"business-hours increment in $\hat\kappa$ (95% interval)", fontsize=8.5)
ax2.set_title("(c) Holiday placebo at matched activity", fontsize=9.5)
ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.005))
fig.savefig(os.path.join(FIG, NAME + ".pdf")); plt.close(fig)

pd.set_option("display.width", 200)
print(pd.DataFrame(prof).T.round(4).to_string())
print(json.dumps(diffs, indent=0))
print("hol deciles", OUT["hol_bh_decile_shares"]); print("ord deciles", OUT["ord_bh_decile_shares"])
print("night deciles", OUT["night_decile_shares"])
for k_, v in res.items():
    print(k_, {c: (np.round(v[c], 4).tolist() if isinstance(v[c], tuple) else v[c]) for c in ["closed", "inc_ord", "inc_hol", "gap", "survives"]})
print(H.round(4).to_string()); print(OUT["speed_by_group"]); print("ess", OUT["ess_ord_bh"], OUT["n_ord_bh"], OUT["n_hol_bh"])
print("checks", OUT["check_vol_a5_vs_klines_5min_corr"], OUT["check_vol_a5_over_klines_median"])

for nm, v in OUT["posthoc_activity"].items():
    print("panel B", nm, v["label"], "means hol/ord/night/eve", round(v["hol_mean"], 3), round(v["ord_bh_mean"], 3), round(v["night_mean"], 3), round(v["eve_mean"], 3))
    for k_ in ["baseline_same_sample", "reweight", "interact"]:
        print("   ", k_, {c: np.round(v[k_][c], 4).tolist() for c in ["closed", "inc_ord", "inc_hol", "gap"]}, v[k_]["survives"])
print(json.dumps(OUT["posthoc_volume_response"], indent=0))
print(pd.DataFrame(OUT["posthoc_speed_by_group_quintile_lagged_enduser"]).round(4).to_string())
