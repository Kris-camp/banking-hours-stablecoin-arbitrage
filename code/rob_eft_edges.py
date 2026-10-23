"""Robustness: rails or desks? The speed of arbitrage at the edges of the EFT day.

EFT, the central bank's lira large-value system, opens at 08:30 and closes at 17:30
Istanbul on full business days (13:00 on half days). The paper codes "banks open" as 09:00-17:00. The
two clocks disagree in two half-hours: 08:30-09:00 and 17:00-17:30, when EFT is open but bank and
trading desks are (mostly) not. If the business-hours increase in the speed of arbitrage comes from
the rail itself, the speed should switch on at 08:30 and off at 17:30; if it comes from desk hours
(staff, treasury limits, OTC desks), it should switch on at 09:00 and off at 17:00.

DESIGN
Data. data_raw/a5.parquet (162,624 five-minute intervals) left-joined with data_raw/ofi_size and
inner-joined with data_raw/state_bands.parquet, built exactly as in empirics_mech.py. Index = start of
the five-minute interval (UTC). Istanbul time = Europe/Istanbul (UTC+3 all year). An interval belongs
to the half-hour bin that contains its start; its dependent variable is the change from the previous
interval's mean premium to its own.
Sample. "Ordinary full business days": Istanbul date Mon-Fri, not a weekday public holiday (HOL list of
empirics_mech.py) and not one of the six half days (HALF list of rob_by_period.py). Windows: sixteen
half-hour bins, 07:00-11:00 (8 bins) and 15:00-19:00 (8 bins) Istanbul time. Standard errors are
clustered by Istanbul date (identical to the UTC date inside these windows), with the finite-sample
correction of empirics_mech.ols.
Specifications (one stacked regression per specification and sample; every slope is bin-specific):
  P  Price side, as Table emp_flow col. 1: R = -s(dy - beta_hat*OFI), beta_hat = emp_numbers thr q;
     state bands (month x 09-17 state); regressors d_out x bin_k and d_in x bin_k (k = 1..16), bin
     intercepts and month fixed effects. kappa_k = coefficient on d_out x bin_k.
  K  kappa+ with monthly bands, as Table emp_main col. 4: dy on OFI x bin_k, -(b_{t-1}-u_m)_+ x bin_k,
     (l_m-b_{t-1})_+ x bin_k, bin intercepts and month FE. kappa+_k = coefficient on the up x bin_k term
     (kappa-_k reported too). Monthly bands do not change at 09:00/17:00, so K is the check that the
     band switch of the state bands in P does not create a jump at 09:00 or 17:00.
  F  Large-trade response, as Table emp_flow col. 2: A_L = -s * z_large (large-trade OFI standardized
     within month on the full sample), same regressors as P; coefficients x100 (s.d. per 100 bp).
Tests (each a linear combination from the same regression, joint day-clustered covariance):
  T1 = kappa(08:30-09:00) - kappa(08:00-08:30)   EFT opens, desks mostly not yet open.
  T2 = kappa(09:00-09:30) - kappa(08:30-09:00)   desks open, EFT already open.
  T3 = kappa(17:00-17:30) - kappa(17:30-18:00)   EFT still open vs closed, both after desk hours.
  T4 = kappa(16:30-17:00) - kappa(17:00-17:30)   desks close, EFT still open (mirror of T2).
  Rail interpretation: T1 > 0 and T3 > 0 (and T2 = T4 = 0). Pure desk-hours interpretation: T1 = 0,
  T2 > 0, T3 = 0 (and T4 > 0). One-sided interpretation, but two-sided p-values are reported.
  Secondary, for power: T1b = kappa(08:30-09:00) - mean kappa(07:00-08:30, three bins);
  T3b = kappa(17:00-17:30) - mean kappa(17:30-19:00, three bins); AVG = (T1+T3)/2; joint Wald test
  of T1 = T3 = 0.
  Reference gaps (the effect size a sharp switch at one edge would produce): G_m = mean kappa over
  09:00-11:00 (4 bins) - mean over 07:00-08:30 (3 bins); G_e = mean over 15:00-17:00 (4 bins) - mean
  over 17:30-19:00 (3 bins); G = (G_m+G_e)/2.
  Location share: theta = (E - N)/(D - N), E = mean kappa over the two EFT-only bins (08:30, 17:00),
  N = mean over the six no-EFT bins, D = mean over the eight desk bins; also morning-only and
  evening-only versions. theta = 1 under the pure rail interpretation, 0 under the pure desk interpretation.
  Delta-method s.e.; descriptive only (a ratio of estimates).
Power (reported for every test): MDE80 = 2.80 x s.e. (5% two-sided, 80% power) and the power to
detect an effect equal to the relevant reference gap (G_m for T1, T2, T1b; G_e for T3, T4, T3b; G for
AVG) and to half of it, using the point estimate of the gap.
Samples / checks (all three specifications in each):
  A  main: ordinary full business days, Mon-Fri; month FE + bin intercepts.
  B  Tue-Fri only (Monday mornings follow the weekend closure).
  C  Tue-Fri, also excluding the first business day after a weekday holiday.
  D  sample A with month x bin fixed effects (= separate regressions by bin, as the cycle in
     empirics_mech.py).
  E/F sample A split by U.S. daylight-saving regime (EDT / EST). In EST the NYSE opens at 17:30
     Istanbul, exactly at the EFT close, which biases T3 downward under a U.S.-session effect; in EDT
     both 17:00 and 17:30 bins are inside the NYSE session. Only T3/T4 are of interest in E/F.
Descriptives by bin (sample A): intervals, share outside the state band, mean d_out when outside,
sum of squared d_out (identifying variation), mean USDT volume, large-trade share of volume,
interbank tick count (fxv).
Supplementary block (stored separately in the JSON): (i) the linear pieces of theta, E-N (rail effect: EFT-only bins vs
no-EFT bins) and D-E (desk effect: desk bins vs EFT-only bins), morning, evening and pooled, because
the ratio theta is ill-conditioned when the gap D-N is imprecise; (ii) the desk-edge average
(T2+T4)/2; (iii) samples A and B without the 19 March - 30 April 2025 lira crisis; (iv) activity at
the edges: log five-minute USDT volume and the large-trade share of volume on bin dummies with month
FE, contrasts 08:30-08:00, 09:00-08:30, 17:00-17:30, 16:30-17:00, day-clustered; (v) power of each test
against the full-sample business-hours increment (emp_flow col. 1 for P, emp_main col. 4 for K, emp_flow
col. 2 for F) instead of the in-window gaps, which are small and imprecise, and the z-statistic
of each test against that increment ("full switch at this edge").
Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, data_raw/ofi_size/*.parquet, tables/emp_numbers.json,
tables/emp_mech.json.
Writes tables/rob_eft_edges.json, tables/rob_eft_edges.tex, figures/rob_eft_edges.pdf.
"""
import os, glob, json, time
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
NAME = "eft_edges"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
t0 = time.time()
OUT = dict(name=NAME)

# ------------------------------------------------------------------ data (as empirics_mech.py)
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()].sort_index()
a = a.join(z, how="left")
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
a = a.join(sb, how="inner")
assert len(a) == 162624
a["l_m"], a["u_m"] = a.l, a.u                      # monthly bands (emp_main)
a["l"], a["u"] = a.l_s, a.u_s                      # state bands (emp_flow / emp_state)
a["z_large"] = a.ofi_large / a.ofi_large.groupby(a.month).transform("std")
mid = (a.l + a.u) / 2
a["s"] = np.where(a.x > a.u, 1.0, np.where(a.x < a.l, -1.0, np.sign(a.x - mid)))
a["out"] = ((a.x > a.u) | (a.x < a.l)).astype(float)
a["d_out"] = np.maximum(a.x - a.u, 0) + np.maximum(a.l - a.x, 0)
a["d_in"] = np.where(a.out == 0, np.abs(a.x - mid), 0.0)
beta = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
a["R"] = -a.s * (a.y - beta * a.q)
a["A_L100"] = 100 * (-a.s * a.z_large)              # coefficients per 100 bp, as emp_flow
a["upm"] = -np.maximum(a.x - a.u_m, 0)
a["dnm"] = np.maximum(a.l_m - a.x, 0)
assert np.allclose(a.upm, a.up) and np.allclose(a.dnm, a.dn)

# ------------------------------------------------------------------ calendars (Istanbul dates)
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]                                    # copied from empirics_mech.py
HALF = ["2024-04-09", "2024-10-28", "2025-06-05", "2025-10-28", "2026-03-19", "2026-05-26"]   # rob_by_period.py
HOLd, HALFd = set(pd.to_datetime(HOL).date), set(pd.to_datetime(HALF).date)
ist = a.index.tz_convert("Europe/Istanbul")
ny = a.index.tz_convert("America/New_York")
a["ist_min"] = np.asarray(ist.hour * 60 + ist.minute)
a["ist_date"] = ist.date
a["dow"] = np.asarray(ist.dayofweek)
a["day"] = a.ist_date                               # cluster
a["hol"] = pd.Series(a.ist_date).isin(HOLd).values
a["half"] = pd.Series(a.ist_date).isin(HALFd).values
a["post_hol"] = pd.Series(pd.to_datetime(a.ist_date) - pd.Timedelta(days=1)).dt.date.isin(HOLd).values
a["edt"] = (ny.tz_localize(None) - a.index.tz_localize(None) == pd.Timedelta(hours=-4))
a["ord"] = (a.dow < 5) & ~a.hol & ~a.half
assert np.allclose(a.bh.values, ((a.ist_min >= 540) & (a.ist_min < 1020)).astype(float))

win = ((a.ist_min >= 420) & (a.ist_min < 660)) | ((a.ist_min >= 900) & (a.ist_min < 1140))
w = a[win & a.ord].copy()
w["hb"] = w.ist_min // 30
lab = lambda k: f"{k // 2:02d}:{30 * (k % 2):02d}"
BINS = list(range(14, 22)) + list(range(30, 38))
LAB = [lab(k) for k in BINS]
w["bin"] = w.hb.map(lab)
w["mb"] = w.month + "_" + w.bin
assert set(w.bin.unique()) == set(LAB)
OUT["sample"] = dict(n_window=int(len(w)), n_days=int(w.day.nunique()),
                     n_days_mon=int(w.loc[w.dow == 0, "day"].nunique()),
                     n_days_post_hol=int(w.loc[w.post_hol & (w.dow > 0), "day"].nunique()),
                     beta_hat=float(beta), edt_share=float(w.edt.mean()))


# ------------------------------------------------------------------ estimator (empirics_mech.ols conventions)
def demean(M, g):
    return M - pd.DataFrame(M).groupby(g).transform("mean").values


def fit(d, ycol, svars, fe="month"):
    """ycol on svar x bin for every bin (+ bin intercepts when fe is month); within-FE OLS, clustered by day."""
    hb = d.bin.values
    labs = [l_ for l_ in LAB if (hb == l_).any()]
    cols, X = [], []
    for v in svars:
        for l_ in labs:
            X.append(d[v].values * (hb == l_)); cols.append(f"{v}|{l_}")
    if fe == "month":
        for l_ in labs[1:]:
            X.append((hb == l_).astype(float)); cols.append(f"bin|{l_}")
    X = np.column_stack(X).astype(float); Y = d[ycol].values.astype(float)
    g = pd.factorize(d[fe])[0]
    X = demean(X, g); Y = demean(Y[:, None], g)[:, 0]
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=pd.Series(b, cols), V=pd.DataFrame(V, cols, cols), n=int(n), n_cl=int(ng), labs=labs)


def lc(r, wts):
    wv = pd.Series(wts, dtype=float)
    est = float((r["b"][wv.index] * wv).sum())
    se = float(np.sqrt(wv.values @ r["V"].loc[wv.index, wv.index].values @ wv.values))
    return est, se


SPECS = {"P": ("R", ["d_out", "d_in"], "d_out"),
         "K": ("y", ["q", "upm", "dnm"], "upm"),
         "F": ("A_L100", ["d_out", "d_in"], "d_out")}
PRE = ["07:00", "07:30", "08:00"]; POST = ["17:30", "18:00", "18:30"]
DESK_M = ["09:00", "09:30", "10:00", "10:30"]; DESK_E = ["15:00", "15:30", "16:00", "16:30"]
EFTONLY = ["08:30", "17:00"]


def tests(r, v):
    k = lambda l_: f"{v}|{l_}"
    avail = set(r["labs"])
    T = {}
    def add(name, wts):
        if all(l_ in avail for l_ in wts):
            T[name] = lc(r, {k(l_): x for l_, x in wts.items()})
    add("T1", {"08:30": 1, "08:00": -1}); add("T2", {"09:00": 1, "08:30": -1})
    add("T3", {"17:00": 1, "17:30": -1}); add("T4", {"16:30": 1, "17:00": -1})
    add("T1b", {"08:30": 1, **{l_: -1 / 3 for l_ in PRE}}); add("T3b", {"17:00": 1, **{l_: -1 / 3 for l_ in POST}})
    add("AVG", {"08:30": .5, "08:00": -.5, "17:00": .5, "17:30": -.5})
    add("G_m", {**{l_: 1 / 4 for l_ in DESK_M}, **{l_: -1 / 3 for l_ in PRE}})
    add("G_e", {**{l_: 1 / 4 for l_ in DESK_E}, **{l_: -1 / 3 for l_ in POST}})
    add("G", {**{l_: 1 / 8 for l_ in DESK_M + DESK_E}, **{l_: -1 / 6 for l_ in PRE + POST}})
    out = {n_: dict(est=e_, se=s_, z=e_ / s_, p=float(2 * norm.sf(abs(e_ / s_)))) for n_, (e_, s_) in T.items()}
    # joint Wald T1 = T3 = 0
    if "T1" in T and "T3" in T:
        Wm = np.zeros((2, len(r["b"])))
        ix = {c: i for i, c in enumerate(r["b"].index)}
        Wm[0, ix[k("08:30")]], Wm[0, ix[k("08:00")]] = 1, -1
        Wm[1, ix[k("17:00")]], Wm[1, ix[k("17:30")]] = 1, -1
        t = Wm @ r["b"].values; VV = Wm @ r["V"].values @ Wm.T
        W = float(t @ np.linalg.solve(VV, t)); out["wald_T1_T3"] = dict(W=W, p=float(chi2.sf(W, 2)))
    # power
    ref = {"T1": "G_m", "T2": "G_m", "T1b": "G_m", "T3": "G_e", "T4": "G_e", "T3b": "G_e", "AVG": "G"}
    for n_, g_ in ref.items():
        if n_ in out and g_ in out:
            se, gap = out[n_]["se"], out[g_]["est"]
            pw = lambda dlt: float(norm.cdf(abs(dlt) / se - 1.96) + norm.cdf(-abs(dlt) / se - 1.96))
            out[n_].update(mde80=2.8 * se, ref=g_, ref_gap=gap, power_full_gap=pw(gap), power_half_gap=pw(gap / 2),
                           mde80_over_gap=2.8 * se / gap if gap != 0 else np.nan)
    # location share theta (delta method on bin coefficients)
    def share(Eb, Nb, Db, tag):
        if not all(l_ in avail for l_ in Eb + Nb + Db): return
        cE = {k(l_): 1 / len(Eb) for l_ in Eb}; cN = {k(l_): 1 / len(Nb) for l_ in Nb}; cD = {k(l_): 1 / len(Db) for l_ in Db}
        bb = r["b"]; E_ = sum(bb[c] * x for c, x in cE.items()); N_ = sum(bb[c] * x for c, x in cN.items())
        D_ = sum(bb[c] * x for c, x in cD.items())
        th = (E_ - N_) / (D_ - N_)
        grad = pd.Series(0.0, index=bb.index)
        for c, x in cE.items(): grad[c] += x / (D_ - N_)
        for c, x in cD.items(): grad[c] += -x * (E_ - N_) / (D_ - N_) ** 2
        for c, x in cN.items(): grad[c] += x * (-(D_ - N_) + (E_ - N_)) / (D_ - N_) ** 2
        se = float(np.sqrt(grad.values @ r["V"].values @ grad.values))
        out[f"theta_{tag}"] = dict(est=float(th), se=se, E=float(E_), N=float(N_), D=float(D_))
    share(EFTONLY, PRE + POST, DESK_M + DESK_E, "pooled")
    share(["08:30"], PRE, DESK_M, "morning")
    share(["17:00"], POST, DESK_E, "evening")
    return out


def bins_table(r, v, extra=None):
    rows = {}
    for l_ in r["labs"]:
        c = f"{v}|{l_}"; rows[l_] = [float(r["b"][c]), float(np.sqrt(r["V"].loc[c, c]))]
    return rows


SAMPLES = {"A_main": (w, "month"), "B_tue_fri": (w[w.dow > 0], "month"),
           "C_tue_fri_no_post_hol": (w[(w.dow > 0) & ~w.post_hol], "month"),
           "D_month_x_bin_fe": (w, "mb"),
           "E_edt": (w[w.edt], "month"), "F_est": (w[~w.edt], "month")}
RES = {}
for sname, (d, fe) in SAMPLES.items():
    RES[sname] = {}
    for spec, (ycol, svars, key) in SPECS.items():
        r = fit(d, ycol, svars, fe=fe)
        ent = dict(n=r["n"], n_days=r["n_cl"], bins=bins_table(r, key), tests=tests(r, key))
        if spec == "K":
            ent["bins_kminus"] = bins_table(r, "dnm"); ent["tests_kminus"] = tests(r, "dnm")
            ent["bins_beta"] = bins_table(r, "q")
        RES[sname][spec] = ent
    print(sname, f"{time.time() - t0:.0f}s", flush=True)
OUT["results"] = RES


# ------------------------------------------------------------------ supplementary block (see docstring)
def posthoc(r, v):
    k = lambda l_: f"{v}|{l_}"
    out = {}
    def mean_w(L, sgn):
        return {k(l_): sgn / len(L) for l_ in L}
    def add(name, *parts):
        wts = {}
        for p_ in parts:
            for c, x in p_.items(): wts[c] = wts.get(c, 0.0) + x
        e_, s_ = lc(r, wts); out[name] = dict(est=e_, se=s_, p=float(2 * norm.sf(abs(e_ / s_))))
    add("E_minus_N_morning", mean_w(["08:30"], 1), mean_w(PRE, -1)); add("D_minus_E_morning", mean_w(DESK_M, 1), mean_w(["08:30"], -1))
    add("E_minus_N_evening", mean_w(["17:00"], 1), mean_w(POST, -1)); add("D_minus_E_evening", mean_w(DESK_E, 1), mean_w(["17:00"], -1))
    add("E_minus_N_pooled", mean_w(EFTONLY, 1), mean_w(PRE + POST, -1)); add("D_minus_E_pooled", mean_w(DESK_M + DESK_E, 1), mean_w(EFTONLY, -1))
    add("T2_T4_avg", {k("09:00"): .5, k("08:30"): -.5, k("16:30"): .5, k("17:00"): -.5})
    return out


CRISIS = (pd.Timestamp("2025-03-19").date(), pd.Timestamp("2025-04-30").date())
nocr = ~((w.ist_date >= CRISIS[0]) & (w.ist_date <= CRISIS[1]))
PH = {"contrasts": {}, "no_crisis": {}, "activity": {}}
for sname in ["A_main", "B_tue_fri", "C_tue_fri_no_post_hol"]:
    d, fe = SAMPLES[sname]
    PH["contrasts"][sname] = {spec: posthoc(fit(d, ycol, svars, fe=fe), key) for spec, (ycol, svars, key) in SPECS.items()}
for sname, d in [("A_main_no_crisis", w[nocr]), ("B_tue_fri_no_crisis", w[nocr & (w.dow > 0)])]:
    PH["no_crisis"][sname] = {}
    for spec, (ycol, svars, key) in SPECS.items():
        r = fit(d, ycol, svars)
        PH["no_crisis"][sname][spec] = dict(n=r["n"], n_days=r["n_cl"], bins=bins_table(r, key),
                                            tests={t_: v_ for t_, v_ in tests(r, key).items() if not t_.startswith("theta")},
                                            post_hoc=posthoc(r, key))
w["lvol"] = np.log(w.vol.where(w.vol > 0))
w["lshare"] = w.vol_large / (w.vol_small + w.vol_mid + w.vol_large)
for sname, d in [("A_main", w), ("B_tue_fri", w[w.dow > 0])]:
    PH["activity"][sname] = {}
    for v_ in ["lvol", "lshare"]:
        r = fit(d.dropna(subset=[v_]), v_, [], fe="month")            # bin intercepts (07:00 omitted) and month FE
        PH["activity"][sname][v_] = {nm: dict(zip(["est", "se"], lc(r, {f"bin|{p}": 1, f"bin|{q}": -1})))
                                     for nm, p, q in [("08:30-08:00", "08:30", "08:00"), ("09:00-08:30", "09:00", "08:30"),
                                                      ("17:00-17:30", "17:00", "17:30"), ("16:30-17:00", "16:30", "17:00")]}
MECH = json.load(open(os.path.join(TAB, "emp_mech.json")))
NUMS = json.load(open(os.path.join(TAB, "emp_numbers.json")))
BENCH = {"P": MECH["price"]["d_out_bh"][0], "K": NUMS["reg"]["bh"]["coef"]["up_bh"], "F": 100 * MECH["flow"]["A_L"]["d_out_bh"][0]}
PH["benchmark_increment"] = BENCH
PH["power_vs_benchmark"] = {}
for sname in ["A_main", "B_tue_fri", "C_tue_fri_no_post_hol", "E_edt"]:
    PH["power_vs_benchmark"][sname] = {}
    for spec in "PKF":
        T, dl = RES[sname][spec]["tests"], BENCH[spec]
        PH["power_vs_benchmark"][sname][spec] = {
            t_: dict(power=float(norm.cdf(dl / T[t_]["se"] - 1.96) + norm.cdf(-dl / T[t_]["se"] - 1.96)),
                     z_vs_full_switch=float((T[t_]["est"] - dl) / T[t_]["se"]))
            for t_ in ["T1", "T2", "T3", "T4", "T1b", "T3b", "AVG"]}
OUT["post_hoc"] = PH

# ------------------------------------------------------------------ descriptives by bin (sample A)
desc = {}
for l_ in LAB:
    d = w[w.bin == l_]
    o = d[d.out == 1]
    desc[l_] = dict(n=int(len(d)), share_out=float(d.out.mean()), mean_dout_if_out=float(o.d_out.mean()),
                    sum_dout2=float((d.d_out ** 2).sum()), n_out_gt30=int((d.d_out > 30).sum()),
                    mean_vol=float(d.vol.mean()), large_share_vol=float(d.vol_large.sum() / (d.vol_small + d.vol_mid + d.vol_large).sum()),
                    mean_fxv=float(d.fxv.mean()) if "fxv" in d else None)
OUT["descriptives_A"] = desc
OUT["runtime_s"] = time.time() - t0
json.dump(OUT, open(os.path.join(TAB, f"rob_{NAME}.json"), "w"), indent=1, default=float)


# ------------------------------------------------------------------ table
def f(e, s, nd):
    t = abs(e / s) if s > 0 else 0
    st = "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""
    return f"${e:.{nd}f}{st}$", f"$({s:.{nd}f})$"


ND = {"P": 4, "K": 4, "F": 3}
with open(os.path.join(TAB, f"rob_{NAME}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccc}\n\\toprule\n & Price side, $\\hat\\kappa$ & $\\hat\\kappa^+$, monthly bands & Large trades, $\\hat\\theta$ \\\\\n"
             " & (1) & (2) & (3) \\\\\n\\midrule\n")
    fh.write("\\multicolumn{4}{l}{\\emph{A. Ordinary full business days, Monday--Friday}} \\\\\n")
    A = RES["A_main"]
    for l_, txt in [("08:00", "08:00--08:30 (EFT closed, desks closed)"), ("08:30", "08:30--09:00 (EFT open, desks closed)"),
                    ("09:00", "09:00--09:30 (EFT open, desks open)"), ("16:30", "16:30--17:00 (EFT open, desks open)"),
                    ("17:00", "17:00--17:30 (EFT open, desks closed)"), ("17:30", "17:30--18:00 (EFT closed, desks closed)")]:
        cells = [f(*A[s_]["bins"][l_], ND[s_]) for s_ in "PKF"]
        fh.write(f"Speed {txt} & " + " & ".join(c[0] for c in cells) + " \\\\\n & " + " & ".join(c[1] for c in cells) + " \\\\\n")
    fh.write("\\addlinespace\n")
    TL = [("T1", "T1: 08:30 bin $-$ 08:00 bin (EFT opens)"), ("T2", "T2: 09:00 bin $-$ 08:30 bin (desks open)"),
          ("T3", "T3: 17:00 bin $-$ 17:30 bin (EFT closes)"), ("T4", "T4: 16:30 bin $-$ 17:00 bin (desks close)"),
          ("AVG", "(T1+T3)/2"), ("G_m", "Reference gap, morning: 09:00--11:00 $-$ 07:00--08:30"),
          ("G_e", "Reference gap, evening: 15:00--17:00 $-$ 17:30--19:00")]
    for t_, txt in TL:
        cells = [f(A[s_]["tests"][t_]["est"], A[s_]["tests"][t_]["se"], ND[s_]) for s_ in "PKF"]
        fh.write(f"{txt} & " + " & ".join(c[0] for c in cells) + " \\\\\n & " + " & ".join(c[1] for c in cells) + " \\\\\n")
    fh.write("Wald test T1 $=$ T3 $=0$, $p$-value & " + " & ".join(f"{A[s_]['tests']['wald_T1_T3']['p']:.2f}" for s_ in "PKF") + " \\\\\n")
    th = [A[s_]["tests"]["theta_pooled"] for s_ in "PKF"]
    fh.write("Share of the gap present in EFT-only bins, $\\vartheta$ & " + " & ".join(f"{t['est']:.2f} ({t['se']:.2f})" if t["D"] > t["N"] else "--" for t in th) + " \\\\\n")
    for t_ in ["T1", "T3"]:
        fh.write(f"Minimum detectable {t_} (80\\% power) & " + " & ".join(f"{A[s_]['tests'][t_]['mde80']:.{ND[s_]}f}" for s_ in "PKF") + " \\\\\n")
    fh.write("Full-sample business-hours increment & " + " & ".join(f"{BENCH[s_]:.{ND[s_]}f}" for s_ in "PKF") + " \\\\\n")
    for t_ in ["T1", "T3"]:
        fh.write(f"Power of {t_} against that increment & " + " & ".join(f"{PH['power_vs_benchmark']['A_main'][s_][t_]['power']:.2f}" for s_ in "PKF") + " \\\\\n")
    fh.write("$N$ (days) & " + " & ".join(f"{A[s_]['n']:,} ({A[s_]['n_days']})" for s_ in "PKF") + " \\\\\n")
    for sname, title in [("B_tue_fri", "B. Tuesday--Friday"), ("E_edt", "C. U.S. daylight-saving time only (NYSE opens 16:30 Istanbul)")]:
        S = RES[sname]
        fh.write("\\midrule\n\\multicolumn{4}{l}{\\emph{" + title + "}} \\\\\n")
        for t_ in (["T1", "T2", "T3", "T4"] if sname == "B_tue_fri" else ["T3", "T4"]):
            cells = [f(S[s_]["tests"][t_]["est"], S[s_]["tests"][t_]["se"], ND[s_]) for s_ in "PKF"]
            fh.write(f"{t_} & " + " & ".join(c[0] for c in cells) + " \\\\\n & " + " & ".join(c[1] for c in cells) + " \\\\\n")
        if sname == "B_tue_fri":
            for t_ in ["T1", "T3"]:
                fh.write(f"Power of {t_} against the increment & " + " & ".join(f"{PH['power_vs_benchmark'][sname][s_][t_]['power']:.2f}" for s_ in "PKF") + " \\\\\n")
        fh.write("$N$ (days) & " + " & ".join(f"{S[s_]['n']:,} ({S[s_]['n_days']})" for s_ in "PKF") + " \\\\\n")
    fh.write("\\midrule\n\\multicolumn{4}{l}{\\emph{D. EFT-only half-hours against their neighbours (both edges pooled)}} \\\\\n")
    for sname, tag in [("A_main", "Mon--Fri"), ("B_tue_fri", "Tue--Fri")]:
        for c_, txt in [("E_minus_N_pooled", "EFT-only $-$ EFT closed"), ("D_minus_E_pooled", "Desk hours $-$ EFT-only")]:
            Q = PH["contrasts"][sname]
            cells = [f(Q[s_][c_]["est"], Q[s_][c_]["se"], ND[s_]) for s_ in "PKF"]
            fh.write(f"{txt}, {tag} & " + " & ".join(c[0] for c in cells) + " \\\\\n & " + " & ".join(c[1] for c in cells) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ figure
xm = {l_: int(l_[:2]) + int(l_[3:]) / 60 + 0.25 for l_ in LAB}       # bin centre, hours
fig, ax = plt.subplots(3, 2, figsize=(7.4, 7.6), sharex="col", gridspec_kw=dict(wspace=0.18, hspace=0.32))
rows = [("P", r"(a) price side, $\hat\kappa$ per 5 min"), ("K", r"(b) $\hat\kappa^+$, monthly bands"),
        ("F", r"(c) large-trade flow, $\hat\theta$ (s.d. per 100 bp)")]
for i, (s_, ttl) in enumerate(rows):
    for j, (lo, hi, labs) in enumerate([(7, 11, LAB[:8]), (15, 19, LAB[8:])]):
        A_ = ax[i, j]
        A_.axvspan(9 if j == 0 else 15, 11 if j == 0 else 17, color=C3, alpha=0.12, lw=0)
        A_.axvspan(8.5 if j == 0 else 17, 9 if j == 0 else 17.5, color=C2, alpha=0.10, lw=0)
        for sname, col, mk, off, lb in [("A_main", C1, "o", -0.04, "Mon--Fri"), ("B_tue_fri", C2, "s", 0.04, "Tue--Fri")]:
            B_ = RES[sname][s_]["bins"]
            xs = np.array([xm[l_] + off for l_ in labs]); ys = np.array([B_[l_][0] for l_ in labs])
            es = np.array([1.96 * B_[l_][1] for l_ in labs])
            A_.errorbar(xs, ys, yerr=es, fmt=mk + "-", ms=4, color=col, ecolor=col, lw=0.9, elinewidth=0.8,
                        capsize=0, alpha=0.95 if sname == "A_main" else 0.7, label=lb)
        A_.axhline(0, color="k", lw=0.4)
        for v_ in ([8.5, 9] if j == 0 else [17, 17.5]):
            A_.axvline(v_, color="k", lw=0.5, ls=":")
        A_.set_xlim(lo, hi); A_.set_xticks(range(lo, hi + 1))
        A_.set_xticklabels([f"{h:02d}:00" for h in range(lo, hi + 1)])
        if j == 0: A_.set_title(ttl, fontsize=9, loc="left")
        if i == 2: A_.set_xlabel("Istanbul time, half-hour bins")
    lims = [min(ax[i, 0].get_ylim()[0], ax[i, 1].get_ylim()[0]), max(ax[i, 0].get_ylim()[1], ax[i, 1].get_ylim()[1])]
    ax[i, 0].set_ylim(lims); ax[i, 1].set_ylim(lims)
ax[0, 0].text(8.75, 1.0, "EFT\nonly", transform=ax[0, 0].get_xaxis_transform(), ha="center", va="top", fontsize=7, color=C2)
ax[0, 1].text(17.25, 1.0, "EFT\nonly", transform=ax[0, 1].get_xaxis_transform(), ha="center", va="top", fontsize=7, color=C2)
ax[0, 0].text(10, 1.0, "desk hours", transform=ax[0, 0].get_xaxis_transform(), ha="center", va="top", fontsize=7, color=C3)
ax[0, 1].text(16, 1.0, "desk hours", transform=ax[0, 1].get_xaxis_transform(), ha="center", va="top", fontsize=7, color=C3)
ax[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.savefig(os.path.join(FIG, f"rob_{NAME}.pdf"), bbox_inches="tight"); plt.close(fig)
print(f"done {time.time() - t0:.0f}s")
