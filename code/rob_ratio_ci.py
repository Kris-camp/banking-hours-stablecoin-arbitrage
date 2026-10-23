"""rob_ratio_ci: how precisely are the paper's open/closed ratios estimated?

The headline ratios divide by closed-hours speeds that are imprecisely estimated:
  R1  kappa+ open/closed, monthly bands (Table emp_main, col 4):                 0.0104/0.0025 = 4.2
  R2  kappa+ open/closed, month x state bands (Table emp_state, col 1):           0.0084/0.0017 = 4.9
  R3  price-side speed outside the band, both sides, state bands, holidays coded
      as closed (Table emp_flow, col 1):                                           3.3
  R4  large-trade response toward the band, same sample (Table emp_flow, col 2):  4.6
      and the log difference R3 - R4 (-0.34, s.e. 0.51)
  R5  long-run impact beta/kappa+, closed over open (emp_main col 4):             257/56 = 4.6
  Extension (fixed bands only): Turkey and Mexico ratios in Table emp_mxn (4.2, 1.6, difference in
  logs 0.94) and the night/day speed shares (about 1/20 and 2/3).

DESIGN
 1. Point estimates must reproduce the existing tables (asserted below).
 2. For every ratio: (a) delta-method CI on the ratio, and on the log ratio exponentiated, from the
    day-clustered covariance with the paper's finite-sample correction; (b) Fieller CI by inverting
    N - theta*D = 0 (for R5, N = beta_c*kappa_o and D = beta_o*kappa_c are linearized by the delta
    method: a Fieller-type interval); (c) day-block bootstrap with bands held fixed: within each month,
    draw that month's number of days with replacement, B = 499, re-estimate every regression (including
    the beta used to net order flow out of R3), percentile CI.
    A ratio whose denominator can be <= 0 is bootstrapped through its inverse (closed/open, whose
    denominator, the open-hours speed, is precise) and inverted; a draw with closed speed <= 0 counts as
    an unboundedly large open/closed ratio.
 3. Full bootstrap: same resampling and the same day draws (rep r uses seed [SEED, r] in both), B = 499
    (>= 199 required). In every replication the monthly bands AND the month x state bands are
    re-profiled with model._ssr_grid exactly as in empirics_main.py / empirics_state.py (5% trimming,
    1 bp grid), then emp_main cols 2-4, emp_state col 1 and emp_flow cols 1-2 are re-estimated.
    Reported: bootstrap distributions of the BH increment above the band (emp_main col 4), the
    open/closed ratio R1, the in-band reversion coefficient (emp_main col 3), and R2-R5.
    Rationale: the band edges are kink points (Hansen 2017); they are root-n consistent, not
    super-consistent, so slope standard errors that condition on them can understate uncertainty.
 4. Confidence levels: 90% (headline) and 95%.
 5. Headline choice, stored in the JSON ("recommendation"). Candidates are the price-side speed ratios R1, R2, R3 (the
    claim is about the speed of arbitrage). The primary headline is the candidate whose 90% intervals
    are bounded above under Fieller AND both bootstraps; if several qualify, the one with the narrowest
    full-bootstrap 90% interval in logs; if none qualifies, the business-hours increment (a difference,
    always defined) becomes the headline and the ratio is stated only as a one-sided lower bound.
    The ratio is then stated as its point estimate in words with the full-bootstrap 90% interval.

Also reported: summaries of the large-trade coefficients of emp_flow col 2 and per-month band diagnostics in
the full bootstrap.

Extended-real ratios. Item 2 maps a draw with closed speed <= 0 to an unboundedly large ratio. The open-hours
speed is not positive in every draw either: for the large-trade response (R4) in the full bootstrap, 44 of 499
draws have an open-hours response <= 0. The percentile intervals are therefore taken on the ratio itself in the
extended reals (closed <= 0 -> +inf; open <= 0 with closed > 0 -> ratio <= 0), and the log difference R3 - R4
likewise. For any ratio whose open-hours speed is positive in every draw (R1, R5, all fixed-band draws) this is
the same as inverting the quantiles of closed/open. Also reported: boot_*_diag90, the share of draws below the
estimate and basic / bias-corrected percentile intervals, because the full-bootstrap distributions are not
centred on the point estimates.

Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, data_raw/ofi_size/*.parquet, tables/emp_numbers.json,
tables/emp_state.json, tables/emp_mech.json, tables/emp_mxn.json (and, for the extension, the raw Binance/Dukascopy
files through build_panel). Outputs:
tables/rob_ratio_ci.json, tables/rob_ratio_ci.tex, figures/rob_ratio_ci.pdf, and the bootstrap draws
tables/rob_ratio_ci_draws_fixed.csv, tables/rob_ratio_ci_draws_full.csv (the full bootstrap appends
to its CSV and resumes from it, so an interrupted run loses nothing).
Env: B_FIX (499), B_FULL (499), DO_MXN (1), SEED (20261022), NWORK (4: the full bootstrap runs in NWORK
single-threaded worker processes, each a re-invocation of this script with WORKER=k), OUT_TAB and OUT_FIG
(output folders, default tables/ and figures/).
"""
import os, sys, json, glob, time, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import _ssr_grid

RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
OTAB, OFIG = os.environ.get("OUT_TAB", TAB), os.environ.get("OUT_FIG", FIG)   # test runs write elsewhere
B_FIX = int(os.environ.get("B_FIX", 499))
B_FULL = int(os.environ.get("B_FULL", 499))
DO_MXN = int(os.environ.get("DO_MXN", 1))
SEED = int(os.environ.get("SEED", 20261022))
WORKER = os.environ.get("WORKER")          # set only in full-bootstrap worker processes
NWORK = int(os.environ.get("NWORK", 4))
NAME = "rob_ratio_ci"
F_FIX = os.path.join(OTAB, f"{NAME}_draws_fixed.csv")
F_FULL = os.path.join(OTAB, f"{NAME}_draws_full.csv")
Z90, Z95 = 1.6448536269514722, 1.959963984540054
OUT = dict(design=dict(B_fixed=B_FIX, B_full_target=B_FULL, seed=SEED,
                       resampling="days drawn with replacement within month (each month keeps its number of days)"))
T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:7.1f}s]", *s, flush=True)


# ====================================================================== data (as in the existing scripts)
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
a["day"] = a.index.date
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
a = a.join(sb, how="inner")
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z[["ofi_small", "ofi_mid", "ofi_large"]], how="left")
assert len(a) == 162624 and a.ofi_large.notna().all()

# HOL: weekday public holidays, copied from empirics_mech.py
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
a["hol"] = np.isin(a.day, pd.to_datetime(HOL).date).astype(float)
a["bh_eff"] = a.bh * (1 - a.hol)
a["z_large"] = a.ofi_large / a.ofi_large.groupby(a.month).transform("std")
a["z_dem"] = (a.ofi_small + a.ofi_mid) / (a.ofi_small + a.ofi_mid).groupby(a.month).transform("std")

# raw arrays used by every estimator
Y_ = a.y.values; X_ = a.x.values; Q_ = a.q.values; BH_ = a.bh.values; BHE_ = a.bh_eff.values
ZL_ = a.z_large.values
month_codes, months = pd.factorize(a.month, sort=True)
day_codes, days = pd.factorize(a.day, sort=True)
NM, ND = len(months), len(days)
day_month = np.zeros(ND, int); day_month[day_codes] = month_codes
state_ = BH_.astype(int)                       # 1 = open (09-17 Istanbul), 0 = closed
ms_codes = month_codes * 2 + state_            # month x state cells
days_in_month = [np.where(day_month == m)[0] for m in range(NM)]
day_start = np.r_[0, np.cumsum(np.bincount(day_codes, minlength=ND))]
assert np.all(np.diff(day_codes) >= 0), "rows must be sorted by day"


# ====================================================================== estimators
def profile(y, x, q, trim=0.05, step=1.0):
    """Band profiling, identical to empirics_main.profile / empirics_state.profile. Returns (l, u)."""
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]


def fe_ols(Y, X, grp, day, cols):
    """Within (fixed-effect) OLS with day-clustered covariance and the paper's finite-sample correction.
    Returns coefficients, covariance, per-day influence (G @ XtXi.T, scaled by sqrt(adj))."""
    g = pd.factorize(grp)[0]; cnt = np.bincount(g)
    dm = lambda v: v - (np.bincount(g, v, minlength=len(cnt)) / cnt)[g]
    Yw = dm(Y.astype(float)); Xw = np.column_stack([dm(X[:, j].astype(float)) for j in range(X.shape[1])])
    XtXi = np.linalg.inv(Xw.T @ Xw); b = XtXi @ Xw.T @ Yw; e = Yw - Xw @ b
    codes = pd.factorize(day)[0]; ng = codes.max() + 1
    Gs = np.zeros((ng, X.shape[1])); np.add.at(Gs, codes, Xw * e[:, None])
    n, k = len(Yw), X.shape[1]
    adj = ng / (ng - 1) * (n - 1) / (n - k)
    V = XtXi @ (Gs.T @ Gs) @ XtXi * adj
    return dict(b=dict(zip(cols, b)), bv=b, V=V, se=dict(zip(cols, np.sqrt(np.diag(V)))), cols=list(cols),
                infl=(Gs @ XtXi.T) * np.sqrt(adj), n=n)


def within_coef(Y, X, g):
    """Point estimates only (bootstrap). Y may have several columns."""
    cnt = np.bincount(g); keep = cnt > 0
    def dm(v):
        mu = np.zeros(len(cnt)); mu[keep] = np.bincount(g, v, minlength=len(cnt))[keep] / cnt[keep]
        return v - mu[g]
    Xw = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    Yw = np.column_stack([dm(Y[:, j]) for j in range(Y.shape[1])])
    return np.linalg.solve(Xw.T @ Xw, Xw.T @ Yw)


def build_design(y, x, q, bh, bhe, zl, l, u, ls, us, mcode):
    """All regressors of emp_main cols 2-4, emp_state col 1 and emp_flow cols 1-2 for given bands."""
    D = {}
    up = -np.maximum(x - u, 0); dn = np.maximum(l - x, 0); inb = ((x >= l) & (x <= u)).astype(float)
    D["c2"] = np.column_stack([q, up, dn])
    D["c3"] = np.column_stack([q, up, dn, -(x - (l + u) / 2) * inb])
    D["c4"] = np.column_stack([q, up, dn, q * bh, up * bh, dn * bh])
    ups = -np.maximum(x - us, 0); dns = np.maximum(ls - x, 0)
    D["st"] = np.column_stack([q, ups, dns, q * bh, ups * bh, dns * bh])
    m = (ls + us) / 2
    s = np.where(x > us, 1.0, np.where(x < ls, -1.0, np.sign(x - m)))
    dout = np.maximum(x - us, 0) + np.maximum(ls - x, 0)
    din = np.where(dout == 0, np.abs(x - m), 0.0)        # (out==0) <=> dout==0, as in empirics_mech
    D["fl"] = np.column_stack([dout, dout * bhe, din, din * bhe])
    D["fl_y"] = np.column_stack([-s * y, s * q, -s * zl])  # R = -s*y + beta*s*q ; A_L = -s*z_large
    D["share_in"] = inb.mean()
    return D


C2 = ["q", "up", "dn"]; C3 = ["q", "up", "dn", "negx_in"]
C4 = ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"]
CS = ["q", "up_s", "dn_s", "q_bh", "up_s_bh", "dn_s_bh"]
CF = ["d_out", "d_out_bh", "d_in", "d_in_bh"]


def derived(c2, c3, c4, st, fp, ff):
    """Scalar quantities recorded in every bootstrap replication (dicts of coefficients)."""
    kc, kb = c4["up"], c4["up_bh"]; bc, bb = c4["q"], c4["q_bh"]
    sc, sbh = st["up_s"], st["up_s_bh"]
    pc, pb = fp["d_out"], fp["d_out_bh"]; tc, tb = ff["d_out"], ff["d_out_bh"]
    r = dict(beta_c2=c2["q"], kp_c2=c2["up"], km_c2=c2["dn"], negx_in=c3["negx_in"],
             c4_q=bc, c4_q_bh=bb, c4_up=kc, c4_up_bh=kb, c4_dn=c4["dn"], c4_dn_bh=c4["dn_bh"],
             st_up_s=sc, st_up_s_bh=sbh, st_dn_s=st["dn_s"], st_dn_s_bh=st["dn_s_bh"], st_q=st["q"], st_q_bh=st["q_bh"],
             fp_c=pc, fp_bh=pb, fp_in=fp["d_in"], fp_in_bh=fp["d_in_bh"], ff_c=tc, ff_bh=tb)
    # inverse ratios closed/open (reported for reference only)
    r["inv_R1"] = kc / (kc + kb); r["inv_R2"] = sc / (sc + sbh); r["inv_R3"] = pc / (pc + pb); r["inv_R4"] = tc / (tc + tb)
    # R5: closed/open long-run impact = (bc/kc)/((bc+bb)/(kc+kb)); its inverse is open/closed
    r["inv_R5"] = (bc + bb) * kc / ((kc + kb) * bc)
    r["lr_closed"] = bc / kc if kc > 0 else np.inf; r["lr_open"] = (bc + bb) / (kc + kb)
    # Open/closed ratios in the extended reals (docstring item 2): a draw with closed speed <= 0 counts as an
    # unboundedly large ratio; a draw with closed speed > 0 and open speed <= 0 has a ratio <= 0 and belongs at the
    # BOTTOM of the distribution. The open-hours speed is not always positive: for the large-trade response (R4) in
    # the full bootstrap 44 of 499 draws have open-hours response <= 0, which also affects the R3-R4 log difference.
    rx = lambda o, c: o / c if c > 0 else np.inf
    r["R1"] = rx(kc + kb, kc); r["R2"] = rx(sc + sbh, sc); r["R3"] = rx(pc + pb, pc); r["R4"] = rx(tc + tb, tc)
    r["R5"] = rx(bc * (kc + kb) / (bc + bb), kc)       # = (bc/kc)/((bc+bb)/(kc+kb)); bc, bc+bb, kc+kb > 0 in every draw
    lx = lambda o, c: np.inf if c <= 0 else (-np.inf if o <= 0 else np.log(o / c))
    r["logdiff_R3_R4"] = lx(pc + pb, pc) - lx(tc + tb, tc)   # inf - inf (same sign) = nan, dropped
    return r


def rederive(df):
    """Recompute the derived ratio columns of stored draws from their stored coefficients (draws on disk written
    by an earlier version of derived() may use a different R4 / logdiff mapping)."""
    rows = []
    for _, x in df.iterrows():
        c2 = {"q": x.beta_c2, "up": x.kp_c2, "dn": x.km_c2}; c3 = {"negx_in": x.negx_in}
        c4 = {"q": x.c4_q, "q_bh": x.c4_q_bh, "up": x.c4_up, "up_bh": x.c4_up_bh, "dn": x.c4_dn, "dn_bh": x.c4_dn_bh}
        st = {"up_s": x.st_up_s, "up_s_bh": x.st_up_s_bh, "dn_s": x.st_dn_s, "dn_s_bh": x.st_dn_s_bh, "q": x.st_q, "q_bh": x.st_q_bh}
        fp = {"d_out": x.fp_c, "d_out_bh": x.fp_bh, "d_in": x.fp_in, "d_in_bh": x.fp_in_bh}
        ff = {"d_out": x.ff_c, "d_out_bh": x.ff_bh}
        rows.append(derived(c2, c3, c4, st, fp, ff))
    new = pd.DataFrame(rows, index=df.index)
    df = df.copy(); df[list(new.columns)] = new
    return df


# ====================================================================== 1. point estimates (reproduction)
l0, u0, ls0, us0 = a.l.values, a.u.values, a.l_s.values, a.u_s.values
D0 = build_design(Y_, X_, Q_, BH_, BHE_, ZL_, l0, u0, ls0, us0, month_codes)
r_c2 = fe_ols(Y_, D0["c2"], month_codes, day_codes, C2)
r_c3 = fe_ols(Y_, D0["c3"], month_codes, day_codes, C3)
r_c4 = fe_ols(Y_, D0["c4"], month_codes, day_codes, C4)
r_st = fe_ols(Y_, D0["st"], ms_codes, day_codes, CS)
beta0 = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
assert abs(beta0 - r_c2["b"]["q"]) < 1e-10
R0 = D0["fl_y"][:, 0] + beta0 * D0["fl_y"][:, 1]
r_fp = fe_ols(R0, D0["fl"], month_codes, day_codes, CF)
r_ff = fe_ols(D0["fl_y"][:, 2], D0["fl"], month_codes, day_codes, CF)

EN = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]
ES = json.load(open(os.path.join(TAB, "emp_state.json")))["state_bands"]["coef"]
EM = json.load(open(os.path.join(TAB, "emp_mech.json")))
for k in C4:
    assert abs(r_c4["b"][k] - EN["bh"]["coef"][k]) < 1e-9 and abs(r_c4["se"][k] - EN["bh"]["se"][k]) < 1e-9, k
assert abs(r_c3["b"]["negx_in"] - EN["thr_in"]["coef"]["negx_in"]) < 1e-10
for k in CS:
    assert abs(r_st["b"][k] - ES[k][0]) < 1e-9 and abs(r_st["se"][k] - ES[k][1]) < 1e-9, k
for k in CF:
    assert abs(r_fp["b"][k] - EM["price"][k][0]) < 1e-9 and abs(r_ff["b"][k] - EM["flow"]["A_L"][k][0]) < 1e-9, k
PT = derived(r_c2["b"], r_c3["b"], r_c4["b"], r_st["b"], r_fp["b"], r_ff["b"])
assert abs(PT["R3"] - EM["cross"]["ratio_price"]) < 1e-9 and abs(PT["R4"] - EM["cross"]["ratio_flow"]) < 1e-9
log("point estimates reproduce emp_main, emp_state, emp_flow:", {k: round(PT[k], 3) for k in ["R1", "R2", "R3", "R4", "R5"]})


# ====================================================================== 2. analytic intervals
def fieller(n, d, Vnn, Vdd, Vnd, z):
    """Fieller set for theta = n/d: {theta: (n - theta d)^2 <= z^2 (Vnn - 2 theta Vnd + theta^2 Vdd)}."""
    A = d * d - z * z * Vdd; B = n * d - z * z * Vnd; C = n * n - z * z * Vnn
    disc = B * B - A * C
    if A > 0:
        r = np.sqrt(max(disc, 0.0))
        return dict(kind="bounded", lo=(B - r) / A, hi=(B + r) / A)
    if disc < 0:
        return dict(kind="whole_line", lo=-np.inf, hi=np.inf)
    r1, r2 = sorted([(B - np.sqrt(disc)) / A, (B + np.sqrt(disc)) / A])
    # set is (-inf, r1] U [r2, inf); for a positive ratio the economically relevant part is [r2, inf) if r1 < 0
    return dict(kind="two_rays", ray_below=r1, ray_above=r2, lo=r2 if r1 < 0 else np.nan, hi=np.inf)


def lin_ratio_ci(b, V, cN, cD, z):
    n, d = cN @ b, cD @ b
    Vnn, Vdd, Vnd = cN @ V @ cN, cD @ V @ cD, cN @ V @ cD
    th = n / d
    g = cN / d - n * cD / d ** 2; se = np.sqrt(g @ V @ g)
    gl = cN / n - cD / d; sel = np.sqrt(gl @ V @ gl)
    return dict(est=th, se=se, delta=[th - z * se, th + z * se], se_log=sel,
                delta_log=[th * np.exp(-z * sel), th * np.exp(z * sel)], fieller=fieller(n, d, Vnn, Vdd, Vnd, z),
                num=n, den=d, se_num=np.sqrt(Vnn), se_den=np.sqrt(Vdd), t_den=d / np.sqrt(Vdd))


def num_grad(f, b, h=1e-7):
    g = np.zeros(len(b))
    for i in range(len(b)):
        e = np.zeros(len(b)); e[i] = h * max(abs(b[i]), 1e-6)
        g[i] = (f(b + e) - f(b - e)) / (2 * e[i])
    return g


def nonlin_ratio_ci(b, V, fN, fD, z):
    n, d = fN(b), fD(b); gN, gD = num_grad(fN, b), num_grad(fD, b)
    Vnn, Vdd, Vnd = gN @ V @ gN, gD @ V @ gD, gN @ V @ gD
    th = n / d; g = gN / d - n * gD / d ** 2; se = np.sqrt(g @ V @ g)
    gl = gN / n - gD / d; sel = np.sqrt(gl @ V @ gl)
    return dict(est=th, se=se, delta=[th - z * se, th + z * se], se_log=sel,
                delta_log=[th * np.exp(-z * sel), th * np.exp(z * sel)], fieller=fieller(n, d, Vnn, Vdd, Vnd, z),
                num=n, den=d, se_num=np.sqrt(Vnn), se_den=np.sqrt(Vdd), t_den=d / np.sqrt(Vdd))


def e(cols, *names, w=None):
    v = np.zeros(len(cols))
    for i, nm in enumerate(names):
        v[cols.index(nm)] += 1 if w is None else w[i]
    return v


# joint covariance of the price (R) and large-trade (A_L) regressions, as in empirics_mech (cross-equation)
Gj = np.hstack([r_fp["infl"], r_ff["infl"]]); Vj = Gj.T @ Gj
bj = np.r_[r_fp["bv"], r_ff["bv"]]; CJ = [f"p_{c}" for c in CF] + [f"f_{c}" for c in CF]
SPEC = {
    "R1": dict(label="kappa+ open/closed, monthly bands (emp_main col 4)", b=r_c4["bv"], V=r_c4["V"], cols=C4,
               cN=e(C4, "up", "up_bh"), cD=e(C4, "up")),
    "R2": dict(label="kappa+ open/closed, month x state bands (emp_state col 1)", b=r_st["bv"], V=r_st["V"], cols=CS,
               cN=e(CS, "up_s", "up_s_bh"), cD=e(CS, "up_s")),
    "R3": dict(label="speed outside band, both sides, open/closed (emp_flow col 1)", b=bj, V=Vj, cols=CJ,
               cN=e(CJ, "p_d_out", "p_d_out_bh"), cD=e(CJ, "p_d_out")),
    "R4": dict(label="large-trade response, open/closed (emp_flow col 2)", b=bj, V=Vj, cols=CJ,
               cN=e(CJ, "f_d_out", "f_d_out_bh"), cD=e(CJ, "f_d_out")),
}
ANA = {}
for k, s in SPEC.items():
    ANA[k] = {lev: lin_ratio_ci(s["b"], s["V"], s["cN"], s["cD"], z) for lev, z in [("90", Z90), ("95", Z95)]}
    ANA[k]["inv"] = {lev: lin_ratio_ci(s["b"], s["V"], s["cD"], s["cN"], z) for lev, z in [("90", Z90), ("95", Z95)]}
i4 = {c: C4.index(c) for c in C4}
fN5 = lambda b: b[i4["q"]] * (b[i4["up"]] + b[i4["up_bh"]])          # beta_c * kappa_o
fD5 = lambda b: (b[i4["q"]] + b[i4["q_bh"]]) * b[i4["up"]]           # beta_o * kappa_c
ANA["R5"] = {lev: nonlin_ratio_ci(r_c4["bv"], r_c4["V"], fN5, fD5, z) for lev, z in [("90", Z90), ("95", Z95)]}
ANA["R5"]["inv"] = {lev: nonlin_ratio_ci(r_c4["bv"], r_c4["V"], fD5, fN5, z) for lev, z in [("90", Z90), ("95", Z95)]}
# long-run impact levels (Fieller applies directly: ratios of linear combinations)
ANA["LR_closed"] = {lev: lin_ratio_ci(r_c4["bv"], r_c4["V"], e(C4, "q"), e(C4, "up"), z) for lev, z in [("90", Z90), ("95", Z95)]}
ANA["LR_open"] = {lev: lin_ratio_ci(r_c4["bv"], r_c4["V"], e(C4, "q", "q_bh"), e(C4, "up", "up_bh"), z) for lev, z in [("90", Z90), ("95", Z95)]}
# log difference of R3 and R4 (delta, joint covariance) - reproduces emp_flow
iC, iB = CJ.index("p_d_out"), CJ.index("p_d_out_bh"); jC, jB = CJ.index("f_d_out"), CJ.index("f_d_out_bh")
gld = np.zeros(len(CJ)); kC, kB, tC, tB = bj[iC], bj[iB], bj[jC], bj[jB]
gld[iC] = 1 / (kC + kB) - 1 / kC; gld[iB] = 1 / (kC + kB); gld[jC] = -(1 / (tC + tB) - 1 / tC); gld[jB] = -1 / (tC + tB)
LD = dict(est=float(np.log(PT["R3"]) - np.log(PT["R4"])), se=float(np.sqrt(gld @ Vj @ gld)))
assert abs(LD["est"] - EM["cross"]["logdiff"]) < 1e-9 and abs(LD["se"] - EM["cross"]["se_logdiff"]) < 1e-9
log("analytic intervals done")


# ====================================================================== 3. day-block bootstrap, bands fixed
def draw_counts(r):
    rng = np.random.default_rng([SEED, r])
    c = np.zeros(ND, int)
    for dm_ in days_in_month:
        c += np.bincount(dm_[rng.integers(0, len(dm_), len(dm_))], minlength=ND)
    return c


def cell_moments(Z, cell, ncell):
    p = Z.shape[1]; M = np.zeros((ncell, p, p))
    for i in range(p):
        for j in range(i, p):
            v = np.bincount(cell, Z[:, i] * Z[:, j], minlength=ncell); M[:, i, j] = v; M[:, j, i] = v
    return M


class Moments:
    """Per (day x FE-group) cross-products of [1, X, Y]; a bootstrap draw re-weights days and applies the
    within transformation group by group, which is exactly OLS with FE on the resampled rows."""
    def __init__(self, X, Y, grp):
        cell, cu = pd.factorize(pd.Series(day_codes) * 1000 + grp)
        self.k, self.ny = X.shape[1], Y.shape[1]
        self.M = cell_moments(np.column_stack([np.ones(len(X)), X, Y]), cell, len(cu))
        cu = np.asarray(cu); self.cday = cu // 1000
        g = pd.factorize(cu % 1000)[0]
        self.A = np.zeros((g.max() + 1, len(cu))); self.A[g, np.arange(len(cu))] = 1.0

    def coef(self, w):
        p = self.M.shape[1]
        Mg = (self.A @ (w[self.cday][:, None] * self.M.reshape(len(self.cday), -1))).reshape(-1, p, p)
        n = Mg[:, 0, 0]; ok = n > 0
        S = (Mg[ok, 1:, 1:] - Mg[ok, 1:, 0][:, :, None] * Mg[ok, 0, 1:][:, None, :] / n[ok][:, None, None]).sum(0)
        return np.linalg.solve(S[:self.k, :self.k], S[:self.k, self.k:])


MO = dict(c2=Moments(D0["c2"], Y_[:, None], month_codes), c3=Moments(D0["c3"], Y_[:, None], month_codes),
          c4=Moments(D0["c4"], Y_[:, None], month_codes), st=Moments(D0["st"], Y_[:, None], ms_codes),
          fl=Moments(D0["fl"], D0["fl_y"], month_codes))


def fixed_rep(w):
    c2 = dict(zip(C2, MO["c2"].coef(w)[:, 0])); c3 = dict(zip(C3, MO["c3"].coef(w)[:, 0]))
    c4 = dict(zip(C4, MO["c4"].coef(w)[:, 0])); st = dict(zip(CS, MO["st"].coef(w)[:, 0]))
    F = MO["fl"].coef(w)
    fp = dict(zip(CF, F[:, 0] + c2["q"] * F[:, 1]))     # R uses this replication's beta
    ff = dict(zip(CF, F[:, 2]))
    return derived(c2, c3, c4, st, fp, ff)


chk = fixed_rep(np.ones(ND))
for k in ["R1", "R2", "R3", "R4", "R5", "negx_in", "c4_up_bh"]:
    assert abs(chk[k] - PT[k]) < 1e-7 * max(1, abs(PT[k])), (k, chk[k], PT[k])
if WORKER is None:
    FX = pd.DataFrame([fixed_rep(draw_counts(r).astype(float)) for r in range(B_FIX)])
    FX.index.name = "rep"
    FX.to_csv(F_FIX)
    log(f"fixed-band bootstrap done: {B_FIX} reps")


# ====================================================================== 4. full bootstrap: re-profile bands each replication
def full_rep(r, c=None):
    c = draw_counts(r) if c is None else c
    rows = np.concatenate([np.tile(np.arange(day_start[d], day_start[d + 1]), c[d]) for d in range(ND) if c[d] > 0])
    y, x, q, bh, bhe, zl, mc = Y_[rows], X_[rows], Q_[rows], BH_[rows], BHE_[rows], ZL_[rows], month_codes[rows]
    l = np.full(len(rows), np.nan); u = l.copy(); ls = l.copy(); us = l.copy()
    rec = {}
    for m in range(NM):
        idx = np.where(mc == m)[0]
        lm, um = profile(y[idx], x[idx], q[idx]); l[idx], u[idx] = lm, um
        rec[f"l_{months[m]}"], rec[f"u_{months[m]}"] = lm, um
        for st in (0, 1):
            jdx = idx[bh[idx] == st]
            if len(jdx) < 300: continue
            lsm, usm = profile(y[jdx], x[jdx], q[jdx]); ls[jdx], us[jdx] = lsm, usm
            rec[f"ls_{months[m]}_{st}"], rec[f"us_{months[m]}_{st}"] = lsm, usm
    ok = np.isfinite(ls)                                  # empirics_state drops cells with < 300 rows
    D = build_design(y, x, q, bh, bhe, zl, l, u, np.where(ok, ls, 0), np.where(ok, us, 0), mc)
    c2 = dict(zip(C2, within_coef(y[:, None], D["c2"], mc)[:, 0]))
    c3 = dict(zip(C3, within_coef(y[:, None], D["c3"], mc)[:, 0]))
    c4 = dict(zip(C4, within_coef(y[:, None], D["c4"], mc)[:, 0]))
    msc = mc * 2 + bh.astype(int)
    st = dict(zip(CS, within_coef(y[ok, None], D["st"][ok], msc[ok])[:, 0]))
    F = within_coef(D["fl_y"][ok], D["fl"][ok], mc[ok])
    fp = dict(zip(CF, F[:, 0] + c2["q"] * F[:, 1])); ff = dict(zip(CF, F[:, 2]))
    out = derived(c2, c3, c4, st, fp, ff)
    out["share_inband"] = D["share_in"]
    wm = np.array([rec[f"u_{mm}"] - rec[f"l_{mm}"] for mm in months])
    out["band_width_median"] = float(np.median(wm))
    wo = np.array([rec.get(f"us_{mm}_1", np.nan) - rec.get(f"ls_{mm}_1", np.nan) for mm in months])
    wc = np.array([rec.get(f"us_{mm}_0", np.nan) - rec.get(f"ls_{mm}_0", np.nan) for mm in months])
    out["state_width_open_mean"], out["state_width_closed_mean"] = np.nanmean(wo), np.nanmean(wc)
    out["state_width_diff"] = np.nanmean(wc - wo)
    out["n_dropped_state_cells"] = int((~ok).sum())
    out.update(rec)
    return out


# replication 'identity': with every day drawn once the full procedure must reproduce the published bands
EXTRA = ["share_inband", "band_width_median", "state_width_open_mean", "state_width_closed_mean", "state_width_diff",
         "n_dropped_state_cells"]
BANDK = [f"{p_}_{m}" for m in months for p_ in ("l", "u")] + [f"{p_}_{m}_{s_}" for m in months for s_ in (0, 1) for p_ in ("ls", "us")]
FULL_COLS = ["rep"] + list(PT.keys()) + EXTRA + BANDK
part_file = lambda k: F_FULL + f".part{k}"


def done_reps():
    s_ = set()
    for f in [F_FULL] + [part_file(k) for k in range(NWORK)]:
        if os.path.exists(f) and os.path.getsize(f) > 0:
            s_ |= set(pd.read_csv(f, usecols=["rep"]).rep.astype(int))
    return s_


if WORKER is not None:                      # worker process: reps r = WORKER (mod NWORK), then exit
    k_ = int(WORKER); have = done_reps(); pf = part_file(k_)
    for r in range(k_, B_FULL, NWORK):
        if r in have: continue
        t = time.time()
        row = {"rep": r, **full_rep(r)}
        hdr = (not os.path.exists(pf)) or os.path.getsize(pf) == 0
        pd.DataFrame([row]).reindex(columns=FULL_COLS).to_csv(pf, mode="a", header=hdr, index=False)
        if (r // NWORK) % 10 == 0:
            log(f"worker {k_}: rep {r} in {time.time() - t:.1f}s")
    sys.exit(0)

ident = full_rep(-1, np.ones(ND, int))
assert set(ident) <= set(FULL_COLS)
for k in ["R1", "R2", "R3", "R4", "R5", "negx_in", "c4_up_bh"]:
    assert abs(ident[k] - PT[k]) < 1e-7 * max(1, abs(PT[k])), (k, ident[k], PT[k])
log("full-bootstrap identity check passed (re-profiling the original sample reproduces all bands and estimates)")

have = done_reps()
todo = [r for r in range(B_FULL) if r not in have]
if todo:
    log(f"full bootstrap: {len(have)} reps on disk, launching {NWORK} workers for {len(todo)} more")
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                              env={**os.environ, "WORKER": str(k), "NWORK": str(NWORK), "NUMBA_NUM_THREADS": "1"})
             for k in range(NWORK)]
    rc = [p_.wait() for p_ in procs]
    assert all(c == 0 for c in rc), f"worker failure {rc}"
parts = [pd.read_csv(f) for f in [F_FULL] + [part_file(k) for k in range(NWORK)] if os.path.exists(f) and os.path.getsize(f) > 0]
FU = pd.concat(parts).drop_duplicates("rep").sort_values("rep").reindex(columns=FULL_COLS)
FU = rederive(FU)                                   # draws on disk may carry an earlier ratio mapping
FU.to_csv(F_FULL, index=False)
for k in range(NWORK):
    if os.path.exists(part_file(k)): os.remove(part_file(k))
FU = FU[FU.rep < B_FULL].set_index("rep")
assert len(FU) == B_FULL, len(FU)
log(f"full bootstrap: {len(FU)} reps")


# ====================================================================== 5. summaries
def q_(v, p):
    v = np.sort(np.asarray(v, float)); v = v[~np.isnan(v)]
    return float(np.quantile(v, p, method="inverted_cdf"))


def boot_ratio(df, k, lev):
    """Percentile CI of the open/closed ratio in the extended reals: closed speed <= 0 -> +inf (docstring item 2);
    closed > 0 and open <= 0 -> ratio <= 0, at the bottom. Equivalent to inverting the quantiles of the closed/open
    inverse only when the open-hours speed is positive in every draw."""
    v = df[k].values.astype(float); inv = df[f"inv_{k}"].values; al = (1 - lev) / 2
    return dict(lo=q_(v, al), hi=q_(v, 1 - al), median=q_(v, 0.5),
                inv_lo=q_(inv, al), inv_hi=q_(inv, 1 - al),
                share_den_le0=float(np.mean(np.isinf(v))), share_ratio_le0=float(np.mean(v <= 0)))


def boot_diag(df, k, est, lev=0.90):
    """Centring of the bootstrap distribution and two alternatives to the percentile interval, on the log scale:
    basic (reverse percentile) and bias-corrected (BC). Reporting only."""
    from scipy.stats import norm
    v = df[k].values.astype(float); al = (1 - lev) / 2
    lr = np.where(np.isinf(v), np.inf, np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), -np.inf))
    le = np.log(est); p_below = float(np.mean(lr < le))
    pl, ph = q_(lr, al), q_(lr, 1 - al)
    z0 = norm.ppf(p_below); zz = norm.ppf(1 - al)
    a_lo, a_hi = norm.cdf(2 * z0 - zz), norm.cdf(2 * z0 + zz)
    return dict(share_draws_below_estimate=p_below, median=float(np.exp(q_(lr, .5))),
                basic_log=[float(np.exp(2 * le - ph)), float(np.exp(2 * le - pl))],
                bc=[float(np.exp(q_(lr, a_lo))), float(np.exp(q_(lr, a_hi)))], bc_levels=[float(a_lo), float(a_hi)])


def boot_scalar(df, col, lev):
    v = df[col].values; al = (1 - lev) / 2
    vv = v[np.isfinite(v)]
    return dict(lo=q_(v, al), hi=q_(v, 1 - al), median=q_(v, 0.5), sd=float(np.std(vv, ddof=1)),
                mean=float(np.mean(vv)), share_le0=float(np.mean(v <= 0)), n_nonfinite=int((~np.isfinite(v)).sum()))


RES = {}
for k in ["R1", "R2", "R3", "R4", "R5"]:
    A90, A95 = ANA[k]["90"], ANA[k]["95"]
    lab = SPEC[k]["label"] if k in SPEC else "long-run impact beta/kappa+, closed over open (emp_main col 4)"
    d = dict(label=lab, estimate=float(A90["est"]), numerator=float(A90["num"]), denominator=float(A90["den"]),
             se_numerator=float(A90["se_num"]), se_denominator=float(A90["se_den"]), t_denominator=float(A90["t_den"]),
             se_delta=float(A90["se"]), se_log_delta=float(A90["se_log"]))
    for lev, A, L in [("90", A90, 0.90), ("95", A95, 0.95)]:
        d[f"delta{lev}"] = [float(A["delta"][0]), float(A["delta"][1])]
        d[f"delta_log{lev}"] = [float(A["delta_log"][0]), float(A["delta_log"][1])]
        d[f"fieller{lev}"] = {kk: (float(vv) if not isinstance(vv, str) else vv) for kk, vv in A["fieller"].items()}
        d[f"boot_fixed{lev}"] = boot_ratio(FX, k, L)
        d[f"boot_full{lev}"] = boot_ratio(FU, k, L)
    inv = ANA[k]["inv"]
    d["inverse_closed_over_open"] = dict(estimate=float(inv["90"]["est"]), se_delta=float(inv["90"]["se"]),
                                         delta90=[float(x_) for x_ in inv["90"]["delta"]],
                                         fieller90={kk: (float(vv) if not isinstance(vv, str) else vv) for kk, vv in inv["90"]["fieller"].items()},
                                         boot_fixed90=[q_(FX[f"inv_{k}"], .05), q_(FX[f"inv_{k}"], .95)],
                                         boot_full90=[q_(FU[f"inv_{k}"], .05), q_(FU[f"inv_{k}"], .95)])
    d["boot_full_diag90"] = boot_diag(FU, k, d["estimate"]); d["boot_fixed_diag90"] = boot_diag(FX, k, d["estimate"])
    d["boot_full_sd_log"] = float(np.std(np.log(FU[k][np.isfinite(FU[k]) & (FU[k] > 0)]), ddof=1))
    d["boot_fixed_sd_log"] = float(np.std(np.log(FX[k][np.isfinite(FX[k]) & (FX[k] > 0)]), ddof=1))
    RES[k] = d
for k in ["LR_closed", "LR_open"]:
    A = ANA[k]["90"]
    col = "lr_closed" if k == "LR_closed" else "lr_open"
    RES[k] = dict(estimate=float(A["est"]), se_delta=float(A["se"]), delta90=[float(x_) for x_ in A["delta"]],
                  fieller90={kk: (float(vv) if not isinstance(vv, str) else vv) for kk, vv in A["fieller"].items()},
                  fieller95={kk: (float(vv) if not isinstance(vv, str) else vv) for kk, vv in ANA[k]["95"]["fieller"].items()},
                  boot_fixed90=[q_(FX[col], .05), q_(FX[col], .95)], boot_full90=[q_(FU[col], .05), q_(FU[col], .95)],
                  share_full_nonpositive_den=float(np.mean(FU.c4_up <= 0)))
RES["logdiff_R3_R4"] = dict(estimate=LD["est"], se_delta=LD["se"], delta90=[LD["est"] - Z90 * LD["se"], LD["est"] + Z90 * LD["se"]],
                            boot_fixed90=[q_(FX.logdiff_R3_R4, .05), q_(FX.logdiff_R3_R4, .95)],
                            boot_full90=[q_(FU.logdiff_R3_R4, .05), q_(FU.logdiff_R3_R4, .95)],
                            boot_fixed95=[q_(FX.logdiff_R3_R4, .025), q_(FX.logdiff_R3_R4, .975)],
                            boot_full95=[q_(FU.logdiff_R3_R4, .025), q_(FU.logdiff_R3_R4, .975)],
                            boot_full_sd=float(np.nanstd(FU.logdiff_R3_R4[np.isfinite(FU.logdiff_R3_R4)], ddof=1)))
OUT["ratios"] = RES

# full-bootstrap distributions of the main coefficients, against the conditional (analytic) s.e.
COEF = {"up_bh (BH increment above band, emp_main col 4)": ("c4_up_bh", r_c4["b"]["up_bh"], r_c4["se"]["up_bh"]),
        "up (closed-hours kappa+, emp_main col 4)": ("c4_up", r_c4["b"]["up"], r_c4["se"]["up"]),
        "negx_in (in-band reversion, emp_main col 3)": ("negx_in", r_c3["b"]["negx_in"], r_c3["se"]["negx_in"]),
        "q (beta, emp_main col 4)": ("c4_q", r_c4["b"]["q"], r_c4["se"]["q"]),
        "up_s_bh (BH increment, emp_state col 1)": ("st_up_s_bh", r_st["b"]["up_s_bh"], r_st["se"]["up_s_bh"]),
        "up_s (closed-hours kappa+, emp_state col 1)": ("st_up_s", r_st["b"]["up_s"], r_st["se"]["up_s"]),
        "d_out_bh (BH increment, emp_flow col 1)": ("fp_bh", r_fp["b"]["d_out_bh"], r_fp["se"]["d_out_bh"]),
        "d_out (closed-hours speed, emp_flow col 1)": ("fp_c", r_fp["b"]["d_out"], r_fp["se"]["d_out"]),
        "kappa+ (emp_main col 2)": ("kp_c2", r_c2["b"]["up"], r_c2["se"]["up"]),
        "d_out_bh (large-trade BH increment, emp_flow col 2, unscaled; table shows x100)": ("ff_bh", r_ff["b"]["d_out_bh"], r_ff["se"]["d_out_bh"]),
        "d_out (large-trade closed-hours response, emp_flow col 2, unscaled; table shows x100)": ("ff_c", r_ff["b"]["d_out"], r_ff["se"]["d_out"])}
CO = {}
for lab, (col, est, se) in COEF.items():
    bf = boot_scalar(FU, col, .90)
    CO[lab] = dict(estimate=float(est), se_analytic=float(se), ci90_analytic=[float(est - Z90 * se), float(est + Z90 * se)],
                   fixed=boot_scalar(FX, col, .90), full=bf, full_bias_mean_minus_est=bf["mean"] - float(est),
                   full_basic90=[2 * float(est) - bf["hi"], 2 * float(est) - bf["lo"]],
                   full95=[q_(FU[col], .025), q_(FU[col], .975)],
                   se_ratio_full_to_analytic=float(boot_scalar(FU, col, .90)["sd"] / se),
                   se_ratio_fixed_to_analytic=float(boot_scalar(FX, col, .90)["sd"] / se))
OUT["coefficients"] = CO
bandcols_u = [c for c in FU.columns if c.startswith("u_")]
PUB_U = a.groupby("month").u.first().to_dict(); PUB_L = a.groupby("month").l.first().to_dict()
OUT["band_variability_full"] = dict(
    median_width_p05_p50_p95=[q_(FU.band_width_median, p) for p in (.05, .5, .95)],
    share_inband_p05_p50_p95=[q_(FU.share_inband, p) for p in (.05, .5, .95)],
    state_width_diff_closed_minus_open=dict(estimate=8.03125, p05=q_(FU.state_width_diff, .05), p50=q_(FU.state_width_diff, .5),
                                            p95=q_(FU.state_width_diff, .95), sd=float(FU.state_width_diff.std())),
    mean_abs_dev_u_bp=float(np.mean([np.mean(np.abs(FU[c] - PUB_U[c[2:]])) for c in bandcols_u])),
    mean_abs_dev_l_bp=float(np.mean([np.mean(np.abs(FU["l_" + c[2:]] - PUB_L[c[2:]])) for c in bandcols_u])),
    median_abs_dev_width_bp=float(np.median([np.median(np.abs((FU[c] - FU["l_" + c[2:]]) - (PUB_U[c[2:]] - PUB_L[c[2:]]))) for c in bandcols_u])),
    share_reps_any_state_cell_dropped=float(np.mean(FU.n_dropped_state_cells > 0)))
BM = {}
for m in months:
    lw, uw = FU[f"l_{m}"], FU[f"u_{m}"]
    BM[m] = dict(l=PUB_L[m], u=PUB_U[m], sd_l=float(lw.std()), sd_u=float(uw.std()),
                 share_within_5bp=float(np.mean((np.abs(lw - PUB_L[m]) <= 5) & (np.abs(uw - PUB_U[m]) <= 5))),
                 share_degenerate=float(np.mean(uw == lw)), width_p05_p95=[q_(uw - lw, .05), q_(uw - lw, .95)])
OUT["band_variability_full"]["by_month"] = BM
OUT["band_variability_full"]["median_sd_edge_bp"] = float(np.median([v["sd_u"] for v in BM.values()] + [v["sd_l"] for v in BM.values()]))
OUT["band_variability_full"]["share_months_edges_within_5bp_in_90pct_of_reps"] = float(np.mean([v["share_within_5bp"] >= 0.9 for v in BM.values()]))
log("summaries done")


# ====================================================================== 6. extension: Turkey vs Mexico (bands fixed)
def mxn_block():
    from build_panel import load_binance, load_fx
    W0, W1 = "2024-11-01", "2026-08-31"; UTC_OFF = {"TRY": 3, "MXN": -6}

    def build(local):        # copied from empirics_mxn.build
        u = load_binance(f"USDT{local}").loc[W0:W1]
        fx = load_fx(f"USD{local}")
        p = pd.DataFrame({"pl": u.vwap, "ofi": 2 * u.tbv - u.v, "vol": u.v}).join(fx.reindex(u.index, method="ffill", limit=2))
        p["b"] = 1e4 * np.log(p.pl / p.fx_mid)
        hr = p.index.hour + p.index.minute / 60
        p.loc[(hr >= 20.5) & (hr < 22.5), "b"] = np.nan
        p.loc[p.index.dayofweek >= 5, "b"] = np.nan
        p.loc[p.b.abs() > 1000, "b"] = np.nan
        g = p.resample("5min")
        d = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum()})
        d = d[d.nb >= 2]
        d["month"] = d.index.tz_localize(None).to_period("M").astype(str)
        d["q"] = d.ofi / d.ofi.groupby(d.month).transform("std")
        consec = d.index.to_series().diff() == pd.Timedelta("5min")
        d["y"] = d.b.diff(); d["x"] = d.b.shift(1)
        d = d[consec & d.y.notna() & d.x.notna() & d.q.notna()].copy()
        lh = (d.index.hour + UTC_OFF[local]) % 24
        d["lh"] = lh; d["bh"] = ((lh >= 9) & (lh < 17)).astype(float)
        d["state"] = np.where(d.bh == 1, "open", "closed"); d["ms"] = d.month + "_" + d.state
        for key, dd in d.groupby("ms"):
            if len(dd) < 300: continue
            l_, u_ = profile(dd.y.values, dd.x.values, dd.q.values)
            d.loc[dd.index, "l"], d.loc[dd.index, "u"] = l_, u_
        d = d[d.l.notna()].copy()
        m = (d.l + d.u) / 2
        d["s"] = np.where(d.x > d.u, 1.0, np.where(d.x < d.l, -1.0, np.sign(d.x - m)))
        d["d_out"] = np.maximum(d.x - d.u, 0) + np.maximum(d.l - d.x, 0)
        d["d_in"] = np.where(d.d_out == 0, np.abs(d.x - m), 0.0)
        d["up"] = -np.maximum(d.x - d.u, 0); d["dn"] = np.maximum(d.l - d.x, 0)
        d["day"] = d.index.date
        return d

    EX = json.load(open(os.path.join(TAB, "emp_mxn.json")))
    A = {k: build(k) for k in ["TRY", "MXN"]}
    alld = np.array(sorted(set(A["TRY"].day) | set(A["MXN"].day)))
    dmon = pd.Series(pd.to_datetime(alld)).dt.to_period("M").astype(str).values
    mlist = sorted(set(dmon)); dim = [np.where(dmon == m)[0] for m in mlist]
    blocks_night, blocks_day = [1, 3, 5, 7], [11, 13, 15]
    res, mom = {}, {}
    for k, d in A.items():
        dcode = np.searchsorted(alld, d.day.values)
        mcode = pd.factorize(d.month, sort=True)[0]; mscode = pd.factorize(d.ms, sort=True)[0]
        b_ = fe_ols(d.y.values, d[["q", "up", "dn"]].values, mscode, dcode, ["q", "up", "dn"])["b"]["q"]
        R = -d.s.values * (d.y.values - b_ * d.q.values)
        Xs = np.column_stack([d.d_out, d.d_out * d.bh, d.d_in, d.d_in * d.bh])
        r = fe_ols(R, Xs, mscode, dcode, CF)
        assert abs(r["b"]["d_out"] - EX[k]["sym"]["d_out"][0]) < 1e-9 and abs(r["b"]["d_out_bh"] - EX[k]["sym"]["d_out_bh"][0]) < 1e-9
        cyc = {}
        for h0 in range(0, 24, 2):
            msk = ((d.lh >= h0) & (d.lh < h0 + 2)).values
            if (d.d_out.values[msk] > 0).sum() < 200: continue
            cyc[h0 + 1] = fe_ols(R[msk], np.column_stack([d.d_out.values[msk], d.d_in.values[msk]]), mcode[msk], dcode[msk], ["d_out", "d_in"])
        night = np.mean([cyc[h]["b"]["d_out"] for h in blocks_night if h in cyc]); day = np.mean([cyc[h]["b"]["d_out"] for h in blocks_day if h in cyc])
        assert abs(night / day - EX[k]["night_over_day"]) < 1e-9
        # joint delta for night/day: stack per-day influence of the block regressions
        hs = [h for h in blocks_night + blocks_day if h in cyc]
        Gb = np.zeros((len(alld), len(hs)))
        for i, h in enumerate(hs):
            msk = ((d.lh >= h - 1) & (d.lh < h + 1)).values
            dc = pd.factorize(dcode[msk])[1]
            Gb[np.asarray(dc), i] = cyc[h]["infl"][:, 0]
        Vb = Gb.T @ Gb; bb = np.array([cyc[h]["b"]["d_out"] for h in hs])
        wn = np.array([1 / len([h for h in blocks_night if h in cyc]) if h in blocks_night else 0 for h in hs])
        wd = np.array([1 / len([h for h in blocks_day if h in cyc]) if h in blocks_day else 0 for h in hs])
        nd_ = lin_ratio_ci(bb, Vb, wn, wd, Z90)
        rr = lin_ratio_ci(r["bv"], r["V"], e(CF, "d_out", "d_out_bh"), e(CF, "d_out"), Z90)
        rr95 = lin_ratio_ci(r["bv"], r["V"], e(CF, "d_out", "d_out_bh"), e(CF, "d_out"), Z95)
        res[k] = dict(ratio=rr, ratio95=rr95, night_day=nd_, n=len(d))
        # moments for the bootstrap (beta re-estimated: R = -s*y + beta*s*q)
        mom[k] = dict(beta=_moments_days(d[["q", "up", "dn"]].values, d.y.values[:, None], mscode, dcode, len(alld)),
                      sym=_moments_days(Xs, np.column_stack([-d.s * d.y, d.s * d.q]), mscode, dcode, len(alld)), cyc={})
        for h in cyc:
            msk = ((d.lh >= h - 1) & (d.lh < h + 1)).values
            mom[k]["cyc"][h] = _moments_days(np.column_stack([d.d_out.values[msk], d.d_in.values[msk]]),
                                             np.column_stack([(-d.s * d.y).values[msk], (d.s * d.q).values[msk]]),
                                             mcode[msk], dcode[msk], len(alld))
    # bootstrap: the same calendar days drawn for both markets (keeps cross-market dependence)
    BS = []
    for r_ in range(B_FIX):
        rng = np.random.default_rng([SEED, 7, r_]); w = np.zeros(len(alld))
        for dm_ in dim:
            w += np.bincount(dm_[rng.integers(0, len(dm_), len(dm_))], minlength=len(alld))
        row = {}
        for k in A:
            be = mom[k]["beta"].coef(w)[0, 0]
            F = mom[k]["sym"].coef(w); c0, c1 = F[0, 0] + be * F[0, 1], F[1, 0] + be * F[1, 1]
            row[f"{k}_inv"] = c0 / (c0 + c1); row[f"{k}_ratio"] = (c0 + c1) / c0 if c0 > 0 else np.inf
            kk = {h: (lambda G: G[0, 0] + be * G[0, 1])(m_.coef(w)) for h, m_ in mom[k]["cyc"].items()}
            row[f"{k}_night_day"] = np.mean([kk[h] for h in blocks_night if h in kk]) / np.mean([kk[h] for h in blocks_day if h in kk])
        row["diff_log"] = (np.log(row["TRY_ratio"]) if np.isfinite(row["TRY_ratio"]) else np.inf) - np.log(row["MXN_ratio"])
        BS.append(row)
    BS = pd.DataFrame(BS)
    O = {}
    for k in A:
        rr, rr95, nd_ = res[k]["ratio"], res[k]["ratio95"], res[k]["night_day"]
        O[k] = dict(ratio=float(rr["est"]), closed=float(rr["den"]), se_closed=float(rr["se_den"]), t_closed=float(rr["t_den"]),
                    delta90=[float(v) for v in rr["delta"]], delta_log90=[float(v) for v in rr["delta_log"]],
                    fieller90={kk: (float(vv) if not isinstance(vv, str) else vv) for kk, vv in rr["fieller"].items()},
                    fieller95={kk: (float(vv) if not isinstance(vv, str) else vv) for kk, vv in rr95["fieller"].items()},
                    boot_fixed90=boot_ratio(BS.rename(columns={f"{k}_inv": "inv_X", f"{k}_ratio": "X"}), "X", .90),
                    night_over_day=float(nd_["est"]), night_over_day_se=float(nd_["se"]),
                    night_over_day_delta90=[float(v) for v in nd_["delta"]],
                    night_over_day_fieller90={kk: (float(vv) if not isinstance(vv, str) else vv) for kk, vv in nd_["fieller"].items()},
                    night_over_day_boot90=[q_(BS[f"{k}_night_day"], .05), q_(BS[f"{k}_night_day"], .95)], n=res[k]["n"])
    dl = np.log(res["TRY"]["ratio"]["est"]) - np.log(res["MXN"]["ratio"]["est"])
    sdl = float(np.hypot(res["TRY"]["ratio"]["se_log"], res["MXN"]["ratio"]["se_log"]))
    O["diff_log_ratio"] = dict(estimate=float(dl), se_delta_independent=sdl, delta90=[dl - Z90 * sdl, dl + Z90 * sdl],
                               boot_se=float(np.std(BS.diff_log[np.isfinite(BS.diff_log)], ddof=1)),
                               boot90=[q_(BS.diff_log, .05), q_(BS.diff_log, .95)], boot95=[q_(BS.diff_log, .025), q_(BS.diff_log, .975)],
                               boot_share_le0=float(np.mean(BS.diff_log <= 0)),
                               note="bootstrap draws the same calendar days for both markets; delta treats them as independent (as in emp_mxn)")
    BS.to_csv(os.path.join(OTAB, f"{NAME}_draws_mxn.csv"), index_label="rep")
    return O


def _moments_days(X, Y, grp, dcode, ndays):
    m = Moments.__new__(Moments)
    cell, cu = pd.factorize(pd.Series(dcode) * 1000 + grp)
    m.k, m.ny = X.shape[1], Y.shape[1]
    m.M = cell_moments(np.column_stack([np.ones(len(X)), X, Y]), cell, len(cu))
    cu = np.asarray(cu); m.cday = cu // 1000
    g = pd.factorize(cu % 1000)[0]
    m.A = np.zeros((g.max() + 1, len(cu))); m.A[g, np.arange(len(cu))] = 1.0
    return m


if DO_MXN:
    OUT["mxn_extension"] = mxn_block()
    log("Turkey/Mexico extension done")


# ====================================================================== 7. recommendation (docstring, item 5)
def bounded90(k):
    d = RES[k]
    return d["fieller90"]["kind"] == "bounded" and np.isfinite(d["boot_fixed90"]["hi"]) and np.isfinite(d["boot_full90"]["hi"])


cand = [k for k in ["R1", "R2", "R3"] if bounded90(k)]
width = {k: np.log(RES[k]["boot_full90"]["hi"]) - np.log(RES[k]["boot_full90"]["lo"]) for k in cand}
prim = min(cand, key=lambda k: width[k]) if cand else None
OUT["recommendation"] = dict(rule="docstring item 5", candidates_bounded=cand, log_width_full90=width, primary=prim,
                             primary_label=RES[prim]["label"] if prim else "business-hours increment (no ratio bounded)",
                             primary_ratio=RES[prim]["estimate"] if prim else None,
                             primary_full90=[RES[prim]["boot_full90"]["lo"], RES[prim]["boot_full90"]["hi"]] if prim else None)
OUT["runtime_sec"] = time.time() - T0


def clean(o):
    """JSON-safe: numpy scalars to float, +inf to the string 'inf', nan to None."""
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return o if np.isfinite(o) else (None if np.isnan(o) else ("inf" if o > 0 else "-inf"))
    return o


json.dump(clean(OUT), open(os.path.join(OTAB, f"{NAME}.json"), "w"), indent=1)
log("json written")


# ====================================================================== 8. LaTeX table
def c_(lo, hi, nd=1):
    isinf = lambda v: v is None or v in ("inf", "-inf") or (isinstance(v, float) and np.isinf(v))
    f = lambda v: ("-\\infty" if (v == "-inf" or (isinstance(v, float) and v < 0)) else "\\infty") if isinf(v) else f"{v:.{nd}f}"
    return f"${'(' if isinf(lo) else '['}{f(lo)}, {f(hi)}{')' if isinf(hi) else ']'}$"


def fiel_tex(F, nd=1):
    if F["kind"] == "bounded": return c_(F["lo"], F["hi"], nd)
    if F["kind"] == "two_rays" and F["ray_below"] < 0:      # show the full Fieller set (union of two rays)
        return f"$(-\\infty, {F['ray_below']:.{nd}f}] \\cup [{F['ray_above']:.{nd}f}, \\infty)$"
    return "unbounded"


ROWS = [("R1", "$\\kappa^+$, monthly bands (Table~\\ref{tab:emp_main}, col.~4)"),
        ("R2", "$\\kappa^+$, state bands (Table~\\ref{tab:emp_state}, col.~1)"),
        ("R3", "Speed outside band, both sides (Table~\\ref{tab:emp_flow}, col.~1)"),
        ("R4", "Large-trade response (Table~\\ref{tab:emp_flow}, col.~2)"),
        ("R5", "Long-run impact $\\beta/\\kappa^+$, closed/open (col.~4)")]
with open(os.path.join(OTAB, f"{NAME}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccccccc}\n\\toprule\n & & $t$ of & \\multicolumn{5}{c}{90\\% confidence interval} \\\\\n"
             "\\cmidrule(lr){4-8}\n"
             " & Estimate & closed-hours & Delta & Delta & Fieller & Bootstrap, & Bootstrap, \\\\\n"
             " & & speed & & (log) & & bands fixed & bands re-estimated \\\\\n\\midrule\n"
             "\\multicolumn{8}{l}{\\emph{A. Open/closed ratios}} \\\\\n")
    for k, lab in ROWS:
        d = RES[k]
        fh.write(f"{lab} & {d['estimate']:.1f} & {d['t_denominator']:.1f} & {c_(*d['delta90'])} & {c_(*d['delta_log90'])} & "
                 f"{fiel_tex(d['fieller90'])} & {c_(d['boot_fixed90']['lo'], d['boot_fixed90']['hi'])} & "
                 f"{c_(d['boot_full90']['lo'], d['boot_full90']['hi'])} \\\\\n")
    ld = RES["logdiff_R3_R4"]
    fh.write(f"$\\log$ ratio, prices $-$ large trades (cols.~1--2) & ${ld['estimate']:.2f}$ & & {c_(*ld['delta90'], nd=2)} & & & "
             f"{c_(*ld['boot_fixed90'], nd=2)} & {c_(*ld['boot_full90'], nd=2)} \\\\\n")
    if "mxn_extension" in OUT:
        M = OUT["mxn_extension"]
        fh.write("\\midrule\n\\multicolumn{8}{l}{\\emph{B. Turkey and Mexico, November 2024 -- August 2026 (Table~\\ref{tab:emp_mxn}); bands fixed}} \\\\\n")
        for kk, lab in [("TRY", "Turkey: speed outside band, open/closed"), ("MXN", "Mexico: speed outside band, open/closed")]:
            m = M[kk]
            fh.write(f"{lab} & {m['ratio']:.1f} & {m['t_closed']:.1f} & {c_(*m['delta90'])} & {c_(*m['delta_log90'])} & "
                     f"{fiel_tex(m['fieller90'])} & {c_(m['boot_fixed90']['lo'], m['boot_fixed90']['hi'])} & -- \\\\\n")
        dd = M["diff_log_ratio"]
        fh.write(f"Difference in $\\log$ ratios, Turkey $-$ Mexico & ${dd['estimate']:.2f}$ & & {c_(*dd['delta90'], nd=2)} & & & {c_(*dd['boot90'], nd=2)} & -- \\\\\n")
        for kk, lab in [("TRY", "Turkey: night speed / day speed"), ("MXN", "Mexico: night speed / day speed")]:
            m = M[kk]
            fh.write(f"{lab} & ${m['night_over_day']:.2f}$ & & {c_(*m['night_over_day_delta90'], nd=2)} & & {fiel_tex(m['night_over_day_fieller90'], nd=2)} & "
                     f"{c_(*m['night_over_day_boot90'], nd=2)} & -- \\\\\n")
    fh.write("\\midrule\n\\multicolumn{8}{l}{\\emph{C. Coefficients (Table~\\ref{tab:emp_main}); the Delta column is the conditional day-clustered interval}} \\\\\n")
    for lab, key in [("Business-hours increment in $\\kappa^+$ (col.~4)", "up_bh (BH increment above band, emp_main col 4)"),
                     ("Closed-hours $\\kappa^+$ (col.~4)", "up (closed-hours kappa+, emp_main col 4)"),
                     ("In-band reversion (col.~3)", "negx_in (in-band reversion, emp_main col 3)")]:
        c = CO[key]
        fh.write(f"{lab} & ${c['estimate']:.4f}$ & & {c_(*c['ci90_analytic'], nd=4)} & & & {c_(c['fixed']['lo'], c['fixed']['hi'], nd=4)} & "
                 f"{c_(c['full']['lo'], c['full']['hi'], nd=4)} \\\\\n")
    fh.write(f"\\midrule\nBootstrap replications & & & & & & {len(FX)} & {len(FU)} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
log("tex written")


# ====================================================================== 9. figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.0))
A_ = ax[0, 0]; XMAX = 40.0
meth = [("Delta", "delta90", "D", C3), ("Fieller", "fieller90", "s", "#5b7a3a"),
        ("Bootstrap, bands fixed", "boot_fixed90", "^", C2), ("Bootstrap, bands re-estimated", "boot_full90", "o", C1)]
for i, (k, lab) in enumerate(ROWS):
    d = RES[k]
    for j, (mn, key, mk, col) in enumerate(meth):
        yv = i + (j - 1.5) * 0.17
        if key == "fieller90":
            F = d[key]
            lo, hi = (F["lo"], F["hi"]) if F["kind"] == "bounded" else ((F["ray_above"], np.inf) if F["kind"] == "two_rays" and F["ray_below"] < 0 else (0.5, np.inf))
        elif key == "delta90":
            lo, hi = d[key]
        else:
            lo, hi = d[key]["lo"], d[key]["hi"]
        lo_p = max(lo, 0.5); hi_p = min(hi, XMAX)
        A_.plot([lo_p, hi_p], [yv, yv], color=col, lw=1.6, solid_capstyle="round")
        if lo < 0.5:                                    # mark lower bounds below the axis (incl. negative)
            A_.annotate("", xy=(0.5 / 1.12, yv), xytext=(0.5 * 1.05, yv), arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        if not np.isfinite(hi) or hi > XMAX:
            A_.annotate("", xy=(XMAX * 1.12, yv), xytext=(XMAX * 0.95, yv), arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        A_.plot([d["estimate"]], [yv], marker=mk, ms=4.5, color=col, mec="white", mew=0.6,
                label=mn if i == 0 else None, ls="none")
A_.set_xscale("log"); A_.set_xlim(0.5 / 1.15, XMAX * 1.15)
A_.set_xticks([0.5, 1, 2, 4, 8, 16, 32]); A_.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32"])
A_.axvline(1, color="k", lw=0.5, ls=":")
A_.set_yticks(range(len(ROWS))); A_.set_yticklabels(["R1 $\\kappa^+$ monthly bands", "R2 $\\kappa^+$ state bands",
                                                     "R3 speed, both sides", "R4 large-trade flow", "R5 long-run impact"], fontsize=8)
A_.invert_yaxis(); A_.set_xlabel("open/closed ratio (log scale); arrow = beyond axis or unbounded", fontsize=8)
A_.set_title("(a) 90% intervals for the headline ratios", fontsize=9.5)
A_.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, handletextpad=0.3, columnspacing=1.0)


def hist2(axx, col_full, col_fixed, est, title, xlabel, log_=False, clip=None, legend_=False):
    vf, vx = FU[col_full].values.astype(float), FX[col_fixed].values.astype(float)
    if log_:
        vf, vx = np.log(np.where(np.isfinite(vf) & (vf > 0), vf, np.nan)), np.log(np.where(np.isfinite(vx) & (vx > 0), vx, np.nan))
        est = np.log(est)
    if clip is not None:
        vf, vx = np.clip(vf, *clip), np.clip(vx, *clip)
    allv = np.r_[vf[np.isfinite(vf)], vx[np.isfinite(vx)]]
    bins = np.linspace(allv.min(), allv.max(), 40)
    axx.hist(vx[np.isfinite(vx)], bins=bins, histtype="step", color=C2, lw=1.3, label="bands fixed")
    axx.hist(vf[np.isfinite(vf)], bins=bins, color=C1, alpha=0.55, label="bands re-estimated", edgecolor="white", lw=0.3)
    axx.axvline(est, color="k", lw=0.8, ls="--", label="estimate")
    axx.set_title(title, fontsize=9.5); axx.set_xlabel(xlabel); axx.set_yticks([])
    if legend_: axx.legend(frameon=False, fontsize=7, loc="upper left")
    if not log_: axx.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))


hist2(ax[0, 1], "c4_up_bh", "c4_up_bh", r_c4["b"]["up_bh"], "(b) Business-hours increment in $\\kappa^+$", "per 5 min", legend_=True)
hist2(ax[1, 0], "R1", "R1", PT["R1"], "(c) Open/closed ratio of $\\kappa^+$ (R1)", "ratio (log scale)", log_=True)
_xl = ax[1, 0].get_xlim(); _tk = [v for v in [0.5, 1, 2, 4, 8, 16, 32, 64] if _xl[0] <= np.log(v) <= _xl[1]]
ax[1, 0].set_xticks(np.log(_tk)); ax[1, 0].set_xticklabels([f"{v:g}" for v in _tk]); ax[1, 0].set_xlim(_xl)
hist2(ax[1, 1], "negx_in", "negx_in", r_c3["b"]["negx_in"], "(d) In-band reversion coefficient", "per 5 min")
fig.tight_layout(h_pad=2.2); fig.savefig(os.path.join(OFIG, f"{NAME}.pdf"), bbox_inches="tight"); plt.close(fig)
log("figure written")

# ---------------------------------------------------------------------- console summary
for k in ["R1", "R2", "R3", "R4", "R5"]:
    d = RES[k]
    print(k, f"{d['estimate']:.2f}", "t_den", f"{d['t_denominator']:.2f}", "delta", np.round(d["delta90"], 2), "dlog", np.round(d["delta_log90"], 2),
          "fieller", {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in d["fieller90"].items()},
          "fixed", (round(d["boot_fixed90"]["lo"], 2), round(d["boot_fixed90"]["hi"], 2), d["boot_fixed90"]["share_den_le0"]),
          "full", (round(d["boot_full90"]["lo"], 2), round(d["boot_full90"]["hi"], 2), d["boot_full90"]["share_den_le0"]))
for lab, c in CO.items():
    print(lab, c["estimate"], c["se_analytic"], "fixed sd", c["fixed"]["sd"], "full sd", c["full"]["sd"], "full90", c["full"]["lo"], c["full"]["hi"])
print(json.dumps(OUT["recommendation"], default=float))
