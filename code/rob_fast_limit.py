"""Natural experiment: the 4 April 2024 increase in the FAST per-transfer limit.

The aim is to separate the banking RAIL from daytime activity. Institutional facts (TCMB press
releases 2024-16 and 2026-36, see tables/rob_tcmb_institutions.json): EFT, the large-value lira rail,
operates 08:30-17:30 on business days (13:00 on half days, shut on weekends and public holidays).
Outside EFT hours lira moves between banks only through FAST, whose per-transfer limit rose from
50,000 TRY to 100,000 TRY on 4 April 2024 (and to 300,000 TRY on 26 August 2026, four business days
before the sample ends; not used). During EFT hours the FAST limit should not bind.

HYPOTHESIS. Raising the FAST limit raises effective arbitrage capacity when EFT is closed but not
when it is open: the closed-hours speed of arbitrage rises relative to the open-hours speed.

DESIGN
------
Notation. C_t = 1 if interval t is coded "closed"; P_t = 1 if the Istanbul time of the interval
label is on/after 2024-04-04 00:00 (post). Bands are the PUBLISHED ones (a5.parquet l,u: monthly;
state_bands.parquet l_s,u_s: month x clock banking state), not re-profiled. s_t, d_out, d_in as in
empirics_mech.py (side of the band; distance outside; distance from the midpoint inside).
Codings of C:  clock  C = 1 - bh (outside 09:00-17:00 Istanbul)             [primary]
               eft    C = 1 - EFT open, EFT open = 08:30-17:30 on full business days, 08:30-13:00
                      on half days (rob_tcmb_institutions calendar), closed on HOL weekday holidays
               bheff  C = 1 - bh*(1-hol) (the emp_flow convention)
Specifications (each regressor v enters as v, v*C, v*P, v*C*P; open/pre is the base cell):
  P  pooled price side, as emp_flow col (1): R = -s(y - beta*q), beta = published full-sample
     column (2) estimate (emp_numbers.json); v in {d_out, d_in}. Key coefficient: d_out*C*P.
  K  emp_main col (4) analogue: y on v in {q, up, dn}. Key coefficient: up*C*P (kappa+);
     dn*C*P (kappa-) reported.
  Bands/FE: M = monthly bands, month FE; S = state bands, month x state FE (state = the coding's C).
  The month containing the break date is split at the break in the FE (so FE = month x post).
Reported per spec: closed speed pre/post, open speed pre/post (delta-method s.e. of sums), the DiD in
levels (= coefficient on v*C*P), the DiD in logs, log(cl_post/cl_pre) - log(op_post/op_pre)
(delta method; only when all four speeds > 0), the d_in*C*P placebo for P, cell counts.
All s.e. are day-clustered (UTC day, as published) with the finite-sample correction and are
CONDITIONAL on the bands (no band-re-estimating bootstrap; the paper's bootstrap roughly doubles the
s.e. of the full-sample increase). Secondary: week-clustered s.e. for the primary spec.
PRIMARY TEST: spec P, state bands, month x state FE, clock coding, window 2024-01-02 to 2024-07-03
(= break - 93 days to break + 90 days), H1: DiD > 0, one-sided 5%. Co-headline: K-M-clock kappa+.
Power: MDE at 80% power for the one-sided 5% test = 2.49 x s.e.; benchmark effect = closed-hours
capacity proportional to the cap (doubling): level DiD = closed pre speed, log DiD = log 2.
(1) Main DiD: P and K, bands M and S, codings clock/eft/bheff, primary window.
(2) Robustness (P-S, P-M, K-M, K-S, clock): windows break -/+ 2 months (2024-02-03..2024-06-02) and
    2024-01-02..2024-08-31; excluding 12-16 Apr 2024 (Istanbul dates; the 590 bp episode); excluding
    9-16 Apr 2024 (half day, Bayram holiday and reopening); demand-only
    order flow (z_dem = small+medium OFI standardized within month; beta_D from the column (2)
    regression with z_dem, monthly bands, month FE, full sample, as rob_demand_ofi.py "fixed");
    free beta (P with -s*y as dependent variable and s*q fully interacted as a regressor);
    week-clustered s.e.; FXCM benchmark (panel rebuilt as in rob_fxcm_benchmark.py for 2024-01..08,
    monthly and state bands re-profiled on FXCM data, beta = FXCM column (2) estimate).
(3) Placebo break dates, same window rule [D-93d, D+90d], FE month split at D, spec P-S-clock and
    K-M-clock. Listed set: 2024-06-04, 2024-09-04, 2025-01-06, 2025-04-04, 2025-09-04, 2026-01-05,
    2026-04-06 (2024-06-04's window contains the true break: reported but excluded from p-values).
    Extended set: the first business day of every month Aug 2024 - May 2026 (all windows lie inside
    the 100,000 TRY regime). Randomization-style p = (1 + #{placebo DiD >= actual}) / (1 + N),
    one-sided, on the level DiD (primary), the t-statistic and the log DiD (secondary).
(4) Dose-response, 2024-01-02..2026-08-25: cap_usd = cap_TRY(date) / monthly mean of exp(pg);
    lc = log(cap_usd) demeaned. P-S: R on d_out x month dummies (month-specific open slopes),
    d_out*C, d_out*C*lc, same for d_in (d_in*C*lc is a placebo), FE month x regime x state. K-M: q, up,
    dn each x month dummies, x C, x C*lc; FE month x regime. Key: d_out*C*lc (up*C*lc). Variants: add
    a linear trend in the closed gap (d_out*C*t, t in years); EFT coding; excluding 19 Mar-30 Apr 2025;
    including the 300,000 TRY days.
(5) Event-time figure by month Jan-Aug 2024 (April 1-3 pooled with March) of closed and open
    speeds, P-S-clock and K-M-clock, figures/rob_fast_limit.pdf.
Classification stored in the JSON (key "verdict"): "supported" if the primary DiD is positive and
significant at 5% one-sided AND its extended-placebo p-value is <= 0.10; "not supported" if the primary
DiD is <= 0 or insignificant; anything in between is "suggestive". Power limits (three months of
pre-period, one break) are reported with the MDE. The JSON also tallies the sign of the closed-hours
change across the main specifications.
Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, data_raw/ofi_size/*.parquet, data_raw/panel_TRY.parquet,
data_raw/fxcm (download_fxcm.py), tables/emp_numbers.json, tables/emp_mech.json, tables/rob_demand_ofi.json,
tables/rob_fxcm_benchmark.json.
Outputs: tables/rob_fast_limit.json, tables/rob_fast_limit.tex, figures/rob_fast_limit.pdf.
Run: python -P code/rob_fast_limit.py
"""
import os, sys, glob, json, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import _ssr_grid
warnings.filterwarnings("ignore")

RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
NAME = "fast_limit"
T0 = time.time()
BREAK = pd.Timestamp("2024-04-04")
PRE_D, POST_D = 93, 90
W_PRIMARY = (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-07-03"))
OUT = dict(name=NAME, break_date=str(BREAK.date()), window_rule=f"[D-{PRE_D}d, D+{POST_D}d] Istanbul dates")

# ------------------------------------------------------------------ data
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
a = a.join(sb, how="inner")
a["day"] = a.index.date
a["z_dem"] = (a.ofi_small + a.ofi_mid) / (a.ofi_small + a.ofi_mid).groupby(a.month).transform("std")

HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
HALF = ["2024-04-09", "2024-10-28", "2025-06-05", "2025-10-28", "2026-03-19", "2026-05-26"]


def add_calendar(d):
    ist = d.index.tz_convert(None) + pd.Timedelta(hours=3)          # Istanbul, UTC+3 all year
    d["ist_date"] = ist.normalize()
    mod = ist.hour * 60 + ist.minute
    hol = d.ist_date.isin(pd.to_datetime(HOL)).values
    half = d.ist_date.isin(pd.to_datetime(HALF)).values
    eft = np.where(hol, False, np.where(half, (mod >= 510) & (mod < 780), (mod >= 510) & (mod < 1050)))
    d["hol"] = hol.astype(float)
    d["eft_open"] = eft.astype(float)
    d["C_clock"] = 1.0 - d.bh
    d["C_eft"] = 1.0 - d.eft_open
    d["C_bheff"] = 1.0 - d.bh * (1 - d.hol)
    d["week"] = pd.to_datetime(d.day).dt.to_period("W").astype(str).values
    return d


a = add_calendar(a)


def add_band_vars(d, l, u, tag, beta, flow="q"):
    m = (d[l] + d[u]) / 2
    s = np.where(d.x > d[u], 1.0, np.where(d.x < d[l], -1.0, np.sign(d.x - m)))
    out = (d.x > d[u]) | (d.x < d[l])
    d["s_" + tag] = s
    d["dout_" + tag] = np.maximum(d.x - d[u], 0) + np.maximum(d[l] - d.x, 0)
    d["din_" + tag] = np.where(out, 0.0, np.abs(d.x - m))
    d["up_" + tag] = -np.maximum(d.x - d[u], 0)
    d["dn_" + tag] = np.maximum(d[l] - d.x, 0)
    d["R_" + tag] = -s * (d.y - beta * d[flow])
    d["R0_" + tag] = -s * d.y                    # free-beta variant: dependent variable without the flow adjustment
    d["sq_" + tag] = s * d[flow]
    return d


BETA = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
a = add_band_vars(a, "l", "u", "M", BETA)
a = add_band_vars(a, "l_s", "u_s", "S", BETA)


# ------------------------------------------------------------------ estimator (conventions of empirics_mech.ols)
def ols(d, ycol, cols, fe, cluster="day", drop_zero=False):
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    gm = d.groupby(fe, sort=False)
    Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    keep = np.abs(X).sum(0) > 1e-12
    if not keep.all():
        if not drop_zero:
            raise ValueError(f"all-zero regressor after FE: {[c for c, k in zip(cols, keep) if not k]}")
        X = X[:, keep]; cols = [c for c, k in zip(cols, keep) if k]   # month-slope columns of near-empty months
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d[cluster])[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=b, V=V, cols=list(cols), n=int(n), n_clusters=int(ng))


def coef(r, c):
    i = r["cols"].index(c)
    return float(r["b"][i]), float(np.sqrt(r["V"][i, i]))


def lin(r, w):
    """Linear combination sum w[c]*b[c] and its s.e."""
    g = np.zeros(len(r["cols"]))
    for c, v in w.items():
        g[r["cols"].index(c)] = v
    return float(g @ r["b"]), float(np.sqrt(g @ r["V"] @ g))


def did_summary(r, v):
    """Speeds by cell and DiD from v, v_C, v_P, v_CP."""
    ix = {c: r["cols"].index(c) for c in (v, v + "_C", v + "_P", v + "_CP")}
    a0, b0, c0, g0 = (r["b"][ix[c]] for c in (v, v + "_C", v + "_P", v + "_CP"))
    o = dict(open_pre=lin(r, {v: 1}), closed_pre=lin(r, {v: 1, v + "_C": 1}),
             open_post=lin(r, {v: 1, v + "_P": 1}), closed_post=lin(r, {v: 1, v + "_C": 1, v + "_P": 1, v + "_CP": 1}),
             closed_change=lin(r, {v + "_P": 1, v + "_CP": 1}), open_change=lin(r, {v + "_P": 1}),
             did=lin(r, {v + "_CP": 1}))
    o["did_t"] = o["did"][0] / o["did"][1]
    o["did_p_one_sided"] = float(0.5 * __import__("math").erfc(o["did_t"] / np.sqrt(2)))
    sp = [a0 + b0, a0 + b0 + c0 + g0, a0, a0 + c0]       # closed pre, closed post, open pre, open post
    if min(sp) > 0:
        cp, cq, op, oq = sp
        g = np.zeros(len(r["cols"]))
        g[ix[v]] = 1 / cq - 1 / cp - 1 / oq + 1 / op
        g[ix[v + "_C"]] = 1 / cq - 1 / cp
        g[ix[v + "_P"]] = 1 / cq - 1 / oq
        g[ix[v + "_CP"]] = 1 / cq
        L = np.log(cq) - np.log(cp) - np.log(oq) + np.log(op)
        o["did_log"] = [float(L), float(np.sqrt(g @ r["V"] @ g))]
        o["ratio_open_closed_pre"] = float(op / cp); o["ratio_open_closed_post"] = float(oq / cq)
    else:
        o["did_log"] = None
        o["ratio_open_closed_pre"] = float(sp[2] / sp[0]); o["ratio_open_closed_post"] = float(sp[3] / sp[1])
    o["mde80_level"] = 2.49 * o["did"][1]
    o["benchmark_level_doubling"] = float(sp[0])             # closed pre speed: what a doubling would add
    o["mde80_log"] = 2.49 * o["did_log"][1] if o["did_log"] else None
    o["n"] = r["n"]; o["n_days"] = r["n_clusters"]
    return o


def interact(d, vars_, C, P):
    cols = []
    for v in vars_:
        d[v + "_C"] = d[v] * C; d[v + "_P"] = d[v] * P; d[v + "_CP"] = d[v] * C * P
        cols += [v, v + "_C", v + "_P", v + "_CP"]
    return cols


def window(d, D, pre=PRE_D, post=POST_D, start=None, end=None):
    lo = pd.Timestamp(start) if start is not None else D - pd.Timedelta(days=pre)
    hi = pd.Timestamp(end) if end is not None else D + pd.Timedelta(days=post)
    return d[(d.ist_date >= lo) & (d.ist_date <= hi)].copy()


def run_did(d, D, spec, bands, coding, flow="q", Rcol=None, cluster="day", free_beta=False):
    """spec P or K on window data d with break D."""
    d = d.copy()
    C = d["C_" + coding].values; P = (d.ist_date >= D).values.astype(float)
    d["per"] = d.month + np.where(P == 1, "_post", "_pre")
    d["fe"] = d.per if bands == "M" else d.per + "_" + C.astype(int).astype(str)
    if spec == "P":
        d["v_dout"], d["v_din"] = d["dout_" + bands], d["din_" + bands]
        vars_ = ["v_dout", "v_din"]
        if free_beta:
            d["v_sq"] = d["sq_" + bands]; vars_ = vars_ + ["v_sq"]; ycol = "R0_" + bands
        else:
            ycol = Rcol or "R_" + bands
        cols = interact(d, vars_, C, P)
        r = ols(d, ycol, cols, "fe", cluster)
        o = dict(d_out=did_summary(r, "v_dout"), d_in=did_summary(r, "v_din"))
        if free_beta:
            o["beta"] = did_summary(r, "v_sq")
    else:
        d["v_q"], d["v_up"], d["v_dn"] = d[flow], d["up_" + bands], d["dn_" + bands]
        cols = interact(d, ["v_q", "v_up", "v_dn"], C, P)
        r = ols(d, "y", cols, "fe", cluster)
        o = dict(kp=did_summary(r, "v_up"), km=did_summary(r, "v_dn"), beta=did_summary(r, "v_q"))
    out = (d["dout_" + bands] > 0)
    o["cells"] = {f"{'closed' if c else 'open'}_{'post' if p else 'pre'}":
                  dict(n=int(((C == c) & (P == p)).sum()), n_out=int(((C == c) & (P == p) & out.values).sum()),
                       n_above=int(((C == c) & (P == p) & (d.x > d["u" if bands == "M" else "u_s"]).values).sum()))
                  for c in (1, 0) for p in (1, 0)}
    o["window"] = [str(d.ist_date.min().date()), str(d.ist_date.max().date())]
    return o


def key(o, spec):
    return o["d_out"] if spec == "P" else o["kp"]


# ------------------------------------------------------------------ replication of published numbers (full sample)
def replicate():
    d = a.copy()
    d["bh_eff"] = d.bh * (1 - d.hol)
    for c in ["dout_S", "din_S"]:
        d[c + "_bh"] = d[c] * d.bh_eff
    r = ols(d, "R_S", ["dout_S", "dout_S_bh", "din_S", "din_S_bh"], "month")
    for c in ["q", "up_M", "dn_M"]:
        d[c + "_bh"] = d[c] * d.bh
    r2 = ols(d, "y", ["q", "up_M", "dn_M", "q_bh", "up_M_bh", "dn_M_bh"], "month")
    mech = json.load(open(os.path.join(TAB, "emp_mech.json")))["price"]
    pub = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["bh"]["coef"]
    return dict(emp_flow_col1_d_out=[coef(r, "dout_S")[0], mech["d_out"][0]],
                emp_flow_col1_d_out_bh=[coef(r, "dout_S_bh")[0], mech["d_out_bh"][0]],
                emp_main_col4_up=[coef(r2, "up_M")[0], pub["up"]], emp_main_col4_up_bh=[coef(r2, "up_M_bh")[0], pub["up_bh"]])


OUT["replication_check"] = replicate()
print("replication", OUT["replication_check"], flush=True)

# ------------------------------------------------------------------ (1) main DiD
W = window(a, BREAK, start=W_PRIMARY[0], end=W_PRIMARY[1])
main = {}
for spec in ("P", "K"):
    for bands in ("S", "M"):
        for coding in ("clock", "eft", "bheff"):
            main[f"{spec}-{bands}-{coding}"] = run_did(W, BREAK, spec, bands, coding)
OUT["main"] = main
pr = main["P-S-clock"]["d_out"]
print("PRIMARY P-S-clock:", {k: pr[k] for k in ("closed_pre", "closed_post", "open_pre", "open_post", "did", "did_log")}, flush=True)
for k, v in main.items():
    kk = key(v, k[0])
    print(f"{k:14s} cl {kk['closed_pre'][0]:.4f}->{kk['closed_post'][0]:.4f}  op {kk['open_pre'][0]:.4f}->{kk['open_post'][0]:.4f}"
          f"  DiD {kk['did'][0]:.4f} ({kk['did'][1]:.4f})  log {kk['did_log']}", flush=True)

# ------------------------------------------------------------------ (2) robustness
# demand-only flow: beta_D from column (2) with z_dem, monthly bands, month FE, full sample
rD = ols(a, "y", ["z_dem", "up_M", "dn_M"], "month")
BETA_D = coef(rD, "z_dem")[0]
OUT["beta_D"] = dict(estimate=coef(rD, "z_dem"), rob_demand_ofi_fixed_D=json.load(open(os.path.join(TAB, "rob_demand_ofi.json")))["thr_col2"]["fixed_D"]["f"][0])
a["RD_S"] = -a.s_S * (a.y - BETA_D * a.z_dem); a["RD_M"] = -a.s_M * (a.y - BETA_D * a.z_dem)
W = window(a, BREAK, start=W_PRIMARY[0], end=W_PRIMARY[1])

ROB_SPECS = [("P", "S"), ("P", "M"), ("K", "M"), ("K", "S")]
rob = {}
variants = {
    "window_pm2m": dict(data=window(a, BREAK, pre=61, post=59)),
    "window_jan_aug": dict(data=window(a, BREAK, start="2024-01-02", end="2024-08-31")),
    "excl_12_16apr": dict(data=W[~W.ist_date.between("2024-04-12", "2024-04-16")]),
    "excl_9_16apr": dict(data=W[~W.ist_date.between("2024-04-09", "2024-04-16")]),
    "demand_ofi": dict(data=W, demand=True),
    "free_beta": dict(data=W, free_beta=True),
    "week_cluster": dict(data=W, cluster="week"),
}
for name, v in variants.items():
    rob[name] = {}
    for spec, bands in ROB_SPECS:
        if v.get("free_beta") and spec == "K":
            continue
        kw = dict(cluster=v.get("cluster", "day"))
        if v.get("demand"):
            kw.update(flow="z_dem", Rcol="RD_" + bands)
        if v.get("free_beta"):
            kw.update(free_beta=True)
        rob[name][f"{spec}-{bands}-clock"] = run_did(v["data"], BREAK, spec, bands, "clock", **kw)
    # EFT coding for the primary spec in every variant
    if not v.get("free_beta"):
        kw = dict(cluster=v.get("cluster", "day"))
        if v.get("demand"):
            kw.update(flow="z_dem", Rcol="RD_S")
        rob[name]["P-S-eft"] = run_did(v["data"], BREAK, "P", "S", "eft", **kw)
OUT["robustness"] = rob
for name, v in rob.items():
    for k, o in v.items():
        kk = key(o, k[0])
        print(f"{name:15s} {k:12s} DiD {kk['did'][0]:.4f} ({kk['did'][1]:.4f})  log {kk['did_log']}", flush=True)


# FXCM benchmark (panel construction copied from rob_fxcm_benchmark.py, Jan-Aug 2024 only)
def load_fxcm(pair="USDTRY"):
    dfs = []
    for f in sorted(glob.glob(os.path.join(RAW, "fxcm", f"{pair}_202[34]_*.csv.gz"))):
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
    m["fxv"] = 1.0
    return m


def profile_lu(y, x, q, trim=0.05, step=1.0):
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return float(grid[j]), float(grid[i])


def fxcm_panel():
    P0 = pd.read_parquet(os.path.join(RAW, "panel_TRY.parquet")).drop(columns=["fx_mid", "fx_spread", "fxv"], errors="ignore")
    P0 = P0.loc["2024-01-01":"2024-08-31"]
    fx = load_fxcm()
    p = P0.join(fx.reindex(P0.index, method="ffill", limit=2))
    p["b"] = 1e4 * np.log(p.p_usdt / p.fx_mid)
    hr = p.index.hour + p.index.minute / 60
    p.loc[(hr >= 20.5) & (hr < 22.5), "b"] = np.nan
    p.loc[p.index.dayofweek >= 5, "b"] = np.nan
    p.loc[p.b.abs() > 1000, "b"] = np.nan
    g = p.resample("5min")
    f = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum()})
    f = f[f.nb >= 2]
    f["month"] = f.index.tz_localize(None).to_period("M").astype(str)
    f["q"] = f.ofi / f.ofi.groupby(f.month).transform("std")
    consec = f.index.to_series().diff() == pd.Timedelta("5min")
    f["y"] = f.b.diff(); f["x"] = f.b.shift(1)
    f = f[consec & f.y.notna() & f.x.notna() & f.q.notna()].copy()
    f["day"] = f.index.date
    ih = (f.index.hour + 3) % 24
    f["bh"] = ((ih >= 9) & (ih < 17)).astype(float)
    mb, sbands = {}, {}
    for m_, d in f.groupby("month"):
        if len(d) >= 500:
            mb[m_] = profile_lu(d.y.values, d.x.values, d.q.values)
    f["state"] = np.where(f.bh == 1, "open", "closed")
    for (m_, st), d in f.groupby(["month", "state"]):
        if len(d) >= 300:
            sbands[(m_, st)] = profile_lu(d.y.values, d.x.values, d.q.values)
    f["l"] = f.month.map(lambda k: mb.get(k, (np.nan, np.nan))[0]); f["u"] = f.month.map(lambda k: mb.get(k, (np.nan, np.nan))[1])
    f["l_s"] = [sbands.get((m_, s), (np.nan, np.nan))[0] for m_, s in zip(f.month, f.state)]
    f["u_s"] = [sbands.get((m_, s), (np.nan, np.nan))[1] for m_, s in zip(f.month, f.state)]
    f = f[f.l.notna() & f.l_s.notna()].copy()
    return f, mb, sbands


try:
    FXJ = json.load(open(os.path.join(TAB, "rob_fxcm_benchmark.json")))
    BETA_FX = FXJ["fxcm"]["thr"]["q"][0]
    F, mbF, sbF = fxcm_panel()
    F = add_calendar(F)
    F = add_band_vars(F, "l", "u", "M", BETA_FX)
    F = add_band_vars(F, "l_s", "u_s", "S", BETA_FX)
    band_check = {m_: [list(v), FXJ["fxcm"]["bands"].get(m_)] for m_, v in mbF.items()}
    WF = window(F, BREAK, start=W_PRIMARY[0], end=W_PRIMARY[1])
    OUT["fxcm"] = dict(beta=BETA_FX)
    OUT["fxcm"]["monthly_bands_vs_published_fxcm"] = band_check
    OUT["fxcm"]["state_bands"] = {f"{k[0]}_{k[1]}": list(v) for k, v in sbF.items()}
    OUT["fxcm"]["n_days_window"] = int(WF.day.nunique())
    rob["fxcm"] = {f"{spec}-{bands}-clock": run_did(WF, BREAK, spec, bands, "clock") for spec, bands in ROB_SPECS}
    rob["fxcm"]["P-S-eft"] = run_did(WF, BREAK, "P", "S", "eft")
    for k, o in rob["fxcm"].items():
        kk = key(o, k[0])
        print(f"fxcm            {k:12s} DiD {kk['did'][0]:.4f} ({kk['did'][1]:.4f})  log {kk['did_log']}", flush=True)
except Exception as ex:                                  # FXCM variant is optional: record a failure, do not stop
    OUT["fxcm"] = dict(error=repr(ex))
    print("FXCM failed:", repr(ex), flush=True)

# ------------------------------------------------------------------ (3) placebo break dates
LISTED = ["2024-06-04", "2024-09-04", "2025-01-06", "2025-04-04", "2025-09-04", "2026-01-05", "2026-04-06"]
hol_set = set(pd.to_datetime(HOL))
EXT = []
for m_ in pd.period_range("2024-08", "2026-05", freq="M"):
    dd = m_.start_time
    while dd.dayofweek >= 5 or dd in hol_set:
        dd += pd.Timedelta(days=1)
    EXT.append(str(dd.date()))
CHANGES = [BREAK, pd.Timestamp("2026-08-26")]


def placebo_row(D):
    D = pd.Timestamp(D)
    d = window(a, D)
    lo, hi = D - pd.Timedelta(days=PRE_D), D + pd.Timedelta(days=POST_D)
    contam = any(lo <= c <= hi for c in CHANGES)
    row = dict(date=str(D.date()), contains_fast_change=bool(contam))
    for spec, bands in (("P", "S"), ("K", "M")):
        try:
            o = run_did(d, D, spec, bands, "clock")
            kk = key(o, spec)
            row[f"{spec}-{bands}"] = dict(did=kk["did"], t=kk["did_t"], did_log=kk["did_log"],
                                         closed_pre=kk["closed_pre"][0], closed_post=kk["closed_post"][0],
                                         open_pre=kk["open_pre"][0], open_post=kk["open_post"][0])
        except Exception as ex:
            row[f"{spec}-{bands}"] = dict(error=repr(ex))
    return row


actual = placebo_row(BREAK)
pl_listed = [placebo_row(D) for D in LISTED]
pl_ext = [placebo_row(D) for D in EXT]


def pvals(rows, act):
    res = {}
    for sk in ("P-S", "K-M"):
        ok = [r for r in rows if not r["contains_fast_change"] and "did" in r[sk]]
        g = np.array([r[sk]["did"][0] for r in ok]); t = np.array([r[sk]["t"] for r in ok])
        lg = np.array([r[sk]["did_log"][0] for r in ok if r[sk]["did_log"]])
        A = act[sk]
        res[sk] = dict(N=len(ok), actual_did=A["did"][0], actual_t=A["t"],
                       p_level=float((1 + np.sum(g >= A["did"][0])) / (1 + len(g))),
                       p_t=float((1 + np.sum(t >= A["t"])) / (1 + len(t))),
                       p_abs_level=float((1 + np.sum(np.abs(g) >= abs(A["did"][0]))) / (1 + len(g))),
                       placebo_did_quantiles=[float(x) for x in np.quantile(g, [0, .1, .5, .9, 1])] if len(g) else None,
                       placebo_t_quantiles=[float(x) for x in np.quantile(t, [0, .1, .5, .9, 1])] if len(t) else None,
                       share_placebo_t_above_1_645=float(np.mean(t > 1.645)) if len(t) else None,
                       share_placebo_abs_t_above_1_96=float(np.mean(np.abs(t) > 1.96)) if len(t) else None)
        if A["did_log"]:
            res[sk].update(actual_did_log=A["did_log"][0], N_log=int(len(lg)),
                           p_log=float((1 + np.sum(lg >= A["did_log"][0])) / (1 + len(lg))))
    return res


OUT["placebo"] = dict(actual=actual, listed=pl_listed, extended=pl_ext,
                      p_listed=pvals(pl_listed, actual), p_extended=pvals(pl_ext, actual))
print("placebo p (listed):", json.dumps(OUT["placebo"]["p_listed"]), flush=True)
print("placebo p (extended):", json.dumps(OUT["placebo"]["p_extended"]), flush=True)
for r in pl_listed + pl_ext:
    print(r["date"], r["contains_fast_change"], {sk: (round(r[sk]["did"][0], 4), round(r[sk]["t"], 2)) if "did" in r[sk] else r[sk] for sk in ("P-S", "K-M")}, flush=True)

# ------------------------------------------------------------------ (4) dose-response with the real cap
fxm = np.exp(a.pg).groupby(a.month).mean()
a["fx_month"] = a.month.map(fxm)
a["cap_try"] = np.where(a.ist_date < BREAK, 50000.0, np.where(a.ist_date < pd.Timestamp("2026-08-26"), 100000.0, 300000.0))
a["cap_usd"] = a.cap_try / a.fx_month
a["regime"] = np.where(a.ist_date < BREAK, "r50", np.where(a.ist_date < pd.Timestamp("2026-08-26"), "r100", "r300"))
cap_tab = a.groupby(["month", "regime"]).agg(usdtry=("fx_month", "first"), cap_usd=("cap_usd", "first")).reset_index()
OUT["cap_usd_by_month"] = cap_tab.round(2).to_dict(orient="records")


def run_dose(d, spec, coding, trend=False):
    d = d.copy()
    C = d["C_" + coding].values
    lc = np.log(d.cap_usd.values); lc = lc - lc.mean()
    d["lc"] = lc
    d["t_yr"] = ((d.ist_date - pd.Timestamp("2024-01-01")).dt.days / 365.25).values
    t_c = d.t_yr.values - d.t_yr.values.mean()
    mon = pd.get_dummies(d.month).astype(float)
    bands = "S" if spec == "P" else "M"
    d["fe"] = d.month + "_" + d.regime + ("_" + C.astype(int).astype(str) if bands == "S" else "")
    vars_ = (["dout_S", "din_S"] if spec == "P" else ["q", "up_M", "dn_M"])
    cols = []
    for v in vars_:
        for m_ in mon.columns:
            d[f"{v}_m{m_}"] = d[v].values * mon[m_].values; cols.append(f"{v}_m{m_}")
        d[v + "_C"] = d[v] * C; d[v + "_Clc"] = d[v] * C * lc; cols += [v + "_C", v + "_Clc"]
        if trend:
            d[v + "_Ct"] = d[v] * C * t_c; cols.append(v + "_Ct")
    r = ols(d, "R_S" if spec == "P" else "y", cols, "fe", drop_zero=True)
    kv = "dout_S" if spec == "P" else "up_M"
    o = dict(theta=coef(r, kv + "_Clc"), closed_gap_at_mean=coef(r, kv + "_C"), n=r["n"], n_days=r["n_clusters"],
             lc_range=[float(lc.min()), float(lc.max())], cap_usd_range=[float(d.cap_usd.min()), float(d.cap_usd.max())])
    o["theta_t"] = o["theta"][0] / o["theta"][1]
    o["effect_of_doubling"] = [o["theta"][0] * np.log(2), o["theta"][1] * np.log(2)]
    if spec == "P":
        o["placebo_din_theta"] = coef(r, "din_S_Clc")
    else:
        o["theta_km"] = coef(r, "dn_M_Clc")
    if trend:
        o["trend"] = coef(r, kv + "_Ct")
    return o


DR = a[a.ist_date <= pd.Timestamp("2026-08-25")]
dose = {}
for spec in ("P", "K"):
    dose[f"{spec}-clock"] = run_dose(DR, spec, "clock")
    dose[f"{spec}-clock-trend"] = run_dose(DR, spec, "clock", trend=True)
    dose[f"{spec}-eft"] = run_dose(DR, spec, "eft")
    dose[f"{spec}-clock-excl_crisis2025"] = run_dose(DR[~DR.ist_date.between("2025-03-19", "2025-04-30")], spec, "clock")
    dose[f"{spec}-clock-with300k"] = run_dose(a, spec, "clock")
OUT["dose_response"] = dose
for k, v in dose.items():
    print(f"dose {k:28s} theta {v['theta'][0]:.4f} ({v['theta'][1]:.4f}) gap {v['closed_gap_at_mean'][0]:.4f}"
          + (f" trend {v['trend'][0]:.4f} ({v['trend'][1]:.4f})" if "trend" in v else ""), flush=True)

# monthly closed and open speeds over the whole sample (descriptive, for the figure)
def by_period(d, spec, period_col):
    d = d.copy()
    C = d.C_clock.values
    bands = "S" if spec == "P" else "M"
    d["fe"] = d[period_col] + ("_" + C.astype(int).astype(str) if bands == "S" else "")
    per = sorted(d[period_col].unique())
    vars_ = ["dout_S", "din_S"] if spec == "P" else ["q", "up_M", "dn_M"]
    cols = []
    for v in vars_:
        for p_ in per:
            m_ = (d[period_col] == p_).values.astype(float)
            d[f"{v}_{p_}_o"] = d[v] * m_ * (1 - C); d[f"{v}_{p_}_c"] = d[v] * m_ * C
            cols += [f"{v}_{p_}_o", f"{v}_{p_}_c"]
    r = ols(d, "R_S" if spec == "P" else "y", cols, "fe", drop_zero=True)   # zero-width state bands: no d_in
    kv = "dout_S" if spec == "P" else "up_M"
    rows = []
    for p_ in per:
        o, c = coef(r, f"{kv}_{p_}_o"), coef(r, f"{kv}_{p_}_c")
        gap = lin(r, {f"{kv}_{p_}_c": 1, f"{kv}_{p_}_o": -1})
        rows.append(dict(period=p_, open=o, closed=c, closed_minus_open=gap))
    return rows


E = a[(a.ist_date >= pd.Timestamp("2024-01-02")) & (a.ist_date <= pd.Timestamp("2024-08-31"))].copy()
E["period"] = E.month.where(~((E.month == "2024-04") & (E.ist_date < BREAK)), "2024-03")
ev = {"P-S": by_period(E, "P", "period"), "K-M": by_period(E, "K", "period")}
OUT["event_time"] = ev
mon_all = by_period(DR, "P", "month")
for r_ in mon_all:
    r_["cap_usd"] = float(DR.loc[DR.month == r_["period"], "cap_usd"].mean())
OUT["monthly_full_sample_P-S"] = mon_all

# ------------------------------------------------------------------ (5) figure
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
CO, CC, CG = "#2a78d6", "#eb6834", "#52514e"          # open, closed, neutral ink
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.8))
labs = ["Jan", "Feb", "Mar*", "Apr", "May", "Jun", "Jul", "Aug"]
for j, (sk, ttl) in enumerate([("P-S", "(a) Pooled, price side, state bands"), ("K-M", r"(b) $\kappa^+$ above the band, monthly bands")]):
    A_ = ax[0, j]; R_ = ev[sk]; xx = np.arange(len(R_))
    for nm, col, mk, off in [("open", CO, "o", -0.1), ("closed", CC, "s", 0.1)]:
        A_.errorbar(xx + off, [r_[nm][0] for r_ in R_], yerr=[1.96 * r_[nm][1] for r_ in R_], fmt=mk + "-", ms=4.5, lw=1.4,
                    color=col, ecolor=col, elinewidth=0.9, capsize=0, mec="white", mew=0.8,
                    label="banks open (09-17)" if nm == "open" else "banks closed")
    A_.axvline(2.5, color=CG, lw=0.8, ls="--"); A_.axhline(0, color=CG, lw=0.5)
    A_.text(2.6, 0.97 if j == 0 else 0.62, "FAST cap\n50k → 100k TRY", fontsize=7, color=CG, va="top",
            transform=matplotlib.transforms.blended_transform_factory(A_.transData, A_.transAxes))
    A_.set_xticks(xx); A_.set_xticklabels(labs[:len(R_)]); A_.set_title(ttl, fontsize=9)
    A_.grid(axis="y", color="#e6e5e1", lw=0.6); A_.set_axisbelow(True)
ax[0, 0].set_ylabel("speed per 5 min")
ax[0, 1].legend(frameon=False, fontsize=7.5, loc="upper right")
# (c) dose-response: monthly closed-minus-open speed against the USD value of the cap
C_ = ax[1, 0]
cu = np.array([r_["cap_usd"] for r_ in mon_all]); gp = np.array([r_["closed_minus_open"][0] for r_ in mon_all])
gs = np.array([r_["closed_minus_open"][1] for r_ in mon_all])
is50 = np.array([r_["period"] < "2024-04" for r_ in mon_all])
C_.errorbar(cu[~is50], gp[~is50], yerr=1.96 * gs[~is50], fmt="o", ms=4, color=CO, ecolor="#b7cdea", elinewidth=0.8, mec="white", mew=0.6,
            label="100k TRY cap (Apr 2024 on)")
C_.errorbar(cu[is50], gp[is50], yerr=1.96 * gs[is50], fmt="s", ms=4.5, color=CC, ecolor="#f3c3ad", elinewidth=0.8, mec="white", mew=0.6,
            label="50k TRY cap (Jan-Mar 2024)")
C_.axhline(0, color=CG, lw=0.5); C_.set_xscale("log")
C_.set_ylim(-0.12, 0.08)          # truncates the 95% interval of one month (reported in the json)
C_.set_xticks([1500, 2000, 2500, 3000]); C_.set_xticklabels(["1,500", "2,000", "2,500", "3,000"])
C_.minorticks_off()
C_.set_xlabel("FAST cap in USD (monthly, log scale)"); C_.set_ylabel("closed minus open speed")
C_.set_title("(c) Closed-hours gap by month, Jan 2024-Aug 2026", fontsize=9)
C_.legend(frameon=False, fontsize=7.5, loc="lower right"); C_.grid(axis="y", color="#e6e5e1", lw=0.6); C_.set_axisbelow(True)
# (d) placebo distribution
D_ = ax[1, 1]
gx = [r_["P-S"]["did"][0] for r_ in pl_ext if "did" in r_["P-S"]]
D_.hist(gx, bins=12, color="#b7cdea", edgecolor="white", linewidth=1.0, label="placebo dates (Aug 2024-May 2026)")
D_.axvline(actual["P-S"]["did"][0], color=CC, lw=2, label="4 April 2024")
D_.axvline(0, color=CG, lw=0.5)
D_.set_xlabel("DiD in speed per 5 min"); D_.set_ylabel("number of break dates")
D_.set_title("(d) Placebo break dates, pooled spec", fontsize=9)
D_.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rob_{NAME}.pdf")); plt.close(fig)


# ------------------------------------------------------------------ classification (see docstring)
p = OUT["main"]["P-S-clock"]["d_out"]
pe = OUT["placebo"]["p_extended"]["P-S"]
sig = p["did"][0] > 0 and p["did_p_one_sided"] < 0.05
if sig and pe["p_level"] <= 0.10:
    verdict = "supported"
elif p["did"][0] <= 0 or p["did_p_one_sided"] >= 0.05:
    verdict = "not supported"
else:
    verdict = "suggestive"
OUT["verdict"] = dict(rule="supported if primary DiD > 0 at 5% one-sided and extended-placebo p <= 0.10; not supported if DiD <= 0 or insignificant",
                      verdict=verdict, primary_did=p["did"], primary_p_one_sided=p["did_p_one_sided"],
                      placebo_p_level=pe["p_level"], mde80_level=p["mde80_level"], benchmark_doubling=p["benchmark_level_doubling"],
                      mde80_log=p["mde80_log"])
# descriptive tally: sign of the closed-hours change across main specs
OUT["verdict"]["posthoc_closed_change_by_spec"] = {k: key(v, k[0])["closed_change"] for k, v in OUT["main"].items()}
OUT["verdict"]["posthoc_n_main_specs_closed_speed_rose"] = int(sum(key(v, k[0])["closed_change"][0] > 0 for k, v in OUT["main"].items()))
print("VERDICT", OUT["verdict"], flush=True)


# ------------------------------------------------------------------ LaTeX table
def star_p(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def f2(v, nd=4):
    if v is None:
        return "", ""
    return f"${v[0]:.{nd}f}{star_p(v[0], v[1])}$", f"$({v[1]:.{nd}f})$"


rows = [("\\emph{A. Main estimates, 2 Jan -- 3 Jul 2024}", None),
        ("Pooled, state bands", OUT["main"]["P-S-clock"]["d_out"]),
        ("Pooled, monthly bands", OUT["main"]["P-M-clock"]["d_out"]),
        ("$\\kappa^+$, monthly bands", OUT["main"]["K-M-clock"]["kp"]),
        ("$\\kappa^+$, state bands", OUT["main"]["K-S-clock"]["kp"]),
        ("Pooled, state bands, EFT hours", OUT["main"]["P-S-eft"]["d_out"]),
        ("$\\kappa^+$, monthly bands, EFT hours", OUT["main"]["K-M-eft"]["kp"]),
        ("\\emph{B. Pooled, state bands: robustness}", None),
        ("Window $\\pm$2 months", rob["window_pm2m"]["P-S-clock"]["d_out"]),
        ("Window Jan -- Aug 2024", rob["window_jan_aug"]["P-S-clock"]["d_out"]),
        ("Excluding 12--16 April 2024", rob["excl_12_16apr"]["P-S-clock"]["d_out"]),
        ("Excluding 9--16 April 2024", rob["excl_9_16apr"]["P-S-clock"]["d_out"]),
        ("Small and medium order flow", rob["demand_ofi"]["P-S-clock"]["d_out"]),
        ("$\\beta$ estimated by cell", rob["free_beta"]["P-S-clock"]["d_out"]),
        ("Clustered by week", rob["week_cluster"]["P-S-clock"]["d_out"])]
if "P-S-clock" in rob.get("fxcm", {}):
    rows.append(("FXCM benchmark", rob["fxcm"]["P-S-clock"]["d_out"]))
with open(os.path.join(TAB, f"rob_{NAME}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{2}{c}{Banks closed} & \\multicolumn{2}{c}{Banks open} & \\multicolumn{2}{c}{Difference-in-differences} \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & Before & After & Before & After & Level & Log \\\\\n\\midrule\n")
    for lab, o in rows:
        if o is None:
            fh.write(f"{lab} & & & & & & \\\\\n"); continue
        cs = [f2(o[k]) for k in ("closed_pre", "closed_post", "open_pre", "open_post", "did")] + [f2(o["did_log"], 2)]
        fh.write(lab + " & " + " & ".join(c[0] for c in cs) + " \\\\\n & " + " & ".join(c[1] for c in cs) + " \\\\\n")
    pl_ = OUT["placebo"]["p_extended"]
    fh.write("\\midrule\n\\emph{C. Placebo break dates (" + f"{pl_['P-S']['N']}" + " dates, Aug 2024 -- May 2026)} & & & & & & \\\\\n")
    fh.write(f"Pooled, state bands: randomization $p$ (one-sided) & \\multicolumn{{6}}{{c}}{{{pl_['P-S']['p_level']:.2f}}} \\\\\n")
    fh.write(f"$\\kappa^+$, monthly bands: randomization $p$ (one-sided) & \\multicolumn{{6}}{{c}}{{{pl_['K-M']['p_level']:.2f}}} \\\\\n")
    dP, dK = dose["P-clock"], dose["K-clock"]
    fh.write("\\emph{D. Dose-response, Jan 2024 -- Aug 2026} & & & & & & \\\\\n")
    c0, s0 = f2(dP["theta"]); fh.write(f"Pooled: closed $\\times\\log$ cap (USD) & \\multicolumn{{6}}{{c}}{{{c0} {s0}}} \\\\\n")
    c0, s0 = f2(dK["theta"]); fh.write(f"$\\kappa^+$: closed $\\times\\log$ cap (USD) & \\multicolumn{{6}}{{c}}{{{c0} {s0}}} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

OUT["runtime_s"] = time.time() - T0
json.dump(OUT, open(os.path.join(TAB, f"rob_{NAME}.json"), "w"), indent=1, default=float)
print("done", OUT["runtime_s"])
