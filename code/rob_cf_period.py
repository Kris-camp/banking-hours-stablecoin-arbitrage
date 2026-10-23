"""rob_cf_period: the 24/7-banking counterfactual of Table emp_cf by sample period, and with period-specific speeds.

The counterfactual of Table emp_cf (empirics_mech.py, section 4, `replay`) applies the full-sample
speeds of Table emp_state col (1) to every month, including April-August 2026, when the business-hours increase in
the speed above the band is about zero (rob_by_period / rob_size_dependence sub-period estimates). This script
(1) decomposes the baseline effects by period, (2) re-runs the counterfactual with speeds estimated separately by
period, and (3) attaches 90% parameter-uncertainty intervals as in Table emp_cf. For reference, Table emp_cf reports a share >50 bp
outside the band -19% [-27,-3], mean distance -13%, wedge -12% = 2.3 bp, ~USD 7m on 31bn USDT of small/medium volume.

DESIGN
------
Data and model: exactly as empirics_mech.py section 4. a5.parquet joined (left) with ofi_size, inner-joined with the
state bands (l_s, u_s); y = db on [q, up_s, dn_s, q*bh, up_s*bh, dn_s*bh] with month-by-state FE (ms), up_s =
-(x-u_s)+, dn_s = (l_s-x)+, bh = 09:00-17:00 Istanbul clock (as in emp_cf; not bh_eff). Month-by-state effects
alpha_ms = ms-mean of (y - X b); realized residual eps = y - alpha - X b, so the replay is exact. Replay resets the
premium to its observed value at the start of each contiguous five-minute segment. Counterfactual: closed-hours
speeds (kappa+ and kappa-) replaced by the open-hours speeds, floored at zero (emp_cf floors only in the draws; the
full-sample point speeds are positive, so the floor does not change the baseline). Everything else held fixed.
Metrics (as emp_cf): share of intervals >50 bp (and >25 bp) outside the band; mean distance outside the band (bp);
SD of the premium around the band midpoint; wedge = small+medium (aggressive trades < 10k USDT) volume-weighted
distance outside the band (bp); USD amount = sum_t dem_t * (exc_t^base - exc_t^cf) / 1e4, in USD (USDT taken
at par), which equals the change in the wedge times total small/medium volume (the paper's "about USD 7 million").
Duration of excursions is not decomposed (not additive).

Periods (UTC timestamps): calendar years 2024, 2025, 2026 (Jan-Aug); sub-periods A = 1 Jan 2024 - 30 Jun 2025,
B = 1 Jul - 12 Nov 2025, C = 13 Nov 2025 - 31 Mar 2026, D = 1 Apr - 31 Aug 2026 (the sub-periods of the paper's
Appendix D, CUTS in rob_size_dependence.py).

(1) Decomposition of the baseline (full-sample speeds; the emp_cf replay itself, first verified to reproduce emp_cf
    to machine precision, including its 90% intervals with the same seed and draws). For each period P: observed
    and counterfactual levels of every metric on P's intervals, the change and % change, and P's contribution to
    the full-sample change: share metrics and mean distance: (sum over P of the change)/N_total; wedge: (sum over
    P of dem*change)/dem_total; USD amount additive. Reported: P's share of the total change (the "how much comes
    from 2024" number is the 2024 share of the USD amount and of the >50 bp reduction). Derived row: the full-sample
    total with Apr-Aug 2026 contributing nothing (total minus D's contribution).
(2) Period-specific speeds. Two schemes: by calendar year, and by the four sub-periods. In each period the model is
    re-estimated on that period's intervals only: bands re-profiled for every (month x state) cell within the period
    (profile() of empirics_state: 1 bp grid, 5% trimming, cells >= 300 intervals); month-by-state FE within the
    period (November 2025 is split between B and C, so its bands and effects are estimated separately on each side
    of 13 Nov); all six slopes period-specific. Year cells coincide with the published cells, so the year bands must
    reproduce the published state bands (checked; max abs difference reported). The replay runs over the full
    sample with interval-specific parameters (each interval uses its own period's bands, effects, beta and speeds);
    it is exact by construction (checked). A segment that crosses a period boundary carries its counterfactual
    deviation forward under the next period's parameters (no extra reset). Metrics are evaluated against each
    scheme's own bands (identical to the published bands except November 2025 in the sub-period scheme).
    Secondary (sub-period and year schemes): counterfactual closed-hours speeds max(open, closed) side by side, i.e. a
    period in which open-hours reversion is slower than closed-hours reversion keeps its closed-hours speed (round-the-clock
    banking cannot slow arbitrage); open-hours speeds as in the primary. Primary is the literal emp_cf counterfactual
    (closed := open, floored at 0). In both, only closed-hours speeds change.
(3) Parameter uncertainty as in emp_cf: 200 draws of each model's slope vector from N(b, V), V the day-clustered
    covariance WITHOUT the finite-sample factor (exactly emp_cf's Vb), alpha and eps recomputed under each draw so the
    replay stays exact, counterfactual speeds floored at 0; 5th and 95th percentiles of each change. Baseline:
    default_rng(1) as emp_cf. Period schemes: periods drawn independently (day clusters do not cross period
    boundaries), default_rng(2) for years, default_rng(3) for sub-periods. Reported s.e. of speeds use the
    finite-sample-corrected day-clustered covariance (ols/ols_fe convention).

Summary classifications stored in the JSON:
  - "concentrated in 2024": 2024 accounts for more than half of the total USD amount in the baseline decomposition.
  - "the headline is robust to period-specific speeds": the full-sample % reduction in the >50 bp share under
    year-specific speeds lies inside the published 90% interval [-27%, -3%]; same check for the sub-period scheme.
  - "Apr-Aug 2026 matters": D's contribution to the baseline USD amount exceeds 10% of the total.

ONE-SIDED COUNTERFACTUALS. The 2024 kappa+ increment (0.0065) is close to the full-sample one (0.0067), while the
2024 kappa- increment is much smaller (0.0015 vs 0.0082). Every counterfactual is therefore also run changing one
side only: "kappa+ only" (closed kappa+ := open kappa+, kappa- at its estimated state values) and
"kappa- only" (the reverse), for the baseline decomposition and both period schemes, with the same draws. The share
of intervals >50 bp outside the band is also split into above (b-u > 50) and below (l-b > 50).
CRISIS WINDOW. Because the kappa- channel carries most of the baseline effect and the 2025 kappa- increment is
large, the full-sample model re-estimated without the 19 March - 30 April 2025 lira-crisis
window (the paper's "_excl" robustness window), published bands kept; the replay and counterfactual then run on the
whole sample with those slopes (alpha and eps recomputed on all intervals, so the replay stays exact); draws with
default_rng(4).

Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, data_raw/ofi_size/*.parquet, tables/emp_mech.json (check).
Outputs: tables/rob_cf_period.json, tables/rob_cf_period.tex. No existing file is modified.
"""
import os, glob, json, time
import numpy as np, pandas as pd
from numba import njit
from model import _ssr_grid

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables"))
NAME = "cf_period"
T_START = time.time()
NDRAW = 200

# ------------------------------------------------------------------ data (identical construction to empirics_mech.py)
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
a["day"] = a.index.date
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
a = a.join(sb, how="inner")
a["l"], a["u"] = a.l_s, a.u_s
assert len(a) == 162624 and a.index.is_monotonic_increasing
a["state"] = np.where(a.bh == 1, "open", "closed")
a["ms"] = a.month + "_" + a.state

YEARS = [("2024", "2024-01-01", "2025-01-01"), ("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", "2026-09-01")]
SUBS = [("A", "2024-01-01", "2025-07-01"), ("B", "2025-07-01", "2025-11-13"), ("C", "2025-11-13", "2026-04-01"),
        ("D", "2026-04-01", "2026-09-01")]
SUB_LAB = {"A": "Jan 2024--Jun 2025", "B": "Jul--12 Nov 2025", "C": "13 Nov 2025--Mar 2026", "D": "Apr--Aug 2026"}


def period_codes(cuts):
    g = np.full(len(a), -1)
    for i, (_, s0, s1) in enumerate(cuts):
        g[(a.index >= pd.Timestamp(s0, tz="UTC")) & (a.index < pd.Timestamp(s1, tz="UTC"))] = i
    assert (g >= 0).all()
    return g


GY, GS = period_codes(YEARS), period_codes(SUBS)
a["yr"] = np.array([YEARS[i][0] for i in GY]); a["sp"] = np.array([SUBS[i][0] for i in GS])
assert (a.yr.values == a.month.str[:4].values).all()

Y_ = a.y.values.astype(float); X0 = a.x.values.astype(float); Q_ = a.q.values.astype(float)
BH_ = a.bh.values.astype(np.float64)
SEG_ = np.r_[True, np.diff(a.index.values).astype("timedelta64[s]").astype(np.int64) != 300]
DEM_ = (a.vol_small.fillna(0) + a.vol_mid.fillna(0)).values
DAYC = pd.factorize(a.day)[0]
PC = ["q", "up_s", "dn_s", "q_bh", "up_s_bh", "dn_s_bh"]
OUT = dict(name=NAME, n=int(len(a)), n_days=int(a.day.nunique()), dem_total_usdt=float(DEM_.sum()),
           periods=dict(years={k: [s0, s1] for k, s0, s1 in YEARS}, subperiods={k: [s0, s1] for k, s0, s1 in SUBS}))


# ------------------------------------------------------------------ estimator pieces
def profile(y, x, q, trim=0.05, step=1.0):
    """empirics_state.profile: returns (l, u)."""
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]


def design(L, U):
    up = -np.maximum(X0 - U, 0); dn = np.maximum(L - X0, 0)
    return np.column_stack([Q_, up, dn, Q_ * BH_, up * BH_, dn * BH_])


def group_mean(v, codes, ng):
    return (np.bincount(codes, v, ng) / np.bincount(codes, minlength=ng))[codes]


def fit(idx, X, mscodes):
    """Within (month x state) OLS on rows idx. Returns b, V (emp_cf: no finite-sample factor), Vadj (ols_fe)."""
    c, ng = pd.factorize(mscodes[idx])[0], None
    ng = c.max() + 1
    Xi, Yi = X[idx], Y_[idx]
    Xw = Xi - np.column_stack([group_mean(Xi[:, k], c, ng) for k in range(Xi.shape[1])])
    Yw = Yi - group_mean(Yi, c, ng)
    XtXi = np.linalg.inv(Xw.T @ Xw); b = XtXi @ Xw.T @ Yw; e = Yw - Xw @ b
    dc = pd.factorize(DAYC[idx])[0]; nd = dc.max() + 1
    G = np.zeros((nd, Xi.shape[1])); np.add.at(G, dc, Xw * e[:, None])
    V = XtXi @ G.T @ G @ XtXi
    n, k = len(Yi), Xi.shape[1]
    Vadj = V * nd / (nd - 1) * (n - 1) / (n - k)
    return b, V, Vadj, c, ng


def components(idx, X, c, ng, b):
    """alpha (ms means of y - Xb), beta_eff, eps, speeds as interval arrays on rows idx."""
    r = Y_[idx] - X[idx] @ b
    al = group_mean(r, c, ng)
    eps = r - al
    return al, b[0] + b[3] * BH_[idx], eps


def speeds_of(b):
    return dict(kpc=b[1], kpo=b[1] + b[4], kmc=b[2], kmo=b[2] + b[5])


@njit(cache=True)
def replay(x0, seg, L, U, BH, alpha, beta_eff, q, eps, KPO, KPC, KMO, KMC):
    """empirics_mech.replay with interval-specific speeds."""
    n = len(q); out = np.empty(n); b = x0[0]
    for t in range(n):
        if seg[t]: b = x0[t]
        out[t] = b
        kp = KPO[t] if BH[t] == 1.0 else KPC[t]
        km = KMO[t] if BH[t] == 1.0 else KMC[t]
        b = b + alpha[t] + beta_eff[t] * q[t] - kp * max(b - U[t], 0.0) + km * max(L[t] - b, 0.0) + eps[t]
    return out


def sums(path, L, U, G, ng):
    exc = np.maximum(path - U, 0) + np.maximum(L - path, 0)
    dev = path - (L + U) / 2
    f = lambda w: np.bincount(G, w, ng)
    return dict(n=f(np.ones(len(G))), c50=f((exc > 50).astype(float)), c25=f((exc > 25).astype(float)), sexc=f(exc),
                c50u=f((path - U > 50).astype(float)), c50d=f((L - path > 50).astype(float)),
                sdexc=f(DEM_ * exc), sdem=f(DEM_), s1=f(dev), s2=f(dev ** 2))


def lev(S, i=None):
    """Levels of the emp_cf metrics for group i (None = full sample)."""
    g = (lambda k: S[k].sum()) if i is None else (lambda k: S[k][i])
    n = g("n")
    return dict(sd=float(np.sqrt(g("s2") / n - (g("s1") / n) ** 2)), p50=float(g("c50") / n), p25=float(g("c25") / n),
                p50_above=float(g("c50u") / n), p50_below=float(g("c50d") / n),
                mean_exc=float(g("sexc") / n), wedge=float(g("sdexc") / g("sdem")), usd_cost=float(g("sdexc") * 1e-4))


def compare(Sb, Sc, labels):
    """Per-period and total levels, changes, contributions. Sb, Sc: sums for base and cf on the same grouping."""
    N, D = Sb["n"].sum(), Sb["sdem"].sum()
    tot_b, tot_c = lev(Sb), lev(Sc)
    tot_usd = float((Sb["sdexc"] - Sc["sdexc"]).sum() * 1e-4)
    tot = dict(base=tot_b, cf=tot_c, pct={k: tot_c[k] / tot_b[k] - 1 for k in ("sd", "p50", "p25", "mean_exc", "wedge", "p50_above", "p50_below")},
               d_p50=tot_c["p50"] - tot_b["p50"], d_mean_exc=tot_c["mean_exc"] - tot_b["mean_exc"],
               d_wedge=tot_c["wedge"] - tot_b["wedge"], usd_saving=tot_usd)
    per = {}
    for i, lab in enumerate(labels):
        lb, lc = lev(Sb, i), lev(Sc, i)
        usd = float((Sb["sdexc"][i] - Sc["sdexc"][i]) * 1e-4)
        c50 = (Sc["c50"][i] - Sb["c50"][i]) / N; cex = (Sc["sexc"][i] - Sb["sexc"][i]) / N
        cwe = (Sc["sdexc"][i] - Sb["sdexc"][i]) / D
        per[lab] = dict(n=int(Sb["n"][i]), share_n=float(Sb["n"][i] / N), share_dem=float(Sb["sdem"][i] / D),
                        dem_usdt=float(Sb["sdem"][i]), base=lb, cf=lc,
                        pct={k: (lc[k] / lb[k] - 1) if lb[k] > 0 else None for k in ("sd", "p50", "p25", "mean_exc", "wedge", "p50_above", "p50_below")},
                        d_p50=lc["p50"] - lb["p50"], d_mean_exc=lc["mean_exc"] - lb["mean_exc"], d_wedge=lc["wedge"] - lb["wedge"],
                        usd_saving=usd,
                        contrib=dict(p50=float(c50), mean_exc=float(cex), wedge=float(cwe)),
                        share_of_total=dict(p50=float(c50 / tot["d_p50"]) if tot["d_p50"] != 0 else None,
                                            mean_exc=float(cex / tot["d_mean_exc"]) if tot["d_mean_exc"] != 0 else None,
                                            usd=float(usd / tot_usd) if tot_usd != 0 else None))
    return dict(total=tot, periods=per)


def flat(C):
    """Scalars to collect across draws."""
    o = {f"total_pct_{k}": v for k, v in C["total"]["pct"].items()}
    o.update(total_d_wedge=C["total"]["d_wedge"], total_usd=C["total"]["usd_saving"], total_d_p50=C["total"]["d_p50"])
    for lab, p in C["periods"].items():
        for k, v in p["pct"].items():
            o[f"{lab}_pct_{k}"] = np.nan if v is None else v
        o[f"{lab}_d_wedge"] = p["d_wedge"]; o[f"{lab}_usd"] = p["usd_saving"]
        for k, v in p["share_of_total"].items():
            o[f"{lab}_share_{k}"] = np.nan if v is None else v
    return o


def ci_of(rows):
    df = pd.DataFrame(rows)
    return {k: [float(df[k].quantile(.05)), float(df[k].quantile(.95))] if df[k].notna().any() else None for k in df}


def const(v):
    return np.full(len(a), float(v))


# ------------------------------------------------------------------ (0) baseline: reproduce emp_cf
L_ = a.l.values.astype(float); U_ = a.u.values.astype(float)
MS_ = a.ms.values
Xb = design(L_, U_)
ALL = np.arange(len(a))
b0, V0, V0adj, c0, ng0 = fit(ALL, Xb, MS_)
al0, be0, ep0 = components(ALL, Xb, c0, ng0, b0)
s0 = speeds_of(b0)
pth_b = replay(X0, SEG_, L_, U_, BH_, al0, be0, Q_, ep0, const(s0["kpo"]), const(s0["kpc"]), const(s0["kmo"]), const(s0["kmc"]))
cfk = dict(KPO=const(max(s0["kpo"], 0)), KPC=const(max(s0["kpo"], 0)), KMO=const(max(s0["kmo"], 0)), KMC=const(max(s0["kmo"], 0)))
pth_c = replay(X0, SEG_, L_, U_, BH_, al0, be0, Q_, ep0, cfk["KPO"], cfk["KPC"], cfk["KMO"], cfk["KMC"])
pub = json.load(open(os.path.join(TAB, "emp_mech.json")))["counterfactual"]
one = np.zeros(len(a), dtype=np.int64)
lb_, lc_ = lev(sums(pth_b, L_, U_, one, 1)), lev(sums(pth_c, L_, U_, one, 1))
OUT["check_baseline"] = dict(
    replay_max_abs_err=float(np.max(np.abs(pth_b - X0))),
    speeds_match_max_abs=float(max(abs(s0[k] - pub["speeds"][k]) for k in s0)),
    levels_max_abs_diff=float(max(max(abs(lb_[k] - pub["baseline"][k]), abs(lc_[k] - pub["cf"][k])) for k in ("sd", "p50", "p25", "mean_exc", "wedge"))))
print("baseline check", OUT["check_baseline"], f"{time.time() - T_START:.0f}s", flush=True)

# (1) decomposition of the baseline by period, point and draws (same seed and draws as emp_cf)
SCH = [("years", GY, YEARS), ("subperiods", GS, SUBS)]


def dec_paths(b):
    """Baseline replay and the three counterfactual paths (primary, kappa+ only, kappa- only) of the full-sample model."""
    al, be, ep = components(ALL, Xb, c0, ng0, b)
    s = speeds_of(b)
    kp, km = max(s["kpo"], 0), max(s["kmo"], 0)
    P0 = (const(s["kpo"]), const(s["kpc"]), const(s["kmo"]), const(s["kmc"]))
    pb = replay(X0, SEG_, L_, U_, BH_, al, be, Q_, ep, *P0)
    V = dict(primary=(const(kp), const(kp), const(km), const(km)), kp_only=(const(kp), const(kp), P0[2], P0[3]),
             km_only=(P0[0], P0[1], const(km), const(km)))
    return pb, {v: replay(X0, SEG_, L_, U_, BH_, al, be, Q_, ep, *k) for v, k in V.items()}


def dec_compare(pb, pcs):
    o = {}
    for sch, G, cuts in SCH:
        labs = [k for k, _, _ in cuts]
        Sb = sums(pb, L_, U_, G, len(labs))
        o[sch] = {v: compare(Sb, sums(pc, L_, U_, G, len(labs)), labs) for v, pc in pcs.items()}
    return o


pb_, pcs_ = dec_paths(b0)
assert np.array_equal(pcs_["primary"], pth_c)
DD = dec_compare(pb_, pcs_)
rng = np.random.default_rng(1)
draws0 = rng.multivariate_normal(b0, V0, size=NDRAW)
rows = {(sch, v): [] for sch, _, _ in SCH for v in pcs_}
for dr in draws0:
    o = dec_compare(*dec_paths(dr))
    for (sch, v) in rows:
        rows[(sch, v)].append(flat(o[sch][v]))
for (sch, v) in rows:
    DD[sch][v]["ci90"] = ci_of(rows[(sch, v)])
DEC = {sch: DD[sch]["primary"] for sch, _, _ in SCH}
DEC["posthoc_side"] = {v: {sch: DD[sch][v] for sch, _, _ in SCH} for v in ("kp_only", "km_only")}
ci_pub = {k: pub["ci90"][k] for k in ("sd", "p50", "p25", "mean_exc", "wedge")}
OUT["check_baseline"]["ci90_max_abs_diff_vs_emp_cf"] = float(max(
    max(abs(DEC["years"]["ci90"][f"total_pct_{k}"][j] - ci_pub[k][j]) for j in (0, 1)) for k in ci_pub))
# derived: full-sample total with Apr-Aug 2026 contributing nothing
dd = DEC["subperiods"]
DEC["total_excl_D"] = dict(
    d_p50=dd["total"]["d_p50"] - dd["periods"]["D"]["contrib"]["p50"],
    pct_p50=(dd["total"]["d_p50"] - dd["periods"]["D"]["contrib"]["p50"]) / dd["total"]["base"]["p50"],
    pct_mean_exc=(dd["total"]["d_mean_exc"] - dd["periods"]["D"]["contrib"]["mean_exc"]) / dd["total"]["base"]["mean_exc"],
    d_wedge=dd["total"]["d_wedge"] - dd["periods"]["D"]["contrib"]["wedge"],
    pct_wedge=(dd["total"]["d_wedge"] - dd["periods"]["D"]["contrib"]["wedge"]) / dd["total"]["base"]["wedge"],
    usd=dd["total"]["usd_saving"] - dd["periods"]["D"]["usd_saving"])
DEC["speeds"] = {k: float(v) for k, v in s0.items()}
DEC["speeds_se"] = dict(kpc=float(np.sqrt(V0adj[1, 1])), kpo=float(np.sqrt(V0adj[1, 1] + V0adj[4, 4] + 2 * V0adj[1, 4])),
                        kmc=float(np.sqrt(V0adj[2, 2])), kmo=float(np.sqrt(V0adj[2, 2] + V0adj[5, 5] + 2 * V0adj[2, 5])),
                        kp_incr=float(np.sqrt(V0adj[4, 4])), km_incr=float(np.sqrt(V0adj[5, 5])))
OUT["decomposition"] = DEC

# F: full-sample slopes estimated without the 19 Mar - 30 Apr 2025 crisis window
CRI = (a.index >= pd.Timestamp("2025-03-19", tz="UTC")) & (a.index < pd.Timestamp("2025-05-01", tz="UTC"))
bE, VE, VEadj, _, _ = fit(np.where(~CRI)[0], Xb, MS_)
pbE, pcsE = dec_paths(bE)
DE = dec_compare(pbE, pcsE)
rowsE = {(sch, v): [] for sch, _, _ in SCH for v in pcsE}
for dr in np.random.default_rng(4).multivariate_normal(bE, VE, size=NDRAW):
    o = dec_compare(*dec_paths(dr))
    for (sch, v) in rowsE:
        rowsE[(sch, v)].append(flat(o[sch][v]))
for (sch, v) in rowsE:
    DE[sch][v]["ci90"] = ci_of(rowsE[(sch, v)])
seE = np.sqrt(np.diag(VEadj))
OUT["posthoc_excl_crisis"] = dict(
    window=["2025-03-19", "2025-05-01"], n_excluded=int(CRI.sum()), replay_max_abs_err=float(np.max(np.abs(pbE - X0))),
    speeds={k: float(v) for k, v in speeds_of(bE).items()},
    coef={c: [float(bE[i]), float(seE[i])] for i, c in enumerate(PC)}, coef_full={c: [float(b0[i]), float(np.sqrt(V0adj[i, i]))] for i, c in enumerate(PC)},
    years=DE["years"], subperiods=DE["subperiods"])
print("F (crisis window excluded) done", f"{time.time() - T_START:.0f}s", flush=True)
print("decomposition done", OUT["check_baseline"], f"{time.time() - T_START:.0f}s", flush=True)


# ------------------------------------------------------------------ (2) period-specific speeds
def band_cells(keys):
    """Re-profile state bands for every cell defined by the tuple of column names `keys`."""
    L, U = np.full(len(a), np.nan), np.full(len(a), np.nan)
    for _, idx in a.groupby(keys).indices.items():
        if len(idx) < 300: continue
        l, u = profile(Y_[idx], X0[idx], Q_[idx])
        L[idx], U[idx] = l, u
    return L, U


def scheme(name, G, cuts, cellkeys, seed):
    labs = [k for k, _, _ in cuts]
    L, U = band_cells(cellkeys)
    assert np.isfinite(L).all()
    msk = (a.ms + "_" + pd.Series([labs[i] for i in G], index=a.index)).values
    X = design(L, U)
    fits, AR = {}, {k: np.empty(len(a)) for k in ("al", "be", "ep", "KPO", "KPC", "KMO", "KMC", "CKP", "CKM", "MKP", "MKM")}
    for i, lab in enumerate(labs):
        idx = np.where(G == i)[0]
        b, V, Vadj, c, ng = fit(idx, X, msk)
        fits[lab] = (idx, b, V, c, ng)
    res = dict(bands_max_abs_diff_vs_published=float(np.max(np.abs(np.r_[L - L_, U - U_]))),
               n_cells_differing=int(pd.Series(list(zip(L != L_, U != U_))).map(any).groupby(a.ms.values).any().sum()))

    def fill(par):
        """par: {lab: b}. Returns arrays for the baseline replay, the primary cf and the secondary cf."""
        for i, lab in enumerate(labs):
            idx, _, _, c, ng = fits[lab]
            b = par[lab]
            al, be, ep = components(idx, X, c, ng, b)
            s = speeds_of(b)
            AR["al"][idx], AR["be"][idx], AR["ep"][idx] = al, be, ep
            AR["KPO"][idx], AR["KPC"][idx], AR["KMO"][idx], AR["KMC"][idx] = s["kpo"], s["kpc"], s["kmo"], s["kmc"]
            AR["CKP"][idx], AR["CKM"][idx] = max(s["kpo"], 0), max(s["kmo"], 0)
            AR["MKP"][idx], AR["MKM"][idx] = max(s["kpo"], s["kpc"], 0), max(s["kmo"], s["kmc"], 0)
        pb = replay(X0, SEG_, L, U, BH_, AR["al"], AR["be"], Q_, AR["ep"], AR["KPO"], AR["KPC"], AR["KMO"], AR["KMC"])
        # the secondary variant changes CLOSED-hours speeds only (closed := max(open, closed, 0)); open hours keep
        # the primary's open speed, so business-hours arbitrage is never sped up and the variant remains a
        # banking-hours counterfactual.
        VAR = dict(primary=(AR["CKP"], AR["CKP"], AR["CKM"], AR["CKM"]), max_open_closed=(AR["CKP"], AR["MKP"], AR["CKM"], AR["MKM"]),
                   kp_only=(AR["CKP"], AR["CKP"], AR["KMO"], AR["KMC"]), km_only=(AR["KPO"], AR["KPC"], AR["CKM"], AR["CKM"]))
        Sb = sums(pb, L, U, G, len(labs))
        return pb, {v: compare(Sb, sums(replay(X0, SEG_, L, U, BH_, AR["al"], AR["be"], Q_, AR["ep"], *k), L, U, G, len(labs)), labs)
                    for v, k in VAR.items()}

    pb, CMP = fill({lab: fits[lab][1] for lab in labs})
    res["replay_max_abs_err"] = float(np.max(np.abs(pb - X0)))
    sp = {}
    for lab in labs:
        idx, b, V, c, ng = fits[lab]
        _, _, Vadj, _, _ = fit(idx, X, msk)
        s = speeds_of(b)
        se = dict(kpc=np.sqrt(Vadj[1, 1]), kpo=np.sqrt(Vadj[1, 1] + Vadj[4, 4] + 2 * Vadj[1, 4]),
                  kmc=np.sqrt(Vadj[2, 2]), kmo=np.sqrt(Vadj[2, 2] + Vadj[5, 5] + 2 * Vadj[2, 5]),
                  kp_incr=np.sqrt(Vadj[4, 4]), km_incr=np.sqrt(Vadj[5, 5]), beta=np.sqrt(Vadj[0, 0]), beta_bh=np.sqrt(Vadj[3, 3]))
        sp[lab] = dict(est={**{k: float(v) for k, v in s.items()}, "kp_incr": float(b[4]), "km_incr": float(b[5]),
                            "beta": float(b[0]), "beta_bh": float(b[3])},
                       se={k: float(v) for k, v in se.items()}, n=int(len(idx)), n_days=int(len(np.unique(DAYC[idx]))))
    res["speeds"] = sp
    rng_ = np.random.default_rng(seed)
    D = {lab: rng_.multivariate_normal(fits[lab][1], fits[lab][2], size=NDRAW) for lab in labs}
    R = {v: [] for v in CMP}
    for j in range(NDRAW):
        _, o = fill({lab: D[lab][j] for lab in labs})
        for v in R:
            R[v].append(flat(o[v]))
    for v in R:
        CMP[v]["ci90"] = ci_of(R[v])
    res["primary"], res["secondary_max_open_closed"] = CMP["primary"], CMP["max_open_closed"]
    res["posthoc_side"] = dict(kp_only=CMP["kp_only"], km_only=CMP["km_only"])
    print(name, "done", res["bands_max_abs_diff_vs_published"], res["replay_max_abs_err"], f"{time.time() - T_START:.0f}s", flush=True)
    return res


OUT["period_speeds"] = dict(years=scheme("years", GY, YEARS, ["month", "state"], 2),
                            subperiods=scheme("subperiods", GS, SUBS, ["month", "state", "sp"], 3))

# ------------------------------------------------------------------ summary classifications
dec = OUT["decomposition"]; ps = OUT["period_speeds"]
lo_pub, hi_pub = pub["ci90"]["p50"]
OUT["reading"] = dict(
    usd_share_2024_baseline=dec["years"]["periods"]["2024"]["share_of_total"]["usd"],
    p50_share_2024_baseline=dec["years"]["periods"]["2024"]["share_of_total"]["p50"],
    concentrated_in_2024=bool(dec["years"]["periods"]["2024"]["share_of_total"]["usd"] > 0.5),
    usd_share_D_baseline=dec["subperiods"]["periods"]["D"]["share_of_total"]["usd"],
    apr_aug_2026_matters=bool(dec["subperiods"]["periods"]["D"]["share_of_total"]["usd"] > 0.10),
    p50_pct_years=ps["years"]["primary"]["total"]["pct"]["p50"],
    p50_pct_subperiods=ps["subperiods"]["primary"]["total"]["pct"]["p50"],
    robust_years=bool(lo_pub <= ps["years"]["primary"]["total"]["pct"]["p50"] <= hi_pub),
    robust_subperiods=bool(lo_pub <= ps["subperiods"]["primary"]["total"]["pct"]["p50"] <= hi_pub))
PH = {}
for lab, C in [("full_sample_speeds", dec["posthoc_side"]), ("year_speeds", ps["years"]["posthoc_side"]),
               ("subperiod_speeds", ps["subperiods"]["posthoc_side"])]:
    for v in ("kp_only", "km_only"):
        Cy = C[v]["years"] if "years" in C[v] else C[v]
        first = "2024" if "2024" in Cy["periods"] else "A"
        PH[f"{lab}_{v}"] = dict(pct_p50=Cy["total"]["pct"]["p50"], ci_pct_p50=Cy["ci90"]["total_pct_p50"],
                                pct_wedge=Cy["total"]["pct"]["wedge"], usd=Cy["total"]["usd_saving"], ci_usd=Cy["ci90"]["total_usd"],
                                first_period=first, usd_first=Cy["periods"][first]["usd_saving"],
                                share_usd_first=Cy["periods"][first]["share_of_total"]["usd"])
OUT["posthoc_summary"] = PH
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


# ------------------------------------------------------------------ LaTeX table
def nz(t):
    return t[1:] if t.startswith("-") and float(t) == 0 else t


def pc(v):
    return "--" if v is None else nz(f"{100 * v:.0f}") + "\\%"


def br(ci, scale=100, fmt="{:.0f}"):
    return "" if ci is None else f" [{nz(fmt.format(scale * ci[0]))}, {nz(fmt.format(scale * ci[1]))}]"


def row(lab, P, ci, key, sp=None):
    """One table row: period label, speeds (optional), share>50bp base, %chg [ci], mean dist %chg [ci], wedge bp chg
    [ci], USD m [ci], share of total USD."""
    pre = "total_" if key == "total" else f"{key}_"
    if key == "total":
        pct, dw, usd, sh = P["pct"], P["d_wedge"], P["usd_saving"], None
    else:
        pct, dw, usd, sh = P["pct"], P["d_wedge"], P["usd_saving"], P["share_of_total"]["usd"]
    spd = " & & & " if sp is None else f" & {sp['kpc']:.4f} & {sp['kpo']:.4f} & "
    return (f"{lab}{spd}{P['base']['p50']:.3f} & {pc(pct['p50'])}{br(ci.get(pre + 'pct_p50'))} & "
            f"{pc(pct['mean_exc'])}{br(ci.get(pre + 'pct_mean_exc'))} & "
            f"{dw:.1f}{br(ci.get(pre + 'd_wedge'), 1, '{:.1f}')} & "
            f"{usd / 1e6:.1f}{br(ci.get(pre + 'usd'), 1e-6, '{:.1f}')} & "
            f"{'' if sh is None else pc(sh)} \\\\\n")


with open(os.path.join(TAB, f"rob_{NAME}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccccccc}\n\\toprule\n"
             " & \\multicolumn{2}{c}{$\\kappa^+$ used} & \\multicolumn{2}{c}{Share $>$50 bp outside} & Mean distance"
             " & Wedge & USD saved & Share of \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
             " & closed & open & observed & change & change & change (bp) & (million) & USD total \\\\\n\\midrule\n")
    for title, sch, labs, src in [
            ("A. Table~\\ref{tab:emp_cf} counterfactual (full-sample speeds), by calendar year", "years", [k for k, _, _ in YEARS], "dec"),
            ("B. Table~\\ref{tab:emp_cf} counterfactual (full-sample speeds), by sub-period", "subperiods", [k for k, _, _ in SUBS], "dec"),
            ("C. Speeds estimated separately by calendar year", "years", [k for k, _, _ in YEARS], "ps"),
            ("D. Speeds estimated separately by sub-period", "subperiods", [k for k, _, _ in SUBS], "ps")]:
        fh.write(f"\\multicolumn{{9}}{{l}}{{\\emph{{{title}}}}} \\\\\n")
        C = dec[sch] if src == "dec" else ps[sch]["primary"]
        for lab in labs:
            s_ = dec["speeds"] if src == "dec" else ps[sch]["speeds"][lab]["est"]
            fh.write(row(("\\quad " + (lab if sch == "years" else SUB_LAB[lab])), C["periods"][lab], C["ci90"], lab, s_))
        fh.write(row("\\quad Full sample", C["total"], C["ci90"], "total"))
        if src == "dec" and sch == "subperiods":
            t = dec["total_excl_D"]
            fh.write(f"\\quad Full sample, no effect Apr--Aug 2026 & & & {dd['total']['base']['p50']:.3f} & {pc(t['pct_p50'])} & {pc(t['pct_mean_exc'])} & "
                     f"{t['d_wedge']:.1f} & {t['usd'] / 1e6:.1f} & \\\\\n")
        if src == "ps":
            sec = ps[sch]["secondary_max_open_closed"]
            fh.write(row("\\quad Full sample, CF closed speed $=\\max$(open, closed)", sec["total"], sec["ci90"], "total"))
        fh.write("\\addlinespace\n")
    fh.write("\\multicolumn{9}{l}{\\emph{E. Counterfactual on one side of the band only (full sample)}} \\\\\n")
    for lab, C in [("Full-sample speeds", dec["posthoc_side"]), ("Speeds by calendar year", ps["years"]["posthoc_side"]),
                   ("Speeds by sub-period", ps["subperiods"]["posthoc_side"])]:
        for v, vl in [("kp_only", "$\\kappa^+$ only"), ("km_only", "$\\kappa^-$ only")]:
            CC = C[v]["years"] if "years" in C[v] else C[v]
            fh.write(row(f"\\quad {lab}, {vl}", CC["total"], CC["ci90"], "total"))
    fh.write("\\addlinespace\n\\multicolumn{9}{l}{\\emph{F. Full-sample speeds estimated without 19 Mar--30 Apr 2025}} \\\\\n")
    EX = OUT["posthoc_excl_crisis"]
    for lab in [k for k, _, _ in YEARS]:
        fh.write(row(f"\\quad {lab}", EX["years"]["primary"]["periods"][lab], EX["years"]["primary"]["ci90"], lab, EX["speeds"]))
    for v, vl in [("primary", "both sides"), ("kp_only", "$\\kappa^+$ only"), ("km_only", "$\\kappa^-$ only")]:
        fh.write(row(f"\\quad Full sample, {vl}", EX["years"][v]["total"], EX["years"][v]["ci90"], "total"))
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(json.dumps(clean(OUT["reading"]), indent=1))
print(f"runtime {time.time() - T_START:.0f}s")
