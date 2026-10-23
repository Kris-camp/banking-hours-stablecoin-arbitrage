"""rob_ratio_by_year: does the price-flow equal-ratio test of Table emp_flow hold period by period?

Table emp_flow tests that bank opening multiplies the price-side speed (col 1, kappa) and the
large-trade response (col 2, theta) by equal factors, as one shift in effective capacity K implies
(kappa = lambda*K, theta = eta*K). Full sample: difference in log open/closed ratios -0.34 (s.e. 0.51), joint
day-clustered cross-equation covariance (empirics_mech.py, section 2). This script runs the same joint test by
calendar year (2024, 2025, 2026) and before/after 13 November 2025. Where a closed-hours coefficient is not
significantly positive the log ratio is undefined or unstable, so it also reports the levels, the ratios and a test
in levels.

REFERENCE VALUES: the published full-sample cross test (emp_mech.json: log difference -0.338, s.e. 0.506; levels
statistic g = kB*tC - tB*kC, z = -0.74) and the by-year price/flow coefficients and log differences of
tables/rob_by_period.json (rob_by_period.py spec (e) and (3)): 2024 log difference -0.88 (0.98) with beta fixed,
-1.03 (0.98) with beta by year; 2025 +1.73 (0.98) / +1.76 (0.97), where the flow open/closed ratio is 0.58 because
the flow increment is negative; 2026 undefined because theta_closed < 0 (-0.064, s.e. 0.085) and, with beta fixed,
kappa_open < kappa_closed.

DESIGN
------
Construction (identical to Table emp_flow cols 1-2, empirics_mech.py; only the sample changes):
  - bands by month x clock banking state (data_raw/state_bands.parquet, empirics_state.py); s = side of the band
    (+1 above u_s, -1 below l_s, sign(x - midpoint) inside); d_out = distance outside, d_in = distance from the
    midpoint for in-band intervals; bh_eff = bh*(1-hol) with the HOL weekday-holiday list (UTC date, as there).
  - regressors [d_out, d_out*bh_eff, d_in, d_in*bh_eff]; month fixed effects; clusters = UTC date; finite-sample
    correction ng/(ng-1)*(n-1)/(n-k) of ols(); cross-equation covariance from the stacked per-day scores.
  - price side R = -s*(y - beta*q); large trades A_L = -s*z_large, z_large = ofi_large / its within-month SD
    (whole calendar month, as published); theta reported x100 (SD of flow per 100 bp).
  - beta. Primary: re-estimated within the period from the emp_main col (2) analogue (y on q, up, dn; monthly
    bands; month FE), exactly as rob_by_period.py spec (e2), which the paper cites for the year-by-year price
    side. This is "the same construction" applied to a sub-sample: in the full sample it equals the published
    beta (0.604). Alternative (e1): beta fixed at the published full-sample value.
  - bands within a period. The published bands are profiled cell by cell (month x state), each cell on its own
    data. Calendar years: every cell lies inside one year, so re-profiling within the period reproduces the
    published bands (verified in rob_by_period.py, max abs difference 0; re-checked here by construction). The
    baseline itself does not re-profile bands on sub-samples, so the primary pre/post estimates keep the published
    November 2025 bands (one cell per state for the whole month). Sensitivity (pre/post only): the two November
    2025 state cells and the November 2025 monthly band (used for beta) re-profiled separately on 1-12 November
    and 13-30 November with the empirics_state/empirics_main profile (q control, 5% trim, 1 bp grid).
Periods: full (replication; must reproduce emp_mech.json to 1e-10), 2024, 2025, 2026 (Jan-Aug), pre = Jan 2024
  to 12 Nov 2025, post = 13 Nov 2025 00:00 UTC to Aug 2026 (T0 as rob_size_dependence.py). Each period is a
  separate pair of regressions on that period's intervals (month FE nest; the November 2025 FE is estimated within
  each side of T0).
Reported per period (and per beta / band variant):
  (1) levels: kappa_C, kappa_B (increment), kappa_O = kappa_C + kappa_B; theta_C, theta_B, theta_O; s.e. and z.
  (2) ratios kappa_O/kappa_C and theta_O/theta_C (point estimates whenever the denominator is non-zero).
  (3) log ratios with delta-method s.e. where the closed and open coefficients are both positive; a log ratio is
      flagged UNSTABLE when its closed coefficient is not significantly positive (z < 1.96).
  (4) difference of log ratios (price minus flow), joint delta-method s.e., z, two-sided p, where both log ratios
      are defined (as published).
  (5) the levels test: h = theta_O*kappa_C - kappa_O*theta_C = theta_B*kappa_C -
      kappa_B*theta_C (= -g of empirics_mech.py); H0: h = 0, i.e. the vectors (kappa_C, kappa_O) and
      (theta_C, theta_O) are proportional, which is equal ratios whenever the closed coefficients are non-zero
      and is defined for any sign. Delta-method s.e. from the joint covariance, z, two-sided p.
  (6) supplementary, interpretable forms of the same null: the closed-hours large-trade response implied by the
      price-side ratio, theta_C* = theta_O*kappa_C/kappa_O, and the gap theta_C* - theta_C = h/kappa_O (delta
      method; flagged unstable if kappa_O is not significantly positive); and the difference of closed/open
      ratios kappa_C/kappa_O - theta_C/theta_O = h/(kappa_O*theta_O) (flagged unstable unless both open
      coefficients are significantly positive).
  (7) joint Wald test of h = 0 in all three calendar years (chi2, 3 df; years use disjoint day clusters, so the
      statistic is the sum of the squared z's), and in both pre and post (chi2, 2 df).
  (8) secondary check on the delta method: day bootstrap with bands held fixed (within each month of the period,
      draw that month's number of days with replacement, as rob_ratio_ci.py), B = 999, seed 20261022, beta
      re-estimated in each draw for the primary. Reported: percentile 95% CI of h, bootstrap two-sided p
      (2*min share on either side of 0), share of draws with kappa_C > 0 and with theta_C > 0, share of draws in
      which the log difference is defined and its percentile 95% CI among those draws.
Classifications stored in the JSON (primary = beta by period, published bands):
  - "equality rejected" in a period if the levels test (5) rejects at 5% two-sided; if the log-difference test
    (4) is defined and disagrees with (5), the period is reported as "mixed" with both numbers.
  - "consistent with equality" if (5) does not reject. It is called "informative" only if the log-difference test
    is defined, neither log ratio is flagged unstable, and its 95% CI lies inside [-log 2, log 2] (ratios within
    a factor of two of each other); otherwise "not rejected but uninformative".

Also reported: the mean, median and SD of the bootstrap draws of h, because the bootstrap and delta-method
p-values differ in 2024 and 2025.

Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, data_raw/ofi_size/*.parquet, tables/emp_numbers.json,
tables/emp_mech.json (replication check). Outputs: tables/rob_ratio_by_year.json,
tables/rob_ratio_by_year.tex.
Env: B_BOOT (bootstrap draws, default 999), OUT_TAB (output folder, default tables/).
"""
import os, sys, glob, json, time
import numpy as np, pandas as pd
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import _ssr_grid

RAW, TAB = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables"))
OTAB = os.environ.get("OUT_TAB", TAB)
NAME = "rob_ratio_by_year"
B_BOOT = int(os.environ.get("B_BOOT", 999))
SEED = 20261022
T0 = pd.Timestamp("2025-11-13", tz="UTC")
t_start = time.time()

# ------------------------------------------------------------------ data (empirics_mech.py / rob_by_period.py)
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
a["day"] = a.index.date
a = a.join(pd.read_parquet(os.path.join(RAW, "state_bands.parquet")), how="inner")
assert len(a) == 162624 and a[["ofi_large", "l_s", "u_s"]].notna().all().all()
a["z_large"] = a.ofi_large / a.ofi_large.groupby(a.month).transform("std")
a["state"] = np.where(a.bh == 1, "open", "closed")
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
a["hol"] = np.isin(a.day, pd.to_datetime(HOL).date).astype(float)
a["bh_eff"] = a.bh * (1 - a.hol)
a["P"] = (a.index >= T0).astype(float)
BETA_PUB = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
PCOLS = ["d_out", "d_out_be", "d_in", "d_in_be"]


def design(d):
    """Orientation and regressors from the state bands (l_s, u_s) and monthly up/dn from (l, u)."""
    d = d.copy()
    m = (d.l_s + d.u_s) / 2
    out = (d.x > d.u_s) | (d.x < d.l_s)
    d["s"] = np.where(d.x > d.u_s, 1.0, np.where(d.x < d.l_s, -1.0, np.sign(d.x - m)))
    d["d_out"] = np.maximum(d.x - d.u_s, 0) + np.maximum(d.l_s - d.x, 0)
    d["d_in"] = np.where(~out, np.abs(d.x - m), 0.0)
    d["d_out_be"] = d.d_out * d.bh_eff
    d["d_in_be"] = d.d_in * d.bh_eff
    d["up"] = -np.maximum(d.x - d.u, 0); d["dn"] = np.maximum(d.l - d.x, 0)
    d["A_L"] = -d.s * d.z_large
    d["u1"] = -d.s * d.y          # R = u1 + beta*u2
    d["u2"] = d.s * d.q
    return d


a = design(a)
assert np.allclose(a.up.values, pd.read_parquet(os.path.join(RAW, "a5.parquet")).up.reindex(a.index).values)


# ------------------------------------------------------------------ estimators (copied conventions)
def ols(d, ycol, cols, fe="month"):
    """Within-FE OLS, day-clustered with the finite-sample correction; per-day scores (rob_by_period.py)."""
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


def joint(regs):
    days = regs[0]["days"]
    for r in regs[1:]:
        assert (r["days"] == days).all()
    S = np.hstack([r["S"] for r in regs])
    b = np.concatenate([[r["b"][c] for c in r["cols"]] for r in regs])
    return b, S.T @ S * regs[0]["adj"]


def beta_col2(d):
    return ols(d, "y", ["q", "up", "dn"], fe="month")["b"]["q"]


def profile(y, x, q, trim=0.05, step=1.0):
    """empirics_main.profile / empirics_state.profile: returns (l, u)."""
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)          # S[i, j]: u = grid[i], l = grid[j]
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return float(grid[j]), float(grid[i])


def pval(zv):
    return float(2 * st.norm.sf(abs(zv)))


def est(v, s):
    return dict(est=float(v), se=float(s), z=float(v / s) if s > 0 else None)


# ------------------------------------------------------------------ the test on one sample
TH = 100.0          # theta reported x100


def statistics(b, V):
    """All statistics from b = (kC, kB, kdin, kdinB, tC, tB, tdin, tdinB) and its covariance."""
    k = 4
    b = b.copy(); V = V.copy()
    sc = np.r_[np.ones(k), TH * np.ones(k)]
    b = b * sc; V = V * np.outer(sc, sc)
    iC, iB, jC, jB = 0, 1, k, k + 1
    kC, kB, tC, tB = b[iC], b[iB], b[jC], b[jB]
    kO, tO = kC + kB, tC + tB

    def lc(w):
        g = np.zeros(2 * k)
        for i_, v_ in w.items():
            g[i_] = v_
        return float(g @ b), float(np.sqrt(g @ V @ g))

    def dm(val, g):
        return dict(est=float(val), se=float(np.sqrt(g @ V @ g)), z=float(val / np.sqrt(g @ V @ g)),
                    p=pval(val / np.sqrt(g @ V @ g)))

    o = dict(kappa_closed=est(*lc({iC: 1})), kappa_incr=est(*lc({iB: 1})), kappa_open=est(*lc({iC: 1, iB: 1})),
             theta_closed=est(*lc({jC: 1})), theta_incr=est(*lc({jB: 1})), theta_open=est(*lc({jC: 1, jB: 1})),
             placebo_in=dict(kappa_closed=est(*lc({2: 1})), kappa_incr=est(*lc({3: 1})),
                             theta_closed=est(*lc({k + 2: 1})), theta_incr=est(*lc({k + 3: 1}))))
    o["kappa_incr"]["p"] = pval(o["kappa_incr"]["z"]); o["theta_incr"]["p"] = pval(o["theta_incr"]["z"])
    o["ratio_price"] = float(kO / kC) if kC != 0 else None
    o["ratio_flow"] = float(tO / tC) if tC != 0 else None
    zk, zt = o["kappa_closed"]["z"], o["theta_closed"]["z"]
    zko, zto = o["kappa_open"]["z"], o["theta_open"]["z"]
    gk = np.zeros(2 * k); gk[iC] = 1 / kO - 1 / kC; gk[iB] = 1 / kO
    gt = np.zeros(2 * k); gt[jC] = 1 / tO - 1 / tC; gt[jB] = 1 / tO
    defk, deft = (kC > 0 and kO > 0), (tC > 0 and tO > 0)
    o["log_ratio_price"] = dict(est=float(np.log(kO / kC)), se=float(np.sqrt(gk @ V @ gk)), defined=True,
                                unstable=bool(zk < 1.96)) if defk else dict(est=None, se=None, defined=False,
                                                                            unstable=True)
    o["log_ratio_flow"] = dict(est=float(np.log(tO / tC)), se=float(np.sqrt(gt @ V @ gt)), defined=True,
                               unstable=bool(zt < 1.96)) if deft else dict(est=None, se=None, defined=False,
                                                                           unstable=True)
    if defk and deft:
        dl = np.log(kO / kC) - np.log(tO / tC)
        r = dm(dl, gk - gt)
        r.update(ci95=[r["est"] - 1.96 * r["se"], r["est"] + 1.96 * r["se"]], defined=True)
        o["logdiff"] = r
    else:
        o["logdiff"] = dict(est=None, se=None, z=None, p=None, ci95=None, defined=False)
    # (5) levels test h = theta_O*kappa_C - kappa_O*theta_C = tB*kC - kB*tC
    gh = np.zeros(2 * k); gh[iC] = tB; gh[iB] = -tC; gh[jC] = -kB; gh[jB] = kC
    h = tB * kC - kB * tC
    o["levels_h"] = dm(h, gh)
    o["levels_h"]["units"] = "speed per 5 min x (SD of flow per 100 bp)"
    # (6) implied theta_C under the price ratio, the gap h/kO, and the difference of closed/open ratios
    gi = np.zeros(2 * k)              # theta_C* = tO*kC/kO
    gi[jC] = gi[jB] = kC / kO; gi[iC] = tO * (1 / kO - kC / kO ** 2); gi[iB] = -tO * kC / kO ** 2
    o["theta_closed_implied"] = dict(**dm(tO * kC / kO, gi), unstable=bool(zko < 1.96))
    o["gap_implied_minus_actual"] = dict(**dm(tO * kC / kO - tC, gi - np.eye(2 * k)[jC]), unstable=bool(zko < 1.96))
    gr = np.zeros(2 * k)              # kC/kO - tC/tO
    gr[iC] = 1 / kO - kC / kO ** 2; gr[iB] = -kC / kO ** 2
    gr[jC] = -(1 / tO - tC / tO ** 2); gr[jB] = tC / tO ** 2
    o["closed_open_ratio_diff"] = dict(**dm(kC / kO - tC / tO, gr), closed_open_price=float(kC / kO),
                                       closed_open_flow=float(tC / tO), unstable=bool(zko < 1.96 or zto < 1.96))
    return o


def run(d, beta=None):
    """Joint price/flow estimation on sample d; beta=None -> re-estimated within the sample (primary)."""
    d = d.copy()
    bP = beta_col2(d) if beta is None else beta
    d["R"] = d.u1 + bP * d.u2
    pr = ols(d, "R", PCOLS); fw = ols(d, "A_L", PCOLS)
    b, V = joint([pr, fw])
    o = statistics(b, V)
    o.update(beta=float(bP), n=pr["n"], n_days=pr["n_days"],
             n_above_closed=int(((d.x > d.u_s) & (d.bh_eff == 0)).sum()),
             n_above_open=int(((d.x > d.u_s) & (d.bh_eff == 1)).sum()),
             n_below_closed=int(((d.x < d.l_s) & (d.bh_eff == 0)).sum()),
             n_below_open=int(((d.x < d.l_s) & (d.bh_eff == 1)).sum()))
    return o, b


# ------------------------------------------------------------------ day bootstrap through per-day sufficient statistics
def day_stats(d):
    X = d[PCOLS].values.astype(float)
    U = d[["u1", "u2", "A_L"]].values.astype(float)
    W = d[["q", "up", "dn"]].values.astype(float)
    yv = d.y.values.astype(float)[:, None]
    n = X.shape[0]
    parts = [np.ones((n, 1)), X, U, W, yv,
             (X[:, :, None] * X[:, None, :]).reshape(n, -1), (X[:, :, None] * U[:, None, :]).reshape(n, -1),
             (W[:, :, None] * W[:, None, :]).reshape(n, -1), W * yv]
    M = np.hstack(parts)
    codes, uniq = pd.factorize(d.day)
    D = np.zeros((len(uniq), M.shape[1])); np.add.at(D, codes, M)
    mon = d.groupby(codes).month.first().sort_index().values
    return D, mon


SL = dict(n=slice(0, 1), X=slice(1, 5), U=slice(5, 8), W=slice(8, 11), y=slice(11, 12), XX=slice(12, 28),
          XU=slice(28, 40), WW=slice(40, 49), Wy=slice(49, 52))


def boot_estimates(D, mon, C, beta=None):
    """C: (B x days) multiplicities. Returns (B x 8) coefficient draws [price 4, flow 4] and beta draws."""
    Bn = C.shape[0]
    XXw = np.zeros((Bn, 4, 4)); XUw = np.zeros((Bn, 4, 3)); WWw = np.zeros((Bn, 3, 3)); Wyw = np.zeros((Bn, 3))
    for m_ in np.unique(mon):
        ix = np.where(mon == m_)[0]
        T = C[:, ix] @ D[ix]                       # B x 52
        n = T[:, SL["n"]][:, 0]
        sX, sU, sW, sy = T[:, SL["X"]], T[:, SL["U"]], T[:, SL["W"]], T[:, SL["y"]]
        XXw += T[:, SL["XX"]].reshape(Bn, 4, 4) - sX[:, :, None] * sX[:, None, :] / n[:, None, None]
        XUw += T[:, SL["XU"]].reshape(Bn, 4, 3) - sX[:, :, None] * sU[:, None, :] / n[:, None, None]
        WWw += T[:, SL["WW"]].reshape(Bn, 3, 3) - sW[:, :, None] * sW[:, None, :] / n[:, None, None]
        Wyw += T[:, SL["Wy"]] - sW * sy / n[:, None]
    bet = np.linalg.solve(WWw, Wyw[:, :, None])[:, 0, 0] if beta is None else np.full(Bn, beta)
    rhs_p = XUw[:, :, 0] + bet[:, None] * XUw[:, :, 1]
    bp = np.linalg.solve(XXw, rhs_p[:, :, None])[:, :, 0]
    bf = np.linalg.solve(XXw, XUw[:, :, 2:3])[:, :, 0]
    return np.hstack([bp, bf]), bet


def bootstrap(d, beta, b_hat, seed_key):
    D, mon = day_stats(d)
    # identity draw reproduces the point estimates
    b1, _ = boot_estimates(D, mon, np.ones((1, len(mon))), beta)
    assert np.allclose(b1[0], b_hat, rtol=1e-7, atol=1e-12), (b1[0], b_hat)
    rng = np.random.default_rng([SEED, seed_key])
    C = np.zeros((B_BOOT, len(mon)))
    for m_ in np.unique(mon):
        ix = np.where(mon == m_)[0]
        draws = rng.integers(0, len(ix), size=(B_BOOT, len(ix)))
        for r in range(B_BOOT):
            C[r, ix] = np.bincount(draws[r], minlength=len(ix))
    bb, bet = boot_estimates(D, mon, C, beta)
    kC, kB, tC, tB = bb[:, 0], bb[:, 1], bb[:, 4] * TH, bb[:, 5] * TH
    kO, tO = kC + kB, tC + tB
    h = tB * kC - kB * tC
    dfn = (kC > 0) & (kO > 0) & (tC > 0) & (tO > 0)
    dl = np.where(dfn, np.log(np.where(dfn, kO / kC, 1)) - np.log(np.where(dfn, tO / tC, 1)), np.nan)
    o = dict(B=int(B_BOOT), h_ci95=[float(np.percentile(h, 2.5)), float(np.percentile(h, 97.5))],
             h_p_two_sided=float(min(1.0, 2 * min((h <= 0).mean(), (h >= 0).mean()))),
             share_kappa_closed_pos=float((kC > 0).mean()), share_theta_closed_pos=float((tC > 0).mean()),
             share_kappa_open_pos=float((kO > 0).mean()), share_theta_open_pos=float((tO > 0).mean()),
             share_logdiff_defined=float(dfn.mean()),
             logdiff_ci95_defined=[float(np.nanpercentile(dl, 2.5)), float(np.nanpercentile(dl, 97.5))]
             if dfn.sum() >= 20 else None,
             beta_mean=float(bet.mean()),
             h_mean=float(h.mean()), h_median=float(np.median(h)), h_sd=float(h.std(ddof=1)))
    return o


# ------------------------------------------------------------------ periods
PERIODS = {"full": a.index >= a.index.min(), "2024": (a.month.str[:4] == "2024").values,
           "2025": (a.month.str[:4] == "2025").values, "2026": (a.month.str[:4] == "2026").values,
           "pre": (a.P == 0).values, "post": (a.P == 1).values}
OUT = dict(name=NAME, T0=str(T0), beta_published=float(BETA_PUB), B_boot=B_BOOT, seed=SEED,
           theta_scale="x100 (SD of large-trade flow per 100 bp)",
           band_treatment=dict(
               years="published month x state bands; every cell lies inside one year, so these equal bands "
                     "re-profiled within the period",
               prepost_primary="published bands (November 2025 cells profiled on the whole month); the baseline "
                               "does not re-profile bands on sub-samples",
               prepost_sensitivity="November 2025 state cells and monthly band re-profiled separately on 1-12 Nov "
                                   "and 13-30 Nov"))
# check: no state-band cell spans two calendar years (month cells), and the November cells straddle T0
cells = a.groupby(["month", "state"]).agg(y0=("month", lambda s: s.str[:4].nunique()))
assert (cells.y0 == 1).all()
OUT["n_cells_straddling_T0"] = int(a.groupby(["month", "state"]).P.nunique().gt(1).sum())

RES = {"primary": {}, "beta_fixed": {}}
BH = {}
for key, (p, msk) in enumerate(PERIODS.items()):
    d = a[msk]
    RES["primary"][p], b_hat = run(d, None)
    RES["primary"][p]["boot"] = bootstrap(d, None, b_hat, key)
    RES["beta_fixed"][p], b_hat1 = run(d, BETA_PUB)
    RES["beta_fixed"][p]["boot"] = bootstrap(d, BETA_PUB, b_hat1, 100 + key)
    r = RES["primary"][p]
    print(p, "beta", round(r["beta"], 3), "kC", round(r["kappa_closed"]["est"], 5), "kO", round(r["kappa_open"]["est"], 5),
          "tC", round(r["theta_closed"]["est"], 4), "tO", round(r["theta_open"]["est"], 4),
          "dl", r["logdiff"]["est"], r["logdiff"]["se"], "h z", round(r["levels_h"]["z"], 2),
          f"{time.time() - t_start:.0f}s", flush=True)

# replication of the published full-sample test (empirics_mech.py)
pub = json.load(open(os.path.join(TAB, "emp_mech.json")))["cross"]
F = RES["primary"]["full"]
OUT["check_replication"] = dict(
    logdiff=[F["logdiff"]["est"], pub["logdiff"]], se_logdiff=[F["logdiff"]["se"], pub["se_logdiff"]],
    ratio_price=[F["ratio_price"], pub["ratio_price"]], ratio_flow=[F["ratio_flow"], pub["ratio_flow"]],
    levels_z=[F["levels_h"]["z"], -pub["z"]], beta=[F["beta"], BETA_PUB])
for k_, (v1, v2) in OUT["check_replication"].items():
    assert abs(v1 - v2) < 1e-10 * max(1, abs(v2)), (k_, v1, v2)
assert abs(RES["beta_fixed"]["full"]["logdiff"]["est"] - pub["logdiff"]) < 1e-10

# ------------------------------------------------------------------ sensitivity: November 2025 re-profiled on each side of T0
ab = a.copy()
nov = (ab.month == "2025-11").values
bands_nov = {}
for p_ in [0.0, 1.0]:
    side = "pre" if p_ == 0 else "post"
    m_side = nov & (ab.P.values == p_)
    dd = ab[m_side]
    l_, u_ = profile(dd.y.to_numpy(float), dd.x.to_numpy(float), dd.q.to_numpy(float))
    ab.loc[m_side, "l"] = l_; ab.loc[m_side, "u"] = u_
    bands_nov[f"monthly_{side}"] = dict(l=l_, u=u_, n=int(m_side.sum()))
    for stt in ["closed", "open"]:
        mm = m_side & (ab.state.values == stt)
        assert mm.sum() >= 300
        dd = ab[mm]
        l_, u_ = profile(dd.y.to_numpy(float), dd.x.to_numpy(float), dd.q.to_numpy(float))
        ab.loc[mm, "l_s"] = l_; ab.loc[mm, "u_s"] = u_
        bands_nov[f"state_{stt}_{side}"] = dict(l=l_, u=u_, n=int(mm.sum()))
for stt in ["closed", "open"]:
    mm = nov & (a.state.values == stt)
    bands_nov[f"state_{stt}_published"] = dict(l=float(a.l_s[mm].iloc[0]), u=float(a.u_s[mm].iloc[0]))
bands_nov["monthly_published"] = dict(l=float(a.l[nov].iloc[0]), u=float(a.u[nov].iloc[0]))
# check the profile function reproduces the published November cells on the whole month
for stt in ["closed", "open"]:
    mm = nov & (a.state.values == stt)
    dd = a[mm]
    assert profile(dd.y.to_numpy(float), dd.x.to_numpy(float), dd.q.to_numpy(float)) == \
        (bands_nov[f"state_{stt}_published"]["l"], bands_nov[f"state_{stt}_published"]["u"])
ab = design(ab)
RES["nov_reprofiled"] = {}
RES["nov_reprofiled_beta_fixed"] = {}
for key, p in enumerate(["pre", "post"]):
    d = ab[PERIODS[p]]
    RES["nov_reprofiled"][p], b_hat = run(d, None)
    RES["nov_reprofiled"][p]["boot"] = bootstrap(d, None, b_hat, 200 + key)
    RES["nov_reprofiled_beta_fixed"][p], _ = run(d, BETA_PUB)
OUT["bands_nov2025"] = bands_nov


# ------------------------------------------------------------------ joint Wald tests and classifications
def wald(var, periods):
    zs = [RES[var][p]["levels_h"]["z"] for p in periods]
    c2 = float(np.sum(np.square(zs)))
    return dict(periods=periods, chi2=c2, df=len(periods), p=float(st.chi2.sf(c2, len(periods))))


OUT["joint_levels_test"] = {var: dict(years=wald(var, ["2024", "2025", "2026"]), prepost=wald(var, ["pre", "post"]))
                            for var in ["primary", "beta_fixed"]}
OUT["joint_levels_test"]["nov_reprofiled"] = dict(prepost=wald("nov_reprofiled", ["pre", "post"]))


def verdict(r):
    rej_h = r["levels_h"]["p"] < 0.05
    ld = r["logdiff"]
    if ld["defined"] and ((ld["p"] < 0.05) != rej_h):
        return "mixed"
    if rej_h:
        return "equality rejected"
    inf = (ld["defined"] and not r["log_ratio_price"]["unstable"] and not r["log_ratio_flow"]["unstable"]
           and ld["ci95"][0] > -np.log(2) and ld["ci95"][1] < np.log(2))
    return "consistent with equality, informative" if inf else "not rejected but uninformative"


OUT["reading"] = {var: {p: verdict(r) for p, r in RES[var].items()} for var in RES}
OUT["results"] = RES
OUT["runtime_s"] = round(time.time() - t_start, 1)
os.makedirs(OTAB, exist_ok=True)
json.dump(OUT, open(os.path.join(OTAB, f"{NAME}.json"), "w"), indent=1)
print(json.dumps(OUT["reading"], indent=1))
print(json.dumps(OUT["joint_levels_test"], indent=1))


# ------------------------------------------------------------------ LaTeX table
def f(v, nd):
    return "--" if v is None else f"{v:.{nd}f}"


def pv(v):
    """Delta-method p-value: avoid printing 0.000."""
    return "--" if v is None else ("$<0.001$" if v < 0.0005 else f"{v:.3f}")


def pb(v):
    """Bootstrap p-value: with no draw on one side of 0 the p-value is only bounded by 2/(B+1)."""
    return "--" if v is None else (f"$<{2 / (B_BOOT + 1):.3f}$" if v == 0 else f"{v:.3f}")


def cell(e, nd, flag=False):
    if e is None or e.get("est") is None:
        return "n.d."
    return f"${e['est']:.{nd}f}$" + ("$^\\dagger$" if flag else "")


def se_cell(e, nd):
    if e is None or e.get("se") is None:
        return ""
    return f"$({e['se']:.{nd}f})$"


COLS = ["full", "2024", "2025", "2026", "pre", "post"]
HEAD = ["Full", "2024", "2025", "2026", "Before", "From"]
HEAD2 = ["sample", "", "", "Jan--Aug", "13 Nov 2025", "13 Nov 2025"]
P_ = RES["primary"]; FX = RES["beta_fixed"]; NR = RES["nov_reprofiled"]
L = []
L.append("\\begin{tabular}{l" + "c" * len(COLS) + "}\n\\toprule\n")
L.append(" & " + " & ".join(HEAD) + " \\\\\n & " + " & ".join(HEAD2) + " \\\\\n")
L.append(" & " + " & ".join(f"({i + 1})" for i in range(len(COLS))) + " \\\\\n\\midrule\n")


def row(lab, getter, nd, flagger=None):
    L.append(lab + " & " + " & ".join(cell(getter(P_[p]), nd, flagger(P_[p]) if flagger else False) for p in COLS)
             + " \\\\\n")
    L.append(" & " + " & ".join(se_cell(getter(P_[p]), nd) for p in COLS) + " \\\\\n")


L.append("\\multicolumn{7}{l}{\\emph{A. Price-side speed $\\kappa$ (Table~\\ref{tab:emp_flow}, col.\\ 1)}} \\\\\n")
row("Banks closed", lambda r: r["kappa_closed"], 4)
row("Banks open", lambda r: r["kappa_open"], 4)
L.append("\\multicolumn{7}{l}{\\emph{B. Large-trade response $\\theta\\times100$ (col.\\ 2)}} \\\\\n")
row("Banks closed", lambda r: r["theta_closed"], 3)
row("Banks open", lambda r: r["theta_open"], 3)
L.append("\\midrule\n\\multicolumn{7}{l}{\\emph{C. Open/closed ratios}} \\\\\n")
L.append("Price, $\\kappa_{open}/\\kappa_{closed}$ & " + " & ".join(f(P_[p]["ratio_price"], 2) for p in COLS) + " \\\\\n")
L.append("Large trades, $\\theta_{open}/\\theta_{closed}$ & " + " & ".join(f(P_[p]["ratio_flow"], 2) for p in COLS) + " \\\\\n")
row("$\\log$ price ratio", lambda r: r["log_ratio_price"], 2, lambda r: r["log_ratio_price"]["unstable"])
row("$\\log$ large-trade ratio", lambda r: r["log_ratio_flow"], 2, lambda r: r["log_ratio_flow"]["unstable"])
row("Difference of log ratios", lambda r: r["logdiff"], 2,
    lambda r: r["logdiff"]["defined"] and (r["log_ratio_price"]["unstable"] or r["log_ratio_flow"]["unstable"]))
L.append("\\midrule\n\\multicolumn{7}{l}{\\emph{D. Test in levels, $h=\\theta_{open}\\kappa_{closed}-\\kappa_{open}\\theta_{closed}$}} \\\\\n")
L.append("$h\\times1000$ & " + " & ".join(f"${1000 * P_[p]['levels_h']['est']:.3f}$" for p in COLS) + " \\\\\n")
L.append(" & " + " & ".join(f"$({1000 * P_[p]['levels_h']['se']:.3f})$" for p in COLS) + " \\\\\n")
L.append("$p$-value, delta method & " + " & ".join(pv(P_[p]["levels_h"]["p"]) for p in COLS) + " \\\\\n")
L.append("$p$-value, day bootstrap & " + " & ".join(pb(P_[p]["boot"]["h_p_two_sided"]) for p in COLS) + " \\\\\n")
row("Implied $\\theta_{closed}=\\theta_{open}\\kappa_{closed}/\\kappa_{open}$", lambda r: r["theta_closed_implied"], 3,
    lambda r: r["theta_closed_implied"]["unstable"])
L.append("Share of bootstrap draws with $\\theta_{closed}>0$ & " +
         " & ".join(f(P_[p]["boot"]["share_theta_closed_pos"], 2) for p in COLS) + " \\\\\n")
L.append("\\midrule\n\\multicolumn{7}{l}{\\emph{E. Alternative constructions}} \\\\\n")


def alt_row(lab, src, getter, nd, cols=COLS):
    L.append(lab + " & " + " & ".join((cell(getter(src[p]), nd) if p in src else "") for p in COLS) + " \\\\\n")
    L.append(" & " + " & ".join((se_cell(getter(src[p]), nd) if p in src else "") for p in COLS) + " \\\\\n")


alt_row("$\\beta$ fixed at full sample: difference of log ratios", FX, lambda r: r["logdiff"], 2)
L.append("\\quad $p$-value of levels test & " + " & ".join(pv(FX[p]["levels_h"]["p"]) for p in COLS) + " \\\\\n")
alt_row("Nov.\\ 2025 bands split at 13 Nov: difference of log ratios", NR, lambda r: r["logdiff"], 2)
L.append("\\quad $p$-value of levels test & " + " & ".join((pv(NR[p]["levels_h"]["p"]) if p in NR else "") for p in COLS)
         + " \\\\\n")
L.append("\\midrule\n$\\hat\\beta$ & " + " & ".join(f(P_[p]["beta"], 3) for p in COLS) + " \\\\\n")
L.append("$N$ & " + " & ".join(f"{P_[p]['n']:,}" for p in COLS) + " \\\\\n")
L.append("Days & " + " & ".join(f"{P_[p]['n_days']:,}" for p in COLS) + " \\\\\n\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(OTAB, f"{NAME}.tex"), "w") as fh:
    fh.write("".join(L))
print("".join(L))
print(f"done {time.time() - t_start:.0f}s")
