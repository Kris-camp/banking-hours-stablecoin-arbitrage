"""rob_dayfe_t1: T+1 settlement of the interbank benchmark, day-varying carry, and boundary timing.

Interbank USD/TRY spot settles T+1 while USDT settles at once. At 40-50% lira rates
tom-next carry is 11-14 bp per calendar day, and the number of carry days embedded in the T+1 rate
changes around value-date shifts (a Friday trade settles on Monday, a trade before a holiday settles
after it). The benchmark therefore carries a day-varying carry component. In addition, business
hours are coded 09:00-17:00 Istanbul, but EFT may run 08:30-17:30, so boundary timing could drive the
business-hours increment.

Timing fact used throughout. The FX value date rolls at 17:00 New York (21:00 or 22:00 UTC), inside
the excluded 20:30-22:30 UTC window, and Delta b is never computed across that gap. On the estimation
sample the Istanbul calendar date (UTC+3) therefore coincides with the FX trade date: the value date,
and so the embedded carry, is constant within an Istanbul date. "Day" below means Istanbul date.

DESIGN
Targets. A = emp_main column (4): Delta b on OFI, kappa+ term, kappa- term, all interacted with BH,
monthly bands. B = emp_state column (1): same with month x state bands. Reported for each: kappa+ when
banks are closed, the business-hours increment in kappa+ (headline), the open/closed ratio of kappa+
with a delta-method s.e. of its log, and the same for kappa- and beta.
Standard errors. Day-clustered with the paper's finite-sample correction. Baselines are reproduced
exactly (asserted) with the paper's UTC-date clusters and also shown with Istanbul-date clusters; all
new specifications cluster by Istanbul date, so that day fixed effects are nested in the clusters.

 1. Day fixed effects. Month FE replaced by Istanbul-date FE (A) and by Istanbul-date x banking-state
    FE (B, the analogue of month x state FE; plain date FE also reported). Bands unchanged (published
    monthly / month x state bands). Alternative: UTC-date FE clustered by UTC date.
    Within-day demeaning of a persistent premium creates a Nickell-type bias, of order 2/T with
    T ~ 260 five-minute intervals per day, which is the size of kappa itself. It is measured by
    simulation: under the month-FE estimates of the same specification (no day
    effects), the premium is regenerated segment by segment (each contiguous run of five-minute
    intervals starts at its observed lagged premium), with realized order flow, bands, BH, and the
    fitted residuals multiplied by one Rademacher sign per Istanbul date; R = 200 paths, seed 20261022.
    Each path is estimated with month FE (validation: bias should be ~0) and with each day-FE variant.
    Bias = mean simulated estimate - true value. Reported: raw day-FE estimate, bias, bias-corrected
    estimate (raw - bias) with the raw analytic s.e.; ratios use the corrected speeds.
 2. Value-date-shift days excluded. Settlement calendar: a value date must be a weekday that is
    neither a Turkish weekday public holiday (HOL, copied from empirics_mech.py) nor a U.S. federal
    (Fedwire) holiday. Carry days n(d) = calendar days from trade date d to its T+1 value date.
    Retained days: Tuesday-Thursday such that none of d-1, d, d+1 is a Turkish holiday, a U.S.
    federal holiday or an arife half-day. This drops Fridays (n = 3), Mondays, holidays and the days
    before and after them, and implies n(d-1) = n(d) = 1 (asserted). Month (A) or month x state (B) FE.
    Reported with (a) bands re-profiled on the retained days only (primary: shifted-carry days could
    contaminate the band edges) and (b) the published bands. Then 2 + 1 combined: retained days,
    re-profiled bands, day FE, with its own simulated bias.
    Complementary: drop only the dates whose T+1 rate embeds more
    than one carry day, n(d) > 1 (Fridays and pre-holiday dates), re-profiled bands.
 3. Business-hour boundaries. (i) Primary: drop 09:00-10:00 and 16:00-17:00 Istanbul; open = 10-16,
    closed = 17-09 unchanged. (ii) Wide: drop 08:00-10:00 and 16:00-18:00; open = 10-16, closed =
    18-08. (iii) Re-coded: open = 08:30-17:30, nothing dropped. (i) and (ii) with published and
    re-profiled bands; (iii) with re-profiled bands only for B (published state bands are defined on
    the 09-17 split). Month (A) or month x state (B) FE.
Band profiling copies empirics_main.profile / empirics_state.profile (1 bp grid, 5% trimming,
model._ssr_grid; >= 500 obs per month, >= 300 per month x state); on the full sample it must
reproduce the published bands (asserted).
Diagnostic of the premise. If the benchmark embeds n(d) days of carry, the premium level is lower by
the daily carry for each extra carry day: x on (n(d) - 1) with month FE, Istanbul-date clusters, and,
separately, on a Friday dummy; and the jump in the premium across each value-date roll (last premium
of a date to the first of the next sample date) on the change in carry days n(d) - n(next), which
should equal the daily carry if the premise holds. Reference value: the CBRT-Fed policy differential / 365
(CBRT one-week repo rate and Fed funds target midpoint by decision date, 2024-2025).
ADDITIONAL CHECKS (rows with keys starting ph_ in the JSON). Excluding value-date-shift days raises the
closed-hours speed and lowers the open/closed ratio; P1-P7 examine why and how the bias correction matters:
 P1. Which excluded days drive it: drop only Mondays and first settlement dates after a holiday
     (dates whose previous weekday is not a settlement date), re-profiled bands.
 P2. Col (4) estimated separately by date type with the published bands and month (A) or month x state
     (B) FE: clean Tue-Thu, Mondays, Fridays, other (holidays, arife and their neighbours).
     The carry premise predicts anomalies on dates with n(d) > 1 (Fridays), not on Mondays (n = 1).
 P3. Premium level (x on month FE, date clusters) on Monday and Friday dummies split by banking state.
 P4. Half-panel jackknife correction of every day-FE estimate (Dhaene and Jochmans 2015):
     2*theta(date FE) - theta(half-date FE), each date (x state) group split at the midpoint of its
     observations; clustered s.e. from the combined influence functions. A second, DGP-free check on the simulated bias; its own bias is also
     measured in the simulation.
 P5. Mean five-minute premium drift by date type and banking state.
 P6. The wild simulation shows a bias of the month-FE estimator itself (see the validation note below).
     Because a +1 sign replays the observed day exactly, that bias mixes finite-sample bias with
     misspecification. Check: the same simulation with iid Gaussian shocks (s.d. of the residual
     by month x state), R = R_SIM/2, for the full and the excluded-days samples, and the date-FE
     estimates corrected by this model-based bias.
 P7. Like-for-like corrections: (a) date-FE estimate minus the bias that date FE add relative to
     month FE on the same simulated paths; (b) month-FE estimates corrected by their own simulated
     bias, the comparison point for the absolute (wild) correction of item 1.
Validation of the wild simulation. The month-FE estimator, which should be close to unbiased, shows a
     simulated bias of +0.0024 (A), +0.0026 (B) and +0.0081 (A, excluded dates) in the increment, against
     MC s.e. of 0.0001-0.0004. The month-FE residuals have day means correlated with the regressor (corr of
     day-mean residual with day-mean up = -0.34), and flipping every sign moves the month-FE increment from
     0.0079 to 0.0130, so the wild "bias" mixes Nickell bias with the removal of day effects correlated with
     the premium. Neither P6 nor P7 resolves this, so the LaTeX table lists every correction side by side,
     together with the combined rows (2 + 1).
Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, tables/emp_numbers.json, tables/emp_state.json.
Outputs: tables/rob_dayfe_t1.json, tables/rob_dayfe_t1.tex.
Env: R_SIM (200), SEED (20261022).
"""
import os, sys, json, time
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import _ssr_grid

RAW, TAB = os.path.join(HERE, "..", "data_raw"), os.path.join(HERE, "..", "tables")
R_SIM = int(os.environ.get("R_SIM", 200))
SEED = int(os.environ.get("SEED", 20261022))
T0 = time.time()
OUT = dict(design=dict(r_sim=R_SIM, seed=SEED))

# ------------------------------------------------------------------ data
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
a["day"] = a.index.date                                   # UTC date: the paper's clusters
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
assert sb.index.equals(a.index)
a["l_s"], a["u_s"] = sb.l_s.values, sb.u_s.values
ist = a.index.tz_convert("Europe/Istanbul")
a["iday"] = ist.date                                      # Istanbul date = FX trade date on this sample
a["imin"] = ist.hour * 60 + ist.minute
a["idow"] = ist.dayofweek
assert a.idow.max() <= 4
# FX value date rolls inside the excluded window: no observation between 20:30 and 22:30 UTC
umin = a.index.hour * 60 + a.index.minute
assert not ((umin >= 20 * 60 + 30) & (umin < 22 * 60 + 30)).any()
# segments: x_t = x_{t-1} + y_{t-1} within contiguous runs
gap = a.index.to_series().diff() != pd.Timedelta("5min")
a["seg"] = gap.values
cont = ~a.seg.values
assert np.allclose(a.x.values[1:][cont[1:]], (a.x.values + a.y.values)[:-1][cont[1:]], atol=1e-6)

# ------------------------------------------------------------------ calendars
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]                                    # copied from empirics_mech.py
ARIFE = ["2024-04-09", "2024-10-28", "2025-06-05", "2025-10-28", "2026-03-19", "2026-05-26"]   # rob_time_of_day.py
US_FED = ["2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
          "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25",
          "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
          "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25",
          "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07"]
D = lambda L: set(pd.to_datetime(L).date)
HOL, ARIFE, US_FED = D(HOL), D(ARIFE), D(US_FED)
cal = pd.date_range("2023-12-01", "2026-10-31", freq="D").date
settle = {d: (d.weekday() < 5) and (d not in HOL) and (d not in US_FED) for d in cal}
carry = {}
for i, d in enumerate(cal[:-10]):
    j = i + 1
    while not settle[cal[j]]:
        j += 1
    carry[d] = (cal[j] - d).days
one = pd.Timedelta(days=1)
special = HOL | US_FED | ARIFE
days = sorted(set(a.iday))
keep = {}
for d in days:
    dm, dp = (pd.Timestamp(d) - one).date(), (pd.Timestamp(d) + one).date()
    keep[d] = (d.weekday() in (1, 2, 3)) and not ({dm, d, dp} & special)
    if keep[d]:
        assert carry[d] == 1 and carry[dm] == 1
a["ncarry"] = a.iday.map(carry).astype(float)
a["clean"] = a.iday.map(keep).astype(bool)
dd = pd.DataFrame({"d": days})
dd["n"] = dd.d.map(carry); dd["keep"] = dd.d.map(keep); dd["dow"] = [d.weekday() for d in days]
OUT["calendar"] = dict(n_days=len(days), n_days_retained=int(dd.keep.sum()), n_obs_retained=int(a.clean.sum()),
                       carry_days_dist={str(k): int(v) for k, v in dd.n.value_counts().sort_index().items()},
                       n_days_multi_carry=int((dd.n > 1).sum()),
                       n_multi_carry_not_friday=int(((dd.n > 1) & (dd.dow != 4)).sum()))


# ------------------------------------------------------------------ estimator (empirics_main.ols_fe, fe and cluster as codes)
def ols(Y, X, fe, cl, names):
    Y = Y.astype(float).copy(); X = X.astype(float).copy()
    if fe is not None:
        cnt = np.bincount(fe)
        Y -= (np.bincount(fe, Y) / cnt)[fe]
        X -= np.column_stack([np.bincount(fe, X[:, j]) / cnt for j in range(X.shape[1])])[fe]
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    G = np.zeros((cl.max() + 1, X.shape[1])); np.add.at(G, cl, X * e[:, None])
    ng, n, k = cl.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=dict(zip(names, b)), se=dict(zip(names, np.sqrt(np.diag(V)))), V=V, names=list(names), n=int(n),
                ncl=int(ng), e=e, IF=G @ XtXi, adj=ng / (ng - 1) * (n - 1) / (n - k), bvec=b)


def jackknife(r_full, r_half):
    """half-panel jackknife 2*full - half (Dhaene and Jochmans 2015); clustered V from combined influence."""
    b = 2 * r_full["bvec"] - r_half["bvec"]
    IF = 2 * r_full["IF"] - r_half["IF"]
    V = IF.T @ IF * r_full["adj"]
    nm = r_full["names"]
    return dict(b=dict(zip(nm, b)), se=dict(zip(nm, np.sqrt(np.diag(V)))), V=V, names=nm, n=r_full["n"], ncl=r_full["ncl"])


NAMES = ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"]


def design(x, q, l, u, bh):
    up = -np.maximum(x - u, 0); dn = np.maximum(l - x, 0)
    return np.column_stack([q, up, dn, q * bh, up * bh, dn * bh])


def codes(s):
    return pd.factorize(s)[0]


def fe_key(d, fe):
    if fe is None: return None
    if fe == "month": return codes(d.month.values)
    if fe == "ms": return codes(d.month.values + "_" + d.st.values)
    if fe == "iday": return codes(d.iday.values)
    if fe == "iday_s": return codes(pd.Series(d.iday.astype(str).values) + "_" + d.st.values)
    if fe == "uday": return codes(d.day.values)
    if fe in ("iday_half", "iday_s_half"):           # split each date (x state) group into its first and second half in time
        g = codes(d.iday.values) if fe == "iday_half" else fe_key(d, "iday_s")
        pos = pd.Series(np.arange(len(d))).groupby(g).cumcount().values
        cnt = np.bincount(g)[g]
        return codes(g * 2 + (pos >= cnt / 2))
    raise ValueError(fe)


def fit(d, fe, cl="iday"):
    X = design(d.x.values, d.q.values, d.l.values, d.u.values, d.bh.values)
    return ols(d.y.values, X, fe_key(d, fe), codes(d[cl].values), NAMES)


def summarize(r, bias=None):
    b = dict(r["b"]); se = r["se"]; V = r["V"]; ix = {k: i for i, k in enumerate(r["names"])}
    if bias is not None:
        b = {k: b[k] - bias[k] for k in b}
    out = dict(n=r["n"], n_clusters=r["ncl"], coef={k: float(b[k]) for k in b}, se={k: float(se[k]) for k in se})
    for side, k in [("kp", "up"), ("km", "dn"), ("beta", "q")]:
        c, inc = b[k], b[k + "_bh"]
        o = c + inc
        rec = dict(closed=[float(c), float(se[k])], inc=[float(inc), float(se[k + "_bh"])], open=float(o))
        # s.e. of open speed
        g = np.zeros(len(ix)); g[ix[k]] = 1; g[ix[k + "_bh"]] = 1
        rec["open_se"] = float(np.sqrt(g @ V @ g))
        if c > 0 and o > 0:
            g = np.zeros(len(ix)); g[ix[k]] = 1 / o - 1 / c; g[ix[k + "_bh"]] = 1 / o
            rec["ratio"] = float(o / c); rec["log_ratio"] = [float(np.log(o / c)), float(np.sqrt(g @ V @ g))]
        else:
            rec["ratio"] = None; rec["log_ratio"] = None
        out[side] = rec
    return out


# ------------------------------------------------------------------ band profiling (copied from empirics_main / empirics_state)
def profile(d, trim=0.05, step=1.0):
    y, x, q = d.y.values, d.x.values, d.q.values
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]          # l, u


def bands_monthly(d):
    B = {m: profile(g) for m, g in d.groupby("month") if len(g) >= 500}
    d = d[d.month.isin(B)].copy()
    d["l"] = d.month.map(lambda m: B[m][0]); d["u"] = d.month.map(lambda m: B[m][1])
    return d, B


def bands_state(d):
    B = {k: profile(g) for k, g in d.groupby(["month", "st"]) if len(g) >= 300}
    key = list(zip(d.month, d.st))
    ok = np.array([k in B for k in key])
    d = d[ok].copy(); key = [k for k, o in zip(key, ok) if o]
    d["l"] = [B[k][0] for k in key]; d["u"] = [B[k][1] for k in key]
    return d, B


def setup(d, bh, bands):
    """bh: Series/array of the open indicator; bands: 'pubM', 'pubS', 'reM', 'reS'"""
    d = d.copy(); d["bh"] = np.asarray(bh, float); d["st"] = np.where(d.bh == 1, "open", "closed")
    if bands == "pubM":
        pass                                         # a5 columns l, u
    elif bands == "pubS":
        d["l"], d["u"] = d.l_s, d.u_s
    elif bands == "reM":
        d, _ = bands_monthly(d)
    elif bands == "reS":
        d, _ = bands_state(d)
    return d


# ------------------------------------------------------------------ Nickell-type bias by simulation
@njit(cache=True)
def _sim(seg, x0, drift0, bq, kp, km, l, u, e):
    n = seg.shape[0]; xs = np.empty(n); ys = np.empty(n); prev = 0.0
    for i in range(n):
        xb = x0[i] if seg[i] else prev
        yb = drift0[i] + bq[i] - kp[i] * max(xb - u[i], 0.0) + km[i] * max(l[i] - xb, 0.0) + e[i]
        xs[i] = xb; ys[i] = yb; prev = xb + yb
    return xs, ys


def sim_bias(d, fe_true, fe_list, R=R_SIM, seed=SEED, shock="wild"):
    """Simulate under the fe_true estimates of col (4) on sample d; estimate under each fe in fe_list."""
    seg = np.r_[True, d.index.to_series().diff().values[1:] != np.timedelta64(5, "m")]
    X = design(d.x.values, d.q.values, d.l.values, d.u.values, d.bh.values)
    f = fe_key(d, fe_true)
    r = ols(d.y.values, X, f, codes(d.iday.values), NAMES)
    th = np.array([r["b"][k] for k in NAMES])
    resid_raw = d.y.values - X @ th
    alpha = (np.bincount(f, resid_raw) / np.bincount(f))[f]
    e = resid_raw - alpha
    bh = d.bh.values
    bq = (th[0] + th[3] * bh) * d.q.values
    kp = th[1] + th[4] * bh; km = th[2] + th[5] * bh
    dcode = codes(d.iday.values)
    cell = fe_key(d, "ms")
    sd_cell = np.sqrt(np.bincount(cell, e ** 2) / np.bincount(cell))[cell]
    rng = np.random.default_rng(seed)
    halves = [fe + "_half" for fe in fe_list if fe in ("iday", "iday_s")]
    keys = {fe: fe_key(d, fe) for fe in [fe_true] + fe_list + halves}
    clc = codes(d.iday.values)
    draws = {fe: [] for fe in keys}
    for _ in range(R):
        if shock == "wild":
            sh = e * rng.choice([-1.0, 1.0], dcode.max() + 1)[dcode]
        else:                                        # iid Gaussian, s.d. of the residual by month x state
            sh = rng.standard_normal(len(e)) * sd_cell
        xs, ys = _sim(seg, d.x.values.astype(float), alpha, bq, kp, km, d.l.values.astype(float),
                      d.u.values.astype(float), sh)
        Xs = design(xs, d.q.values, d.l.values, d.u.values, bh)
        for fe, k in keys.items():
            rr = ols(ys, Xs, k, clc, NAMES)
            draws[fe].append([rr["b"][n] for n in NAMES])
    draws = {fe: np.array(v) for fe, v in draws.items()}
    for h in halves:
        draws[h.replace("_half", "_jk")] = 2 * draws[h.replace("_half", "")] - draws[h]
    out = {}
    for fe, v in draws.items():
        out[fe] = dict(bias={n: float(v[:, i].mean() - th[i]) for i, n in enumerate(NAMES)},
                       mc_se={n: float(v[:, i].std() / np.sqrt(R)) for i, n in enumerate(NAMES)},
                       sim_sd={n: float(v[:, i].std()) for i, n in enumerate(NAMES)})
    out["true"] = {n: float(th[i]) for i, n in enumerate(NAMES)}
    return out


# ================================================================== baselines
bh09 = a.bh.values
A_pub = setup(a, bh09, "pubM")
B_pub = setup(a, bh09, "pubS")
pubM = json.load(open(os.path.join(TAB, "emp_numbers.json")))
pubS = json.load(open(os.path.join(TAB, "emp_state.json")))
rA0 = fit(A_pub, "month", cl="day")
for k in NAMES:
    assert abs(rA0["b"][k] - pubM["reg"]["bh"]["coef"][k]) < 1e-9 and abs(rA0["se"][k] - pubM["reg"]["bh"]["se"][k]) < 1e-9
rB0 = fit(B_pub, "ms", cl="day")
for k, kk in zip(NAMES, ["q", "up_s", "dn_s", "q_bh", "up_s_bh", "dn_s_bh"]):
    assert abs(rB0["b"][k] - pubS["state_bands"]["coef"][kk][0]) < 1e-9
    assert abs(rB0["se"][k] - pubS["state_bands"]["coef"][kk][1]) < 1e-9
# profiling code reproduces the published bands
_, BM = bands_monthly(A_pub)
assert all(abs(BM[m][0] - v["l"]) < 1e-9 and abs(BM[m][1] - v["u"]) < 1e-9 for m, v in pubM["bands"].items())
_, BS = bands_state(B_pub)
chk = B_pub.groupby(["month", "st"])[["l_s", "u_s"]].first()
assert all(abs(BS[k][0] - chk.loc[k, "l_s"]) < 1e-9 and abs(BS[k][1] - chk.loc[k, "u_s"]) < 1e-9 for k in BS)
print(f"baselines reproduced [{time.time() - T0:.0f}s]", flush=True)

RES = {"A": {}, "B": {}}
RES["A"]["base_utc_cl"] = summarize(rA0)
RES["A"]["base"] = summarize(fit(A_pub, "month"))
RES["B"]["base_utc_cl"] = summarize(rB0)
RES["B"]["base"] = summarize(fit(B_pub, "ms"))

# ================================================================== 1. day fixed effects
rA_d = fit(A_pub, "iday"); rA_du = fit(A_pub, "uday", cl="day")
rB_ds = fit(B_pub, "iday_s"); rB_d = fit(B_pub, "iday"); rB_du = fit(B_pub, "uday", cl="day")
simA = sim_bias(A_pub, "month", ["iday", "uday"])
print(f"sim A done [{time.time() - T0:.0f}s]", flush=True)
simB = sim_bias(B_pub, "ms", ["iday_s", "iday"])
print(f"sim B done [{time.time() - T0:.0f}s]", flush=True)
RES["A"]["dayfe_raw"] = summarize(rA_d)
RES["A"]["dayfe_bc"] = summarize(rA_d, simA["iday"]["bias"])
RES["A"]["utcdayfe_raw"] = summarize(rA_du)
RES["A"]["utcdayfe_bc"] = summarize(rA_du, simA["uday"]["bias"])
RES["B"]["daystatefe_raw"] = summarize(rB_ds)
RES["B"]["daystatefe_bc"] = summarize(rB_ds, simB["iday_s"]["bias"])
RES["B"]["dayfe_raw"] = summarize(rB_d)
RES["B"]["dayfe_bc"] = summarize(rB_d, simB["iday"]["bias"])
OUT["sim_bias"] = {"A": simA, "B": simB}

# ================================================================== 2. value-date-shift days excluded
ac = a[a.clean].copy()
A_cl_re = setup(ac, ac.bh.values, "reM"); B_cl_re = setup(ac, ac.bh.values, "reS")
A_cl_pub = setup(ac, ac.bh.values, "pubM"); B_cl_pub = setup(ac, ac.bh.values, "pubS")
RES["A"]["excl_re"] = summarize(fit(A_cl_re, "month"))
RES["A"]["excl_pub"] = summarize(fit(A_cl_pub, "month"))
RES["B"]["excl_re"] = summarize(fit(B_cl_re, "ms"))
RES["B"]["excl_pub"] = summarize(fit(B_cl_pub, "ms"))
# 2 + 1 combined
simAc = sim_bias(A_cl_re, "month", ["iday"])
simBc = sim_bias(B_cl_re, "ms", ["iday_s"])
RES["A"]["excl_dayfe_raw"] = summarize(fit(A_cl_re, "iday"))
RES["A"]["excl_dayfe_bc"] = summarize(fit(A_cl_re, "iday"), simAc["iday"]["bias"])
RES["B"]["excl_daystatefe_raw"] = summarize(fit(B_cl_re, "iday_s"))
RES["B"]["excl_daystatefe_bc"] = summarize(fit(B_cl_re, "iday_s"), simBc["iday_s"]["bias"])
OUT["sim_bias_excl"] = {"A": simAc, "B": simBc}
print(f"exclusion done [{time.time() - T0:.0f}s]", flush=True)

# complementary: only the days whose T+1 rate embeds extra carry (n > 1) are dropped
am = a[a.ncarry == 1].copy()
RES["A"]["n1_re"] = summarize(fit(setup(am, am.bh.values, "reM"), "month"))
RES["B"]["n1_re"] = summarize(fit(setup(am, am.bh.values, "reS"), "ms"))

# ================================================================== 3. business-hour boundaries
im = a.imin.values
don1 = ~(((im >= 540) & (im < 600)) | ((im >= 960) & (im < 1020)))
don2 = ~(((im >= 480) & (im < 600)) | ((im >= 960) & (im < 1080)))
bh1016 = ((im >= 600) & (im < 960)).astype(float)
bh0830 = ((im >= 510) & (im < 1050)).astype(float)
a1, a2 = a[don1].copy(), a[don2].copy()
for tag, dsub, bh in [("donut1", a1, bh1016[don1]), ("donut2", a2, bh1016[don2])]:
    RES["A"][tag + "_pub"] = summarize(fit(setup(dsub, bh, "pubM"), "month"))
    RES["A"][tag + "_re"] = summarize(fit(setup(dsub, bh, "reM"), "month"))
    RES["B"][tag + "_pub"] = summarize(fit(setup(dsub, bh, "pubS"), "ms"))
    RES["B"][tag + "_re"] = summarize(fit(setup(dsub, bh, "reS"), "ms"))
RES["A"]["bh0830_pub"] = summarize(fit(setup(a, bh0830, "pubM"), "month"))
RES["B"]["bh0830_re"] = summarize(fit(setup(a, bh0830, "reS"), "ms"))
print(f"boundaries done [{time.time() - T0:.0f}s]", flush=True)

# ================================================================== additional checks P1-P7 (see docstring)
prevw = {d: (pd.Timestamp(d) - pd.Timedelta(days=3 if d.weekday() == 0 else 1)).date() for d in days}
after = {d: (d.weekday() == 0) or (not settle[prevw[d]]) for d in days}
a["after_shift"] = a.iday.map(after).astype(bool)
ap = a[~a.after_shift].copy()
RES["A"]["ph_excl_mon_re"] = summarize(fit(setup(ap, ap.bh.values, "reM"), "month"))
RES["B"]["ph_excl_mon_re"] = summarize(fit(setup(ap, ap.bh.values, "reS"), "ms"))
def _nb(d, offs):
    return {(pd.Timestamp(d) + pd.Timedelta(days=o)).date() for o in offs}


mon_ok = {d: d.weekday() == 0 and not (_nb(d, [-3, 0, 1]) & special) for d in days}
fri_ok = {d: d.weekday() == 4 and not (_nb(d, [-1, 0, 3]) & special) for d in days}
dtype = np.where(a.clean, "clean", np.where(a.iday.map(mon_ok), "mon", np.where(a.iday.map(fri_ok), "fri", "other")))
a["dtype"] = dtype
OUT["posthoc_daytype_counts"] = {k: [int(v), int(a.loc[a.dtype == k, "iday"].nunique())] for k, v in a.dtype.value_counts().items()}
for k in ["clean", "mon", "fri", "other"]:
    m_ = (a.dtype == k).values
    RES["A"]["ph_type_" + k] = summarize(fit(A_pub[m_], "month"))
    RES["B"]["ph_type_" + k] = summarize(fit(B_pub[m_], "ms"))

# P4. half-panel jackknife bias correction of the day-FE estimates (independent of the simulation DGP)
RES["A"]["ph_dayfe_jk"] = summarize(jackknife(rA_d, fit(A_pub, "iday_half")))
RES["B"]["ph_daystatefe_jk"] = summarize(jackknife(rB_ds, fit(B_pub, "iday_s_half")))
RES["B"]["ph_dayfe_jk"] = summarize(jackknife(rB_d, fit(B_pub, "iday_half")))
RES["A"]["ph_excl_dayfe_jk"] = summarize(jackknife(fit(A_cl_re, "iday"), fit(A_cl_re, "iday_half")))
RES["B"]["ph_excl_daystatefe_jk"] = summarize(jackknife(fit(B_cl_re, "iday_s"), fit(B_cl_re, "iday_s_half")))
# P5. mean five-minute drift in the premium by date type and banking state (bp), date-clustered s.e.
drift = {}
for k in ["clean", "mon", "fri", "other"]:
    for st_, v in [("open", 1.0), ("closed", 0.0)]:
        g = a[(a.dtype == k) & (a.bh == v)]
        sums = g.y.groupby(g.iday).sum(); cnts = g.y.groupby(g.iday).count()
        mu = sums.sum() / cnts.sum(); z_ = (sums - mu * cnts) / cnts.sum()
        drift[f"{k}_{st_}"] = [float(mu), float(np.sqrt((z_ ** 2).sum() * len(z_) / (len(z_) - 1))), int(cnts.sum())]
OUT["posthoc_drift_by_daytype_state"] = drift

# P6. model-based simulation with iid Gaussian shocks (no residual dependence, no misspecification):
#     is the month-FE estimator itself biased under the model, and how large is the pure date-FE bias?
R6 = max(R_SIM // 2, 2)
OUT["posthoc_sim_gauss"] = {"A": sim_bias(A_pub, "month", ["iday"], R=R6, shock="gauss"),
                            "B": sim_bias(B_pub, "ms", ["iday_s", "iday"], R=R6, shock="gauss"),
                            "A_excl": sim_bias(A_cl_re, "month", ["iday"], R=R6, shock="gauss"),
                            "B_excl": sim_bias(B_cl_re, "ms", ["iday_s"], R=R6, shock="gauss")}
print(f"gauss sims done [{time.time() - T0:.0f}s]", flush=True)
GS = OUT["posthoc_sim_gauss"]
RES["A"]["ph_dayfe_gc"] = summarize(rA_d, GS["A"]["iday"]["bias"])
RES["B"]["ph_daystatefe_gc"] = summarize(rB_ds, GS["B"]["iday_s"]["bias"])
RES["B"]["ph_dayfe_gc"] = summarize(rB_d, GS["B"]["iday"]["bias"])
RES["A"]["ph_excl_dayfe_gc"] = summarize(fit(A_cl_re, "iday"), GS["A_excl"]["iday"]["bias"])
RES["B"]["ph_excl_daystatefe_gc"] = summarize(fit(B_cl_re, "iday_s"), GS["B_excl"]["iday_s"]["bias"])


# P7. like-for-like corrections. (a) differential: raw date-FE estimate minus the extra bias that date FE
#     add relative to month FE on the same simulated paths; (b) the month-FE estimate corrected by its own
#     simulated bias, the comparison point for the absolute (wild) correction of item 1.
def diffbias(sim, fe, base):
    return {k: sim[fe]["bias"][k] - sim[base]["bias"][k] for k in NAMES}


RES["A"]["ph_dayfe_diff"] = summarize(rA_d, diffbias(simA, "iday", "month"))
RES["B"]["ph_daystatefe_diff"] = summarize(rB_ds, diffbias(simB, "iday_s", "ms"))
RES["B"]["ph_dayfe_diff"] = summarize(rB_d, diffbias(simB, "iday", "ms"))
RES["A"]["ph_excl_dayfe_diff"] = summarize(fit(A_cl_re, "iday"), diffbias(simAc, "iday", "month"))
RES["B"]["ph_excl_daystatefe_diff"] = summarize(fit(B_cl_re, "iday_s"), diffbias(simBc, "iday_s", "ms"))
RES["A"]["ph_base_bc"] = summarize(fit(A_pub, "month"), simA["month"]["bias"])
RES["B"]["ph_base_bc"] = summarize(fit(B_pub, "ms"), simB["ms"]["bias"])
RES["A"]["ph_excl_re_bc"] = summarize(fit(A_cl_re, "month"), simAc["month"]["bias"])
RES["B"]["ph_excl_re_bc"] = summarize(fit(B_cl_re, "ms"), simBc["ms"]["bias"])

# ================================================================== diagnostic: does the premium level move with carry days?
def lvl(d, cols):
    X = d[cols].values.astype(float)
    r = ols(d.x.values, X, codes(d.month.values), codes(d.iday.values), cols)
    return {k: [float(r["b"][k]), float(r["se"][k])] for k in cols} | dict(n=r["n"])


a["extra"] = a.ncarry - 1; a["fri"] = (a.idow == 4).astype(float); a["mon"] = (a.idow == 0).astype(float)
diag = dict(extra_carry_days=lvl(a, ["extra"]), friday=lvl(a, ["fri"]), friday_monday=lvl(a, ["fri", "mon"]))
for c_ in ["fri", "mon"]:
    a[c_ + "_open"] = a[c_] * a.bh; a[c_ + "_closed"] = a[c_] * (1 - a.bh)
diag["posthoc_level_by_state"] = lvl(a, ["fri_open", "fri_closed", "mon_open", "mon_closed", "bh"])
# reference daily carry from the CBRT-Fed policy-rate paths (2024-2025)
cb = [("2023-12-21", 42.5), ("2024-01-25", 45), ("2024-03-21", 50), ("2024-12-26", 47.5), ("2025-01-23", 45),
      ("2025-03-06", 42.5), ("2025-04-17", 46), ("2025-07-24", 43), ("2025-09-11", 40.5), ("2025-10-23", 39.5),
      ("2025-12-11", 38)]
fed = [("2023-07-27", 5.375), ("2024-09-19", 4.875), ("2024-11-08", 4.625), ("2024-12-19", 4.375),
       ("2025-09-18", 4.125), ("2025-10-30", 3.875), ("2025-12-11", 3.625)]


def path(ev):
    s = pd.Series({pd.Timestamp(d): v for d, v in ev})
    dd_ = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    return s.reindex(s.index.union(dd_)).ffill().reindex(dd_)


pdiff = (path(cb) - path(fed))
diag["policy_carry_bp_per_day_2024_25"] = dict(mean=float((pdiff / 365 * 100).mean()),
                                               min=float((pdiff / 365 * 100).min()), max=float((pdiff / 365 * 100).max()))
a24 = a[a.index.year <= 2025]
diag["extra_carry_days_2024_25"] = lvl(a24, ["extra"])
# premium jump across the value-date roll (last premium of sample day d to first lagged premium of the next
# sample day) by the change in embedded carry days, n(d) - n(next day); premise: jump = daily carry x change
lastx = (a.x + a.y).groupby(a.iday).last(); firstx = a.x.groupby(a.iday).first()
ov = pd.DataFrame({"jump": firstx.shift(-1) - lastx, "n": [carry[d] for d in lastx.index],
                   "n_next": [carry[d] for d in firstx.index[1:]] + [np.nan]}, index=lastx.index).dropna()
ov["dn_"] = ov.n - ov.n_next
diag["overnight_jump_by_carry_change"] = {str(int(k)): [float(v.jump.mean()), float(v.jump.std() / np.sqrt(len(v))), int(len(v))]
                                          for k, v in ov.groupby("dn_")}
Xo = np.column_stack([np.ones(len(ov)), ov.dn_.values]); bo = np.linalg.lstsq(Xo, ov.jump.values, rcond=None)[0]
eo = ov.jump.values - Xo @ bo; Vo = np.linalg.inv(Xo.T @ Xo) @ (Xo.T * eo ** 2) @ Xo @ np.linalg.inv(Xo.T @ Xo)
diag["overnight_jump_slope_per_carry_day"] = [float(bo[1]), float(np.sqrt(Vo[1, 1])), int(len(ov))]
OUT["diagnostic"] = diag

OUT["results"] = RES
json.dump(OUT, open(os.path.join(TAB, "rob_dayfe_t1.json"), "w"), indent=1)

# ================================================================== LaTeX
def star(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def row(lab, s):
    kp = s["kp"]; c, inc = kp["closed"], kp["inc"]
    rt = f"{kp['ratio']:.1f}" if kp["ratio"] else "--"
    lr = f"({kp['log_ratio'][1]:.2f})" if kp["log_ratio"] else ""
    km = s["km"]["inc"]
    top = (f"{lab} & ${c[0]:.4f}{star(*c)}$ & ${inc[0]:.4f}{star(*inc)}$ & {rt} & ${km[0]:.4f}{star(*km)}$ "
           f"& {s['n']:,} & {s['n_clusters']} \\\\\n")
    bot = f" & $({c[1]:.4f})$ & $({inc[1]:.4f})$ & {lr} & $({km[1]:.4f})$ & & \\\\\n"
    return top + bot


# the table shows every correction of every day-FE estimator (wild, Gaussian, like-for-like differential,
# jackknife) and the combined rows (2 + 1).
LAB = [("base", "Published specification$^f$"),
       ("dayfe_raw", "Date FE, raw"), ("dayfe_bc", "Date FE, bias-corrected$^a$"), ("ph_dayfe_gc", "Date FE, model-based correction$^b$"),
       ("ph_dayfe_diff", "Date FE, net of month-FE simulated bias$^d$"), ("ph_dayfe_jk", "Date FE, half-panel jackknife$^e$"),
       ("daystatefe_raw", "Date $\\times$ state FE, raw"), ("daystatefe_bc", "Date $\\times$ state FE, bias-corrected$^a$"),
       ("ph_daystatefe_gc", "Date $\\times$ state FE, model-based correction$^b$"),
       ("ph_daystatefe_diff", "Date $\\times$ state FE, net of month-FE simulated bias$^d$"),
       ("ph_daystatefe_jk", "Date $\\times$ state FE, half-panel jackknife$^e$"),
       ("excl_re", "Excl.\\ value-date-shift dates$^c$"), ("excl_pub", "Excl.\\ value-date-shift dates, published bands"),
       ("excl_dayfe_raw", "Excl.\\ dates$^c$ + date FE, raw"), ("excl_dayfe_bc", "Excl.\\ dates$^c$ + date FE, bias-corrected$^a$"),
       ("ph_excl_dayfe_gc", "Excl.\\ dates$^c$ + date FE, model-based correction$^b$"),
       ("ph_excl_dayfe_diff", "Excl.\\ dates$^c$ + date FE, net of month-FE simulated bias$^d$"),
       ("excl_daystatefe_raw", "Excl.\\ dates$^c$ + date $\\times$ state FE, raw"),
       ("excl_daystatefe_bc", "Excl.\\ dates$^c$ + date $\\times$ state FE, bias-corrected$^a$"),
       ("ph_excl_daystatefe_gc", "Excl.\\ dates$^c$ + date $\\times$ state FE, model-based correction$^b$"),
       ("ph_excl_daystatefe_diff", "Excl.\\ dates$^c$ + date $\\times$ state FE, net of month-FE simulated bias$^d$"),
       ("n1_re", "Excl.\\ dates with multi-day carry only"),
       ("ph_excl_mon_re", "Excl.\\ Mondays and post-holiday dates"),
       ("donut1_re", "Drop 09--10 and 16--17"), ("donut1_pub", "Drop 09--10 and 16--17, published bands"),
       ("donut2_re", "Drop 08--10 and 16--18"),
       ("bh0830_pub", "Open 08:30--17:30, published bands"), ("bh0830_re", "Open 08:30--17:30, bands re-estimated")]
with open(os.path.join(TAB, "rob_dayfe_t1.tex"), "w") as fh:
    fh.write("% rob_dayfe_t1.py. Day-clustered s.e. (Istanbul date) in parentheses; under Ratio, s.e. of the log ratio.\n"
             "% a: raw minus bias simulated under the month-FE estimates with day-level wild (Rademacher) residuals.\n"
             "% b: raw minus bias simulated under the month-FE estimates with iid Gaussian shocks.\n"
             "% c: Tue-Thu dates with no Turkish holiday, U.S. federal holiday or arife on the date or either neighbour; bands re-estimated.\n"
             "% d: raw minus (wild-simulated bias of the date-FE estimator minus wild-simulated bias of the month-FE estimator).\n"
             "% e: 2 x date FE minus half-date FE; halves differ in time of day, so the correction is unreliable.\n"
             "% f: Istanbul-date clusters; with the paper's UTC-date clusters the s.e. of the increment in panel A is 0.0014.\n")
    fh.write("\\begin{tabular}{lcccccc}\n\\toprule\n & $\\kappa^+$ closed & $\\Delta\\kappa^+$ (BH) & Ratio & "
             "$\\Delta\\kappa^-$ (BH) & $N$ & Days \\\\\n\\midrule\n")
    for P, title in [("A", "A. Monthly bands (Table~\\ref{tab:emp_main}, col.\\ 4)"),
                     ("B", "B. Month $\\times$ state bands (Table~\\ref{tab:emp_state}, col.\\ 1)")]:
        fh.write(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\\n")
        for k, lab in LAB:
            if k in RES[P]:
                fh.write(row(lab, RES[P][k]))
        if P == "A": fh.write("\\midrule\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# console summary
for P in "AB":
    for k, s in RES[P].items():
        kp = s["kp"]
        print(f"{P} {k:20s} kp_c {kp['closed'][0]:.4f} ({kp['closed'][1]:.4f})  inc {kp['inc'][0]:.4f} ({kp['inc'][1]:.4f})"
              f"  ratio {kp['ratio'] if kp['ratio'] is None else round(kp['ratio'], 2)}"
              f"  km_c {s['km']['closed'][0]:.4f} inc {s['km']['inc'][0]:.4f} ({s['km']['inc'][1]:.4f})"
              f"  beta_inc {s['beta']['inc'][0]:.3f}  n {s['n']} days {s['n_clusters']}")
print(json.dumps(OUT["diagnostic"], indent=1))
print(json.dumps(OUT["calendar"], indent=1))
print(f"done [{time.time() - T0:.0f}s]")
