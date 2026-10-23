"""rob_partner_banks: do new round-the-clock partner-bank lira channels raise the closed-hours speed of arbitrage?

Aim: separate the banking RAIL from the Turkish working day with events that change night-time lira
rails without changing who is awake. A new 24/7 partner bank of an exchange lets that bank's customers move lira to
and from the exchange's account at the same bank at any hour (same-bank "havale", outside EFT and outside the FAST
cap). It changes the night-time rail for those customers and
leaves the Istanbul clock, the global FX session and the time zone of traders unchanged.

EVENTS (Istanbul dates, break at 00:00 Istanbul; see the paper's Appendix C)
  E1  2025-01-27  Binance TR + N Kolay (Aktif Bank): 24/7 fee-free TRY deposits.                       [primary]
  E3  2026-03-12  Binance TR + Yapi Kredi, a large private bank: 24/7 TRY deposits and withdrawals.    [primary]
  E2  2025-11-19  Binance TR (and reportedly other exchanges) + GetirFinans on Fibabanka.  [secondary, separate]
      E2 comes 6 days after the 13 Nov 2025 restriction of Binance.com lira books to verified Binance TR users
      (the venue of our data), and Fibabanka may already have been a Binance TR 24/7 partner, so E2 may add a
      retail front end rather than a new bank. E2 is reported separately and enters only the all-three average.
  Other dated changes inside the windows, not separable by this design:
      E1: FAST request-to-pay for corporates (18 Jan 2025, marginal); MASAK travel rule from 25 Feb 2025 (crypto leg,
          all hours); CMB rule III-35/B.1 (13 Mar 2025); the 19 Mar 2025 lira crisis falls in the last five days of
          the 8-week post window (check c4 drops 19 Mar - 30 Apr 2025).
      E2: the 13 Nov 2025 restriction; 27 Nov 2025 cancellation of remaining orders and delisting of Binance TR
          USD pairs.
      E3: night-time (22:00-06:00) NFC/video verification for digital transfers >= TRY 20k, reported by banks from
          March 2026 (not verified from a primary source; it lowers off-hours capacity, the opposite direction; check c5 drops 22:00-06:00);
          MASAK description rule for transfers above TRY 400k/day from 15 Mar 2026 (all hours); CMB deadline
          postponed 26 Mar 2026.

NOTE. The business-hours increment of kappa+ is about zero in 2026 (0.0010, s.e. 0.0014), and -0.0006 (0.0011) in
Apr-Aug 2026, when reversion during business hours slows to the closed-hours pace. That period overlaps the E3 post
window and on its own pushes the E3 closed-minus-open change UP through the open-hours side, not through the
closed-hours side.

DESIGN
------
Data. data_raw/a5.parquet with the PUBLISHED monthly bands (l, u), not re-profiled. y = five-minute change in the
premium (bp); q = OFI standardized within month; up = -(x-u)+ (coefficient kappa+), dn = (l-x)+ (kappa-),
pooled = up + dn (one common speed on both sides). C = 1 - bh: closed outside 09:00-17:00 Istanbul (the clock coding
of emp_main column 4) [primary coding]. exc = distance outside the band.
Sample [primary]: intervals at most 30 bp outside the band (exc <= 30); in-band intervals are kept, as in column 4,
where they identify beta and the fixed effects. This holds the size mix of dislocations roughly fixed across
periods. Secondary: all intervals (the linear full-distance version of column 4).
Window [primary]: Istanbul dates [D-56d, D+56d): 8 weeks before against 8 weeks after; P = 1{Istanbul date >= D}.
Check: 12 weeks, [D-84d, D+84d).
Spec K2 (column 4 analogue): y on v in {q, up, dn}, each x {1, C, P, C*P}; fixed effects month x P (the event month
  split at D). For kappa+: closed-hours change = up*P + up*C*P; open-hours change = up*P; closed-minus-open change
  (the DiD) = up*C*P. kappa- reported, not tested.
Spec K1 (pooled both sides): the same with v in {q, pooled}; DiD = pooled*C*P.
Standard errors clustered by UTC day (as published), with the finite-sample correction of ols_fe, conditional on the
bands.
Pooled across events: stacked regression of the event windows with event-specific slopes and fixed effects
(event x month x P), clustered by UTC day (a day in two windows is one cluster). Pooled statistic = equal-weight mean
of the event DiDs. "Pooled primary" = mean(E1, E3); "pooled all" = mean(E1, E2, E3).
H1 (rail interpretation): DiD > 0, one-sided.
Randomization inference (RI). Placebo break dates: every Monday D whose window [D-56d, D+56d) lies inside the
  sample (Istanbul dates 2024-01-02 to 2026-08-31) and with |D - E| >= 56 days for each of E1, E2, E3 [primary
  placebo set]. Each placebo date gets the identical regression. Event-specific p = (1 + #{placebo DiD >= actual})
  / (1 + N). Pooled p: every pair (for all three: every triple) of placebo dates with non-overlapping windows
  (|Di - Dj| >= 112 days) gives one draw of the mean DiD; p = (1 + #{draws >= actual}) / (1 + #draws). Primary RI
  statistic: the level DiD; secondary: the t-statistic (for placebo means, s.e. = sqrt(sum s.e.^2)/k, exact for
  non-overlapping windows). Check placebo set: additionally excludes Mondays within 56 days of the other dated
  in-sample changes to 24/7 lira channels or to the venue: 2024-02-27 (OKX TR launch), 2024-03-12 (Binance TR +
  Turkiye Finans), 2024-04-04 (FAST cap 50k -> 100k TRY), 2025-11-13 (Binance.com lira-book restriction).
  For 12-week windows: 84 days from the events, windows inside the sample, pairs at least 168 days apart.
MDE. Analytic: MDE80 = (z_0.95 + z_0.80) x s.e. = 2.486 x s.e. (one-sided 5%, 80% power). Randomization: the 90th
  minus the 20th percentile of the placebo null (the shift that puts the statistic above the 90th placebo
  percentile with 80% probability). Benchmark: the pre-period open-minus-closed speed, i.e. the DiD if night-time
  speed rose all the way to the daytime level.
Checks (analytic s.e., kappa+ and pooled sides, each event and the pooled averages):
  c1 12-week windows [with RI]; c2 full distance [with RI]; c3 C = 1 - bh*(1-hol) (weekday public holidays
  closed, HOL list of empirics_mech.py); c4 excluding 19 Mar - 30 Apr 2025 (Istanbul dates); c5 dropping
  22:00-06:00 Istanbul, so closed = 06:00-09:00 and 17:00-22:00 (the window of the night-time verification rule);
  c6 standard errors clustered by week.
Event time (descriptive, figure): two-week bins k = -4..3 relative to D; y on q, dn (each x {1, C, P, C*P}) and
  up x 1{bin k} x (1-C), up x 1{bin k} x C; fixed effects month x P; <= 30 bp sample.

Classification stored in the JSON ("verdict"). The rail interpretation is "supported" if the pooled primary (E1 + E3)
kappa+ DiD (<= 30 bp, 8 weeks, clock coding) is positive with one-sided day-clustered p < 0.05 AND its RI p < 0.10
(primary placebo set, level statistic). "Suggestive" if exactly one of the two conditions holds with a positive point
estimate. "Not supported" otherwise. The same classification is reported for the pooled-both-sides outcome.
Qualifier: the rail interpretation predicts that the closed-hours speed itself rises. If the pooled DiD is positive
while the pooled closed-hours change is <= 0, the DiD comes from a fall in the open-hours speed, which the rail
interpretation does not predict. If the verdict is not "supported", the JSON reports whether the design could detect
an effect of the benchmark size (analytic and RI MDE against the benchmark).

DISPLAY AND DESCRIPTION. (a) In the
event-time panels, two-week bins with fewer than 30 above-band intervals in a banking state are estimated but not
plotted (some bins have 0-15 such intervals and estimates of +-10). (b) Placebo histograms are drawn on
[-0.12, 0.12]; two placebo dates (2024-07-08, 2024-09-02, whose windows share a near-empty open-hours cell in
Jul-Aug 2024) give DiDs of about -4.2 and +4.2 and are counted in the panel title. (c) The json adds, for the pooled
primary, the analytic MDE of the closed-hours change itself and the sign of the closed-hours change across all
variants.

Inputs: data_raw/a5.parquet, tables/emp_numbers.json.
Outputs: tables/rob_partner_banks.json, tables/rob_partner_banks.tex, figures/rob_partner_banks.pdf.
Run: python -P code/rob_partner_banks.py
"""
import os, sys, json, time, itertools, warnings, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
NAME = "partner_banks"
T0 = time.time()
EVENTS = {"E1": pd.Timestamp("2025-01-27"), "E2": pd.Timestamp("2025-11-19"), "E3": pd.Timestamp("2026-03-12")}
LABEL = {"E1": "Jan 2025 (N Kolay)", "E2": "Nov 2025 (GetirFinans)", "E3": "Mar 2026 (Yapi Kredi)"}
PRIMARY = ["E1", "E3"]
ALL3 = ["E1", "E3", "E2"]
OTHER_CHANGES = [pd.Timestamp(s) for s in ("2024-02-27", "2024-03-12", "2024-04-04", "2025-11-13")]
S0, S1 = pd.Timestamp("2024-01-02"), pd.Timestamp("2026-08-31")
CRISIS = (pd.Timestamp("2025-03-19"), pd.Timestamp("2025-04-30"))
Z80 = 1.6448536269514722 + 0.8416212335729143
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
OUT = dict(name=NAME, events={k: dict(date=str(v.date()), label=LABEL[k], primary=k in PRIMARY) for k, v in EVENTS.items()})

# ------------------------------------------------------------------ data
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
ist = a.index.tz_convert(None) + pd.Timedelta(hours=3)          # Istanbul, UTC+3 all year
a["ist_date"] = ist.normalize()
a["ist_hour"] = ist.hour
a["hol"] = a.ist_date.isin(pd.to_datetime(HOL)).astype(float)
a["C_clock"] = 1.0 - a.bh
a["C_bheff"] = 1.0 - a.bh * (1 - a.hol)
a["pooled"] = a.up + a.dn
a["week"] = pd.to_datetime(pd.Series(a.day.values)).dt.to_period("W").astype(str).values
a["dayc"] = pd.Series(a.day.values).astype(str).values
a["night"] = ((a.ist_hour >= 22) | (a.ist_hour < 6)).astype(bool)
a["near"] = a.exc <= 30
a["crisis"] = a.ist_date.between(*CRISIS)
MONTH = a.month.values


# ------------------------------------------------------------------ estimator (ols_fe conventions)
def ols_arr(Y, X, fe, cl, names):
    fc = pd.factorize(fe)[0]
    cnt = np.bincount(fc).astype(float)
    Y = Y - (np.bincount(fc, Y) / cnt)[fc]
    Xm = np.column_stack([np.bincount(fc, X[:, j], minlength=len(cnt)) for j in range(X.shape[1])]) / cnt[:, None]
    X = X - Xm[fc]
    keep = np.abs(X).sum(0) > 1e-10
    X = X[:, keep]; names = [n for n, k in zip(names, keep) if k]
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(cl)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=b, V=V, cols=names, n=int(n), n_clusters=int(ng))


def lin(r, w):
    g = np.zeros(len(r["cols"]))
    for c, v in w.items():
        if c not in r["cols"]:
            return [float("nan"), float("nan")]
        g[r["cols"].index(c)] += v
    return [float(g @ r["b"]), float(np.sqrt(g @ r["V"] @ g))]


def p_one(est):
    t = est[0] / est[1]
    return float(0.5 * math.erfc(t / math.sqrt(2)))


def win(d, D, days):
    return d[(d.ist_date >= D - pd.Timedelta(days=days)) & (d.ist_date < D + pd.Timedelta(days=days))]


def build(d, D, vars_, Ccol, tag):
    """Regressor block for one event window: v, v*C, v*P, v*C*P for v in vars_; FE month x P."""
    C = d[Ccol].values; P = (d.ist_date >= D).values.astype(float)
    cols, names = [], []
    for v in vars_:
        x = d[v].values.astype(float)
        for suf, m in (("", 1.0), ("_C", C), ("_P", P), ("_CP", C * P)):
            cols.append(x * m); names.append(f"{tag}{v}{suf}")
    fe = np.char.add(np.char.add(tag, d.month.values.astype(str)), np.where(P == 1, "_post", "_pre"))
    return np.column_stack(cols), names, fe


def fit(blocks, cluster="dayc"):
    """blocks: list of (d, D, vars_, Ccol, tag). Stacked OLS with block-specific slopes and FE."""
    Xs, Ys, FEs, CLs, names = [], [], [], [], []
    k_tot = sum(len(b_[2]) * 4 for b_ in blocks)
    off = 0
    for d, D, vars_, Ccol, tag in blocks:
        X, nm, fe = build(d, D, vars_, Ccol, tag)
        Xf = np.zeros((len(d), k_tot)); Xf[:, off:off + X.shape[1]] = X; off += X.shape[1]
        Xs.append(Xf); names += nm; Ys.append(d.y.values.astype(float)); FEs.append(fe); CLs.append(d[cluster].values)
    return ols_arr(np.concatenate(Ys), np.vstack(Xs), np.concatenate(FEs), np.concatenate(CLs), names)


def summ(r, v, tag="", n_cells=None):
    p = tag + v
    o = dict(open_pre=lin(r, {p: 1}), closed_pre=lin(r, {p: 1, p + "_C": 1}),
             open_post=lin(r, {p: 1, p + "_P": 1}), closed_post=lin(r, {p: 1, p + "_C": 1, p + "_P": 1, p + "_CP": 1}),
             closed_change=lin(r, {p + "_P": 1, p + "_CP": 1}), open_change=lin(r, {p + "_P": 1}),
             did=lin(r, {p + "_CP": 1}), benchmark_gap_pre=lin(r, {p + "_C": -1}))
    o["did_t"] = o["did"][0] / o["did"][1]
    o["did_p_one_sided"] = p_one(o["did"])
    o["mde80"] = Z80 * o["did"][1]
    return o


def summ_pool(r, v, tags):
    w = 1.0 / len(tags)
    def comb(suffixes, sign=1.0):
        dct = {}
        for t in tags:
            for s in suffixes:
                dct[t + v + s] = dct.get(t + v + s, 0.0) + sign * w
        return lin(r, dct)
    o = dict(closed_change=comb(["_P", "_CP"]), open_change=comb(["_P"]), did=comb(["_CP"]),
             benchmark_gap_pre=comb(["_C"], -1.0), closed_pre=comb(["", "_C"]), open_pre=comb([""]))
    o["did_t"] = o["did"][0] / o["did"][1]
    o["did_p_one_sided"] = p_one(o["did"])
    o["mde80"] = Z80 * o["did"][1]
    return o


SPECS = {"K2": ["q", "up", "dn"], "K1": ["q", "pooled"]}
KEY = {"K2": "up", "K1": "pooled"}


def sample(variant):
    d = a
    if variant.get("near", True):
        d = d[d.near]
    if variant.get("excl_crisis"):
        d = d[~d.crisis]
    if variant.get("drop_night"):
        d = d[~d.night]
    return d


def run_variant(variant, events=("E1", "E3", "E2")):
    """Per-event (separate regressions) and pooled (stacked) results for K2 and K1 under one variant."""
    days = variant.get("days", 56); Ccol = variant.get("C", "C_clock"); cl = variant.get("cluster", "dayc")
    d0 = sample(variant)
    res = {}
    for spec, vars_ in SPECS.items():
        rs = {}
        for e in events:
            dw = win(d0, EVENTS[e], days)
            r = fit([(dw, EVENTS[e], vars_, Ccol, "")], cl)
            o = summ(r, KEY[spec])
            if spec == "K2":
                o["kappa_minus"] = summ(r, "dn")
                o["beta"] = summ(r, "q")
            C = dw[Ccol].values; P = (dw.ist_date >= EVENTS[e]).values
            o["cells"] = {f"{'closed' if c else 'open'}_{'post' if p else 'pre'}":
                          dict(n=int(((C == c) & (P == p)).sum()),
                               n_above=int(((C == c) & (P == p) & (dw.x > dw.u).values).sum()),
                               n_below=int(((C == c) & (P == p) & (dw.x < dw.l).values).sum()))
                          for c in (1, 0) for p in (True, False)}
            o["n"], o["n_days"] = r["n"], r["n_clusters"]
            o["window"] = [str(dw.ist_date.min().date()), str(dw.ist_date.max().date())]
            rs[e] = o
        for pname, tags in (("pooled_primary", PRIMARY), ("pooled_all", ALL3)):
            blocks = [(win(d0, EVENTS[e], days), EVENTS[e], vars_, Ccol, e + ":") for e in tags]
            r = fit(blocks, cl)
            rs[pname] = summ_pool(r, KEY[spec], [e + ":" for e in tags])
            rs[pname]["n"], rs[pname]["n_days"] = r["n"], r["n_clusters"]
        res[spec] = rs
    return res


# ------------------------------------------------------------------ replication of emp_main column (4)
def replicate():
    X = np.column_stack([a.q, a.up, a.dn, a.q * a.bh, a.up * a.bh, a.dn * a.bh])
    r = ols_arr(a.y.values.astype(float), X, MONTH, a.dayc.values, ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
    pub = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["bh"]
    return {k: dict(here=[float(r["b"][i]), float(np.sqrt(r["V"][i, i]))], published=[pub["coef"][k], pub["se"][k]])
            for i, k in enumerate(r["cols"])}


OUT["replication_emp_main_col4"] = replicate()
print("replication", {k: (round(v["here"][0], 5), round(v["published"][0], 5)) for k, v in OUT["replication_emp_main_col4"].items()}, flush=True)

# ------------------------------------------------------------------ main and checks
VARIANTS = {
    "main": dict(),
    "c1_12w": dict(days=84),
    "c2_full": dict(near=False),
    "c2_full_12w": dict(near=False, days=84),
    "c3_bheff": dict(C="C_bheff"),
    "c4_excl_crisis": dict(excl_crisis=True),
    "c5_drop_22_06": dict(drop_night=True),
    "c6_week_cluster": dict(cluster="week"),
}
RES = {}
for vn, v in VARIANTS.items():
    RES[vn] = run_variant(v)
    for spec in ("K2", "K1"):
        for e in ("E1", "E3", "pooled_primary", "E2", "pooled_all"):
            o = RES[vn][spec][e]
            print(f"{vn:16s} {spec} {e:15s} closed chg {o['closed_change'][0]:+.4f} ({o['closed_change'][1]:.4f})  "
                  f"open chg {o['open_change'][0]:+.4f} ({o['open_change'][1]:.4f})  DiD {o['did'][0]:+.4f} ({o['did'][1]:.4f}) "
                  f"p1 {o['did_p_one_sided']:.3f}  bench {o['benchmark_gap_pre'][0]:+.4f}", flush=True)
OUT["results"] = RES

# ------------------------------------------------------------------ randomization inference
def mondays(days, excl_dates):
    lo, hi = S0 + pd.Timedelta(days=days), S1 + pd.Timedelta(days=1) - pd.Timedelta(days=days)
    out = []
    for D in pd.date_range(lo, hi, freq="W-MON"):
        if D - pd.Timedelta(days=days) < S0 or D + pd.Timedelta(days=days) > S1 + pd.Timedelta(days=1):
            continue
        if all(abs((D - E).days) >= days for E in excl_dates):
            out.append(D)
    return out


def placebo_stats(variant, dates):
    days = variant.get("days", 56); Ccol = variant.get("C", "C_clock")
    d0 = sample(variant)
    rows = []
    for D in dates:
        dw = win(d0, D, days)
        row = dict(date=str(D.date()))
        for spec, vars_ in SPECS.items():
            try:
                r = fit([(dw, D, vars_, Ccol, "")])
                o = summ(r, KEY[spec])
                row[spec] = dict(did=o["did"], t=o["did_t"], closed_change=o["closed_change"][0], open_change=o["open_change"][0])
            except Exception as ex:
                row[spec] = dict(error=repr(ex))
        rows.append(row)
    return rows


def ri(rows, actual, spec, days, check_dates=None):
    ok = [r_ for r_ in rows if "did" in r_[spec] and np.isfinite(r_[spec]["did"][0])
          and (check_dates is None or r_["date"] in check_dates)]
    dates = [pd.Timestamp(r_["date"]) for r_ in ok]
    g = np.array([r_[spec]["did"][0] for r_ in ok]); s = np.array([r_[spec]["did"][1] for r_ in ok])
    out = dict(N=len(ok))
    for e in ("E1", "E3", "E2"):
        A = actual[spec][e]
        out[e] = dict(actual=A["did"][0], p_level=float((1 + np.sum(g >= A["did"][0])) / (1 + len(g))),
                      p_t=float((1 + np.sum(g / s >= A["did_t"])) / (1 + len(g))),
                      mde_ri=float(np.quantile(g, 0.9) - np.quantile(g, 0.2)))
    out["placebo_quantiles"] = [float(x) for x in np.quantile(g, [0, .05, .1, .2, .5, .8, .9, .95, 1])]
    out["placebo_sd"] = float(g.std(ddof=1))
    out["share_placebo_t_above_1_645"] = float(np.mean(g / s > 1.645))
    for pname, k in (("pooled_primary", 2), ("pooled_all", 3)):
        A = actual[spec][pname]
        idx = [c for c in itertools.combinations(range(len(dates)), k)
               if all(abs((dates[i] - dates[j]).days) >= 2 * days for i, j in itertools.combinations(c, 2))]
        idx = np.array(idx)
        m = g[idx].mean(1); tt = m / (np.sqrt((s[idx] ** 2).sum(1)) / k)
        out[pname] = dict(actual=A["did"][0], actual_t=A["did_t"], N_draws=int(len(idx)),
                          p_level=float((1 + np.sum(m >= A["did"][0])) / (1 + len(m))),
                          p_t=float((1 + np.sum(tt >= A["did_t"])) / (1 + len(m))),
                          mde_ri=float(np.quantile(m, 0.9) - np.quantile(m, 0.2)),
                          null_quantiles=[float(x) for x in np.quantile(m, [.05, .1, .2, .5, .8, .9, .95])])
        if pname == "pooled_primary":
            out[pname]["draws"] = m            # kept for the figure; dropped before writing json
    return out


RI = {}
for vn in ("main", "c1_12w", "c2_full", "c2_full_12w"):
    v = VARIANTS[vn]; days = v.get("days", 56)
    dates = mondays(days, list(EVENTS.values()))
    dates_chk = {str(D.date()) for D in mondays(days, list(EVENTS.values()) + OTHER_CHANGES)}
    rows = placebo_stats(v, dates)
    RI[vn] = dict(dates=[str(D.date()) for D in dates], rows=rows,
                  primary={s: ri(rows, RES[vn], s, days) for s in SPECS},
                  check_excl_other_changes={s: ri(rows, RES[vn], s, days, dates_chk) for s in SPECS})
    for s in SPECS:
        q = RI[vn]["primary"][s]; qc = RI[vn]["check_excl_other_changes"][s]
        print(f"RI {vn:12s} {s} N={q['N']} p E1 {q['E1']['p_level']:.3f} E3 {q['E3']['p_level']:.3f} E2 {q['E2']['p_level']:.3f} "
              f"pooledP {q['pooled_primary']['p_level']:.3f} (t {q['pooled_primary']['p_t']:.3f}, draws {q['pooled_primary']['N_draws']}) "
              f"all3 {q['pooled_all']['p_level']:.3f} | check N={qc['N']} pooledP {qc['pooled_primary']['p_level']:.3f}", flush=True)
print("runtime so far", time.time() - T0, flush=True)

# ------------------------------------------------------------------ event time (two-week bins), <= 30 bp
def event_time(e, days=56, binw=14):
    D = EVENTS[e]
    dw = win(sample({}), D, days).copy()
    C = dw.C_clock.values; P = (dw.ist_date >= D).values.astype(float)
    k = ((dw.ist_date - D).dt.days // binw).values
    cols, names = [], []
    for v in ("q", "dn"):
        x = dw[v].values
        for suf, m in (("", 1.0), ("_C", C), ("_P", P), ("_CP", C * P)):
            cols.append(x * m); names.append(v + suf)
    bins = sorted(set(k))
    for b_ in bins:
        mk = (k == b_).astype(float)
        cols.append(dw.up.values * mk * (1 - C)); names.append(f"up_o{b_}")
        cols.append(dw.up.values * mk * C); names.append(f"up_c{b_}")
    fe = np.char.add(dw.month.values.astype(str), np.where(P == 1, "_post", "_pre"))
    r = ols_arr(dw.y.values.astype(float), np.column_stack(cols), fe, dw.dayc.values, names)
    rows = []
    for b_ in bins:
        rows.append(dict(bin=int(b_), open=lin(r, {f"up_o{b_}": 1}), closed=lin(r, {f"up_c{b_}": 1}),
                         n_above_open=int(((k == b_) & (C == 0) & (dw.x > dw.u).values).sum()),
                         n_above_closed=int(((k == b_) & (C == 1) & (dw.x > dw.u).values).sum())))
    return rows


EV = {e: event_time(e) for e in ("E1", "E3", "E2")}
OUT["event_time_kappa_plus_le30"] = EV

# ------------------------------------------------------------------ verdict (see docstring)
def verdict(spec):
    o = RES["main"][spec]["pooled_primary"]; q = RI["main"]["primary"][spec]["pooled_primary"]
    c1 = o["did"][0] > 0 and o["did_p_one_sided"] < 0.05
    c2 = o["did"][0] > 0 and q["p_level"] < 0.10
    v = "supported" if (c1 and c2) else ("suggestive" if (c1 or c2) else "not supported")
    out = dict(verdict=v, did=o["did"], p_one_sided=o["did_p_one_sided"], ri_p=q["p_level"], ri_p_t=q["p_t"],
               closed_change=o["closed_change"], open_change=o["open_change"],
               benchmark_gap_pre=o["benchmark_gap_pre"], mde80_analytic=o["mde80"], mde_ri=q["mde_ri"])
    out["qualifier_did_from_open_side"] = bool(o["did"][0] > 0 and o["closed_change"][0] <= 0)
    out["benchmark_detectable_analytic"] = bool(o["benchmark_gap_pre"][0] > o["mde80"])
    out["benchmark_detectable_ri"] = bool(o["benchmark_gap_pre"][0] > q["mde_ri"])
    return out


OUT["verdict"] = dict(rule="supported if the pooled primary (E1+E3) kappa+ DiD (<=30 bp, 8 weeks, clock) > 0 with one-sided "
                           "day-clustered p < 0.05 AND RI p < 0.10; suggestive if exactly one holds; not supported otherwise",
                      kappa_plus=verdict("K2"), pooled_sides_secondary=verdict("K1"))
# descriptive additions (c)
pp = RES["main"]["K2"]["pooled_primary"]
OUT["verdict"]["posthoc"] = dict(
    mde80_closed_change_pooled_primary=Z80 * pp["closed_change"][1], closed_pre_pooled_primary=pp["closed_pre"],
    closed_change_pooled_primary_by_variant={vn: {sp: RES[vn][sp]["pooled_primary"]["closed_change"] for sp in SPECS} for vn in RES},
    open_change_pooled_primary_by_variant={vn: {sp: RES[vn][sp]["pooled_primary"]["open_change"] for sp in SPECS} for vn in RES},
    extreme_placebos="2024-07-08 and 2024-09-02 give kappa+ DiDs of about -4.2 and +4.2 (near-empty open-hours cell, Jul-Aug 2024)",
    full_sample_bh_increment_kappa_plus=json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["bh"]["coef"]["up_bh"])
print("VERDICT", json.dumps(OUT["verdict"], default=float), flush=True)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
CO, CC, CG, CL = "#2a78d6", "#eb6834", "#52514e", "#b7cdea"
fig, ax = plt.subplots(2, 3, figsize=(7.6, 5.6))
for j, e in enumerate(("E1", "E3", "E2")):
    A_ = ax[0, j]; R_ = EV[e]
    xx = np.array([r_["bin"] for r_ in R_]) * 2 + 1.0                 # midpoint of the two-week bin, in weeks
    for nm, col, mk, off in (("open", CO, "o", -0.25), ("closed", CC, "s", 0.25)):
        ok = np.array([r_["n_above_" + nm] >= 30 for r_ in R_])        # display rule (a)
        yv = np.where(ok, [r_[nm][0] for r_ in R_], np.nan); ev_ = np.where(ok, [1.96 * r_[nm][1] for r_ in R_], np.nan)
        A_.errorbar(xx + off, yv, yerr=ev_, fmt=mk + "-", ms=4.5, lw=1.3,
                    color=col, ecolor=col, elinewidth=0.9, capsize=0, mec="white", mew=0.8,
                    label="banks open (09-17)" if nm == "open" else "banks closed")
    A_.axvline(0, color=CG, lw=0.8, ls="--"); A_.axhline(0, color=CG, lw=0.5)
    YL = (-0.05, 0.11)                                                 # display rule (a): common axis, off-axis points counted
    n_off = sum(r_["n_above_" + nm] >= 30 and not (YL[0] <= r_[nm][0] <= YL[1]) for r_ in R_ for nm in ("open", "closed"))
    A_.set_ylim(*YL)
    A_.set_title(f"({'abc'[j]}) {LABEL[e]}" + ("" if e != "E2" else ",\n6 days after lira-book restriction")
                 + (f"\n({n_off} estimate off axis)" if n_off else ""), fontsize=8.5)
    A_.set_xlabel("weeks from event"); A_.set_xticks([-8, -4, 0, 4, 8])
    A_.grid(axis="y", color="#e6e5e1", lw=0.6); A_.set_axisbelow(True)
ax[0, 0].set_ylabel(r"$\kappa^+$ within 30 bp, per 5 min")
ax[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
# (d) DiD by event: kappa+ <= 30 bp, pooled sides <= 30 bp, kappa+ full distance
D_ = ax[1, 0]
groups = [("E1", "Jan 2025"), ("E3", "Mar 2026"), ("pooled_primary", "Jan+Mar"), ("E2", "Nov 2025"), ("pooled_all", "All three")]
series = [("main", "K2", r"$\kappa^+$, $\leq$30 bp", CC, "s"), ("main", "K1", r"both sides, $\leq$30 bp", CO, "o"),
          ("c2_full", "K2", r"$\kappa^+$, full distance", CG, "D")]
for i, (vn, sp, lab, col, mk) in enumerate(series):
    ys = np.arange(len(groups)) + (i - 1) * 0.22
    est = [RES[vn][sp][g]["did"] for g, _ in groups]
    D_.errorbar([x_[0] for x_ in est], ys, xerr=[1.96 * x_[1] for x_ in est], fmt=mk, ms=4, color=col, ecolor=col,
                elinewidth=0.9, capsize=0, mec="white", mew=0.6, label=lab)
D_.axvline(0, color=CG, lw=0.5)
D_.set_yticks(np.arange(len(groups))); D_.set_yticklabels([g[1] for g in groups]); D_.invert_yaxis()
D_.set_xlabel("closed-minus-open change, per 5 min")
D_.set_title("(d) Change in closed minus open", fontsize=8.5)
D_.set_xlim(-0.07, 0.095); D_.set_ylim(len(groups) + 0.9, -0.5); D_.legend(frameon=False, fontsize=6.5, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
# (e) single-date placebo null, kappa+ <= 30 bp
E_ = ax[1, 1]
g = np.array([r_["K2"]["did"][0] for r_ in RI["main"]["rows"] if "did" in r_["K2"]])
XL = 0.12                                                            # display rule (b)
E_.hist(g[np.abs(g) <= XL], bins=np.linspace(-XL, XL, 25), color=CL, edgecolor="white", linewidth=1.0,
        label=f"placebo Mondays (N={len(g)})")
E_.set_xlim(-XL, XL); E_.set_ylim(0, E_.get_ylim()[1] * 1.6)
for e, ls in (("E1", "-"), ("E3", "--"), ("E2", ":")):
    E_.axvline(RES["main"]["K2"][e]["did"][0], color=CC, lw=1.6, ls=ls, label=LABEL[e].split(" (")[0])
E_.axvline(0, color=CG, lw=0.5)
E_.set_xlabel(r"$\kappa^+$ closed minus open"); E_.set_ylabel("number of dates")
E_.set_title(f"(e) Placebo break dates ({int(np.sum(np.abs(g) > XL))} off axis)", fontsize=8.5)
E_.legend(frameon=False, fontsize=6.5, loc="upper left")
# (f) pooled-primary null (pairs of non-overlapping placebo windows)
F_ = ax[1, 2]
m = RI["main"]["primary"]["K2"]["pooled_primary"]["draws"]
F_.hist(m[np.abs(m) <= XL], bins=np.linspace(-XL, XL, 25), color=CL, edgecolor="white", linewidth=0.6,
        label=f"placebo pairs ({int(np.sum(np.abs(m) > XL))} off axis)")
F_.set_xlim(-XL, XL); F_.set_ylim(0, F_.get_ylim()[1] * 1.5)
F_.axvline(RES["main"]["K2"]["pooled_primary"]["did"][0], color=CC, lw=2, label="Jan 2025 + Mar 2026")
F_.axvline(0, color=CG, lw=0.5)
F_.set_xlabel(r"mean of two placebo changes"); F_.set_ylabel("number of pairs")
F_.set_title(f"(f) Pooled primary: RI p = {RI['main']['primary']['K2']['pooled_primary']['p_level']:.2f}", fontsize=8.5)
F_.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rob_{NAME}.pdf")); plt.close(fig)


# ------------------------------------------------------------------ LaTeX table
def star_p(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def cell(v, nd=4):
    return f"${v[0]:.{nd}f}{star_p(v[0], v[1])}$", f"$({v[1]:.{nd}f})$"


COLS = ["E1", "E3", "pooled_primary", "E2", "pooled_all"]
with open(os.path.join(TAB, f"rob_{NAME}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccccc}\n\\toprule\n & Jan 2025 & Mar 2026 & Mean of & Nov 2025 & Mean of \\\\\n"
             " & N Kolay & Yap\\i{} Kredi & Jan + Mar & GetirFinans & all three \\\\\n\\midrule\n")

    def block(title, vn, spec, ri_key=None, full=True):
        fh.write(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\\n")
        rows = [("Closed-hours change", "closed_change"), ("Open-hours change", "open_change"),
                ("Closed minus open", "did")] if full else [("Closed minus open", "did")]
        for lab, k in rows:
            cs = [cell(RES[vn][spec][c][k]) for c in COLS]
            fh.write(lab + " & " + " & ".join(c[0] for c in cs) + " \\\\\n & " + " & ".join(c[1] for c in cs) + " \\\\\n")
        if ri_key:
            q = RI[ri_key]["primary"][spec]
            fh.write("\\quad Randomization $p$ (one-sided) & " + " & ".join(f"{q[c]['p_level']:.2f}" for c in COLS) + " \\\\\n")
        if full:
            fh.write("\\quad MDE, 80\\% power & " + " & ".join(f"{RES[vn][spec][c]['mde80']:.4f}" for c in COLS) + " \\\\\n")
            fh.write("\\quad Pre-period open minus closed & " + " & ".join(f"{RES[vn][spec][c]['benchmark_gap_pre'][0]:.4f}" for c in COLS) + " \\\\\n")

    block("A. Above the band, within 30 bp ($\\kappa^+$)", "main", "K2", "main")
    block("B. Both sides pooled, within 30 bp", "main", "K1", "main")
    block("C. Above the band, full distance", "c2_full", "K2", "c2_full", full=False)
    fh.write("\\multicolumn{6}{l}{\\emph{D. Checks: closed minus open, $\\kappa^+$ within 30 bp}} \\\\\n")
    for lab, vn in (("12-week windows", "c1_12w"), ("Weekday holidays coded closed", "c3_bheff"),
                    ("Excluding 19 Mar -- 30 Apr 2025", "c4_excl_crisis"), ("Excluding 22:00--06:00 Istanbul", "c5_drop_22_06"),
                    ("Clustered by week", "c6_week_cluster")):
        cs = [cell(RES[vn]["K2"][c]["did"]) for c in COLS]
        fh.write(lab + " & " + " & ".join(c[0] for c in cs) + " \\\\\n & " + " & ".join(c[1] for c in cs) + " \\\\\n")
        if vn == "c1_12w":
            q = RI["c1_12w"]["primary"]["K2"]
            fh.write("\\quad Randomization $p$ (one-sided) & " + " & ".join(f"{q[c]['p_level']:.2f}" for c in COLS) + " \\\\\n")
    nP = RI["main"]["primary"]["K2"]["N"]; nPP = RI["main"]["primary"]["K2"]["pooled_primary"]["N_draws"]
    fh.write("\\midrule\n")
    fh.write(f"Placebo dates (pairs, triples) & \\multicolumn{{5}}{{c}}{{{nP} Mondays ({nPP:,} pairs, "
             f"{RI['main']['primary']['K2']['pooled_all']['N_draws']:,} triples)}} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ json
for vn in RI:
    for s in SPECS:
        for blk in ("primary", "check_excl_other_changes"):
            RI[vn][blk][s]["pooled_primary"].pop("draws", None)
OUT["randomization"] = RI
OUT["runtime_s"] = time.time() - T0
json.dump(OUT, open(os.path.join(TAB, f"rob_{NAME}.json"), "w"), indent=1, default=float)
print("done", OUT["runtime_s"])
