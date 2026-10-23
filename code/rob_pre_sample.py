"""rob_pre_sample: when did the business-hours gap in the speed of arbitrage appear? The baseline extended back to
January 2021 - December 2023.

Extends the main analysis (paper sample Jan 2024 - Aug 2026) back to 2021-2023 to show when the business-hours
increase in kappa+ appeared. rob_btcturk_venue.py finds no business-hours increase above the band in Mar-Dec 2020 on
Binance or BtcTurk (paper's column-(4) specification on Binance, Dec 2019 - Dec 2020: increment -0.0061, s.e.
0.0108; closed speed 0.043). Reference values for 2024-2026: tables/rob_by_period.json (increments 0.0077 (0.0016)
in 2024, 0.0094 (0.0033) in 2025, 0.0010 (0.0014) in 2026; rolling 6-month increments 2024-2026) and the bin-average
contrast of tables/rob_size_dependence.json (1c_binavg_full).

DESIGN
------
Data (code/rob_pre_sample_download.py; new files under data_raw/pre_sample/ only):
  Binance USDTTRY one-minute klines 2021-01..2023-12 (download_binance.get); Dukascopy USD/TRY one-minute BID and
  ASK candles for the weekdays of 2021-2023 (download_dukascopy URL scheme), downloaded 2023 first, then 2022, then
  2021, by one sequential process. If a year's FX files are incomplete when the analysis is run, that year uses the
  days available and the JSON records the coverage. (In the authors' run coverage was complete: all 781 weekdays
  x 2 sides.)

Panel: EXACTLY the construction of data_raw/a5.parquet (build_panel.load_binance, build_panel.load_fx and
  empirics_main.py, copied below with the raw folder as an argument):
  P^L = Binance minute VWAP (qv/v, missing when v = 0); OFI = 2 TBV - V; interbank minutes with tick volume > 0,
  forward-filled at most 2 minutes onto the Binance minute index; b = 1e4 log(P^L / mid); b missing on weekends,
  in 20:30-22:30 UTC and when |b| > 1000 bp; five-minute means with at least 2 valid minutes; OFI summed and
  standardized by its within-month SD; consecutive five-minute intervals only; y = delta b, x = lagged b;
  bh = 1 for 09:00-17:00 Istanbul = 06:00-14:00 UTC (UTC+3 throughout; Turkey has kept UTC+3 since Sept 2016).
  Bands: month by month (months with >= 500 intervals), profiled with model._ssr_grid on a 1 bp grid between the
  5th and 95th percentiles of x (5% trimming), with OFI in the profile, l <= u.
  Mid-quote. In the 2024-2026 panel, 836 of 848 USD/TRY days have only BID files, so its mid is bid x (1 + half
  the median relative bid-ask spread) on almost every day (load_fx falls back to this when the ASK is missing).
  PRIMARY (comparable with the published 2024-2026 numbers): the same BID-based mid, mid = bid x (1 + h), with
  h = half the median relative spread over 2021-2023 minutes where both sides exist (load_fx's own formula; if
  no ASK file is available, h of the 2024-2026 panel). A constant h only shifts b and the bands.
  CHECK C1: load_fx verbatim on BID and ASK files, i.e. mid = (bid + ask)/2 wherever the ASK exists; bands
  re-profiled on that premium.
  Replication check (run first): the same code applied to the 2024-2026 raw files must reproduce a5.parquet
  (N, b, y, x, q, bh, l, u). The script stops if it does not.

Holidays. HOL (weekday public holidays; banks and EFT closed) extended to 2021-2023 from Law No. 2429 on National
  and General Holidays (fixed-date holidays: 1 Jan, 23 Apr, 1 May, 19 May, 15 Jul, 30 Aug, 29 Oct; 28 Oct
  afternoon and the eves (arife) of the religious holidays are half days) and the Presidency of Religious Affairs
  (Diyanet) dates of Ramazan Bayrami (2021: 13-15 May; 2022: 2-4 May; 2023: 21-23 Apr) and Kurban Bayrami (2021:
  20-23 Jul; 2022: 9-12 Jul; 2023: 28 Jun - 1 Jul). The list is cross-checked against the CBRT indicative-rate
  archive (www.tcmb.gov.tr/kurlar/YYYYMM/DDMMYYYY.xml returns 404 on public holidays and half days), the same
  check as data_raw/tcmb_indicative_rate_calendar.csv for 2024-2026; mismatches are reported, not silently fixed.
  As in the baseline, the column-(4) specification uses the clock indicator bh; holidays enter only check C2.

(1) Main estimates by calendar year 2021, 2022, 2023 (separate regression per year on that year's intervals), and
    the same code on a5.parquet for 2024, 2025, 2026 (must reproduce tables/rob_by_period.json, spec a; the
    published values are reported alongside): Table emp_main column (4),
       y = a_m + beta q + beta_bh q*bh - kappa+ (x-u)+ - d+ (x-u)+ *bh + kappa- (l-x)+ + d- (l-x)+ *bh + e,
    month FE, standard errors clustered by UTC day with the finite-sample correction of ols_fe, conditional on the
    bands. Reported per year: business-hours increment d+ of kappa+ and its s.e., closed speed kappa+, open speed
    kappa+ + d+ (s.e. from the joint covariance), open/closed ratio when both are positive, one-sided p of d+ > 0;
    the same for kappa-; beta by state; N; days; median monthly band width; the share of intervals more than 30 bp
    and more than 60 bp above the monthly band (all intervals, and by banking state); mean five-minute USDT
    volume (thousand USDT; all, open, closed); mean and SD of the premium.
    The JSON flags the gap as "present" in a year if d+ > 0 at 5% one-sided (z > 1.645).
(2) Six-month rolling windows (6 consecutive calendar months, step one month) across 2021-01..2026-08, pre-sample
    panel and a5.parquet stacked (same construction): closed and open kappa+, increment with 95% CI, the share of
    intervals more than 30 and 60 bp above the band in the window. Descriptive: the first window
    whose increment is positive at 5% one-sided, and for each calendar year (by window end month) the number of
    windows with a significant increment.
(3) Bin-average contrast by year (rob_size_dependence.binavg, spec 1c): regressors -e*1{e in bin k} and their bh
    interactions above the band (bins (0,10], (10,30], (30,60], >60 bp), the same below, q and q*bh, month FE.
    Reported per year for the >60 bp bin: closed and open proportional speeds and the contrast (open - closed),
    with s.e., and the number of intervals more than 60 bp above the band by state. A bin x state column with no
    nonzero entry in a year is dropped (its coefficient is not identified); the >60 bp contrast is reported only
    when each banking state has at least 30 intervals more than 60 bp above the band.
(4) Checks (2021-2023, by year): C1 bid/ask mid as above; C2 bh replaced by bh_eff = bh x (1 - hol) (HOL on the UTC
    date, as empirics_mech.py), also computed for 2024-2026; C3 2021-2023 pooled, and the difference between the
    pooled 2021-2023 increment and the pooled 2024-2025 increment (disjoint days, independent; normal test).
(5) Descriptives by year: share of Binance weekday minutes outside 20:30-22:30 UTC with a usable interbank mid, by
    banking state; median relative bid-ask spread (bp) where the ASK exists, by banking state; mean USD/TRY; the
    per-transfer FAST limit in force (tables/rob_xc_inst_try.json) and its USD value at the year's mean rate
    (verified limits only). Institutional facts are listed by year without any causal claim.

Diagnostics: (i) the six-month rolling windows of (2) re-estimated on the bid/ask-mid panel of check C1, windows
inside 2021-2023 only, because C1 changes the 2021 result; shown in the figure as a thin line; (ii) the number of
months whose profiled band has zero width (l = u), by year (descriptive).

Inputs: data_raw/pre_sample/ (rob_pre_sample_download.py), data_raw/binance and data_raw/dukascopy (2024-2026, for the
replication check), data_raw/a5.parquet, tables/rob_by_period.json, tables/rob_xc_inst_try.json (FAST limits).

Outputs: tables/rob_pre_sample.json, tables/rob_pre_sample.tex, figures/rob_pre_sample.pdf.
Env: REPL (1: run the a5 replication check), MINYEAR_PRE/MAXYEAR_PRE (default 2021/2023; for debugging only),
     ONLY_REPL (stop after the replication check), OUTDIR (write the table, JSON and figure there instead of
     tables/ and figures/, and leave data_raw/pre_sample/a5_pre.parquet unchanged).
"""
import os, sys, glob, lzma, json, time, datetime as dt
import numpy as np, pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import _ssr_grid
from build_panel import COLS

RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
TAB_OUT = os.environ.get("OUTDIR", TAB); FIG_OUT = os.environ.get("OUTDIR", FIG)
PRE = os.path.join(RAW, "pre_sample")
NAME = "pre_sample"
REPL = int(os.environ.get("REPL", 1))
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
T_START = time.time()
OUT = dict(name=NAME)

# ------------------------------------------------------------------ holidays
HOL_MAIN = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17",
            "2024-06-18", "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
            "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06",
            "2025-06-09", "2025-07-15", "2025-10-29",
            "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28",
            "2026-05-29", "2026-07-15"]          # = empirics_mech.HOL
HOL_PRE = ["2021-01-01", "2021-04-23", "2021-05-13", "2021-05-14", "2021-05-19", "2021-07-15", "2021-07-20",
           "2021-07-21", "2021-07-22", "2021-07-23", "2021-08-30", "2021-10-29",
           "2022-05-02", "2022-05-03", "2022-05-04", "2022-05-19", "2022-07-11", "2022-07-12", "2022-07-15",
           "2022-08-30",
           "2023-04-21", "2023-05-01", "2023-05-19", "2023-06-28", "2023-06-29", "2023-06-30", "2023-08-30"]
HALF_PRE = ["2021-05-12", "2021-07-19", "2021-10-28", "2022-07-08", "2022-10-28", "2023-04-20", "2023-06-27"]
HOL = HOL_PRE + HOL_MAIN
HOL_SOURCE = ("Law No. 2429 on National and General Holidays (Ulusal Bayram ve Genel Tatiller Hakkinda Kanun, "
              "17 Mar 1981, as amended: 1 Jan, 23 Apr, 1 May, 19 May, 15 Jul, 30 Aug, 29 Oct; half day on 28 Oct and "
              "on the eves of the religious holidays) and the Presidency of Religious Affairs (Diyanet Isleri "
              "Baskanligi) calendar for Ramazan Bayrami (2021-05-13..15, 2022-05-02..04, 2023-04-21..23) and Kurban "
              "Bayrami (2021-07-20..23, 2022-07-09..12, 2023-06-28..07-01); weekday dates only. Cross-checked "
              "against the CBRT indicative-rate archive (no file on holidays and half days).")


# ------------------------------------------------------------------ loaders (build_panel with a folder argument)
def load_binance_dir(folder, sym):
    """build_panel.load_binance, reading <folder>/<sym>-1m-*.csv."""
    fs = sorted(glob.glob(os.path.join(folder, f"{sym}-1m-*.csv")))
    dfs = []
    for f in fs:
        d = pd.read_csv(f, header=None, names=COLS)
        t = d.open_time.astype("int64")
        unit = np.where(t > 1e14, "us", "ms")
        ts = np.where(unit == "us", t // 1000, t)
        d.index = pd.to_datetime(ts, unit="ms", utc=True)
        dfs.append(d)
    d = pd.concat(dfs)
    d = d[~d.index.duplicated()]
    vwap = np.where(d.v > 0, d.qv / d.v.replace(0, np.nan), np.nan)
    return pd.DataFrame({"vwap": vwap, "close": d.c, "v": d.v, "tbv": d.tbv, "n": d.n}, index=d.index)


def read_fx_raw(folder, pair="USDTRY", years=None):
    rows = []
    for side in ("BID", "ASK"):
        for f in sorted(glob.glob(os.path.join(folder, f"{pair}_{side}_*.bi5"))):
            if years is not None and os.path.basename(f).split("_")[2][:4] not in years: continue
            if os.path.getsize(f) == 0: continue
            try:
                raw = lzma.decompress(open(f, "rb").read())
            except Exception:
                continue
            day = dt.datetime.strptime(os.path.basename(f).split("_")[2][:8], "%Y%m%d").replace(tzinfo=dt.timezone.utc)
            a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6)
            secs = a[:, 0].astype("int64")
            close = a[:, 2] / 1e5
            vol = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5]
            idx = pd.Timestamp(day) + pd.to_timedelta(secs, unit="s")
            rows.append(pd.DataFrame({"side": side, "px": close, "fxv": vol}, index=idx))
    return pd.concat(rows)


def load_fx_dir(folder, pair="USDTRY", use_ask=True, half=None, years=None):
    """build_panel.load_fx on <folder>. use_ask=True is load_fx verbatim (mid = (bid+ask)/2 where the ASK exists,
    bid x (1+h) otherwise); use_ask=False gives mid = bid x (1+h) everywhere, the construction of almost every
    day of the 2024-2026 panel. h = half the median relative spread over minutes with both sides (load_fx), or
    `half` if given."""
    fx = read_fx_raw(folder, pair, years)
    bid = fx[fx.side == "BID"]; ask = fx[fx.side == "ASK"]
    bid = bid[~bid.index.duplicated()]; ask = ask[~ask.index.duplicated()]
    m = pd.DataFrame({"bid": bid.px, "fxv": bid.fxv})
    m["ask"] = ask.px.reindex(m.index)
    m = m[(m.fxv > 0)]  # Dukascopy pads non-trading minutes with zero-volume flat candles
    rel = ((m.ask - m.bid) / m.bid).dropna()
    h = 0.5 * (rel.median() if len(rel) else 1e-4)
    if half is not None:
        h = half
    if use_ask:
        m["fx_mid"] = np.where(m.ask.notna(), (m.bid + m.ask) / 2, m.bid * (1 + h))
    else:
        m["fx_mid"] = m.bid * (1 + h)
    m["fx_spread"] = (m.ask - m.bid) / m.fx_mid
    info = dict(half_bp=float(1e4 * h), n_minutes=int(len(m)), n_minutes_with_ask=int(m.ask.notna().sum()),
                median_rel_spread_bp=float(1e4 * rel.median()) if len(rel) else None)
    return m[["fx_mid", "fx_spread", "fxv"]], info


# ------------------------------------------------------------------ panel (empirics_main.py, verbatim steps)
def build_minutes(bin_folder, fx, start, end):
    u = load_binance_dir(bin_folder, "USDTTRY")
    p = pd.DataFrame(index=u.index)
    p["p_usdt"] = u.vwap; p["ofi"] = 2 * u.tbv - u.v; p["vol"] = u.v; p["ntrades"] = u.n
    p = p.loc[start:end]
    p = p.join(fx.reindex(p.index, method="ffill", limit=2))
    p["b"] = 1e4 * np.log(p.p_usdt / p.fx_mid)
    hr = p.index.hour + p.index.minute / 60
    p.loc[(hr >= 20.5) & (hr < 22.5), "b"] = np.nan      # FX rollover: stale/wide interbank quotes
    p.loc[p.index.dayofweek >= 5, "b"] = np.nan           # FX market closed
    p.loc[p.b.abs() > 1000, "b"] = np.nan                 # obviously bad prints
    return p


def five_min(p, FREQ="5min"):
    g = p.resample(FREQ)
    a = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum(), "vol": g.vol.sum(), "fxv": g.fxv.sum(),
                      "pl": np.log(g.p_usdt.mean()), "pg": np.log(g.fx_mid.mean())})
    a = a[a.nb >= 2]
    naive = a.index.tz_localize(None)
    a["month"] = naive.to_period("M").astype(str)
    a["bandper"] = a["month"]
    a["q"] = a.ofi / a.ofi.groupby(a.month).transform("std")
    a["dpl"] = 1e4 * a.pl.diff(); a["dpg"] = 1e4 * a.pg.diff()
    consec = a.index.to_series().diff() == pd.Timedelta(FREQ)
    a["y"] = a.b.diff(); a["x"] = a.b.shift(1)
    a = a[consec & a.y.notna() & a.x.notna() & a.q.notna()].copy()
    a["day"] = a.index.date
    ih = (a.index.hour + 3) % 24
    a["bh"] = ((ih >= 9) & (ih < 17)).astype(float)       # Istanbul business hours 09-17
    return a


def profile(y, x, q, step=1.0, TRIM=0.05):
    lo_, hi_ = np.percentile(x, [100 * TRIM, 100 * (1 - TRIM)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)     # S[i, j]: u = grid[i], l = grid[j]; inf where l > u
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[i], grid[j]


def add_bands(a, tag=""):
    bands = {}
    for per, d in a.groupby("bandper"):
        if len(d) < 500: continue
        u, l = profile(d.y.values, d.x.values, d.q.values)
        bands[per] = dict(l=float(l), u=float(u), n=int(len(d)))
        print(f"  band {tag} {per}: l={l:.0f} u={u:.0f} n={len(d)}  {time.time() - T_START:.0f}s", flush=True)
    a["l"] = a.bandper.map(lambda k: bands.get(k, {}).get("l", np.nan))
    a["u"] = a.bandper.map(lambda k: bands.get(k, {}).get("u", np.nan))
    a = a[a.l.notna()].copy()
    a["up"] = -np.maximum(a.x - a.u, 0)
    a["dn"] = np.maximum(a.l - a.x, 0)
    a["exc"] = np.maximum(a.x - a.u, 0) + np.maximum(a.l - a.x, 0)
    a["inband"] = (a.exc == 0).astype(float)
    return a, bands


# ------------------------------------------------------------------ estimators (rob_by_period / rob_size_dependence)
def ols(d, ycol, cols, fe="month"):
    """Within-FE OLS, day-clustered with the finite-sample correction of ols_fe (rob_by_period.ols)."""
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
    return dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), V=V, cols=list(cols), n=int(n),
                n_days=int(ng))


def lin(r, w):
    g = np.array([w.get(c, 0.0) for c in r["cols"]]); b = np.array([r["b"][c] for c in r["cols"]])
    return [float(g @ b), float(np.sqrt(g @ r["V"] @ g))]


def speeds(r, c0, c1):
    k0, k1 = r["b"][c0], r["b"][c1]
    inc = lin(r, {c1: 1})
    out = dict(closed=lin(r, {c0: 1}), incr=inc, open=lin(r, {c0: 1, c1: 1}),
               z=float(inc[0] / inc[1]), p1=float(st.norm.sf(inc[0] / inc[1])))
    if k0 > 0 and k0 + k1 > 0:
        ix = {c: i for i, c in enumerate(r["cols"])}
        g = np.zeros(len(r["cols"])); g[ix[c0]] = 1 / (k0 + k1) - 1 / k0; g[ix[c1]] = 1 / (k0 + k1)
        out["ratio"] = float((k0 + k1) / k0); out["se_logratio"] = float(np.sqrt(g @ r["V"] @ g))
    else:
        out["ratio"] = None; out["se_logratio"] = None
    return out


SPEC = ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"]


def col4(d, S="bh"):
    d = d.copy()
    for c in ["q", "up", "dn"]:
        d[c + "_bh"] = d[c] * d[S]
    r = ols(d, "y", SPEC)
    return dict(kp=speeds(r, "up", "up_bh"), km=speeds(r, "dn", "dn_bh"),
                beta=dict(closed=lin(r, {"q": 1}), open=lin(r, {"q": 1, "q_bh": 1})), n=r["n"], n_days=r["n_days"])


def fit(d, ycol, cols, fe="month"):
    """rob_size_dependence.fit: one-way FE (within), day-clustered s.e. with the finite-sample correction."""
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
    return dict(b=b, V=V, cols=list(cols), idx={c: i for i, c in enumerate(cols)}, n=int(n), ng=int(ng))


def lc(r, w):
    g = np.zeros(len(r["b"]))
    for k, v in w.items():
        if k not in r["idx"]:
            return None
        g[r["idx"][k]] += v
    return [float(g @ r["b"]), float(np.sqrt(g @ r["V"] @ g))]


SEGLAB = ["0-10", "10-30", "30-60", ">60"]


def binavg(d, fe="month", min_n=30):
    """rob_size_dependence.binavg (spec 1c); bin x state columns with no nonzero entry are dropped."""
    d = d.copy()
    d["e"] = np.maximum(d.x - d.u, 0.0); d["f"] = np.maximum(d.l - d.x, 0.0)
    d["q_bh"] = d.q * d.bh
    edges = [0, 10, 30, 60, np.inf]
    cols = ["q", "q_bh"]; cnt = {}
    for side, var, sign in [("ub", "e", -1.0), ("db", "f", 1.0)]:
        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            c = f"{side}{k + 1}"
            inb = (d[var] > lo) & (d[var] <= hi)
            d[c] = sign * d[var] * inb
            d[c + "_bh"] = d[c] * d.bh
            cnt[c] = dict(closed=int((inb & (d.bh == 0)).sum()), open=int((inb & (d.bh == 1)).sum()))
            cols += [cc for cc in (c, c + "_bh") if (d[cc] != 0).any()]
    r = fit(d, "y", cols, fe)
    res = {"n": r["n"]}
    for side, nm in [("above", "ub"), ("below", "db")]:
        segs = {}
        for k in range(4):
            c = f"{nm}{k + 1}"
            ok = cnt[c]["closed"] >= min_n and cnt[c]["open"] >= min_n
            segs[SEGLAB[k]] = dict(n=cnt[c], reported=bool(ok),
                                   closed=lc(r, {c: 1}) if ok else None,
                                   open=lc(r, {c: 1, c + "_bh": 1}) if ok else None,
                                   incr=lc(r, {c + "_bh": 1}) if ok else None)
        res[side] = segs
    return res


def describe(d):
    e = np.maximum(d.x - d.u, 0.0)
    o = dict(n=int(len(d)), n_days=int(d.day.nunique()), prem_mean=float(d.x.mean()), prem_sd=float(d.x.std()))
    for lab, m in [("all", np.ones(len(d), bool)), ("open", d.bh.values == 1), ("closed", d.bh.values == 0)]:
        o[lab] = dict(share_above_30=float((e[m] > 30).mean()), share_above_60=float((e[m] > 60).mean()),
                      n_above_30=int((e[m] > 30).sum()), n_above_60=int((e[m] > 60).sum()),
                      share_above=float((e[m] > 0).mean()), vol_mean_k=float(d.vol[m].mean() / 1e3), n=int(m.sum()))
    bw = (d.groupby("month").u.first() - d.groupby("month").l.first())
    o["band_width_median"] = float(bw.median()); o["band_width_mean"] = float(bw.mean())
    o["n_months_zero_width_band"] = int((bw == 0).sum())     # descriptive
    return o


# ================================================================== (0) replication of a5.parquet
a5 = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
a5["day"] = a5.index.date
if REPL:
    print("replication of a5 ...", flush=True)
    fxm, infom = load_fx_dir(os.path.join(RAW, "dukascopy"), use_ask=True)
    pm = build_minutes(os.path.join(RAW, "binance"), fxm, "2024-01-01", "2026-08-31")
    am, bm = add_bands(five_min(pm), "a5")
    common = am.index.intersection(a5.index)
    rep = dict(n_rebuilt=int(len(am)), n_a5=int(len(a5)), n_common=int(len(common)))
    for c in ["b", "y", "x", "q", "l", "u", "bh", "vol"]:
        rep[f"maxabs_{c}"] = float(np.nanmax(np.abs(am.loc[common, c].values - a5.loc[common, c].values)))
    rep["ok"] = bool(rep["n_rebuilt"] == rep["n_a5"] == rep["n_common"] and
                     max(rep[f"maxabs_{c}"] for c in ["b", "y", "x", "q", "l", "u", "bh"]) < 1e-6)
    OUT["replication_a5"] = rep
    print(rep, flush=True)
    assert rep["ok"], "construction does not reproduce a5.parquet"
    del pm, am
if os.environ.get("ONLY_REPL"):
    raise SystemExit

# ================================================================== (1) pre-sample panel
YEARS_PRE = [str(y) for y in range(int(os.environ.get("MINYEAR_PRE", 2021)), int(os.environ.get("MAXYEAR_PRE", 2023)) + 1)]
fx_files = {y: dict(BID=0, ASK=0, BID_nonempty=0, ASK_nonempty=0) for y in YEARS_PRE}
for f in glob.glob(os.path.join(PRE, "dukascopy", "USDTRY_*.bi5")):
    _, side, ds = os.path.basename(f)[:-4].split("_")
    if ds[:4] in fx_files:
        fx_files[ds[:4]][side] += 1; fx_files[ds[:4]][side + "_nonempty"] += int(os.path.getsize(f) > 0)
for y in YEARS_PRE:
    fx_files[y]["weekdays"] = int(np.sum(pd.bdate_range(f"{y}-01-01", f"{y}-12-31").size))
OUT["fx_file_coverage"] = fx_files
print("FX coverage", fx_files, flush=True)

fx_pre_ask, info_ask = load_fx_dir(os.path.join(PRE, "dukascopy"), use_ask=True, years=YEARS_PRE)
H = info_ask["half_bp"] / 1e4 if info_ask["n_minutes_with_ask"] > 0 else infom["half_bp"] / 1e4 if REPL else 0.5e-4
fx_pre_bid, info_bid = load_fx_dir(os.path.join(PRE, "dukascopy"), use_ask=False, half=H, years=YEARS_PRE)
OUT["fx_info"] = dict(primary_bid_based=info_bid, check_bid_ask=info_ask,
                      baseline_2024_2026=infom if REPL else None)
END_PRE = f"{YEARS_PRE[-1]}-12-31"
p_pre = build_minutes(os.path.join(PRE, "binance"), fx_pre_bid, f"{YEARS_PRE[0]}-01-01", END_PRE)
pre, bands_pre = add_bands(five_min(p_pre), "pre")
OUT["bands_pre"] = bands_pre
if "OUTDIR" not in os.environ:
    pre.drop(columns=["day"]).to_parquet(os.path.join(PRE, "a5_pre.parquet"))

# benchmark quality descriptives (minute level)
fxq = p_pre.copy()
hr = fxq.index.hour + fxq.index.minute / 60
fxq = fxq[(fxq.index.dayofweek < 5) & ~((hr >= 20.5) & (hr < 22.5))]
ihm = (fxq.index.hour + 3) % 24
fxq["bh"] = ((ihm >= 9) & (ihm < 17)).astype(float)
sp = fx_pre_ask.fx_spread.reindex(fxq.index)
fxq["sp"] = sp.values
FXQ = {}
for y in YEARS_PRE:
    d = fxq.loc[y]
    FXQ[y] = {s: dict(share_minutes_with_mid=float(d.fx_mid[d.bh == v].notna().mean()),
                      median_spread_bp=float(1e4 * d.sp[d.bh == v].median()) if d.sp[d.bh == v].notna().any() else None)
              for s, v in [("open", 1), ("closed", 0)]}
    FXQ[y]["usdtry_mean"] = float(d.fx_mid.mean())
OUT["benchmark_quality_pre"] = FXQ
del fxq, p_pre

# ================================================================== (2) by-year estimates
a5["year"] = a5.month.str[:4]; pre["year"] = pre.month.str[:4]
for d in (a5, pre):
    d["hol"] = np.isin(d.day, pd.to_datetime(HOL).date).astype(float)
    d["bh_eff"] = d.bh * (1 - d.hol)
PUB = json.load(open(os.path.join(TAB, "rob_by_period.json")))
BY = {}
for y in YEARS_PRE + ["2024", "2025", "2026"]:
    d = pre[pre.year == y] if y < "2024" else a5[a5.year == y]
    BY[y] = dict(col4=col4(d), col4_bh_eff=col4(d, "bh_eff"), describe=describe(d),
                 n_holidays=int(d.loc[d.hol == 1, "day"].nunique()))
    k = BY[y]["col4"]["kp"]
    BY[y]["present_5pct_one_sided"] = bool(k["z"] > 1.645)
    if y >= "2024":
        BY[y]["published_rob_by_period"] = PUB["by_period"][y]["a"]["kp"]
    print(y, "incr", np.round(k["incr"], 5), "closed", np.round(k["closed"], 5), "open", np.round(k["open"], 5),
          "z", round(k["z"], 2), flush=True)
OUT["by_year"] = BY
OUT["check_by_year_vs_published"] = {y: [BY[y]["col4"]["kp"]["incr"][0], PUB["by_period"][y]["a"]["kp"]["incr"][0]]
                                     for y in ["2024", "2025", "2026"]}
OUT["reading_rule"] = {y: ("present" if BY[y]["present_5pct_one_sided"] else "not present") for y in BY}

# pooled 2021-2023 and difference from pooled 2024-2025
pp = col4(pre); p45 = col4(a5[a5.year.isin(["2024", "2025"])])
dif = pp["kp"]["incr"][0] - p45["kp"]["incr"][0]; sdif = np.hypot(pp["kp"]["incr"][1], p45["kp"]["incr"][1])
OUT["pooled"] = dict(pre_2021_2023=pp, main_2024_2025=p45,
                     diff_incr_pre_minus_2024_25=dict(est=float(dif), se=float(sdif), z=float(dif / sdif),
                                                      p2=float(2 * st.norm.sf(abs(dif / sdif)))))
print("pooled pre", pp["kp"]["incr"], "2024-25", p45["kp"]["incr"], flush=True)

# ================================================================== (3) bin averages by year
BA = {}
for y in YEARS_PRE + ["2024", "2025", "2026"]:
    d = pre[pre.year == y] if y < "2024" else a5[a5.year == y]
    try:
        BA[y] = binavg(d)
    except np.linalg.LinAlgError as e_:
        BA[y] = dict(error=str(e_))
OUT["binavg_by_year"] = BA

# ================================================================== (4) check C1: bid/ask mid
C1R = {}
if info_ask["n_minutes_with_ask"] > 0:
    p_c1 = build_minutes(os.path.join(PRE, "binance"), fx_pre_ask, f"{YEARS_PRE[0]}-01-01", END_PRE)
    pre_c1, bands_c1 = add_bands(five_min(p_c1), "C1")
    del p_c1
    pre_c1["year"] = pre_c1.month.str[:4]
    for y in YEARS_PRE:
        d = pre_c1[pre_c1.year == y]
        C1R[y] = dict(col4=col4(d), describe=describe(d))
    C1R["pooled"] = col4(pre_c1)
    C1R["bands"] = bands_c1
OUT["check_C1_bid_ask_mid"] = C1R

# ================================================================== (5) rolling 6-month windows 2021-2026
cols_keep = ["y", "x", "q", "up", "dn", "bh", "month", "day", "l", "u", "vol"]
allp = pd.concat([pre[cols_keep], a5[cols_keep]]).sort_index()
months = sorted(allp.month.unique())
mi = {m: pd.Period(m, "M").ordinal for m in months}
ROLL = []
for i in range(len(months)):
    w = [m for m in months if 0 <= mi[m] - mi[months[i]] < 6]
    if pd.Period(months[i], "M") + 5 > pd.Period(months[-1], "M"): break
    d = allp[allp.month.isin(w)]
    r = col4(d); e = np.maximum(d.x - d.u, 0.0)
    ROLL.append(dict(start=w[0], end=(pd.Period(months[i], "M") + 5).strftime("%Y-%m"), months=len(w),
                     kp=r["kp"], n=r["n"], share_above_30=float((e > 30).mean()),
                     share_above_60=float((e > 60).mean())))
OUT["rolling_6m"] = ROLL
sig = [w for w in ROLL if w["kp"]["z"] > 1.645]
OUT["rolling_summary"] = dict(
    first_significant_window=(sig[0]["start"] + ".." + sig[0]["end"]) if sig else None,
    n_windows=len(ROLL),
    significant_by_end_year={yy: f"{sum(1 for w in ROLL if w['end'][:4] == yy and w['kp']['z'] > 1.645)} of "
                                 f"{sum(1 for w in ROLL if w['end'][:4] == yy)}"
                             for yy in sorted({w['end'][:4] for w in ROLL})})
pub_roll = {w["start"]: w["a"]["incr"][0] for w in PUB["rolling_6m"]}
OUT["check_rolling_vs_published_maxabs"] = float(max(abs(w["kp"]["incr"][0] - pub_roll[w["start"]])
                                                     for w in ROLL if w["start"] in pub_roll))
# diagnostic: the same rolling windows on the bid/ask-mid panel of check C1, windows inside
# 2021-01..2023-12 only.
ROLL_C1 = []
if C1R:
    mc = sorted(pre_c1.month.unique())
    for i in range(len(mc)):
        if pd.Period(mc[i], "M") + 5 > pd.Period(mc[-1], "M"): break
        w = [m for m in mc if 0 <= pd.Period(m, "M").ordinal - pd.Period(mc[i], "M").ordinal < 6]
        r = col4(pre_c1[pre_c1.month.isin(w)])
        ROLL_C1.append(dict(start=w[0], end=(pd.Period(mc[i], "M") + 5).strftime("%Y-%m"), kp=r["kp"], n=r["n"]))
OUT["posthoc_rolling_6m_C1"] = ROLL_C1
OUT["posthoc_rolling_C1_summary"] = dict(
    first_significant_window=next((w["start"] + ".." + w["end"] for w in ROLL_C1 if w["kp"]["z"] > 1.645), None),
    significant_by_end_year={yy: f"{sum(1 for w in ROLL_C1 if w['end'][:4] == yy and w['kp']['z'] > 1.645)} of "
                                 f"{sum(1 for w in ROLL_C1 if w['end'][:4] == yy)}"
                             for yy in sorted({w['end'][:4] for w in ROLL_C1})})
monthly = []
for m, d in allp.groupby("month"):
    e = np.maximum(d.x - d.u, 0.0)
    monthly.append(dict(month=m, share_above_30=float((e > 30).mean()), share_above_60=float((e > 60).mean()),
                        vol_mean_k=float(d.vol.mean() / 1e3), n=int(len(d))))
OUT["monthly"] = monthly

# ================================================================== (6) institutions by year (no causal claim)
INST = json.load(open(os.path.join(TAB, "rob_xc_inst_try.json")))
fast = INST["instant_systems"][0]["per_transaction_limit_history_TRY"]
usd = {y: FXQ[y]["usdtry_mean"] for y in YEARS_PRE}
usd.update({y: float(np.exp(a5[a5.year == y].pg).mean()) for y in ["2024", "2025", "2026"]})
OUT["institutions"] = dict(
    fast_limit_history=[dict(effective=f["effective"], limit_try=f["money_transfer_limit"],
                             note=f.get("note", "")[:200], url=f.get("url")) for f in fast],
    usdtry_mean=usd,
    by_year={
        "2021": "FAST (24/7 instant transfers) live from 8 Jan 2021: per-transfer limit TRY 50, TRY 1,000 from 29 Jan, "
                "TRY 2,000 from 4 Aug 2021 (about USD %.0f-%.0f at the 2021 mean rate). From 30 Apr 2021 payment and "
                "e-money institutions may not intermediate transfers to crypto platforms (CBRT regulation, Official "
                "Gazette 16 Apr 2021), so lira legs go through bank accounts. EFT 08:30-17:30 on weekdays." % (
                    1000 / usd.get("2021", np.nan), 2000 / usd.get("2021", np.nan)),
        "2022": "FAST limit TRY 5,000 from 21 Feb 2022 (about USD %.0f at the 2022 mean rate); later steps up to TRY "
                "50,000 before Apr 2024 are unverified. EFT hours unchanged." % (5000 / usd.get("2022", np.nan)),
        "2023": "FAST limit between TRY 5,000 and TRY 50,000 (steps unverified); payment and e-money institutions "
                "admitted to FAST in 2023. EFT hours unchanged.",
        "2024": "FAST limit TRY 50,000, TRY 100,000 from 4 Apr 2024 (about USD %.0f-%.0f). Redesigned CBRT payment "
                "systems from 23 Feb 2024, hours unchanged." % (50000 / usd.get("2024", np.nan), 100000 / usd.get("2024", np.nan)),
        "2025": "FAST limit TRY 100,000 (about USD %.0f). EFT hours unchanged." % (100000 / usd.get("2025", np.nan)),
        "2026": "FAST limit TRY 100,000 (about USD %.0f) until 25 Aug 2026, TRY 300,000 from 26 Aug 2026; Binance "
                "lira books restricted from 13 Nov 2025. EFT hours unchanged." % (100000 / usd.get("2026", np.nan))},
    eft_hours="Weekdays except public holidays 08:30-17:30, 13:00 on half days (CBRT; PFMI disclosure 29 Dec 2021 "
              "section 2.9; unchanged through 2026 per tables/rob_xc_inst_try.json).")

# ================================================================== holiday cross-check against the CBRT calendar
calf = os.path.join(PRE, "tcmb_indicative_rate_calendar_2021_2023.csv")
if os.path.exists(calf):
    cal = pd.read_csv(calf, dtype=str)
    no_rate = set(cal.loc[cal.status == "404", "day"]); ok_rate = set(cal.loc[cal.status == "200", "day"])
    listed = set(HOL_PRE) | set(HALF_PRE)
    OUT["holiday_check"] = dict(source=HOL_SOURCE, n_weekdays_checked=int(len(cal)),
                                hol_or_half_with_rate=sorted(listed & ok_rate),
                                no_rate_not_listed=sorted(no_rate - listed),
                                other_status=sorted(cal.loc[~cal.status.isin(["200", "404"]), "day"]))
else:
    OUT["holiday_check"] = dict(source=HOL_SOURCE, note="CBRT calendar not downloaded")
OUT["hol_pre"] = HOL_PRE; OUT["half_pre"] = HALF_PRE

# ================================================================== table
YRS = YEARS_PRE + ["2024", "2025", "2026"]


def f4(v, nd=4):
    return "" if v is None else f"${v[0]:.{nd}f}$"


def s4(v, nd=4):
    return "" if v is None else f"$({v[1]:.{nd}f})$"


with open(os.path.join(TAB_OUT, f"rob_{NAME}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{l" + "c" * len(YRS) + "}\n\\toprule\n & " + " & ".join(
        y if y != "2026" else "2026 (Jan--Aug)" for y in YRS) + " \\\\\n\\midrule\n")
    rows = [("$\\kappa^+$, banks closed", lambda y: BY[y]["col4"]["kp"]["closed"]),
            ("$\\kappa^+$, banks open", lambda y: BY[y]["col4"]["kp"]["open"]),
            ("Business-hours increment", lambda y: BY[y]["col4"]["kp"]["incr"]),
            ("$\\kappa^-$, banks closed", lambda y: BY[y]["col4"]["km"]["closed"]),
            ("$\\kappa^-$ increment", lambda y: BY[y]["col4"]["km"]["incr"])]
    for lab, fn in rows:
        fh.write(lab + " & " + " & ".join(f4(fn(y)) for y in YRS) + " \\\\\n & " +
                 " & ".join(s4(fn(y)) for y in YRS) + " \\\\\n")
    fh.write("One-sided $p$ (increment $>0$) & " + " & ".join(f"{BY[y]['col4']['kp']['p1']:.3f}" for y in YRS) + " \\\\\n")
    fh.write("Increment, holidays closed & " + " & ".join(f4(BY[y]["col4_bh_eff"]["kp"]["incr"]) for y in YRS) +
             " \\\\\n & " + " & ".join(s4(BY[y]["col4_bh_eff"]["kp"]["incr"]) for y in YRS) + " \\\\\n")
    if C1R:
        fh.write("Increment, bid/ask mid & " + " & ".join(f4(C1R[y]["col4"]["kp"]["incr"]) if y in C1R else "--"
                                                        for y in YRS) + " \\\\\n & " +
                 " & ".join(s4(C1R[y]["col4"]["kp"]["incr"]) if y in C1R else "" for y in YRS) + " \\\\\n")
    fh.write("\\midrule\n")

    def bav(y, k):
        s = BA[y].get("above", {}).get(">60", {})
        return s.get(k) if s.get("reported") else None
    for lab, k in [("$>$60 bp above: speed, closed", "closed"), ("$>$60 bp above: speed, open", "open"),
                   ("$>$60 bp above: open $-$ closed", "incr")]:
        fh.write(lab + " & " + " & ".join(f4(bav(y, k)) if bav(y, k) else "--" for y in YRS) + " \\\\\n & " +
                 " & ".join(s4(bav(y, k)) for y in YRS) + " \\\\\n")
    fh.write("\\midrule\n")
    D = {y: BY[y]["describe"] for y in YRS}
    fh.write("Share $>$30 bp above band (\\%) & " + " & ".join(f"{100 * D[y]['all']['share_above_30']:.1f}" for y in YRS) + " \\\\\n")
    fh.write("Share $>$60 bp above band (\\%) & " + " & ".join(f"{100 * D[y]['all']['share_above_60']:.1f}" for y in YRS) + " \\\\\n")
    fh.write("Mean volume (k USDT per 5 min) & " + " & ".join(f"{D[y]['all']['vol_mean_k']:.0f}" for y in YRS) + " \\\\\n")
    fh.write("Median band width (bp) & " + " & ".join(f"{D[y]['band_width_median']:.0f}" for y in YRS) + " \\\\\n")
    FASTLAB = {"2021": "0.05k--2k", "2022": "5k--50k$^\\dagger$", "2023": "5k--50k$^\\dagger$", "2024": "50k--100k",
               "2025": "100k", "2026": "100k"}
    fh.write("FAST limit (TRY) & " + " & ".join(FASTLAB[y] for y in YRS) + " \\\\\n")
    fh.write("$N$ & " + " & ".join(f"{D[y]['n']:,}" for y in YRS) + " \\\\\n")
    fh.write("Days & " + " & ".join(f"{D[y]['n_days']}" for y in YRS) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ================================================================== figure
fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
tt = np.array([pd.Period(w["end"], "M").to_timestamp(how="end") for w in ROLL])
inc = np.array([w["kp"]["incr"][0] for w in ROLL]); ise = np.array([w["kp"]["incr"][1] for w in ROLL])
clo = np.array([w["kp"]["closed"][0] for w in ROLL]); opn = np.array([w["kp"]["open"][0] for w in ROLL])
A = ax[0]
A.fill_between(tt, inc - 1.96 * ise, inc + 1.96 * ise, color=C1, alpha=0.15, lw=0)
A.plot(tt, inc, color=C1, lw=1.4, label="business-hours increment (95% CI)")
A.plot(tt, opn, color=C2, lw=0.9, ls="--", label=r"$\kappa^+$ banks open")
A.plot(tt, clo, color=C3, lw=0.9, ls=":", label=r"$\kappa^+$ banks closed")
if ROLL_C1:
    tc = np.array([pd.Period(w["end"], "M").to_timestamp(how="end") for w in ROLL_C1])
    A.plot(tc, [w["kp"]["incr"][0] for w in ROLL_C1], color=C1, lw=0.8, ls="-.",
           label="increment, bid/ask mid (2021-23)")
A.axhline(0, color="k", lw=0.4)
A.axvline(pd.Timestamp("2024-01-01"), color="k", lw=0.6, ls="-.")
A.set_ylim(min(-0.01, A.get_ylim()[0]), 0.046)
A.text(pd.Timestamp("2024-01-20"), 0.031, "paper's sample", fontsize=7.5)
A.set_ylabel("per five minutes")
A.set_title("(a) Speed above the band, six-month rolling windows (plotted at window end)", fontsize=9.5)
A.legend(frameon=False, fontsize=7.2, loc="upper left", ncol=2)
B = ax[1]
mm = pd.DataFrame(monthly)
tm = pd.PeriodIndex(mm.month, freq="M").to_timestamp(how="start") + pd.Timedelta(days=14)
B.bar(tm, 100 * mm.share_above_30, width=25, color=C1, alpha=0.45, label="more than 30 bp above the band")
B.bar(tm, 100 * mm.share_above_60, width=25, color=C2, alpha=0.85, label="more than 60 bp above the band")
B.axvline(pd.Timestamp("2024-01-01"), color="k", lw=0.6, ls="-.")
B.set_ylim(0, 118); B.set_yticks([0, 20, 40, 60, 80, 100])
for dte, lab in [("2021-01-08", "FAST launch"), ("2024-04-04", "FAST limit TRY 100k"),
                 ("2025-11-13", "Binance TR only")]:
    B.axvline(pd.Timestamp(dte), color=C3, lw=0.5, ls=":")
    B.text(pd.Timestamp(dte), 68, " " + lab, fontsize=6.5, color=C3, va="bottom", rotation=90)
B.set_ylabel("% of five-minute intervals")
B.set_title("(b) Large dislocations above the monthly band, by month", fontsize=9.5)
B.legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 1.02))
B.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
B.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); B.tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig(os.path.join(FIG_OUT, f"rob_{NAME}.pdf")); plt.close(fig)

OUT["runtime_s"] = float(time.time() - T_START)
json.dump(OUT, open(os.path.join(TAB_OUT, f"rob_{NAME}.json"), "w"), indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ["reading_rule", "rolling_summary", "check_by_year_vs_published",
                                      "check_rolling_vs_published_maxabs"]}, indent=1, default=float))
print("done", f"{time.time() - T_START:.0f}s")
