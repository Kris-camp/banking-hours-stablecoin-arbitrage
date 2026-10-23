"""rob_btcturk_venue: is the banking-hours result specific to one venue (Binance)?

All main estimates use Binance.com USDT/TRY, whereas Turkish onshore exchanges such as BtcTurk are
directly integrated with Turkish banks. This script compares the two venues where both can be observed.

PART A - DATA AVAILABILITY (re-probed at run time and recorded in the JSON).
The public BtcTurk archive at graph-api.btcturk.com ends at 2020-12-31 23:59 UTC:
  * /v1/klines/history (TradingView UDF; supported resolutions 1,5,15,30,60,240,1D,1W,1M) returns
    {"s": null, "nb": 1609459140} (next bar = 2020-12-31 23:59 UTC) for every window after 2020, at
    every resolution, for USDTTRY, BTCTRY, ETHTRY and BTCUSDT; pairs listed after 2020 (USDCTRY,
    AVAXTRY, SOLTRY) return nb = 0 (no data at all). Millisecond timestamps are rejected
    ("invalidTimestampValue"); USDT_TRY is "pairNotFound"; countback, exchange=, lower-case symbols
    and /v1/klines change nothing; /v2/... is "UnsupportedApiVersion".
  * /v1/ohlcs?pair=USDTTRY (daily) runs 2019-01-01 to 2020-12-31 and returns [] for any later window.
  * Other paths (/v1/ohlc, /v1/candles, /v1/trades, /v1/ticker, /v1/klines/symbols, /v1/klines/time,
    swagger) are 404.
No BtcTurk minute price is therefore available for the paper's sample (2024-2026). The script re-runs the
key probes, and records whether other BtcTurk hosts and two price aggregators respond, in the JSON.

PART B - OUT-OF-PERIOD VENUE COMPARISON, DECEMBER 2019 - DECEMBER 2020.
The archive contains one-minute BtcTurk USDT/TRY klines through 2020, and Binance's public archive has
USDT/TRY one-minute klines from December 2019, so both venues and the Dukascopy interbank mid can be
observed from 2019-12-01 to 2020-12-31. This window predates the FAST 24/7 instant-payment system
(launched January 2021), the 2021-23 lira crisis and the 2024-26 regime, and BtcTurk USDT/TRY was much
thinner then.
 1. Data. BtcTurk: /v1/klines/history, resolution 1, 7-day requests (OHLCV, volume in USDT; no taker
    side, no quote volume). Binance: data.binance.vision USDTTRY 1m klines. Interbank: Dukascopy USD/TRY
    one-minute BID candles, minutes with tick volume > 0, mid = bid x (1 + half median spread) as in
    build_panel.load_fx (BID only, as in the main panel), forward-filled at most 2 minutes as in
    empirics_main.py. All raw files go to data_raw/btcturk_venue/ (separate from the main raw folders
    so that build_panel.py is unaffected).
 2. Timestamp check. BtcTurk bar times are taken as UTC bar-open times. Verified by the offset in
    [-240, +240] minutes that maximizes the correlation of BtcTurk and Binance one-minute log price
    changes (expected within +-5 minutes of 0, allowing for a short lead-lag between venues; a
    time-zone error would show up at +-180); if |offset| > 5 the script stops.
 3. Premium. b_v = 1e4 log(P_v / FX mid), P_v = one-minute CLOSE on both venues (BtcTurk has no VWAP;
    Binance close is used too so the venues are treated alike). Binance klines include zero-volume
    minutes whose close is the last trade carried forward, while the BtcTurk feed has bars only for
    traded minutes, so the Binance close is set to missing when v = 0 and both premia are built from
    traded minutes only (as the paper's VWAP premium is). Same filters as empirics_main.py:
    weekends and 20:30-22:30 UTC dropped, |b| > 1000 bp dropped. Five-minute mean, at least 2 minute
    observations per interval, consecutive intervals only, y = delta b, x = lagged b.
 4. Primary sample: COMMON five-minute intervals in which both venues have a valid y and x
    (identical intervals, so venue differences are not sample composition). Secondary: each venue's
    own full sample.
 5. Primary specification S1 (no order-flow term, because BtcTurk klines carry no trade side):
       y = a_m + k+ up + k- dn + d+ up*BH + d- dn*BH + e,
    up = -(x-u_m)+, dn = (l_m-x)+, monthly bands profiled per venue with model._ssr_grid (q = 0),
    5% trimming, 1 bp grid, months with >= 500 intervals; BH = 09:00-17:00 Istanbul on non-holiday
    weekdays (Turkish weekday public holidays of Dec 2019 - Dec 2020 listed below coded as closed);
    month fixed effects; day-clustered s.e. with the finite-sample correction of empirics_main.ols_fe.
    Secondary S2 (Binance only): adds its own standardized OFI and OFI*BH, bands profiled with OFI,
    to show what dropping the flow term does on the venue where it is observed.
    Secondary S3: S1 at fifteen-minute frequency (min 4 minute observations), same rules.
 6. Tests. (i) d+ > 0 and d- on each venue; open/closed ratio k+_open/k+_closed with delta-method s.e.
    (ii) Venue difference d+_BtcTurk - d+_Binance (and for d-, k+, k-), two-sided, from a stacked
    regression on the common sample with venue x month FE, venue-specific slopes and day clustering
    (both venues share days, so the covariance across venues is accounted for).
 7. Descriptives: minutes with a trade, mean/s.d. of the premium, correlation of the two venues'
    five-minute premia, mean and s.d. of the BtcTurk - Binance price gap, band widths.
 8. Diagnostic: the paper's column-(4) specification (Binance minute VWAP, own standardized OFI and
    OFI*BH, bands profiled with OFI, five-minute frequency, Binance's own sample) re-estimated on
    Dec 2019 - Dec 2020, to check that the 2020 estimates are not an artefact of last-trade prices in a
    thin market.
Outputs: tables/rob_btcturk_venue.json, tables/rob_btcturk_venue.tex.
Env: SKIP_PROBE (0), NTHREAD_DL (8), BINANCE_STALE (0; 1 = use every Binance close, including the
     carried-forward zero-volume minutes, and write rob_btcturk_venue_stale.{json,tex}, for comparison only).
"""
import os, sys, io, json, time, glob, lzma, zipfile, random, datetime as dt, urllib.request, urllib.error
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import _ssr_grid

RAW, TAB = os.path.join(HERE, "..", "data_raw"), os.path.join(HERE, "..", "tables")
DD = os.path.join(RAW, "btcturk_venue")
for s in ("btcturk", "binance", "dukascopy"):
    os.makedirs(os.path.join(DD, s), exist_ok=True)
GA = "https://graph-api.btcturk.com"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
D0, D1 = dt.date(2019, 12, 1), dt.date(2020, 12, 31)
# Turkish public holidays on weekdays, Dec 2019 - Dec 2020 (banks and EFT closed)
HOL = ["2020-01-01", "2020-04-23", "2020-05-01", "2020-05-19", "2020-05-25", "2020-05-26",
       "2020-07-15", "2020-07-31", "2020-08-03", "2020-10-29"]
OUT = {}
SUF = "_stale" if os.environ.get("BINANCE_STALE", "0") == "1" else ""


def http(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return None, str(e).encode()


# ------------------------------------------------------------------ PART A: probes
def probe():
    now = int(time.time()); wk = now - 7 * 86400
    P = [("USDTTRY 1m, last 7 days", f"/v1/klines/history?symbol=USDTTRY&resolution=1&from={wk}&to={now}"),
         ("USDTTRY 60, last 7 days", f"/v1/klines/history?symbol=USDTTRY&resolution=60&from={wk}&to={now}"),
         ("USDTTRY 1D, 2024-2026", f"/v1/klines/history?symbol=USDTTRY&resolution=1D&from=1704067200&to={now}"),
         ("USDTTRY 60, Jan 2024", "/v1/klines/history?symbol=USDTTRY&resolution=60&from=1704067200&to=1706745600"),
         ("USDTTRY 60, Jan 2021", "/v1/klines/history?symbol=USDTTRY&resolution=60&from=1609459200&to=1612137600"),
         ("USDTTRY 60, ms timestamps", f"/v1/klines/history?symbol=USDTTRY&resolution=60&from={wk}000&to={now}000"),
         ("USDTTRY 60, countback", f"/v1/klines/history?symbol=USDTTRY&resolution=60&to={now}&countback=500"),
         ("BTCTRY 60, Jan 2024", "/v1/klines/history?symbol=BTCTRY&resolution=60&from=1704067200&to=1706745600"),
         ("USDCTRY 60, Jan 2024 (pair listed after 2020)",
          "/v1/klines/history?symbol=USDCTRY&resolution=60&from=1704067200&to=1706745600"),
         ("USDTTRY 1m, last day of 2020", "/v1/klines/history?symbol=USDTTRY&resolution=1&from=1609372800&to=1609459200"),
         ("daily ohlcs, 2024-2026", f"/v1/ohlcs?pair=USDTTRY&from=1704067200&to={now}"),
         ("daily ohlcs, full archive", "/v1/ohlcs?pair=USDTTRY"),
         ("v2 klines", f"/v2/klines/history?symbol=USDTTRY&resolution=60&from={wk}&to={now}"),
         ("UDF config", "/v1/klines/config")]
    rec = []
    for lab, path in P:
        code, body = http(GA + path)
        s = dict(probe=lab, path=path, http=code)
        try:
            j = json.loads(body)
            if isinstance(j, dict) and "t" in j:
                s.update(status=j.get("s"), nb=j.get("nb"), n_bars=len(j["t"]))
                if j["t"]:
                    s.update(first=str(pd.Timestamp(j["t"][0], unit="s")), last=str(pd.Timestamp(j["t"][-1], unit="s")))
            elif isinstance(j, list):
                s.update(n_rows=len(j))
                if j:
                    s.update(first=str(pd.Timestamp(j[0]["time"], unit="s").date()),
                             last=str(pd.Timestamp(j[-1]["time"], unit="s").date()))
            else:
                s.update(body=json.dumps(j)[:200])
        except Exception:
            s.update(body=body[:200].decode(errors="replace"))
        rec.append(s)
        time.sleep(0.3)
    hosts = {}
    for h in ["api.btcturk.com", "pro.btcturk.com", "www.btcturk.com", "ws-feed-pro.btcturk.com",
              "min-api.cryptocompare.com", "api.coingecko.com"]:
        code, body = http(f"https://{h}/", timeout=15)
        hosts[h] = code if code else body[:120].decode(errors="replace")
    return rec, hosts


# ------------------------------------------------------------------ PART B: downloads (cached)
def dl_btcturk():
    t0 = int(pd.Timestamp(D0, tz="UTC").timestamp()); t1 = int(pd.Timestamp(D1 + dt.timedelta(1), tz="UTC").timestamp())
    parts = []
    for f in range(t0, t1, 7 * 86400):
        fn = os.path.join(DD, "btcturk", f"USDTTRY_1m_{f}.json")
        if not os.path.exists(fn):
            for k in range(6):
                code, body = http(f"{GA}/v1/klines/history?symbol=USDTTRY&resolution=1&from={f}&to={min(f + 7 * 86400 - 60, t1)}")
                if code == 200:
                    open(fn, "wb").write(body); break
                time.sleep(2 ** k)
            else:
                raise RuntimeError(f"BtcTurk chunk {f} failed")
            time.sleep(0.5)
        j = json.load(open(fn))
        if j.get("t"):
            parts.append(pd.DataFrame({k: j[k] for k in ["t", "o", "h", "l", "c", "v"]}))
    d = pd.concat(parts).drop_duplicates("t").sort_values("t")
    d.index = pd.to_datetime(d.t, unit="s", utc=True)
    return d[["o", "h", "l", "c", "v"]].astype(float)


COLS = ["open_time", "o", "h", "l", "c", "v", "close_time", "qv", "n", "tbv", "tbqv", "ign"]


def dl_binance():
    out = []
    for ym in pd.period_range(D0, D1, freq="M").astype(str):
        fn = os.path.join(DD, "binance", f"USDTTRY-1m-{ym}.csv")
        if not os.path.exists(fn):
            code, body = http(f"https://data.binance.vision/data/spot/monthly/klines/USDTTRY/1m/USDTTRY-1m-{ym}.zip", 120)
            if code != 200:
                print("binance", ym, code); continue
            z = zipfile.ZipFile(io.BytesIO(body)); open(fn, "wb").write(z.read(z.namelist()[0]))
        d = pd.read_csv(fn, header=None, names=COLS)
        t = d.open_time.astype("int64").values; t = np.where(t > 1e14, t // 1000, t)
        d.index = pd.to_datetime(t, unit="ms", utc=True)
        out.append(d)
    d = pd.concat(out); d = d[~d.index.duplicated()]
    vwap = np.where(d.v > 0, d.qv / d.v.replace(0, np.nan), np.nan)
    return pd.DataFrame({"c": d.c.astype(float), "v": d.v.astype(float), "ofi": 2 * d.tbv - d.v, "vwap": vwap},
                        index=d.index)


def _duk(day):
    fn = os.path.join(DD, "dukascopy", f"USDTRY_BID_{day:%Y%m%d}.bi5")
    if os.path.exists(fn): return
    url = f"https://datafeed.dukascopy.com/datafeed/USDTRY/{day.year}/{day.month-1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    for k in range(30):
        code, body = http(url, 30)
        if code == 200: open(fn, "wb").write(body); return
        if code == 404: open(fn, "wb").close(); return
        time.sleep(1 + 2 * random.random())
    raise RuntimeError(f"dukascopy {day} failed")


def dl_fx():
    from concurrent.futures import ThreadPoolExecutor
    days = [D0 + dt.timedelta(n) for n in range((D1 - D0).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    with ThreadPoolExecutor(int(os.environ.get("NTHREAD_DL", 8))) as ex:
        list(ex.map(_duk, days))
    rows = []
    for f in sorted(glob.glob(os.path.join(DD, "dukascopy", "USDTRY_BID_*.bi5"))):
        if os.path.getsize(f) == 0: continue
        raw = lzma.decompress(open(f, "rb").read())
        day = pd.Timestamp(os.path.basename(f).split("_")[2][:8], tz="UTC")
        a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6)
        vol = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5]
        rows.append(pd.DataFrame({"bid": a[:, 2] / 1e5, "fxv": vol}, index=day + pd.to_timedelta(a[:, 0].astype("int64"), unit="s")))
    m = pd.concat(rows); m = m[~m.index.duplicated()]
    m = m[m.fxv > 0]
    m["fx_mid"] = m.bid * (1 + 0.5 * 1e-4)   # build_panel.load_fx with BID only: half of the 1e-4 default spread
    return m[["fx_mid", "fxv"]]


# ------------------------------------------------------------------ estimation helpers
def ols_fe(d, ycol, cols, fe="month"):
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    gm = d.groupby(fe)
    Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), V=pd.DataFrame(V, cols, cols), n=int(n),
                n_days=int(ng))


def profile(y, x, q, trim=0.05, step=1.0):
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]       # l, u


def panel(minute_b, freq, minobs, q=None):
    """minute_b: Series of minute premium (NaN where filtered). Returns interval panel."""
    g = minute_b.resample(freq)
    a = pd.DataFrame({"b": g.mean(), "nb": g.count()})
    if q is not None:
        a["ofi"] = q.resample(freq).sum().reindex(a.index)
    a = a[a.nb >= minobs]
    naive = a.index.tz_localize(None)
    a["month"] = naive.to_period("M").astype(str)
    consec = a.index.to_series().diff() == pd.Timedelta(freq)
    a["y"] = a.b.diff(); a["x"] = a.b.shift(1)
    a = a[consec & a.y.notna() & a.x.notna()].copy()
    if q is not None:
        a["q"] = a.ofi / a.ofi.groupby(a.month).transform("std")
    a["day"] = a.index.date
    ih = (a.index.hour + 3) % 24
    hol = np.isin(a.day, pd.to_datetime(HOL).date)
    a["bh"] = (((ih >= 9) & (ih < 17)) & ~hol).astype(float)
    return a


def add_bands(a, use_q=False, minn=500):
    bands = {}
    for m, d in a.groupby("month"):
        if len(d) < minn: continue
        qq = d.q.values if use_q else np.zeros(len(d))
        bands[m] = profile(d.y.values, d.x.values, qq)
    a = a[a.month.isin(list(bands))].copy()
    a["l"] = a.month.map(lambda m: bands[m][0]); a["u"] = a.month.map(lambda m: bands[m][1])
    a["up"] = -np.maximum(a.x - a.u, 0); a["dn"] = np.maximum(a.l - a.x, 0)
    for c in ["up", "dn"] + (["q"] if use_q else []):
        a[c + "_bh"] = a[c] * a.bh
    return a, {m: dict(l=float(l), u=float(u)) for m, (l, u) in bands.items()}


def summarize(r):
    b, se, V = r["b"], r["se"], r["V"]
    o = {k: [float(b[k]), float(se[k])] for k in b}
    for s in ["up", "dn"]:
        kc, ko = b[s], b[s] + b[s + "_bh"]
        vo = V.loc[s, s] + V.loc[s + "_bh", s + "_bh"] + 2 * V.loc[s, s + "_bh"]
        o[f"{s}_open"] = [float(ko), float(np.sqrt(vo))]
        # ratio open/closed, delta method: R = (kc+d)/kc
        g = np.array([-b[s + "_bh"] / kc ** 2, 1 / kc])
        Vs = V.loc[[s, s + "_bh"], [s, s + "_bh"]].values
        o[f"{s}_ratio_open_closed"] = [float(ko / kc), float(np.sqrt(g @ Vs @ g))]
    o["n"] = r["n"]; o["n_days"] = r["n_days"]
    return o


def fit_S1(a):
    a2, bands = add_bands(a)
    r = ols_fe(a2, "y", ["up", "dn", "up_bh", "dn_bh"])
    w = [v["u"] - v["l"] for v in bands.values()]
    return a2, r, dict(coef=summarize(r), bands=bands, band_width_median=float(np.median(w)),
                       share_inband=float(((a2.x >= a2.l) & (a2.x <= a2.u)).mean()))


def stacked(aT, aB):
    """Venue-specific slopes, venue x month FE, day clustering; returns BtcTurk - Binance differences."""
    cols = ["up", "dn", "up_bh", "dn_bh"]
    T = aT[["y", "month", "day"] + cols].copy(); B = aB[["y", "month", "day"] + cols].copy()
    for c in cols:
        T[c + "_T"] = T[c]; T[c + "_B"] = 0.0; B[c + "_T"] = 0.0; B[c + "_B"] = B[c]
    T["vm"] = "T" + T.month; B["vm"] = "B" + B.month
    S = pd.concat([T, B])
    xc = [c + "_T" for c in cols] + [c + "_B" for c in cols]
    r = ols_fe(S, "y", xc, fe="vm")
    out = {}
    for c in cols:
        g = pd.Series(0.0, index=xc); g[c + "_T"] = 1; g[c + "_B"] = -1
        out[c + "_diff_T_minus_B"] = [float(r["b"][c + "_T"] - r["b"][c + "_B"]), float(np.sqrt(g @ r["V"] @ g))]
    for s in ["up", "dn"]:
        g = pd.Series(0.0, index=xc); g[[s + "_T", s + "_bh_T"]] = 1; g[[s + "_B", s + "_bh_B"]] = -1
        out[s + "_open_diff_T_minus_B"] = [float(g @ pd.Series(r["b"])[xc]), float(np.sqrt(g @ r["V"] @ g))]
    out["n"] = r["n"]
    return out


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    if os.environ.get("SKIP_PROBE", "0") != "1":
        rec, hosts = probe()
        OUT["probe"] = rec; OUT["hosts"] = hosts
        print(json.dumps(rec, indent=1)); print(hosts)
    bt = dl_btcturk(); bn = dl_binance(); fx = dl_fx()
    print("btcturk", bt.index.min(), bt.index.max(), len(bt))
    print("binance", bn.index.min(), bn.index.max(), len(bn))
    print("fx", fx.index.min(), fx.index.max(), len(fx))
    s0 = max(bn.index.min(), bt.index.min(), fx.index.min()).floor("D")
    s1 = pd.Timestamp(D1 + dt.timedelta(1), tz="UTC")
    bt, bn, fx = bt.loc[s0:s1], bn.loc[s0:s1], fx.loc[s0:s1]

    # timestamp check: offset maximizing corr of 1-minute log changes (BtcTurk vs Binance)
    full = pd.date_range(s0, s1, freq="1min", inclusive="left")
    rT = np.log(bt.c.reindex(full).ffill(limit=5)).diff()
    rB = np.log(bn.c.reindex(full).ffill(limit=5)).diff()
    offs = list(range(-240, 241, 1))
    cc = {k: rT.shift(-k).corr(rB) for k in offs}
    best = max(cc, key=lambda k: cc[k])
    OUT["timestamp_check"] = dict(best_offset_min=int(best), corr_at_0=float(cc[0]), corr_at_best=float(cc[best]),
                                  corr_at_m180=float(cc[-180]), corr_at_p180=float(cc[180]))
    print(OUT["timestamp_check"])
    if abs(best) > 5:
        json.dump(OUT, open(os.path.join(TAB, f"rob_btcturk_venue{SUF}.json"), "w"), indent=1)
        raise SystemExit("timestamp offset beyond 5 minutes; stopping")

    # minute premia
    fxm = fx.fx_mid.reindex(full, method="ffill", limit=2)
    def prem(c):
        b = 1e4 * np.log(c.reindex(full) / fxm)
        hr = full.hour + full.minute / 60
        b[(hr >= 20.5) & (hr < 22.5)] = np.nan
        b[full.dayofweek >= 5] = np.nan
        b[b.abs() > 1000] = np.nan
        return b
    # Binance klines contain a bar for every minute (54% have zero volume, with the close carried forward
    # from the last trade without limit), whereas the BtcTurk UDF feed returns bars only for minutes with a
    # trade. Binance closes are therefore used only in minutes with v > 0, as in the paper's traded-minute
    # premium (VWAP is NaN when v = 0). BINANCE_STALE=1 uses every Binance close, for comparison.
    if os.environ.get("BINANCE_STALE", "0") == "1":
        bB_px = bn.c
    else:
        bB_px = bn.c.where(bn.v > 0)
    bT, bB = prem(bt.c), prem(bB_px)
    wk = (full.dayofweek < 5)
    D = dict(minutes_weekday=int(wk.sum()),
             btcturk_minutes_with_trade=int(bt.c.reindex(full)[wk].notna().sum()),
             binance_minutes_with_trade=int((bn.v.reindex(full)[wk] > 0).sum()),
             btcturk_valid_premium_minutes=int(bT.notna().sum()), binance_valid_premium_minutes=int(bB.notna().sum()),
             btcturk_median_daily_volume_usdt=float(bt.v.resample("D").sum().loc[lambda s: s > 0].median()),
             binance_median_daily_volume_usdt=float(bn.v.resample("D").sum().loc[lambda s: s > 0].median()),
             sample_start=str(s0.date()), sample_end=str(D1))
    print(D)

    RES = {}
    for tag, freq, minobs in [("5min", "5min", 2), ("15min", "15min", 4)]:
        aT = panel(bT, freq, minobs); aB = panel(bB, freq, minobs, q=bn.ofi.reindex(full))
        com = aT.index.intersection(aB.index)
        R = {}
        cT, cB = aT.loc[com].copy(), aB.loc[com].copy()
        R["n_common"] = int(len(com)); R["n_btcturk_own"] = int(len(aT)); R["n_binance_own"] = int(len(aB))
        R["desc"] = dict(mean_b_btcturk=float(cT.b.mean()), sd_b_btcturk=float(cT.b.std()),
                         mean_b_binance=float(cB.b.mean()), sd_b_binance=float(cB.b.std()),
                         corr_b=float(cT.b.corr(cB.b)), corr_dy=float(cT.y.corr(cB.y)),
                         gap_T_minus_B_mean=float((cT.b - cB.b).mean()), gap_T_minus_B_sd=float((cT.b - cB.b).std()),
                         share_bh=float(cT.bh.mean()))
        fT, rT_, R["S1_btcturk_common"] = fit_S1(cT)
        fB, rB_, R["S1_binance_common"] = fit_S1(cB)
        R["S1_diff_common"] = stacked(fT, fB)
        _, _, R["S1_btcturk_own"] = fit_S1(aT)
        _, _, R["S1_binance_own"] = fit_S1(aB)
        if tag == "5min":
            a2, bands = add_bands(cB.copy(), use_q=True)
            r = ols_fe(a2, "y", ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
            R["S2_binance_common_with_ofi"] = dict(coef=summarize(r), bands=bands)
        RES[tag] = R
        print(tag, json.dumps({k: (v["coef"] if isinstance(v, dict) and "coef" in v else v) for k, v in R.items()
                               if k != "desc"}, indent=0, default=str)[:4000])
        print(R["desc"])
    # Diagnostic: the paper's column-(4) specification on Binance 2020 (VWAP, own OFI, own sample)
    bV = prem(pd.Series(bn.vwap, index=bn.index))
    aV = panel(bV, "5min", 2, q=bn.ofi.reindex(full))
    a2, bandsV = add_bands(aV, use_q=True)
    r = ols_fe(a2, "y", ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
    RES["posthoc_binance_vwap_ofi_own_5min"] = dict(coef=summarize(r), bands=bandsV,
                                                    band_width_median=float(np.median([v["u"] - v["l"] for v in bandsV.values()])))
    print("diagnostic: Binance VWAP, own OFI, 5 min", RES["posthoc_binance_vwap_ofi_own_5min"]["coef"])
    OUT["descriptives"] = D; OUT["results"] = RES
    OUT["binance_close_minutes"] = "all klines (stale closes kept)" if SUF else "traded minutes only (v > 0)"
    json.dump(OUT, open(os.path.join(TAB, f"rob_btcturk_venue{SUF}.json"), "w"), indent=1, default=str)

    # LaTeX table: S1 five-minute, common sample
    R = RES["5min"]; T, B, Df = R["S1_btcturk_common"]["coef"], R["S1_binance_common"]["coef"], R["S1_diff_common"]
    def st(c, s):
        t = abs(c / s); return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""
    rows = [("$\\kappa^+$, banks closed", "up"), ("$\\quad\\times$ banks open", "up_bh"),
            ("$\\kappa^-$, banks closed", "dn"), ("$\\quad\\times$ banks open", "dn_bh")]
    with open(os.path.join(TAB, f"rob_btcturk_venue{SUF}.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lccc}\n\\toprule\n & BtcTurk & Binance & Difference \\\\\n & (1) & (2) & (1)$-$(2) \\\\\n\\midrule\n")
        for lab, k in rows:
            c1, s1 = T[k]; c2, s2 = B[k]; c3, s3 = Df[k + "_diff_T_minus_B"]
            fh.write(f"{lab} & ${c1:.4f}{st(c1, s1)}$ & ${c2:.4f}{st(c2, s2)}$ & ${c3:.4f}{st(c3, s3)}$ \\\\\n"
                     f" & $({s1:.4f})$ & $({s2:.4f})$ & $({s3:.4f})$ \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"Median band width (bp) & {R['S1_btcturk_common']['band_width_median']:.0f} & "
                 f"{R['S1_binance_common']['band_width_median']:.0f} & \\\\\n")
        fh.write(f"Premium s.d. (bp) & {R['desc']['sd_b_btcturk']:.1f} & {R['desc']['sd_b_binance']:.1f} & \\\\\n")
        fh.write(f"$N$ (common intervals) & {R['n_common']:,} & {R['n_common']:,} & \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    print("done")
