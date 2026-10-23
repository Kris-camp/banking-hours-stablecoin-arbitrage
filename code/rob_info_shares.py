"""Robustness: linear information shares by regime, and lead times on 19 March 2025.

The paper argues that linear information-share measures (Hasbrouck 1995; Gonzalo-Granger 1995)
average the threshold transmission S_h(delta) over regimes. This script computes them by regime, and
measures the lead times of the two prices on 19 March 2025.

DESIGN
(a) Information shares.
  Prices: p^L = 1e4 log USDT/TRY (Binance), p^G = 1e4 log USD/TRY interbank mid (Dukascopy), bp.
  Minute sample and filters exactly as empirics_main.py: interbank mid forward-filled at most
  2 minutes, weekdays only, 20:30-22:30 UTC dropped, |b|>1000 bp dropped, 2024-01..2026-08.
  Frequencies: 1 minute; 5 minutes (means over valid minutes, >= 2 valid minutes, as in a5).
  Price variants: main = Binance minute VWAP (the paper's price) vs Dukascopy minute close at 1 min
  and interval means at 5 min. Timing check = Binance minute close vs Dukascopy minute close
  (both end-of-minute; at 5 min, the last valid minute of the interval). The VWAP averages over
  the minute while the interbank close is an end-of-minute quote, which by itself makes the
  interbank rate look ~30 seconds ahead, so the close-close variant is reported alongside.
  VECM: d p_t = c + alpha * z_{t-1} + sum_{i=1..k} Gamma_i d p_{t-i} + e_t, with
  z = p^L - p^G - m_month (cointegrating vector (1,-1) fixed; m = monthly band midpoint from
  emp_numbers.json, month of t). Estimated equation by equation by OLS on observations whose
  current value and 11 lags lie on a contiguous grid (common sample for every k).
  k chosen by BIC over 1..10 on the full sample, separately for each frequency x variant, and held
  fixed across regimes.
  Shares: alpha_perp = (alpha_G, -alpha_L), normalised to sum to one.
  Component share CS_L = alpha_G/(alpha_G - alpha_L) (equals the paper's omega).
  Hasbrouck IS_L: upper bound with p^L first in the Cholesky ordering, lower bound with p^G first;
  midpoint reported. Information leadership share ILS_L (Putnins 2013) from IS midpoints and CS.
  Regimes (subsample estimation; lags taken from the full contiguous series):
    full; banks open (bh_eff = Istanbul 09:00-17:00 on non-holiday weekdays, at t) vs closed;
    premium inside vs outside the month band, classified on b_{t-1} (the error-correction term's
    timing), at the same frequency (1-min premium for 1-min data). Secondary: above vs below band;
    open x outside, closed x outside, open x inside, closed x inside.
  Inference: day-clustered (UTC date) sandwich for (alpha_L, alpha_G) with the finite-sample
  correction of ols(); delta-method s.e. for CS_L; day-cluster pairs bootstrap (B=499, seed 0,
  computed exactly from per-day cross-product matrices) for CS_L, IS bounds/midpoint and ILS_L,
  reporting bootstrap s.e. and 2.5/97.5 percentiles.
  Secondary, descriptive: month-by-month CS_L and IS midpoint (1 min, VWAP), Spearman correlation
  across months with the monthly s.d. of the 5-minute premium, the share of 5-minute intervals
  outside the band and the s.d. of 5-minute interbank returns.
(b) 19 March 2025, one-minute data. Base = first valid observation at or after 05:00 UTC.
  First minute each price is >= +1, +3, +6% (100 x log change) above base, using the minute VWAP
  (Binance) and the minute close (both), and the first-touch minute from the minute high
  (Binance kline high; Dukascopy bid-candle high). First minute the premium is <= -100, -300,
  -600 bp; trough; first minute after the trough with b >= l_March (back inside the band) and
  first minute after the trough above -600, -300, -100. Lead = local time minus interbank time.

SENSITIVITY AND DESCRIPTION
  - The paper's shock-exclusion window (19 Mar - 30 Apr 2025, as in robust_table.py "_excl") for the
    full sample and the open/closed/outside/above/below regimes, and the single day 19 Mar 2025 alone,
    because March 2025 dominates alpha_G in the month-by-month estimates.
  - Event: reversal timing from each price's own peak; SWFX tick activity in the shock window against
    the same clock window on the 20 prior weekdays; one-minute return cross-correlations separately
    for the run-up (06:30-07:42) and the reversal (07:43-09:00).

FURTHER CHECKS (descriptive/sensitivity)
  - Event: the first-touch return times are driven by one-minute interbank noise (the one-minute premium
    touched the band at 09:01-09:02 and was back below -50 bp by 09:03). Also reported: (i) the first minute after
    the trough from which the one-minute premium stays >= threshold for 15 consecutive valid minutes, and
    (ii) the first five-minute mean premium (a5 construction: mean of valid minutes, >= 2) >= threshold after
    the five-minute trough; Spearman rank versions of the window cross-correlations.
  - Lags: BIC sits at the k=10 bound, so k is re-selected over 1..30 on the common sample with 31 lags
    (point estimates only) for full, full_x0319, open, closed, outside at 1 and 5 minutes (VWAP).

Inputs: data_raw/panel_TRY.parquet, data_raw/a5.parquet, data_raw/binance (USDTTRY), data_raw/dukascopy (USDTRY),
tables/emp_numbers.json.
Outputs: tables/rob_info_shares.json, tables/rob_info_shares.tex, figures/rob_info_shares.pdf.
Env: NBOOT (cluster-bootstrap draws, default 499; 0 skips the bootstrap).
"""
import os, json, glob, lzma, datetime as dt, warnings
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build_panel import load_fx, load_binance, RAW as _R
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
NBOOT = int(os.environ.get("NBOOT", 499))
KMAX = 10
OUT = dict(design=dict(kmax=KMAX, nboot=NBOOT, cluster="UTC date", ect="p^L-p^G-band midpoint (month of t)"))

# Turkish weekday public holidays, copied from empirics_mech.py
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
HOLD = set(pd.to_datetime(HOL).date)
BANDS = json.load(open(os.path.join(TAB, "emp_numbers.json")))["bands"]

# ------------------------------------------------------------------ minute data (as empirics_main.py)
p = pd.read_parquet(os.path.join(RAW, "panel_TRY.parquet")).drop(columns=["fx_mid", "fx_spread", "fxv"], errors="ignore")
p = p.loc["2024-01-01":"2026-08-31"]
FX = load_fx("USDTRY")
p = p.join(FX.reindex(p.index, method="ffill", limit=2))
kl = load_binance("USDTTRY")
p["c_usdt"] = kl.close.reindex(p.index)


def clean_b(pl_level, pg_level):
    b = 1e4 * np.log(pl_level / pg_level)
    hr = b.index.hour + b.index.minute / 60
    b[(hr >= 20.5) & (hr < 22.5)] = np.nan
    b[b.index.dayofweek >= 5] = np.nan
    b[b.abs() > 1000] = np.nan
    return b


p["b"] = clean_b(p.p_usdt, p.fx_mid)
p["b_c"] = clean_b(p.c_usdt, p.fx_mid)
OUT["n_valid_minutes"] = dict(vwap=int(p.b.notna().sum()), close=int(p.b_c.notna().sum()))


def series(freq, variant):
    """Return DataFrame on the regular grid with PL, PG (bp levels) and NaN where invalid."""
    if variant == "vwap":
        w = p[["p_usdt", "fx_mid"]].where(p.b.notna())
    else:
        w = p[["c_usdt", "fx_mid"]].where(p.b_c.notna()).rename(columns={"c_usdt": "p_usdt"})
    if freq == "1min":
        s = pd.DataFrame({"PL": 1e4 * np.log(w.p_usdt), "PG": 1e4 * np.log(w.fx_mid)})
    else:
        g = w.resample(freq)
        nb = g.p_usdt.count()
        if variant == "vwap":
            s = pd.DataFrame({"PL": 1e4 * np.log(g.p_usdt.mean()), "PG": 1e4 * np.log(g.fx_mid.mean())})
        else:
            s = pd.DataFrame({"PL": 1e4 * np.log(g.p_usdt.last()), "PG": 1e4 * np.log(g.fx_mid.last())})
        s = s.where(nb >= 2)
    grid = pd.date_range("2024-01-01", "2026-09-01", freq=freq, tz="UTC", inclusive="left")
    return s.reindex(grid)


def design(s):
    """Regressor matrix [const, z_{t-1}, dPL_{t-1}, dPG_{t-1}, ..., dPL_{t-K}, dPG_{t-K}], Y=[dPL_t, dPG_t]."""
    PL, PG = s.PL.values, s.PG.values
    n = len(PL)
    dPL = np.r_[np.nan, np.diff(PL)]; dPG = np.r_[np.nan, np.diff(PG)]
    naive = s.index.tz_localize(None)
    month = naive.to_period("M").astype(str)
    mid = np.array([(BANDS[m]["l"] + BANDS[m]["u"]) / 2 if m in BANDS else np.nan for m in month])
    lo = np.array([BANDS[m]["l"] if m in BANDS else np.nan for m in month])
    hi = np.array([BANDS[m]["u"] if m in BANDS else np.nan for m in month])
    def lag(v, i):
        return np.r_[np.full(i, np.nan), v[:n - i]]
    b1 = lag(PL - PG, 1)
    cols = [np.ones(n), b1 - mid]
    for i in range(1, KMAX + 1):
        cols += [lag(dPL, i), lag(dPG, i)]
    X = np.column_stack(cols); Y = np.column_stack([dPL, dPG])
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    idx = s.index[ok]
    ih = (idx.hour + 3) % 24
    day_ist = (idx + pd.Timedelta(hours=3)).date
    bh = ((ih >= 9) & (ih < 17)) & ~np.isin(day_ist, list(HOLD))
    b1o, loo, hio = b1[ok], lo[ok], hi[ok]
    reg = dict(open=bh, closed=~bh, inside=(b1o >= loo) & (b1o <= hio), above=b1o > hio, below=b1o < loo)
    reg["outside"] = ~reg["inside"]
    for a_ in ("open", "closed"):
        for c_ in ("inside", "outside"):
            reg[f"{a_}_{c_}"] = reg[a_] & reg[c_]
    reg["full"] = np.ones(ok.sum(), bool)
    # sensitivity: the paper's shock exclusion window,
    # 19 Mar - 30 Apr 2025 (robust_table.py "_excl"), because March 2025 dominates alpha_G by month.
    xev = ~((idx >= pd.Timestamp("2025-03-19", tz="UTC")) & (idx < pd.Timestamp("2025-05-01", tz="UTC")))
    for r_ in ("full", "open", "closed", "outside", "above", "below"):
        reg[r_ + "_xev"] = reg[r_] & xev
    # sensitivity: drop the single day 19 March 2025 only
    reg["full_x0319"] = ~((idx >= pd.Timestamp("2025-03-19", tz="UTC")) & (idx < pd.Timestamp("2025-03-20", tz="UTC")))
    day = pd.factorize(idx.date)[0]
    return X[ok], Y[ok], day, reg, idx, month[ok], b1o


def day_cross(X, Y, day):
    Z = np.hstack([X, Y])
    st = np.r_[0, np.flatnonzero(np.diff(day)) + 1, len(day)]
    M = np.empty((len(st) - 1, Z.shape[1], Z.shape[1]))
    for g in range(len(st) - 1):
        z = Z[st[g]:st[g + 1]]
        M[g] = z.T @ z
    return M


def sel(M, k):
    """Sub-matrix for k lags: X columns 0..2+2k-1, Y the last two."""
    K = 2 + 2 * k; KT = M.shape[-1]
    ii = np.r_[np.arange(K), KT - 2, KT - 1]
    return M[..., ii[:, None], ii[None, :]]


def fit(Ms, K):
    XX, XY, YY = Ms[:K, :K], Ms[:K, K:], Ms[K:, K:]
    B = np.linalg.solve(XX, XY)
    n = Ms[0, 0]
    S = YY - XY.T @ B
    return B, S / (n - K), n


def shares(aL, aG, Om):
    g = np.array([aG, -aL]); g = g / g.sum()
    v = g @ Om @ g
    F = np.linalg.cholesky(Om)
    isL_up = (g @ F)[0] ** 2 / v
    P = np.array([[0, 1], [1, 0]])
    F2 = np.linalg.cholesky(P @ Om @ P)
    isG_up = (g[::-1] @ F2)[0] ** 2 / v
    isL_alt = 1 - isG_up                                   # IS_L with p^G first in the ordering
    isL_lo, isL_up = min(isL_up, isL_alt), max(isL_up, isL_alt)
    mid = 0.5 * (isL_up + isL_lo)
    cs = g[0]
    IL = abs((mid / (1 - mid)) * ((1 - cs) / cs)) if 0 < mid < 1 and cs != 0 else np.nan
    ils = IL / (IL + 1 / IL) if np.isfinite(IL) and IL > 0 else np.nan
    return dict(CS_L=cs, IS_L_lo=isL_lo, IS_L_up=isL_up, IS_L_mid=mid, ILS_L=ils)


def estimate(Md, k, rng):
    """Md: per-day cross-products (days x (KT) x (KT)) for k lags."""
    K = 2 + 2 * k
    Ms = Md.sum(0)
    B, Om, n = fit(Ms, K)
    XXi = np.linalg.inv(Ms[:K, :K])
    # day-clustered covariance of (alpha_L, alpha_G)
    Sd = Md[:, :K, K:] - Md[:, :K, :K] @ B                 # days x K x 2 scores
    inf = np.einsum("j,djk->dk", XXi[1], Sd)               # influence on the alpha row
    G = len(Md)
    adj = G / (G - 1) * (n - 1) / (n - K)
    Va = adj * inf.T @ inf
    aL, aG = B[1, 0], B[1, 1]
    sh = shares(aL, aG, Om)
    d = aG - aL
    grad = np.array([aG / d ** 2, -aL / d ** 2])
    res = dict(n=int(n), days=int(G), k=k, alpha_L=float(aL), se_alpha_L=float(np.sqrt(Va[0, 0])),
               alpha_G=float(aG), se_alpha_G=float(np.sqrt(Va[1, 1])),
               sd_eL=float(np.sqrt(Om[0, 0])), sd_eG=float(np.sqrt(Om[1, 1])),
               corr_e=float(Om[0, 1] / np.sqrt(Om[0, 0] * Om[1, 1])),
               se_CS_L_delta=float(np.sqrt(grad @ Va @ grad)))
    res.update({kk: float(v) for kk, v in sh.items()})
    if NBOOT:
        draws = []
        for _ in range(NBOOT):
            w = rng.multinomial(G, np.full(G, 1 / G)).astype(float)
            Mb = np.tensordot(w, Md, axes=1)
            try:
                Bb, Ob, _n = fit(Mb, K)
                draws.append(list(shares(Bb[1, 0], Bb[1, 1], Ob).values()))
            except np.linalg.LinAlgError:
                pass
        D = np.array(draws)
        for j, kk in enumerate(sh.keys()):
            col = D[:, j]; col = col[np.isfinite(col)]
            res[f"bse_{kk}"] = float(np.std(col, ddof=1))
            res[f"ci_{kk}"] = [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))]
    return res


REGS = ["full", "open", "closed", "inside", "outside", "above", "below",
        "open_inside", "open_outside", "closed_inside", "closed_outside",
        "full_xev", "open_xev", "closed_xev", "outside_xev", "above_xev", "below_xev", "full_x0319"]
rng = np.random.default_rng(0)
INFO = {}
monthly = {}
for freq in ("1min", "5min"):
    for variant in ("vwap", "close"):
        key = f"{freq}_{variant}"
        s = series(freq, variant)
        X, Y, day, reg, idx, month, b1 = design(s)
        Mfull = day_cross(X, Y, day)
        # BIC on the common sample
        Ms = Mfull.sum(0); n = Ms[0, 0]; bic = {}
        for k in range(1, KMAX + 1):
            Mk = sel(Ms, k); K = 2 + 2 * k
            B, Om, _ = fit(Mk, K)
            Oml = Om * (n - K) / n
            bic[k] = float(n * np.log(np.linalg.det(Oml)) + 2 * K * np.log(n))
        kbest = min(bic, key=bic.get)
        INFO[key] = dict(k_bic=kbest, bic=bic, n_common=int(n), share_outside=float(reg["outside"].mean()),
                         share_open=float(reg["open"].mean()), regimes={})
        print(key, "k=", kbest, "n=", int(n), flush=True)
        # lag-order sensitivity of the full-sample point estimates (BIC may sit at the upper bound)
        ksens = {}
        for k in (1, 2, 5, KMAX):
            B, Om, _ = fit(sel(Ms, k), 2 + 2 * k)
            ksens[k] = {kk: float(v) for kk, v in shares(B[1, 0], B[1, 1], Om).items()}
        INFO[key]["k_sensitivity_full"] = ksens
        for r in REGS:
            m = reg[r]
            if m.sum() < 1000:
                continue
            Md = sel(day_cross(X[m], Y[m], day[m]), kbest) if r != "full" else sel(Mfull, kbest)
            # drop days with no observations (zero matrices are harmless, but keep G honest)
            Md = Md[Md[:, 0, 0] > 0]
            INFO[key]["regimes"][r] = estimate(Md, kbest, rng)
            e = INFO[key]["regimes"][r]
            print(f"  {r:15s} n={e['n']:7d} aL={e['alpha_L']:.5f}({e['se_alpha_L']:.5f}) aG={e['alpha_G']:.5f}({e['se_alpha_G']:.5f}) "
                  f"CS_L={e['CS_L']:.3f}[{e.get('ci_CS_L')}] IS_L=[{e['IS_L_lo']:.3f},{e['IS_L_up']:.3f}] ILS={e['ILS_L']:.3f}", flush=True)
        if key == "1min_vwap":
            mos = sorted(set(month))
            for mo in mos:
                m = month == mo
                if m.sum() < 2000:
                    continue
                Md = sel(day_cross(X[m], Y[m], day[m]), kbest)
                K = 2 + 2 * kbest
                B, Om, nn = fit(Md.sum(0), K)
                sh = shares(B[1, 0], B[1, 1], Om)
                monthly[mo] = dict(n=int(nn), alpha_L=float(B[1, 0]), alpha_G=float(B[1, 1]),
                                   **{kk: float(v) for kk, v in sh.items()})
OUT["vecm"] = INFO

# ---- lag order re-selected over 1..30 (point estimates; common sample with 31 lags) ----
KMAX = 30                                     # design() reads the global KMAX at call time
LAGX = {}
for freq in ("1min", "5min"):
    X, Y, day, reg, idx, month, b1 = design(series(freq, "vwap"))
    for r in ("full", "full_x0319", "open", "closed", "outside"):
        m = reg[r]
        M = day_cross(X[m], Y[m], day[m]).sum(0); n = M[0, 0]
        bic, pts = {}, {}
        for k in range(1, KMAX + 1):
            K = 2 + 2 * k
            B, Om, _ = fit(sel(M, k), K)
            bic[k] = float(n * np.log(np.linalg.det(Om * (n - K) / n)) + 2 * K * np.log(n))
            pts[k] = {kk: float(v) for kk, v in shares(B[1, 0], B[1, 1], Om).items()}
        kb = min(bic, key=bic.get)
        LAGX[f"{freq}_vwap_{r}"] = dict(n=int(n), k_bic=kb, at_bic=pts[kb], at_10=pts[10],
                                        by_k={k: pts[k] for k in (1, 5, 10, 15, 20, 25, 30)})
        print("lagX", freq, r, "k_bic", kb, {kk: round(v, 3) for kk, v in pts[kb].items()}, flush=True)
    del X, Y
KMAX = 10
OUT["lag_extension_audit"] = LAGX

# ------------------------------------------------------------------ monthly co-movement
a5 = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
mstat = pd.DataFrame({"sd_prem": a5.groupby("month").b.std(), "out_share": 1 - a5.groupby("month").inband.mean(),
                      "sd_dpg": a5.groupby("month").dpg.std()})
mm = pd.DataFrame(monthly).T.join(mstat, how="inner")
corr = {}
for v in ("CS_L", "IS_L_mid"):
    for c in ("sd_prem", "out_share", "sd_dpg"):
        rho, pv = stats.spearmanr(mm[v], mm[c])
        corr[f"{v}~{c}"] = [float(rho), float(pv)]
OUT["monthly"] = dict(table=mm.reset_index().rename(columns={"index": "month"}).to_dict("records"), spearman=corr,
                      CS_L_range=[float(mm.CS_L.min()), float(mm.CS_L.max())],
                      CS_L_iqr=[float(mm.CS_L.quantile(.25)), float(mm.CS_L.quantile(.75))],
                      IS_mid_range=[float(mm.IS_L_mid.min()), float(mm.IS_L_mid.max())],
                      IS_mid_iqr=[float(mm.IS_L_mid.quantile(.25)), float(mm.IS_L_mid.quantile(.75))])
print(mm.round(3).to_string())
print(corr)

# ------------------------------------------------------------------ (b) 19 March 2025 lead times
D0, D1 = "2025-03-19 05:00", "2025-03-19 15:00"
ev = p.loc[D0:D1].copy()


def duk_high(day="20250319"):
    f = os.path.join(RAW, "dukascopy", f"USDTRY_BID_{day}.bi5")
    raw = lzma.decompress(open(f, "rb").read())
    a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6).astype(np.int64)
    vol = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5].astype(np.float64)
    t0 = pd.Timestamp(day, tz="UTC")
    idx = t0 + pd.to_timedelta(a[:, 0].astype("int64"), unit="s")
    return pd.DataFrame({"o": a[:, 1] / 1e5, "c": a[:, 2] / 1e5, "lo": a[:, 3] / 1e5, "hi": a[:, 4] / 1e5, "v": vol}, index=idx)


dk = duk_high()
dk = dk[dk.v > 0]
kraw = pd.concat([pd.read_csv(f, header=None, names=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbv", "tbqv", "i"])
                  for f in glob.glob(os.path.join(RAW, "binance", "USDTTRY-1m-2025-03.csv"))])
tt = kraw.t.astype("int64"); tt = np.where(tt > 1e14, tt // 1000, tt)
kraw.index = pd.to_datetime(tt, unit="ms", utc=True)
ev["h_usdt"] = kraw.h.reindex(ev.index)
ev["hi_fx"] = dk.hi.reindex(ev.index)              # bid-candle high (no ffill: only minutes with FX activity)
bm = BANDS["2025-03"]


def first(series_, cond):
    s_ = series_[cond(series_)]
    return s_.index[0] if len(s_) else None


def fmt(t):
    return None if t is None else t.strftime("%H:%M")


def mins(t1, t0):
    return None if (t1 is None or t0 is None) else float((t1 - t0).total_seconds() / 60)


def sustained(s_, th, N):
    """First index from which s_ >= th for N consecutive grid points (NaN breaks the run)."""
    ok = (s_ >= th).values
    run = 0
    for i, o in enumerate(ok):
        run = run + 1 if o else 0
        if run >= N:
            return s_.index[i - N + 1]
    return None


E = dict(band=dict(l=bm["l"], u=bm["u"]))
lvl = {}
for nm, colL in (("vwap", "p_usdt"), ("close", "c_usdt")):
    L = ev[colL].dropna(); G_ = ev.fx_mid.dropna()
    rL = 100 * (np.log(L) - np.log(L.iloc[0])); rG = 100 * (np.log(G_) - np.log(G_.iloc[0]))
    lvl[nm] = dict(base_time_L=str(L.index[0]), base_time_G=str(G_.index[0]), base_L=float(L.iloc[0]), base_G=float(G_.iloc[0]))
    cross = {}
    for th in (1, 3, 6):
        tL = first(rL, lambda s_: s_ >= th); tG = first(rG, lambda s_: s_ >= th)
        cross[f"+{th}%"] = dict(local=fmt(tL), interbank=fmt(tG), lead_local_minus_interbank_min=mins(tL, tG))
    cross["peak"] = dict(local=fmt(rL.idxmax()), local_pct=float(rL.max()), interbank=fmt(rG.idxmax()), interbank_pct=float(rG.max()))
    b = (ev.b if nm == "vwap" else ev.b_c).dropna()
    tmin = b.idxmin(); after = b.loc[tmin:]
    prem = dict(b_at_0500=float(b.iloc[0]), first_below_band=fmt(first(b, lambda s_: s_ < bm["l"])),
                trough=fmt(tmin), trough_bp=float(b.min()))
    for th in (-100, -300, -600):
        prem[f"first<= {th}"] = fmt(first(b, lambda s_, th=th: s_ <= th))
        prem[f"back> {th} after trough"] = fmt(first(after, lambda s_, th=th: s_ > th))
    tb = first(after, lambda s_: s_ >= bm["l"])
    prem["back_inside_band"] = fmt(tb); prem["minutes_trough_to_band"] = mins(tb, tmin)
    # sustained returns (first-touch times reflect one-minute interbank noise)
    bg = (ev.b if nm == "vwap" else ev.b_c).loc[tmin:]          # minute grid, NaN breaks a run
    sus = {}
    for th in (-600, -300, -100, bm["l"]):
        ts_ = sustained(bg, th, 15)
        sus[f">= {th} for 15 min from"] = fmt(ts_)
        sus[f"minutes_trough_to_sustained_{th}"] = mins(ts_, tmin)
    bfull = ev.b if nm == "vwap" else ev.b_c
    g5 = bfull.resample("5min").agg(["mean", "count"]); b5 = g5["mean"].where(g5["count"] >= 2).dropna()
    t5 = b5.idxmin(); a5_ = b5.loc[t5:]
    sus["5min_trough"] = fmt(t5); sus["5min_trough_bp"] = float(b5.min())
    for th in (-300, -100, bm["l"]):
        sus[f"5min_first_mean>= {th}"] = fmt(first(a5_, lambda s_, th=th: s_ >= th))
    tb5 = first(a5_, lambda s_: s_ >= bm["l"])
    sus["5min_minutes_trough_to_band"] = mins(tb5, t5)
    sus["share_of_minutes_inside_band_0901_1000"] = float(((bfull >= bm["l"]) & (bfull <= bm["u"])).loc["2025-03-19 09:01":"2025-03-19 09:59"].mean())
    prem["sustained_audit"] = sus
    E[nm] = dict(price_crossings=cross, premium=prem)
# first-touch from intraminute highs
L0 = ev.p_usdt.dropna().iloc[0]; G0 = ev.fx_mid.dropna().iloc[0]
rLh = 100 * (np.log(ev.h_usdt.dropna()) - np.log(L0)); rGh = 100 * (np.log(ev.hi_fx.dropna()) - np.log(G0))
E["first_touch_high"] = {f"+{th}%": dict(local=fmt(first(rLh, lambda s_: s_ >= th)), interbank=fmt(first(rGh, lambda s_: s_ >= th)),
                                         lead_local_minus_interbank_min=mins(first(rLh, lambda s_: s_ >= th), first(rGh, lambda s_: s_ >= th)))
                         for th in (1, 3, 6)}
E["levels"] = lvl
# ---- descriptive additions (see docstring) ----
L0 = ev.p_usdt.dropna().iloc[0]; G0 = ev.fx_mid.dropna().iloc[0]
rLv = 100 * (np.log(ev.p_usdt) - np.log(L0)); rGv = 100 * (np.log(ev.fx_mid) - np.log(G0))
# reversal: first minute after each price's own peak at which it is >= 1, 2, 3 pp below that peak
rev = {}
for nm, r_ in (("local", rLv.dropna()), ("interbank", rGv.dropna())):
    tpk = r_.idxmax(); aft = r_.loc[tpk:]
    rev[nm] = dict(peak=fmt(tpk), **{f"-{d}pp": fmt(first(aft, lambda s_, d=d: s_ <= r_.max() - d)) for d in (1, 2, 3)})
E["reversal_from_peak"] = rev
# interbank (Dukascopy SWFX) activity: tick-volume during the shock vs the same clock window on the 20 prior weekdays
fxa = FX.fxv.reindex(p.index).fillna(0.0)          # raw SWFX tick volume, 0 when the minute has no update
days_prior = [d for d in pd.bdate_range("2025-02-01", "2025-03-18") if d.date() not in HOLD][-20:]
def win_med(d, a_, b_):
    return float(np.nanmedian(fxa.loc[f"{d:%Y-%m-%d} {a_}":f"{d:%Y-%m-%d} {b_}"].values))
act = {}
for a_, b_ in (("06:30", "07:29"), ("07:30", "07:59"), ("08:00", "08:59")):
    act[f"{a_}-{b_}"] = dict(event=win_med(pd.Timestamp("2025-03-19"), a_, b_),
                             prior20_median=float(np.median([win_med(d, a_, b_) for d in days_prior])))
# minutes in 07:30-07:59 whose interbank mid is forward-filled (no SWFX update that minute)
raw_fx = FX.loc["2025-03-19 07:30":"2025-03-19 07:59"]
act["minutes_with_fx_update_0730_0759"] = int(len(raw_fx))
E["interbank_activity"] = act
# lead-lag cross-correlation of one-minute log returns, by window
xcs = {}
for wn, (a_, b_) in dict(morning=("06:00", "10:00"), runup=("06:30", "07:42"), reversal=("07:43", "09:00")).items():
    w = ev.loc[f"2025-03-19 {a_}":f"2025-03-19 {b_}"]
    rl = 1e4 * np.log(w.p_usdt).diff(); rg = 1e4 * np.log(w.fx_mid).diff()
    # L>0: interbank return L minutes earlier against the local return now
    xcs[wn] = {L_: float(rl.corr(rg.shift(L_))) for L_ in range(-5, 6)}
    # rank correlation, robust to the few very large one-minute moves
    xcs[wn + "_spearman"] = {L_: float(rl.corr(rg.shift(L_), method="spearman")) for L_ in range(-5, 6)}
E["xcorr_rL_t__rG_t-L"] = xcs
pth = pd.DataFrame({"rL": rLv, "rG": rGv, "b": ev.b, "fxv": ev.fxv}).loc["2025-03-19 06:45":"2025-03-19 09:05"]
E["path_0645_0905"] = {t.strftime("%H:%M"): [None if not np.isfinite(v) else round(float(v), 2) for v in row]
                       for t, row in zip(pth.index, pth.values)}
OUT["event"] = E
print(json.dumps({k: v for k, v in E.items() if k != "path_0645_0905"}, indent=1))

json.dump(OUT, open(os.path.join(TAB, "rob_info_shares.json"), "w"), indent=1, default=str)

# ------------------------------------------------------------------ table
def fx3(v):
    return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"


def rowcells(e, inside=False):
    if inside:
        return ["n.d.", "n.d."], ["", ""]
    ci = e["ci_CS_L"]; cim = e["ci_IS_L_mid"]
    return ([fx3(e["CS_L"]), f"[{e['IS_L_lo']:.3f}, {e['IS_L_up']:.3f}]"],
            [f"({ci[0]:.2f}, {ci[1]:.2f})", f"({cim[0]:.2f}, {cim[1]:.2f})"])


lab = dict(full="All observations", open="Banks open", closed="Banks closed", inside="Inside band",
           outside="Outside band", above="Above band", below="Below band")
K1, K5 = "1min_vwap", "5min_vwap"
lines = ["\\begin{tabular}{lcccc}", "\\toprule",
         " & \\multicolumn{2}{c}{One minute} & \\multicolumn{2}{c}{Five minutes} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         " & $CS_L$ & $IS_L$ bounds & $CS_L$ & $IS_L$ bounds \\\\", "\\midrule"]
for blk, rs, suf in (("Full sample", ["full", "open", "closed", "inside", "outside", "above", "below"], ""),
                     ("Excluding 19 March -- 30 April 2025", ["full", "open", "closed", "outside", "above", "below"], "_xev")):
    lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{blk}}}}} \\\\")
    for r in rs:
        c1, d1 = rowcells(INFO[K1]["regimes"][r + suf], r == "inside")
        c5, d5 = rowcells(INFO[K5]["regimes"][r + suf], r == "inside")
        lines.append(f"{lab[r]} & {c1[0]} & {c1[1]} & {c5[0]} & {c5[1]} \\\\")
        if r != "inside":
            lines.append(f" & {d1[0]} & {d1[1]} & {d5[0]} & {d5[1]} \\\\")
lines += ["\\midrule",
          f"Lags (BIC) & \\multicolumn{{2}}{{c}}{{{INFO[K1]['k_bic']}}} & \\multicolumn{{2}}{{c}}{{{INFO[K5]['k_bic']}}} \\\\",
          f"Observations & \\multicolumn{{2}}{{c}}{{{INFO[K1]['regimes']['full']['n']:,}}} & \\multicolumn{{2}}{{c}}{{{INFO[K5]['regimes']['full']['n']:,}}} \\\\",
          "\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "rob_info_shares.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
mo = pd.PeriodIndex(mm.index, freq="M").to_timestamp()
ax[0].plot(mo, mm.IS_L_mid, color=C1, lw=1.4, marker="o", ms=3, label="Hasbrouck $IS_L$ (midpoint)")
ax[0].plot(mo, mm.CS_L.clip(-0.2, 1), color=C2, lw=1.0, ls="--", marker="s", ms=3, label="Gonzalo-Granger $CS_L$")
ax[0].axhline(0, color="k", lw=0.4)
ax[0].set_ylabel("stablecoin share"); ax[0].set_ylim(-0.25, 1.3)
ax[0].set_title("(a) One-minute shares by month", fontsize=10); ax[0].legend(frameon=False, fontsize=7.5, loc="upper right")
ax[0].tick_params(axis="x", rotation=30)
import matplotlib.dates as mdates
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
z = ev.loc["2025-03-19 06:45":"2025-03-19 09:15"]
tt = z.index.tz_localize(None)
ax[1].plot(tt, 100 * (np.log(z.p_usdt) - np.log(L0)), color=C1, lw=1.2, label="USDT/TRY (Binance)")
ax[1].plot(tt, 100 * (np.log(z.fx_mid) - np.log(G0)), color=C2, lw=1.2, ls="--", label="USD/TRY (interbank)")
for th in (1, 3, 6):
    cr = OUT["event"]["vwap"]["price_crossings"][f"+{th}%"]
    for who, col, mk in (("local", C1, "o"), ("interbank", C2, "s")):
        if cr[who]:
            t_ = pd.Timestamp(f"2025-03-19 {cr[who]}")
            ax[1].plot([t_], [th], marker=mk, color=col, ms=5, ls="none")
ax[1].set_ylabel("% change since 05:00 UTC"); ax[1].set_xlabel("UTC")
ax[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30])); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax[1].tick_params(axis="x", rotation=30)
ax[1].set_title("(b) 19 March 2025", fontsize=10)
ax[1].text(0.98, 0.10, "markers: first minute\nat +1%, +3%, +6%", transform=ax[1].transAxes, ha="right", va="bottom", fontsize=7.5, color=C3)
ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rob_info_shares.pdf")); plt.close(fig)
print("done")
