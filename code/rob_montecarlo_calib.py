"""Calibrated Monte Carlo for the empirical estimator of Table emp_main, columns (3)-(4).

The baseline Monte Carlo of Appendix F (run_all.py) uses speeds ~50x the empirical ones,
one-minute data and no five-minute averaging. This study calibrates the data-generating process
to the empirical regime and also asks whether profiling the band can mechanically create a flat
band with zero in-band reversion.

DESIGN
------
Minute-level DGP (bp, minutes), on the real calendar 2024-01-01 .. 2026-08-31 (UTC) plus a
four-week burn-in, weekends and the FX rollover simulated but unobserved:
    b_m = b_{m-1} + beta1*q_m - kp(s)(b_{m-1}-u)_+ + km(s)(l-b_{m-1})_+ - kl(s) b_{m-1} + sig(s) e_m
    s = open on weekdays 09:00-17:00 Istanbul (06:00-14:00 UTC), closed otherwise (incl. weekends).
    q_m = sq(s) * [sqrt(w1) z1 + sqrt(w2) z2 + sqrt(1-w1-w2) n],  z1, z2 unit-variance AR(1)s
          (rho1, rho2); (w1, rho1, w2, rho2) fitted to the ACF (lags 1-60) of one-minute OFI
          standardised within month; sq(open)/sq(closed) = ratio of one-minute OFI SDs.
    beta1, sig(open), sig(closed) calibrated by simulation (DGP1, fixed seeds) to the
          empirical OFI coefficient of column (2) and the SD of the five-minute premium
          change in each banking state.
    Five-minute speeds k5 are converted to per-minute speeds k1 = 1-(1-k5)^(1/5); the
    estimand for the five-minute regression is k5.
DGP1 (truth): band [-16, 16] (median empirical width 32 bp); kp, km per five minutes equal to
     emp_main column (4): closed 0.0025 / 0.0074, open 0.0104 / 0.0097; kl = 0.
DGP2: linear error correction toward 0, kl = 0.0039 per five minutes (column 1), no band.
DGP3: band [-16, 16], same speeds in both states, kp = 0.0054, km = 0.0085 (column 2).
Common random numbers across DGPs (rep r uses the same seed in every DGP).

Estimation replicates empirics_main.py exactly: five-minute means of the minute premium on
weekdays outside 20:30-22:30 UTC, OFI summed and standardised by its within-month SD,
y = db, x = b_{t-1}, consecutive intervals only, monthly bands profiled over a 1 bp grid
between the 5th and 95th percentiles of x with l <= u, the _ssr_grid objective (evaluated
here by an algebraically identical prefix-sum routine, `fast_ssr`, whose output is checked
against model._ssr_grid on the empirical months and on simulated months), month FE,
day-clustered SEs with the same finite-sample correction.

Outputs, per DGP, over R simulated 32-month samples:
  * bias and RMSE of the band edges l_m, u_m (pooled over months x replications);
  * bias and RMSE of kappa+_closed, kappa+_open, kappa-_closed, kappa-_open (column 4),
    of the ratio kappa+_open/kappa+_closed (mean, median, RMSE, median abs. error), and of
    the in-band coefficient (column 3); coverage of the 95% day-clustered CIs;
  * rejection rates of two-sided 5% t-tests: (a) in-band coefficient = 0 [size under DGP1
    and DGP3, power under DGP2]; (b) BH increment on kappa+ = 0 [size under DGP3, power
    under DGP1]; also BH increment on kappa- and on OFI;
  * DGP2 "spurious flat band": share of samples in which (i) column-3 kappa+ and kappa-
    are both positive with t > 1.96, (ii) the in-band coefficient has t < 1.96, and
    (iii) the median monthly band width is >= 20 bp; plus P(in-band coef <= -0.0006, the
    empirical value) and P(sum of monthly sup-LR >= 464.1, the empirical value).
Volatility also varies by month: sig_m(s) = sig(s) * h_m, h_m = SD of the five-minute premium
change in month m divided by its root-mean-square across months.

Two calibrations of the same three DGPs (both reported in full):
  Design A: OFI with the fitted one-minute persistence; beta1, sig(open),
    sig(closed) matched to the column-(2) OFI coefficient and the SD of the five-minute
    premium change by banking state.
  Design B (dispersion-matched; calibration diagnostics show that design A puts the
    within-month SD of the premium at about twice its empirical value and that the data
    show no lagged-flow predictability of premium changes beyond 15 minutes): only the
    short-memory OFI component
    (w1 = 0), beta1 matched to the column-(2) coefficient, and a common scale on sig
    (open/closed ratio as in A) matched to the median within-month SD of the
    five-minute premium (21.5 bp).
Variants on design A (secondary, fewer replications): Student-t(3) minute shocks scaled to
unit variance (A1t, A2t, A3t); month-specific bands equal to the empirical monthly estimates
(A1e, A3e; A2e reverts to the empirical band midpoints).

Inputs: data_raw/a5.parquet, data_raw/panel_TRY.parquet, tables/emp_numbers.json.
Run: python rob_montecarlo_calib.py   (env R, RV, NPROC; defaults 500, 200, 4)
Further env: SCRATCH (a folder in which to save the replications as rob_montecarlo_calib_rows.npy) and ROWS (path
of such a file: re-summarise the saved replications instead of simulating; the simulation time is then read
from the existing tables/rob_montecarlo_calib.json).
Writes tables/rob_montecarlo_calib.{json,tex}, figures/rob_montecarlo_calib.pdf.
"""
import os, sys, json, time
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
NAME = "rob_montecarlo_calib"
R_MAIN = int(os.environ.get("R", 500))
R_VAR = int(os.environ.get("RV", 200))
NPROC = int(os.environ.get("NPROC", 4))
TRIM = 0.05
SEED0 = 20261022

# ------------------------------------------------------------------ calendar (shared, cheap)
T_BURN = np.datetime64("2023-12-04T00:00")
T_START = np.datetime64("2024-01-01T00:00")
T_END = np.datetime64("2026-09-01T00:00")
M = int((T_END - T_BURN) / np.timedelta64(1, "m"))
_ts = T_BURN + np.arange(M).astype("timedelta64[m]")
_days = _ts.astype("datetime64[D]")
_dow = (_days.astype(np.int64) + 3) % 7                        # Monday = 0
_mod = (_ts - _days).astype(np.int64)                          # UTC minute of day
WEEKDAY = _dow < 5
OPEN = WEEKDAY & (_mod >= 6 * 60) & (_mod < 14 * 60)           # 09-17 Istanbul (UTC+3)
OBS = WEEKDAY & ~((_mod >= 20 * 60 + 30) & (_mod < 22 * 60 + 30)) & (_ts >= T_START)
_mon_all = (_ts.astype("datetime64[M]") - T_START.astype("datetime64[M]")).astype(np.int64)
MIN_MON = np.clip(_mon_all, 0, None).astype(np.int64)          # burn-in uses month 0's band
NM = int(MIN_MON.max() + 1)
NB = M // 5
_bin_obs = OBS.reshape(NB, 5).sum(1) >= 2
RET = np.where(_bin_obs)[0]                                    # retained five-minute bins
_bts = _ts[::5][RET]
_bmon = MIN_MON[::5][RET]
_bday = (_bts.astype("datetime64[D]") - T_START.astype("datetime64[D]")).astype(np.int64)
_bbh = OPEN[::5][RET]
_cons = np.r_[False, np.diff(RET) == 1]
POS = np.where(_cons)[0]                                       # regression rows: positions k with k-1 consecutive
MCODE = _bmon[POS]
_, DCODE = np.unique(_bday[POS], return_inverse=True)
ND = int(DCODE.max() + 1)
BH = _bbh[POS].astype(float)
BMON_RET = _bmon                                               # month of every retained bin (for q std)


# ------------------------------------------------------------------ simulation kernel
@njit(cache=True)
def simulate(seed, open_, mon, lo_b, hi_b, ctr, kp, km, kl, beta1, sig, hm, sq, w1, r1, w2, r2, tdf):
    np.random.seed(seed)
    n = open_.shape[0]
    b = np.empty(n); q = np.empty(n)
    z1 = np.random.standard_normal(); z2 = np.random.standard_normal()
    s1 = np.sqrt(1 - r1 * r1); s2 = np.sqrt(1 - r2 * r2)
    a1 = np.sqrt(w1); a2 = np.sqrt(w2); a0 = np.sqrt(1 - w1 - w2)
    tsc = np.sqrt((tdf - 2.0) / tdf) if tdf > 0 else 1.0
    bp = ctr[0]
    for m in range(n):
        s = 1 if open_[m] else 0
        z1 = r1 * z1 + s1 * np.random.standard_normal()
        z2 = r2 * z2 + s2 * np.random.standard_normal()
        qm = sq[s] * (a1 * z1 + a2 * z2 + a0 * np.random.standard_normal())
        mi = mon[m]
        rev = -kl[s] * (bp - ctr[mi])
        if bp > hi_b[mi]:
            rev -= kp[s] * (bp - hi_b[mi])
        elif bp < lo_b[mi]:
            rev += km[s] * (lo_b[mi] - bp)
        e = np.random.standard_normal()
        if tdf > 0:
            e = e / np.sqrt(np.random.chisquare(tdf) / tdf) * tsc
        bp = bp + beta1 * qm + rev + sig[s] * hm[mi] * e
        b[m] = bp; q[m] = qm
    return b, q


# ------------------------------------------------------------------ fast exact SSR surface
@njit(cache=True)
def fast_ssr(y, x, q, grid):
    """Same objective and layout as model._ssr_grid(y, x, q, grid, -grid):
    S[i, j] = SSR of y on [1, q, -(x-grid[i])_+, (grid[j]-x)_+], inf where grid[j] > grid[i]."""
    n = y.shape[0]; G = grid.shape[0]
    o = np.argsort(x)
    xs = x[o]; ys = y[o]; qs = q[o]
    P = np.zeros((n + 1, 7))       # cumulative: 1, x, x^2, q, qx, y, xy
    for t in range(n):
        xt = xs[t]; qt = qs[t]; yt = ys[t]
        P[t + 1, 0] = P[t, 0] + 1.0
        P[t + 1, 1] = P[t, 1] + xt
        P[t + 1, 2] = P[t, 2] + xt * xt
        P[t + 1, 3] = P[t, 3] + qt
        P[t + 1, 4] = P[t, 4] + qt * xt
        P[t + 1, 5] = P[t, 5] + yt
        P[t + 1, 6] = P[t, 6] + xt * yt
    Sq = P[n, 3]; Sy = P[n, 5]
    Sqq = 0.0; Sqy = 0.0; yy = 0.0
    for t in range(n):
        Sqq += q[t] * q[t]; Sqy += q[t] * y[t]; yy += y[t] * y[t]
    iR = np.searchsorted(xs, grid, side="right")
    iL = np.searchsorted(xs, grid, side="left")
    out = np.empty((G, G))
    A = np.zeros((4, 4)); v = np.zeros(4)
    for i in range(G):
        u = grid[i]; k = iR[i]
        U1 = P[n, 0] - P[k, 0]; Ux = P[n, 1] - P[k, 1]; Uxx = P[n, 2] - P[k, 2]
        Uq = P[n, 3] - P[k, 3]; Uqx = P[n, 4] - P[k, 4]; Uy = P[n, 5] - P[k, 5]; Uxy = P[n, 6] - P[k, 6]
        for j in range(G):
            l = grid[j]
            if l > u:
                out[i, j] = np.inf
                continue
            h = iL[j]
            L1 = P[h, 0]; Lx = P[h, 1]; Lxx = P[h, 2]; Lq = P[h, 3]; Lqx = P[h, 4]; Ly = P[h, 5]; Lxy = P[h, 6]
            A[0, 0] = n; A[0, 1] = Sq; A[1, 1] = Sqq
            A[0, 2] = -(Ux - u * U1); A[1, 2] = -(Uqx - u * Uq); A[2, 2] = Uxx - 2 * u * Ux + u * u * U1
            A[0, 3] = l * L1 - Lx; A[1, 3] = l * Lq - Lqx; A[3, 3] = l * l * L1 - 2 * l * Lx + Lxx
            A[2, 3] = 0.0
            for r in range(4):
                for c in range(r):
                    A[r, c] = A[c, r]
            for r in range(4):
                A[r, r] += 1e-10
            v[0] = Sy; v[1] = Sqy; v[2] = -(Uxy - u * Uy); v[3] = l * Ly - Lxy
            coef = np.linalg.solve(A, v)
            out[i, j] = yy - (coef[0] * v[0] + coef[1] * v[1] + coef[2] * v[2] + coef[3] * v[3])
    return out


def make_grid(x):
    lo_, hi_ = np.percentile(x, [100 * TRIM, 100 * (1 - TRIM)])
    return np.arange(np.floor(lo_), np.ceil(hi_) + 1.0, 1.0)


def profile(y, x, q, use_numba_grid=False):
    grid = make_grid(x)
    if use_numba_grid:
        from model import _ssr_grid
        S = _ssr_grid(y, x, q, grid, -grid)
    else:
        S = fast_ssr(y, x, q, grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i], S.min(), S, grid       # l, u


# ------------------------------------------------------------------ regression (as ols_fe)
def ols_fe(Y, X, mcode, dcode):
    cnt = np.bincount(mcode, minlength=NM).astype(float); cnt[cnt == 0] = 1
    Yd = Y - (np.bincount(mcode, Y, NM) / cnt)[mcode]
    Xd = np.empty_like(X)
    for k in range(X.shape[1]):
        Xd[:, k] = X[:, k] - (np.bincount(mcode, X[:, k], NM) / cnt)[mcode]
    XtXi = np.linalg.inv(Xd.T @ Xd); b = XtXi @ (Xd.T @ Yd); e = Yd - Xd @ b
    Gm = np.empty((ND, X.shape[1]))
    for k in range(X.shape[1]):
        Gm[:, k] = np.bincount(dcode, Xd[:, k] * e, ND)
    ng, n, kk = ND, len(Y), X.shape[1]
    V = XtXi @ (Gm.T @ Gm) @ XtXi * ng / (ng - 1) * (n - 1) / (n - kk)
    return b, np.sqrt(np.diag(V))


# ------------------------------------------------------------------ one replication
def five_minute(b, q1):
    bb = b.reshape(NB, 5).mean(1)[RET]
    oo = q1.reshape(NB, 5).sum(1)[RET]
    cnt = np.bincount(BMON_RET, minlength=NM).astype(float)
    mu = np.bincount(BMON_RET, oo, NM) / cnt
    var = np.bincount(BMON_RET, (oo - mu[BMON_RET]) ** 2, NM) / (cnt - 1)
    qq = oo / np.sqrt(var)[BMON_RET]
    y = bb[POS] - bb[POS - 1]; x = bb[POS - 1]; q = qq[POS]
    return y, x, q


BIN_LABELS = ["in_inner", "in_outer", "out_0_5", "out_5_10", "out_10_20", "out_20_40", "out_40p"]


def estimate(y, x, q, keep_S=False):
    """Replicates empirics_main.py steps 1-2 on one sample. Returns a flat dict."""
    l_m = np.full(NM, np.nan); u_m = np.full(NM, np.nan); lr = np.zeros(NM)
    for mth in range(NM):
        idx = np.where(MCODE == mth)[0]
        if len(idx) < 500: continue
        ym, xm, qm = y[idx], x[idx], q[idx]
        l, u, smin, _, _ = profile(ym, xm, qm)
        Xl = np.column_stack([np.ones(len(ym)), qm, -xm])
        s0 = np.sum((ym - Xl @ np.linalg.lstsq(Xl, ym, rcond=None)[0]) ** 2)
        l_m[mth], u_m[mth], lr[mth] = l, u, len(ym) * np.log(s0 / smin)
    L = l_m[MCODE]; U = u_m[MCODE]
    up = -np.maximum(x - U, 0); dn = np.maximum(L - x, 0)
    exc = np.maximum(x - U, 0) + np.maximum(L - x, 0)
    inb = (exc == 0).astype(float)
    negx_in = -(x - (L + U) / 2) * inb
    out = {}
    c1, s1 = ols_fe(y, np.column_stack([q, -x]), MCODE, DCODE)
    c2, s2 = ols_fe(y, np.column_stack([q, up, dn]), MCODE, DCODE)
    c3, s3 = ols_fe(y, np.column_stack([q, up, dn, negx_in]), MCODE, DCODE)
    c4, s4 = ols_fe(y, np.column_stack([q, up, dn, q * BH, up * BH, dn * BH]), MCODE, DCODE)
    for tag, c, s, names in [("c1", c1, s1, ["q", "negx"]), ("c2", c2, s2, ["q", "up", "dn"]),
                             ("c3", c3, s3, ["q", "up", "dn", "negx_in"]),
                             ("c4", c4, s4, ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])]:
        for k, nm in enumerate(names):
            out[f"{tag}_{nm}"] = c[k]; out[f"{tag}_{nm}_se"] = s[k]
    # reversion by size of dislocation (emp_bins), net of column-2 beta
    center = (L + U) / 2; half = (U - L) / 2
    sgn = np.where(x > U, 1.0, np.where(x < L, -1.0, np.sign(x - center)))
    ry = -(y - c2[0] * q) * sgn
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(inb == 1, np.abs(x - center) / np.where(half == 0, np.nan, half), np.nan)
    masks = [(inb == 1) & (rel < 0.5), (inb == 1) & (rel >= 0.5)]
    edges = [0, 5, 10, 20, 40, np.inf]
    masks += [(exc > lo_) & (exc <= hi_) for lo_, hi_ in zip(edges[:-1], edges[1:])]
    dev = np.abs(x - center)
    base = np.where(inb == 1, dev, exc)
    for lab, m in zip(BIN_LABELS, masks):
        out[f"rho_{lab}"] = ry[m].sum() / base[m].sum() if m.sum() >= 100 else np.nan
    out["l_m"] = l_m; out["u_m"] = u_m
    out["width_med"] = float(np.nanmedian(u_m - l_m))
    out["share_in"] = float(inb.mean())
    out["sumLR"] = float(lr.sum())
    # calibration moments
    out["sd_y_closed"] = float(y[BH == 0].std()); out["sd_y_open"] = float(y[BH == 1].std())
    out["sd_x_month_med"] = float(np.median([x[MCODE == k].std() for k in range(NM)]))
    out["acf1_y"] = float(np.corrcoef(y[1:][POS[1:] - POS[:-1] == 1], y[:-1][POS[1:] - POS[:-1] == 1])[0, 1])
    c_ = POS[1:] - POS[:-1] == 1
    out["acf1_q"] = float(np.corrcoef(q[1:][c_], q[:-1][c_])[0, 1])
    out["corr_y_qlag"] = float(np.corrcoef(y[1:][c_], q[:-1][c_])[0, 1])
    return out


def run_one(args):
    dgp, r, P = args
    b, q1 = simulate(SEED0 + r, OPEN, MIN_MON, P["lo"], P["hi"], P["ctr"], P["kp"], P["km"], P["kl"],
                     P["beta1"], P["sig"], P["hm"], P["sq"], P["w1"], P["r1"], P["w2"], P["r2"], P["tdf"])
    y, x, q = five_minute(b, q1)
    out = estimate(y, x, q)
    out["dgp"] = dgp; out["r"] = r
    return out


# ------------------------------------------------------------------ helpers for calibration/DGPs
def k1(k5):
    return 1.0 - (1.0 - np.asarray(k5, float)) ** 0.2


def make_dgp(kind, base, emp_bands=None, tdf=0.0):
    """kind: 1 (state-dependent threshold), 2 (linear ECM), 3 (common-speed threshold)."""
    K = base["K5"]
    if emp_bands is None:
        lo = np.full(NM, -base["half_width"]); hi = np.full(NM, base["half_width"]); ctr = np.zeros(NM)
    else:
        lo, hi = emp_bands; ctr = (lo + hi) / 2
    P = dict(beta1=base["beta1"], sig=np.array([base["sig_c"], base["sig_o"]]), hm=np.asarray(base["hm"], float),
             sq=np.array([base["sq_c"], base["sq_o"]]), w1=base["w1"], r1=base["r1"],
             w2=base["w2"], r2=base["r2"], tdf=float(tdf), lo=lo.astype(float), hi=hi.astype(float),
             ctr=ctr.astype(float))
    z = np.zeros(2)
    if kind == 1:
        P.update(kp=k1([K["kp_c"], K["kp_o"]]), km=k1([K["km_c"], K["km_o"]]), kl=z)
    elif kind == 2:
        P.update(kp=z, km=z, kl=k1([K["k_lin"], K["k_lin"]]))
    else:
        P.update(kp=k1([K["kp_com"], K["kp_com"]]), km=k1([K["km_com"], K["km_com"]]), kl=z)
    return P


# ================================================================== main
if __name__ == "__main__":
    import pandas as pd
    from scipy.optimize import least_squares
    t_all = time.time()
    OUT = {"design": {}, "calibration": {}, "validation": {}, "results": {}}
    EMP = json.load(open(os.path.join(TAB, "emp_numbers.json")))
    rg = EMP["reg"]
    K5 = dict(kp_c=rg["bh"]["coef"]["up"], kp_o=rg["bh"]["coef"]["up"] + rg["bh"]["coef"]["up_bh"],
              km_c=rg["bh"]["coef"]["dn"], km_o=rg["bh"]["coef"]["dn"] + rg["bh"]["coef"]["dn_bh"],
              kp_com=rg["thr"]["coef"]["up"], km_com=rg["thr"]["coef"]["dn"], k_lin=rg["linear"]["coef"]["negx"])
    EMP_INB = rg["thr_in"]["coef"]["negx_in"]
    EMP_SUMLR = EMP["sum_supLR"]
    EMP_UPBH = rg["bh"]["coef"]["up_bh"]
    HALF = EMP["band_width_median"] / 2

    # ---------------- empirical moments (targets and checks)
    a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
    ai = a.index
    cons = np.r_[False, (ai[1:] - ai[:-1]) == pd.Timedelta("5min")]
    yv, qv = a.y.values, a.q.values
    emp_mom = dict(sd_y_closed=float(a.y[a.bh == 0].std()), sd_y_open=float(a.y[a.bh == 1].std()),
                   acf1_y=float(np.corrcoef(yv[1:][cons[1:]], yv[:-1][cons[1:]])[0, 1]),
                   acf1_q=float(np.corrcoef(qv[1:][cons[1:]], qv[:-1][cons[1:]])[0, 1]),
                   corr_y_qlag=float(np.corrcoef(yv[1:][cons[1:]], qv[:-1][cons[1:]])[0, 1]),
                   sd_x_month_med=float(a.groupby("month").x.std().median()),
                   share_in=float(EMP["share_inband"]), beta_c2=float(rg["thr"]["coef"]["q"]),
                   width_med=float(EMP["band_width_median"]), n_obs=int(len(a)),
                   robust_sd_y=float(np.subtract(*np.percentile(a.y, [75, 25])) / 1.349),
                   kurtosis_y=float(a.y.kurt()))
    OUT["calibration"]["empirical_moments"] = emp_mom
    OUT["design"]["sim_n_obs"] = int(len(POS))
    sdm = a.groupby("month").y.std()
    months_emp = sorted(sdm.index)
    assert len(months_emp) == NM, (len(months_emp), NM)
    hm = sdm.loc[months_emp].values / np.sqrt(np.mean(sdm.values ** 2))

    # ---------------- one-minute OFI: ACF fit and state scale
    p = pd.read_parquet(os.path.join(RAW, "panel_TRY.parquet")).loc["2024-01-01":"2026-08-31"]
    p = p[p.index.dayofweek < 5]
    mon = p.index.tz_localize(None).to_period("M")
    o_std = p.ofi / p.ofi.groupby(mon).transform("std")
    ih = (p.index.hour + 3) % 24
    bhm = (ih >= 9) & (ih < 17)
    sq_o, sq_c = float(o_std[bhm].std()), float(o_std[~bhm].std())
    o = o_std.reindex(pd.date_range(p.index.min(), p.index.max(), freq="1min")).values
    Kl = np.arange(1, 61); ac = []
    for k in Kl:
        mm = ~np.isnan(o[k:]) & ~np.isnan(o[:-k]); ac.append(np.corrcoef(o[k:][mm], o[:-k][mm])[0, 1])
    ac = np.array(ac)
    fit = least_squares(lambda th: th[0] * th[1] ** Kl + th[2] * th[3] ** Kl - ac, [0.04, 0.99, 0.1, 0.5],
                        bounds=([0, 0.5, 0, 0], [1, 0.9999, 1, 0.95]))
    w1, r1, w2, r2 = map(float, fit.x)
    del p, o, o_std
    OUT["calibration"]["ofi_acf_fit"] = dict(w1=w1, rho1=r1, w2=w2, rho2=r2, max_abs_err=float(np.abs(fit.fun).max()),
                                             acf_1m_lag1_5_60=[float(ac[0]), float(ac[4]), float(ac[59])],
                                             sd_open=sq_o, sd_closed=sq_c)
    OUT["calibration"]["month_vol_scale_h"] = dict(zip(months_emp, map(float, hm)))

    # ---------------- calibration by simulation on DGP1 (fixed seeds, 4 samples per step)
    cal_seeds = [900001, 900002, 900003, 900004]

    def cal_moments(base):
        P1 = make_dgp(1, base)
        sims = []
        for sd in cal_seeds:
            b, q1 = simulate(sd, OPEN, MIN_MON, P1["lo"], P1["hi"], P1["ctr"], P1["kp"], P1["km"], P1["kl"],
                             P1["beta1"], P1["sig"], P1["hm"], P1["sq"], P1["w1"], P1["r1"], P1["w2"], P1["r2"], 0.0)
            y, x, q = five_minute(b, q1)
            c2, _ = ols_fe(y, np.column_stack([q, -np.maximum(x - HALF, 0), np.maximum(-HALF - x, 0)]), MCODE, DCODE)
            sims.append([c2[0], y[BH == 0].std(), y[BH == 1].std(),
                         np.median([x[MCODE == k].std() for k in range(NM)])])
        return np.mean(sims, axis=0)

    print("calibrating design A ...", flush=True)
    baseA = dict(K5=K5, half_width=HALF, w1=w1, r1=r1, w2=w2, r2=r2, sq_o=sq_o, sq_c=sq_c, hm=hm,
                 beta1=0.35, sig_c=1.45, sig_o=1.75)
    for it in range(7):
        bq, sdc, sdo, sdx = cal_moments(baseA)
        print(f"  it {it}: beta1={baseA['beta1']:.4f} sig_c={baseA['sig_c']:.4f} sig_o={baseA['sig_o']:.4f} -> "
              f"beta={bq:.4f} sd_c={sdc:.4f} sd_o={sdo:.4f} sdx={sdx:.1f}", flush=True)
        if it == 6: break
        baseA["beta1"] *= emp_mom["beta_c2"] / bq
        baseA["sig_c"] *= emp_mom["sd_y_closed"] / sdc
        baseA["sig_o"] *= emp_mom["sd_y_open"] / sdo
    OUT["calibration"]["design_A"] = dict(beta1=baseA["beta1"], sig_closed=baseA["sig_c"], sig_open=baseA["sig_o"],
                                          w1=w1, rho1=r1, w2=w2, rho2=r2,
                                          fit=dict(beta_c2=float(bq), sd_y_closed=float(sdc), sd_y_open=float(sdo), sd_x_month_med=float(sdx)))
    print("calibrating design B ...", flush=True)
    baseB = dict(baseA, w1=0.0)
    for it in range(8):
        bq, sdc, sdo, sdx = cal_moments(baseB)
        print(f"  it {it}: beta1={baseB['beta1']:.4f} sig_c={baseB['sig_c']:.4f} sig_o={baseB['sig_o']:.4f} -> "
              f"beta={bq:.4f} sd_c={sdc:.4f} sd_o={sdo:.4f} sdx={sdx:.1f}", flush=True)
        if it == 7: break
        baseB["beta1"] *= emp_mom["beta_c2"] / bq
        fct = (emp_mom["sd_x_month_med"] / sdx) ** 1.5
        baseB["sig_c"] *= fct; baseB["sig_o"] *= fct
    OUT["calibration"]["design_B"] = dict(beta1=baseB["beta1"], sig_closed=baseB["sig_c"], sig_open=baseB["sig_o"],
                                          w1=0.0, w2=w2, rho2=r2,
                                          fit=dict(beta_c2=float(bq), sd_y_closed=float(sdc), sd_y_open=float(sdo), sd_x_month_med=float(sdx)))
    OUT["calibration"]["truth_K5"] = K5
    OUT["calibration"]["truth_K1"] = {k: float(k1(v)) for k, v in K5.items()}
    OUT["calibration"]["band"] = [-HALF, HALF]

    # ---------------- DGP list
    eb = EMP["bands"]; months = sorted(eb)
    assert months == months_emp
    emp_lo = np.array([eb[m]["l"] for m in months], float); emp_hi = np.array([eb[m]["u"] for m in months], float)
    DGPS = {}
    for kind in (1, 2, 3):
        DGPS[f"A{kind}"] = (make_dgp(kind, baseA), R_MAIN)
    for kind in (1, 2, 3):
        DGPS[f"B{kind}"] = (make_dgp(kind, baseB), R_MAIN)
    for kind in (1, 2, 3):
        DGPS[f"A{kind}t"] = (make_dgp(kind, baseA, tdf=3.0), R_VAR)
    for kind in (1, 2, 3):
        DGPS[f"A{kind}e"] = (make_dgp(kind, baseA, emp_bands=(emp_lo, emp_hi)), R_VAR)
    TRUE_BANDS = {k: (v[0]["lo"], v[0]["hi"]) for k, v in DGPS.items()}

    # ---------------- validation: fast_ssr == model._ssr_grid
    print("validating fast_ssr against _ssr_grid ...", flush=True)
    from model import _ssr_grid
    val = []
    for mth, d in a.groupby("month"):
        y_, x_, q_ = d.y.values, d.x.values, d.q.values
        g = make_grid(x_)
        S0 = _ssr_grid(y_, x_, q_, g, -g); S1 = fast_ssr(y_, x_, q_, g)
        fin = np.isfinite(S0)
        i0, j0 = np.unravel_index(np.argmin(S0), S0.shape)
        val.append(dict(src="empirical", same_argmin=bool(np.argmin(S0) == np.argmin(S1)),
                        max_rel=float(np.max(np.abs(S0[fin] - S1[fin]) / S0[fin])),
                        l_u=[float(g[j0]), float(g[i0])], emp_l_u=[eb[mth]["l"], eb[mth]["u"]]))
    for dg in ["A1", "A2", "A3", "B1", "B2", "B3"]:
        P = DGPS[dg][0]
        b, q1 = simulate(SEED0 + 999_000, OPEN, MIN_MON, P["lo"], P["hi"], P["ctr"], P["kp"], P["km"], P["kl"],
                         P["beta1"], P["sig"], P["hm"], P["sq"], P["w1"], P["r1"], P["w2"], P["r2"], P["tdf"])
        y, x, q = five_minute(b, q1)
        for mth in range(0, NM, 4):
            idx = MCODE == mth
            g = make_grid(x[idx])
            S0 = _ssr_grid(y[idx], x[idx], q[idx], g, -g); S1 = fast_ssr(y[idx], x[idx], q[idx], g)
            fin = np.isfinite(S0)
            val.append(dict(src=dg, same_argmin=bool(np.argmin(S0) == np.argmin(S1)),
                            max_rel=float(np.max(np.abs(S0[fin] - S1[fin]) / S0[fin]))))
    OUT["validation"] = dict(n_surfaces=len(val), share_same_argmin=float(np.mean([v["same_argmin"] for v in val])),
                             max_rel_diff=float(max(v["max_rel"] for v in val)),
                             empirical_same_argmin=all(v["same_argmin"] for v in val if v["src"] == "empirical"),
                             empirical_bands_reproduced=all(v["l_u"] == v["emp_l_u"] for v in val if v["src"] == "empirical"))
    print(OUT["validation"], flush=True)

    # ---------------- Monte Carlo
    import multiprocessing as mp
    os.environ["NUMBA_NUM_THREADS"] = "1"
    tasks = [(dg, r, P) for dg, (P, R) in DGPS.items() for r in range(R)]
    t0 = time.time(); rows = []
    if os.environ.get("ROWS"):          # re-summarise saved replications (same seeds, same DGPs)
        rows = list(np.load(os.environ["ROWS"], allow_pickle=True))
        assert sorted((r_["dgp"], r_["r"]) for r_ in rows) == sorted((t[0], t[1]) for t in tasks)
        print(f"loaded {len(rows)} saved replications from {os.environ['ROWS']}", flush=True)
        OUT["design"]["mc_seconds"] = float(json.load(open(os.path.join(TAB, f"{NAME}.json")))["design"]["mc_seconds"])
    else:
        print(f"running {len(tasks)} replications on {NPROC} processes ...", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(NPROC) as pool:
            for k, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=4)):
                rows.append(res)
                if (k + 1) % 250 == 0:
                    print(f"  {k+1}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
        OUT["design"]["mc_seconds"] = time.time() - t0
    if os.environ.get("SCRATCH") and not os.environ.get("ROWS"):
        np.save(os.path.join(os.environ["SCRATCH"], f"{NAME}_rows.npy"), np.array(rows, dtype=object), allow_pickle=True)

    # ---------------- summarise
    def S(dg, key):
        return np.array([r_[key] for r_ in rows if r_["dgp"] == dg], float)

    def stat(est, truth=None, se=None):
        est = np.asarray(est, float)
        d = dict(mean=float(np.nanmean(est)), median=float(np.nanmedian(est)), sd=float(np.nanstd(est)),
                 p05=float(np.nanpercentile(est, 5)), p95=float(np.nanpercentile(est, 95)))
        if truth is not None:
            d.update(truth=float(truth), bias=float(np.nanmean(est) - truth),
                     rmse=float(np.sqrt(np.nanmean((est - truth) ** 2))),
                     median_abs_err=float(np.nanmedian(np.abs(est - truth))))
            if se is not None:
                d["coverage95"] = float(np.mean(np.abs(est - truth) <= 1.96 * se))
        if se is not None:
            d["mean_se"] = float(np.nanmean(se))
        return d

    for dg, (P, R) in DGPS.items():
        kind = int(dg[1])
        rr = [r_ for r_ in rows if r_["dgp"] == dg]
        res = dict(R=len(rr), tdf=P["tdf"], ofi_w1=P["w1"],
                   bands="empirical monthly" if dg.endswith("e") else f"[{-HALF:.0f},{HALF:.0f}]")
        Lh = np.vstack([r_["l_m"] for r_ in rr]); Uh = np.vstack([r_["u_m"] for r_ in rr])
        lo_t, hi_t = TRUE_BANDS[dg]
        res["width_median"] = stat(S(dg, "width_med"), None if kind == 2 else float(np.median(hi_t - lo_t)))
        res["share_inband"] = stat(S(dg, "share_in"))
        if kind != 2:
            res["l_edge"] = dict(truth_mean=float(lo_t.mean()), bias=float(np.mean(Lh - lo_t)), rmse=float(np.sqrt(np.mean((Lh - lo_t) ** 2))),
                                 median_abs_err=float(np.median(np.abs(Lh - lo_t))), sd=float(np.std(Lh - lo_t)))
            res["u_edge"] = dict(truth_mean=float(hi_t.mean()), bias=float(np.mean(Uh - hi_t)), rmse=float(np.sqrt(np.mean((Uh - hi_t) ** 2))),
                                 median_abs_err=float(np.median(np.abs(Uh - hi_t))), sd=float(np.std(Uh - hi_t)))
            res["width_all_months"] = dict(truth_median=float(np.median(hi_t - lo_t)), est_median=float(np.median(Uh - Lh)),
                                           bias_mean=float(np.mean((Uh - Lh) - (hi_t - lo_t))),
                                           share_zero_width=float(np.mean(Uh - Lh == 0)))
        else:
            res["width_all_months"] = dict(est_median=float(np.median(Uh - Lh)), est_mean=float(np.mean(Uh - Lh)),
                                           share_zero_width=float(np.mean(Uh - Lh == 0)),
                                           share_ge_20=float(np.mean(Uh - Lh >= 20)))
        K = K5
        if kind == 1:
            tr = dict(kp_c=K["kp_c"], kp_o=K["kp_o"], km_c=K["km_c"], km_o=K["km_o"], up_bh=K["kp_o"] - K["kp_c"],
                      dn_bh=K["km_o"] - K["km_c"], ratio=K["kp_o"] / K["kp_c"], inb=0.0, kp=None, km=None)
        elif kind == 3:
            tr = dict(kp_c=K["kp_com"], kp_o=K["kp_com"], km_c=K["km_com"], km_o=K["km_com"], up_bh=0.0, dn_bh=0.0,
                      ratio=1.0, inb=0.0, kp=K["kp_com"], km=K["km_com"])
        else:
            tr = dict(kp_c=None, kp_o=None, km_c=None, km_o=None, up_bh=0.0, dn_bh=0.0, ratio=None, inb=K["k_lin"], kp=None, km=None)
        kp_c, kp_c_se = S(dg, "c4_up"), S(dg, "c4_up_se")
        kp_o = kp_c + S(dg, "c4_up_bh"); km_c, km_c_se = S(dg, "c4_dn"), S(dg, "c4_dn_se"); km_o = km_c + S(dg, "c4_dn_bh")
        res["kp_closed"] = stat(kp_c, tr["kp_c"], kp_c_se)
        res["kp_open"] = stat(kp_o, tr["kp_o"])
        res["km_closed"] = stat(km_c, tr["km_c"], km_c_se)
        res["km_open"] = stat(km_o, tr["km_o"])
        res["up_bh"] = stat(S(dg, "c4_up_bh"), tr["up_bh"], S(dg, "c4_up_bh_se"))
        res["dn_bh"] = stat(S(dg, "c4_dn_bh"), tr["dn_bh"], S(dg, "c4_dn_bh_se"))
        res["q_bh"] = stat(S(dg, "c4_q_bh"), 0.0, S(dg, "c4_q_bh_se"))
        ratio = kp_o / kp_c
        res["ratio_open_closed_up"] = stat(ratio, tr["ratio"])
        res["ratio_open_closed_up"]["share_closed_nonpositive"] = float(np.mean(kp_c <= 0))
        inbc, inbs = S(dg, "c3_negx_in"), S(dg, "c3_negx_in_se")
        res["inband_coef"] = stat(inbc, tr["inb"], inbs)
        res["c3_kp"] = stat(S(dg, "c3_up"), tr["kp"], S(dg, "c3_up_se"))
        res["c3_km"] = stat(S(dg, "c3_dn"), tr["km"], S(dg, "c3_dn_se"))
        res["c2_kp"] = stat(S(dg, "c2_up"), tr["kp"], S(dg, "c2_up_se"))
        res["c2_km"] = stat(S(dg, "c2_dn"), tr["km"], S(dg, "c2_dn_se"))
        res["c1_linear"] = stat(S(dg, "c1_negx"), K["k_lin"] if kind == 2 else None, S(dg, "c1_negx_se"))
        res["beta_c2"] = stat(S(dg, "c2_q"))
        t_inb = inbc / inbs; t_up = S(dg, "c4_up_bh") / S(dg, "c4_up_bh_se")
        t_dn = S(dg, "c4_dn_bh") / S(dg, "c4_dn_bh_se"); t_q = S(dg, "c4_q_bh") / S(dg, "c4_q_bh_se")
        res["reject5"] = dict(inband_eq0=float(np.mean(np.abs(t_inb) > 1.96)),
                              inband_pos_onesided=float(np.mean(t_inb > 1.645)),
                              inband_neg_onesided=float(np.mean(t_inb < -1.645)),
                              inband_sig_positive=float(np.mean(t_inb > 1.96)),
                              upbh_eq0=float(np.mean(np.abs(t_up) > 1.96)),
                              upbh_sig_positive=float(np.mean(t_up > 1.96)),
                              upbh_t_ge_empirical=float(np.mean(t_up >= EMP_UPBH / rg["bh"]["se"]["up_bh"])),
                              dnbh_eq0=float(np.mean(np.abs(t_dn) > 1.96)),
                              qbh_eq0=float(np.mean(np.abs(t_q) > 1.96)))
        t3p = S(dg, "c3_up") / S(dg, "c3_up_se"); t3m = S(dg, "c3_dn") / S(dg, "c3_dn_se")
        flat = (t3p > 1.96) & (t3m > 1.96) & (t_inb < 1.96) & (S(dg, "width_med") >= 20)
        res["flat_band_pattern"] = float(np.mean(flat))
        res["flat_band_pattern_without_width"] = float(np.mean((t3p > 1.96) & (t3m > 1.96) & (t_inb < 1.96)))
        res["mean_t_upbh"] = float(np.mean(t_up))
        res["P_inband_le_emp"] = float(np.mean(inbc <= EMP_INB))
        res["P_inband_ub_le_0008"] = float(np.mean(inbc + 1.96 * inbs <= 0.0008 + 1e-12))
        res["P_sumLR_ge_emp"] = float(np.mean(S(dg, "sumLR") >= EMP_SUMLR))
        res["sumLR"] = stat(S(dg, "sumLR"))
        res["sumLR"].update(max=float(S(dg, "sumLR").max()), p99=float(np.percentile(S(dg, "sumLR"), 99)))
        res["P_upbh_ge_emp"] = float(np.mean(S(dg, "c4_up_bh") >= EMP_UPBH))
        res["bins_rho"] = {lab: stat(S(dg, f"rho_{lab}")) for lab in BIN_LABELS}
        res["moments"] = {k: float(np.nanmean(S(dg, k))) for k in
                          ["sd_y_closed", "sd_y_open", "acf1_y", "acf1_q", "corr_y_qlag", "sd_x_month_med", "share_in"]}
        OUT["results"][dg] = res
    OUT["empirical_reference"] = dict(inband=EMP_INB, inband_se=rg["thr_in"]["se"]["negx_in"], sumLR=EMP_SUMLR,
                                      up_bh=EMP_UPBH, up_bh_se=rg["bh"]["se"]["up_bh"],
                                      bins={b_["label"]: b_["rho"] for b_ in EMP["bins"]})
    OUT["design"].update(R_main=R_MAIN, R_variants=R_VAR, seeds=f"{SEED0}+r, common across DGPs",
                         calendar="2024-01-01..2026-08-31 UTC, weekdays observed outside 20:30-22:30 UTC, 4-week burn-in",
                         n_months=NM, n_days=ND)
    OUT["design"]["total_seconds"] = time.time() - t_all
    json.dump(OUT, open(os.path.join(TAB, f"{NAME}.json"), "w"), indent=1, default=float)

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    COL = {"1": "#1f4e79", "2": "#c0504d", "3": "#7f7f7f"}
    LAB = {"1": "DGP1: state-dependent threshold", "2": "DGP2: linear ECM", "3": "DGP3: common-speed threshold"}
    fig, ax = plt.subplots(2, 4, figsize=(8.6, 4.4))
    for row_, des in enumerate(["A", "B"]):
        for kk in "123":
            dg = des + kk
            ax[row_, 0].hist(S(dg, "c3_negx_in"), bins=40, color=COL[kk], alpha=0.5, label=LAB[kk])
            ax[row_, 1].hist(S(dg, "c4_up_bh"), bins=40, color=COL[kk], alpha=0.5)
            ax[row_, 2].hist(S(dg, "width_med"), bins=40, color=COL[kk], alpha=0.5)
            ax[row_, 3].hist(S(dg, "sumLR"), bins=40, color=COL[kk], alpha=0.5)
        ax[row_, 0].axvline(EMP_INB, color="k", lw=1, ls="--"); ax[row_, 0].axvline(0, color="k", lw=0.4)
        ax[row_, 0].axvline(K5["k_lin"], color=COL["2"], lw=0.8, ls=":")
        ax[row_, 1].axvline(EMP_UPBH, color="k", lw=1, ls="--"); ax[row_, 1].axvline(K5["kp_o"] - K5["kp_c"], color=COL["1"], lw=0.8, ls=":")
        ax[row_, 1].axvline(0, color="k", lw=0.4)
        ax[row_, 2].axvline(EMP["band_width_median"], color="k", lw=1, ls="--")
        ax[row_, 3].axvline(EMP_SUMLR, color="k", lw=1, ls="--")
        ax[row_, 0].set_ylabel(f"Design {des}")
    for j, t in enumerate(["In-band coefficient, col. (3)", r"BH increment on $\kappa^+$, col. (4)",
                           "Median monthly band width (bp)", "Sum of monthly sup-LR"]):
        ax[0, j].set_title(t, fontsize=8)
    ax[0, 0].legend(frameon=False, fontsize=6, loc="upper left")
    for a_ in ax.ravel():
        a_.set_yticks([])
    fig.text(0.5, 0.005, "Dashed black: empirical estimate. Dotted: true value (DGP1 increment; DGP2 linear speed).",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(FIG, f"{NAME}.pdf")); plt.close(fig)

    # ---------------- table
    def f4(v): return "" if v is None or not np.isfinite(v) else f"{v:.4f}"
    def f1(v): return "" if v is None or not np.isfinite(v) else f"{v:.1f}"
    def f2(v): return "" if v is None or not np.isfinite(v) else f"{v:.2f}"
    Rr = OUT["results"]
    L = ["\\begin{tabular}{lccccccc}", "\\toprule",
         " & & \\multicolumn{3}{c}{Design A} & \\multicolumn{3}{c}{Design B} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         " & True & Mean & RMSE & Cover. & Mean & RMSE & Cover. \\\\", "\\midrule"]

    def prow(lab, key, kind, fmt=f4):
        dA, dB = Rr[f"A{kind}"][key], Rr[f"B{kind}"][key]
        return (f"{lab} & {fmt(dA.get('truth'))} & {fmt(dA['mean'])} & {fmt(dA.get('rmse'))} & {f2(dA.get('coverage95'))} & "
                f"{fmt(dB['mean'])} & {fmt(dB.get('rmse'))} & {f2(dB.get('coverage95'))} \\\\")

    def erow(lab, key, kind):
        dA, dB = Rr[f"A{kind}"][key], Rr[f"B{kind}"][key]
        return (f"{lab} & {f1(dA['truth_mean'])} & {f1(dA['truth_mean'] + dA['bias'])} & {f1(dA['rmse'])} & & "
                f"{f1(dB['truth_mean'] + dB['bias'])} & {f1(dB['rmse'])} & \\\\")

    for kind, title in [(1, "DGP1: threshold, banking-state speeds"), (3, "DGP3: threshold, same speed in both states")]:
        L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}} \\\\")
        L.append(erow("Lower edge $\\ell_m$ (bp)", "l_edge", kind))
        L.append(erow("Upper edge $u_m$ (bp)", "u_edge", kind))
        L.append(prow("$\\kappa^+$, banks closed", "kp_closed", kind))
        L.append(prow("$\\kappa^+$, banks open", "kp_open", kind))
        L.append(prow("$\\kappa^+$, BH increment", "up_bh", kind))
        L.append(prow("$\\kappa^-$, banks closed", "km_closed", kind))
        L.append(prow("$\\kappa^-$, banks open", "km_open", kind))
        rA, rB = Rr[f"A{kind}"]["ratio_open_closed_up"], Rr[f"B{kind}"]["ratio_open_closed_up"]
        L.append(f"$\\kappa^+_{{open}}/\\kappa^+_{{closed}}$, median & {f2(rA['truth'])} & {f2(rA['median'])} & & & {f2(rB['median'])} & & \\\\")
        L.append(prow("In-band coefficient, col.\\ (3)", "inband_coef", kind))
        L.append("\\addlinespace")
    L.append("\\multicolumn{8}{l}{\\textit{DGP2: linear error correction, no band}} \\\\")
    L.append(prow("Linear $\\kappa$, col.\\ (1)", "c1_linear", 2))
    L.append(prow("In-band coefficient, col.\\ (3)", "inband_coef", 2))
    L.append(f"Median band width (bp) & -- & {f1(Rr['A2']['width_median']['mean'])} & & & {f1(Rr['B2']['width_median']['mean'])} & & \\\\")
    L.append("\\midrule")
    L.append(" & \\multicolumn{3}{c}{In-band $=0$} & \\multicolumn{2}{c}{$\\kappa^+$ BH incr.\\ $=0$} & \\multicolumn{2}{c}{DGP2} \\\\")
    L.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
    L.append("Rejection rate at 5\\% & DGP1 & DGP3 & DGP2 & DGP1 & DGP3 & Flat band & $\\sum$LR$\\ge$464 \\\\")
    for lab, des in [("Design A", "A"), ("Design B", "B"), ("A, $t_3$ shocks", "A@t"), ("A, empirical bands", "A@e")]:
        nm = lambda k: des.replace("@", str(k)) if "@" in des else des + str(k)
        L.append(f"{lab} & {f2(Rr[nm(1)]['reject5']['inband_eq0'])} & {f2(Rr[nm(3)]['reject5']['inband_eq0'])} & "
                 f"{f2(Rr[nm(2)]['reject5']['inband_eq0'])} & {f2(Rr[nm(1)]['reject5']['upbh_eq0'])} & "
                 f"{f2(Rr[nm(3)]['reject5']['upbh_eq0'])} & {f2(Rr[nm(2)]['flat_band_pattern'])} & {f2(Rr[nm(2)]['P_sumLR_ge_emp'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, f"{NAME}.tex"), "w").write("\n".join(L) + "\n")
    print(json.dumps({k: OUT["results"][k]["reject5"] for k in OUT["results"]}, indent=1))
    print(f"done in {time.time()-t_all:.0f}s")
