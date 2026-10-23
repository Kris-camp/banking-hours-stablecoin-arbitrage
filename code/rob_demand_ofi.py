"""Robustness: demand-only order-flow control in equation (emp).

Equation (emp) controls for TOTAL signed order flow q. Total flow contains the
arbitrageurs' own aggressive large trades, which lean against the premium outside the band, so
beta*q may absorb part of the arbitrage correction and kappa may understate arbitrage.

DESIGN
------
Order-flow controls
  T  total OFI q, standardized within month (published specification).
  D  demand OFI z_dem = (ofi_small + ofi_mid) / its within-month SD on the a5 sample, exactly as
     in empirics_mech.py (aggressive trades < 10,000 USDT).
  N  no order-flow control (reduced form).
Band treatment
  primary   "reprofiled": every band (monthly; month x banking state) is re-profiled with the same
            control that enters the slope regression (profile on y, x, flow; under N the flow
            column is zero, so the profile is the threshold model without order flow). This is
            the full re-estimation of the model under each control.
  secondary "fixed": bands held at their published values; isolates the slope effect.
  Check: re-profiling under T must reproduce the published bands exactly (and T/fixed must
  reproduce the published coefficients).
Specifications (all day-clustered with the finite-sample correction of ols/ols_fe)
  A  Table emp_main col (4): y on [f, up, dn, f*BH, up*BH, dn*BH], month FE, monthly bands.
  B  Table emp_state col (1): same with month-by-state bands and month-by-state FE.
  C  Table emp_flow col (1): R = -s*(y - beta_c*f_c) with beta_c the col (2) (thr, monthly bands,
     month FE) estimate under control c (R = -s*y under N); regressors d_out, d_out*bh_eff, d_in,
     d_in*bh_eff (bh_eff excludes the HOL weekday holidays); state bands; month FE. Col (2):
     A_L = -s*z_large on the same bands. Cross-equation test: joint day-clustered covariance of
     (1) and (2), difference of log open/closed ratios (delta method), as in empirics_mech.py.
     Supplements: (C2) d_out split into above/below components to report kappa- on the price
     side; (C3) holiday interaction (d_out*bh_hol) as in the published holiday placebo.
Reported per control: closed speed, BH increment (s.e.), open speed, open/closed ratio and the
delta-method s.e. of its log, for kappa+ and kappa-; beta by state; long-run impact beta/kappa by
state and the closed/open ratio of long-run impacts with the s.e. of its log (undefined under N).
Cross-control test: joint day-clustered covariance of the T and D (and T and N) regressions on the
same days; s.e. of the difference in the BH increment of kappa+ and in the log open/closed ratio.
Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, data_raw/ofi_size/*.parquet, tables/emp_numbers.json,
        tables/emp_state.json, tables/emp_mech.json.
Outputs: tables/rob_demand_ofi.json, tables/rob_demand_ofi.tex (primary, re-profiled bands),
tables/rob_demand_ofi_fixedbands.tex (secondary).

BAND DIAGNOSTICS. Re-profiling moves many cell-level band edges by large amounts, so the script also reports
(i) for every band cell, the LR statistic n*log(SSR(published band)/SSR(re-profiled band)) under
the D and N models, against the Hansen (2000) 95% value 7.35; (ii) the single band cell whose
restoration to the published band moves each headline statistic most; (iii) leave-one-month-out
ranges of the headline ratios.
"""
import os, glob, json, time
import numpy as np, pandas as pd
from model import _ssr_grid

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB = os.path.join(HERE, "..", "data_raw"), os.path.join(HERE, "..", "tables")
NAME = "demand_ofi"
t0 = time.time()

# ------------------------------------------------------------------ data
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
a["day"] = a.index.date
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
a = a.join(sb, how="inner")
assert a[["ofi_small", "ofi_mid", "ofi_large"]].notna().all().all(), "size-class flow missing"
for c in ["small", "mid", "large"]:
    a[f"z_{c}"] = a[f"ofi_{c}"] / a[f"ofi_{c}"].groupby(a.month).transform("std")
a["z_dem"] = (a.ofi_small + a.ofi_mid) / (a.ofi_small + a.ofi_mid).groupby(a.month).transform("std")
a["zero"] = 0.0
a["state"] = np.where(a.bh == 1, "open", "closed")
a["ms"] = a.month + "_" + a.state

HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
a["hol"] = np.isin(a.day, pd.to_datetime(HOL).date).astype(float)
a["bh_eff"] = a.bh * (1 - a.hol)
a["bh_hol"] = a.bh * a.hol

OUT = dict(name=NAME, n=int(len(a)), n_days=int(a.day.nunique()))
# how total flow relates to its size components
cz = a[["q", "z_dem", "z_large", "z_small", "z_mid"]].corr()
X_ = np.column_stack([np.ones(len(a)), a.z_dem, a.z_large])
cq = np.linalg.lstsq(X_, a.q.values, rcond=None)[0]
OUT["flow_diag"] = dict(corr_q_zdem=float(cz.loc["q", "z_dem"]), corr_q_zlarge=float(cz.loc["q", "z_large"]),
                        corr_zdem_zlarge=float(cz.loc["z_dem", "z_large"]),
                        q_on_zdem_zlarge=[float(cq[1]), float(cq[2])],
                        share_vol_large=float(a.vol_large.sum() / (a.vol_small + a.vol_mid + a.vol_large).sum()))

CTRL = {"T": "q", "D": "z_dem", "N": "zero"}


# ------------------------------------------------------------------ estimators
def profile(y, x, f, trim=0.05, step=1.0):
    """Identical to empirics_main/empirics_state: returns (l, u)."""
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, f, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]


def ols(d, ycol, cols, fe="month"):
    """Within-FE OLS, day-clustered with the finite-sample correction; returns per-day scores."""
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    gm = d.groupby(fe)
    Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes, uniq = pd.factorize(d.day)
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    adj = ng / (ng - 1) * (n - 1) / (n - k)
    S = G @ XtXi.T                                  # per-day influence of each coefficient
    V = S.T @ S * adj
    return dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), V=V, S=S, days=uniq,
                adj=adj, cols=list(cols), n=n)


def joint(regs):
    """Stack per-day scores of several regressions estimated on the same days."""
    days = regs[0]["days"]
    for r in regs[1:]:
        assert (r["days"] == days).all()
    S = np.hstack([r["S"] for r in regs])
    b = np.concatenate([[r["b"][c] for c in r["cols"]] for r in regs])
    return b, S.T @ S * regs[0]["adj"]


def dm(grad, V):
    return float(np.sqrt(grad @ V @ grad))


def state_summary(r, f, up, dn, fb, upb, dnb):
    """Speeds, ratios and long-run impacts by banking state from one regression."""
    cols = r["cols"]; b = np.array([r["b"][c] for c in cols]); V = r["V"]; k = len(cols)
    ix = {c: i for i, c in enumerate(cols)}
    out = {}
    for side, c0, c1 in [("kp", up, upb), ("km", dn, dnb)]:
        k0, k1 = r["b"][c0], r["b"][c1]
        g = np.zeros(k); g[ix[c0]] = 1 / (k0 + k1) - 1 / k0; g[ix[c1]] = 1 / (k0 + k1)
        go = np.zeros(k); go[ix[c0]] = 1; go[ix[c1]] = 1
        out[side] = dict(closed=[float(k0), float(r["se"][c0])], incr=[float(k1), float(r["se"][c1])],
                         open=[float(k0 + k1), dm(go, V)], ratio=float((k0 + k1) / k0) if k0 > 0 else None,
                         se_logratio=dm(g, V) if k0 > 0 and k0 + k1 > 0 else None)
    if f is not None:
        b0, b1 = r["b"][f], r["b"][fb]
        gbo = np.zeros(k); gbo[ix[f]] = 1; gbo[ix[fb]] = 1
        out["beta"] = dict(closed=[float(b0), float(r["se"][f])], incr=[float(b1), float(r["se"][fb])],
                           open=[float(b0 + b1), dm(gbo, V)])
        for side, c0, c1 in [("kp", up, upb), ("km", dn, dnb)]:
            k0, k1 = r["b"][c0], r["b"][c1]
            lrc, lro = b0 / k0, (b0 + b1) / (k0 + k1)
            gc = np.zeros(k); gc[ix[f]] = 1 / k0; gc[ix[c0]] = -b0 / k0 ** 2
            go = np.zeros(k); go[ix[f]] = 1 / (k0 + k1); go[ix[fb]] = 1 / (k0 + k1)
            go[ix[c0]] = -(b0 + b1) / (k0 + k1) ** 2; go[ix[c1]] = -(b0 + b1) / (k0 + k1) ** 2
            # log(LR_closed / LR_open) = log b0 - log k0 - log(b0+b1) + log(k0+k1)
            gl = np.zeros(k)
            gl[ix[f]] = 1 / b0 - 1 / (b0 + b1); gl[ix[fb]] = -1 / (b0 + b1)
            gl[ix[c0]] = -1 / k0 + 1 / (k0 + k1); gl[ix[c1]] = 1 / (k0 + k1)
            out["lr_" + side] = dict(closed=[float(lrc), dm(gc, V)], open=[float(lro), dm(go, V)],
                                     ratio_closed_open=float(lrc / lro), se_logratio=dm(gl, V))
    return out


def cross_control(rT, rX, c0, c1, c0x=None, c1x=None):
    """Difference X - T in the BH increment and in the log open/closed ratio, joint covariance."""
    c0x, c1x = c0x or c0, c1x or c1
    b, V = joint([rT, rX]); kT = len(rT["cols"])
    i0, i1 = rT["cols"].index(c0), rT["cols"].index(c1)
    j0, j1 = kT + rX["cols"].index(c0x), kT + rX["cols"].index(c1x)
    g = np.zeros(len(b)); g[j1] = 1; g[i1] = -1
    d_incr = b[j1] - b[i1]
    out = dict(d_incr=[float(d_incr), dm(g, V)])
    if b[i0] > 0 and b[j0] > 0 and b[i0] + b[i1] > 0 and b[j0] + b[j1] > 0:
        g = np.zeros(len(b))
        g[j0] = 1 / (b[j0] + b[j1]) - 1 / b[j0]; g[j1] = 1 / (b[j0] + b[j1])
        g[i0] = -(1 / (b[i0] + b[i1]) - 1 / b[i0]); g[i1] = -1 / (b[i0] + b[i1])
        d_lr = np.log((b[j0] + b[j1]) / b[j0]) - np.log((b[i0] + b[i1]) / b[i0])
        out["d_logratio"] = [float(d_lr), dm(g, V)]
    return out


# ------------------------------------------------------------------ bands under each control
def monthly_bands(f):
    l, u = pd.Series(np.nan, index=a.index), pd.Series(np.nan, index=a.index)
    for per, d in a.groupby("bandper"):
        if len(d) < 500: continue
        l_, u_ = profile(d.y.values, d.x.values, d[f].values.astype(float))
        l[d.index], u[d.index] = l_, u_
    return l.values, u.values


def state_bands(f):
    l, u = pd.Series(np.nan, index=a.index), pd.Series(np.nan, index=a.index)
    for (m, st), d in a.groupby(["month", "state"]):
        if len(d) < 300: continue
        l_, u_ = profile(d.y.values, d.x.values, d[f].values.astype(float))
        l[d.index], u[d.index] = l_, u_
    return l.values, u.values


BANDS = {"fixed": {}, "reprofiled": {}}
for c, f in CTRL.items():
    BANDS["fixed"][c] = dict(M=(a.l.values, a.u.values), S=(a.l_s.values, a.u_s.values))
    BANDS["reprofiled"][c] = dict(M=monthly_bands(f), S=state_bands(f))
    print(f"bands {c} done {time.time() - t0:.0f}s", flush=True)
chk = BANDS["reprofiled"]["T"]
OUT["check_reprofile_T"] = dict(monthly_max_abs_diff=float(np.nanmax(np.abs(np.r_[chk["M"][0] - a.l.values, chk["M"][1] - a.u.values]))),
                                state_max_abs_diff=float(np.nanmax(np.abs(np.r_[chk["S"][0] - a.l_s.values, chk["S"][1] - a.u_s.values]))))


def band_stats(l, u, by):
    w = pd.Series(u - l, index=a.index).groupby(by).first()
    return float(w.mean()), float(w.median())


OUT["band_widths"] = {}
for bt in BANDS:
    for c in CTRL:
        lM, uM = BANDS[bt][c]["M"]; lS, uS = BANDS[bt][c]["S"]
        wM = band_stats(lM, uM, a.month.values)
        ws = pd.DataFrame({"w": uS - lS, "m": a.month.values, "s": a.state.values}).groupby(["m", "s"]).w.first().unstack()
        inb = float(np.mean((a.x.values >= lM) & (a.x.values <= uM)))
        OUT["band_widths"][f"{bt}_{c}"] = dict(monthly_mean=wM[0], monthly_median=wM[1], share_inband=inb,
                                              state_open_mean=float(ws.open.mean()), state_closed_mean=float(ws.closed.mean()))


# ------------------------------------------------------------------ A and B: emp_main col (4), emp_state col (1)
def design(l, u, f):
    d = a[["y", "x", "day", "month", "ms", "bh", "bh_eff", "bh_hol", "z_large"]].copy()
    d["f"] = a[f].values
    d["up"] = -np.maximum(d.x - u, 0); d["dn"] = np.maximum(l - d.x, 0)
    for c in ["f", "up", "dn"]:
        d[c + "_bh"] = d[c] * d.bh
    return d


RES = {}
for bt in BANDS:
    for c, f in CTRL.items():
        flow_cols = [] if c == "N" else ["f"]
        flow_bh = [] if c == "N" else ["f_bh"]
        # A: monthly bands, month FE
        dA = design(*BANDS[bt][c]["M"], f)
        rA = ols(dA, "y", flow_cols + ["up", "dn"] + flow_bh + ["up_bh", "dn_bh"], fe="month")
        rA2 = ols(dA, "y", flow_cols + ["up", "dn"], fe="month")          # column (2) analogue, for beta_c
        # B: state bands, month x state FE
        dB = design(*BANDS[bt][c]["S"], f)
        rB = ols(dB, "y", flow_cols + ["up", "dn"] + flow_bh + ["up_bh", "dn_bh"], fe="ms")
        RES[(bt, c, "A")] = rA; RES[(bt, c, "A2")] = rA2; RES[(bt, c, "B")] = rB


def pack(r):
    return {k: [float(r["b"][k]), float(r["se"][k])] for k in r["cols"]}


OUT["A_emp_main_col4"] = {}; OUT["B_emp_state_col1"] = {}; OUT["thr_col2"] = {}
for bt in BANDS:
    for c in CTRL:
        fb = (None, None) if c == "N" else ("f", "f_bh")
        for key, lab in [("A", "A_emp_main_col4"), ("B", "B_emp_state_col1")]:
            r = RES[(bt, c, key)]
            OUT[lab][f"{bt}_{c}"] = dict(coef=pack(r), n=r["n"],
                                         **state_summary(r, fb[0], "up", "dn", fb[1], "up_bh", "dn_bh"))
        OUT["thr_col2"][f"{bt}_{c}"] = pack(RES[(bt, c, "A2")])
    for c in ["D", "N"]:
        for key, lab in [("A", "A_emp_main_col4"), ("B", "B_emp_state_col1")]:
            OUT[lab][f"{bt}_{c}"]["vs_T_kp"] = cross_control(RES[(bt, "T", key)], RES[(bt, c, key)], "up", "up_bh")
            OUT[lab][f"{bt}_{c}"]["vs_T_km"] = cross_control(RES[(bt, "T", key)], RES[(bt, c, key)], "dn", "dn_bh")
print(f"A/B done {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ C: emp_flow price side and cross-equation test
OUT["C_emp_flow"] = {}
for bt in BANDS:
    for c, f in CTRL.items():
        l, u = BANDS[bt][c]["S"]
        beta_c = 0.0 if c == "N" else RES[(bt, c, "A2")]["b"]["f"]
        if bt == "fixed" and c == "T":       # published convention: beta read from emp_numbers.json
            beta_c = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
        d = a[["y", "x", "day", "month", "bh", "bh_eff", "bh_hol", "z_large"]].copy()
        fl = a[f].values
        m = (l + u) / 2
        x = d.x.values
        d["s"] = np.where(x > u, 1.0, np.where(x < l, -1.0, np.sign(x - m)))
        out_ = (x > u) | (x < l)
        d["d_out"] = np.maximum(x - u, 0) + np.maximum(l - x, 0)
        d["d_in"] = np.where(~out_, np.abs(x - m), 0.0)
        d["d_up"] = np.maximum(x - u, 0); d["d_dn"] = np.maximum(l - x, 0)
        for v in ["d_out", "d_in", "d_up", "d_dn"]:
            d[v + "_bh"] = d[v] * d.bh_eff
            d[v + "_hol"] = d[v] * d.bh_hol
        d["R"] = -d.s * (d.y - beta_c * fl)
        d["A_L"] = -d.s * d.z_large
        cols = ["d_out", "d_out_bh", "d_in", "d_in_bh"]
        pr = ols(d, "R", cols); fw = ols(d, "A_L", cols)
        b, V = joint([pr, fw]); k = len(cols)
        iC, iB = cols.index("d_out"), cols.index("d_out_bh")
        kC, kB, tC, tB = b[iC], b[iB], b[k + iC], b[k + iB]
        g = np.zeros(2 * k)
        g[iC] = 1 / (kC + kB) - 1 / kC; g[iB] = 1 / (kC + kB)
        g[k + iC] = -(1 / (tC + tB) - 1 / tC); g[k + iB] = -1 / (tC + tB)
        gp = np.zeros(2 * k); gp[iC] = 1 / (kC + kB) - 1 / kC; gp[iB] = 1 / (kC + kB)
        gf = np.zeros(2 * k); gf[k + iC] = 1 / (tC + tB) - 1 / tC; gf[k + iB] = 1 / (tC + tB)
        dl = np.log((kC + kB) / kC) - np.log((tC + tB) / tC)
        # C2: side split of the price side; C3: holiday interaction
        cs = ["d_up", "d_up_bh", "d_dn", "d_dn_bh", "d_in", "d_in_bh"]
        ps = ols(d, "R", cs)
        side = state_summary(ps, None, "d_up", "d_dn", None, "d_up_bh", "d_dn_bh")
        ch = ["d_out", "d_out_bh", "d_out_hol", "d_in", "d_in_bh", "d_in_hol"]
        ph = ols(d, "R", ch)
        OUT["C_emp_flow"][f"{bt}_{c}"] = dict(
            beta_used=float(beta_c), price=pack(pr), flow_large=pack(fw),
            kappa_closed=float(kC), kappa_open=float(kC + kB), ratio_price=float((kC + kB) / kC),
            se_log_ratio_price=dm(gp, V), theta_closed=float(tC), theta_open=float(tC + tB),
            ratio_flow=float((tC + tB) / tC), se_log_ratio_flow=dm(gf, V),
            logdiff=float(dl), se_logdiff=dm(g, V), z_logdiff=float(dl / dm(g, V)), n=pr["n"],
            side_split=dict(coef=pack(ps), **side), holiday=pack(ph))
        RES[(bt, c, "C")] = pr
    for c in ["D", "N"]:
        OUT["C_emp_flow"][f"{bt}_{c}"]["vs_T"] = cross_control(RES[(bt, "T", "C")], RES[(bt, c, "C")], "d_out", "d_out_bh")
print(f"C done {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ replication check against the published tables
pub_main = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["bh"]["coef"]
pub_state = json.load(open(os.path.join(TAB, "emp_state.json")))["state_bands"]["coef"]
pub_mech = json.load(open(os.path.join(TAB, "emp_mech.json")))["cross"]
rA = RES[("fixed", "T", "A")]["b"]; rB = RES[("fixed", "T", "B")]["b"]; rC = OUT["C_emp_flow"]["fixed_T"]
OUT["check_replication"] = dict(
    main_max_abs_diff=float(max(abs(rA[k2] - pub_main[k1]) for k1, k2 in
                                [("q", "f"), ("up", "up"), ("dn", "dn"), ("q_bh", "f_bh"), ("up_bh", "up_bh"), ("dn_bh", "dn_bh")])),
    state_max_abs_diff=float(max(abs(rB[k2] - pub_state[k1][0]) for k1, k2 in
                                 [("q", "f"), ("up_s", "up"), ("dn_s", "dn"), ("q_bh", "f_bh"), ("up_s_bh", "up_bh"), ("dn_s_bh", "dn_bh")])),
    flow_ratio_price=[rC["ratio_price"], pub_mech["ratio_price"]], flow_logdiff=[rC["logdiff"], pub_mech["logdiff"]],
    flow_se_logdiff=[rC["se_logdiff"], pub_mech["se_logdiff"]])

# ------------------------------------------------------------------ band diagnostics (see docstring)
def ssr_at(y, x, f, l, u):
    return _ssr_grid(y, x, f, np.array([u], float), np.array([-l], float))[0, 0]


def cells(kind):
    return a.groupby("bandper").indices if kind == "M" else a.groupby(["month", "state"]).indices


DIAG = {"cell_LR": {}}
for kind in ["M", "S"]:
    for c in ["D", "N"]:
        f = CTRL[c]; lrs = []
        for key, ix in cells(kind).items():
            d = a.iloc[ix]; fv = d[f].values.astype(float)
            lP, uP = BANDS["fixed"][c][kind][0][ix[0]], BANDS["fixed"][c][kind][1][ix[0]]
            lR, uR = BANDS["reprofiled"][c][kind][0][ix[0]], BANDS["reprofiled"][c][kind][1][ix[0]]
            lr = len(d) * np.log(ssr_at(d.y.values, d.x.values, fv, lP, uP) / ssr_at(d.y.values, d.x.values, fv, lR, uR))
            lrs.append((str(key), float(lr), [float(lP), float(uP)], [float(lR), float(uR)]))
        L = np.array([v[1] for v in lrs])
        top = sorted(lrs, key=lambda v: -v[1])[:3]
        DIAG["cell_LR"][f"{kind}_{c}"] = dict(n_cells=len(L), n_changed=int(sum(v[2] != v[3] for v in lrs)),
                                              median=float(np.median(L)), share_below_7_35=float(np.mean(L < 7.35)),
                                              largest=[dict(cell=v[0], LR=v[1], published=v[2], reprofiled=v[3]) for v in top])


def stat_A(l, u, f, fe="month", flow=True, sub=None):
    d = design(l, u, f) if sub is None else design(l, u, f)[sub]
    fc = ["f"] if flow else []; fb = ["f_bh"] if flow else []
    r = ols(d, "y", fc + ["up", "dn"] + fb + ["up_bh", "dn_bh"], fe=fe)
    return (r["b"]["up"] + r["b"]["up_bh"]) / r["b"]["up"]


def stat_C(l, u, f, beta_c, sub=None):
    d = a[["y", "x", "day", "month", "bh_eff", "z_large"]].copy(); fl = a[f].values
    m = (l + u) / 2; x = d.x.values
    d["s"] = np.where(x > u, 1.0, np.where(x < l, -1.0, np.sign(x - m))); out_ = (x > u) | (x < l)
    d["d_out"] = np.maximum(x - u, 0) + np.maximum(l - x, 0); d["d_in"] = np.where(~out_, np.abs(x - m), 0.0)
    for v in ["d_out", "d_in"]:
        d[v + "_bh"] = d[v] * d.bh_eff
    d["R"] = -d.s * (d.y - beta_c * fl); d["A_L"] = -d.s * d.z_large
    if sub is not None: d = d[sub]
    cols = ["d_out", "d_out_bh", "d_in", "d_in_bh"]
    pr = ols(d, "R", cols); fw = ols(d, "A_L", cols)
    rp = (pr["b"]["d_out"] + pr["b"]["d_out_bh"]) / pr["b"]["d_out"]
    rf = (fw["b"]["d_out"] + fw["b"]["d_out_bh"]) / fw["b"]["d_out"]
    return rp, rf, (np.log(rp) - np.log(rf)) if rp > 0 and rf > 0 else np.nan


# (ii) single most influential band cell under D, re-profiled bands
DIAG["single_cell"] = {}
betaD = RES[("reprofiled", "D", "A2")]["b"]["f"]
for spec, kind in [("A", "M"), ("B", "S"), ("C", "S")]:
    lR, uR = (v.copy() for v in BANDS["reprofiled"]["D"][kind]); lP, uP = BANDS["fixed"]["D"][kind]
    base = {"A": lambda l, u: stat_A(l, u, "z_dem"), "B": lambda l, u: stat_A(l, u, "z_dem", fe="ms"),
            "C": lambda l, u: stat_C(l, u, "z_dem", betaD)[1]}[spec]
    b0 = base(lR, uR); best = None
    for key, ix in cells(kind).items():
        if lR[ix[0]] == lP[ix[0]] and uR[ix[0]] == uP[ix[0]]: continue
        l, u = lR.copy(), uR.copy(); l[ix], u[ix] = lP[ix], uP[ix]
        v = base(l, u)
        if best is None or abs(np.log(v / b0)) > abs(np.log(best[1] / b0)):
            best = (str(key), float(v), [float(lP[ix[0]]), float(uP[ix[0]])], [float(lR[ix[0]]), float(uR[ix[0]])])
    DIAG["single_cell"][spec] = dict(stat="kappa+ open/closed ratio" if spec != "C" else "large-trade open/closed ratio",
                                     reprofiled_D=float(b0), cell=best[0], restored_value=best[1],
                                     published_band=best[2], reprofiled_band=best[3])
    if spec == "C":
        l, u = lR.copy(), uR.copy(); ix = cells("S")[tuple(best[0].strip("()").replace("'", "").split(", "))]
        l[ix], u[ix] = lP[ix], uP[ix]
        rp, rf, ld = stat_C(l, u, "z_dem", betaD)
        DIAG["single_cell"]["C"].update(restored_ratio_price=float(rp), restored_ratio_flow=float(rf), restored_logdiff=float(ld))
print(f"single-cell done {time.time() - t0:.0f}s", flush=True)

# (iii) leave-one-month-out ranges
DIAG["lomo"] = {}
months = sorted(a.month.unique())
for bt, c in [("fixed", "T"), ("fixed", "D"), ("reprofiled", "D"), ("reprofiled", "N")]:
    f = CTRL[c]; fl_ = c != "N"
    beta_c = 0.0 if c == "N" else RES[(bt, c, "A2")]["b"]["f"]
    vA, vB, vC = [], [], []
    for mo in months:
        sub = (a.month != mo).values
        vA.append(stat_A(*BANDS[bt][c]["M"], f, flow=fl_, sub=sub))
        vB.append(stat_A(*BANDS[bt][c]["S"], f, fe="ms", flow=fl_, sub=sub))
        vC.append(stat_C(*BANDS[bt][c]["S"], f, beta_c, sub=sub))
    vC = np.array(vC)
    DIAG["lomo"][f"{bt}_{c}"] = dict(
        A_kp_ratio=[float(np.min(vA)), float(np.median(vA)), float(np.max(vA)), months[int(np.argmin(vA))], months[int(np.argmax(vA))]],
        B_kp_ratio=[float(np.min(vB)), float(np.median(vB)), float(np.max(vB)), months[int(np.argmin(vB))], months[int(np.argmax(vB))]],
        C_ratio_price=[float(np.min(vC[:, 0])), float(np.median(vC[:, 0])), float(np.max(vC[:, 0]))],
        C_ratio_flow=[float(np.min(vC[:, 1])), float(np.median(vC[:, 1])), float(np.max(vC[:, 1])), months[int(np.argmin(vC[:, 1]))]],
        C_logdiff=[float(np.nanmin(vC[:, 2])), float(np.nanmedian(vC[:, 2])), float(np.nanmax(vC[:, 2]))])
    print(f"lomo {bt}_{c} done {time.time() - t0:.0f}s", flush=True)
OUT["diagnostics_posthoc"] = DIAG

OUT["runtime_s"] = float(time.time() - t0)
json.dump(OUT, open(os.path.join(TAB, f"rob_{NAME}.json"), "w"), indent=1, default=float)


# ------------------------------------------------------------------ LaTeX table (primary: re-profiled bands)
def star(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def cell(v, nd=4):
    return f"${v[0]:.{nd}f}{star(*v)}$", f"$({v[1]:.{nd}f})$"


LAB = {"T": "Total OFI (published)", "D": "Small and medium OFI", "N": "None"}
for bt in ["reprofiled", "fixed"]:
    with open(os.path.join(TAB, f"rob_{NAME}{'' if bt == 'reprofiled' else '_fixedbands'}.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lccc}\n\\toprule\nOrder-flow control & Total OFI & Small and medium OFI & None \\\\\n"
                 " & (published) & (demand) & (reduced form) \\\\\n\\midrule\n")
        for panel, lab in [("A_emp_main_col4", "A. Monthly bands (Table~\\ref{tab:emp_main}, column 4)"),
                           ("B_emp_state_col1", "B. Bands by month and banking state (Table~\\ref{tab:emp_state}, column 1)")]:
            fh.write(f"\\multicolumn{{4}}{{l}}{{\\emph{{{lab}}}}} \\\\\n")
            P = {c: OUT[panel][f"{bt}_{c}"] for c in CTRL}
            for rl, side, key in [("$\\kappa^+$, banks closed", "kp", "closed"), ("$\\quad\\times$ banks open", "kp", "incr"),
                                  ("$\\kappa^-$, banks closed", "km", "closed"), ("$\\quad\\times$ banks open", "km", "incr")]:
                cc = [cell(P[c][side][key]) for c in CTRL]
                fh.write(rl + " & " + " & ".join(x[0] for x in cc) + " \\\\\n & " + " & ".join(x[1] for x in cc) + " \\\\\n")
            fh.write("Open/closed ratio, $\\kappa^+$ & " + " & ".join(f"{P[c]['kp']['ratio']:.1f}" for c in CTRL) + " \\\\\n")
            fh.write("Open/closed ratio, $\\kappa^-$ & " + " & ".join(f"{P[c]['km']['ratio']:.1f}" for c in CTRL) + " \\\\\n")
            fh.write("$\\beta$ closed / open & " + " & ".join(
                f"{P[c]['beta']['closed'][0]:.3f} / {P[c]['beta']['open'][0]:.3f}" if "beta" in P[c] else "--" for c in CTRL) + " \\\\\n")
            fh.write("Long-run impact $\\beta/\\kappa^+$, closed / open (bp) & " + " & ".join(
                f"{P[c]['lr_kp']['closed'][0]:.0f} / {P[c]['lr_kp']['open'][0]:.0f}" if "lr_kp" in P[c] else "--" for c in CTRL) + " \\\\\n")
            fh.write("\\quad ratio closed/open & " + " & ".join(
                f"{P[c]['lr_kp']['ratio_closed_open']:.1f}" if "lr_kp" in P[c] else "--" for c in CTRL) + " \\\\\n")
            dd = [P[c].get("vs_T_kp", {}).get("d_incr") for c in CTRL]
            fh.write("Change in $\\kappa^+\\times$open vs.\\ total OFI & -- & " + " & ".join(
                f"${v[0]:.4f}$ $({v[1]:.4f})$" for v in dd[1:]) + " \\\\\n\\midrule\n")
        fh.write("\\multicolumn{4}{l}{\\emph{C. Price side and large trades (Table~\\ref{tab:emp_flow}, columns 1--2)}} \\\\\n")
        P = {c: OUT["C_emp_flow"][f"{bt}_{c}"] for c in CTRL}
        for rl, k_ in [("$d^{out}$, banks closed", "d_out"), ("$\\quad\\times$ banks open", "d_out_bh")]:
            cc = [cell(P[c]["price"][k_]) for c in CTRL]
            fh.write(rl + " & " + " & ".join(x[0] for x in cc) + " \\\\\n & " + " & ".join(x[1] for x in cc) + " \\\\\n")
        fh.write("Open/closed ratio, price side & " + " & ".join(f"{P[c]['ratio_price']:.1f}" for c in CTRL) + " \\\\\n")
        fh.write("Open/closed ratio, large trades & " + " & ".join(f"{P[c]['ratio_flow']:.1f}" for c in CTRL) + " \\\\\n")
        fh.write("Equal ratios: $\\log$ difference (s.e.) & " + " & ".join(
            f"{P[c]['logdiff']:.2f} ({P[c]['se_logdiff']:.2f})" for c in CTRL) + " \\\\\n")
        fh.write(f"$N$ & \\multicolumn{{3}}{{c}}{{{OUT['n']:,}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

print(json.dumps({k: OUT[k] for k in ["flow_diag", "check_reprofile_T", "check_replication", "band_widths"]}, indent=1))
print(f"total {time.time() - t0:.0f}s")
