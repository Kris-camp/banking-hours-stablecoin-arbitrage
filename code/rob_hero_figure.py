"""Hero figure with a folded distance axis: the flat in-band region made visible.

Panel (a) of figures/hero4.pdf (section 6 of empirics_mech.py) puts every
in-band observation at zero on the horizontal axis, so the figure cannot show the flat in-band
region that is the core threshold claim, nor the kink at the band edge. This script rebuilds the
same four panels into figures/hero4_v2.pdf on a signed distance-from-the-nearest-edge axis.
figures/hero4_v2.pdf is Figure 1 of the paper; figures/hero4.pdf is not used.

DESIGN
Sample, bands and transformations are those of section 6 of empirics_mech.py: a5.parquet joined
with the aggressive-trade size files and the month x banking-state bands (state_bands.parquet);
beta from emp_numbers.json; variables demeaned by month; bin means with the same day-clustered
standard error; bins with fewer than 150 observations dropped; "banks open" = bh_eff (business hours,
not a weekday public holiday), "banks closed" = bh == 0.

Horizontal axis. m = (l+u)/2. d = x - u if x > m, d = l - x if x <= m. d > 0 is the distance outside
the band; d <= 0 inside, where |d| is the distance to the nearest edge (0 at the edge, -(u-l)/2 at
the midpoint). Side s as in empirics_mech.py: +1 above u, -1 below l, sign(x - m) inside. Each
panel's dependent variable is the original one multiplied by -s, so positive values mean movement
toward the band (outside) or toward the band midpoint (inside):
 (a) R = -s * yy, yy = (dy - beta*OFI) demeaned by month (the original panel-(a) variable).
     Fitted line: threshold model on the folded axis, R = kappa * max(d,0) with month FE, kappa by
     OLS on all observations (day-clustered s.e.); drawn flat at 0 inside, slope kappa outside.
 (b) R by banking state. Fitted lines: the state-band model of Table emp_state, re-estimated here
     (month x state FE) to obtain the full covariance matrix; coefficients must reproduce
     emp_state.json. Aggregated to the folded axis as kappa_bar = sum_out kappa_{side,state} d_i /
     sum_out d_i over the state's outside observations with d <= 60 (the proportional reversion of
     Table emp_bins), delta-method s.e. Drawn flat at 0 inside, slope kappa_bar outside.
 (c) -s * z_large (banks open, closed) and -s * z_dem (all hours), month-demeaned: order flow toward
     the band in the same interval.
 (d) contributions of each leg to the reversion: -s * dpl (USDT/TRY) and +s * dpg (interbank),
     month-demeaned; they sum to -s * (dpl - dpg).
All fitted lines are drawn only between the outermost plotted bin means of their series (fixes the
overshoot of the original panel (b) lines to +-50 bp and the clipping of its error bars).
Bin edges in d: -60, -40, -25, -15, -8, -3, 0, 3, 8, 15, 25, 40, 60 (the edges of hero4.pdf, mirrored
inside the band). d = 0 counts as inside.

Summary statistics on the plotted variables (written to tables/rob_hero_figure.json):
 T1 in-band level: mean of R over in-band observations (d <= 0), day-clustered s.e.
 T2 kink regression: R on const, min(d,0), max(d,0) (pooling, no further FE), day-clustered s.e.:
    in-band slope gamma_in, outside slope kappa_out, kink = kappa_out - gamma_in. On |d| <= 60
    (the plotted range; main) and on all observations. Pooled and by banking state.
 T3 the same kink regression for the panel-(c) and panel-(d) variables (|d| <= 60).
Note: the bands are profiled to fit a kink at the edges, so T1-T3 describe the
fitted shape conditional on the estimated bands; they are not an independent test of linearity
(that is the bootstrap sup-LR test in the paper).
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
OUT = {}

# ------------------------------------------------------------------ data, exactly as empirics_mech.py
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
a["day"] = a.index.date
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
a = a.join(sb, how="inner")
a["l"], a["u"] = a.l_s, a.u_s
for c in ["small", "mid", "large"]:
    a[f"z_{c}"] = a[f"ofi_{c}"] / a[f"ofi_{c}"].groupby(a.month).transform("std")
a["z_dem"] = (a.ofi_small + a.ofi_mid) / (a.ofi_small + a.ofi_mid).groupby(a.month).transform("std")
mid = (a.l + a.u) / 2
a["s"] = np.where(a.x > a.u, 1.0, np.where(a.x < a.l, -1.0, np.sign(a.x - mid)))
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
a["hol"] = np.isin(a.day, pd.to_datetime(HOL).date).astype(float)
a["bh_eff"] = a.bh * (1 - a.hol)
beta = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]

# folded axis: signed distance from the nearest band edge, negative inside
d = np.where(a.x > mid, a.x - a.u, a.l - a.x).astype(float)
a["d"] = d
s = a.s.values
dm = lambda v: (pd.Series(np.asarray(v, float)) - pd.Series(np.asarray(v, float)).groupby(a.month.values).transform("mean")).values
yy = dm(a.y - beta * a.q)                       # original panel-(a) variable
a["R"] = -s * yy
a["F_L"] = -s * dm(a.z_large)                   # large-trade flow toward the band
a["F_D"] = -s * dm(a.z_dem)                     # small and medium flow toward the band
a["P_L"] = -s * dm(a.dpl)                       # local-leg contribution to reversion
a["P_G"] = s * dm(a.dpg)                        # interbank-leg contribution to reversion
assert np.allclose((a.P_L + a.P_G).values, -s * (dm(a.dpl) - dm(a.dpg)), equal_nan=True)


def ols(dd, ycol, cols, fe="month"):
    Y = dd[ycol].values.astype(float); X = dd[cols].values.astype(float)
    if fe:
        gm = dd.groupby(fe)
        Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    else:
        X = np.column_stack([np.ones(len(Y)), X]); cols = ["const"] + list(cols)
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(dd.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), V=V, cols=cols, n=n)


EDG = np.r_[-60, -40, -25, -15, -8, -3, 0, 3, 8, 15, 25, 40, 60]


def binned(v, mask):
    """Bin means of v by d, day-clustered s.e. (same formula as empirics_mech.py)."""
    v = np.asarray(v, float); rows = []
    for lo_, hi_ in zip(EDG[:-1], EDG[1:]):
        mm = mask & (d > lo_) & (d <= hi_) & np.isfinite(v)
        if mm.sum() < 150: continue
        mu = np.nanmean(v[mm])
        gs = pd.Series(v[mm] - mu).groupby(a.day.values[mm]).sum()
        rows.append((d[mm].mean(), mu, np.sqrt(np.sum(gs ** 2)) / mm.sum(), int(mm.sum())))
    return np.array(rows)


allm = np.ones(len(a), bool)
op = (a.bh_eff == 1).values; cl = (a.bh == 0).values
inrng = np.abs(d) <= 60

# ------------------------------------------------------------------ fitted lines
# (a) pooled threshold model on the folded axis
a["d_out"] = np.maximum(d, 0.0)
fa = ols(a, "R", ["d_out"])
kap_a, kap_a_se = fa["b"]["d_out"], fa["se"]["d_out"]

# (b) state-band model of Table emp_state, re-estimated for the full covariance matrix
a["up_s"] = -np.maximum(a.x - a.u, 0); a["dn_s"] = np.maximum(a.l - a.x, 0)
for c in ["q", "up_s", "dn_s"]:
    a[c + "_bh"] = a[c] * a.bh
a["ms"] = a.month + "_" + np.where(a.bh == 1, "open", "closed")
pc = ["q", "up_s", "dn_s", "q_bh", "up_s_bh", "dn_s_bh"]
rs = ols(a, "y", pc, fe="ms")
ref = json.load(open(os.path.join(TAB, "emp_state.json")))["state_bands"]["coef"]
OUT["state_model_matches_emp_state"] = bool(all(abs(rs["b"][c] - ref[c][0]) < 1e-9 and abs(rs["se"][c] - ref[c][1]) < 1e-9 for c in pc))
assert OUT["state_model_matches_emp_state"], "state-band re-estimate does not reproduce emp_state.json"
Vs = rs["V"]; ix = {c: i for i, c in enumerate(pc)}
kbar = {}
for st, mask in [("open", op), ("closed", cl)]:
    w_up = np.sum(np.maximum(a.x - a.u, 0).values[mask & inrng])      # total excess above, plotted range
    w_dn = np.sum(np.maximum(a.l - a.x, 0).values[mask & inrng])      # total excess below
    sh = w_up / (w_up + w_dn)
    g = np.zeros(len(pc))
    if st == "open":
        val = sh * (rs["b"]["up_s"] + rs["b"]["up_s_bh"]) + (1 - sh) * (rs["b"]["dn_s"] + rs["b"]["dn_s_bh"])
        g[ix["up_s"]] = g[ix["up_s_bh"]] = sh; g[ix["dn_s"]] = g[ix["dn_s_bh"]] = 1 - sh
    else:
        val = sh * rs["b"]["up_s"] + (1 - sh) * rs["b"]["dn_s"]
        g[ix["up_s"]] = sh; g[ix["dn_s"]] = 1 - sh
    kbar[st] = dict(kappa_bar=float(val), se=float(np.sqrt(g @ Vs @ g)), share_excess_above=float(sh))
# ratio open/closed of kappa_bar, delta-method s.e. of the log ratio
sh_o, sh_c = kbar["open"]["share_excess_above"], kbar["closed"]["share_excess_above"]
go = np.zeros(len(pc)); go[ix["up_s"]] = go[ix["up_s_bh"]] = sh_o; go[ix["dn_s"]] = go[ix["dn_s_bh"]] = 1 - sh_o
gc = np.zeros(len(pc)); gc[ix["up_s"]] = sh_c; gc[ix["dn_s"]] = 1 - sh_c
gl = go / kbar["open"]["kappa_bar"] - gc / kbar["closed"]["kappa_bar"]
OUT["fit_a"] = dict(kappa=float(kap_a), se=float(kap_a_se), n=fa["n"])
OUT["fit_b"] = dict(kbar, ratio_open_closed=float(kbar["open"]["kappa_bar"] / kbar["closed"]["kappa_bar"]),
                    se_log_ratio=float(np.sqrt(gl @ Vs @ gl)),
                    state_model={c: [float(rs["b"][c]), float(rs["se"][c])] for c in pc})


# ------------------------------------------------------------------ summary statistics
def mean_se(v, mask):
    v = np.asarray(v, float); mm = mask & np.isfinite(v)
    mu = v[mm].mean(); gs = pd.Series(v[mm] - mu).groupby(a.day.values[mm]).sum()
    ng, n = gs.size, mm.sum()
    return [float(mu), float(np.sqrt(np.sum(gs ** 2) * ng / (ng - 1)) / n), int(n)]


a["d_neg"] = np.minimum(d, 0.0)


def kink(ycol, mask):
    dd = a[mask].dropna(subset=[ycol])
    r = ols(dd, ycol, ["d_neg", "d_out"], fe=None)
    i, o = r["cols"].index("d_neg"), r["cols"].index("d_out")
    gv = np.zeros(len(r["cols"])); gv[o], gv[i] = 1, -1
    return dict(const=[float(r["b"]["const"]), float(r["se"]["const"])],
                gamma_in=[float(r["b"]["d_neg"]), float(r["se"]["d_neg"])],
                kappa_out=[float(r["b"]["d_out"]), float(r["se"]["d_out"])],
                kink=[float(r["b"]["d_out"] - r["b"]["d_neg"]), float(np.sqrt(gv @ r["V"] @ gv))], n=r["n"])


inside = d <= 0
OUT["T1_inband_mean_R"] = {k: mean_se(a.R, inside & inrng & m) for k, m in [("all", allm), ("open", op), ("closed", cl)]}
OUT["T1_inband_mean_R_allobs"] = mean_se(a.R, inside)
OUT["T2_kink_R"] = {k: kink("R", inrng & m) for k, m in [("all", allm), ("open", op), ("closed", cl)]}
OUT["T2_kink_R_allobs"] = kink("R", allm)
OUT["T3_kink_flow"] = {"large_open": kink("F_L", inrng & op), "large_closed": kink("F_L", inrng & cl),
                       "small_mid_all": kink("F_D", inrng)}
OUT["T3_kink_legs"] = {"usdttry": kink("P_L", inrng), "interbank": kink("P_G", inrng)}
OUT["share_inside_plotted"] = float(np.mean(inside & inrng) / np.mean(inrng))
OUT["share_inside_below_minus60"] = float(np.mean(d < -60))

# ------------------------------------------------------------------ figure
XL = (-62, 62)


def frame(ax, ylab=None, xlab=False):
    ax.axvspan(XL[0], 0, color=C3, alpha=0.10, lw=0)
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(*XL); ax.set_xticks(range(-60, 61, 20))
    if ylab: ax.set_ylabel(ylab)
    if xlab: ax.set_xlabel("distance from nearest band edge (bp)")


def eb(ax, B_, col, mk, lab=None, line=False, ms=3.2, lw=0.8, alpha=0.85, dx=0.0):
    ax.errorbar(B_[:, 0] + dx, B_[:, 1], yerr=1.96 * B_[:, 2], fmt=mk + ("-" if line else ""), ms=ms, color=col,
                ecolor=col, alpha=alpha, lw=lw, label=lab)


def thr_line(ax, B_, k_, col, lw=1.1):
    gx = np.linspace(B_[:, 0].min(), B_[:, 0].max(), 200)
    ax.plot(gx, k_ * np.maximum(gx, 0), color=col, lw=lw)


BIN = {}
fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.0))
# A
A = ax[0, 0]; frame(A, "expected 5-min reversion (bp)")
BIN["a"] = Ba = binned(a.R, allm)
thr_line(A, Ba, kap_a, C2)
eb(A, Ba, C1, "o", ms=3.5, lw=0.8, alpha=1.0)
A.set_title("(a) Arbitrage switches on outside the band", fontsize=9.5)
# B
B = ax[0, 1]; frame(B)
for mask, col, mk, lab, st, dx in [(op, C1, "o", "banks open", "open", -0.5), (cl, C2, "s", "banks closed", "closed", 0.5)]:
    Bb = binned(a.R, mask); BIN["b_" + st] = Bb
    thr_line(B, Bb, kbar[st]["kappa_bar"], col, lw=1.0)
    eb(B, Bb, col, mk, lab=lab, lw=0.6, dx=dx)
B.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
B.set_title("(b) Faster arbitrage when banks are open", fontsize=9.5)
# C
C = ax[1, 0]; frame(C, "order flow toward the band (s.d.)", xlab=True)
for v, mask, col, mk, lab, key, dx in [(a.F_L, op, C1, "o", "large trades, banks open", "c_large_open", -0.5),
                                       (a.F_L, cl, C2, "s", "large trades, banks closed", "c_large_closed", 0.5),
                                       (a.F_D, allm, C3, "^", "small and medium trades", "c_small_mid", 0.0)]:
    Bc = binned(v, mask); BIN[key] = Bc
    eb(C, Bc, col, mk, lab=lab, line=True, dx=dx)
C.legend(frameon=False, fontsize=7.5, loc="lower left")
C.set_title("(c) Large trades lean against the premium", fontsize=9.5)
# D
D = ax[1, 1]; frame(D, "contribution to 5-min reversion (bp)", xlab=True)
for v, col, mk, lab, key, dx in [(a.P_L, C1, "o", "USDT/TRY price", "d_usdttry", -0.5),
                                 (a.P_G, C2, "s", "interbank USD/TRY", "d_interbank", 0.5)]:
    Bd = binned(v, allm); BIN[key] = Bd
    eb(D, Bd, col, mk, lab=lab, line=True, lw=0.8, alpha=1.0, dx=dx)
D.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
D.set_title("(d) Stablecoin adjusts; interbank does not", fontsize=9.5)
# label the two regions in every panel, in a strip of headroom above the data
for axx in ax.flat:
    y0, y1 = axx.get_ylim()
    y1 = y1 + 0.08 * (y1 - y0); axx.set_ylim(y0, y1)
    yt = y1 - 0.015 * (y1 - y0)
    axx.text(-2.5, yt, r"$\leftarrow$ inside band", ha="right", va="top", fontsize=7, color="0.35")
    axx.text(2.5, yt, r"outside band $\rightarrow$", ha="left", va="top", fontsize=7, color="0.35")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "hero4_v2.pdf")); plt.close(fig)

OUT["bins"] = {k: [dict(d=float(r[0]), mean=float(r[1]), se=float(r[2]), n=int(r[3])) for r in v] for k, v in BIN.items()}
OUT["bin_edges"] = EDG.tolist()
json.dump(OUT, open(os.path.join(TAB, "rob_hero_figure.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k != "bins"}, indent=1, default=float))
