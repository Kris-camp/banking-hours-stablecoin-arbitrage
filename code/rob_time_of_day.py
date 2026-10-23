"""Robustness: is the after-17:00 speed of arbitrage a U.S.-session liquidity effect?

After 17:00 Istanbul the speed of arbitrage outside the band stays at about half its
business-hours level (cycle in emp_mech.json). This could reflect U.S./NYSE session liquidity
rather than Turkish banking rails. Istanbul is UTC+3 all year; New York is
UTC-4 (EDT) or UTC-5 (EST). NYSE 09:30-16:00 ET is therefore 16:30-23:00 Istanbul in EDT and
17:30-24:00 Istanbul in EST. The sample already excludes 23:30-01:30 Istanbul (FX rollover).

TESTS
Speed = coefficient on d_out in a regression of R = -s(dy - beta*OFI) on d_out and d_in, as in the
cycle of empirics_mech.py: state-specific bands, month (or month x state) fixed effects, standard
errors clustered by Istanbul calendar date (by spell date in test b). Flow side (theta, x100) uses
A_L = -s * z_large as dependent variable. Turkish dates (holidays, arife, weekdays) are Istanbul
dates; NYSE dates and session are New York local time (zoneinfo America/New_York).
"Ordinary" Istanbul day = Mon-Fri, not a Turkish weekday public holiday (HOL), not an arife day.

(a) U.S. session inside closed Turkish-bank hours (17:00-09:00).
 a1 DST: ordinary days, closed hours, NYSE regular days only (US holidays and early closes dropped).
    R on d_out x half-hour-bin, d_in x half-hour-bin, d_out x EDT, d_in x EDT, d_out x NYSEopen,
    d_in x NYSEopen; month FE. With clock bins and the regime (season) both absorbed, the NYSEopen
    slope is identified only by the bins whose NYSE status flips with U.S. DST: 17:00-17:30
    (open in EDT only) and 23:00-23:30 (open in EST only). Also reported: raw speeds by bin x regime.
    Robustness: add d_out x London-open (08:00-16:30 London; flips 18:30-19:30 Istanbul).
 a2 U.S. holidays: ordinary Turkish days, 17:30-23:00 Istanbul (NYSE open in both regimes on
    regular days). d_out x US-holiday with bin and regime controls. Main set: U.S.-only NYSE holidays
    (excludes 1 Jan, Good Friday, 25 Dec, which close Europe too); robustness: all NYSE holidays.
    Daytime placebo: same regression in 09:00-16:30 Istanbul (NYSE closed on all days anyway).
 a3 Pooled: all closed-hours intervals on ordinary days; NYSEopen varies by DST, U.S. holidays and
    early closes; bin and regime controls. Headline estimate of the NYSE effect on the closed speed.
 a4 Turkish-holiday evenings: 17:30-23:00 Istanbul on NYSE regular days. d_out x TR-holiday and
    d_out x arife (banks closed all day / since about 12:30) against ordinary evenings. NYSE is open
    in all three groups. U.S.-session story: no difference. Rails/depletion story: lower speed.
 a5 Inside banking hours, 13:00-17:00 Istanbul, ordinary days, NYSE regular days: d_out x bin,
    d_out x EDT, d_out x NYSEopen (flips 16:30-17:00) and d_out x NY-FX-session (08:00-17:00 ET; flips
    15:00-16:00). Also the raw comparison of 13:00-17:00 vs 09:00-13:00 speeds in EDT vs EST (DiD).
(b) Capacity depletion. Closed spell = 17:00 on ordinary day D to 09:00 on D+1 (Monday mornings and
    post-holiday mornings are excluded because weekend/holiday flow is unobserved). Within spell,
    C_t = sum over earlier observed intervals of -s*z_large (oriented large-trade arbitrage flow since
    close), standardized by its s.d. over the closed-spell sample. R on d_out x Istanbul hour,
    d_in x hour, d_out x C, d_in x C; month FE; clustered by spell date. Depletion predicts a negative
    d_out x C slope at a given clock hour. Open-hours analogue: C since 09:00 within 09:00-17:00 on
    ordinary days; replenishment predicts no decline. Difference tested in a stacked regression
    (month x state FE, cluster by spell date). Robustness: C from out-of-band intervals only; C from
    same-side flow only; C over the full 5-minute grid with the rollover gap filled (s carried forward).
    Also: speed by hour since close, pooled and separately in EDT and EST. Rails story: profile
    anchored at 17:00 in both regimes. U.S.-session story: EST profile shifted one hour later.
(c) Mornings. Ordinary days, 06:00-12:00 Istanbul, bins 06-09 (closed), 09-10 (first hour), 10-12.
    Speeds by bin for Mondays vs Tuesday-Friday; Monday minus other, and the 09:00 jump. NYSE and
    London are closed at 09:00 Istanbul in both regimes, so a jump at 09:00 is not a U.S. or London
    session effect. Also first hour (09-10) vs rest of the business day (10-17), all ordinary days.
(d) Arife half-days (2024-04-09, 2024-10-28, 2025-06-05, 2025-10-28, 2026-03-19, 2026-05-26; banks
    and EFT close around midday): speed 13:00-17:00 on arife days vs ordinary days, and the DiD
    against mornings 09:00-12:00. All six arife days fall in EDT, so the robustness restricts the
    comparison group to EDT days.

Predictions. The U.S.-session explanation predicts positive NYSE effects in a1-a3 of the order of the
evening speed (about 0.01), no effect in a4, a profile shifted with DST, no jump at 09:00, and no
arife effect. The banking explanation predicts NYSE effects near zero, lower holiday and arife
evenings, a 09:00 jump, lower speed on arife afternoons, and (if depletion) a negative slope in C
in closed hours only.
Diagnostics P1-P5 (stored separately in the JSON): evening levels by group, activity filter and volume
control for holiday evenings, in-band slope in the flip bins, C without the last hour, and randomization
inference for the six arife days.
Sensitivity checks A1-A6 (stored separately in the JSON): the 17:00-17:30 bin, month x bin fixed effects, the
mechanical link between lagged flow and R, the depletion test without the last hour split by evening and night,
the 09:00 jump under other intercepts, and a London-open check. The pre-open bin of the morning jump in the
table is 06:00-09:00, as in (c).
Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, data_raw/ofi_size/*.parquet, tables/emp_numbers.json.
Writes tables/rob_time_of_day.json, tables/rob_time_of_day.tex, figures/rob_time_of_day.pdf.
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

# ------------------------------------------------------------------ data (identical construction to empirics_mech.py)
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()].sort_index()
a = a.join(z, how="left")
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
a = a.join(sb, how="inner")
a["l"], a["u"] = a.l_s, a.u_s
sd_large = a.ofi_large.groupby(a.month).std()
a["z_large"] = a.ofi_large / a.month.map(sd_large)
m = (a.l + a.u) / 2
a["s"] = np.where(a.x > a.u, 1.0, np.where(a.x < a.l, -1.0, np.sign(a.x - m)))
a["out"] = ((a.x > a.u) | (a.x < a.l)).astype(float)
a["d_out"] = np.maximum(a.x - a.u, 0) + np.maximum(a.l - a.x, 0)
a["d_in"] = np.where(a.out == 0, np.abs(a.x - m), 0.0)
beta = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
a["R"] = -a.s * (a.y - beta * a.q)
a["A_L"] = -a.s * a.z_large

# ------------------------------------------------------------------ calendars
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]                                    # copied from empirics_mech.py
ARIFE = ["2024-04-09", "2024-10-28", "2025-06-05", "2025-10-28", "2026-03-19", "2026-05-26"]
NYSE_HOL = ["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04",
            "2024-09-02", "2024-11-28", "2024-12-25",
            "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
            "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
            "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03"]
GLOBAL_HOL = ["2024-01-01", "2024-03-29", "2024-12-25", "2025-01-01", "2025-04-18", "2025-12-25", "2026-01-01", "2026-04-03"]
NYSE_EARLY = ["2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24"]   # 13:00 ET close
D = lambda L: set(pd.to_datetime(L).date)
HOL, ARIFE, NYSE_HOL, GLOBAL_HOL, NYSE_EARLY = map(D, (HOL, ARIFE, NYSE_HOL, GLOBAL_HOL, NYSE_EARLY))
US_ONLY_HOL = NYSE_HOL - GLOBAL_HOL

ist = a.index.tz_convert("Europe/Istanbul")
ny = a.index.tz_convert("America/New_York")
ldn = a.index.tz_convert("Europe/London")
a["ist_min"] = ist.hour * 60 + ist.minute
a["ist_date"] = ist.date
a["ist_dow"] = ist.dayofweek
a["ist_hb"] = a.ist_min // 30                        # half-hour bin
a["ist_h"] = a.ist_min // 60
a["day"] = a.ist_date                                # cluster: Istanbul calendar date
ny_min = ny.hour * 60 + ny.minute
ny_date = pd.Series(ny.date, index=a.index)
a["edt"] = (ny.tz_localize(None) - a.index.tz_localize(None) == pd.Timedelta(hours=-4)).astype(float)
a["ny_dow"] = ny.dayofweek
a["ny_hol"] = ny_date.isin(NYSE_HOL).values.astype(float)
a["ny_hol_us"] = ny_date.isin(US_ONLY_HOL).values.astype(float)
a["ny_early"] = ny_date.isin(NYSE_EARLY).values.astype(float)
close_ny = np.where(a.ny_early == 1, 13 * 60, 16 * 60)
a["nyse"] = ((a.ny_dow < 5) & (a.ny_hol == 0) & (ny_min >= 570) & (ny_min < close_ny)).astype(float)
a["nyfx"] = ((a.ny_dow < 5) & (a.ny_hol == 0) & (ny_min >= 480) & (ny_min < 17 * 60)).astype(float)
ldn_min = ldn.hour * 60 + ldn.minute
a["ldn"] = ((ldn.dayofweek < 5) & (ldn_min >= 480) & (ldn_min < 990)).astype(float)
a["trhol"] = pd.Series(a.ist_date).isin(HOL).values.astype(float)
a["arife"] = pd.Series(a.ist_date).isin(ARIFE).values.astype(float)
a["ord"] = ((a.ist_dow < 5) & (a.trhol == 0) & (a.arife == 0)).astype(float)
a["closed"] = ((a.ist_min < 540) | (a.ist_min >= 1020)).astype(float)
a["state"] = np.where(a.closed == 1, "closed", "open")
a["ms"] = a.month + "_" + a.state
OUT["calendar"] = dict(n_arife_in_sample=int(a.loc[a.arife == 1, "ist_date"].nunique()),
                       arife_days=sorted(str(d) for d in a.loc[a.arife == 1, "ist_date"].unique()),
                       n_us_only_hol=int(a.loc[(a.ny_hol_us == 1) & (a.ord == 1) & (a.ist_min >= 540), "ist_date"].nunique()),
                       us_only_hol=sorted(str(d) for d in a.loc[(a.ny_hol_us == 1) & (a.ord == 1) & (a.ist_min >= 540), "ist_date"].unique()),
                       share_edt=float(a.edt.mean()))
assert set(a.loc[a.arife == 1, "edt"].unique()) == {1.0}


# ------------------------------------------------------------------ estimator (copied from empirics_mech.ols, cluster column added)
def ols(d, ycol, cols, fe="month", cl="day"):
    d = d.dropna(subset=[ycol] + list(cols))
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    if fe:
        gm = d.groupby(fe)
        Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    else:
        X = np.column_stack([np.ones(len(Y)), X]); cols = ["const"] + list(cols)
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d[cl])[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    adj = ng / (ng - 1) * (n - 1) / (n - k)
    V = XtXi @ (G.T @ G) @ XtXi * adj
    return dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), V=pd.DataFrame(V, cols, cols), n=n,
                n_out=int((d.d_out > 0).sum()), n_cl=int(ng))


def pr(r, c, scale=1.0):
    return [float(r["b"][c] * scale), float(r["se"][c] * scale)]


def lincomb(r, w, scale=1.0):
    w = pd.Series(w); V = r["V"].loc[w.index, w.index].values
    return [float(sum(r["b"][k] * v for k, v in w.items()) * scale), float(np.sqrt(w.values @ V @ w.values) * scale)]


def dummies(d, var, prefix, base="d_out"):
    cols = []
    for v in sorted(d[var].unique()):
        c = f"{prefix}{v}"; d[c] = d[base] * (d[var] == v); cols.append(c)
    return cols


def speed(d, fe="month", cl="day", y="R"):
    r = ols(d, y, ["d_out", "d_in"], fe=fe, cl=cl)
    return pr(r, "d_out", 100 if y == "A_L" else 1) + [r["n"], r["n_out"]]


def inter(d, cols_by):
    """add d_out x v and d_in x v columns for each v in cols_by; return names"""
    out = []
    for v in cols_by:
        d[f"o_{v}"] = d.d_out * d[v]; d[f"i_{v}"] = d.d_in * d[v]; out += [f"o_{v}", f"i_{v}"]
    return out


# ================================================================== (a) U.S. session inside closed hours
base = a[(a.ord == 1) & (a.closed == 1)].copy()
regular = (base.ny_hol == 0) & (base.ny_early == 0)
# a1 DST
d1 = base[regular].copy()
binc = dummies(d1, "ist_hb", "ob") + dummies(d1, "ist_hb", "ib", base="d_in")
x1 = binc + inter(d1, ["edt", "nyse"])
r1 = ols(d1, "R", x1)
r1f = ols(d1, "A_L", x1)
x1l = x1 + inter(d1, ["ldn"])
r1l = ols(d1, "R", x1l)
flip = {}
for nm, lo, hi in [("17:00-17:30", 1020, 1050), ("23:00-23:30", 1380, 1410), ("17:30-23:00", 1050, 1380),
                   ("01:30-09:00", 0, 540), ("16:30-17:00 (banks open)", 990, 1020)]:
    src = a[(a.ord == 1) & (a.ny_hol == 0) & (a.ny_early == 0)]
    for reg, val in [("EDT", 1.0), ("EST", 0.0)]:
        dd = src[(src.ist_min >= lo) & (src.ist_min < hi) & (src.edt == val)]
        flip[f"{nm} {reg}"] = dict(R=speed(dd), A_L=speed(dd, y="A_L"), nyse_share=float(dd.nyse.mean()))
OUT["a1_dst"] = dict(nyse_out=pr(r1, "o_nyse"), nyse_in_placebo=pr(r1, "i_nyse"), edt_out=pr(r1, "o_edt"),
                     nyse_out_flow=pr(r1f, "o_nyse", 100), nyse_out_london_ctrl=pr(r1l, "o_nyse"),
                     london_out=pr(r1l, "o_ldn"), n=r1["n"], n_out=r1["n_out"],
                     n_flip_out=int(((d1.d_out > 0) & (((d1.ist_hb == 34) | (d1.ist_hb == 46)))).sum()),
                     raw_by_bin_regime=flip)
# evening reference speed on ordinary regular days
ev_ref = d1[(d1.ist_min >= 1050) & (d1.ist_min < 1380)]
OUT["ref_evening_17h30_23h"] = dict(R=speed(ev_ref), A_L=speed(ev_ref, y="A_L"))
OUT["ref_night_01h30_09h"] = dict(R=speed(d1[d1.ist_min < 540]), A_L=speed(d1[d1.ist_min < 540], y="A_L"))
dd = a[(a.ord == 1) & (a.closed == 0) & (a.ist_min >= 600) & (a.ist_min < 960)]
OUT["ref_day_10h_16h"] = dict(R=speed(dd), A_L=speed(dd, y="A_L"))

# a2 US holidays, evening window where NYSE is open on regular days in both regimes
def ushol_test(lo, hi, holcol):
    d = a[(a.ord == 1) & (a.ist_min >= lo) & (a.ist_min < hi) & (a.ny_early == 0)].copy()
    if holcol == "ny_hol_us":
        d = d[(d.ny_hol == 0) | (d.ny_hol_us == 1)]
    d["ush"] = d[holcol]
    bc = dummies(d, "ist_hb", "ob") + dummies(d, "ist_hb", "ib", base="d_in")
    xc = bc + inter(d, ["edt", "ush"])
    fe = "ms" if d.state.nunique() > 1 else "month"
    r = ols(d, "R", xc, fe=fe); rf = ols(d, "A_L", xc, fe=fe)
    sp_h = speed(d[d.ush == 1], fe=None); sp_o = speed(d[d.ush == 0], fe=None)
    return dict(ush_out=pr(r, "o_ush"), ush_in_placebo=pr(r, "i_ush"), ush_out_flow=pr(rf, "o_ush", 100),
                speed_hol_nofe=sp_h, speed_ord_nofe=sp_o, n_hol_days=int(d.loc[d.ush == 1, "ist_date"].nunique()),
                n_hol_out=int(((d.ush == 1) & (d.d_out > 0)).sum()), n=r["n"])
OUT["a2_ushol"] = dict(evening_us_only=ushol_test(1050, 1380, "ny_hol_us"),
                       evening_all_nyse_hol=ushol_test(1050, 1380, "ny_hol"),
                       daytime_placebo_09_1630=ushol_test(540, 990, "ny_hol_us"))

# a3 pooled
d3 = base.copy()
x3 = dummies(d3, "ist_hb", "ob") + dummies(d3, "ist_hb", "ib", base="d_in") + inter(d3, ["edt", "nyse"])
r3 = ols(d3, "R", x3); r3f = ols(d3, "A_L", x3)
x3l = x3 + inter(d3, ["ldn"])
r3l = ols(d3, "R", x3l)
OUT["a3_pooled"] = dict(nyse_out=pr(r3, "o_nyse"), nyse_in_placebo=pr(r3, "i_nyse"), nyse_out_flow=pr(r3f, "o_nyse", 100),
                        nyse_out_london_ctrl=pr(r3l, "o_nyse"), n=r3["n"], n_out=r3["n_out"])

# a4 Turkish-holiday and arife evenings, NYSE open
d4 = a[(a.ist_dow < 5) & (a.ist_min >= 1050) & (a.ist_min < 1380) & (a.ny_hol == 0) & (a.ny_early == 0)].copy()
x4 = dummies(d4, "ist_hb", "ob") + dummies(d4, "ist_hb", "ib", base="d_in") + inter(d4, ["edt", "trhol", "arife"])
r4 = ols(d4, "R", x4); r4f = ols(d4, "A_L", x4)
OUT["a4_tr_evenings"] = dict(trhol_out=pr(r4, "o_trhol"), arife_out=pr(r4, "o_arife"), trhol_in_placebo=pr(r4, "i_trhol"),
                             trhol_out_flow=pr(r4f, "o_trhol", 100), arife_out_flow=pr(r4f, "o_arife", 100),
                             speed_trhol_evening=speed(d4[d4.trhol == 1], fe=None),
                             speed_arife_evening=speed(d4[d4.arife == 1], fe=None),
                             speed_ord_evening=speed(d4[d4.ord == 1], fe=None),
                             nyse_share_trhol=float(d4.loc[d4.trhol == 1, "nyse"].mean()),
                             n_trhol_days=int(d4.loc[d4.trhol == 1, "ist_date"].nunique()),
                             n_trhol_out=int(((d4.trhol == 1) & (d4.d_out > 0)).sum()),
                             n_arife_out=int(((d4.arife == 1) & (d4.d_out > 0)).sum()))

# a5 inside banking hours, 13:00-17:00
d5 = a[(a.ord == 1) & (a.ist_min >= 780) & (a.ist_min < 1020) & (a.ny_hol == 0) & (a.ny_early == 0)].copy()
x5 = dummies(d5, "ist_hb", "ob") + dummies(d5, "ist_hb", "ib", base="d_in") + inter(d5, ["edt", "nyse", "nyfx"])
r5 = ols(d5, "R", x5); r5f = ols(d5, "A_L", x5)
dd = a[(a.ord == 1) & (a.closed == 0) & (a.ny_hol == 0) & (a.ny_early == 0)].copy()
dd["pm"] = (dd.ist_min >= 780).astype(float)
dd["pm_edt"] = dd.pm * dd.edt
xd = inter(dd, ["pm", "edt", "pm_edt"])
rd = ols(dd, "R", ["d_out", "d_in"] + xd)
OUT["a5_bankhours"] = dict(nyse_out=pr(r5, "o_nyse"), nyfx_out=pr(r5, "o_nyfx"), nyse_out_flow=pr(r5f, "o_nyse", 100),
                           nyfx_out_flow=pr(r5f, "o_nyfx", 100), edt_out=pr(r5, "o_edt"), n=r5["n"],
                           did_pm_x_edt=pr(rd, "o_pm_edt"),
                           speeds={f"{w} {g}": speed(dd[(dd.pm == p) & (dd.edt == e)])
                                   for w, p in [("09-13", 0), ("13-17", 1)] for g, e in [("EDT", 1), ("EST", 0)]})

# ================================================================== (b) capacity depletion
ordset = set(a.loc[a.ord == 1, "ist_date"].unique())
prev = (pd.to_datetime(a.ist_date) - pd.Timedelta(days=1)).dt.date.values
a["spell_c"] = np.where(a.ist_min >= 1020, a.ist_date.astype(str), np.where(a.ist_min < 540, pd.Series(prev).astype(str).values, ""))
a["spell_c_ok"] = pd.Series(np.where(a.ist_min >= 1020, a.ist_date, prev)).isin(ordset).values & (a.closed == 1).values
a["spell_o_ok"] = (a.ord == 1) & (a.closed == 0)
a["spell"] = np.where(a.closed == 1, a.spell_c, a.ist_date.astype(str))
a["Afill"] = a.A_L.fillna(0)
a["Aout"] = (a.A_L * a.out).fillna(0)
a["Ap"] = (a.A_L * (a.s > 0) * a.out).fillna(0)     # flow leaning against premium above the band
a["Am"] = (a.A_L * (a.s < 0) * a.out).fillna(0)     # ... below the band

# gap-filled flow on the full 5-minute grid (s carried forward through the rollover window)
g = pd.DataFrame(index=z.index[(z.index >= a.index.min()) & (z.index <= a.index.max())])
g["s"] = a.s.reindex(g.index).ffill(limit=30)
g["mon"] = g.index.strftime("%Y-%m")
g["zl"] = z.ofi_large.reindex(g.index) / g.mon.map(sd_large)
gi = g.index.tz_convert("Europe/Istanbul"); gmin = gi.hour * 60 + gi.minute
gprev = (pd.to_datetime(gi.date) - pd.Timedelta(days=1)).date
g["spell"] = np.where(gmin >= 1020, pd.Series(gi.date).astype(str).values,
                      np.where(gmin < 540, pd.Series(gprev).astype(str).values, pd.Series(gi.date).astype(str).values))
g["A"] = (-g.s * g.zl).fillna(0)
g["Cg"] = g.groupby("spell").A.cumsum() - g.A
a["C_gap"] = g.Cg.reindex(a.index).values


def build_C(d):
    d = d.sort_index()
    for src, nm in [("Afill", "C"), ("Aout", "C_out"), ("Ap", "Cp"), ("Am", "Cm")]:
        d[nm] = d.groupby("spell")[src].cumsum() - d[src]
    d["C_same"] = np.where(d.s > 0, d.Cp, d.Cm)   # same-side flow (Ap and Am are both oriented against the premium)
    for nm in ["C", "C_out", "C_same", "C_gap"]:
        d[nm + "_z"] = d[nm] / d[nm].std()
    return d


dc = build_C(a[a.spell_c_ok].copy())
do = build_C(a[a.spell_o_ok].copy())
OUT["b_spells"] = dict(n_closed=int(dc.spell.nunique()), n_open=int(do.spell.nunique()),
                       sd_C_closed=float(dc.C.std()), sd_C_open=float(do.C.std()),
                       mean_C_end_closed=float(dc.groupby("spell").C.last().mean()))


def depl(d, Ccol):
    d = d.copy(); d["Cz"] = d[Ccol + "_z"]
    hc = dummies(d, "ist_h", "oh") + dummies(d, "ist_h", "ih", base="d_in")
    r = ols(d, "R", hc + inter(d, ["Cz"]), cl="spell")
    return r


dep = {}
for Ccol in ["C", "C_out", "C_same", "C_gap"]:
    rc, ro = depl(dc, Ccol), depl(do, Ccol)
    dep[Ccol] = dict(closed=pr(rc, "o_Cz"), closed_in_placebo=pr(rc, "i_Cz"), open=pr(ro, "o_Cz"),
                     open_in_placebo=pr(ro, "i_Cz"), n_closed=rc["n"], n_open=ro["n"])
# stacked: difference closed - open
st = pd.concat([dc.assign(Cz=dc.C_z, st_c=1.0), do.assign(Cz=do.C_z, st_c=0.0)])
st["Cz_c"] = st.Cz * st.st_c; st["Cz_o"] = st.Cz * (1 - st.st_c)
hc = dummies(st, "ist_h", "oh") + dummies(st, "ist_h", "ih", base="d_in")
rs = ols(st, "R", hc + inter(st, ["Cz_c", "Cz_o"]), fe="ms", cl="spell")
dep["stacked_diff_closed_minus_open"] = lincomb(rs, {"o_Cz_c": 1, "o_Cz_o": -1})
OUT["b_depletion"] = dep
# closed-hours speed relative to C: terciles of C within Istanbul hour
for d_ in (dc, do):
    d_["Ct"] = d_.groupby("ist_h").C.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
terc = {st_: [speed(d_[d_.Ct == k], cl="spell") for k in range(3)] for st_, d_ in [("closed", dc), ("open", do)]}
OUT["b_terciles_within_hour"] = terc
# speed by hour since close, pooled and by regime
hrs = {}
for reg, msk in [("all", dc.edt >= 0), ("EDT", dc.edt == 1), ("EST", dc.edt == 0)]:
    hrs[reg] = {int(h): speed(dc[msk & (dc.ist_h == h)], cl="spell") for h in sorted(dc.ist_h.unique())}
OUT["b_hour_since_close"] = hrs
# regime-specific late-evening contrast: 22:00-23:30 relative to 17:00-22:00, EDT vs EST
de = dc[(dc.ist_min >= 1020) & (dc.ist_min < 1410)].copy()
de["late"] = (de.ist_min >= 1320).astype(float); de["late_edt"] = de.late * de.edt
re_ = ols(de, "R", ["d_out", "d_in"] + inter(de, ["late", "edt", "late_edt"]), cl="spell")
OUT["b_late_evening_did"] = dict(late_x_edt=pr(re_, "o_late_edt"), late=pr(re_, "o_late"))

# ================================================================== (c) mornings
dm = a[(a.ord == 1) & (a.ist_min >= 360) & (a.ist_min < 720)].copy()
pday = pd.to_datetime(dm.ist_date)
prev_bd = (pday - pd.offsets.BDay(1)).dt.date
dm["mon"] = (dm.ist_dow == 0).astype(float)
dm["after_long"] = ((dm.ist_dow == 0) | pd.Series(prev_bd.values).isin(HOL | ARIFE).values).astype(float)
dm["bin"] = np.select([dm.ist_min < 540, dm.ist_min < 600], ["pre", "h1"], "later")
mor = {}
for grp in ["mon", "after_long"]:
    cols = []
    for bn in ["pre", "h1", "later"]:
        for gv, gl in [(1, "g"), (0, "o")]:
            c = f"o_{bn}_{gl}"; dm[c] = dm.d_out * ((dm.bin == bn) & (dm[grp] == gv)); cols.append(c)
            c = f"i_{bn}_{gl}"; dm[c] = dm.d_in * ((dm.bin == bn) & (dm[grp] == gv)); cols.append(c)
    r = ols(dm, "R", cols, fe="ms")
    mor[grp] = {f"{bn}_{gl}": pr(r, f"o_{bn}_{gl}") for bn in ["pre", "h1", "later"] for gl in ["g", "o"]}
    for bn in ["pre", "h1", "later"]:
        mor[grp][f"diff_{bn}"] = lincomb(r, {f"o_{bn}_g": 1, f"o_{bn}_o": -1})
    mor[grp]["jump_g"] = lincomb(r, {"o_h1_g": 1, "o_pre_g": -1})
    mor[grp]["jump_o"] = lincomb(r, {"o_h1_o": 1, "o_pre_o": -1})
    mor[grp]["jump_diff"] = lincomb(r, {"o_h1_g": 1, "o_pre_g": -1, "o_h1_o": -1, "o_pre_o": 1})
    mor[grp]["n_days_g"] = int(dm.loc[dm[grp] == 1, "ist_date"].nunique())
# backlog at the open: excursion size 08:00-09:00
bk = dm[(dm.ist_min >= 480) & (dm.ist_min < 540)]
mor["backlog_0800_0900"] = {k: dict(share_out=float(v.out.mean()), mean_d_out=float(v.d_out.mean()))
                            for k, v in [("monday", bk[bk.mon == 1]), ("other", bk[bk.mon == 0])]}
# first hour vs rest of business day
dbd = a[(a.ord == 1) & (a.closed == 0)].copy()
dbd["h1"] = (dbd.ist_min < 600).astype(float)
rb = ols(dbd, "R", ["d_out", "d_in"] + inter(dbd, ["h1"]))
mor["first_hour_minus_rest"] = pr(rb, "o_h1"); mor["rest_10_17"] = pr(rb, "d_out")
OUT["c_mornings"] = mor

# ================================================================== (d) arife afternoons
dA = a[((a.ord == 1) | (a.arife == 1)) & (a.closed == 0) & ((a.ist_min < 720) | (a.ist_min >= 780))].copy()
dA["pm"] = (dA.ist_min >= 780).astype(float)
dA["ar_pm"] = dA.arife * dA.pm
ar = {}
for nm, sub in [("all_days", dA), ("edt_days", dA[dA.edt == 1])]:
    sub = sub.copy()
    r = ols(sub, "R", ["d_out", "d_in"] + inter(sub, ["pm", "arife", "ar_pm"]))
    rf = ols(sub, "A_L", ["d_out", "d_in"] + inter(sub, ["pm", "arife", "ar_pm"]))
    ar[nm] = dict(arife_pm_minus_ord_pm=lincomb(r, {"o_arife": 1, "o_ar_pm": 1}), did=pr(r, "o_ar_pm"),
                  arife_am_minus_ord_am=pr(r, "o_arife"), ord_pm=lincomb(r, {"d_out": 1, "o_pm": 1}),
                  arife_pm=lincomb(r, {"d_out": 1, "o_pm": 1, "o_arife": 1, "o_ar_pm": 1}),
                  flow_arife_pm_minus_ord_pm=lincomb(rf, {"o_arife": 1, "o_ar_pm": 1}, 100),
                  n_arife_pm_out=int(((sub.arife == 1) & (sub.pm == 1) & (sub.d_out > 0)).sum()))
ar["speed_arife_pm_nofe"] = speed(dA[(dA.arife == 1) & (dA.pm == 1)], fe=None)
ar["speed_arife_am_nofe"] = speed(dA[(dA.arife == 1) & (dA.pm == 0)], fe=None)
OUT["d_arife"] = ar

# ================================================================== DIAGNOSTICS P1-P5
# They probe inference and mechanics and are stored separately in the output.
PH = {}
# P1 evening (17:30-23:00) levels by group with common month FE (descriptive companion to a2, a4)
de2 = a[(a.ist_dow < 5) & (a.ist_min >= 1050) & (a.ist_min < 1380) & (a.ny_early == 0)].copy()
G4 = {"g_ord": (de2.ord == 1) & (de2.ny_hol == 0), "g_ush": (de2.ord == 1) & (de2.ny_hol_us == 1),
      "g_trh": (de2.trhol == 1) & (de2.ny_hol == 0), "g_arf": (de2.arife == 1) & (de2.ny_hol == 0)}
for k, v in G4.items():
    de2[k] = v.astype(float)
de2 = de2[de2[list(G4)].sum(axis=1) == 1].copy()
gc = inter(de2, list(G4))
rl, rlf = ols(de2, "R", gc), ols(de2, "A_L", gc)
PH["P1_evening_levels"] = {k: dict(R=pr(rl, "o_" + k), A_L=pr(rlf, "o_" + k, 100), n_out=int(((de2[k] == 1) & (de2.d_out > 0)).sum()),
                                   n_days=int(de2.loc[de2[k] == 1, "ist_date"].nunique())) for k in G4}
# P2 Turkish-holiday evenings: interbank-activity filter (as in emp_quality col. 2) and a local-volume control
thr = a.loc[a.bh == 1, "fxv"].quantile(0.25)
d4f = d4[d4.fxv >= thr].copy()
r4a = ols(d4f, "R", x4)
d4["lv"] = np.log(d4.vol.clip(lower=1.0)); d4["lv"] -= d4.groupby("month").lv.transform("mean")
r4v = ols(d4, "R", x4 + inter(d4, ["lv"]))
PH["P2_tr_evenings_robust"] = dict(activity_filter=dict(trhol_out=pr(r4a, "o_trhol"), arife_out=pr(r4a, "o_arife"),
                                                        share_trhol_kept=float((d4f.trhol == 1).sum() / (d4.trhol == 1).sum()),
                                                        share_ord_kept=float((d4f.ord == 1).sum() / (d4.ord == 1).sum())),
                                   volume_ctrl=dict(trhol_out=pr(r4v, "o_trhol"), arife_out=pr(r4v, "o_arife"),
                                                    lv_out=pr(r4v, "o_lv"),
                                                    mean_lv_trhol=float(d4.loc[d4.trhol == 1, "lv"].mean())))
# P3 the in-band slope in the DST flip bins (the a3 in-band placebo is positive)
ib = {}
src = a[(a.ord == 1) & (a.ny_hol == 0) & (a.ny_early == 0)]
for nm, lo, hi in [("17:00-17:30", 1020, 1050), ("23:00-23:30", 1380, 1410)]:
    for reg, val in [("EDT", 1.0), ("EST", 0.0)]:
        r = ols(src[(src.ist_min >= lo) & (src.ist_min < hi) & (src.edt == val)], "R", ["d_out", "d_in"])
        ib[f"{nm} {reg}"] = dict(d_in=pr(r, "d_in"), d_out=pr(r, "d_out"))
PH["P3_inband_flip_bins"] = ib
# P4 depletion with the last hour of flow left out of C (flow persistence plus the large-trade price impact
# makes C_t mechanically correlated with R_t when C includes the most recent intervals)
for d_ in (dc, do):
    d_["C_l12"] = d_.groupby("spell").Afill.transform(lambda v: v.cumsum().shift(13).fillna(0))
    d_["C_l12_z"] = d_.C_l12 / d_.C_l12.std()
rc, ro = depl(dc, "C_l12"), depl(do, "C_l12")
PH["P4_depletion_C_excl_last_hour"] = dict(closed=pr(rc, "o_Cz"), open=pr(ro, "o_Cz"))
# P5 arife afternoons. Six days give six effective clusters, so day-clustered s.e. are unreliable.
# (i) randomization inference: the same estimator (month FE, group-specific d_out and d_in slopes) on
#     20,000 random sets of six ordinary EDT afternoons (13:00-17:00); (ii) the slope of each arife day;
# (iii) the arife-minus-ordinary afternoon difference with the month FE estimated on afternoons only and the
#     d_out slope identified from within-group variation (d_out demeaned by month within each group).
P = a[((a.ord == 1) | (a.arife == 1)) & (a.edt == 1) & (a.ist_min >= 780) & (a.ist_min < 1020)].dropna(subset=["R"]).copy()


def grp_est(Pd, inS):
    cm = pd.factorize(Pd.month)[0]; cnt = np.bincount(cm)
    X = np.column_stack([Pd.d_out * inS, Pd.d_out * ~inS, Pd.d_in * inS, Pd.d_in * ~inS])
    Xd = X - np.column_stack([np.bincount(cm, X[:, j]) / cnt for j in range(4)])[cm]
    Yd = Pd.R.values - (np.bincount(cm, Pd.R.values) / cnt)[cm]
    b = np.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ Yd)       # pinv: a draw can have no in-band interval
    return b[0] - b[1]


act = grp_est(P, P.arife.values == 1)
P0 = P[P.arife == 0]; dcodes = pd.factorize(P0.ist_date)[0]; nd = dcodes.max() + 1
rng = np.random.default_rng(7)
draws = np.array([grp_est(P0, np.isin(dcodes, rng.choice(nd, 6, replace=False))) for _ in range(20000)])
med = np.median(draws)
per = P.copy(); dcols = []
for d_ in sorted(ARIFE):
    c = "o_" + str(d_); per[c] = per.d_out * (per.ist_date == d_); dcols.append(c)
per["o_ord"] = per.d_out * per.ord; per["i_ar"] = per.d_in * per.arife; per["i_ord"] = per.d_in * per.ord
rp = ols(per, "R", dcols + ["o_ord", "i_ar", "i_ord"])
# within-group demeaning: month x group FE
P["mg"] = P.month + "_" + P.arife.astype(int).astype(str)
P["o_ar"] = P.d_out * P.arife; P["o_or"] = P.d_out * (1 - P.arife); P["i_ar"] = P.d_in * P.arife; P["i_or"] = P.d_in * (1 - P.arife)
rw = ols(P, "R", ["o_ar", "o_or", "i_ar", "i_or"], fe="mg")
PH["P5_arife"] = dict(estimate_edt_afternoons=float(act), placebo_median=float(med), placebo_p05=float(np.quantile(draws, .05)),
                      placebo_p95=float(np.quantile(draws, .95)),
                      p_two_sided=float(np.mean(np.abs(draws - med) >= abs(act - med))),
                      per_day={c[2:]: pr(rp, c) for c in dcols}, ordinary_pm=pr(rp, "o_ord"),
                      within_group_fe_diff=lincomb(rw, {"o_ar": 1, "o_or": -1}), within_group_fe_arife=pr(rw, "o_ar"),
                      share_out_arife_pm=float(P.loc[P.arife == 1, "out"].mean()), share_out_ord_pm=float(P.loc[P.arife == 0, "out"].mean()),
                      mean_d_out_arife_pm=float(P.loc[P.arife == 1, "d_out"].mean()), mean_d_out_ord_pm=float(P.loc[P.arife == 0, "d_out"].mean()))
OUT["post_hoc"] = PH

# ================================================================== SENSITIVITY CHECKS A1-A6
# Sensitivity of the results of the tests above; stored separately in the output.
AU = {}
# A1 the in-band placebo in a3 is not a single-cell artefact: drop the 17:00-17:30 bin (bank-close transition)
d3x = base[base.ist_hb != 34].copy()
xx = dummies(d3x, "ist_hb", "ob") + dummies(d3x, "ist_hb", "ib", base="d_in") + inter(d3x, ["edt", "nyse"])
rx = ols(d3x, "R", xx)
AU["A1_a3_excl_1700_bin"] = dict(nyse_out=pr(rx, "o_nyse"), nyse_in_placebo=pr(rx, "i_nyse"), n=rx["n"])
# A2 finer fixed effects: month x half-hour bin
for nm, dd_ in [("a1", d1.copy()), ("a3", d3.copy())]:
    dd_["mb"] = dd_.month + "_" + dd_.ist_hb.astype(str)
    rx = ols(dd_, "R", x1 if nm == "a1" else x3, fe="mb")
    AU[f"A2_{nm}_month_x_bin_fe"] = dict(nyse_out=pr(rx, "o_nyse"), nyse_in_placebo=pr(rx, "i_nyse"))
dx = d4.copy(); dx["mb"] = dx.month + "_" + dx.ist_hb.astype(str)
rx = ols(dx, "R", x4, fe="mb")
AU["A2_a4_month_x_bin_fe"] = dict(trhol_out=pr(rx, "o_trhol"), arife_out=pr(rx, "o_arife"))
dx = a[(a.ord == 1) & (a.ist_min >= 1050) & (a.ist_min < 1380) & (a.ny_early == 0)].copy()
dx = dx[(dx.ny_hol == 0) | (dx.ny_hol_us == 1)].copy(); dx["ush"] = dx.ny_hol_us; dx["mb"] = dx.month + "_" + dx.ist_hb.astype(str)
xx = dummies(dx, "ist_hb", "ob") + dummies(dx, "ist_hb", "ib", base="d_in") + inter(dx, ["edt", "ush"])
AU["A2_a2_month_x_bin_fe"] = dict(ush_out=pr(ols(dx, "R", xx, fe="mb"), "o_ush"))
# A3 mechanical link between C and R: b is a five-minute mean, so flow in t-1 shows up in y_t
mech = {}
for nm, d_ in [("closed", dc), ("open", do)]:
    dx = d_.copy()
    dx["A1"] = dx.groupby("spell").Afill.shift(1).fillna(0)
    dx["A2_12"] = dx.groupby("spell").Afill.transform(lambda v: v.shift(2).rolling(11, min_periods=1).sum()).fillna(0)
    rx = ols(dx, "R", ["d_out", "d_in", "A1", "A2_12"], cl="spell")
    mech[nm] = dict(R_on_A_lag1=pr(rx, "A1"), R_on_A_lag2_12=pr(rx, "A2_12"))
AU["A3_R_on_lagged_oriented_flow"] = mech
# A4 depletion with C excluding the last hour: stacked closed-minus-open, and evening vs night in closed hours
stx = pd.concat([dc.assign(Cz=dc.C_l12_z, st_c=1.0), do.assign(Cz=do.C_l12_z, st_c=0.0)])
stx["Cz_c"] = stx.Cz * stx.st_c; stx["Cz_o"] = stx.Cz * (1 - stx.st_c)
hx = dummies(stx, "ist_h", "oh") + dummies(stx, "ist_h", "ih", base="d_in")
rx = ols(stx, "R", hx + inter(stx, ["Cz_c", "Cz_o"]), fe="ms", cl="spell")
AU["A4_depletion_excl_last_hour"] = dict(stacked_diff_closed_minus_open=lincomb(rx, {"o_Cz_c": 1, "o_Cz_o": -1}),
                                         closed_evening_1700_2330=pr(depl(dc[dc.ist_min >= 1020], "C_l12"), "o_Cz"),
                                         closed_night_0130_0900=pr(depl(dc[dc.ist_min < 540], "C_l12"), "o_Cz"))
# A5 the 09:00 jump under alternative intercepts: month x bin FE, and 08:00-09:00 as the pre-open bin
mj = {}
for lab, lo, fe_ in [("month_x_bin_fe_pre0600", 360, "mb"), ("month_x_state_fe_pre0800", 480, "ms")]:
    dx = dm[dm.ist_min >= lo].copy(); dx["mb"] = dx.month + "_" + dx.bin
    cols = []
    for bn in ["pre", "h1", "later"]:
        for gv, gl in [(1, "g"), (0, "o")]:
            c = f"o_{bn}_{gl}"; dx[c] = dx.d_out * ((dx.bin == bn) & (dx.mon == gv)); cols.append(c)
            c = f"i_{bn}_{gl}"; dx[c] = dx.d_in * ((dx.bin == bn) & (dx.mon == gv)); cols.append(c)
    rx = ols(dx, "R", cols, fe=fe_)
    mj[lab] = dict(h1_other=pr(rx, "o_h1_o"), h1_monday=pr(rx, "o_h1_g"),
                   jump_other=lincomb(rx, {"o_h1_o": 1, "o_pre_o": -1}), jump_monday=lincomb(rx, {"o_h1_g": 1, "o_pre_g": -1}),
                   jump_diff=lincomb(rx, {"o_h1_g": 1, "o_pre_g": -1, "o_h1_o": -1, "o_pre_o": 1}))
AU["A5_morning_jump_sensitivity"] = mj
# A6 London open inside banking hours (08:00 London = 10:00 Istanbul in BST, 11:00 in GMT), 09:00-13:00, ordinary days
a["bst"] = ((ldn.tz_localize(None) - a.index.tz_localize(None)) == pd.Timedelta(hours=1)).astype(float)
dx = a[(a.ord == 1) & (a.ist_min >= 540) & (a.ist_min < 780)].copy()
xx = dummies(dx, "ist_hb", "ob") + dummies(dx, "ist_hb", "ib", base="d_in") + inter(dx, ["bst", "ldn"])
rx = ols(dx, "R", xx)
AU["A6_london_open_0900_1300"] = dict(ldn_out=pr(rx, "o_ldn"), ldn_in_placebo=pr(rx, "i_ldn"),
                                      speed_10h_bst=speed(dx[(dx.ist_h == 10) & (dx.bst == 1)])[:2],
                                      speed_10h_gmt=speed(dx[(dx.ist_h == 10) & (dx.bst == 0)])[:2],
                                      speed_9h_bst=speed(dx[(dx.ist_h == 9) & (dx.bst == 1)])[:2],
                                      speed_9h_gmt=speed(dx[(dx.ist_h == 9) & (dx.bst == 0)])[:2])
OUT["audit"] = AU

# ------------------------------------------------------------------ hourly profile by DST regime (figure)
prof = {}
srcp = a[(a.ord == 1) & (a.ny_hol == 0) & (a.ny_early == 0)]
for reg, val in [("EDT", 1.0), ("EST", 0.0)]:
    prof[reg] = {int(h): speed(srcp[(srcp.edt == val) & (srcp.ist_h == h)]) for h in sorted(srcp.ist_h.unique())}
OUT["hourly_profile_by_regime"] = prof

json.dump(OUT, open(os.path.join(TAB, "rob_time_of_day.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0]))
A = ax[0]
wins = ["17:00-17:30", "17:30-23:00", "23:00-23:30", "01:30-09:00"]
RB = OUT["a1_dst"]["raw_by_bin_regime"]
for reg, col, mk, off in [("EDT", C1, "o", -0.1), ("EST", C2, "s", 0.1)]:
    for i, w in enumerate(wins):
        v = RB[f"{w} {reg}"]; op_ = RB[f"{w} {reg}"]["nyse_share"] > 0.5
        A.errorbar([i + off], [v["R"][0]], yerr=[1.96 * v["R"][1]], fmt=mk, ms=5, color=col, ecolor=col, elinewidth=0.7,
                   mfc=col if op_ else "white", mew=1.1,
                   label=(("U.S. summer time (EDT)" if reg == "EDT" else "U.S. winter time (EST)") if i == 0 else None))
A.axhline(0, color="k", lw=0.4); A.set_xticks(range(4)); A.set_xticklabels(["17:00-\n17:30", "17:30-\n23:00", "23:00-\n23:30", "01:30-\n09:00"], fontsize=7.5)
A.set_ylabel(r"speed outside the band, $\hat\kappa$"); A.set_xlabel("Istanbul time (banks closed)", fontsize=8)
A.set_title("(a) NYSE open (filled) or shut (hollow)", fontsize=9)
from matplotlib.lines import Line2D
A.legend(handles=[Line2D([], [], marker="o", ls="", color=C1, ms=5, label="U.S. summer time (EDT)"),
                  Line2D([], [], marker="s", ls="", color=C2, ms=5, label="U.S. winter time (EST)")],
         frameon=False, fontsize=6.8, loc="upper right")
B = ax[1]
labs = [("g_ord", "ordinary"), ("g_ush", "U.S.\nholiday"), ("g_arf", "arife"), ("g_trh", "Turkish\nholiday")]
for i, (k, lb) in enumerate(labs):
    v = PH["P1_evening_levels"][k]["R"]
    B.errorbar([i], [v[0]], yerr=[1.96 * v[1]], fmt="o", ms=5, color=C2 if k in ("g_trh", "g_arf") else C1, ecolor=C3, elinewidth=0.8,
               mfc="white" if k == "g_ush" else (C2 if k in ("g_trh", "g_arf") else C1), mew=1.1)
B.axhline(0, color="k", lw=0.4); B.set_xticks(range(4)); B.set_xticklabels([l for _, l in labs], fontsize=7)
B.set_xlim(-0.5, 3.5); B.set_title("(b) Evenings, 17:30-23:00", fontsize=9)
Cx = ax[2]
for st_, col, mk, off, ls in [("closed", C2, "s", 0.06, "--"), ("open", C1, "o", -0.06, "-")]:
    v = np.array(terc[st_])
    Cx.errorbar(np.arange(3) + off, v[:, 0], yerr=1.96 * v[:, 1], fmt=mk + ls, ms=4, color=col, ecolor=col, elinewidth=0.7, lw=0.9,
                label="banks closed" if st_ == "closed" else "banks open")
Cx.axhline(0, color="k", lw=0.4); Cx.set_xticks(range(3)); Cx.set_xticklabels(["low", "mid", "high"], fontsize=8)
Cx.set_xlabel("arbitrage flow since close/open\n(tercile within clock hour)", fontsize=7.5)
Cx.set_title("(c) Cumulative arbitrage", fontsize=9); Cx.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rob_time_of_day.pdf")); plt.close(fig)

# ------------------------------------------------------------------ LaTeX table
def cell(v, sc=1.0, f="{:.4f}"):
    return f"${f.format(v[0] * sc)}{star(v[0], v[1])}$", f"$({f.format(v[1] * sc)})$"


def star(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


rows = [("A. U.S. session within closed Turkish-bank hours (17:00--09:00)", None, None),
        ("$d^{out}\\times$ NYSE open, identified by U.S. DST (a1)", OUT["a1_dst"]["nyse_out"], OUT["a1_dst"]["nyse_out_flow"]),
        ("$d^{out}\\times$ NYSE open, DST and U.S. holidays (a3)", OUT["a3_pooled"]["nyse_out"], OUT["a3_pooled"]["nyse_out_flow"]),
        ("$d^{out}\\times$ U.S.-only NYSE holiday, 17:30--23:00 (a2)", OUT["a2_ushol"]["evening_us_only"]["ush_out"],
         OUT["a2_ushol"]["evening_us_only"]["ush_out_flow"]),
        ("Reference: $d^{out}$, ordinary evenings 17:30--23:00", OUT["ref_evening_17h30_23h"]["R"][:2], OUT["ref_evening_17h30_23h"]["A_L"][:2]),
        ("B. Turkish banks shut while the NYSE is open (17:30--23:00)", None, None),
        ("$d^{out}\\times$ Turkish public holiday (a4)", OUT["a4_tr_evenings"]["trhol_out"], OUT["a4_tr_evenings"]["trhol_out_flow"]),
        ("$d^{out}\\times$ arife half-day (a4)", OUT["a4_tr_evenings"]["arife_out"], OUT["a4_tr_evenings"]["arife_out_flow"]),
        ("C. Turkish banks open, 13:00--17:00 (a5)", None, None),
        ("$d^{out}\\times$ NYSE open (16:30--17:00 in EDT)", OUT["a5_bankhours"]["nyse_out"], OUT["a5_bankhours"]["nyse_out_flow"]),
        ("$d^{out}\\times$ New York FX session (15:00--16:00 in EDT)", OUT["a5_bankhours"]["nyfx_out"], OUT["a5_bankhours"]["nyfx_out_flow"]),
        ("D. Cumulative arbitrage flow since the banks closed or opened (b)", None, None),
        ("$d^{out}\\times$ cumulative flow, banks closed (per s.d.)", OUT["b_depletion"]["C"]["closed"], None),
        ("$d^{out}\\times$ cumulative flow, banks open (per s.d.)", OUT["b_depletion"]["C"]["open"], None),
        ("Difference, closed minus open", OUT["b_depletion"]["stacked_diff_closed_minus_open"], None),
        ("E. Bank opening at 09:00, when New York and London are closed (c)", None, None),
        ("Jump in $\\hat\\kappa$, 06:00--09:00 to 09:00--10:00, Tuesday--Friday", OUT["c_mornings"]["mon"]["jump_o"], None),
        ("Jump in $\\hat\\kappa$, 06:00--09:00 to 09:00--10:00, Monday", OUT["c_mornings"]["mon"]["jump_g"], None),
        ("F. Arife half-days, banks shut from about noon (d)", None, None),
        ("$\\hat\\kappa$ 13:00--17:00, arife minus ordinary days", OUT["d_arife"]["all_days"]["arife_pm_minus_ord_pm"],
         OUT["d_arife"]["all_days"]["flow_arife_pm_minus_ord_pm"])]
with open(os.path.join(TAB, "rob_time_of_day.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcc}\n\\toprule\n & Price side, $\\hat\\kappa$ & Large-trade flow, $\\hat\\theta$ \\\\\n\\midrule\n")
    for lab, pv, fv in rows:
        if pv is None:
            fh.write(f"\\multicolumn{{3}}{{l}}{{\\emph{{{lab}}}}} \\\\\n"); continue
        p1, p2 = cell(pv); f1, f2 = cell(fv, 1, "{:.3f}") if fv is not None else ("", "")
        fh.write(f"\\quad {lab} & {p1} & {f1} \\\\\n & {p2} & {f2} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(json.dumps({k: OUT[k] for k in ["calendar", "post_hoc"]}, indent=1, default=float))

