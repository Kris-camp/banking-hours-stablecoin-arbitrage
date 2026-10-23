"""rob_carry_adjusted: the headline business-hours result on a carry-adjusted premium.

The interbank benchmark (Dukascopy USD/TRY) settles T+1 while USDT settles at once. A trade
date d whose T+1 value date lies n(d) calendar days ahead embeds n(d) days of lira-dollar carry (Fridays n = 3,
pre-holiday dates more), so the measured premium b = log(USDT/TRY) - log(USD/TRY T+1) is lower on multi-carry
dates. This script removes that carry from the premium and re-estimates the headline.

Timing fact (documented and asserted in rob_dayfe_t1.py, re-asserted here). The FX value date rolls at 17:00 New
York (21:00 or 22:00 UTC), inside the excluded 20:30-22:30 UTC window, so the embedded carry is constant within an
Istanbul date (= FX trade date) and Delta b is never computed across a roll. Hence the adjustment shifts the lagged
premium of a whole date by a constant and leaves Delta b unchanged:
    x_adj = x + g_d * (n(d) - 1),   y_adj = y,
which normalizes every date to the one carry day of an ordinary Monday-Thursday date. Sign: with i_TRY > i_USD the
T+1 rate exceeds the value-today rate by the carry, so the measured premium is LOWER on multi-carry dates and the
adjustment RAISES it (g_d > 0, n(d) - 1 >= 0). The script checks this against the diagnostic of rob_dayfe_t1.json:
the premium level falls with extra carry days and the premium jumps by +4.12 bp per carry day lost across a roll.

Carry per day g_d (bp per calendar day).
 (1) measured: the jump across value-date rolls per carry day, 4.12 bp (s.e. 0.75), read from
     tables/rob_dayfe_t1.json["diagnostic"]["overnight_jump_slope_per_carry_day"].
 (2) policy: (CBRT one-week repo rate - Fed funds target midpoint) / 365, date by date on the trade date.
     Decision-by-decision paths for 2024-2026; the January-August 2026 decisions are taken from tcmb.gov.tr and
     federalreserve.gov and listed with their sources in the JSON (key "rates").
 (3) stress: twice the policy differential.
 Supplementary (labelled so in the JSON):
 (2e) effective funding: as (2) but with the CBRT overnight lending rate while one-week repo auctions were suspended
      (20 Mar - 16 Apr 2025 at 46%; 2 Mar - 23 Aug 2026 at 40%, when the CBRT's average funding cost was its
      overnight lending rate), since tom-next carry follows the marginal funding rate.
 (1p) theta x (2), theta = slope of the roll jump on the change in embedded policy carry (measured share, date by date).
 (1-), (1+) measured carry -/+ 1.96 s.e.
 Sensitivity grids: constant g from 0 to 50 bp per day, and multiples 0.25-4 of the policy differential.

Estimation, for every g. Bands are re-profiled on x_adj exactly as in the paper (empirics_main.profile and
empirics_state.profile: 1 bp grid, 5% trimming, model._ssr_grid; >= 500 obs per month, >= 300 per month x state).
 A = Table emp_main column (4): Delta b on OFI, kappa+ term, kappa- term, all interacted with BH (09-17 Istanbul),
     monthly bands, month FE, standard errors clustered by UTC date with the paper's finite-sample correction.
 B = Table emp_state column (1): the same with bands by month x banking state and month x state FE.
 Also reported: Istanbul-date clusters, and x_adj with the published (unadjusted) bands.
 With g = 0 the code must reproduce the published bands and estimates (asserted): A kappa+ closed 0.0025,
 increment 0.0079 (s.e. 0.0014); B increment 0.0067 (s.e. 0.0023).
Reference (copied from tables/rob_dayfe_t1.json so this JSON is self-contained): the Tuesday-Thursday clean-day
row (increment 0.0079, s.e. 0.0029, Istanbul-date clusters) and the row that drops only multi-carry dates.

Pieces copied (not imported) from rob_dayfe_t1.py: data checks, settlement calendar and n(d), OLS with FE and
clusters, summarize, band profiling, premium-level and roll-jump diagnostics. Importing it would run its simulation.
Inputs: data_raw/a5.parquet, data_raw/state_bands.parquet, tables/rob_dayfe_t1.json, tables/emp_numbers.json,
        tables/emp_state.json.
Output: tables/rob_carry_adjusted.json.
"""
import os, sys, json, time, datetime
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import _ssr_grid

RAW, TAB = os.path.join(HERE, "..", "data_raw"), os.path.join(HERE, "..", "tables")
T0 = time.time()
OUT = dict(meta=dict(script="code/rob_carry_adjusted.py", run_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                     adjustment="x_adj = x + g_d * (n(d) - 1); y unchanged (carry constant within an Istanbul date)",
                     sign="measured premium is lower on multi-carry dates (T+1 rate embeds n(d) days of i_TRY - i_USD); the adjustment raises it",
                     g_units="bp per calendar day", ratio_se="s.e. of log(open/closed) by the delta method",
                     clusters="paper: UTC date (the published s.e.); also Istanbul date"))

# ------------------------------------------------------------------ data (copied from rob_dayfe_t1.py)
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
a["day"] = a.index.date                                   # UTC date: the paper's clusters
sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
assert sb.index.equals(a.index)
a["l_s"], a["u_s"] = sb.l_s.values, sb.u_s.values
ist = a.index.tz_convert("Europe/Istanbul")
a["iday"] = ist.date                                      # Istanbul date = FX trade date on this sample
a["idow"] = ist.dayofweek
assert a.idow.max() <= 4
umin = a.index.hour * 60 + a.index.minute
assert not ((umin >= 20 * 60 + 30) & (umin < 22 * 60 + 30)).any()   # value date rolls inside the excluded window
gap = a.index.to_series().diff() != pd.Timedelta("5min")
cont = ~gap.values
assert np.allclose(a.x.values[1:][cont[1:]], (a.x.values + a.y.values)[:-1][cont[1:]], atol=1e-6)
# check: a contiguous five-minute run never crosses an Istanbul date, so a date-constant shift leaves y unchanged
idv = np.asarray(a.iday.values)
assert (idv[1:][cont[1:]] == idv[:-1][cont[1:]]).all()

# ------------------------------------------------------------------ settlement calendar (copied from rob_dayfe_t1.py)
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
a["ncarry"] = a.iday.map(carry).astype(float)
a["extra"] = a.ncarry - 1
assert a.ncarry.notna().all() and (a.ncarry >= 1).all()
days = sorted(set(a.iday))
dd = pd.DataFrame({"d": days}); dd["n"] = dd.d.map(carry); dd["dow"] = [d.weekday() for d in days]
OUT["calendar"] = dict(n_days=len(days), carry_days_dist={str(k): int(v) for k, v in dd.n.value_counts().sort_index().items()},
                       n_days_multi_carry=int((dd.n > 1).sum()), n_multi_carry_not_friday=int(((dd.n > 1) & (dd.dow != 4)).sum()),
                       n_obs_multi_carry=int((a.ncarry > 1).sum()), share_obs_multi_carry=float((a.ncarry > 1).mean()),
                       mean_extra_carry_days_all_obs=float(a.extra.mean()))

# ------------------------------------------------------------------ policy-rate paths
# 2024-2025: CBRT one-week repo rate and Fed funds target midpoint, by decision date.
CB_2425 = [("2023-12-21", 42.5), ("2024-01-25", 45), ("2024-03-21", 50), ("2024-12-26", 47.5), ("2025-01-23", 45),
           ("2025-03-06", 42.5), ("2025-04-17", 46), ("2025-07-24", 43), ("2025-09-11", 40.5), ("2025-10-23", 39.5),
           ("2025-12-11", 38)]
FED_2425 = [("2023-07-27", 5.375), ("2024-09-19", 4.875), ("2024-11-08", 4.625), ("2024-12-19", 4.375),
            ("2025-09-18", 4.125), ("2025-10-30", 3.875), ("2025-12-11", 3.625)]
# January-August 2026: every scheduled decision, with its source and cross-checks.
TCMB = "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/MPC/MPC+Meeting+Decisions/"
FRB = "https://www.federalreserve.gov/newsevents/pressreleases/"
CB_2026 = [
    dict(date="2026-01-22", rate=37.0, previous=38.0, decision="cut 100 bp", overnight_lending=40.0, overnight_borrowing=35.5,
         source=TCMB + "ANO2026-01", release="Press Release on Interest Rates No. 2026-01",
         cross_check=["https://www.bloomberg.com/news/articles/2026-01-22/turkey-lowers-key-interest-rate-to-37-in-fifth-straight-cut",
                      "https://www.hurriyetdailynews.com/amp/central-bank-cuts-policy-rate-by-100-bps-to-37-percent-218120",
                      "https://think.ing.com/snaps/the-turkish-central-bank-delivered-a-cautious-100bp-cut/",
                      "https://www.intellinews.com/turkey-delivers-another-100bp-rate-cut-market-bet-on-150bp-421642/"]),
    dict(date="2026-03-12", rate=37.0, previous=37.0, decision="hold", overnight_lending=40.0, overnight_borrowing=35.5,
         source=TCMB + "ANO2026-12", release="Press Release on Interest Rates No. 2026-12",
         cross_check=["https://tradingeconomics.com/turkey/interest-rate/forecast",
                      "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Announcements/Press+Releases/2026/ANO2026-14"]),
    dict(date="2026-04-22", rate=37.0, previous=37.0, decision="hold", overnight_lending=40.0, overnight_borrowing=35.5,
         source=TCMB + "ANO2026-17", release="Press Release on Interest Rates No. 2026-17",
         cross_check=["https://www.focus-economics.com/countries/turkey/news/monetary-policy/turkey-central-bank-meeting-22-04-2026-central-bank-leaves-rates-unchanged-in-april/"]),
    dict(date="2026-06-11", rate=37.0, previous=37.0, decision="hold", overnight_lending=40.0, overnight_borrowing=35.5,
         source="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/MPC/MPC+Meeting+Decisions/ANO2026-23",
         release="Press Release on Interest Rates No. 2026-23",
         cross_check=["https://www.intellinews.com/turkey-announces-third-straight-hold-on-policy-rate-at-37-448123/",
                      "https://think.ing.com/downloads/pdf/snap/the-turkish-central-bank-remained-on-hold-keeping-an-eye-on-inflationary-risks"]),
    dict(date="2026-07-23", rate=37.0, previous=37.0, decision="hold", overnight_lending=40.0, overnight_borrowing=35.5,
         source="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Announcements/Press+Releases/2026/ANO2026-28",
         release="Press Release on Interest Rates No. 2026-28",
         cross_check=["https://www.focus-economics.com/countries/turkey/news/monetary-policy/turkey-central-bank-meeting-23-07-2026-central-bank-leaves-rates-unchanged-in-july/",
                      "https://www.rttnews.com/3669999/turkey-central-bank-leaves-key-rate-unchanged.aspx"]),
]
FED_2026 = [
    dict(date="2026-01-28", range="3.50-3.75", midpoint=3.625, decision="maintain", vote_note="Miran and Waller preferred a 25 bp cut",
         source=FRB + "monetary20260128a.htm", cross_check=["https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm"]),
    dict(date="2026-03-18", range="3.50-3.75", midpoint=3.625, decision="maintain", vote_note="Miran preferred a 25 bp cut",
         source=FRB + "monetary20260318a.htm",
         cross_check=["https://www.pcbb.com/industry-insights/economy/2026-03-18-fomc-mar-2026",
                      "https://www.agwestfc.com/education-and-resources/industry-and-economic-insights/industry-insights/monthly-market-report/economic-updates/2026/03/24/fomc-results-march-2026"]),
    dict(date="2026-04-29", range="3.50-3.75", midpoint=3.625, decision="maintain",
         vote_note="Miran preferred a 25 bp cut; Hammack, Kashkari and Logan supported maintaining the range but voted against the "
                   "statement's easing bias",
         source=FRB + "monetary20260429a.htm", cross_check=["https://www.federalreserve.gov/monetarypolicy/fomcminutes20260429.htm",
                                                            "https://www.atlantafed.org/news-and-events/2026/04/29/fomc-statement"]),
    dict(date="2026-06-17", range="3.50-3.75", midpoint=3.625, decision="maintain", vote_note="12-0",
         source=FRB + "monetary20260617a.htm", cross_check=["https://www.federalreserve.gov/monetarypolicy/fomcminutes20260617.htm"]),
    dict(date="2026-07-29", range="3.50-3.75", midpoint=3.625, decision="maintain",
         vote_note="9-3; Hammack, Kashkari and Logan preferred a 25 bp increase",
         source=FRB + "monetary20260729a.htm", cross_check=["https://www.federalreserve.gov/monetarypolicy/fomcminutes20260729.htm",
                                                            "https://www.federalreserve.gov/monetarypolicy/files/20260710_mprfullreport.pdf"]),
]
# Supplementary (2e): CBRT effective funding rate while one-week repo auctions were suspended.
CB_EFF_EPISODES = [
    dict(start="2025-03-20", end="2025-04-16", rate=46.0,
         note="interim MPC of 20 Mar 2025 raised the overnight lending rate to 46% and suspended one-week repo auctions; "
              "the TCMB 'Monetary Policy for 2026' text dates the suspension 20 March - 17 April 2025; from 17 Apr 2025 the "
              "policy rate is 46% (rates_band.py), so the path is unchanged from that date",
         source=["https://tcmb.gov.tr/wps/wcm/connect/en/tcmb+en/main+menu/announcements/press+releases/2025/ano2025-23",
                 "https://tcmb.gov.tr/wps/wcm/connect/c2ac62b6-3112-4f33-a6ad-3817defff0be/December28.pdf?MOD=AJPERES",
                 "https://www.centralbanking.com/central-banks/monetary-policy/7972564/turkey-raises-overnight-lending-rate-in-emergency-move"]),
    dict(start="2026-03-02", end="2026-08-23", rate=40.0,
         note="one-week repo auctions suspended (press release of 1 Mar 2026; 'suspended as of March 2' and 'average funding cost "
              "to stand at 40%, which is the CBRT's overnight lending rate' in the MPC summary of 18 Mar 2026); resumption "
              "decided 23 Aug 2026, first auction (at 37%) on 24 Aug 2026",
         source=["https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Announcements/Press+Releases/2026/ANO2026-11",
                 "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Announcements/Press+Releases/2026/ANO2026-14",
                 "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Announcements/Press+Releases/2026/ANO2026-35",
                 "https://www.intellinews.com/turkey-revives-one-week-repo-auctions-463063/"]),
]
CB = CB_2425 + [(e["date"], e["rate"]) for e in CB_2026]
FED = FED_2425 + [(e["date"], e["midpoint"]) for e in FED_2026]
DAYS = pd.date_range("2024-01-01", "2026-10-31", freq="D")


def path(ev):
    s = pd.Series({pd.Timestamp(d): v for d, v in ev}).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(DAYS)).ffill().reindex(DAYS)


cb_p, fed_p = path(CB), path(FED)
# the 2024-2025 part is built from the CB_2425 and FED_2425 decision lists
cb_eff = cb_p.copy()
for e in CB_EFF_EPISODES:
    cb_eff.loc[e["start"]:e["end"]] = e["rate"]
assert cb_p.notna().all() and fed_p.notna().all()
assert cb_p.loc["2026-01-21"] == 38 and cb_p.loc["2026-01-22"] == 37 and cb_p.loc["2026-08-31"] == 37
assert fed_p.loc["2026-01-01":"2026-08-31"].eq(3.625).all()
g_pol_day = (cb_p - fed_p) / 365 * 100                   # bp per calendar day
g_eff_day = (cb_eff - fed_p) / 365 * 100
key = pd.to_datetime(pd.Series(a.iday.values).astype(str)).values
a["g_pol"] = g_pol_day.reindex(key).values
a["g_eff"] = g_eff_day.reindex(key).values
assert a.g_pol.notna().all() and a.g_eff.notna().all()
gd = pd.DataFrame({"d": days})
gd["g_pol"] = g_pol_day.reindex(pd.to_datetime([str(d) for d in days])).values
gd["g_eff"] = g_eff_day.reindex(pd.to_datetime([str(d) for d in days])).values
gd["n"] = gd.d.map(carry)
mon = (cb_p - fed_p).resample("MS").mean().loc["2024-01":"2026-08"]
mon_eff = (cb_eff - fed_p).resample("MS").mean().loc["2024-01":"2026-08"]
OUT["rates"] = dict(
    definition="policy differential = CBRT one-week repo rate - Fed funds target midpoint (pp per year); carry per day = differential / 365 x 100 bp",
    cbrt_path_2024_25_from_rates_band=[list(e) for e in CB_2425], fed_path_2024_25_from_rates_band=[list(e) for e in FED_2425],
    cbrt_decisions_2026=CB_2026, fomc_decisions_2026=FED_2026,
    cbrt_meetings_2026_source="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Core+Functions/Monetary+Policy/Monetary+Policy+Committee/2026",
    cbrt_meetings_2026_note="scheduled decisions Jan 22, Mar 12, Apr 22, Jun 11, Jul 23 (next Sep 10); the second date of each pair on that page "
                            "is the summary release; no interim MPC meeting in Jan-Aug 2026",
    fomc_meetings_2026_source="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    fomc_meetings_2026_note="Jan 27-28, Mar 17-18, Apr 28-29, Jun 16-17, Jul 28-29 (next Sep 15-16); no unscheduled meeting found; the July 2026 "
                            "Monetary Policy Report states the range was kept at 3.50-3.75 since the beginning of the year",
    verification=("All ten 2026 decisions verified; none held for lack of verification. CBRT: each press release read directly on "
                  "www.tcmb.gov.tr on 2026-10-23 (rates, overnight lending and borrowing rates), cross-checked with the listed secondary "
                  "sources. FOMC: each statement "
                  "was verified from the federalreserve.gov text returned by web search restricted to that domain (statement, "
                  "implementation note, minutes), cross-checked with the listed sources. Fed changes take effect the day after the "
                  "decision as in rates_band.py; with no 2026 change this does not matter."),
    audit_note=("Independent audit, 2026-10-23: every 2026 rate re-verified. CBRT ANO2026-01, -12, -17, -23 and -28 read directly on "
                "www.tcmb.gov.tr (37% after 22 Jan, overnight lending/borrowing 40/35.5); ANO2026-11 (1 Mar 2026, one-week repo "
                "auctions suspended) and ANO2026-35 (23 Aug 2026, auctions resumed) also read directly. FOMC: all five 2026 "
                "statements to July confirmed at 3-1/2 to 3-3/4 percent via federalreserve.gov-restricted search; "
                "the April vote note was corrected to list all four dissents. The "
                "CBRT 'Monetary Policy for 2026' text dates the 2025 suspension 20 March - 17 April 2025. Rates, estimates and "
                "all other outputs unchanged by the audit."),
    effective_funding_episodes=CB_EFF_EPISODES,
    effective_funding_caveat=("supplementary path. The 2026 episode is documented by CBRT releases (ANO2026-35 and the 18 Mar 2026 MPC summary "
                              "were read directly; ANO2026-11 returned HTTP 503 and its text was confirmed via search). The 2025 episode is "
                              "from search-indexed TCMB text (direct fetches returned 503): if overnight funding at 49% continued after "
                              "17 Apr 2025, this path understates that carry by 3 pp (0.8 bp per day)."),
    monthly_mean_differential_pp=dict(policy={k.strftime("%Y-%m"): float(v) for k, v in mon.items()},
                                      effective={k.strftime("%Y-%m"): float(v) for k, v in mon_eff.items()}),
    carry_bp_per_day_on_sample_dates=dict(
        policy=dict(mean=float(gd.g_pol.mean()), min=float(gd.g_pol.min()), max=float(gd.g_pol.max()),
                    mean_multi_carry_dates=float(gd.g_pol[gd.n > 1].mean()),
                    mean_2024_25=float(gd.g_pol[[d.year <= 2025 for d in gd.d]].mean()),
                    mean_2026=float(gd.g_pol[[d.year == 2026 for d in gd.d]].mean())),
        effective=dict(mean=float(gd.g_eff.mean()), min=float(gd.g_eff.min()), max=float(gd.g_eff.max()),
                       mean_multi_carry_dates=float(gd.g_eff[gd.n > 1].mean()))),
    policy_2024_25_daily_mean_check=float(g_pol_day.loc["2024-01-01":"2025-12-31"].mean()))
# the 2024-25 mean equals the reference value of rob_dayfe_t1.json (11.30 bp per day)
DF = json.load(open(os.path.join(TAB, "rob_dayfe_t1.json")))
assert abs(OUT["rates"]["policy_2024_25_daily_mean_check"] - DF["diagnostic"]["policy_carry_bp_per_day_2024_25"]["mean"]) < 1e-9
print(f"rates built: policy carry on sample dates mean {gd.g_pol.mean():.2f} bp/day [{time.time() - T0:.0f}s]", flush=True)


# ------------------------------------------------------------------ estimator (copied from rob_dayfe_t1.py)
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
                ncl=int(ng), ssr=float(e @ e), r2=float(1 - (e @ e) / (Y @ Y)))


NAMES = ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"]


def design(x, q, l, u, bh):
    up = -np.maximum(x - u, 0); dn = np.maximum(l - x, 0)
    return np.column_stack([q, up, dn, q * bh, up * bh, dn * bh])


def codes(s):
    return pd.factorize(s)[0]


def fe_key(d, fe):
    if fe == "month": return codes(d.month.values)
    if fe == "ms": return codes(d.month.values + "_" + d.st.values)
    raise ValueError(fe)


def fit(d, fe, cl):
    X = design(d.x.values, d.q.values, d.l.values, d.u.values, d.bh.values)
    return ols(d.y.values, X, fe_key(d, fe), codes(d[cl].values), NAMES)


def summarize(r):
    b = r["b"]; se = r["se"]; V = r["V"]; ix = {k: i for i, k in enumerate(r["names"])}
    out = dict(n=r["n"], n_clusters=r["ncl"], within_r2=r["r2"], ssr=r["ssr"],
               coef={k: float(b[k]) for k in b}, se={k: float(se[k]) for k in se})
    for side, k in [("kp", "up"), ("km", "dn"), ("beta", "q")]:
        c, inc = b[k], b[k + "_bh"]
        o = c + inc
        rec = dict(closed=[float(c), float(se[k])], inc=[float(inc), float(se[k + "_bh"])], open=float(o))
        g = np.zeros(len(ix)); g[ix[k]] = 1; g[ix[k + "_bh"]] = 1
        rec["open_se"] = float(np.sqrt(g @ V @ g))
        if c > 0 and o > 0:
            g = np.zeros(len(ix)); g[ix[k]] = 1 / o - 1 / c; g[ix[k + "_bh"]] = 1 / o
            rec["ratio"] = float(o / c); rec["log_ratio"] = [float(np.log(o / c)), float(np.sqrt(g @ V @ g))]
        else:
            rec["ratio"] = None; rec["log_ratio"] = None
        out[side] = rec
    out["kp"]["inc_t"] = out["kp"]["inc"][0] / out["kp"]["inc"][1]
    return out


# ------------------------------------------------------------------ band profiling (copied from rob_dayfe_t1.py)
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


def setup(d, bands):
    d = d.copy(); d["bh"] = d.bh.astype(float); d["st"] = np.where(d.bh == 1, "open", "closed")
    B = None
    if bands == "pubM":
        pass                                         # a5 columns l, u (published monthly bands)
    elif bands == "pubS":
        d["l"], d["u"] = d.l_s, d.u_s
    elif bands == "reM":
        d, B = bands_monthly(d)
    elif bands == "reS":
        d, B = bands_state(d)
    return d, B


# ------------------------------------------------------------------ diagnostics (copied from rob_dayfe_t1.py)
def lvl(d, xcol, cols):
    """premium level on regressors, month FE, Istanbul-date clusters"""
    r = ols(d[xcol].values, d[cols].values.astype(float), codes(d.month.values), codes(d.iday.values), cols)
    return {k: [float(r["b"][k]), float(r["se"][k])] for k in cols}


def jump_slope(d, xcol):
    """premium jump across each value-date roll (last premium of date d to first lagged premium of the next sample
    date) on the change in carry days n(d) - n(next); HC s.e."""
    lastx = (d[xcol] + d.y).groupby(d.iday).last(); firstx = d[xcol].groupby(d.iday).first()
    ov = pd.DataFrame({"jump": firstx.shift(-1) - lastx, "n": [carry[x] for x in lastx.index],
                       "n_next": [carry[x] for x in firstx.index[1:]] + [np.nan]}, index=lastx.index).dropna()
    ov["dn_"] = ov.n - ov.n_next
    Xo = np.column_stack([np.ones(len(ov)), ov.dn_.values]); bo = np.linalg.lstsq(Xo, ov.jump.values, rcond=None)[0]
    eo = ov.jump.values - Xo @ bo; XtXi = np.linalg.inv(Xo.T @ Xo); Vo = XtXi @ (Xo.T * eo ** 2) @ Xo @ XtXi
    return [float(bo[1]), float(np.sqrt(Vo[1, 1])), int(len(ov))]


# ------------------------------------------------------------------ published numbers and the sign check
pubM = json.load(open(os.path.join(TAB, "emp_numbers.json")))
pubS = json.load(open(os.path.join(TAB, "emp_state.json")))
diag0 = DF["diagnostic"]
G_MEAS, G_MEAS_SE = diag0["overnight_jump_slope_per_carry_day"][0], diag0["overnight_jump_slope_per_carry_day"][1]
assert abs(G_MEAS - 4.12) < 0.005 and abs(G_MEAS_SE - 0.75) < 0.005
# sign: our copies of the diagnostics reproduce rob_dayfe_t1.json on the unadjusted premium ...
js0 = jump_slope(a, "x"); lv0 = lvl(a, "x", ["extra"])["extra"]
assert abs(js0[0] - G_MEAS) < 1e-9 and abs(js0[1] - G_MEAS_SE) < 1e-9
assert abs(lv0[0] - diag0["extra_carry_days"]["extra"][0]) < 1e-9 and abs(lv0[1] - diag0["extra_carry_days"]["extra"][1]) < 1e-9
# ... and they say the measured premium is LOWER on multi-carry dates: level falls with extra carry days and the
# premium rises across a roll that removes carry days. The adjustment g * (n - 1) with g > 0 therefore RAISES it.
assert lv0[0] < 0 and js0[0] > 0
OUT["sign_check"] = dict(level_on_extra_carry_days_unadjusted=lv0, roll_jump_per_carry_day_unadjusted=js0,
                         friday_level_unadjusted=diag0["friday"]["fri"],
                         reading="premium level lower by 2.4 bp per extra carry day and +4.1 bp jump per carry day lost at a roll: "
                                 "the measured premium is lower on multi-carry dates, so x_adj = x + g (n - 1) with g > 0 raises it")


# supplementary: the measured carry as a share theta of the policy carry, theta estimated from the roll jumps.
# Across the roll from date d to the next sample date the embedded carry changes by g_d n(d) - g_next n(next).
def jump_share(d, gcol):
    lastx = (d.x + d.y).groupby(d.iday).last(); firstx = d.x.groupby(d.iday).first(); gg = d[gcol].groupby(d.iday).first()
    idx = list(lastx.index)
    ov = pd.DataFrame({"jump": (firstx.shift(-1) - lastx).values,
                       "c": [gg[x] * carry[x] for x in idx], "c_next": [gg[x] * carry[x] for x in idx[1:]] + [np.nan]}, index=idx).dropna()
    Xo = np.column_stack([np.ones(len(ov)), (ov.c - ov.c_next).values]); bo = np.linalg.lstsq(Xo, ov.jump.values, rcond=None)[0]
    eo = ov.jump.values - Xo @ bo; XtXi = np.linalg.inv(Xo.T @ Xo); Vo = XtXi @ (Xo.T * eo ** 2) @ Xo @ XtXi
    return [float(bo[1]), float(np.sqrt(Vo[1, 1])), int(len(ov))]


THETA = jump_share(a, "g_pol")
OUT["sign_check"]["share_of_policy_carry_embedded"] = dict(
    theta_from_roll_jumps=THETA, measured_over_mean_policy=float(G_MEAS / gd.g_pol.mean()),
    note="theta: slope of the roll jump on the change in embedded policy carry g_d n(d) - g_next n(next) (HC s.e.)")


# ------------------------------------------------------------------ one run
def run(gvec, full=True):
    """gvec: carry per day (bp) for every observation (array) or a scalar."""
    gvec = np.broadcast_to(np.asarray(gvec, float), (len(a),))
    d = a.copy()
    shift = gvec * a.extra.values
    assert (shift >= 0).all()                                     # adjustment never lowers the premium
    d["x"] = a.x.values + shift
    # shift is constant within an Istanbul date, so y and the segment identity x_t = x_{t-1} + y_{t-1} are unchanged
    xv = d.x.values
    assert np.allclose(xv[1:][cont[1:]], (xv + d.y.values)[:-1][cont[1:]], atol=1e-6)
    A, BA = setup(d, "reM"); B, BB = setup(d, "reS")
    assert len(A) == len(a) and len(B) == len(a)
    rA = fit(A, "month", "day"); rB = fit(B, "ms", "day")
    res = dict(A=summarize(rA), B=summarize(rB))
    if not full:
        return res, BA, BB
    res["A_iday_cl"] = summarize(fit(A, "month", "iday")); res["B_iday_cl"] = summarize(fit(B, "ms", "iday"))
    Ap, _ = setup(d, "pubM"); Bp, _ = setup(d, "pubS")
    res["A_pubbands"] = summarize(fit(Ap, "month", "day")); res["B_pubbands"] = summarize(fit(Bp, "ms", "day"))
    # band changes relative to the published bands
    bl = {m: (v["l"], v["u"]) for m, v in pubM["bands"].items()}
    dl = np.array([BA[m][0] - bl[m][0] for m in BA]); du = np.array([BA[m][1] - bl[m][1] for m in BA])
    pS = a.assign(st=np.where(a.bh == 1, "open", "closed")).groupby(["month", "st"])[["l_s", "u_s"]].first()
    dlS = np.array([BB[k][0] - pS.loc[k, "l_s"] for k in BB]); duS = np.array([BB[k][1] - pS.loc[k, "u_s"] for k in BB])
    res["bands"] = dict(
        monthly={m: [float(BA[m][0]), float(BA[m][1])] for m in sorted(BA)},
        monthly_change_vs_published=dict(mean_dl=float(dl.mean()), mean_du=float(du.mean()), mean_dwidth=float((du - dl).mean()),
                                         n_months_changed=int(((dl != 0) | (du != 0)).sum()), n_months=len(BA)),
        state_change_vs_published=dict(mean_dl=float(dlS.mean()), mean_du=float(duS.mean()), mean_dwidth=float((duS - dlS).mean()),
                                       n_cells_changed=int(((dlS != 0) | (duS != 0)).sum()), n_cells=len(BB)),
        share_above_band_A=float((A.x > A.u).mean()), share_below_band_A=float((A.x < A.l).mean()))
    res["shift_bp"] = dict(mean_all=float(shift.mean()), mean_multi_carry=float(shift[a.extra.values > 0].mean()) if (a.extra > 0).any() else 0.0,
                           max=float(shift.max()), mean_friday=float(shift[a.idow.values == 4].mean()))
    res["diagnostic_after_adjustment"] = dict(level_on_extra_carry_days=lvl(d, "x", ["extra"])["extra"],
                                              friday_level=lvl(d.assign(fri=(d.idow == 4).astype(float)), "x", ["fri"])["fri"],
                                              roll_jump_per_carry_day=jump_slope(d, "x"))
    return res, BA, BB


def line(tag, r):
    s = ""
    for P in ["A", "B"]:
        kp = r[P]["kp"]
        s += (f" {P}: kp_c {kp['closed'][0]:.4f} ({kp['closed'][1]:.4f}) inc {kp['inc'][0]:.4f} ({kp['inc'][1]:.4f}) "
              f"ratio {kp['ratio'] if kp['ratio'] is None else round(kp['ratio'], 2)} |")
    return f"{tag:28s}{s} [{time.time() - T0:.0f}s]"


# ================================================================== g = 0: reproduce the published results
R0, BA0, BB0 = run(0.0)
for k in NAMES:
    assert abs(R0["A"]["coef"][k] - pubM["reg"]["bh"]["coef"][k]) < 1e-9 and abs(R0["A"]["se"][k] - pubM["reg"]["bh"]["se"][k]) < 1e-9
for k, kk in zip(NAMES, ["q", "up_s", "dn_s", "q_bh", "up_s_bh", "dn_s_bh"]):
    assert abs(R0["B"]["coef"][k] - pubS["state_bands"]["coef"][kk][0]) < 1e-9
    assert abs(R0["B"]["se"][k] - pubS["state_bands"]["coef"][kk][1]) < 1e-9
assert all(abs(BA0[m][0] - v["l"]) < 1e-9 and abs(BA0[m][1] - v["u"]) < 1e-9 for m, v in pubM["bands"].items()) and len(BA0) == len(pubM["bands"])
pS0 = a.assign(st=np.where(a.bh == 1, "open", "closed")).groupby(["month", "st"])[["l_s", "u_s"]].first()
assert all(abs(BB0[k][0] - pS0.loc[k, "l_s"]) < 1e-9 and abs(BB0[k][1] - pS0.loc[k, "u_s"]) < 1e-9 for k in BB0) and len(BB0) == len(pS0)
kpA, kpB = R0["A"]["kp"], R0["B"]["kp"]
assert round(kpA["closed"][0], 4) == 0.0025 and round(kpA["inc"][0], 4) == 0.0079 and round(kpA["inc"][1], 4) == 0.0014
assert round(kpB["inc"][0], 4) == 0.0067 and round(kpB["inc"][1], 4) == 0.0023
assert R0["A"]["n"] == 162624 and R0["B"]["n"] == 162624
OUT["baseline_reproduced"] = dict(ok=True, checks="g = 0: monthly and month x state bands equal the published bands; all six coefficients and "
                                                  "UTC-date-clustered s.e. of emp_main col (4) and emp_state col (1) equal the published values (|diff| < 1e-9)")
print(line("g=0 (published)", R0), flush=True)

# ================================================================== variants
VAR = {}
VAR["g0_published"] = dict(label="no adjustment (published)", g="0", res=R0)
spec = [("measured", "(1) measured carry: roll jump per carry day, 4.12 bp (s.e. 0.75)", G_MEAS, f"{G_MEAS:.4f} bp/day, constant"),
        ("policy", "(2) policy differential (CBRT one-week repo - Fed funds midpoint) / 365, by date", a.g_pol.values,
         "date by date; mean %.2f bp/day on sample dates" % gd.g_pol.mean()),
        ("policy_x2", "(3) stress: twice the policy differential", 2 * a.g_pol.values, "date by date; mean %.2f bp/day" % (2 * gd.g_pol.mean())),
        ("policy_effective_supp", "(2e, supplementary) effective CBRT funding rate during repo-auction suspensions - Fed funds midpoint, / 365",
         a.g_eff.values, "date by date; mean %.2f bp/day" % gd.g_eff.mean()),
        ("measured_share_policy_supp", "(1p, supplementary) measured share theta of the policy carry, date by date: theta x (2)",
         THETA[0] * a.g_pol.values, "theta = %.3f (s.e. %.3f); mean %.2f bp/day" % (THETA[0], THETA[1], THETA[0] * gd.g_pol.mean())),
        ("measured_lo_supp", "(1-, supplementary) measured carry minus 1.96 s.e.", G_MEAS - 1.96 * G_MEAS_SE, f"{G_MEAS - 1.96 * G_MEAS_SE:.4f} bp/day"),
        ("measured_hi_supp", "(1+, supplementary) measured carry plus 1.96 s.e.", G_MEAS + 1.96 * G_MEAS_SE, f"{G_MEAS + 1.96 * G_MEAS_SE:.4f} bp/day")]
for tag, lab, g, gdesc in spec:
    r, _, _ = run(g)
    VAR[tag] = dict(label=lab, g=gdesc, res=r)
    print(line(tag, r), flush=True)
OUT["variants"] = VAR

# ================================================================== sensitivity grids (bands re-profiled at every point)
GRID_C = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50]
GRID_M = [0.25, 0.5, 0.75, 1.25, 1.5, 2.5, 3, 4]


def compact(r):
    o = {}
    for P in ["A", "B"]:
        s = r[P]
        o[P] = dict(kp_closed=s["kp"]["closed"], kp_inc=s["kp"]["inc"], kp_inc_t=s["kp"]["inc_t"], kp_ratio=s["kp"]["ratio"],
                    kp_log_ratio=s["kp"]["log_ratio"], km_closed=s["km"]["closed"], km_inc=s["km"]["inc"],
                    beta_closed=s["beta"]["closed"], beta_inc=s["beta"]["inc"], ssr=s["ssr"], n=s["n"])
    return o


grid_c, grid_m = {}, {}
for g in GRID_C:
    if g == 0:
        grid_c["0"] = compact(R0); continue
    r, _, _ = run(float(g), full=False)
    grid_c[str(g)] = compact(r); print(line(f"grid const g={g}", r), flush=True)
for m in GRID_M:
    r, _, _ = run(m * a.g_pol.values, full=False)
    grid_m[str(m)] = compact(r); print(line(f"grid policy x{m}", r), flush=True)
grid_m["1"] = compact(VAR["policy"]["res"]); grid_m["2"] = compact(VAR["policy_x2"]["res"]); grid_m["0"] = compact(R0)
grid_m = {k: grid_m[k] for k in sorted(grid_m, key=float)}


def breakeven(grid):
    out = {}
    for P in ["A", "B"]:
        ks = sorted(grid, key=float)
        sig = [k for k in ks if grid[k][P]["kp_inc_t"] > 1.96]
        first_ns = next((k for k in ks if grid[k][P]["kp_inc_t"] <= 1.96), None)
        tmin = min(ks, key=lambda k: grid[k][P]["kp_inc_t"])
        out[P] = dict(grid=[float(k) for k in ks], all_significant_5pct=bool(len(sig) == len(ks)),
                      largest_with_inc_significant_5pct=float(sig[-1]) if sig else None,
                      smallest_with_inc_not_significant=float(first_ns) if first_ns is not None else None,
                      all_positive=bool(all(grid[k][P]["kp_inc"][0] > 0 for k in ks)),
                      min_t=float(grid[tmin][P]["kp_inc_t"]), at=float(tmin),
                      inc_range=[float(min(grid[k][P]["kp_inc"][0] for k in ks)), float(max(grid[k][P]["kp_inc"][0] for k in ks))],
                      closed_kp_negative_at=[float(k) for k in ks if grid[k][P]["kp_closed"][0] <= 0])
    return out


def within(grid, lo, hi):
    """restrict a grid to keys in [lo, hi]"""
    return {k: v for k, v in grid.items() if lo <= float(k) <= hi}


ssr_c = {P: min(grid_c, key=lambda k: grid_c[k][P]["ssr"]) for P in ["A", "B"]}
ssr_m = {P: min(grid_m, key=lambda k: grid_m[k][P]["ssr"]) for P in ["A", "B"]}
OUT["sensitivity"] = dict(
    note="bands re-profiled at every grid point; UTC-date clusters as in the paper",
    constant_g_bp_per_day=grid_c, policy_multiple=grid_m,
    constant_g_breakeven=breakeven(grid_c), policy_multiple_breakeven=breakeven(grid_m),
    constant_g_up_to_2x_policy=breakeven(within(grid_c, 0, 2 * gd.g_pol.mean())),
    policy_multiple_up_to_2=breakeven(within(grid_m, 0, 2)),
    ssr_relative_to_g0=dict(constant={k: {P: float(v[P]["ssr"] / grid_c["0"][P]["ssr"] - 1) for P in "AB"} for k, v in grid_c.items()},
                            policy_multiple={k: {P: float(v[P]["ssr"] / grid_m["0"][P]["ssr"] - 1) for P in "AB"} for k, v in grid_m.items()}),
    ssr_minimizing=dict(constant_g=dict(A=float(ssr_c["A"]), B=float(ssr_c["B"])), policy_multiple=dict(A=float(ssr_m["A"]), B=float(ssr_m["B"])),
                        note="descriptive: grid point with the smallest within SSR of the regression (same y, same number of "
                             "parameters, bands re-profiled); the criterion is flat (relative SSR changes below 0.3%) and this is "
                             "not used as an estimator of the carry"))

# ================================================================== reference rows from rob_dayfe_t1.json
REF = {}
for P in ["A", "B"]:
    for k, lab in [("excl_re", "Tuesday-Thursday clean dates (no holiday/half day on the date or a neighbour), bands re-estimated, Istanbul-date clusters"),
                   ("excl_pub", "Tuesday-Thursday clean dates, published bands, Istanbul-date clusters"),
                   ("n1_re", "dates with one carry day only (n(d) = 1), bands re-estimated, Istanbul-date clusters")]:
        s = DF["results"][P][k]
        REF[f"{P}_{k}"] = dict(label=lab, n=s["n"], n_clusters=s["n_clusters"], kp=s["kp"], km=s["km"], beta=s["beta"])
assert round(REF["A_excl_re"]["kp"]["inc"][0], 4) == 0.0079 and round(REF["A_excl_re"]["kp"]["inc"][1], 4) == 0.0029
REF["source"] = "tables/rob_dayfe_t1.json, results.{A,B}.{excl_re, excl_pub, n1_re} (A = emp_main col 4 monthly bands; B = emp_state col 1 month x state bands)"
REF["diagnostic"] = {k: diag0[k] for k in ["extra_carry_days", "friday", "overnight_jump_slope_per_carry_day", "policy_carry_bp_per_day_2024_25",
                                            "extra_carry_days_2024_25", "overnight_jump_by_carry_change"]}
REF["diagnostic"]["overnight_jump_by_carry_change_note"] = "key = n(d) - n(next date): [mean jump bp, s.e., number of rolls]"
REF["drift_by_daytype_state"] = DF["posthoc_drift_by_daytype_state"]
REF["drift_by_daytype_state_note"] = ("mean five-minute change in the premium (bp) [mean, date-clustered s.e., N] by date type (clean Tue-Thu, "
                                      "Monday, Friday, other) and banking state, from rob_dayfe_t1.json")
OUT["reference_rob_dayfe_t1"] = REF


# ================================================================== headline summary
def head(r, P):
    s = r[P]
    return dict(kp_closed=s["kp"]["closed"], kp_inc=s["kp"]["inc"], kp_open=[s["kp"]["open"], s["kp"]["open_se"]],
                kp_ratio=s["kp"]["ratio"], kp_log_ratio=s["kp"]["log_ratio"],
                kp_inc_relative_to_published=s["kp"]["inc"][0] / R0[P]["kp"]["inc"][0],
                km_closed=s["km"]["closed"], km_inc=s["km"]["inc"], km_ratio=s["km"]["ratio"],
                beta_closed=s["beta"]["closed"], beta_inc=s["beta"]["inc"], n=s["n"], n_clusters=s["n_clusters"])


OUT["summary"] = {tag: dict(label=v["label"], g=v["g"],
                            A_emp_main_col4=head(v["res"], "A"), B_emp_state_col1=head(v["res"], "B"),
                            A_inc_istanbul_cl=v["res"]["A_iday_cl"]["kp"]["inc"], B_inc_istanbul_cl=v["res"]["B_iday_cl"]["kp"]["inc"],
                            A_inc_published_bands=v["res"]["A_pubbands"]["kp"]["inc"], B_inc_published_bands=v["res"]["B_pubbands"]["kp"]["inc"],
                            roll_jump_per_carry_day_after=v["res"]["diagnostic_after_adjustment"]["roll_jump_per_carry_day"],
                            level_on_extra_carry_days_after=v["res"]["diagnostic_after_adjustment"]["level_on_extra_carry_days"])
                  for tag, v in VAR.items()}
OUT["summary"]["reference_tue_thu_clean_A"] = dict(kp_closed=REF["A_excl_re"]["kp"]["closed"], kp_inc=REF["A_excl_re"]["kp"]["inc"],
                                                   kp_ratio=REF["A_excl_re"]["kp"]["ratio"], n=REF["A_excl_re"]["n"],
                                                   note="from rob_dayfe_t1.json; Istanbul-date clusters")
json.dump(OUT, open(os.path.join(TAB, "rob_carry_adjusted.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ console
print("\nHEADLINE (UTC-date clusters, bands re-profiled on x_adj)")
for tag, v in VAR.items():
    for P in ["A", "B"]:
        s = v["res"][P]; kp, km, be = s["kp"], s["km"], s["beta"]
        print(f"{tag:22s} {P} kp_c {kp['closed'][0]:.4f} ({kp['closed'][1]:.4f}) inc {kp['inc'][0]:.4f} ({kp['inc'][1]:.4f}) "
              f"ratio {kp['ratio'] if kp['ratio'] is None else round(kp['ratio'], 2)} "
              f"logr_se {kp['log_ratio'][1] if kp['log_ratio'] else float('nan'):.2f} | km_c {km['closed'][0]:.4f} ({km['closed'][1]:.4f}) "
              f"inc {km['inc'][0]:.4f} ({km['inc'][1]:.4f}) | beta_c {be['closed'][0]:.3f} ({be['closed'][1]:.3f}) inc {be['inc'][0]:.3f} "
              f"({be['inc'][1]:.3f}) | N {s['n']}")
    da = v["res"]["diagnostic_after_adjustment"]
    print(f"{'':22s}   roll jump after {da['roll_jump_per_carry_day'][0]:.2f} ({da['roll_jump_per_carry_day'][1]:.2f}); "
          f"level on extra days after {da['level_on_extra_carry_days'][0]:.2f} ({da['level_on_extra_carry_days'][1]:.2f}); "
          f"inc Istanbul cl A {v['res']['A_iday_cl']['kp']['inc'][1]:.4f} B {v['res']['B_iday_cl']['kp']['inc'][1]:.4f}; "
          f"pub bands A {v['res']['A_pubbands']['kp']['inc'][0]:.4f} ({v['res']['A_pubbands']['kp']['inc'][1]:.4f})")
for k in ["constant_g_breakeven", "policy_multiple_breakeven", "constant_g_up_to_2x_policy", "policy_multiple_up_to_2", "ssr_minimizing"]:
    print(k, json.dumps(OUT["sensitivity"][k]))
print("theta", THETA, "measured/policy", OUT["sign_check"]["share_of_policy_carry_embedded"]["measured_over_mean_policy"])
print(f"done [{time.time() - T0:.0f}s]")
