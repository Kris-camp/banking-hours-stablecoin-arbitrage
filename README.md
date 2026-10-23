# Replication package: Banking Hours and the Speed of Stablecoin Arbitrage

**Jiayi Wang** (University of Technology Sydney) and **Lanxin Zhao** (The University of Sydney)

This repository contains the code that downloads the data and reproduces the tables and figures of the paper
"Banking Hours and the Speed of Stablecoin Arbitrage" (Tables 7 and 16 collect rows from the outputs of other
scripts; see "Tables 7 and 16" below), together with the generated tables, number files and figures for
comparison. The paper studies minute data on USDT against the Turkish lira from January 2024 to
August 2026. The USDT/TRY premium over the interbank dollar reverts toward a no-arbitrage band, and above the
band it reverts several times faster during Istanbul business hours than outside them. The code builds the
premium from Binance and Dukascopy data, estimates month-by-month threshold bands and the speed of reversion
inside and outside business hours, tests the arbitrage policy on order flow split by trade size, runs a
counterfactual with business-hours capacity around the clock, and repeats the analysis for 2021-2023 and for
Mexico, the euro area, South Africa and Poland. A Monte Carlo study checks the estimator.

## Contents

```
code/        scripts (Python); each finds its data in ../data_raw and writes to ../tables and ../figures
data_raw/    empty except README.md, which explains how the download scripts rebuild it
tables/      the LaTeX tables input by the paper, and the JSON and text files with the numbers behind them
figures/     the figures included in the paper (PDF)
tools/       check_identical.py (see "Code provenance" below)
```

## Data sources

| Source | Data | URL |
|---|---|---|
| Binance public data archive | spot 1-minute klines (USDT/TRY, BTC/TRY, BTC/USDT, USDT/MXN, EUR/USDT, USDT/ZAR, USDT/PLN) and aggregated trades (USDT/TRY, USDT/MXN) | <https://data.binance.vision> |
| Dukascopy historical data | 1-minute bid and ask candles of the SWFX marketplace: USD/TRY, USD/MXN, EUR/USD, USD/ZAR, USD/PLN | <https://www.dukascopy.com/swiss/english/marketwatch/historical/> (files from `https://datafeed.dukascopy.com/datafeed/`) |
| FXCM | 1-minute USD/TRY bid and ask candles (second dollar benchmark) | <https://candledata.fxcorporate.com> |
| Central Bank of the Republic of Turkey | indicative exchange-rate archive (used to verify the holiday calendar) | <https://www.tcmb.gov.tr/kurlar/> |
| BtcTurk | public API, USDT/TRY 1-minute klines through December 2020 | `https://graph-api.btcturk.com` |

The raw data are not included: they take about 2 GB, and the Dukascopy and FXCM terms may not permit
redistribution. `data_raw/README.md` lists every file and the command that downloads it. Institutional
facts (payment-system hours, FAST transfer limits, policy-rate decisions, holiday calendars) are entered in the
scripts with their sources; `tables/rob_xc_inst_try.json` holds the Turkish payment-rail facts read by
`code/rob_pre_sample.py`.

## Software

- Python 3.11 (the authors used 3.11.15 on Linux).
- Packages: `pip install -r requirements.txt` (numpy, pandas, pyarrow, scipy, matplotlib, numba, requests; the
  versions used for the paper are pinned). On Windows, also `pip install tzdata` for the time zones used by
  `zoneinfo`.
- Hardware used: 4 CPU cores, 16 GB of memory. The raw data and panels need about 2.5 GB of disk.
- LaTeX is needed only to compile the paper, not to produce the tables.

## How to run

Run every command from `code/`. Each script reads the files written by the earlier steps, so keep the order.

**1. Download the data** (several hours; Dukascopy rate-limits its feed)

```
python download_binance.py
python download_binance.py USDTMXN 2024-11 2026-08
python download_binance.py EURUSDT 2024-01 2026-08
python download_binance.py USDTZAR 2024-01 2026-08
python download_binance.py USDTPLN 2024-01 2026-07
python download_dukascopy.py
python download_dukascopy.py USDMXN 2024-11-01 2026-08-31
python download_dukascopy.py EURUSD 2024-01-01 2026-08-31
python download_dukascopy.py USDZAR 2024-01-01 2026-08-31
python download_dukascopy.py USDPLN 2024-01-01 2026-07-01
python download_fxcm.py
python aggtrades_ofi.py 2024-01 2026-08
python rob_mexico_utc_aggtrades.py 2024-11 2026-08
python rob_pre_sample_download.py all
```

Then fetch the Dukascopy ask candles and the CBRT calendar with the two short snippets in `data_raw/README.md`.

**2. Build the panel and the main estimates**

```
python build_panel.py TRY          # data_raw/panel_TRY.parquet
BOOT=99 python empirics_main.py    # data_raw/a5.parquet; Tables 1, 2, 5, 8, 9; Figures 3, 9
python empirics_state.py           # data_raw/state_bands.parquet; Table 3
python empirics_mech.py            # Tables 4, 6; Figure 4
python empirics_size.py            # Table 11
python empirics_extra.py           # Table 12
python rob_hero_figure.py          # Figure 1
python fig_ramadan_2024.py         # Figure 5
python fig_model.py                # Figure 2
python run_all.py                  # Figures 6-8; Tables 34-35
```

**3. Alternative specifications (Table 10)**

```
TAG=_f1 FREQ=1min BOOT=0 python empirics_main.py
TAG=_f15 FREQ=15min BOOT=0 python empirics_main.py
TAG=_close PRICE=close BOOT=0 python empirics_main.py
TAG=_excl EXCL=2025-03-19,2025-04-30 BOOT=0 python empirics_main.py
TAG=_q BAND=Q BOOT=0 python empirics_main.py
TAG=_trim10 TRIM=0.10 BOOT=0 python empirics_main.py
python robust_table.py
```

With `TAG` set, `empirics_main.py` writes only the tagged tables and `tables/emp_numbers{TAG}.json`, and leaves
`data_raw/a5.parquet` unchanged.

**4. Robustness (Appendix C)**

```
python rob_tcmb_institutions.py    # institutional checks; EFT-hours row of Table 7
python rob_demand_ofi.py           # Tables 13-14
python rob_fxcm_benchmark.py       # Table 15
python rob_dayfe_t1.py             # Table 16; rows of Table 7
python rob_carry_adjusted.py       # carry-adjusted rows of Table 7 (reads rob_dayfe_t1.json)
python rob_time_of_day.py          # Table 17
python rob_holiday_activity.py     # Table 18
python rob_info_shares.py          # Table 19
python rob_fast_limit.py           # Table 20; Figure 10
python rob_eft_edges.py            # Table 21
python rob_partner_banks.py        # Table 22
```

**5. The gap over time and by size of dislocation (Appendix D)**

```
python rob_by_period.py            # Table 23; Figure 11
python rob_size_dependence.py      # Table 24; Figure 12
python rob_flow_spline.py          # Table 25
python rob_ratio_by_year.py        # Table 26
python rob_cf_period.py            # Table 27
python rob_btcturk_venue.py        # 2020 comparison with BtcTurk (text of Appendix D)
python rob_pre_sample.py           # Table 28; Figure 13
python rob_jackknife_month.py      # Table 29
```

**6. Other local stablecoin markets (Appendix E)**

```
python empirics_mxn.py             # Table 31; Figure 14
python rob_mexico_utc.py           # Table 32
python rob_xc_eur.py
python rob_xc_eur_premica.py       # euro area before 31 March 2025 (text of Appendix E)
python rob_xc_zar.py
python rob_xc_pln.py
python rob_xc_summary.py           # Table 30
python rob_ratio_ci.py             # bootstrap and Fieller intervals for the open/closed ratios (text)
```

**7. Monte Carlo (Appendix F)**

```
python rob_montecarlo_calib.py     # Table 33 (environment variables R, RV, NPROC; defaults 500, 200, 4)
```

Several scripts accept further environment variables (number of bootstrap draws, seeds, number of worker
processes); each script's docstring lists them, with its inputs and outputs. The random seeds are fixed in the
code, so with the same data and package versions a rerun reproduces the tables in `tables/` and the estimates in
its JSON files. Run times recorded in some JSON files will differ, and so will the docstring text that
`rob_mexico_utc.py` and `rob_xc_pln.py` copy into their JSON (see "Code provenance").

## Tables and figures

Numbers refer to the full paper (main text and Appendices A-F). In the version with a separate Internet
Appendix, Appendices C-F are Sections IA.1-IA.4, Tables 7-35 are Tables IA.1-IA.29 and Figures 9-14 are
Figures IA.1-IA.6.

| Paper | IA | Title | Script | Output |
|---|---|---|---|---|
| Figure 1 | | Threshold arbitrage, banking hours, order flow and price discovery in USDT/TRY | `rob_hero_figure.py` | `figures/hero4_v2.pdf` |
| Figure 2 | | The model's predictions | `fig_model.py` | `figures/fig_model.pdf` |
| Table 1 | | Summary statistics: USDT/TRY, January 2024 - August 2026 | `empirics_main.py` | `tables/emp_summary.tex` |
| Figure 3 | | The USDT/TRY premium over the interbank rate | `empirics_main.py` | `figures/emp_premium.pdf` |
| Table 2 | | Threshold arbitrage in the USDT/TRY premium | `empirics_main.py` | `tables/emp_main.tex` |
| Table 3 | | Separating the friction from effective capacity | `empirics_state.py` | `tables/emp_state.tex` |
| Figure 4 | | The banking day and a holiday placebo | `empirics_mech.py` | `figures/emp_cycle.pdf` |
| Figure 5 | | The Ramadan holiday of April 2024 | `fig_ramadan_2024.py` | `figures/fig_ramadan_2024.pdf` |
| Table 4 | | The arbitrage policy in prices and in trades | `empirics_mech.py` | `tables/emp_flow.tex` |
| Table 5 | | Which price adjusts? | `empirics_main.py` | `tables/emp_legs.tex` |
| Table 6 | | Counterfactual: business-hours capacity around the clock | `empirics_mech.py` | `tables/emp_cf.tex` |
| Figure 6 | | Stationary distribution of the premium | `run_all.py` | `figures/fig2_density.pdf` |
| Figure 7 | | Capacity depletion | `run_all.py` | `figures/fig3_capacity.pdf` |
| Figure 8 | | Transmission of local information to the global price | `run_all.py` | `figures/fig4_discovery.pdf` |
| Table 7 | IA.1 | The business-hours increase in the speed of arbitrage | assembled from the JSON outputs (see below) | `tables/rob_bh_summary.tex` |
| Table 8 | IA.2 | Reversion by size of dislocation | `empirics_main.py` | `tables/emp_bins.tex` |
| Table 9 | IA.3 | Estimates by year | `empirics_main.py` | `tables/emp_year.tex` |
| Table 10 | IA.4 | Alternative specifications | `empirics_main.py` (six `TAG` runs), `robust_table.py` | `tables/emp_robust.tex` |
| Table 11 | IA.5 | Order flow by trade size and the excess premium | `empirics_size.py` | `tables/emp_size.tex` |
| Table 12 | IA.6 | Banking hours and the interbank benchmark | `empirics_extra.py` | `tables/emp_quality.tex` |
| Table 13 | IA.7 | Order-flow controls | `rob_demand_ofi.py` | `tables/rob_demand_ofi_fixedbands.tex` |
| Table 14 | IA.8 | Order-flow controls with bands re-estimated | `rob_demand_ofi.py` | `tables/rob_demand_ofi.tex` |
| Table 15 | IA.9 | A second dollar benchmark: FXCM | `rob_fxcm_benchmark.py` | `tables/rob_fxcm_benchmark.tex` |
| Table 16 | IA.10 | Value-date carry, date fixed effects and the definition of banking hours | `rob_dayfe_t1.py` (rows of its `rob_dayfe_t1.tex`, see below) | `tables/rob_dayfe_main.tex` |
| Table 17 | IA.11 | Time of day: the New York session, holidays and bank opening | `rob_time_of_day.py` | `tables/rob_time_of_day.tex` |
| Table 18 | IA.12 | Bank holidays and trading activity | `rob_holiday_activity.py` | `tables/rob_holiday_activity.tex` |
| Figure 9 | IA.1 | The 19 March 2025 lira shock | `empirics_main.py` | `figures/emp_event.pdf` |
| Table 19 | IA.13 | Information shares | `rob_info_shares.py` | `tables/rob_info_shares.tex` |
| Table 20 | IA.14 | The April 2024 increase in the FAST per-transfer limit | `rob_fast_limit.py` | `tables/rob_fast_limit.tex` |
| Figure 10 | IA.2 | The April 2024 increase in the FAST limit | `rob_fast_limit.py` | `figures/rob_fast_limit.pdf` |
| Table 21 | IA.15 | The edges of EFT | `rob_eft_edges.py` | `tables/rob_eft_edges.tex` |
| Table 22 | IA.16 | Round-the-clock bank channels | `rob_partner_banks.py` | `tables/rob_partner_banks.tex` |
| Table 23 | IA.17 | The banking-hours gap by year | `rob_by_period.py` | `tables/rob_by_period.tex` |
| Table 24 | IA.18 | The banking-hours gap by size of dislocation | `rob_size_dependence.py` | `tables/rob_size_dependence.tex` |
| Table 25 | IA.19 | Large-trade flow by size of dislocation | `rob_flow_spline.py` | `tables/rob_flow_spline.tex` |
| Figure 11 | IA.3 | The banking-hours gap over the sample | `rob_by_period.py` | `figures/rob_by_period.pdf` |
| Figure 12 | IA.4 | Speed by size of dislocation and around the November 2025 restriction | `rob_size_dependence.py` | `figures/rob_size_dependence.pdf` |
| Table 26 | IA.20 | Prices and large-trade flow by period | `rob_ratio_by_year.py` | `tables/rob_ratio_by_year.tex` |
| Table 27 | IA.21 | The counterfactual by period | `rob_cf_period.py` | `tables/rob_cf_period.tex` |
| Table 28 | IA.22 | The banking-hours gap from 2021 to 2026 | `rob_pre_sample.py` | `tables/rob_pre_sample.tex` |
| Figure 13 | IA.5 | The banking-hours gap from 2021 to 2026 | `rob_pre_sample.py` | `figures/rob_pre_sample.pdf` |
| Table 29 | IA.23 | Dropping one month at a time | `rob_jackknife_month.py` | `tables/rob_jackknife_month.tex` |
| Table 30 | IA.24 | Banking hours and the speed of arbitrage in five markets | `rob_xc_summary.py` (from `rob_xc_eur.py`, `rob_xc_zar.py`, `rob_xc_pln.py`, `rob_mexico_utc.py`, `empirics_mxn.py`) | `tables/rob_xc_summary.tex` |
| Table 31 | IA.25 | Turkey and Mexico | `empirics_mxn.py` | `tables/emp_mxn.tex` |
| Table 32 | IA.26 | Turkey and Mexico in common UTC windows | `rob_mexico_utc.py` | `tables/rob_mexico_utc.tex` |
| Figure 14 | IA.6 | The daily cycle of arbitrage in Turkey and Mexico | `empirics_mxn.py` | `figures/emp_mxn_cycle.pdf` |
| Table 33 | IA.27 | Calibrated Monte Carlo: the estimator in the empirical regime | `rob_montecarlo_calib.py` | `tables/rob_montecarlo_calib.tex` |
| Table 34 | IA.28 | Monte Carlo: recovery of structural parameters | `run_all.py` | `tables/tab1_montecarlo.tex` |
| Table 35 | IA.29 | Reversion speed by size of dislocation (baseline design) | `run_all.py` | `tables/tab2_bins.tex` |

### Tables 7 and 16

Table 7 (`tables/rob_bh_summary.tex`) collects estimates from several scripts. Each row reports the closed-hours
speed above the band, its business-hours increment (both with standard errors) and their open/closed ratio:

| Row | Source (file: entry) |
|---|---|
| Baseline | `emp_numbers.json`: `reg/bh`, coefficients `up` and `up_bh` |
| Small and medium trades only | `rob_demand_ofi.json`: `A_emp_main_col4/fixed_D/kp` |
| Small and medium trades only, bands re-estimated | `rob_demand_ofi.json`: `A_emp_main_col4/reprofiled_D/kp` |
| No order-flow control | `rob_demand_ofi.json`: `A_emp_main_col4/fixed_N/kp` |
| FXCM USD/TRY mid-quote | `rob_fxcm_benchmark.json`: `fxcm/bh`, entries `up` and `up_bh` |
| Date fixed effects, bias-corrected | `rob_dayfe_t1.json`: `results/A/dayfe_bc/kp` |
| Excluding dates with multi-day value-date carry | `rob_dayfe_t1.json`: `results/A/n1_re/kp` |
| Tuesday-Thursday dates away from value-date shifts | `rob_dayfe_t1.json`: `results/A/excl_re/kp` |
| Benchmark adjusted for value-date carry: measured, policy-rate differential, twice the differential | `rob_carry_adjusted.json`: `variants/{measured, policy, policy_x2}/res/A/kp` |
| EFT operating hours | `rob_tcmb_institutions.json`: the EFT-hours entry of `audit_ratio_check` (also `rob_by_period.json`: `by_period/full/d/kp`) |
| Excluding 09:00-10:00 and 16:00-17:00 | `rob_dayfe_t1.json`: `results/A/donut1_re/kp` |
| 2024, 2025, 2026 | `rob_by_period.json`: `by_period/{2024, 2025, 2026}/a/kp` |
| Bands by month and banking state | `emp_state.json`: `state_bands/coef`, entries `up_s` and `up_s_bh` |
| ... with date x state fixed effects, bias-corrected | `rob_dayfe_t1.json`: `results/B/daystatefe_bc/kp` |

Table 16 (`tables/rob_dayfe_main.tex`) consists of rows of `tables/rob_dayfe_t1.tex`, which `rob_dayfe_t1.py`
writes; the full table there adds further bias corrections and footnote markers.

### Numbers quoted in the text

Besides the tables, the text quotes numbers from these files: `tables/emp_numbers.json`, `emp_mech.json`,
`emp_state.json`, `emp_extra.json`, `emp_mxn.json` and the `rob_*.json` files of the scripts above, in particular
`rob_ratio_ci.json` (bootstrap and Fieller intervals of the open/closed ratios, `rob_ratio_ci.py`),
`rob_btcturk_venue.json` (the 2020 comparison with BtcTurk), `rob_xc_eur_premica.json` (the euro area before
31 March 2025), `rob_carry_adjusted.json` (value-date carry), `rob_tcmb_institutions.json` (institutional
facts) and `numbers.txt` (values in the captions of Figures 6 and 7, written by `run_all.py`).

## Run times

Approximate times on 4 cores, for a run with the data already downloaded:

| Script | Time |
|---|---|
| `empirics_main.py` with `BOOT=0` (each robustness run) | under 1 minute |
| `empirics_size.py`, `empirics_extra.py` | a few seconds each |
| `rob_flow_spline.py`, `rob_eft_edges.py`, `rob_ratio_by_year.py`, `rob_jackknife_month.py` | under 30 seconds each |
| `rob_by_period.py`, `rob_size_dependence.py`, `rob_partner_banks.py`, `rob_fast_limit.py`, `rob_xc_eur.py`, `rob_xc_zar.py`, `rob_mexico_utc.py` | about 1 minute each |
| `rob_xc_pln.py` | about 1.5 minutes |
| `rob_pre_sample.py`, `rob_cf_period.py`, `rob_demand_ofi.py` | about 2 minutes each |
| `rob_montecarlo_calib.py` (defaults R = 500, RV = 200, 4 processes) | about 50 minutes for the simulations |
| downloads | several hours |

The remaining scripts were not timed. The slowest of them are `empirics_main.py` with the default 99 bootstrap
draws per month, `rob_ratio_ci.py` (a full bootstrap that re-profiles every band in each of 499 draws) and
`rob_dayfe_t1.py` (200 simulated panels).

## Code provenance

The scripts in `code/` are the authors' working scripts. For publication only their comments and docstrings
were edited; the executable code is unchanged. `tools/check_identical.py` verifies this: it removes the
docstrings from each script, parses both versions and compares their abstract syntax trees, which do not
contain comments (`python tools/check_identical.py ORIGINAL_CODE_DIR`). Two scripts read their own docstring
when they run: `rob_mexico_utc.py` stores the part before `Outputs:` in its JSON as `prespec`, and
`rob_xc_pln.py` stores that part and the part after the upper-case word `AUDIT` as `prespec` and `audit_note`.
Their edited docstrings keep both markers. The shipped `rob_mexico_utc.json` and `rob_xc_pln.json` hold the
text of the docstrings before editing; a rerun writes the edited text and the same numbers.

## Citation

```bibtex
@unpublished{WangZhao2026,
  author = {Wang, Jiayi and Zhao, Lanxin},
  title  = {Banking Hours and the Speed of Stablecoin Arbitrage},
  note   = {Working paper},
  year   = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The data are not part of this repository and remain
subject to the terms of their providers.
