# data_raw/

This folder holds the raw market data and the intermediate panels. It is empty in the repository: the full
folder is about 2 GB, and the Dukascopy and FXCM data may not be redistributed. The scripts in `code/`
download every file from public sources and rebuild the panels. All commands below are run from `code/`;
every script finds this folder as `../data_raw` relative to its own location.

Downloads skip files that already exist, so an interrupted download can simply be restarted. A file that the
source reports as missing (HTTP 404) is stored as an empty file so that it is not requested again.

## 1. Files downloaded from public sources

| Folder | Source | Command (from `code/`) | Coverage used in the paper |
|---|---|---|---|
| `binance/` | Binance public data archive, spot monthly 1-minute klines, <https://data.binance.vision> | `python download_binance.py` | USDTTRY, BTCTRY, BTCUSDT, 2023-06 to 2026-08 (the default run also fetches USDTNGN and BTCNGN, which the paper does not use) |
| | | `python download_binance.py USDTMXN 2024-11 2026-08` | USDT/MXN |
| | | `python download_binance.py EURUSDT 2024-01 2026-08` | EUR/USDT |
| | | `python download_binance.py USDTZAR 2024-01 2026-08` | USDT/ZAR |
| | | `python download_binance.py USDTPLN 2024-01 2026-07` | USDT/PLN (Binance has no 2026-08 file) |
| `dukascopy/` | Dukascopy historical data feed, 1-minute candles (<https://www.dukascopy.com/swiss/english/marketwatch/historical/>; files from `https://datafeed.dukascopy.com/datafeed/`) | `python download_dukascopy.py` | USD/TRY bid, every weekday 2023-06-01 to 2026-08-31 |
| | | `python download_dukascopy.py USDMXN 2024-11-01 2026-08-31` | USD/MXN bid |
| | | `python download_dukascopy.py EURUSD 2024-01-01 2026-08-31` | EUR/USD bid |
| | | `python download_dukascopy.py USDZAR 2024-01-01 2026-08-31` | USD/ZAR bid |
| | | `python download_dukascopy.py USDPLN 2024-01-01 2026-07-01` | USD/PLN bid |
| | | ask candles: see below | |
| `fxcm/` | FXCM historical candle data, <https://candledata.fxcorporate.com> | `python download_fxcm.py` | USD/TRY bid and ask, 2023 weeks 52-53 and 2024 to 2026 week 37 |
| `ofi_size/` | Binance aggTrades, USDT/TRY | `python aggtrades_ofi.py 2024-01 2026-08` | signed flow by trade size per five minutes (the monthly archives are processed in memory and not kept) |
| `ofi_size_mxn/` | Binance aggTrades, USDT/MXN | `python rob_mexico_utc_aggtrades.py 2024-11 2026-08` | as above, for Mexico |
| `pre_sample/` | Binance, Dukascopy and the CBRT indicative-rate archive, 2021-2023 | `python rob_pre_sample_download.py all` | USDTTRY klines 2021-01 to 2023-12, USD/TRY bid and ask for every weekday of 2021-2023, and `tcmb_indicative_rate_calendar_2021_2023.csv` |
| `btcturk_venue/` | BtcTurk public API (`https://graph-api.btcturk.com`), Binance and Dukascopy, December 2019 to December 2020 | downloaded by `python rob_btcturk_venue.py` itself | |
| `tcmb_indicative_rate_calendar.csv` | CBRT indicative exchange-rate archive, <https://www.tcmb.gov.tr/kurlar/> | see below | HTTP status for every weekday 2024-01-01 to 2026-08-31 |

### Dukascopy ask candles

`download_dukascopy.py` downloads bid candles only. The ask candles below were fetched with its `fetch` function.
They matter: `build_panel.load_fx` uses the bid/ask mid on days with an ask file and, on every other day, the bid
plus half of the median relative spread measured on those days. In particular, the twelve USD/TRY ask files of
14-31 August 2026 set that spread for the whole USD/TRY benchmark, so they are needed to reproduce
`a5.parquet` exactly.

| Pair | Ask files used |
|---|---|
| USDTRY | every weekday 2026-08-14 to 2026-08-31 (12 days) |
| EURUSD | every weekday 2024-01-01 to 2026-08-31 |
| USDZAR | every weekday 2024-01-01 to 2026-08-31 |
| USDPLN | one day a month: 2024-01-10, 2024-02-14, 2024-03-13, 2024-04-10, 2024-05-08, 2024-06-12, 2024-07-10, 2024-08-14, 2024-09-11, 2024-10-09, 2024-11-13, 2024-12-11, 2025-01-08, 2025-02-12, 2025-03-12, 2025-04-09, 2025-05-14, 2025-06-11, 2025-07-09, 2025-08-13, 2025-09-10, 2025-10-08, 2025-11-12, 2025-12-10, 2026-01-14, 2026-02-11, 2026-03-11, 2026-04-08, 2026-05-13, 2026-06-10 |

Run from `code/`:

```python
import datetime as dt
import download_dukascopy as D

def weekdays(first, last):
    d0, d1 = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    days = (d0 + dt.timedelta(n) for n in range((d1 - d0).days + 1))
    return [d for d in days if d.weekday() < 5]

jobs = [("USDTRY", d) for d in weekdays("2026-08-14", "2026-08-31")]
jobs += [(p, d) for p in ("EURUSD", "USDZAR") for d in weekdays("2024-01-01", "2026-08-31")]
jobs += [("USDPLN", dt.date.fromisoformat(s)) for s in [
    "2024-01-10", "2024-02-14", "2024-03-13", "2024-04-10", "2024-05-08", "2024-06-12", "2024-07-10", "2024-08-14",
    "2024-09-11", "2024-10-09", "2024-11-13", "2024-12-11", "2025-01-08", "2025-02-12", "2025-03-12", "2025-04-09",
    "2025-05-14", "2025-06-11", "2025-07-09", "2025-08-13", "2025-09-10", "2025-10-08", "2025-11-12", "2025-12-10",
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-08", "2026-05-13", "2026-06-10"]]
for pair, day in jobs:
    print(pair, day, D.fetch(day, "ASK", pair=pair), flush=True)
```

### CBRT indicative-rate calendar, 2024-2026

`rob_tcmb_institutions.py` reads `tcmb_indicative_rate_calendar.csv` (columns `day,status`): the HTTP status of
the CBRT indicative-rate file for every weekday from 2024-01-01 to 2026-08-31. The CBRT publishes no indicative
rate on public holidays and half working days, so a 404 marks such a day. It is the same check that `tcmb()` in
`rob_pre_sample_download.py` runs for 2021-2023. Run from `code/`:

```python
import datetime as dt, time, requests

rows, day, last = ["day,status"], dt.date(2024, 1, 1), dt.date(2026, 8, 31)
s = requests.Session()
while day <= last:
    if day.weekday() < 5:
        url = f"https://www.tcmb.gov.tr/kurlar/{day:%Y%m}/{day:%d%m%Y}.xml"
        rows.append(f"{day},{s.get(url, timeout=30).status_code}")
        time.sleep(0.3)
    day += dt.timedelta(1)
open("../data_raw/tcmb_indicative_rate_calendar.csv", "w").write("\n".join(rows) + "\n")
```

## 2. Files built by the scripts

| File | Written by |
|---|---|
| `panel_TRY.parquet` | `python build_panel.py TRY` (minute panel: Binance USDT/TRY VWAP, order flow, volume, the BTC-implied price) |
| `a5.parquet` | `python empirics_main.py` (run without `TAG`): the five-minute estimation panel with the monthly bands, read by most other scripts |
| `state_bands.parquet` | `python empirics_state.py`: bands by month and banking state |
| `pre_sample/a5_pre.parquet` | `python rob_pre_sample.py`: the 2021-2023 panel |

## 3. Notes

- Dukascopy rate-limits its feed. The download scripts retry with pauses; a full download can take several
  hours.
- FXCM has no file for some weeks (404, stored as empty files; 23 of 145 USD/TRY week files in the authors'
  copy).
- The authors downloaded the data in October 2026. The providers can revise or withdraw archived files, so a
  later download may differ slightly.
- Files in the authors' copy, for comparison: `binance/` USDTTRY, BTCTRY and BTCUSDT 39 months each, EURUSDT 32,
  USDTZAR 32, USDTPLN 31, USDTMXN 22; `dukascopy/` USDTRY 848 bid and 12 ask files, EURUSD and USDZAR 696 bid
  and 696 ask files each, USDPLN 653 bid and 30 ask files, USDMXN 477 bid files; `ofi_size/` 32 and
  `ofi_size_mxn/` 22 monthly files; `pre_sample/dukascopy/` 781 bid and 781 ask files.
