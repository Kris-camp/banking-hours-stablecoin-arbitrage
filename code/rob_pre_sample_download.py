"""Raw data for rob_pre_sample.py (Jan 2021 - Dec 2023), written under data_raw/pre_sample/ only.

Binance: USDTTRY one-minute klines 2021-01..2023-12 with download_binance.get (its OUT redirected to
         data_raw/pre_sample/binance, so data_raw/binance and the 2024-2026 panel are untouched).
CBRT: tcmb() writes data_raw/pre_sample/tcmb_indicative_rate_calendar_2021_2023.csv (see its docstring).
Dukascopy: USD/TRY one-minute BID and ASK candles for every weekday of 2021-2023, same URL scheme, file
         names and 404 convention (empty file) as download_dukascopy.fetch, written to
         data_raw/pre_sample/dukascopy. Dukascopy rate-limits, so this runs as one sequential process with a
         pause after every request (1-1.5 s) and exponential back-off on errors or HTTP 429
         (5, 10, 20, 40, 60, 60, ... s), over one keep-alive HTTPS session. Order: 2023 (BID and ASK), then
         the BID files of 2022 and of 2021, then the ASK files of 2022 and of 2021: the primary premium of
         rob_pre_sample.py needs only the BID (as in the 2024-2026 panel), the ASK serves its check C1, and
         an interrupted run leaves the most recent years complete. A second pass re-requests any file that
         failed; files that do not decompress are deleted and re-requested (validate()).
Usage: python rob_pre_sample_download.py [binance|tcmb|dukascopy|all]
"""
import os, sys, time, random, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import download_binance as DB
import download_dukascopy as DD

ROOT = os.path.join(HERE, "..", "data_raw", "pre_sample")
OUT_B, OUT_D = os.path.join(ROOT, "binance"), os.path.join(ROOT, "dukascopy")
os.makedirs(OUT_B, exist_ok=True); os.makedirs(OUT_D, exist_ok=True)


def binance():
    DB.OUT = OUT_B
    for ym in DB.months("2021-01", "2023-12"):
        print("USDTTRY", ym, DB.get("USDTTRY", ym), flush=True)


def dukascopy(pair="USDTRY"):
    import requests
    s = requests.Session(); s.headers["User-Agent"] = DD.UA
    def wdays(yr):
        d0, d1 = dt.date(yr, 1, 1), dt.date(yr, 12, 31)
        return [d0 + dt.timedelta(n) for n in range((d1 - d0).days + 1) if (d0 + dt.timedelta(n)).weekday() < 5]
    # 2023 BID and ASK; then the BID files of 2022 and 2021 (the primary, BID-based mid); then the ASK files of
    # 2022 and 2021 (check C1 only)
    jobs = [(d, side) for d in wdays(2023) for side in ("BID", "ASK")]
    jobs += [(d, "BID") for d in wdays(2022)] + [(d, "BID") for d in wdays(2021)]
    jobs += [(d, "ASK") for d in wdays(2022)] + [(d, "ASK") for d in wdays(2021)]
    t0, n_ok = time.time(), 0
    for i, (day, side) in enumerate(jobs):
        fn = os.path.join(OUT_D, f"{pair}_{side}_{day:%Y%m%d}.bi5")
        if os.path.exists(fn):
            continue
        url = (f"https://datafeed.dukascopy.com/datafeed/{pair}/{day.year}/{day.month - 1:02d}/{day.day:02d}/"
               f"{side}_candles_min_1.bi5")
        status, wait = "fail", 5
        for k in range(30):
            try:
                r = s.get(url, timeout=30)
                if r.status_code == 200:
                    open(fn, "wb").write(r.content); status = "ok"; n_ok += 1; break
                if r.status_code == 404:
                    open(fn, "wb").close(); status = "404"; break
                status = f"http{r.status_code}"
            except Exception as e:
                status = type(e).__name__
                s = requests.Session(); s.headers["User-Agent"] = DD.UA
            time.sleep(wait + random.random()); wait = min(2 * wait, 60)
        print(i, day, side, status, f"{time.time() - t0:.0f}s", flush=True)
        time.sleep(1.0 + 0.5 * random.random())
    print("done", n_ok, f"{time.time() - t0:.0f}s", flush=True)


def tcmb():
    """HTTP status of the CBRT indicative-rate file https://www.tcmb.gov.tr/kurlar/YYYYMM/DDMMYYYY.xml for every
    weekday of 2021-2023 (the CBRT publishes no indicative rate on public holidays or half working days), the same
    check as data_raw/tcmb_indicative_rate_calendar.csv for 2024-2026. Used to verify the 2021-2023 holiday list."""
    import requests
    s = requests.Session(); s.headers["User-Agent"] = DD.UA
    fn = os.path.join(ROOT, "tcmb_indicative_rate_calendar_2021_2023.csv")
    d0, d1 = dt.date(2021, 1, 1), dt.date(2023, 12, 31)
    days = [d0 + dt.timedelta(n) for n in range((d1 - d0).days + 1)]
    rows = ["day,status"]
    for d in days:
        if d.weekday() >= 5:
            continue
        url = f"https://www.tcmb.gov.tr/kurlar/{d:%Y%m}/{d:%d%m%Y}.xml"
        st = "err"
        for k in range(5):
            try:
                st = str(s.get(url, timeout=30).status_code); break
            except Exception:
                time.sleep(2 + 2 * k)
        rows.append(f"{d},{st}")
        time.sleep(0.3)
    open(fn, "w").write("\n".join(rows) + "\n")
    print("tcmb calendar", len(rows) - 1, "days", flush=True)


def validate(pair="USDTRY"):
    """Delete files that do not decompress (e.g. a write interrupted by a restart) so the next pass refetches them."""
    import lzma
    bad = 0
    for f in sorted(os.listdir(OUT_D)):
        fn = os.path.join(OUT_D, f)
        if os.path.getsize(fn) == 0:
            continue
        try:
            lzma.decompress(open(fn, "rb").read())
        except Exception:
            os.remove(fn); bad += 1
    print("validate: removed", bad, "corrupt files", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("binance", "all"):
        binance()
    if what in ("tcmb", "all"):
        tcmb()
    if what in ("dukascopy", "all"):
        validate(); dukascopy(); validate(); dukascopy()
