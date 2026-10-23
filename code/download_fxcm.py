"""Download FXCM USD/TRY one-minute bid/ask candles (one gzipped CSV per week).

Source: https://candledata.fxcorporate.com/m1/USDTRY/{year}/{week}.csv.gz
Columns: DateTime (MM/DD/YYYY HH:MM:SS.fff, UTC, bar start), BidOpen, BidHigh, BidLow,
BidClose, AskOpen, AskHigh, AskLow, AskClose. FXCM weeks run Sunday-Friday; files are
stored verbatim (no parsing here) in data_raw/fxcm/USDTRY_{year}_{week:02d}.csv.gz.
A 404 is recorded as an empty file so that it is not retried.
Usage: python -P code/download_fxcm.py [PAIR] [Y0] [Y1]
"""
import os, sys, time, random, urllib.request, urllib.error

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_raw", "fxcm")
os.makedirs(OUT, exist_ok=True)
UA = "Mozilla/5.0 (X11; Linux x86_64) research download (academic, polite)"


def fetch(pair, year, week, tries=8):
    fn = os.path.join(OUT, f"{pair}_{year}_{week:02d}.csv.gz")
    if os.path.exists(fn): return "cached"
    url = f"https://candledata.fxcorporate.com/m1/{pair}/{year}/{week}.csv.gz"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if data[:2] != b"\x1f\x8b":          # not gzip: treat as a failed fetch
                raise ValueError("not gzip")
            tmp = fn + ".part"
            open(tmp, "wb").write(data); os.replace(tmp, fn)
            return f"ok {len(data)}"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(fn, "wb").close(); return "404"
        except Exception:
            pass
        time.sleep(2 ** min(k, 5) + random.random())
    return "fail"


if __name__ == "__main__":
    pair = sys.argv[1] if len(sys.argv) > 1 else "USDTRY"
    y0, y1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2024, 2026)
    jobs = [(2023, w) for w in (52, 53)] + [(y, w) for y in range(y0, y1 + 1) for w in range(1, 54)]
    for y, w in jobs:
        if y == 2026 and w > 37: continue        # sample ends 31 August 2026
        r = fetch(pair, y, w)
        print(y, w, r, flush=True)
        if r != "cached": time.sleep(0.5 + random.random())   # politeness
