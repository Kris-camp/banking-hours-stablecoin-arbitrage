"""Figure: the USDT/TRY premium around the Ramadan holiday of April 2024 (figures/fig_ramadan_2024.pdf).

Five-minute premium from data_raw/a5.parquet, Istanbul time, 8-17 April 2024. Banks were closed from 13:00 on
9 April (half day), on the public holidays of 10-12 April and over the weekend of 13-14 April (weekends are not
in the panel). Shading marks hours in which banks are closed; the dashed line is the upper edge of the April
2024 monthly band.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "figures"))
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"

a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
s = a.loc["2024-04-07 21:00":"2024-04-17 21:00"].copy()
t = s.index.tz_convert("Europe/Istanbul").tz_localize(None)
b = pd.Series(s.b.values, index=t)
# break the line across gaps longer than 30 minutes
b = b.reindex(pd.date_range(t.min(), t.max(), freq="5min"))
hol = [pd.Timestamp(f"2024-04-{d}") for d in (10, 11, 12, 13, 14)]
fig, ax = plt.subplots(figsize=(7.0, 3.2))
for d in pd.date_range("2024-04-08", "2024-04-17"):
    if d in hol:
        ax.axvspan(d, d + pd.Timedelta(days=1), color=C3, alpha=0.18, lw=0)
    else:
        close = pd.Timedelta(hours=13) if d == pd.Timestamp("2024-04-09") else pd.Timedelta(hours=17)
        ax.axvspan(d, d + pd.Timedelta(hours=9), color=C3, alpha=0.18, lw=0)
        ax.axvspan(d + close, d + pd.Timedelta(days=1), color=C3, alpha=0.18, lw=0)
ax.plot(b.index, b.values, color=C1, lw=0.8)
ax.axhline(s.u.iloc[0], color=C2, lw=0.9, ls="--", label="upper edge of the April band")
ax.set_xlim(pd.Timestamp("2024-04-08"), pd.Timestamp("2024-04-18"))
ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d Apr"))
ax.tick_params(axis="x", labelsize=7.5)
ax.set_ylabel("premium (bp)")
ax.text(pd.Timestamp("2024-04-11 12:00"), 600, "public holidays", ha="center", va="top", fontsize=8)
ax.text(pd.Timestamp("2024-04-14 00:00"), 600, "weekend\n(not in panel)", ha="center", va="top", fontsize=8, color="#444444")
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ramadan_2024.pdf")); plt.close(fig)
print("ok", int(b.notna().sum()))
