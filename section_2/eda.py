"""Exploratory analysis figures for Section 2.4 of the report.

Run from the repository root:
    uv run --locked python -m section_2.eda

Reads data/raw/consumption_temp.csv and writes three figures (PDF) to
outputs/section_2/figures/. Also prints the numbers quoted in Section 2.4.
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", default="data/raw/consumption_temp.csv", help="path to the raw CSV")
parser.add_argument("--out", default="outputs/section_2/figures", help="folder for the figures")
args = parser.parse_args()
SRC, OUT = args.data, args.out
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False,
})

AREA = {"oslo": "NO1 Oslo", "stavanger": "NO2 Stavanger", "trondheim": "NO3 Trondheim",
        "tromsø": "NO4 Tromsø", "bergen": "NO5 Bergen", "helsingfors": "FI Helsinki"}
ORDER = list(AREA)

d = pd.read_csv(SRC, parse_dates=["time"])
d["local"] = [t.tz_localize("UTC").tz_convert("Europe/Helsinki" if l == "helsingfors" else "Europe/Oslo").tz_localize(None)
              for t, l in zip(d.time, d.location)]
d["date"] = d.local.dt.normalize()
d["hour"] = d.local.dt.hour
d["weekend"] = d.local.dt.dayofweek >= 5

# drop Helsinki start-up zeros for plotting
d = d[~((d.location == "helsingfors") & (d.consumption <= 0))]

daily = d.groupby(["location", "date"]).agg(cons=("consumption", "mean"), temp=("temperature", "mean")).reset_index()
first_last = daily.groupby("location").date.transform("min").eq(daily.date) | daily.groupby("location").date.transform("max").eq(daily.date)
daily = daily[~first_last].reset_index(drop=True)  # drop partial first/last days

def grid():
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
    return fig, axes.ravel()

# 1. Daily mean consumption over time
fig, axes = grid()
for ax, loc in zip(axes, ORDER):
    x = daily[daily.location == loc]
    ax.plot(x.date, x.cons, color=BLUE, lw=1.4)
    ax.set_title(AREA[loc], loc="left")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 7, 10, 1]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
for ax in axes[::3]: ax.set_ylabel("Daily mean consumption (MW)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_consumption_timeseries.pdf")

# 2. Consumption vs temperature, early vs late period
early = daily.date < "2022-10-01"
late = daily.date >= "2022-12-01"
fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
axes = axes.ravel()
for ax, loc in zip(axes, ORDER):
    m = daily.location == loc
    ax.scatter(daily[m & early].temp, daily[m & early].cons, s=10, color=BLUE, alpha=.7, lw=0, label="Apr–Sep 2022")
    ax.scatter(daily[m & late].temp, daily[m & late].cons, s=10, color=ORANGE, alpha=.7, lw=0, label="Dec 2022–Apr 2023")
    ax.set_title(AREA[loc], loc="left"); ax.set_ylim(bottom=0)
for ax in axes[3:]: ax.set_xlabel("Daily mean forecast temperature (°C)")
for ax in axes[::3]: ax.set_ylabel("Daily mean consumption (MW)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, markerscale=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{OUT}/fig_consumption_vs_temperature.pdf")

# 3. Normalised daily profile (winter, Dec–Feb), weekday vs weekend
w = d[d.local.dt.month.isin([12, 1, 2])].copy()
w["rel"] = w.consumption / w.groupby(["location", "date"]).consumption.transform("mean")
prof = w.groupby(["location", "weekend", "hour"]).rel.mean().reset_index()
fig, axes = grid()
for ax, loc in zip(axes, ORDER):
    for we, c, lab in [(False, BLUE, "Weekday"), (True, ORANGE, "Weekend")]:
        x = prof[(prof.location == loc) & (prof.weekend == we)]
        ax.plot(x.hour, x.rel, color=c, lw=2, label=lab)
    ax.axhline(1, color=MUTED, lw=.8, ls=":")
    ax.set_title(AREA[loc], loc="left"); ax.set_xticks([0, 6, 12, 18, 23])
for ax in axes[3:]: ax.set_xlabel("Hour of day (local time)")
for ax in axes[::3]: ax.set_ylabel("Consumption / daily mean")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{OUT}/fig_daily_profile_winter.pdf")

# Numbers for the text
print("Profile amplitude (max/min of weekday winter profile):")
print(prof[~prof.weekend].groupby("location").rel.agg(lambda s: round(s.max() / s.min(), 2)))
print("Peak hour weekday winter:", prof[~prof.weekend].loc[lambda x: x.groupby("location").rel.idxmax()][["location", "hour"]].values.tolist())
wk = w.groupby(["location", "weekend"]).consumption.mean().unstack()
print("Winter weekend vs weekday %:", ((wk[True] / wk[False] - 1) * 100).round(1).to_dict())

# Seasonal-naive baseline (same hour 168 h earlier) and lag correlations
hh = d.sort_values(["location", "time"]).copy()
rows = []
for loc, x in hh.groupby("location"):
    s = x.set_index("time").consumption
    b = s.shift(168, freq="h").reindex(s.index)
    e = (s - b).dropna()
    rows.append({"area": AREA[loc], "MAE": e.abs().mean(), "RMSE": np.sqrt((e**2).mean()),
                     "rel_MAE_pct": 100 * e.abs().mean() / s.loc[e.index].mean(),
                     "bias": e.mean(), "corr_lag144": s.corr(s.shift(144, freq="h").reindex(s.index)),
                     "corr_lag168": s.corr(b)})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# Christmas check: Oslo daily consumption 18 Dec – 3 Jan
o = daily[(daily.location == "oslo") & (daily.date >= "2022-12-15") & (daily.date <= "2023-01-05")]
print(o[["date", "cons", "temp"]].round({"cons": 2, "temp": 2}).to_string(index=False))

#-----------------------------------------------------------------------#
# Numbers quoted in Sections 2.3 and 2.4 that are not read off a figure #
#-----------------------------------------------------------------------#

# Table 1 (Section 2.3): mean hourly consumption in hours with forecast
# temperature in (0, 5] degrees C, spring 2022 vs late winter 2023.
mild = d[(d.temperature > 0) & (d.temperature <= 5)]
spring22 = mild[mild.time < "2022-06-01"].groupby("location").consumption.mean()
winter23 = mild[mild.time >= "2023-02-15"].groupby("location").consumption.mean()
tab1 = pd.DataFrame({"Apr-May 2022": spring22, "15 Feb-Apr 2023": winter23})
tab1["ratio"] = tab1["15 Feb-Apr 2023"] / tab1["Apr-May 2022"]
tab1.index = [AREA[l] for l in tab1.index]
print("\nTable 1 - mean hourly consumption (MW) at 0-5 C forecast temperature:")
print(tab1.round(2).to_string())

# Helsinki step in July 2022 (Section 2.3): daily mean consumption.
h = daily[(daily.location == "helsingfors") & (daily.date >= "2022-07-20") & (daily.date <= "2022-07-31")]
print("\nHelsinki daily mean consumption, 20-31 July 2022 (MW):")
print(h[["date", "cons"]].round({"cons": 2}).to_string(index=False))

# Table 3 and Section 2.4 text: Pearson correlation between daily mean
# consumption and daily mean forecast temperature, for the whole dataset and
# for Dec 2022-Apr 2023, plus the slope of a linear fit in the winter period
# relative to mean consumption (% change per degree C).
rows = []
for loc in ORDER:
    x = daily[daily.location == loc]
    w = x[x.date >= "2022-12-01"]
    slope = np.polyfit(w.temp, w.cons, 1)[0]
    rows.append({"area": AREA[loc],
                 "corr_full": x.cons.corr(x.temp),
                 "corr_winter": w.cons.corr(w.temp),
                 "pct_per_degree_colder": -100 * slope / w.cons.mean()})
print("\nTable 3 - consumption vs temperature (daily means):")
print(pd.DataFrame(rows).round(2).to_string(index=False))