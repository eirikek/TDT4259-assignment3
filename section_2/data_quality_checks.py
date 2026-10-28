"""Data-quality checks behind Section 2.3 of the report.

Run from the repository root:
    uv run --locked python -m section_2.data_quality_checks

Reads data/raw/consumption_temp.csv and prints missing values, duplicate
(time, location) pairs, gaps, zero values and coverage per price area.
"""
import argparse

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", default="data/raw/consumption_temp.csv", help="path to the raw CSV")
args = parser.parse_args()

d = pd.read_csv(args.data, parse_dates=["time"])
print(f"Rows: {len(d)}")

# 1. Missing values per column
print("\nMissing values per column:\n", d.isna().sum().to_string())

# 2. Duplicate (time, location) pairs
print("\nDuplicate (time, location) rows:", d.duplicated(["time", "location"]).sum())

# 3. Coverage, gaps and zero/negative values per location
rows = []
for loc, x in d.groupby("location"):
    full = pd.date_range(x.time.min(), x.time.max(), freq="h")  # every hour that should exist
    missing = full.difference(x.time)
    zeros = x[x.consumption <= 0]
    rows.append({
        "location": loc, "n_hours": len(x), "start_utc": x.time.min(), "end_utc": x.time.max(),
        "missing_hours": len(missing), "zero_or_negative": len(zeros),
        "zero_period": f"{zeros.time.min()} - {zeros.time.max()}" if len(zeros) else "",
        "temp_min": x.temperature.min(), "temp_max": x.temperature.max(),
    })
print("\nPer location:\n", pd.DataFrame(rows).to_string(index=False))