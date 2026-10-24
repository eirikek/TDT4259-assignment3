"""Prepare Aneo data and features for section 3.1"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


AREAS = {
    "bergen": ("NO5", "Europe/Oslo"),
    "helsingfors": ("FI", "Europe/Helsinki"),
    "oslo": ("NO1", "Europe/Oslo"),
    "stavanger": ("NO2", "Europe/Oslo"),
    "tromsø": ("NO4", "Europe/Oslo"),
    "trondheim": ("NO3", "Europe/Oslo"),
}

FEATURE_COLUMNS = [
    "temperature",
    "hour_local",
    "weekday_local",
    "is_weekend",
    "month_local",
    "consumption_lag_168h",
    "consumption_lag_336h",
    "available_mean_168h",
]


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected = {"time", "location", "consumption", "temperature"}
    if set(data.columns) != expected:
        raise ValueError(f"Expected columns {sorted(expected)}, got {list(data.columns)}")
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="raise")
    for column in ["consumption", "temperature"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data.isna().any().any():
        raise ValueError("Missing source values must be investigated before preparation.")
    if not np.isfinite(data[["consumption", "temperature"]].to_numpy()).all():
        raise ValueError("Source contains non-finite numeric values.")
    if data.duplicated(["location", "time"]).any():
        raise ValueError("Duplicate location-time pairs must be resolved explicitly.")
    if not data["time"].eq(data["time"].dt.floor("h")).all():
        raise ValueError("All timestamps must be at the start of an hour.")
    unknown = set(data["location"]) - set(AREAS)
    if unknown:
        raise ValueError(f"Unknown locations: {sorted(unknown)}")
    if data["consumption"].lt(0).any():
        raise ValueError("Negative consumption must be investigated.")
    return data.sort_values(["location", "time"]).reset_index(drop=True)


def consumption_cutoff(forecast_day: object) -> pd.Timestamp:
    day = pd.Timestamp(forecast_day)
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    return day.normalize() - pd.Timedelta(days=5)


def clean_data(data: pd.DataFrame, exclude_startup_zeros: bool = True) -> pd.DataFrame:
    cleaned = data.sort_values(["location", "time"]).copy()
    if exclude_startup_zeros:
        helsinki = cleaned.loc[cleaned["location"].eq("helsingfors")]
        prefix = helsinki["consumption"].eq(0).cumprod().astype(bool)
        cleaned = cleaned.drop(index=helsinki.index[prefix])
    return cleaned.reset_index(drop=True)


def build_features(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts = []
    for location, group in cleaned.groupby("location", sort=True):
        block = group.sort_values("time").copy()
        area, time_zone = AREAS[location]
        block["price_area"] = area
        block["forecast_day_utc"] = block["time"].dt.floor("D")
        block["consumption_available_before_utc"] = (
            block["forecast_day_utc"] - pd.Timedelta(days=5)
        )
        local = block["time"].dt.tz_convert(time_zone)
        block["hour_local"] = local.dt.hour
        block["weekday_local"] = local.dt.weekday
        block["is_weekend"] = block["weekday_local"].ge(5).astype(int)
        block["month_local"] = local.dt.month

        history = block.set_index("time")["consumption"]
        for hours in [168, 336]:
            source_times = block["time"] - pd.Timedelta(hours=hours)
            block[f"lag_{hours}h_source_utc"] = source_times
            block[f"consumption_lag_{hours}h"] = history.reindex(source_times).to_numpy()
            if not source_times.lt(block["consumption_available_before_utc"]).all():
                raise ValueError(f"Unavailable lag for {location}.")

        # (end - 168h, end] contains exactly the 168 hours before the cutoff.
        block["mean_window_end_utc"] = (
            block["consumption_available_before_utc"] - pd.Timedelta(hours=1)
        )
        available_means = history.rolling("168h", min_periods=168).mean()
        block["available_mean_168h"] = available_means.reindex(
            block["mean_window_end_utc"]
        ).to_numpy()
        parts.append(block)

    featured = pd.concat(parts, ignore_index=True)
    sufficient_history = featured[FEATURE_COLUMNS].notna().all(axis=1)
    ready = featured.loc[sufficient_history].copy()
    day_sizes = ready.groupby(["location", "forecast_day_utc"])["time"].transform("size")
    complete = ready.loc[day_sizes.eq(24)].copy()
    complete = complete.sort_values(["time", "location"]).reset_index(drop=True)
    if complete.empty:
        raise ValueError("No complete forecast days have sufficient history.")

    details = {
        "rows_without_required_history": int((~sufficient_history).sum()),
        "rows_excluded_from_incomplete_forecast_days": int(len(ready) - len(complete)),
        "prepared_rows": int(len(complete)),
        "predictors": FEATURE_COLUMNS,
        "target": "consumption",
        "forecast_day_convention": "24 consecutive UTC hours; local calendar features",
        "consumption_cutoff": "strictly before forecast_day_utc minus 5 days",
        "holiday_features": "not included in this initial feature set",
        "by_location": [],
    }
    for location, group in complete.groupby("location", sort=True):
        details["by_location"].append({
            "location": location,
            "price_area": AREAS[location][0],
            "rows": int(len(group)),
            "forecast_days": int(group["forecast_day_utc"].nunique()),
            "first_hour_utc": group["time"].min().isoformat(),
            "last_hour_utc": group["time"].max().isoformat(),
        })
    return complete, details


def training_rows_at_origin(prepared: pd.DataFrame, forecast_day: object) -> pd.DataFrame:
    return prepared.loc[prepared["time"].lt(consumption_cutoff(forecast_day))].copy()


def describe_source(data: pd.DataFrame) -> dict:
    areas = []
    for location, group in data.groupby("location", sort=True):
        expected = pd.date_range(group["time"].min(), group["time"].max(), freq="h")
        areas.append({
            "location": location,
            "rows": int(len(group)),
            "first_hour_utc": group["time"].min().isoformat(),
            "last_hour_utc": group["time"].max().isoformat(),
            "missing_hours": int(len(expected.difference(group["time"]))),
            "zero_consumption_rows": int(group["consumption"].eq(0).sum()),
        })
    return {
        "rows": int(len(data)),
        "missing_values": {key: int(value) for key, value in data.isna().sum().items()},
        "duplicate_location_time_pairs": int(data.duplicated(["location", "time"]).sum()),
        "by_location": areas,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/raw/consumption_temp.csv")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--output", type=Path, default=root / "outputs/section_3")
    parser.add_argument("--keep-startup-zeros", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Dataset not found: {args.input}. See README.md for setup.")
    raw_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    raw = load_data(args.input)
    cleaned = clean_data(raw, exclude_startup_zeros=not args.keep_startup_zeros)
    prepared, preparation = build_features(cleaned)
    summary = {
        "source_sha256": raw_hash,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "source": describe_source(raw),
        "startup_zero_exclusion_enabled": not args.keep_startup_zeros,
        "startup_zero_rows_excluded": int(len(raw) - len(cleaned)),
        "cleaned_rows": int(len(cleaned)),
        "preparation": preparation,
    }
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.processed_dir / "consumption_cleaned.csv", index=False)
    prepared.to_csv(args.processed_dir / "forecast_features.csv", index=False)
    (args.output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Source: {len(raw):,} rows; cleaned: {len(cleaned):,}; prepared: {len(prepared):,}")
    print(f"Startup zeros excluded: {summary['startup_zero_rows_excluded']}")
    print(f"Forecast days by area: {[(r['price_area'], r['forecast_days']) for r in preparation['by_location']]}")
    print(f"Processed data: {args.processed_dir}")
    print(f"Summary: {args.output}")


if __name__ == "__main__":
    main()
