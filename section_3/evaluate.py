import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from section_3.models import load_features, predict_day, train_models
from section_3.prepare_data import FEATURE_COLUMNS, consumption_cutoff


VALIDATION_START = "2022-11-01"
VALIDATION_END = "2022-12-26"
TEST_START = "2023-01-01"
TEST_END = "2023-04-01"

FEATURE_SETS = {
    "full": FEATURE_COLUMNS,
    "without_14_day_lag": [
        column for column in FEATURE_COLUMNS if column != "consumption_lag_336h"
    ],
    "without_history_mean": [
        column for column in FEATURE_COLUMNS if column != "available_mean_168h"
    ],
    "without_both": [
        column for column in FEATURE_COLUMNS
        if column not in ["consumption_lag_336h", "available_mean_168h"]
    ],
}

PARAMETER_GRID = {
    "ridge": [{"ridge_alpha": value} for value in [0.1, 1.0, 10.0]],
    "gradient_boosting": [
        {"boosting_iterations": 100, "boosting_leaves": value} for value in [7, 15]
    ],
}


def check_period(data, start, end):
    days = pd.date_range(start, end, freq="D", tz="UTC")
    rows = data.loc[data["forecast_day_utc"].between(days[0], days[-1])]
    counts = rows.groupby(["forecast_day_utc", "price_area"]).size()
    expected = pd.MultiIndex.from_product(
        [days, sorted(data["price_area"].unique())],
        names=["forecast_day_utc", "price_area"],
    )
    if not counts.reindex(expected, fill_value=0).eq(24).all():
        raise ValueError(f"Incomplete forecast days between {start} and {end}.")


def calculate_metrics(predictions, method):
    values = predictions[["consumption", "weekly_baseline", method]]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Observed consumption and forecasts must be finite.")

    error = predictions["consumption"] - predictions[method]
    mae = float(error.abs().mean())
    baseline_mae = float(
        (predictions["weekly_baseline"] - predictions["consumption"]).abs().mean()
    )
    improvement = 100 * (1 - mae / baseline_mae) if baseline_mae > 0 else None
    return {
        "hours": len(predictions),
        "mae_mw": mae,
        "rmse_mw": float(np.sqrt((error ** 2).mean())),
        "mean_error_mw": float(error.mean()),
        "mae_improvement_pct": improvement,
    }


def summarise(predictions):
    rows = []
    groups = [("all", predictions)] + list(predictions.groupby("price_area"))
    for area, group in groups:
        for method in ["weekly_baseline", "ridge", "gradient_boosting"]:
            rows.append({
                "price_area": area,
                "method": method,
                **calculate_metrics(group, method),
            })
    return pd.DataFrame(rows)


def run_period(data, start, end, method, parameters, feature_set):
    predictions = []
    days = pd.date_range(start, end, freq="D", tz="UTC")
    for number, day in enumerate(days, start=1):
        models = train_models(
            data,
            day,
            features=FEATURE_SETS[feature_set],
            model_name=method,
            **parameters,
        )
        daily = predict_day(data, day, models)
        daily["forecast_day_utc"] = day
        daily["training_cutoff_utc"] = consumption_cutoff(day)
        predictions.append(daily)
        if number % 14 == 0 or number == len(days):
            print(f"  Completed {number}/{len(days)} days", flush=True)
    return pd.concat(predictions, ignore_index=True)


def combine_predictions(predictions):
    columns = [
        "time", "location", "price_area", "consumption", "weekly_baseline",
        "forecast_day_utc", "training_cutoff_utc",
    ]
    combined = next(iter(predictions.values()))[columns].copy()
    for method, rows in predictions.items():
        if len(rows) != len(combined):
            raise ValueError("Methods must be evaluated on the same hours.")
        combined = combined.merge(
            rows[["time", "price_area", method]],
            on=["time", "price_area"],
            how="left",
            validate="one_to_one",
        )
    return combined.sort_values(["time", "price_area"]).reset_index(drop=True)


def evaluate(data, output_dir, input_hash):
    check_period(data, VALIDATION_START, VALIDATION_END)
    check_period(data, TEST_START, TEST_END)
    validation_end = pd.Timestamp(VALIDATION_END, tz="UTC")
    if validation_end >= consumption_cutoff(TEST_START):
        raise ValueError("Validation targets are not available at the first test origin.")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    selected = {}
    validation_predictions = {}
    candidate_count = len(FEATURE_SETS) * sum(len(settings) for settings in PARAMETER_GRID.values())

    for method, settings in PARAMETER_GRID.items():
        for feature_set, features in FEATURE_SETS.items():
            for parameters in settings:
                candidate_id = len(candidates) + 1
                print(
                    f"Validation candidate {candidate_id}/{candidate_count} "
                    f"{method} {feature_set} {parameters}",
                    flush=True,
                )
                predictions = run_period(
                    data, VALIDATION_START, VALIDATION_END,
                    method, parameters, feature_set,
                )
                metrics = calculate_metrics(predictions, method)
                candidates.append({
                    "candidate_id": candidate_id,
                    "method": method,
                    "feature_set": feature_set,
                    "feature_count": len(features),
                    **parameters,
                    **metrics,
                })
                pd.DataFrame(candidates).to_csv(
                    output_dir / "validation_candidates.csv", index=False
                )

                if method not in selected or metrics["mae_mw"] < selected[method]["validation_mae_mw"]:
                    selected[method] = {
                        "candidate_id": candidate_id,
                        "parameters": parameters,
                        "feature_set": feature_set,
                        "features": features,
                        "validation_mae_mw": metrics["mae_mw"],
                    }
                    validation_predictions[method] = predictions
                print(f"  Validation MAE {metrics['mae_mw']:.6f} MW", flush=True)

    candidate_table = pd.DataFrame(candidates)
    selected_ids = [choice["candidate_id"] for choice in selected.values()]
    candidate_table["selected"] = candidate_table["candidate_id"].isin(selected_ids)
    candidate_table.to_csv(output_dir / "validation_candidates.csv", index=False)

    selection = {
        "prepared_data_sha256": input_hash,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "validation_period": [VALIDATION_START, VALIDATION_END],
        "test_period": [TEST_START, TEST_END],
        "selection_metric": "pooled validation MAE across all six price areas",
        "training_window": "all prepared targets strictly before forecast date minus 5 days",
        "error_definition": "observed consumption minus prediction; positive means underforecasting",
        "selected_models": selected,
    }
    (output_dir / "selected_models.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    validation = combine_predictions(validation_predictions)
    validation.to_csv(output_dir / "validation_predictions.csv", index=False)
    summarise(validation).to_csv(output_dir / "validation_metrics.csv", index=False)
    print("Model settings and feature sets are now fixed before test evaluation.", flush=True)

    test_predictions = {}
    for method, choice in selected.items():
        print(f"Test period for {method}", flush=True)
        test_predictions[method] = run_period(
            data, TEST_START, TEST_END,
            method, choice["parameters"], choice["feature_set"],
        )
    test = combine_predictions(test_predictions)
    test.to_csv(output_dir / "test_predictions.csv", index=False)
    test_metrics = summarise(test)
    test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    print(test_metrics.loc[test_metrics["price_area"].eq("all")].round(4).to_string(index=False))
    print(f"Saved evaluation results to {output_dir}", flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate models and evaluate the final test period.")
    parser.add_argument("--input", type=Path, default=root / "data/processed/forecast_features.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/section_3/evaluation")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Prepared data not found. Run section_3.prepare_data first.")

    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    data = load_features(args.input)
    with threadpool_limits(limits=1):
        evaluate(data, args.output_dir, input_hash)


if __name__ == "__main__":
    main()
