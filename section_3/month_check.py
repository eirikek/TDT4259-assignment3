import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from section_3.evaluate import (
    calculate_metrics, check_period, combine_predictions, run_period, summarise,
)
from section_3.models import load_features
from section_3.results import analyse_test_errors, plot_test_errors


def read_predictions(path):
    data = pd.read_csv(path)
    for column in ["time", "forecast_day_utc", "training_cutoff_utc"]:
        data[column] = pd.to_datetime(data[column], utc=True)
    return data


def area_metrics(predictions, method):
    return pd.DataFrame([
        {"price_area": area, **calculate_metrics(rows, method)}
        for area, rows in predictions.groupby("price_area")
    ])


def compare_month(data, original_dir, output_dir, original):
    validation_start, validation_end = original["validation_period"]
    test_start, test_end = original["test_period"]
    check_period(data, validation_start, validation_end)
    check_period(data, test_start, test_end)
    initial_validation = read_predictions(original_dir / "validation_predictions.csv")
    initial_test = read_predictions(original_dir / "test_predictions.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = []
    area_comparisons = []
    selected = {}
    validation_forecasts = {}
    for method, choice in original["selected_models"].items():
        best_mae = float("inf")
        for month_included in [True, False]:
            features = [column for column in choice["features"]
                        if month_included or column != "month_local"]
            print(f"Validation {method}, month included {month_included}", flush=True)
            predictions = initial_validation if month_included else run_period(
                data, validation_start, validation_end, method,
                choice["parameters"], choice["feature_set"], features=features,
            )
            metrics = calculate_metrics(predictions, method)
            by_area = area_metrics(predictions, method)
            comparisons.append({
                "method": method,
                "month_included": month_included,
                "feature_count": len(features),
                **metrics,
                "equal_area_mae_improvement_pct": by_area["mae_improvement_pct"].mean(),
            })
            area_comparisons.append(by_area.assign(method=method, month_included=month_included))
            if metrics["mae_mw"] < best_mae:
                best_mae = metrics["mae_mw"]
                selected[method] = {
                    "parameters": choice["parameters"],
                    "feature_set": choice["feature_set"],
                    "features": features,
                    "month_included": month_included,
                    "validation_mae_mw": best_mae,
                }
                validation_forecasts[method] = predictions

    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output_dir / "month_validation.csv", index=False)
    pd.concat(area_comparisons).to_csv(output_dir / "month_validation_by_area.csv", index=False)
    selection = {
        "analysis_status": "Follow-up designed after inspecting the original test results",
        "prepared_data_sha256": original["prepared_data_sha256"],
        "versions": original["versions"],
        "validation_period": original["validation_period"],
        "test_period": original["test_period"],
        "selection_metric": original["selection_metric"],
        "comparison": "Keep each original model's settings and other features fixed; compare month included or removed",
        "selected_models": selected,
    }
    (output_dir / "selected_models.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("Month choices are fixed from validation before follow-up test scoring.", flush=True)

    test_forecasts = {}
    for method, choice in selected.items():
        print(f"Follow-up test {method}, month included {choice['month_included']}", flush=True)
        test_forecasts[method] = initial_test if choice["month_included"] else run_period(
            data, test_start, test_end, method,
            choice["parameters"], choice["feature_set"], features=choice["features"],
        )

    balanced = []
    for phase, forecasts in [("validation", validation_forecasts), ("test", test_forecasts)]:
        predictions = combine_predictions(forecasts)
        predictions.to_csv(output_dir / f"{phase}_predictions.csv", index=False)
        metrics = summarise(predictions)
        metrics.to_csv(output_dir / f"{phase}_metrics.csv", index=False)
        per_area = metrics.loc[metrics["price_area"].ne("all")]
        means = per_area.groupby("method")["mae_improvement_pct"].mean()
        balanced.extend({"phase": phase, "method": method,
                         "equal_area_mae_improvement_pct": value}
                        for method, value in means.items())
        if phase == "test":
            results_dir = output_dir / "results"
            plot = plot_test_errors(predictions, results_dir)
            plot.rename(results_dir / "test_weekly_mae_followup.png")
            analyse_test_errors(predictions, results_dir)
    pd.DataFrame(balanced).to_csv(output_dir / "area_balanced_metrics.csv", index=False)
    print(comparison.round(5).to_string(index=False), flush=True)
    print(f"Saved follow-up results to {output_dir}", flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compare the selected models with and without month.")
    parser.add_argument("--input", type=Path, default=root / "data/processed/forecast_features.csv")
    parser.add_argument("--original-dir", type=Path, default=root / "outputs/section_3/evaluation")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/section_3/month_check")
    args = parser.parse_args()
    original = json.loads((args.original_dir / "selected_models.json").read_text())
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != original["prepared_data_sha256"]:
        parser.error("Prepared data differs from the original evaluation.")
    data = load_features(args.input)
    with threadpool_limits(limits=1):
        compare_month(data, args.original_dir, args.output_dir, original)


if __name__ == "__main__":
    main()
