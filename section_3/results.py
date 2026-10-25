import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


METHODS = {
    "weekly_baseline": ("Weekly baseline", "#666666", "--"),
    "ridge": ("Ridge", "#0072B2", "-."),
    "gradient_boosting": ("Gradient boosting", "#D55E00", "-"),
}


def plot_test_errors(predictions, output_dir):
    errors = pd.DataFrame()
    errors["date"] = pd.to_datetime(predictions["time"], utc=True).dt.floor("D")
    for method in METHODS:
        errors[method] = (predictions["consumption"] - predictions[method]).abs()

    daily = errors.groupby("date")[list(METHODS)].mean()
    weekly = daily.rolling(7, min_periods=7).mean().dropna()

    figure, axis = plt.subplots(figsize=(8.4, 3.6))
    for method, (label, colour, style) in METHODS.items():
        axis.plot(weekly.index, weekly[method], label=label,
                  color=colour, linestyle=style, linewidth=1.8)

    axis.set_xlim(daily.index.min(), daily.index.max())
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Target date in 2023")
    axis.set_ylabel("Seven-day mean absolute error (MW)")
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    image = output_dir / "test_weekly_mae.png"
    figure.savefig(image, dpi=300)
    plt.close(figure)
    weekly.to_csv(output_dir / "test_weekly_mae.csv", index_label="date")
    return image


def analyse_test_errors(predictions, output_dir):
    rows = predictions.copy()
    rows["time"] = pd.to_datetime(rows["time"], utc=True)
    dates = rows["time"].dt.floor("D")
    week_number = (dates - dates.min()).dt.days // 7
    rows["week_start"] = dates.min() + pd.to_timedelta(week_number * 7, unit="D")

    error_columns = {}
    for method in METHODS:
        column = f"{method}_absolute_error"
        rows[column] = (rows["consumption"] - rows[method]).abs()
        error_columns[column] = method

    weekly = rows.groupby("week_start")[list(error_columns)].mean()
    weekly = weekly.rename(columns=error_columns)
    weekly_by_area = rows.groupby(["week_start", "price_area"])[list(error_columns)].mean()
    weekly_by_area = weekly_by_area.rename(columns=error_columns).reset_index()
    weekly_all = weekly.reset_index().assign(price_area="all")
    weekly_scores = pd.concat([weekly_all, weekly_by_area], ignore_index=True)
    weekly_scores["week_end"] = weekly_scores["week_start"] + pd.Timedelta(days=6)

    summary = []
    worst_forecasts = []
    sensitivity = []
    for method in METHODS:
        column = f"{method}_absolute_error"
        improvements = []
        for week in weekly.index:
            remaining = rows.loc[rows["week_start"].ne(week)]
            baseline_mae = remaining["weekly_baseline_absolute_error"].mean()
            model_mae = remaining[column].mean()
            improvement = 100 * (1 - model_mae / baseline_mae)
            improvements.append(improvement)
            sensitivity.append({
                "excluded_week_start": week,
                "method": method,
                "remaining_hours": len(remaining),
                "mae_mw": model_mae,
                "baseline_mae_mw": baseline_mae,
                "mae_improvement_pct": improvement,
            })

        summary.append({
            "method": method,
            "p95_absolute_error_mw": rows[column].quantile(0.95),
            "maximum_absolute_error_mw": rows[column].max(),
            "weeks_better_than_baseline": int(weekly[method].lt(weekly["weekly_baseline"]).sum()),
            "total_weeks": len(weekly),
            "leave_one_week_out_min_improvement_pct": min(improvements),
            "leave_one_week_out_max_improvement_pct": max(improvements),
        })

        worst = rows.nlargest(5, column)[["time", "price_area", "consumption", method]].copy()
        worst = worst.rename(columns={method: "forecast_mw"})
        worst["method"] = method
        worst["error_mw"] = worst["consumption"] - worst["forecast_mw"]
        worst["absolute_error_mw"] = worst["error_mw"].abs()
        worst_forecasts.append(worst)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(output_dir / "error_summary.csv", index=False)
    weekly_scores.to_csv(output_dir / "weekly_mae_by_area.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output_dir / "week_sensitivity.csv", index=False)
    pd.concat(worst_forecasts).to_csv(output_dir / "worst_forecasts.csv", index=False)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot and analyse errors from the completed test evaluation.")
    parser.add_argument("--input", type=Path, default=root / "outputs/section_3/evaluation/test_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/section_3/results")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Test predictions not found. Run section_3.evaluate first.")

    predictions = pd.read_csv(args.input)
    image = plot_test_errors(predictions, args.output_dir)
    analyse_test_errors(predictions, args.output_dir)
    print(f"Saved the test error figure to {image}")
    print(f"Saved error diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
