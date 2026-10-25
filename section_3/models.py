import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from section_3.prepare_data import FEATURE_COLUMNS, training_rows_at_origin


CALENDAR_FEATURES = ["hour_local", "weekday_local", "is_weekend", "month_local"]
NUMERIC_FEATURES = [
    "temperature",
    "consumption_lag_168h",
    "consumption_lag_336h",
    "available_mean_168h",
]


def load_features(path):
    data = pd.read_csv(path)
    for column in ["time", "forecast_day_utc"]:
        data[column] = pd.to_datetime(data[column], utc=True)
    return data


def create_models(ridge_alpha=1.0, boosting_iterations=100, boosting_leaves=15,
                  features=None):
    if features is None:
        features = FEATURE_COLUMNS
    calendar = [column for column in CALENDAR_FEATURES if column in features]
    numeric = [column for column in NUMERIC_FEATURES if column in features]
    preprocessing = ColumnTransformer([
        (
            "calendar",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            calendar,
        ),
        ("numeric", StandardScaler(), numeric),
    ])

    ridge = Pipeline([
        ("preprocessing", preprocessing),
        ("model", Ridge(alpha=ridge_alpha)),
    ])

    boosting = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=boosting_iterations,
        max_leaf_nodes=boosting_leaves,
        categorical_features=calendar,
        early_stopping=False,
        random_state=42,
    )

    return {"ridge": ridge, "gradient_boosting": boosting}


def train_models(data, forecast_day, ridge_alpha=1.0,
                 boosting_iterations=100, boosting_leaves=15,
                 features=None, model_name=None):
    if features is None:
        features = FEATURE_COLUMNS
    training = training_rows_at_origin(data, forecast_day)
    if training.empty:
        raise ValueError("No training data is available. Choose a later forecast day.")

    models_by_area = {}
    for area, rows in training.groupby("price_area"):
        models = create_models(ridge_alpha, boosting_iterations, boosting_leaves, features)
        if model_name is not None:
            models = {model_name: models[model_name]}
        for model in models.values():
            model.fit(rows[features], rows["consumption"])
        models_by_area[area] = models

    return models_by_area


def predict_day(data, forecast_day, models_by_area):
    day = pd.to_datetime(forecast_day, utc=True).normalize()
    target = data.loc[data["forecast_day_utc"].eq(day)]
    if target.empty:
        raise ValueError(f"No prepared data is available for {day.date()}.")

    day_sizes = target.groupby("price_area")["time"].nunique()
    duplicates = target.duplicated(["price_area", "time"]).any()
    if not day_sizes.eq(24).all() or duplicates:
        raise ValueError("Each area must have exactly 24 distinct target hours.")

    predictions = target[["time", "location", "price_area", "consumption"]].copy()
    predictions["weekly_baseline"] = target["consumption_lag_168h"]

    for area, rows in target.groupby("price_area"):
        if area not in models_by_area:
            raise ValueError(f"No trained models for {area}. Choose a later forecast day.")
        for name, model in models_by_area[area].items():
            predictions.loc[rows.index, name] = model.predict(rows[model.feature_names_in_])

    return predictions.sort_values(["time", "price_area"]).reset_index(drop=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Forecast one day for each price area.")
    parser.add_argument("--forecast-day", required=True, help="Target date as YYYY-MM-DD")
    parser.add_argument("--input", type=Path, default=root / "data/processed/forecast_features.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/section_3")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--boosting-iterations", type=int, default=100)
    parser.add_argument("--boosting-leaves", type=int, default=15)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error("Prepared data not found. Run section_3.prepare_data first.")

    data = load_features(args.input)
    models = train_models(
        data,
        args.forecast_day,
        args.ridge_alpha,
        args.boosting_iterations,
        args.boosting_leaves,
    )
    predictions = predict_day(data, args.forecast_day, models)

    day = pd.to_datetime(args.forecast_day, utc=True).date()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"predictions_{day}.csv"
    predictions.to_csv(output, index=False)
    print(f"Saved forecasts for {len(predictions)} area-hours to {output}")


if __name__ == "__main__":
    main()
