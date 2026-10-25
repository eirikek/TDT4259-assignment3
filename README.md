# TDT4259 Assignment 3

Day-ahead electricity consumption forecasting for the Aneo retailer case.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then run the following commands from the repository root.

```bash
uv sync --locked
```

## Data

Use the original `consumption_temp.csv` and place your local copy at:

```text
data/raw/consumption_temp.csv
```

Keep the filename unchanged. The data file is excluded from Git.

## Run

Prepare the data for section 3.1:

```bash
uv run --locked python -m section_3.prepare_data
```

This creates:

- `data/processed/consumption_cleaned.csv`
- `data/processed/forecast_features.csv`
- `outputs/section_3/preparation_summary.json`

Generate example forecasts for a historical day with the initial model settings.
For example, to forecast 1 November 2022, run

```bash
uv run --locked python -m section_3.models --forecast-day 2022-11-01
```

This saves `outputs/section_3/predictions_2022-11-01.csv` with the observed
consumption and predictions from the weekly baseline, Ridge and gradient boosting.
Each price area has its own models, trained on consumption from 26 October
2022 or earlier for this example.

Run the full validation and test evaluation for section 3.3

```bash
uv run --locked python -m section_3.evaluate
```

This selects model settings and feature sets using 1 November to 26 December
2022, then evaluates the selected models from 1 January to 1 April 2023.
Results are saved in `outputs/section_3/evaluation/`, including candidate
scores, selected settings, daily predictions and metrics for each price area.

The raw file is never overwritten. Generated files are excluded from Git and
can be recreated by running the commands again.
