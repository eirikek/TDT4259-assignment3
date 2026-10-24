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

The raw file is never overwritten. Generated files are excluded from Git and
can be recreated by running the command again.
