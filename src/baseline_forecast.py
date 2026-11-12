from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_hourly_demand.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_forecast_results.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_forecast_metrics.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

data = pd.read_csv(INPUT_FILE)

data["hour"] = pd.to_datetime(
    data["hour"]
)

data = data.sort_values(
    [
        "site_id",
        "hour"
    ]
)


# ---------------------------------------------------------
# CREATE TIME FEATURES
# ---------------------------------------------------------

data["hour_of_day"] = (
    data["hour"].dt.hour
)

data["day_of_week"] = (
    data["hour"].dt.dayofweek
)


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
#
# Train = Jan-May
# Test  = June
#
# Important:
# We use the PAST to predict the FUTURE.
# ---------------------------------------------------------

split_date = pd.Timestamp(
    "2025-06-01"
)

train = data[
    data["hour"] < split_date
].copy()

test = data[
    data["hour"] >= split_date
].copy()


print(
    "\n--- TIME SPLIT ---"
)

print(
    "Training:",
    train["hour"].min(),
    "to",
    train["hour"].max()
)

print(
    "Testing:",
    test["hour"].min(),
    "to",
    test["hour"].max()
)


# ---------------------------------------------------------
# HISTORICAL AVERAGE BASELINE
#
# Typical demand for:
#
# Site + Day of week + Hour of day
#
# Example:
# Frankfurt + Monday + 17:00
# ---------------------------------------------------------

baseline = (
    train
    .groupby(
        [
            "site_id",
            "day_of_week",
            "hour_of_day"
        ]
    )
    .agg(
        predicted_arrivals=(
            "arrivals",
            "mean"
        ),

        predicted_energy_kwh=(
            "delivered_energy_kwh",
            "mean"
        )
    )
    .reset_index()
)


# ---------------------------------------------------------
# APPLY FORECAST TO TEST PERIOD
# ---------------------------------------------------------

forecast = test.merge(
    baseline,
    on=[
        "site_id",
        "day_of_week",
        "hour_of_day"
    ],
    how="left"
)


# Safety fallback
forecast["predicted_arrivals"] = (
    forecast["predicted_arrivals"]
    .fillna(
        train["arrivals"].mean()
    )
)

forecast["predicted_energy_kwh"] = (
    forecast["predicted_energy_kwh"]
    .fillna(
        train["delivered_energy_kwh"].mean()
    )
)


# ---------------------------------------------------------
# FORECAST ERRORS
# ---------------------------------------------------------

forecast["arrival_error"] = (
    forecast["arrivals"]
    - forecast["predicted_arrivals"]
)

forecast["energy_error_kwh"] = (
    forecast["delivered_energy_kwh"]
    - forecast["predicted_energy_kwh"]
)


# ---------------------------------------------------------
# EVALUATION PER SITE
# ---------------------------------------------------------

results = []

for site_id, site_data in forecast.groupby(
    "site_id"
):

    arrival_mae = mean_absolute_error(
        site_data["arrivals"],
        site_data["predicted_arrivals"]
    )

    arrival_rmse = root_mean_squared_error(
        site_data["arrivals"],
        site_data["predicted_arrivals"]
    )

    energy_mae = mean_absolute_error(
        site_data["delivered_energy_kwh"],
        site_data["predicted_energy_kwh"]
    )

    energy_rmse = root_mean_squared_error(
        site_data["delivered_energy_kwh"],
        site_data["predicted_energy_kwh"]
    )


    # WAPE avoids some of the problems MAPE has
    # when actual demand is zero.
    arrival_wape = (
        np.abs(
            site_data["arrival_error"]
        ).sum()
        /
        site_data["arrivals"].sum()
        * 100
    )

    energy_wape = (
        np.abs(
            site_data["energy_error_kwh"]
        ).sum()
        /
        site_data[
            "delivered_energy_kwh"
        ].sum()
        * 100
    )


    results.append(
        {
            "site_id":
                site_id,

            "arrival_mae":
                round(
                    arrival_mae,
                    2
                ),

            "arrival_rmse":
                round(
                    arrival_rmse,
                    2
                ),

            "arrival_wape_pct":
                round(
                    arrival_wape,
                    2
                ),

            "energy_mae_kwh":
                round(
                    energy_mae,
                    2
                ),

            "energy_rmse_kwh":
                round(
                    energy_rmse,
                    2
                ),

            "energy_wape_pct":
                round(
                    energy_wape,
                    2
                )
        }
    )


metrics = pd.DataFrame(
    results
)


# ---------------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------------

forecast.to_csv(
    OUTPUT_FILE,
    index=False
)

metrics.to_csv(
    METRICS_FILE,
    index=False
)


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print(
    "\n--- BASELINE FORECAST PERFORMANCE ---"
)

print(
    metrics.to_string(
        index=False
    )
)


print(
    "\nSaved forecast:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSaved metrics:"
)

print(
    METRICS_FILE
)