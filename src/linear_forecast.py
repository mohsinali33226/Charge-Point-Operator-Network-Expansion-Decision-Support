from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
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
    / "linear_forecast_results.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "linear_forecast_metrics.csv"
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
).reset_index(drop=True)


# ---------------------------------------------------------
# TIME FEATURES
# ---------------------------------------------------------

data["hour_of_day"] = (
    data["hour"].dt.hour
)

data["day_of_week"] = (
    data["hour"].dt.dayofweek
)

data["month"] = (
    data["hour"].dt.month
)

data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)


# ---------------------------------------------------------
# LAG FEATURES
#
# These only use historical information.
# ---------------------------------------------------------

for lag in [
    24,
    48,
    72,
    96,
    120,
    144,
    168
]:

    data[
        f"arrivals_lag_{lag}"
    ] = (
        data
        .groupby("site_id")["arrivals"]
        .shift(lag)
    )

    data[
        f"energy_lag_{lag}"
    ] = (
        data
        .groupby("site_id")["delivered_energy_kwh"]
        .shift(lag)
    )


# ---------------------------------------------------------
# PREVIOUS-WEEK SAME-HOUR AVERAGES
# ---------------------------------------------------------

arrival_lag_columns = [
    f"arrivals_lag_{lag}"
    for lag in [
        24,
        48,
        72,
        96,
        120,
        144,
        168
    ]
]

energy_lag_columns = [
    f"energy_lag_{lag}"
    for lag in [
        24,
        48,
        72,
        96,
        120,
        144,
        168
    ]
]


data[
    "arrivals_previous_week_mean"
] = (
    data[
        arrival_lag_columns
    ]
    .mean(axis=1)
)


data[
    "energy_previous_week_mean"
] = (
    data[
        energy_lag_columns
    ]
    .mean(axis=1)
)


# ---------------------------------------------------------
# REMOVE ROWS WITHOUT FULL HISTORY
# ---------------------------------------------------------

data = data.dropna().copy()


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
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
# FEATURES
# ---------------------------------------------------------

categorical_features = [
    "site_id",
    "hour_of_day",
    "day_of_week",
    "month",
]

numeric_features = [
    "is_weekend",

    "arrivals_lag_24",
    "arrivals_lag_168",
    "arrivals_previous_week_mean",

    "energy_lag_24",
    "energy_lag_168",
    "energy_previous_week_mean",

    "connector_count",
    "charger_power_kw",
    "grid_capacity_kw",
]


features = (
    categorical_features
    + numeric_features
)


X_train = train[
    features
]

X_test = test[
    features
]


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ---------------------------------------------------------
# ARRIVAL MODEL
# ---------------------------------------------------------

arrival_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


arrival_model.fit(
    X_train,
    train["arrivals"]
)


arrival_predictions = (
    arrival_model.predict(
        X_test
    )
)


# Demand cannot be negative
arrival_predictions = np.clip(
    arrival_predictions,
    0,
    None
)


# ---------------------------------------------------------
# ENERGY MODEL
# ---------------------------------------------------------

energy_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


energy_model.fit(
    X_train,
    train["delivered_energy_kwh"]
)


energy_predictions = (
    energy_model.predict(
        X_test
    )
)


energy_predictions = np.clip(
    energy_predictions,
    0,
    None
)


# ---------------------------------------------------------
# STORE FORECASTS
# ---------------------------------------------------------

forecast = test.copy()

forecast[
    "predicted_arrivals"
] = arrival_predictions

forecast[
    "predicted_energy_kwh"
] = energy_predictions


forecast[
    "arrival_error"
] = (
    forecast["arrivals"]
    - forecast["predicted_arrivals"]
)

forecast[
    "energy_error_kwh"
] = (
    forecast["delivered_energy_kwh"]
    - forecast["predicted_energy_kwh"]
)


# ---------------------------------------------------------
# EVALUATE EACH SITE
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

    arrival_wape = (
        np.abs(
            site_data[
                "arrival_error"
            ]
        ).sum()
        /
        site_data[
            "arrivals"
        ].sum()
        * 100
    )


    energy_mae = mean_absolute_error(
        site_data[
            "delivered_energy_kwh"
        ],
        site_data[
            "predicted_energy_kwh"
        ]
    )

    energy_rmse = root_mean_squared_error(
        site_data[
            "delivered_energy_kwh"
        ],
        site_data[
            "predicted_energy_kwh"
        ]
    )

    energy_wape = (
        np.abs(
            site_data[
                "energy_error_kwh"
            ]
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
# SAVE
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
    "\n--- LINEAR MODEL PERFORMANCE ---"
)

print(
    metrics.to_string(
        index=False
    )
)

print(
    "\nSaved forecasts:"
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