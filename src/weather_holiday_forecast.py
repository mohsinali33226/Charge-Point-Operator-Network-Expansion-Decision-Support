from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)

from lightgbm import LGBMRegressor


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_holiday_forecast_results.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_holiday_forecast_metrics.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_csv(
    INPUT_FILE
)

data["hour"] = pd.to_datetime(
    data["hour"]
)

data = (
    data
    .sort_values(
        [
            "site_id",
            "hour"
        ]
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# TIME FEATURES
# =========================================================

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


# =========================================================
# LAG FEATURES
#
# For day-ahead forecasting, we only use demand
# information from at least 24 hours ago.
# =========================================================

lags = [
    24,
    48,
    72,
    96,
    120,
    144,
    168
]


for lag in lags:

    data[
        f"arrivals_lag_{lag}"
    ] = (
        data
        .groupby(
            "site_id"
        )["arrivals"]
        .shift(
            lag
        )
    )

    data[
        f"energy_lag_{lag}"
    ] = (
        data
        .groupby(
            "site_id"
        )["delivered_energy_kwh"]
        .shift(
            lag
        )
    )


# =========================================================
# PREVIOUS-WEEK FEATURES
# =========================================================

arrival_lag_columns = [
    f"arrivals_lag_{lag}"
    for lag in lags
]

energy_lag_columns = [
    f"energy_lag_{lag}"
    for lag in lags
]


data[
    "arrivals_previous_week_mean"
] = (
    data[
        arrival_lag_columns
    ]
    .mean(
        axis=1
    )
)


data[
    "arrivals_previous_week_std"
] = (
    data[
        arrival_lag_columns
    ]
    .std(
        axis=1
    )
)


data[
    "energy_previous_week_mean"
] = (
    data[
        energy_lag_columns
    ]
    .mean(
        axis=1
    )
)


data[
    "energy_previous_week_std"
] = (
    data[
        energy_lag_columns
    ]
    .std(
        axis=1
    )
)


# =========================================================
# RECENT CHANGE FEATURES
# =========================================================

data[
    "arrival_change_vs_last_week"
] = (
    data[
        "arrivals_lag_24"
    ]
    -
    data[
        "arrivals_lag_168"
    ]
)


data[
    "energy_change_vs_last_week"
] = (
    data[
        "energy_lag_24"
    ]
    -
    data[
        "energy_lag_168"
    ]
)


# =========================================================
# REMOVE ONLY ROWS WITHOUT ENOUGH HISTORY
#
# IMPORTANT:
# We do NOT use data.dropna() on the whole dataset.
# That was the reason our training period became too short.
# =========================================================

required_history_columns = (
    arrival_lag_columns
    + energy_lag_columns
)


data = (
    data
    .dropna(
        subset=
            required_history_columns
    )
    .copy()
)


# =========================================================
# CHECK IMPORTANT FEATURES FOR MISSING VALUES
# =========================================================

important_features = [
    "temperature_c",
    "precipitation_mm",
    "is_public_holiday",
    "day_before_holiday",
    "day_after_holiday"
]


print(
    "\n--- FEATURE MISSING VALUE CHECK ---"
)

missing_features = (
    data[
        important_features
    ]
    .isna()
    .sum()
)

print(
    missing_features
)


if missing_features.sum() > 0:

    raise ValueError(
        "Some weather or holiday features still contain missing values."
    )


# =========================================================
# TRAIN / TEST SPLIT
#
# January-May = training
# June = unseen test period
# =========================================================

split_date = pd.Timestamp(
    "2025-06-01"
)


train = data[
    data["hour"]
    < split_date
].copy()


test = data[
    data["hour"]
    >= split_date
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

print(
    "Training rows:",
    len(train)
)

print(
    "Testing rows:",
    len(test)
)


# =========================================================
# FEATURES
# =========================================================

categorical_features = [
    "site_id",
    "hour_of_day",
    "day_of_week",
    "month"
]


numeric_features = [

    # -----------------------------------------------------
    # CALENDAR
    # -----------------------------------------------------

    "is_weekend",
    "is_public_holiday",
    "day_before_holiday",
    "day_after_holiday",

    # -----------------------------------------------------
    # WEATHER
    # -----------------------------------------------------

    "temperature_c",
    "precipitation_mm",
    "is_raining",
    "is_cold",
    "is_hot",

    # -----------------------------------------------------
    # ARRIVAL HISTORY
    # -----------------------------------------------------

    "arrivals_lag_24",
    "arrivals_lag_48",
    "arrivals_lag_72",
    "arrivals_lag_96",
    "arrivals_lag_120",
    "arrivals_lag_144",
    "arrivals_lag_168",

    "arrivals_previous_week_mean",
    "arrivals_previous_week_std",
    "arrival_change_vs_last_week",

    # -----------------------------------------------------
    # ENERGY HISTORY
    # -----------------------------------------------------

    "energy_lag_24",
    "energy_lag_48",
    "energy_lag_72",
    "energy_lag_96",
    "energy_lag_120",
    "energy_lag_144",
    "energy_lag_168",

    "energy_previous_week_mean",
    "energy_previous_week_std",
    "energy_change_vs_last_week",

    # -----------------------------------------------------
    # SITE INFORMATION
    # -----------------------------------------------------

    "connector_count",
    "charger_power_kw",
    "grid_capacity_kw"
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


# =========================================================
# PREPROCESSOR
# =========================================================

def create_preprocessor():

    return ColumnTransformer(
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


# =========================================================
# ARRIVAL MODEL
# =========================================================

arrival_model = Pipeline(
    steps=[

        (
            "preprocessor",
            create_preprocessor()
        ),

        (
            "model",
            LGBMRegressor(
                objective="poisson",
                n_estimators=300,
                learning_rate=0.04,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            )
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


# Charging demand cannot be negative
arrival_predictions = np.clip(
    arrival_predictions,
    0,
    None
)


# =========================================================
# ENERGY MODEL
# =========================================================

energy_model = Pipeline(
    steps=[

        (
            "preprocessor",
            create_preprocessor()
        ),

        (
            "model",
            LGBMRegressor(
                objective="regression",
                n_estimators=300,
                learning_rate=0.04,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            )
        )
    ]
)


energy_model.fit(
    X_train,
    train[
        "delivered_energy_kwh"
    ]
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


# =========================================================
# STORE FORECASTS
# =========================================================

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
    -
    forecast["predicted_arrivals"]
)


forecast[
    "energy_error_kwh"
] = (
    forecast["delivered_energy_kwh"]
    -
    forecast["predicted_energy_kwh"]
)


# =========================================================
# CALCULATE METRICS PER SITE
# =========================================================

results = []


for site_id, site_data in forecast.groupby(
    "site_id"
):

    # -----------------------------------------------------
    # ARRIVAL ERROR
    # -----------------------------------------------------

    arrival_mae = mean_absolute_error(
        site_data[
            "arrivals"
        ],
        site_data[
            "predicted_arrivals"
        ]
    )


    arrival_rmse = root_mean_squared_error(
        site_data[
            "arrivals"
        ],
        site_data[
            "predicted_arrivals"
        ]
    )


    actual_arrivals_total = (
        site_data[
            "arrivals"
        ]
        .sum()
    )


    if actual_arrivals_total > 0:

        arrival_wape = (
            np.abs(
                site_data[
                    "arrival_error"
                ]
            )
            .sum()
            /
            actual_arrivals_total
            * 100
        )

    else:

        arrival_wape = np.nan


    # -----------------------------------------------------
    # ENERGY ERROR
    # -----------------------------------------------------

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


    actual_energy_total = (
        site_data[
            "delivered_energy_kwh"
        ]
        .sum()
    )


    if actual_energy_total > 0:

        energy_wape = (
            np.abs(
                site_data[
                    "energy_error_kwh"
                ]
            )
            .sum()
            /
            actual_energy_total
            * 100
        )

    else:

        energy_wape = np.nan


    # -----------------------------------------------------
    # STORE SITE RESULTS
    # -----------------------------------------------------

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


# =========================================================
# SAVE RESULTS
# =========================================================

forecast.to_csv(
    OUTPUT_FILE,
    index=False
)


metrics.to_csv(
    METRICS_FILE,
    index=False
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print(
    "\n--- WEATHER + HOLIDAY LIGHTGBM PERFORMANCE ---"
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