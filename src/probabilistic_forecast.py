from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_pinball_loss

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
    / "probabilistic_forecast_results.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "probabilistic_forecast_summary.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_csv(INPUT_FILE)

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
    .reset_index(drop=True)
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
        .groupby("site_id")["arrivals"]
        .shift(lag)
    )

    data[
        f"energy_lag_{lag}"
    ] = (
        data
        .groupby("site_id")[
            "delivered_energy_kwh"
        ]
        .shift(lag)
    )


arrival_lag_columns = [
    f"arrivals_lag_{lag}"
    for lag in lags
]

energy_lag_columns = [
    f"energy_lag_{lag}"
    for lag in lags
]


# =========================================================
# WEEK HISTORY FEATURES
# =========================================================

data[
    "arrivals_previous_week_mean"
] = (
    data[
        arrival_lag_columns
    ]
    .mean(axis=1)
)


data[
    "arrivals_previous_week_std"
] = (
    data[
        arrival_lag_columns
    ]
    .std(axis=1)
)


data[
    "energy_previous_week_mean"
] = (
    data[
        energy_lag_columns
    ]
    .mean(axis=1)
)


data[
    "energy_previous_week_std"
] = (
    data[
        energy_lag_columns
    ]
    .std(axis=1)
)


data[
    "arrival_change_vs_last_week"
] = (
    data["arrivals_lag_24"]
    -
    data["arrivals_lag_168"]
)


data[
    "energy_change_vs_last_week"
] = (
    data["energy_lag_24"]
    -
    data["energy_lag_168"]
)


# =========================================================
# REMOVE FIRST WEEK
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
# TRAIN / TEST SPLIT
# =========================================================

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

    # Calendar
    "is_weekend",
    "is_public_holiday",
    "day_before_holiday",
    "day_after_holiday",

    # Weather
    "temperature_c",
    "precipitation_mm",
    "is_raining",
    "is_cold",
    "is_hot",

    # Arrival history
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

    # Energy history
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

    # Site
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
# CREATE QUANTILE MODEL
# =========================================================

def create_quantile_model(alpha):

    return Pipeline(
        steps=[

            (
                "preprocessor",
                create_preprocessor()
            ),

            (
                "model",
                LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    n_estimators=350,
                    learning_rate=0.04,
                    num_leaves=31,
                    random_state=42,
                    verbosity=-1
                )
            )
        ]
    )


# =========================================================
# ARRIVAL P50
# =========================================================

arrival_p50_model = (
    create_quantile_model(
        0.50
    )
)


arrival_p50_model.fit(
    X_train,
    train["arrivals"]
)


arrival_p50 = (
    arrival_p50_model.predict(
        X_test
    )
)


# =========================================================
# ARRIVAL P95
# =========================================================

arrival_p95_model = (
    create_quantile_model(
        0.95
    )
)


arrival_p95_model.fit(
    X_train,
    train["arrivals"]
)


arrival_p95 = (
    arrival_p95_model.predict(
        X_test
    )
)


# =========================================================
# ENERGY P50
# =========================================================

energy_p50_model = (
    create_quantile_model(
        0.50
    )
)


energy_p50_model.fit(
    X_train,
    train[
        "delivered_energy_kwh"
    ]
)


energy_p50 = (
    energy_p50_model.predict(
        X_test
    )
)


# =========================================================
# ENERGY P95
# =========================================================

energy_p95_model = (
    create_quantile_model(
        0.95
    )
)


energy_p95_model.fit(
    X_train,
    train[
        "delivered_energy_kwh"
    ]
)


energy_p95 = (
    energy_p95_model.predict(
        X_test
    )
)


# =========================================================
# CLEAN PREDICTIONS
# =========================================================

arrival_p50 = np.clip(
    arrival_p50,
    0,
    None
)

arrival_p95 = np.clip(
    arrival_p95,
    0,
    None
)

energy_p50 = np.clip(
    energy_p50,
    0,
    None
)

energy_p95 = np.clip(
    energy_p95,
    0,
    None
)


# P95 should never be below P50
arrival_p95 = np.maximum(
    arrival_p95,
    arrival_p50
)

energy_p95 = np.maximum(
    energy_p95,
    energy_p50
)


# =========================================================
# STORE RESULTS
# =========================================================

forecast = test.copy()


forecast[
    "arrival_p50"
] = arrival_p50


forecast[
    "arrival_p95"
] = arrival_p95


forecast[
    "energy_p50_kwh"
] = energy_p50


forecast[
    "energy_p95_kwh"
] = energy_p95


# =========================================================
# P95 COVERAGE
#
# Example:
# If coverage is 94%, actual demand was below
# our P95 forecast in 94% of hours.
# =========================================================

forecast[
    "arrival_below_p95"
] = (
    forecast["arrivals"]
    <=
    forecast["arrival_p95"]
).astype(int)


forecast[
    "energy_below_p95"
] = (
    forecast[
        "delivered_energy_kwh"
    ]
    <=
    forecast[
        "energy_p95_kwh"
    ]
).astype(int)


# =========================================================
# SITE SUMMARY
# =========================================================

summary_rows = []


for site_id, site_data in forecast.groupby(
    "site_id"
):

    arrival_coverage = (
        site_data[
            "arrival_below_p95"
        ]
        .mean()
        * 100
    )


    energy_coverage = (
        site_data[
            "energy_below_p95"
        ]
        .mean()
        * 100
    )


    arrival_band = (
        site_data[
            "arrival_p95"
        ]
        -
        site_data[
            "arrival_p50"
        ]
    ).mean()


    energy_band = (
        site_data[
            "energy_p95_kwh"
        ]
        -
        site_data[
            "energy_p50_kwh"
        ]
    ).mean()


    arrival_p50_loss = mean_pinball_loss(
        site_data[
            "arrivals"
        ],
        site_data[
            "arrival_p50"
        ],
        alpha=0.50
    )


    arrival_p95_loss = mean_pinball_loss(
        site_data[
            "arrivals"
        ],
        site_data[
            "arrival_p95"
        ],
        alpha=0.95
    )


    energy_p50_loss = mean_pinball_loss(
        site_data[
            "delivered_energy_kwh"
        ],
        site_data[
            "energy_p50_kwh"
        ],
        alpha=0.50
    )


    energy_p95_loss = mean_pinball_loss(
        site_data[
            "delivered_energy_kwh"
        ],
        site_data[
            "energy_p95_kwh"
        ],
        alpha=0.95
    )


    summary_rows.append(
        {
            "site_id":
                site_id,

            "arrival_p95_coverage_pct":
                round(
                    arrival_coverage,
                    2
                ),

            "energy_p95_coverage_pct":
                round(
                    energy_coverage,
                    2
                ),

            "average_arrival_p50_p95_gap":
                round(
                    arrival_band,
                    2
                ),

            "average_energy_p50_p95_gap_kwh":
                round(
                    energy_band,
                    2
                ),

            "arrival_p50_pinball_loss":
                round(
                    arrival_p50_loss,
                    3
                ),

            "arrival_p95_pinball_loss":
                round(
                    arrival_p95_loss,
                    3
                ),

            "energy_p50_pinball_loss":
                round(
                    energy_p50_loss,
                    2
                ),

            "energy_p95_pinball_loss":
                round(
                    energy_p95_loss,
                    2
                )
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# =========================================================
# SAVE
# =========================================================

forecast.to_csv(
    OUTPUT_FILE,
    index=False
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# =========================================================
# DISPLAY
# =========================================================

print(
    "\n--- P50 / P95 FORECAST SUMMARY ---"
)


print(
    summary.to_string(
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
    "\nSaved summary:"
)

print(
    SUMMARY_FILE
)