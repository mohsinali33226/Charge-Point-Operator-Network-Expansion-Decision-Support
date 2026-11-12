from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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
    / "calibrated_p95_forecast_results.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "calibrated_p95_summary.csv"
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
# HISTORY FEATURES
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

required_history = (
    arrival_lag_columns
    + energy_lag_columns
)

data = (
    data
    .dropna(
        subset=required_history
    )
    .copy()
)


# =========================================================
# TRAIN / CALIBRATION / TEST
# =========================================================

calibration_start = pd.Timestamp(
    "2025-05-01"
)

test_start = pd.Timestamp(
    "2025-06-01"
)


train = data[
    data["hour"] < calibration_start
].copy()

calibration = data[
    (
        data["hour"] >= calibration_start
    )
    &
    (
        data["hour"] < test_start
    )
].copy()

test = data[
    data["hour"] >= test_start
].copy()


print(
    "\n--- DATA SPLIT ---"
)

print(
    "Training:",
    train["hour"].min(),
    "to",
    train["hour"].max()
)

print(
    "Calibration:",
    calibration["hour"].min(),
    "to",
    calibration["hour"].max()
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

    "is_weekend",
    "is_public_holiday",
    "day_before_holiday",
    "day_after_holiday",

    "temperature_c",
    "precipitation_mm",
    "is_raining",
    "is_cold",
    "is_hot",

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

    "connector_count",
    "charger_power_kw",
    "grid_capacity_kw"
]


features = (
    categorical_features
    + numeric_features
)


# =========================================================
# PREPROCESSOR
# =========================================================

def make_preprocessor():

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
# P95 MODEL
# =========================================================

def make_p95_model():

    return Pipeline(
        steps=[

            (
                "preprocessor",
                make_preprocessor()
            ),

            (
                "model",
                LGBMRegressor(
                    objective="quantile",
                    alpha=0.95,
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
# ARRIVAL P95
# =========================================================

arrival_model = make_p95_model()

arrival_model.fit(
    train[features],
    train["arrivals"]
)


calibration[
    "raw_arrival_p95"
] = np.clip(
    arrival_model.predict(
        calibration[features]
    ),
    0,
    None
)


test[
    "raw_arrival_p95"
] = np.clip(
    arrival_model.predict(
        test[features]
    ),
    0,
    None
)


# =========================================================
# ENERGY P95
# =========================================================

energy_model = make_p95_model()

energy_model.fit(
    train[features],
    train[
        "delivered_energy_kwh"
    ]
)


calibration[
    "raw_energy_p95_kwh"
] = np.clip(
    energy_model.predict(
        calibration[features]
    ),
    0,
    None
)


test[
    "raw_energy_p95_kwh"
] = np.clip(
    energy_model.predict(
        test[features]
    ),
    0,
    None
)


# =========================================================
# CALIBRATION
#
# Find how much each site's P95 needs to move upward
# so approximately 95% of May demand stays below it.
# =========================================================

corrections = []


for site_id, site_data in calibration.groupby(
    "site_id"
):

    arrival_residual = (
        site_data["arrivals"]
        -
        site_data["raw_arrival_p95"]
    )


    energy_residual = (
        site_data["delivered_energy_kwh"]
        -
        site_data["raw_energy_p95_kwh"]
    )


    arrival_correction = np.quantile(
        arrival_residual,
        0.95
    )


    energy_correction = np.quantile(
        energy_residual,
        0.95
    )


    # We only increase P95 here.
    # We do not reduce the safety level.
    arrival_correction = max(
        arrival_correction,
        0
    )

    energy_correction = max(
        energy_correction,
        0
    )


    corrections.append(
        {
            "site_id":
                site_id,

            "arrival_p95_correction":
                arrival_correction,

            "energy_p95_correction_kwh":
                energy_correction
        }
    )


corrections = pd.DataFrame(
    corrections
)


# =========================================================
# APPLY CORRECTION TO JUNE
# =========================================================

test = test.merge(
    corrections,
    on="site_id",
    how="left"
)


test[
    "calibrated_arrival_p95"
] = (
    test["raw_arrival_p95"]
    +
    test["arrival_p95_correction"]
)


test[
    "calibrated_energy_p95_kwh"
] = (
    test["raw_energy_p95_kwh"]
    +
    test["energy_p95_correction_kwh"]
)


# =========================================================
# COVERAGE CHECK
# =========================================================

test[
    "raw_arrival_below_p95"
] = (
    test["arrivals"]
    <=
    test["raw_arrival_p95"]
).astype(int)


test[
    "calibrated_arrival_below_p95"
] = (
    test["arrivals"]
    <=
    test["calibrated_arrival_p95"]
).astype(int)


test[
    "raw_energy_below_p95"
] = (
    test["delivered_energy_kwh"]
    <=
    test["raw_energy_p95_kwh"]
).astype(int)


test[
    "calibrated_energy_below_p95"
] = (
    test["delivered_energy_kwh"]
    <=
    test["calibrated_energy_p95_kwh"]
).astype(int)


# =========================================================
# SUMMARY
# =========================================================

summary_rows = []


for site_id, site_data in test.groupby(
    "site_id"
):

    summary_rows.append(
        {
            "site_id":
                site_id,

            "raw_arrival_coverage_pct":
                round(
                    site_data[
                        "raw_arrival_below_p95"
                    ].mean()
                    * 100,
                    2
                ),

            "calibrated_arrival_coverage_pct":
                round(
                    site_data[
                        "calibrated_arrival_below_p95"
                    ].mean()
                    * 100,
                    2
                ),

            "arrival_p95_correction":
                round(
                    site_data[
                        "arrival_p95_correction"
                    ].iloc[0],
                    2
                ),

            "raw_energy_coverage_pct":
                round(
                    site_data[
                        "raw_energy_below_p95"
                    ].mean()
                    * 100,
                    2
                ),

            "calibrated_energy_coverage_pct":
                round(
                    site_data[
                        "calibrated_energy_below_p95"
                    ].mean()
                    * 100,
                    2
                ),

            "energy_p95_correction_kwh":
                round(
                    site_data[
                        "energy_p95_correction_kwh"
                    ].iloc[0],
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

test.to_csv(
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
    "\n--- CALIBRATED P95 SUMMARY ---"
)

print(
    summary.to_string(
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
    "\nSaved summary:"
)

print(
    SUMMARY_FILE
)