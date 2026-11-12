from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ---------------------------------------------------------
# LOAD MODEL RESULTS
# ---------------------------------------------------------

baseline = pd.read_csv(
    DATA_DIR
    / "baseline_forecast_metrics.csv"
)

linear = pd.read_csv(
    DATA_DIR
    / "linear_forecast_metrics.csv"
)

lightgbm = pd.read_csv(
    DATA_DIR
    / "gradient_boosting_forecast_metrics.csv"
)

weather = pd.read_csv(
    DATA_DIR
    / "weather_holiday_forecast_metrics.csv"
)


# ---------------------------------------------------------
# KEEP WAPE ONLY
# ---------------------------------------------------------

baseline = baseline[
    [
        "site_id",
        "arrival_wape_pct",
        "energy_wape_pct"
    ]
].rename(
    columns={
        "arrival_wape_pct":
            "baseline_arrival_wape",

        "energy_wape_pct":
            "baseline_energy_wape"
    }
)


linear = linear[
    [
        "site_id",
        "arrival_wape_pct",
        "energy_wape_pct"
    ]
].rename(
    columns={
        "arrival_wape_pct":
            "linear_arrival_wape",

        "energy_wape_pct":
            "linear_energy_wape"
    }
)


lightgbm = lightgbm[
    [
        "site_id",
        "arrival_wape_pct",
        "energy_wape_pct"
    ]
].rename(
    columns={
        "arrival_wape_pct":
            "lightgbm_arrival_wape",

        "energy_wape_pct":
            "lightgbm_energy_wape"
    }
)


weather = weather[
    [
        "site_id",
        "arrival_wape_pct",
        "energy_wape_pct"
    ]
].rename(
    columns={
        "arrival_wape_pct":
            "weather_arrival_wape",

        "energy_wape_pct":
            "weather_energy_wape"
    }
)


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

comparison = (
    baseline
    .merge(
        linear,
        on="site_id"
    )
    .merge(
        lightgbm,
        on="site_id"
    )
    .merge(
        weather,
        on="site_id"
    )
)


# ---------------------------------------------------------
# FIND BEST ARRIVAL MODEL
# ---------------------------------------------------------

arrival_columns = {
    "baseline_arrival_wape":
        "BASELINE",

    "linear_arrival_wape":
        "LINEAR",

    "lightgbm_arrival_wape":
        "LIGHTGBM",

    "weather_arrival_wape":
        "LIGHTGBM + WEATHER/HOLIDAY"
}


comparison[
    "best_arrival_model"
] = (
    comparison[
        list(
            arrival_columns.keys()
        )
    ]
    .idxmin(
        axis=1
    )
    .map(
        arrival_columns
    )
)


comparison[
    "best_arrival_wape"
] = (
    comparison[
        list(
            arrival_columns.keys()
        )
    ]
    .min(
        axis=1
    )
)


# ---------------------------------------------------------
# FIND BEST ENERGY MODEL
# ---------------------------------------------------------

energy_columns = {
    "baseline_energy_wape":
        "BASELINE",

    "linear_energy_wape":
        "LINEAR",

    "lightgbm_energy_wape":
        "LIGHTGBM",

    "weather_energy_wape":
        "LIGHTGBM + WEATHER/HOLIDAY"
}


comparison[
    "best_energy_model"
] = (
    comparison[
        list(
            energy_columns.keys()
        )
    ]
    .idxmin(
        axis=1
    )
    .map(
        energy_columns
    )
)


comparison[
    "best_energy_wape"
] = (
    comparison[
        list(
            energy_columns.keys()
        )
    ]
    .min(
        axis=1
    )
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

OUTPUT_FILE = (
    DATA_DIR
    / "final_model_comparison.csv"
)


comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print(
    "\n--- FINAL MODEL COMPARISON ---"
)


print(
    comparison[
        [
            "site_id",

            "baseline_arrival_wape",
            "linear_arrival_wape",
            "lightgbm_arrival_wape",
            "weather_arrival_wape",

            "best_arrival_model",
            "best_arrival_wape",

            "baseline_energy_wape",
            "linear_energy_wape",
            "lightgbm_energy_wape",
            "weather_energy_wape",

            "best_energy_model",
            "best_energy_wape"
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)