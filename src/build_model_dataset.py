from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEMAND_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_hourly_demand_with_holidays.csv"
)

WEATHER_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dwd_hourly_weather.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

demand = pd.read_csv(
    DEMAND_FILE
)

weather = pd.read_csv(
    WEATHER_FILE
)


# ---------------------------------------------------------
# CONVERT TIME
# ---------------------------------------------------------

demand["hour"] = pd.to_datetime(
    demand["hour"]
)

weather["hour"] = pd.to_datetime(
    weather["hour"]
)


# ---------------------------------------------------------
# CHECK FOR DUPLICATE SITE-HOURS
# ---------------------------------------------------------

demand_duplicates = (
    demand[
        [
            "site_id",
            "hour"
        ]
    ]
    .duplicated()
    .sum()
)

weather_duplicates = (
    weather[
        [
            "site_id",
            "hour"
        ]
    ]
    .duplicated()
    .sum()
)


print(
    "\n--- DUPLICATE CHECK ---"
)

print(
    "Demand duplicates:",
    demand_duplicates
)

print(
    "Weather duplicates:",
    weather_duplicates
)


if demand_duplicates > 0:

    raise ValueError(
        "Demand data contains duplicate site-hours."
    )


if weather_duplicates > 0:

    raise ValueError(
        "Weather data contains duplicate site-hours."
    )


# ---------------------------------------------------------
# KEEP ONLY WEATHER COLUMNS WE NEED
# ---------------------------------------------------------

weather = weather[
    [
        "site_id",
        "hour",
        "temperature_c",
        "precipitation_mm"
    ]
]


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

model_data = demand.merge(
    weather,
    on=[
        "site_id",
        "hour"
    ],
    how="left",
    validate="one_to_one"
)


# ---------------------------------------------------------
# ADD SIMPLE WEATHER FEATURES
# ---------------------------------------------------------

model_data[
    "is_raining"
] = (
    model_data[
        "precipitation_mm"
    ] > 0
).astype(int)


model_data[
    "is_cold"
] = (
    model_data[
        "temperature_c"
    ] < 5
).astype(int)


model_data[
    "is_hot"
] = (
    model_data[
        "temperature_c"
    ] > 25
).astype(int)


# ---------------------------------------------------------
# FINAL QA
# ---------------------------------------------------------

missing_temperature = (
    model_data[
        "temperature_c"
    ]
    .isna()
    .sum()
)

missing_precipitation = (
    model_data[
        "precipitation_mm"
    ]
    .isna()
    .sum()
)

missing_holiday = (
    model_data[
        "is_public_holiday"
    ]
    .isna()
    .sum()
)


print(
    "\n--- MODEL DATASET QA ---"
)

print(
    "Rows:",
    len(model_data)
)

print(
    "Sites:",
    model_data[
        "site_id"
    ].nunique()
)

print(
    "From:",
    model_data[
        "hour"
    ].min()
)

print(
    "To:",
    model_data[
        "hour"
    ].max()
)

print(
    "Missing temperature:",
    missing_temperature
)

print(
    "Missing precipitation:",
    missing_precipitation
)

print(
    "Missing holiday values:",
    missing_holiday
)


if len(model_data) != 21720:

    raise ValueError(
        "Model dataset row count is wrong."
    )


if missing_temperature > 0:

    raise ValueError(
        "Temperature is missing."
    )


if missing_precipitation > 0:

    raise ValueError(
        "Precipitation is missing."
    )


if missing_holiday > 0:

    raise ValueError(
        "Holiday information is missing."
    )


# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

model_data = (
    model_data
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


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

model_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# DISPLAY SAMPLE
# ---------------------------------------------------------

print(
    "\n--- SAMPLE MODEL DATA ---"
)

print(
    model_data[
        [
            "site_id",
            "city",
            "hour",
            "arrivals",
            "delivered_energy_kwh",
            "temperature_c",
            "precipitation_mm",
            "is_public_holiday",
            "is_raining"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)