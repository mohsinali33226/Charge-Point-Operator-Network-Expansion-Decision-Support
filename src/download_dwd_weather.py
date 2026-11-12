from pathlib import Path
import pandas as pd

from wetterdienst.provider.dwd.observation import (
    DwdObservationRequest
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SITE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "site_master.csv"
)

WEATHER_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dwd_hourly_weather.csv"
)

STATION_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dwd_station_mapping.csv"
)


# =========================================================
# LOAD SITES
# =========================================================

sites = pd.read_csv(SITE_FILE)


# =========================================================
# PROJECT PERIOD
# =========================================================

TARGET_START = pd.Timestamp(
    "2025-01-01 00:00:00"
)

TARGET_END = pd.Timestamp(
    "2025-06-30 23:00:00"
)

TARGET_HOURS = pd.date_range(
    TARGET_START,
    TARGET_END,
    freq="h"
)


# Slightly wider range because DWD timestamps are UTC
REQUEST_START = "2024-12-31"
REQUEST_END = "2025-07-02"


# =========================================================
# SETTINGS
# =========================================================

SEARCH_RADIUS_KM = 100
MAX_CANDIDATES = 8


# =========================================================
# OUTPUT STORAGE
# =========================================================

all_weather = []
station_records = []


# =========================================================
# TIME COLUMN
# Wetterdienst 0.139 uses "timestamp".
# =========================================================

def detect_time_column(df):

    if "timestamp" in df.columns:
        return "timestamp"

    if "date" in df.columns:
        return "date"

    raise ValueError(
        f"No time column found. Columns: {df.columns.tolist()}"
    )


# =========================================================
# GET DATA FROM ONE STATION
# =========================================================

def get_station_values(
    request,
    station_id
):

    values = (
        request
        .filter_by_station_id(
            station_id=str(station_id)
        )
        .values
        .all()
        .df
        .to_pandas()
    )

    if values.empty:

        return pd.DataFrame(
            columns=[
                "hour",
                "value"
            ]
        )


    time_column = detect_time_column(
        values
    )


    values["hour"] = (
        pd.to_datetime(
            values[time_column],
            utc=True
        )
        .dt
        .tz_convert(
            "Europe/Berlin"
        )
        .dt
        .tz_localize(None)
        .dt
        .floor("h")
    )


    values["value"] = pd.to_numeric(
        values["value"],
        errors="coerce"
    )


    values = values[
        (
            values["hour"] >= TARGET_START
        )
        &
        (
            values["hour"] <= TARGET_END
        )
    ].copy()


    # Protect against duplicate local timestamps
    values = (
        values
        .groupby(
            "hour",
            as_index=False
        )
        .agg(
            value=(
                "value",
                "mean"
            )
        )
    )


    return values


# =========================================================
# BUILD WEATHER VARIABLE USING MULTIPLE STATIONS
# =========================================================

def build_weather_variable(
    parameter_tuple,
    latitude,
    longitude,
    variable_name
):

    request = DwdObservationRequest(
        parameters=parameter_tuple,
        start_date=REQUEST_START,
        end_date=REQUEST_END
    )


    nearby = (
        request
        .filter_by_distance(
            latlon=(
                latitude,
                longitude
            ),
            distance=SEARCH_RADIUS_KM,
            unit="km"
        )
        .df
        .to_pandas()
    )


    if nearby.empty:

        raise ValueError(
            f"No DWD stations found for {variable_name}"
        )


    nearby = (
        nearby
        .sort_values(
            "distance"
        )
        .head(
            MAX_CANDIDATES
        )
        .reset_index(
            drop=True
        )
    )


    # Complete hourly grid
    result = pd.DataFrame(
        {
            "hour":
                TARGET_HOURS
        }
    )

    result["value"] = float("nan")


    stations_used = []


    # -----------------------------------------------------
    # CHECK STATIONS FROM NEAREST TO FARTHEST
    # -----------------------------------------------------

    for _, station in nearby.iterrows():

        station_id = str(
            station["station_id"]
        )

        station_name = str(
            station["name"]
        )

        distance_km = float(
            station["distance"]
        )


        print(
            f"  Checking {station_name} "
            f"({station_id}) - "
            f"{distance_km:.1f} km..."
        )


        try:

            values = get_station_values(
                request,
                station_id
            )

        except Exception as error:

            print(
                f"    Could not use station: {error}"
            )

            continue


        if values.empty:

            print(
                "    No usable data."
            )

            continue


        station_column = (
            f"station_{station_id}"
        )


        values = values.rename(
            columns={
                "value":
                    station_column
            }
        )


        result = result.merge(
            values,
            on="hour",
            how="left"
        )


        # How many empty hours can this station fill?
        can_fill = (
            result["value"].isna()
            &
            result[station_column].notna()
        )


        hours_added = int(
            can_fill.sum()
        )


        result.loc[
            can_fill,
            "value"
        ] = result.loc[
            can_fill,
            station_column
        ]


        total_available = int(
            result["value"]
            .notna()
            .sum()
        )


        coverage_pct = (
            total_available
            / len(TARGET_HOURS)
            * 100
        )


        print(
            f"    Added hours: {hours_added}"
        )

        print(
            f"    Total coverage now: "
            f"{coverage_pct:.2f}%"
        )


        if hours_added > 0:

            stations_used.append(
                {
                    "station_id":
                        station_id,

                    "station_name":
                        station_name,

                    "distance_km":
                        round(
                            distance_km,
                            2
                        ),

                    "hours_contributed":
                        hours_added
                }
            )


        result = result.drop(
            columns=[
                station_column
            ]
        )


        # If everything is filled, stop
        if result["value"].notna().all():

            break


    # -----------------------------------------------------
    # COUNT REMAINING MISSING VALUES
    # -----------------------------------------------------

    missing_before_imputation = int(
        result["value"]
        .isna()
        .sum()
    )


    # -----------------------------------------------------
    # ONLY FILL SMALL REMAINING GAPS
    #
    # Example:
    # German daylight-saving-time missing local hour.
    # -----------------------------------------------------

    result["value"] = (
        result["value"]
        .interpolate(
            method="linear",
            limit=2,
            limit_direction="both"
        )
    )


    missing_after_imputation = int(
        result["value"]
        .isna()
        .sum()
    )


    return (
        result,
        stations_used,
        missing_before_imputation,
        missing_after_imputation
    )


# =========================================================
# PROCESS EACH SITE
# =========================================================

for _, site in sites.iterrows():

    site_id = site[
        "site_id"
    ]

    city = site[
        "city"
    ]

    latitude = float(
        site[
            "latitude"
        ]
    )

    longitude = float(
        site[
            "longitude"
        ]
    )


    print(
        "\n========================================"
    )

    print(
        f"WEATHER FOR {city}"
    )

    print(
        "========================================"
    )


    # =====================================================
    # TEMPERATURE
    # =====================================================

    print(
        "\nTemperature:"
    )


    (
        temperature,
        temperature_stations,
        temp_missing_before,
        temp_missing_after
    ) = build_weather_variable(
        (
            "hourly",
            "temperature_air",
            "temperature_air_mean_2m"
        ),
        latitude,
        longitude,
        "temperature"
    )


    temperature = temperature.rename(
        columns={
            "value":
                "temperature_c"
        }
    )


    # =====================================================
    # PRECIPITATION
    # =====================================================

    print(
        "\nPrecipitation:"
    )


    (
        precipitation,
        precipitation_stations,
        precip_missing_before,
        precip_missing_after
    ) = build_weather_variable(
        (
            "hourly",
            "precipitation",
            "precipitation_amount"
        ),
        latitude,
        longitude,
        "precipitation"
    )


    precipitation = precipitation.rename(
        columns={
            "value":
                "precipitation_mm"
        }
    )


    # =====================================================
    # MERGE TEMPERATURE + RAIN
    # =====================================================

    weather = temperature.merge(
        precipitation,
        on="hour",
        how="outer"
    )


    weather["site_id"] = site_id
    weather["city"] = city


    all_weather.append(
        weather
    )


    # =====================================================
    # STATION REPORT
    # =====================================================

    station_records.append(
        {
            "site_id":
                site_id,

            "city":
                city,

            "temperature_stations_used":
                len(
                    temperature_stations
                ),

            "temperature_missing_before_imputation":
                temp_missing_before,

            "temperature_missing_after_imputation":
                temp_missing_after,

            "precipitation_stations_used":
                len(
                    precipitation_stations
                ),

            "precipitation_missing_before_imputation":
                precip_missing_before,

            "precipitation_missing_after_imputation":
                precip_missing_after,

            "temperature_station_details":
                " | ".join(
                    [
                        (
                            f"{x['station_name']} "
                            f"({x['station_id']}), "
                            f"{x['distance_km']} km, "
                            f"{x['hours_contributed']} h"
                        )
                        for x
                        in temperature_stations
                    ]
                ),

            "precipitation_station_details":
                " | ".join(
                    [
                        (
                            f"{x['station_name']} "
                            f"({x['station_id']}), "
                            f"{x['distance_km']} km, "
                            f"{x['hours_contributed']} h"
                        )
                        for x
                        in precipitation_stations
                    ]
                )
        }
    )


# =========================================================
# COMBINE ALL SITES
# =========================================================

weather_df = pd.concat(
    all_weather,
    ignore_index=True
)


weather_df = (
    weather_df
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


station_df = pd.DataFrame(
    station_records
)


# =========================================================
# FINAL QA
# =========================================================

expected_rows = (
    len(sites)
    * len(TARGET_HOURS)
)


actual_rows = len(
    weather_df
)


missing_temperature = int(
    weather_df[
        "temperature_c"
    ]
    .isna()
    .sum()
)


missing_precipitation = int(
    weather_df[
        "precipitation_mm"
    ]
    .isna()
    .sum()
)


print(
    "\n========================================"
)

print(
    "FINAL WEATHER QA"
)

print(
    "========================================"
)

print(
    "Expected rows:",
    expected_rows
)

print(
    "Actual rows:",
    actual_rows
)

print(
    "Missing temperature:",
    missing_temperature
)

print(
    "Missing precipitation:",
    missing_precipitation
)


# =========================================================
# STOP IF STILL INCOMPLETE
# =========================================================

if actual_rows != expected_rows:

    raise ValueError(
        "Weather row count is wrong."
    )


if missing_temperature > 0:

    raise ValueError(
        "Temperature still contains missing values."
    )


if missing_precipitation > 0:

    raise ValueError(
        "Precipitation still contains missing values."
    )


# =========================================================
# SAVE
# =========================================================

weather_df.to_csv(
    WEATHER_OUTPUT,
    index=False
)


station_df.to_csv(
    STATION_OUTPUT,
    index=False
)


# =========================================================
# FINISH
# =========================================================

print(
    "\n--- WEATHER PIPELINE PASSED ---"
)

print(
    "\nSite summary:"
)

print(
    station_df[
        [
            "site_id",
            "city",
            "temperature_stations_used",
            "temperature_missing_before_imputation",
            "precipitation_stations_used",
            "precipitation_missing_before_imputation"
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nWeather saved to:"
)

print(
    WEATHER_OUTPUT
)


print(
    "\nStation mapping saved to:"
)

print(
    STATION_OUTPUT
)