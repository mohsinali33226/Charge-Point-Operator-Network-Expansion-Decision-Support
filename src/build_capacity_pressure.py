from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_hourly_demand.csv"
)

OUTPUT_HOURLY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_hourly_capacity_pressure.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_capacity_summary.csv"
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


# =========================================================
# HOURLY PRESSURE FLAGS
# =========================================================

data[
    "connector_pressure_80"
] = (
    data[
        "peak_connector_utilisation_pct"
    ] >= 80
).astype(int)


data[
    "connector_pressure_100"
] = (
    data[
        "peak_connector_utilisation_pct"
    ] >= 100
).astype(int)


data[
    "grid_pressure_80"
] = (
    data[
        "peak_grid_utilisation_pct"
    ] >= 80
).astype(int)


data[
    "grid_pressure_100"
] = (
    data[
        "peak_grid_utilisation_pct"
    ] >= 100
).astype(int)


# =========================================================
# SITE SUMMARY
# =========================================================

summary_rows = []


for site_id, site_data in data.groupby(
    "site_id"
):

    city = (
        site_data[
            "city"
        ]
        .iloc[0]
    )


    charger_count = int(
        site_data[
            "charger_count"
        ]
        .iloc[0]
    )


    connector_count = int(
        site_data[
            "connector_count"
        ]
        .iloc[0]
    )


    installed_charger_power_kw = float(
        site_data[
            "installed_charger_power_kw"
        ]
        .iloc[0]
    )


    grid_capacity_kw = float(
        site_data[
            "grid_capacity_kw"
        ]
        .iloc[0]
    )


    total_hours = len(
        site_data
    )


    # =====================================================
    # CONNECTOR PRESSURE
    # =====================================================

    maximum_connected_vehicles = (
        site_data[
            "peak_connected_vehicles"
        ]
        .max()
    )


    p95_connected_vehicles = (
        site_data[
            "peak_connected_vehicles"
        ]
        .quantile(
            0.95
        )
    )


    p95_connector_utilisation_pct = (
        site_data[
            "peak_connector_utilisation_pct"
        ]
        .quantile(
            0.95
        )
    )


    hours_connector_ge_80 = int(
        site_data[
            "connector_pressure_80"
        ]
        .sum()
    )


    hours_connector_full = int(
        site_data[
            "connector_pressure_100"
        ]
        .sum()
    )


    pct_hours_connector_ge_80 = (
        hours_connector_ge_80
        /
        total_hours
        *
        100
    )


    pct_hours_connector_full = (
        hours_connector_full
        /
        total_hours
        *
        100
    )


    # =====================================================
    # GRID / POWER PRESSURE
    # =====================================================

    maximum_site_power_kw = (
        site_data[
            "peak_site_power_kw"
        ]
        .max()
    )


    p95_site_power_kw = (
        site_data[
            "peak_site_power_kw"
        ]
        .quantile(
            0.95
        )
    )


    maximum_grid_utilisation_pct = (
        site_data[
            "peak_grid_utilisation_pct"
        ]
        .max()
    )


    p95_grid_utilisation_pct = (
        site_data[
            "peak_grid_utilisation_pct"
        ]
        .quantile(
            0.95
        )
    )


    hours_grid_ge_80 = int(
        site_data[
            "grid_pressure_80"
        ]
        .sum()
    )


    hours_grid_full = int(
        site_data[
            "grid_pressure_100"
        ]
        .sum()
    )


    pct_hours_grid_ge_80 = (
        hours_grid_ge_80
        /
        total_hours
        *
        100
    )


    # =====================================================
    # CONGESTION
    # =====================================================

    total_arrivals = (
        site_data[
            "arrivals"
        ]
        .sum()
    )


    total_congestion_events = (
        site_data[
            "congestion_events"
        ]
        .sum()
    )


    if total_arrivals > 0:

        congestion_rate_pct = (
            total_congestion_events
            /
            total_arrivals
            *
            100
        )

    else:

        congestion_rate_pct = 0


    # =====================================================
    # SIMPLE BOTTLENECK LABEL
    # =====================================================

    connector_pressure = (
        p95_connector_utilisation_pct >= 80
        or
        total_congestion_events > 0
    )


    grid_pressure = (
        p95_grid_utilisation_pct >= 80
    )


    if (
        connector_pressure
        and grid_pressure
    ):

        bottleneck_type = (
            "CONNECTOR + GRID"
        )

    elif connector_pressure:

        bottleneck_type = (
            "CONNECTOR"
        )

    elif grid_pressure:

        bottleneck_type = (
            "GRID"
        )

    else:

        bottleneck_type = (
            "NO MAJOR PRESSURE"
        )


    # =====================================================
    # STORE
    # =====================================================

    summary_rows.append(
        {
            "site_id":
                site_id,

            "city":
                city,

            "charger_count":
                charger_count,

            "connector_count":
                connector_count,

            "installed_charger_power_kw":
                installed_charger_power_kw,

            "grid_capacity_kw":
                grid_capacity_kw,

            "maximum_connected_vehicles":
                maximum_connected_vehicles,

            "p95_connected_vehicles":
                p95_connected_vehicles,

            "p95_connector_utilisation_pct":
                p95_connector_utilisation_pct,

            "hours_connector_ge_80":
                hours_connector_ge_80,

            "pct_hours_connector_ge_80":
                pct_hours_connector_ge_80,

            "hours_connector_full":
                hours_connector_full,

            "pct_hours_connector_full":
                pct_hours_connector_full,

            "maximum_site_power_kw":
                maximum_site_power_kw,

            "p95_site_power_kw":
                p95_site_power_kw,

            "maximum_grid_utilisation_pct":
                maximum_grid_utilisation_pct,

            "p95_grid_utilisation_pct":
                p95_grid_utilisation_pct,

            "hours_grid_ge_80":
                hours_grid_ge_80,

            "pct_hours_grid_ge_80":
                pct_hours_grid_ge_80,

            "hours_grid_full":
                hours_grid_full,

            "total_congestion_events":
                total_congestion_events,

            "congestion_rate_pct":
                congestion_rate_pct,

            "bottleneck_type":
                bottleneck_type,
        }
    )


# =========================================================
# CREATE SUMMARY
# =========================================================

summary = pd.DataFrame(
    summary_rows
)


# =========================================================
# ROUND
# =========================================================

numeric_columns = (
    summary
    .select_dtypes(
        include=[
            "number"
        ]
    )
    .columns
)


summary[
    numeric_columns
] = (
    summary[
        numeric_columns
    ]
    .round(2)
)


# =========================================================
# SAVE
# =========================================================

data.to_csv(
    OUTPUT_HOURLY,
    index=False
)


summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# =========================================================
# DISPLAY
# =========================================================

print(
    "\n--- SITE CAPACITY PRESSURE SUMMARY ---"
)


print(
    summary[
        [
            "site_id",
            "city",
            "charger_count",
            "connector_count",
            "p95_connected_vehicles",
            "p95_connector_utilisation_pct",
            "hours_connector_full",
            "congestion_rate_pct",
            "p95_site_power_kw",
            "p95_grid_utilisation_pct",
            "maximum_grid_utilisation_pct",
            "bottleneck_type"
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nSaved hourly capacity data:"
)

print(
    OUTPUT_HOURLY
)


print(
    "\nSaved site summary:"
)

print(
    OUTPUT_SUMMARY
)