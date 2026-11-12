from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SITE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "site_master.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

sites = pd.read_csv(
    SITE_FILE,
    skipinitialspace=True
)


# ---------------------------------------------------------
# CLEAN COLUMN NAMES
# ---------------------------------------------------------

sites.columns = (
    sites.columns
    .str.strip()
    .str.replace(
        "\ufeff",
        "",
        regex=False
    )
)


# ---------------------------------------------------------
# SHOW WHAT FILE / COLUMNS ARE BEING USED
# ---------------------------------------------------------

print(
    "\nReading:"
)

print(
    SITE_FILE
)

print(
    "\nColumns found:"
)

print(
    sites.columns.tolist()
)


# ---------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = [
    "site_id",
    "city",
    "charger_count",
    "connectors_per_charger",
    "connector_count",
    "charger_power_kw",
    "grid_capacity_kw"
]


missing_columns = [
    column
    for column in required_columns
    if column not in sites.columns
]


if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ---------------------------------------------------------
# CHECK CONNECTOR STRUCTURE
# ---------------------------------------------------------

sites[
    "calculated_connector_count"
] = (
    sites["charger_count"]
    *
    sites["connectors_per_charger"]
)


sites[
    "connector_count_correct"
] = (
    sites[
        "calculated_connector_count"
    ]
    ==
    sites[
        "connector_count"
    ]
)


# ---------------------------------------------------------
# PHYSICAL CHARGER POWER
#
# charger_power_kw means power of ONE physical charger.
# ---------------------------------------------------------

sites[
    "installed_charger_power_kw"
] = (
    sites["charger_count"]
    *
    sites["charger_power_kw"]
)


# ---------------------------------------------------------
# GRID CHECK
# ---------------------------------------------------------

sites[
    "grid_power_gap_kw"
] = (
    sites[
        "installed_charger_power_kw"
    ]
    -
    sites[
        "grid_capacity_kw"
    ]
)


sites[
    "grid_to_charger_power_ratio"
] = (
    sites[
        "grid_capacity_kw"
    ]
    /
    sites[
        "installed_charger_power_kw"
    ]
)


# ---------------------------------------------------------
# DISPLAY CONNECTOR CHECK
# ---------------------------------------------------------

print(
    "\n--- CHARGER / CONNECTOR CHECK ---"
)

print(
    sites[
        [
            "site_id",
            "city",
            "charger_count",
            "connectors_per_charger",
            "connector_count",
            "calculated_connector_count",
            "connector_count_correct"
        ]
    ]
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# DISPLAY POWER CHECK
# ---------------------------------------------------------

print(
    "\n--- SITE POWER CHECK ---"
)

print(
    sites[
        [
            "site_id",
            "city",
            "charger_count",
            "charger_power_kw",
            "installed_charger_power_kw",
            "grid_capacity_kw",
            "grid_power_gap_kw",
            "grid_to_charger_power_ratio"
        ]
    ]
    .to_string(
        index=False
    )
)