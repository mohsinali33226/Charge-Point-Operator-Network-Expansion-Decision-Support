
from pathlib import Path
import pandas as pd

# =====================================================
# FILE PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "site_capacity_summary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "site_bottleneck_diagnosis.csv"
)

# =====================================================
# LOAD DATA
# =====================================================

data = pd.read_csv(INPUT_FILE)

required_columns = [
    "site_id",
    "city",
    "charger_count",
    "connector_count",
    "hours_connector_full",
    "congestion_rate_pct",
    "p95_connector_utilisation_pct",
    "installed_charger_power_kw",
    "grid_capacity_kw",
    "maximum_site_power_kw",
    "maximum_grid_utilisation_pct",
]

missing = [
    col for col in required_columns
    if col not in data.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if data["site_id"].duplicated().any():
    raise ValueError("Duplicate site IDs found.")

# =====================================================
# EXPANSION STUDY PRIORITY
#
# These are initial project assumptions,
# not official CPO industry thresholds.
# =====================================================

def determine_priority(row):

    congestion = row["congestion_rate_pct"]
    full_hours = row["hours_connector_full"]

    if congestion >= 2 and full_hours >= 100:
        return "HIGH"

    elif congestion >= 1 or full_hours >= 200:
        return "MEDIUM"

    else:
        return "MONITOR"


data["expansion_study_priority"] = data.apply(
    determine_priority,
    axis=1
)

# =====================================================
# GRID CAPACITY
# =====================================================

data["grid_headroom_at_peak_kw"] = (
    data["grid_capacity_kw"]
    - data["maximum_site_power_kw"]
)

data["installed_power_headroom_kw"] = (
    data["grid_capacity_kw"]
    - data["installed_charger_power_kw"]
)

# Initial warning threshold
data["grid_warning"] = (
    data["maximum_grid_utilisation_pct"] >= 85
)

# =====================================================
# RECOMMENDED NEXT ANALYSIS
# =====================================================

def next_action(row):

    priority = row["expansion_study_priority"]

    if priority == "HIGH":
        return "Prioritise expansion scenario analysis"

    elif priority == "MEDIUM":
        return "Compare additional connector options"

    else:
        return "Monitor and reassess demand"


data["next_action"] = data.apply(
    next_action,
    axis=1
)

# =====================================================
# PRIORITY RANKING
# =====================================================

priority_order = {
    "HIGH": 1,
    "MEDIUM": 2,
    "MONITOR": 3
}

data["priority_rank"] = (
    data["expansion_study_priority"]
    .map(priority_order)
)

data = data.sort_values(
    [
        "priority_rank",
        "congestion_rate_pct",
        "hours_connector_full"
    ],
    ascending=[True, False, False]
).reset_index(drop=True)

# =====================================================
# SAVE
# =====================================================

data.to_csv(OUTPUT_FILE, index=False)

# =====================================================
# DISPLAY
# =====================================================

print("\n--- NETWORK DEVELOPMENT PRIORITIES ---")

print(
    data[
        [
            "city",
            "charger_count",
            "connector_count",
            "hours_connector_full",
            "congestion_rate_pct",
            "expansion_study_priority"
        ]
    ].to_string(index=False)
)

print("\n--- GRID CAPACITY CHECK ---")

print(
    data[
        [
            "city",
            "grid_capacity_kw",
            "maximum_site_power_kw",
            "grid_headroom_at_peak_kw",
            "grid_warning"
        ]
    ].round(2).to_string(index=False)
)

print("\n--- NEXT ACTIONS ---")

print(
    data[
        [
            "city",
            "next_action"
        ]
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)
