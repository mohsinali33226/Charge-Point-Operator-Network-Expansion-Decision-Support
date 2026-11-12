from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "charging_sessions.csv"
)

DEMAND_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "demand_attempts.csv"
)

SITE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "site_master.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_baseline_kpis.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

sessions = pd.read_csv(
    SESSION_FILE
)

demand = pd.read_csv(
    DEMAND_FILE
)

sites = pd.read_csv(
    SITE_FILE,
    skipinitialspace=True
)

sites.columns = (
    sites.columns
    .str.strip()
)


# =========================================================
# CONVERT TIMES
# =========================================================

sessions["start_time"] = pd.to_datetime(
    sessions["start_time"]
)

sessions["end_time"] = pd.to_datetime(
    sessions["end_time"]
)

demand["start_time"] = pd.to_datetime(
    demand["start_time"]
)


# =========================================================
# REQUIRED COLUMNS CHECK
# =========================================================

required_session_columns = [
    "session_id",
    "site_id",
    "charger_id",
    "connector_id",
    "start_time",
    "end_time",
    "duration_minutes",
    "requested_power_kw",
    "requested_energy_kwh",
    "average_power_kw",
    "energy_kwh",
    "energy_shortfall_kwh",
]

required_demand_columns = [
    "arrival_id",
    "site_id",
    "start_time",
    "served",
    "congestion_event",
]

required_site_columns = [
    "site_id",
    "city",
    "charger_count",
    "connectors_per_charger",
    "connector_count",
    "charger_power_kw",
    "max_connector_power_kw",
    "grid_capacity_kw",
]


for column in required_session_columns:

    if column not in sessions.columns:

        raise ValueError(
            f"Missing session column: {column}"
        )


for column in required_demand_columns:

    if column not in demand.columns:

        raise ValueError(
            f"Missing demand column: {column}"
        )


for column in required_site_columns:

    if column not in sites.columns:

        raise ValueError(
            f"Missing site column: {column}"
        )


# =========================================================
# OBSERVATION PERIOD
# =========================================================

observation_start = (
    demand["start_time"]
    .min()
    .normalize()
)

observation_end = (
    demand["start_time"]
    .max()
    .normalize()
)


number_days = (
    (
        observation_end
        -
        observation_start
    ).days
    +
    1
)


number_hours = (
    number_days
    *
    24
)


print(
    "\n--- OBSERVATION PERIOD ---"
)

print(
    "From:",
    observation_start
)

print(
    "To:",
    observation_end
)

print(
    "Days:",
    number_days
)


# =========================================================
# BUILD KPI TABLE
# =========================================================

kpi_rows = []


for _, site in sites.iterrows():

    site_id = site[
        "site_id"
    ]

    city = site[
        "city"
    ]

    site_sessions = sessions[
        sessions[
            "site_id"
        ] == site_id
    ].copy()

    site_demand = demand[
        demand[
            "site_id"
        ] == site_id
    ].copy()


    # =====================================================
    # BASIC DEMAND
    # =====================================================

    total_arrivals = len(
        site_demand
    )

    served_sessions = len(
        site_sessions
    )

    congestion_events = int(
        site_demand[
            "congestion_event"
        ]
        .sum()
    )


    if total_arrivals > 0:

        congestion_rate_pct = (
            congestion_events
            /
            total_arrivals
            *
            100
        )

        service_rate_pct = (
            served_sessions
            /
            total_arrivals
            *
            100
        )

    else:

        congestion_rate_pct = 0

        service_rate_pct = 0


    # =====================================================
    # DAILY DEMAND
    # =====================================================

    average_daily_arrivals = (
        total_arrivals
        /
        number_days
    )

    average_daily_sessions = (
        served_sessions
        /
        number_days
    )


    # =====================================================
    # SESSION DURATION
    # =====================================================

    average_session_duration_min = (
        site_sessions[
            "duration_minutes"
        ]
        .mean()
    )


    p95_session_duration_min = (
        site_sessions[
            "duration_minutes"
        ]
        .quantile(
            0.95
        )
    )


    # =====================================================
    # ENERGY
    # =====================================================

    average_requested_energy_kwh = (
        site_sessions[
            "requested_energy_kwh"
        ]
        .mean()
    )

    average_delivered_energy_kwh = (
        site_sessions[
            "energy_kwh"
        ]
        .mean()
    )


    total_requested_energy_kwh = (
        site_sessions[
            "requested_energy_kwh"
        ]
        .sum()
    )

    total_delivered_energy_kwh = (
        site_sessions[
            "energy_kwh"
        ]
        .sum()
    )

    total_energy_shortfall_kwh = (
        site_sessions[
            "energy_shortfall_kwh"
        ]
        .sum()
    )


    if total_requested_energy_kwh > 0:

        energy_delivery_pct = (
            total_delivered_energy_kwh
            /
            total_requested_energy_kwh
            *
            100
        )

    else:

        energy_delivery_pct = 0


    # =====================================================
    # POWER
    # =====================================================

    average_requested_power_kw = (
        site_sessions[
            "requested_power_kw"
        ]
        .mean()
    )

    average_delivered_power_kw = (
        site_sessions[
            "average_power_kw"
        ]
        .mean()
    )


    # =====================================================
    # POWER SHARING
    #
    # If energy_shortfall > small tolerance,
    # charger sharing affected that session.
    # =====================================================

    power_sharing_sessions = int(
        (
            site_sessions[
                "energy_shortfall_kwh"
            ]
            >
            0.01
        )
        .sum()
    )


    if served_sessions > 0:

        power_sharing_session_pct = (
            power_sharing_sessions
            /
            served_sessions
            *
            100
        )

    else:

        power_sharing_session_pct = 0


    # =====================================================
    # CONNECTOR OCCUPANCY
    #
    # Total time all connectors were occupied compared
    # with the total connector-hours available.
    # =====================================================

    total_connected_hours = (
        site_sessions[
            "duration_minutes"
        ]
        .sum()
        /
        60
    )


    connector_count = int(
        site[
            "connector_count"
        ]
    )


    total_connector_hours_available = (
        connector_count
        *
        number_hours
    )


    if total_connector_hours_available > 0:

        average_connector_utilisation_pct = (
            total_connected_hours
            /
            total_connector_hours_available
            *
            100
        )

    else:

        average_connector_utilisation_pct = 0


    # =====================================================
    # AVERAGE NUMBER OF CONNECTED VEHICLES
    # =====================================================

    average_connected_vehicles = (
        total_connected_hours
        /
        number_hours
    )


    # =====================================================
    # SITE POWER STRUCTURE
    # =====================================================

    charger_count = int(
        site[
            "charger_count"
        ]
    )

    charger_power_kw = float(
        site[
            "charger_power_kw"
        ]
    )

    grid_capacity_kw = float(
        site[
            "grid_capacity_kw"
        ]
    )


    installed_charger_power_kw = (
        charger_count
        *
        charger_power_kw
    )


    if installed_charger_power_kw > 0:

        grid_to_charger_power_ratio = (
            grid_capacity_kw
            /
            installed_charger_power_kw
        )

    else:

        grid_to_charger_power_ratio = 0


    # =====================================================
    # STORE RESULT
    # =====================================================

    kpi_rows.append(
        {
            "site_id":
                site_id,

            "city":
                city,

            "charger_count":
                charger_count,

            "connectors_per_charger":
                int(
                    site[
                        "connectors_per_charger"
                    ]
                ),

            "connector_count":
                connector_count,

            "charger_power_kw":
                charger_power_kw,

            "max_connector_power_kw":
                float(
                    site[
                        "max_connector_power_kw"
                    ]
                ),

            "installed_charger_power_kw":
                installed_charger_power_kw,

            "grid_capacity_kw":
                grid_capacity_kw,

            "grid_to_charger_power_ratio":
                grid_to_charger_power_ratio,

            "total_arrivals":
                total_arrivals,

            "served_sessions":
                served_sessions,

            "congestion_events":
                congestion_events,

            "congestion_rate_pct":
                congestion_rate_pct,

            "service_rate_pct":
                service_rate_pct,

            "average_daily_arrivals":
                average_daily_arrivals,

            "average_daily_sessions":
                average_daily_sessions,

            "average_session_duration_min":
                average_session_duration_min,

            "p95_session_duration_min":
                p95_session_duration_min,

            "average_requested_energy_kwh":
                average_requested_energy_kwh,

            "average_delivered_energy_kwh":
                average_delivered_energy_kwh,

            "total_requested_energy_kwh":
                total_requested_energy_kwh,

            "total_delivered_energy_kwh":
                total_delivered_energy_kwh,

            "total_energy_shortfall_kwh":
                total_energy_shortfall_kwh,

            "energy_delivery_pct":
                energy_delivery_pct,

            "average_requested_power_kw":
                average_requested_power_kw,

            "average_delivered_power_kw":
                average_delivered_power_kw,

            "power_sharing_sessions":
                power_sharing_sessions,

            "power_sharing_session_pct":
                power_sharing_session_pct,

            "average_connector_utilisation_pct":
                average_connector_utilisation_pct,

            "average_connected_vehicles":
                average_connected_vehicles,
        }
    )


# =========================================================
# CREATE DATAFRAME
# =========================================================

kpis = pd.DataFrame(
    kpi_rows
)


# =========================================================
# ROUND NUMBERS
# =========================================================

numeric_columns = (
    kpis
    .select_dtypes(
        include=[
            np.number
        ]
    )
    .columns
)


kpis[
    numeric_columns
] = (
    kpis[
        numeric_columns
    ]
    .round(2)
)


# =========================================================
# SAVE
# =========================================================

kpis.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# DISPLAY MAIN KPIS
# =========================================================

print(
    "\n--- SITE BASELINE KPIS ---"
)


print(
    kpis[
        [
            "site_id",
            "city",
            "charger_count",
            "connector_count",
            "average_daily_sessions",
            "average_session_duration_min",
            "average_delivered_energy_kwh",
            "congestion_rate_pct",
            "average_connector_utilisation_pct",
            "power_sharing_session_pct",
            "energy_delivery_pct"
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\n--- SITE POWER STRUCTURE ---"
)


print(
    kpis[
        [
            "site_id",
            "installed_charger_power_kw",
            "grid_capacity_kw",
            "grid_to_charger_power_ratio",
            "total_requested_energy_kwh",
            "total_delivered_energy_kwh",
            "total_energy_shortfall_kwh"
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