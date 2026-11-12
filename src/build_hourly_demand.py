from pathlib import Path

import numpy as np
import pandas as pd


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
    / "site_hourly_demand.csv"
)


# =========================================================
# PERIOD
# =========================================================

START_TIME = pd.Timestamp(
    "2025-01-01 00:00:00"
)

END_TIME_EXCLUSIVE = pd.Timestamp(
    "2025-07-01 00:00:00"
)


HOURLY_INDEX = pd.date_range(
    START_TIME,
    END_TIME_EXCLUSIVE
    - pd.Timedelta(hours=1),
    freq="h"
)


TOTAL_HOURS = len(
    HOURLY_INDEX
)

TOTAL_MINUTES = (
    TOTAL_HOURS
    * 60
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


sessions[
    "start_time"
] = pd.to_datetime(
    sessions[
        "start_time"
    ]
)


sessions[
    "end_time"
] = pd.to_datetime(
    sessions[
        "end_time"
    ]
)


demand[
    "start_time"
] = pd.to_datetime(
    demand[
        "start_time"
    ]
)


# =========================================================
# STORAGE
# =========================================================

all_site_hourly = []


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

    charger_count = int(
        site[
            "charger_count"
        ]
    )

    connector_count = int(
        site[
            "connector_count"
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


    print(
        f"\nBuilding hourly demand for {city}..."
    )


    # =====================================================
    # MINUTE-LEVEL CONNECTED VEHICLES
    # =====================================================

    occupancy_change = np.zeros(
        TOTAL_MINUTES + 1,
        dtype=int
    )


    for _, session in (
        site_sessions.iterrows()
    ):

        session_start = max(
            session[
                "start_time"
            ],
            START_TIME
        )

        session_end = min(
            session[
                "end_time"
            ],
            END_TIME_EXCLUSIVE
        )


        if (
            session_end
            <= session_start
        ):

            continue


        start_index = int(
            (
                session_start
                -
                START_TIME
            )
            .total_seconds()
            /
            60
        )


        end_index = int(
            (
                session_end
                -
                START_TIME
            )
            .total_seconds()
            /
            60
        )


        occupancy_change[
            start_index
        ] += 1


        occupancy_change[
            end_index
        ] -= 1


    connected_vehicles = np.cumsum(
        occupancy_change[:-1]
    )


    # =====================================================
    # MINUTE-LEVEL SITE POWER
    #
    # Rebuild the same physical charger sharing logic:
    #
    # actual charger power
    # =
    # min(total EV request, charger power limit)
    # =====================================================

    site_actual_power = np.zeros(
        TOTAL_MINUTES,
        dtype=float
    )


    site_requested_power = np.zeros(
        TOTAL_MINUTES,
        dtype=float
    )


    for charger_id in range(
        1,
        charger_count + 1
    ):

        charger_sessions = (
            site_sessions[
                site_sessions[
                    "charger_id"
                ] == charger_id
            ]
            .copy()
        )


        if charger_sessions.empty:

            continue


        power_change = np.zeros(
            TOTAL_MINUTES + 1,
            dtype=float
        )


        for _, session in (
            charger_sessions.iterrows()
        ):

            session_start = max(
                session[
                    "start_time"
                ],
                START_TIME
            )

            session_end = min(
                session[
                    "end_time"
                ],
                END_TIME_EXCLUSIVE
            )


            if (
                session_end
                <= session_start
            ):

                continue


            start_index = int(
                (
                    session_start
                    -
                    START_TIME
                )
                .total_seconds()
                /
                60
            )


            end_index = int(
                (
                    session_end
                    -
                    START_TIME
                )
                .total_seconds()
                /
                60
            )


            requested_power = float(
                session[
                    "requested_power_kw"
                ]
            )


            power_change[
                start_index
            ] += requested_power


            power_change[
                end_index
            ] -= requested_power


        charger_requested_power = np.cumsum(
            power_change[:-1]
        )


        charger_actual_power = np.minimum(
            charger_requested_power,
            charger_power_kw
        )


        site_requested_power += (
            charger_requested_power
        )


        site_actual_power += (
            charger_actual_power
        )


    # =====================================================
    # RESHAPE MINUTE DATA INTO HOURS
    # =====================================================

    connected_matrix = (
        connected_vehicles
        .reshape(
            TOTAL_HOURS,
            60
        )
    )


    actual_power_matrix = (
        site_actual_power
        .reshape(
            TOTAL_HOURS,
            60
        )
    )


    requested_power_matrix = (
        site_requested_power
        .reshape(
            TOTAL_HOURS,
            60
        )
    )


    average_connected_vehicles = (
        connected_matrix
        .mean(
            axis=1
        )
    )


    peak_connected_vehicles = (
        connected_matrix
        .max(
            axis=1
        )
    )


    average_site_power_kw = (
        actual_power_matrix
        .mean(
            axis=1
        )
    )


    peak_site_power_kw = (
        actual_power_matrix
        .max(
            axis=1
        )
    )


    # kW × minute / 60 = kWh
    delivered_energy_kwh = (
        actual_power_matrix
        .sum(
            axis=1
        )
        /
        60
    )


    requested_charging_energy_kwh = (
        requested_power_matrix
        .sum(
            axis=1
        )
        /
        60
    )


    power_sharing_shortfall_kwh = (
        requested_charging_energy_kwh
        -
        delivered_energy_kwh
    )


    power_sharing_shortfall_kwh = (
        np.clip(
            power_sharing_shortfall_kwh,
            0,
            None
        )
    )


    # =====================================================
    # CREATE BASE HOURLY TABLE
    # =====================================================

    hourly = pd.DataFrame(
        {
            "site_id":
                site_id,

            "city":
                city,

            "hour":
                HOURLY_INDEX,

            "average_connected_vehicles":
                average_connected_vehicles,

            "peak_connected_vehicles":
                peak_connected_vehicles,

            "average_site_power_kw":
                average_site_power_kw,

            "peak_site_power_kw":
                peak_site_power_kw,

            "requested_charging_energy_kwh":
                requested_charging_energy_kwh,

            "delivered_energy_kwh":
                delivered_energy_kwh,

            "power_sharing_shortfall_kwh":
                power_sharing_shortfall_kwh,
        }
    )


    # =====================================================
    # ARRIVAL / CONGESTION DATA
    # =====================================================

    site_demand[
        "hour"
    ] = (
        site_demand[
            "start_time"
        ]
        .dt
        .floor("h")
    )


    demand_hourly = (
        site_demand
        .groupby(
            "hour",
            as_index=False
        )
        .agg(
            arrivals=(
                "arrival_id",
                "count"
            ),

            served_sessions=(
                "served",
                "sum"
            ),

            congestion_events=(
                "congestion_event",
                "sum"
            ),

            arrival_requested_energy_kwh=(
                "requested_energy_kwh",
                "sum"
            )
        )
    )


    hourly = hourly.merge(
        demand_hourly,
        on="hour",
        how="left"
    )


    # =====================================================
    # HOURS WITH ZERO ARRIVALS
    # =====================================================

    zero_fill_columns = [
        "arrivals",
        "served_sessions",
        "congestion_events",
        "arrival_requested_energy_kwh",
    ]


    hourly[
        zero_fill_columns
    ] = (
        hourly[
            zero_fill_columns
        ]
        .fillna(0)
    )


    # =====================================================
    # ADD SITE STRUCTURE
    # =====================================================

    hourly[
        "charger_count"
    ] = charger_count


    hourly[
        "connectors_per_charger"
    ] = int(
        site[
            "connectors_per_charger"
        ]
    )


    hourly[
        "connector_count"
    ] = connector_count


    hourly[
        "charger_power_kw"
    ] = charger_power_kw


    hourly[
        "max_connector_power_kw"
    ] = float(
        site[
            "max_connector_power_kw"
        ]
    )


    hourly[
        "installed_charger_power_kw"
    ] = (
        installed_charger_power_kw
    )


    hourly[
        "grid_capacity_kw"
    ] = grid_capacity_kw


    # =====================================================
    # UTILISATION
    # =====================================================

    hourly[
        "average_connector_utilisation_pct"
    ] = (
        hourly[
            "average_connected_vehicles"
        ]
        /
        connector_count
        *
        100
    )


    hourly[
        "peak_connector_utilisation_pct"
    ] = (
        hourly[
            "peak_connected_vehicles"
        ]
        /
        connector_count
        *
        100
    )


    hourly[
        "average_grid_utilisation_pct"
    ] = (
        hourly[
            "average_site_power_kw"
        ]
        /
        grid_capacity_kw
        *
        100
    )


    hourly[
        "peak_grid_utilisation_pct"
    ] = (
        hourly[
            "peak_site_power_kw"
        ]
        /
        grid_capacity_kw
        *
        100
    )


    all_site_hourly.append(
        hourly
    )


# =========================================================
# COMBINE ALL SITES
# =========================================================

hourly_data = pd.concat(
    all_site_hourly,
    ignore_index=True
)


hourly_data = (
    hourly_data
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
# ROUND
# =========================================================

float_columns = (
    hourly_data
    .select_dtypes(
        include=[
            "float"
        ]
    )
    .columns
)


hourly_data[
    float_columns
] = (
    hourly_data[
        float_columns
    ]
    .round(2)
)


# =========================================================
# FINAL PHYSICAL QA
# =========================================================

expected_rows = (
    len(sites)
    *
    TOTAL_HOURS
)


connector_violations = (
    hourly_data[
        "peak_connected_vehicles"
    ]
    >
    hourly_data[
        "connector_count"
    ]
).sum()


charger_power_violations = (
    hourly_data[
        "peak_site_power_kw"
    ]
    >
    hourly_data[
        "installed_charger_power_kw"
    ]
    +
    0.1
).sum()


grid_violations = (
    hourly_data[
        "peak_site_power_kw"
    ]
    >
    hourly_data[
        "grid_capacity_kw"
    ]
    +
    0.1
).sum()


print(
    "\n--- HOURLY DEMAND QA ---"
)

print(
    "Expected rows:",
    expected_rows
)

print(
    "Actual rows:",
    len(
        hourly_data
    )
)

print(
    "Connector capacity violations:",
    connector_violations
)

print(
    "Installed charger power violations:",
    charger_power_violations
)

print(
    "Grid capacity violations:",
    grid_violations
)


if (
    len(
        hourly_data
    )
    != expected_rows
):

    raise ValueError(
        "Hourly row count is incorrect."
    )


if connector_violations > 0:

    raise ValueError(
        "Connected vehicles exceed available connectors."
    )


if charger_power_violations > 0:

    raise ValueError(
        "Site power exceeds installed charger power."
    )


if grid_violations > 0:

    raise ValueError(
        "Site power exceeds grid capacity."
    )


# =========================================================
# SAVE
# =========================================================

hourly_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# DISPLAY SITE SUMMARY
# =========================================================

summary = (
    hourly_data
    .groupby(
        [
            "site_id",
            "city"
        ]
    )
    .agg(
        maximum_connected_vehicles=(
            "peak_connected_vehicles",
            "max"
        ),

        maximum_site_power_kw=(
            "peak_site_power_kw",
            "max"
        ),

        maximum_connector_utilisation_pct=(
            "peak_connector_utilisation_pct",
            "max"
        ),

        maximum_grid_utilisation_pct=(
            "peak_grid_utilisation_pct",
            "max"
        ),

        total_delivered_energy_kwh=(
            "delivered_energy_kwh",
            "sum"
        ),

        total_congestion_events=(
            "congestion_events",
            "sum"
        )
    )
    .reset_index()
)


print(
    "\n--- SITE HOURLY SUMMARY ---"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)