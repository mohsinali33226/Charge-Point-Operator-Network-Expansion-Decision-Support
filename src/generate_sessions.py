from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# SETUP
# =========================================================

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SITE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "site_master.csv"
)

SESSION_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "charging_sessions.csv"
)

DEMAND_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "demand_attempts.csv"
)


# =========================================================
# LOAD SITE MASTER
# =========================================================

sites = pd.read_csv(
    SITE_FILE,
    skipinitialspace=True
)

sites.columns = (
    sites.columns
    .str.strip()
)


# =========================================================
# REQUIRED SITE COLUMNS
# =========================================================

required_columns = [
    "site_id",
    "city",
    "charger_count",
    "connectors_per_charger",
    "connector_count",
    "charger_power_kw",
    "max_connector_power_kw",
    "grid_capacity_kw",
]

missing_columns = [
    column
    for column in required_columns
    if column not in sites.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns in site_master.csv: {missing_columns}"
    )


# =========================================================
# SIMULATION PERIOD
# =========================================================

start_date = pd.Timestamp(
    "2025-01-01"
)

end_date = pd.Timestamp(
    "2025-06-30"
)

dates = pd.date_range(
    start_date,
    end_date,
    freq="D"
)


# =========================================================
# DAILY ARRIVAL DEMAND
#
# Approximate average arrivals per day for our synthetic
# network.
# =========================================================

daily_demand = {
    "DE_NBG_001": 35,
    "DE_HAM_001": 70,
    "DE_FRA_001": 110,
    "DE_MUC_001": 85,
    "DE_LEI_001": 30,
}


# =========================================================
# OUTPUT CONTAINERS
# =========================================================

served_sessions = []
demand_attempts = []

session_number = 1
arrival_number = 1


# =========================================================
# PROCESS EACH SITE
# =========================================================

for _, site in sites.iterrows():

    site_id = site["site_id"]

    charger_count = int(
        site["charger_count"]
    )

    connectors_per_charger = int(
        site["connectors_per_charger"]
    )

    connector_count = int(
        site["connector_count"]
    )

    charger_power_kw = float(
        site["charger_power_kw"]
    )

    max_connector_power_kw = float(
        site["max_connector_power_kw"]
    )


    # =====================================================
    # SITE STRUCTURE CHECK
    # =====================================================

    expected_connectors = (
        charger_count
        *
        connectors_per_charger
    )

    if expected_connectors != connector_count:

        raise ValueError(
            f"{site_id}: "
            f"{charger_count} chargers × "
            f"{connectors_per_charger} connectors "
            f"= {expected_connectors}, "
            f"but connector_count is {connector_count}."
        )


    if site_id not in daily_demand:

        raise ValueError(
            f"No daily demand value defined for {site_id}"
        )


    average_daily_arrivals = (
        daily_demand[
            site_id
        ]
    )


    # =====================================================
    # CONNECTOR -> PHYSICAL CHARGER MAP
    #
    # Example:
    #
    # Charger 1 -> Connector 1, Connector 2
    # Charger 2 -> Connector 3, Connector 4
    # =====================================================

    connector_to_charger = {}

    next_connector_id = 1

    for charger_id in range(
        1,
        charger_count + 1
    ):

        for _ in range(
            connectors_per_charger
        ):

            connector_to_charger[
                next_connector_id
            ] = charger_id

            next_connector_id += 1


    # =====================================================
    # GENERATE VEHICLE ARRIVALS
    # =====================================================

    arrivals = []


    for date in dates:

        # Slightly higher demand on weekends
        weekend_factor = (
            1.15
            if date.dayofweek >= 5
            else 1.0
        )


        number_arrivals = np.random.poisson(
            average_daily_arrivals
            *
            weekend_factor
        )


        for _ in range(
            number_arrivals
        ):

            # -------------------------------------------------
            # ARRIVAL TIME
            # -------------------------------------------------

            hour = int(
                np.clip(
                    np.random.normal(
                        loc=15,
                        scale=5
                    ),
                    0,
                    23
                )
            )


            minute = int(
                np.random.randint(
                    0,
                    60
                )
            )


            arrival_time = (
                date
                +
                pd.Timedelta(
                    hours=hour
                )
                +
                pd.Timedelta(
                    minutes=minute
                )
            )


            # -------------------------------------------------
            # SESSION / PARKING DURATION
            # -------------------------------------------------

            duration_minutes = int(
                np.clip(
                    np.random.normal(
                        loc=35,
                        scale=15
                    ),
                    10,
                    120
                )
            )


            duration_hours = (
                duration_minutes
                / 60
            )


            # -------------------------------------------------
            # VEHICLE REQUESTED POWER
            #
            # One vehicle cannot request more than the
            # connector maximum.
            # -------------------------------------------------

            power_fraction = np.random.beta(
                2.5,
                3.5
            )


            requested_power_kw = (
                max_connector_power_kw
                *
                power_fraction
            )


            requested_power_kw = max(
                requested_power_kw,
                20
            )


            requested_power_kw = min(
                requested_power_kw,
                max_connector_power_kw
            )


            # -------------------------------------------------
            # REQUESTED ENERGY
            # -------------------------------------------------

            requested_energy_kwh = (
                requested_power_kw
                *
                duration_hours
            )


            requested_energy_kwh = float(
                np.clip(
                    requested_energy_kwh,
                    5,
                    120
                )
            )


            # Keep energy and power physically consistent
            requested_power_kw = (
                requested_energy_kwh
                /
                duration_hours
            )


            requested_power_kw = min(
                requested_power_kw,
                max_connector_power_kw
            )


            requested_energy_kwh = (
                requested_power_kw
                *
                duration_hours
            )


            arrivals.append(
                {
                    "arrival_id":
                        f"A{arrival_number:07d}",

                    "site_id":
                        site_id,

                    "start_time":
                        arrival_time,

                    "duration_minutes":
                        duration_minutes,

                    "requested_power_kw":
                        round(
                            requested_power_kw,
                            2
                        ),

                    "requested_energy_kwh":
                        round(
                            requested_energy_kwh,
                            2
                        ),
                }
            )


            arrival_number += 1


    # =====================================================
    # SORT ARRIVALS
    # =====================================================

    arrivals = sorted(
        arrivals,
        key=lambda row:
            row["start_time"]
    )


    # =====================================================
    # CONNECTOR AVAILABILITY
    # =====================================================

    connector_available_until = {
        connector_id:
            start_date

        for connector_id in range(
            1,
            connector_count + 1
        )
    }


    # =====================================================
    # ASSIGN ARRIVALS TO CONNECTORS
    # =====================================================

    for arrival in arrivals:

        arrival_time = (
            arrival["start_time"]
        )

        duration_minutes = (
            arrival[
                "duration_minutes"
            ]
        )


        # -------------------------------------------------
        # FIND FREE CONNECTORS
        # -------------------------------------------------

        free_connectors = [

            connector_id

            for (
                connector_id,
                available_until
            )
            in connector_available_until.items()

            if available_until <= arrival_time
        ]


        # =================================================
        # VEHICLE CAN BE SERVED
        # =================================================

        if free_connectors:

            # -------------------------------------------------
            # COUNT CURRENTLY OCCUPIED CONNECTORS
            # ON EACH PHYSICAL CHARGER
            # -------------------------------------------------

            occupied_per_charger = {
                charger_id: 0

                for charger_id in range(
                    1,
                    charger_count + 1
                )
            }


            for (
                connector_id,
                available_until
            ) in connector_available_until.items():

                if available_until > arrival_time:

                    charger_id = (
                        connector_to_charger[
                            connector_id
                        ]
                    )

                    occupied_per_charger[
                        charger_id
                    ] += 1


            # -------------------------------------------------
            # SELECT CONNECTOR
            #
            # Prefer a connector belonging to a charger
            # with fewer active vehicles.
            #
            # Example:
            # if Charger 1 already has one car and
            # Charger 2 is empty, use Charger 2 first.
            # -------------------------------------------------

            selected_connector = min(
                free_connectors,
                key=lambda connector_id: (
                    occupied_per_charger[
                        connector_to_charger[
                            connector_id
                        ]
                    ],
                    connector_available_until[
                        connector_id
                    ],
                    connector_id
                )
            )


            selected_charger = (
                connector_to_charger[
                    selected_connector
                ]
            )


            end_time = (
                arrival_time
                +
                pd.Timedelta(
                    minutes=
                        duration_minutes
                )
            )


            connector_available_until[
                selected_connector
            ] = end_time


            # -------------------------------------------------
            # SAVE SERVED SESSION
            # -------------------------------------------------

            served_sessions.append(
                {
                    "session_id":
                        f"S{session_number:07d}",

                    "arrival_id":
                        arrival[
                            "arrival_id"
                        ],

                    "site_id":
                        site_id,

                    "charger_id":
                        selected_charger,

                    "connector_id":
                        selected_connector,

                    "start_time":
                        arrival_time,

                    "end_time":
                        end_time,

                    "duration_minutes":
                        duration_minutes,

                    "requested_power_kw":
                        arrival[
                            "requested_power_kw"
                        ],

                    "requested_energy_kwh":
                        arrival[
                            "requested_energy_kwh"
                        ],

                    "charger_power_kw":
                        charger_power_kw,

                    "max_connector_power_kw":
                        max_connector_power_kw,
                }
            )


            demand_attempts.append(
                {
                    "arrival_id":
                        arrival[
                            "arrival_id"
                        ],

                    "site_id":
                        site_id,

                    "start_time":
                        arrival_time,

                    "served":
                        1,

                    "congestion_event":
                        0,

                    "requested_energy_kwh":
                        arrival[
                            "requested_energy_kwh"
                        ],
                }
            )


            session_number += 1


        # =================================================
        # ALL CONNECTORS ARE BUSY
        # =================================================

        else:

            demand_attempts.append(
                {
                    "arrival_id":
                        arrival[
                            "arrival_id"
                        ],

                    "site_id":
                        site_id,

                    "start_time":
                        arrival_time,

                    "served":
                        0,

                    "congestion_event":
                        1,

                    "requested_energy_kwh":
                        arrival[
                            "requested_energy_kwh"
                        ],
                }
            )


# =========================================================
# CREATE DATAFRAMES
# =========================================================

sessions_df = pd.DataFrame(
    served_sessions
)

demand_df = pd.DataFrame(
    demand_attempts
)


if sessions_df.empty:

    raise ValueError(
        "No served charging sessions were generated."
    )


# =========================================================
# CONVERT TIME COLUMNS
# =========================================================

sessions_df[
    "start_time"
] = pd.to_datetime(
    sessions_df[
        "start_time"
    ]
)


sessions_df[
    "end_time"
] = pd.to_datetime(
    sessions_df[
        "end_time"
    ]
)


# =========================================================
# PREPARE ACTUAL POWER COLUMN
# =========================================================

sessions_df[
    "average_power_kw"
] = 0.0


# =========================================================
# SIMULATION TIME RANGE
#
# IMPORTANT FIX:
#
# A session starting late on June 30 may finish on July 1.
# Therefore, the simulation must continue until the
# LAST ACTUAL SESSION finishes.
# =========================================================

simulation_start = start_date


simulation_end = (
    sessions_df[
        "end_time"
    ]
    .max()
    .ceil("min")
    +
    pd.Timedelta(
        minutes=1
    )
)


total_minutes = int(
    (
        simulation_end
        -
        simulation_start
    )
    .total_seconds()
    / 60
)


print(
    "\nPower simulation from:",
    simulation_start
)

print(
    "Power simulation to:",
    simulation_end
)

print(
    "Total simulated minutes:",
    total_minutes
)


# =========================================================
# SHARED PHYSICAL CHARGER POWER
#
# If two connectors of the same charger are being used,
# their combined power cannot exceed charger_power_kw.
# =========================================================

for (
    site_id,
    charger_id
), charger_sessions in sessions_df.groupby(
    [
        "site_id",
        "charger_id"
    ]
):

    charger_sessions = (
        charger_sessions
        .copy()
    )


    charger_power_kw = float(
        charger_sessions[
            "charger_power_kw"
        ]
        .iloc[0]
    )


    # -----------------------------------------------------
    # ARRAY SHOWING CHANGES IN REQUESTED POWER
    # -----------------------------------------------------

    power_change = np.zeros(
        total_minutes + 1,
        dtype=float
    )


    # -----------------------------------------------------
    # SESSION START MINUTES
    # -----------------------------------------------------

    start_indices = (
        (
            charger_sessions[
                "start_time"
            ]
            -
            simulation_start
        )
        .dt
        .total_seconds()
        .div(60)
        .astype(int)
        .to_numpy()
    )


    # -----------------------------------------------------
    # SESSION END MINUTES
    # -----------------------------------------------------

    end_indices = (
        (
            charger_sessions[
                "end_time"
            ]
            -
            simulation_start
        )
        .dt
        .total_seconds()
        .div(60)
        .astype(int)
        .to_numpy()
    )


    requested_powers = (
        charger_sessions[
            "requested_power_kw"
        ]
        .to_numpy(
            dtype=float
        )
    )


    # -----------------------------------------------------
    # EXTRA SAFETY CHECK
    # -----------------------------------------------------

    if end_indices.max() > total_minutes:

        raise ValueError(
            f"Session end exceeds simulation window "
            f"for site {site_id}, charger {charger_id}."
        )


    # -----------------------------------------------------
    # ADD START / END POWER EVENTS
    # -----------------------------------------------------

    for (
        start_index,
        end_index,
        requested_power
    ) in zip(
        start_indices,
        end_indices,
        requested_powers
    ):

        power_change[
            start_index
        ] += requested_power


        power_change[
            end_index
        ] -= requested_power


    # -----------------------------------------------------
    # TOTAL REQUESTED POWER AT EACH MINUTE
    # -----------------------------------------------------

    total_requested_power = np.cumsum(
        power_change[:-1]
    )


    # -----------------------------------------------------
    # SHARING FACTOR
    #
    # Example:
    #
    # Car A asks for 200 kW
    # Car B asks for 200 kW
    #
    # Total request = 400 kW
    # Charger limit = 300 kW
    #
    # Sharing factor = 300 / 400 = 0.75
    # -----------------------------------------------------

    sharing_factor = np.ones(
        total_minutes,
        dtype=float
    )


    overloaded = (
        total_requested_power
        >
        charger_power_kw
    )


    sharing_factor[
        overloaded
    ] = (
        charger_power_kw
        /
        total_requested_power[
            overloaded
        ]
    )


    # -----------------------------------------------------
    # CUMULATIVE SHARING FACTOR
    #
    # Allows fast calculation for every session.
    # -----------------------------------------------------

    cumulative_factor = np.concatenate(
        [
            [0.0],
            np.cumsum(
                sharing_factor
            )
        ]
    )


    # -----------------------------------------------------
    # ACTUAL AVERAGE POWER FOR EACH SESSION
    # -----------------------------------------------------

    actual_average_powers = []


    for (
        start_index,
        end_index,
        requested_power
    ) in zip(
        start_indices,
        end_indices,
        requested_powers
    ):

        session_minutes = (
            end_index
            -
            start_index
        )


        if session_minutes <= 0:

            actual_average_powers.append(
                0.0
            )

            continue


        average_sharing_factor = (
            cumulative_factor[
                end_index
            ]
            -
            cumulative_factor[
                start_index
            ]
        ) / session_minutes


        actual_average_power = (
            requested_power
            *
            average_sharing_factor
        )


        # One connector cannot exceed its own limit
        max_connector_power = float(
            charger_sessions[
                "max_connector_power_kw"
            ]
            .iloc[0]
        )


        actual_average_power = min(
            actual_average_power,
            max_connector_power
        )


        actual_average_powers.append(
            actual_average_power
        )


    sessions_df.loc[
        charger_sessions.index,
        "average_power_kw"
    ] = actual_average_powers


# =========================================================
# ACTUAL DELIVERED ENERGY
# =========================================================

sessions_df[
    "energy_kwh"
] = (
    sessions_df[
        "average_power_kw"
    ]
    *
    (
        sessions_df[
            "duration_minutes"
        ]
        / 60
    )
)


# =========================================================
# ENERGY SHORTFALL CAUSED BY POWER SHARING
# =========================================================

sessions_df[
    "energy_shortfall_kwh"
] = (
    sessions_df[
        "requested_energy_kwh"
    ]
    -
    sessions_df[
        "energy_kwh"
    ]
)


sessions_df[
    "energy_shortfall_kwh"
] = (
    sessions_df[
        "energy_shortfall_kwh"
    ]
    .clip(
        lower=0
    )
)


# =========================================================
# ROUND NUMERIC RESULTS
# =========================================================

round_columns = [
    "requested_power_kw",
    "requested_energy_kwh",
    "average_power_kw",
    "energy_kwh",
    "energy_shortfall_kwh",
]


sessions_df[
    round_columns
] = (
    sessions_df[
        round_columns
    ]
    .round(2)
)


# =========================================================
# SAVE DATA
# =========================================================

sessions_df.to_csv(
    SESSION_OUTPUT,
    index=False
)


demand_df.to_csv(
    DEMAND_OUTPUT,
    index=False
)


# =========================================================
# OVERALL SUMMARY
# =========================================================

total_arrivals = len(
    demand_df
)

served_count = int(
    demand_df[
        "served"
    ]
    .sum()
)

congestion_count = int(
    demand_df[
        "congestion_event"
    ]
    .sum()
)


congestion_rate = (
    congestion_count
    /
    total_arrivals
    *
    100
)


print(
    "\n--- CHARGING SITE SIMULATION COMPLETE ---"
)

print(
    "Total vehicle arrivals:",
    total_arrivals
)

print(
    "Served sessions:",
    served_count
)

print(
    "Congestion events:",
    congestion_count
)

print(
    "Congestion rate:",
    round(
        congestion_rate,
        2
    ),
    "%"
)


# =========================================================
# PHYSICAL SITE STRUCTURE
# =========================================================

print(
    "\n--- PHYSICAL SITE STRUCTURE ---"
)

print(
    sites[
        [
            "site_id",
            "city",
            "charger_count",
            "connectors_per_charger",
            "connector_count",
            "charger_power_kw",
            "max_connector_power_kw",
            "grid_capacity_kw"
        ]
    ]
    .to_string(
        index=False
    )
)


# =========================================================
# POWER SHARING SUMMARY
# =========================================================

total_requested_energy = (
    sessions_df[
        "requested_energy_kwh"
    ]
    .sum()
)

total_delivered_energy = (
    sessions_df[
        "energy_kwh"
    ]
    .sum()
)

total_energy_shortfall = (
    sessions_df[
        "energy_shortfall_kwh"
    ]
    .sum()
)


print(
    "\n--- POWER SHARING SUMMARY ---"
)

print(
    "Requested energy:",
    round(
        total_requested_energy,
        2
    ),
    "kWh"
)

print(
    "Delivered energy:",
    round(
        total_delivered_energy,
        2
    ),
    "kWh"
)

print(
    "Energy not delivered because of shared charger power:",
    round(
        total_energy_shortfall,
        2
    ),
    "kWh"
)


# =========================================================
# SITE LEVEL SUMMARY
# =========================================================

print(
    "\n--- SERVED SESSIONS PER SITE ---"
)

print(
    sessions_df
    .groupby(
        "site_id"
    )
    .size()
)


print(
    "\n--- CONGESTION EVENTS PER SITE ---"
)

site_congestion = (
    demand_df[
        demand_df[
            "congestion_event"
        ] == 1
    ]
    .groupby(
        "site_id"
    )
    .size()
)


print(
    site_congestion
)