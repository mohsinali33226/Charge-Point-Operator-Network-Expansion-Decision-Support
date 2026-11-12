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

SITE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "site_master.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

sessions = pd.read_csv(
    SESSION_FILE
)

sites = pd.read_csv(
    SITE_FILE,
    skipinitialspace=True
)

sites.columns = (
    sites.columns
    .str.strip()
)


sessions["start_time"] = pd.to_datetime(
    sessions["start_time"]
)

sessions["end_time"] = pd.to_datetime(
    sessions["end_time"]
)


# =========================================================
# TOLERANCE
#
# Small tolerance because CSV values are rounded.
# =========================================================

POWER_TOLERANCE_KW = 0.2


print(
    "\n========================================"
)

print(
    "CHARGING SESSION PHYSICS VALIDATION"
)

print(
    "========================================"
)

print(
    "Sessions checked:",
    len(sessions)
)


# =========================================================
# CHECK 0
# CONNECTOR -> CHARGER MAPPING
#
# Example:
#
# Charger 1 -> connectors 1, 2
# Charger 2 -> connectors 3, 4
# =========================================================

site_structure = sites.set_index(
    "site_id"
)


mapping_errors = 0


for _, row in sessions.iterrows():

    site_id = row["site_id"]

    connectors_per_charger = int(
        site_structure.loc[
            site_id,
            "connectors_per_charger"
        ]
    )


    expected_charger_id = (
        (
            int(row["connector_id"]) - 1
        )
        //
        connectors_per_charger
        +
        1
    )


    if (
        int(row["charger_id"])
        != expected_charger_id
    ):

        mapping_errors += 1


print(
    "\n--- CHECK 0: CONNECTOR / CHARGER MAPPING ---"
)

print(
    "Mapping errors:",
    mapping_errors
)


# =========================================================
# CHECK 1
# ONE CONNECTOR CANNOT CHARGE TWO CARS AT ONCE
# =========================================================

connector_overlap_count = 0
overlap_examples = []


for (
    site_id,
    connector_id
), group in sessions.groupby(
    [
        "site_id",
        "connector_id"
    ]
):

    group = (
        group
        .sort_values(
            "start_time"
        )
        .reset_index(
            drop=True
        )
    )


    previous_end = None
    previous_session = None


    for _, row in group.iterrows():

        if (
            previous_end is not None
            and row["start_time"] < previous_end
        ):

            connector_overlap_count += 1


            if len(
                overlap_examples
            ) < 5:

                overlap_examples.append(
                    {
                        "site_id":
                            site_id,

                        "connector_id":
                            connector_id,

                        "previous_session":
                            previous_session,

                        "current_session":
                            row["session_id"],

                        "current_start":
                            row["start_time"],

                        "previous_end":
                            previous_end
                    }
                )


        if (
            previous_end is None
            or row["end_time"]
            > previous_end
        ):

            previous_end = (
                row["end_time"]
            )

            previous_session = (
                row["session_id"]
            )


print(
    "\n--- CHECK 1: CONNECTOR OVERLAP ---"
)

print(
    "Overlapping sessions:",
    connector_overlap_count
)


if overlap_examples:

    print(
        "\nExamples:"
    )

    print(
        pd.DataFrame(
            overlap_examples
        ).to_string(
            index=False
        )
    )


# =========================================================
# CHECK 2
# ONE CAR CANNOT EXCEED CONNECTOR POWER
# =========================================================

connector_power_violations = sessions[
    (
        sessions[
            "requested_power_kw"
        ]
        >
        sessions[
            "max_connector_power_kw"
        ]
        +
        POWER_TOLERANCE_KW
    )
    |
    (
        sessions[
            "average_power_kw"
        ]
        >
        sessions[
            "max_connector_power_kw"
        ]
        +
        POWER_TOLERANCE_KW
    )
]


print(
    "\n--- CHECK 2: CONNECTOR POWER LIMIT ---"
)

print(
    "Sessions exceeding connector power:",
    len(
        connector_power_violations
    )
)


if not connector_power_violations.empty:

    print(
        "\nExamples:"
    )

    print(
        connector_power_violations[
            [
                "session_id",
                "site_id",
                "charger_id",
                "connector_id",
                "requested_power_kw",
                "average_power_kw",
                "max_connector_power_kw"
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )


# =========================================================
# CHECK 3
# PHYSICAL CHARGER POWER SHARING
#
# If two connectors of a charger are active:
#
# Example:
#
# Car A requests 200 kW
# Car B requests 200 kW
#
# Total request = 400 kW
# Charger limit = 300 kW
#
# Actual total charger power must be capped at 300 kW.
# =========================================================

charger_power_violations = 0

maximum_actual_power_ratio = 0

hours_with_power_sharing = 0

recomputed_power_errors = 0

largest_power_difference = 0


# =========================================================
# CHECK EACH PHYSICAL CHARGER
# =========================================================

for (
    site_id,
    charger_id
), charger_sessions in sessions.groupby(
    [
        "site_id",
        "charger_id"
    ]
):

    charger_sessions = (
        charger_sessions
        .copy()
        .sort_values(
            "start_time"
        )
    )


    charger_power_kw = float(
        charger_sessions[
            "charger_power_kw"
        ]
        .iloc[0]
    )


    # -----------------------------------------------------
    # Use only the time range needed for this charger.
    # -----------------------------------------------------

    charger_start = (
        charger_sessions[
            "start_time"
        ]
        .min()
        .floor("min")
    )


    charger_end = (
        charger_sessions[
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
            charger_end
            -
            charger_start
        )
        .total_seconds()
        /
        60
    )


    power_change = np.zeros(
        total_minutes + 1,
        dtype=float
    )


    start_indices = (
        (
            charger_sessions[
                "start_time"
            ]
            -
            charger_start
        )
        .dt
        .total_seconds()
        .div(60)
        .astype(int)
        .to_numpy()
    )


    end_indices = (
        (
            charger_sessions[
                "end_time"
            ]
            -
            charger_start
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
    # Build total requested charger power minute-by-minute
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


    total_requested_power = np.cumsum(
        power_change[:-1]
    )


    # -----------------------------------------------------
    # Actual physical charger power after sharing
    # -----------------------------------------------------

    actual_total_power = np.minimum(
        total_requested_power,
        charger_power_kw
    )


    # -----------------------------------------------------
    # CHECK CHARGER LIMIT
    # -----------------------------------------------------

    violation_minutes = int(
        np.sum(
            actual_total_power
            >
            charger_power_kw
            +
            POWER_TOLERANCE_KW
        )
    )


    charger_power_violations += (
        violation_minutes
    )


    if len(
        actual_total_power
    ) > 0:

        charger_ratio = (
            actual_total_power.max()
            /
            charger_power_kw
        )

        maximum_actual_power_ratio = max(
            maximum_actual_power_ratio,
            charger_ratio
        )


    # -----------------------------------------------------
    # COUNT MINUTES WHERE SHARING WAS REQUIRED
    # -----------------------------------------------------

    sharing_minutes = int(
        np.sum(
            total_requested_power
            >
            charger_power_kw
        )
    )


    hours_with_power_sharing += (
        sharing_minutes
        /
        60
    )


    # -----------------------------------------------------
    # RECREATE THE SAME SHARING FACTOR USED BY GENERATOR
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


    cumulative_factor = np.concatenate(
        [
            [0.0],
            np.cumsum(
                sharing_factor
            )
        ]
    )


    # -----------------------------------------------------
    # CHECK SAVED SESSION POWER AGAINST RECOMPUTED POWER
    # -----------------------------------------------------

    saved_average_power = (
        charger_sessions[
            "average_power_kw"
        ]
        .to_numpy(
            dtype=float
        )
    )


    for (
        start_index,
        end_index,
        requested_power,
        saved_power
    ) in zip(
        start_indices,
        end_indices,
        requested_powers,
        saved_average_power
    ):

        duration_minutes = (
            end_index
            -
            start_index
        )


        if duration_minutes <= 0:

            continue


        average_factor = (
            cumulative_factor[
                end_index
            ]
            -
            cumulative_factor[
                start_index
            ]
        ) / duration_minutes


        recomputed_power = (
            requested_power
            *
            average_factor
        )


        difference = abs(
            recomputed_power
            -
            saved_power
        )


        largest_power_difference = max(
            largest_power_difference,
            difference
        )


        if (
            difference
            >
            POWER_TOLERANCE_KW
        ):

            recomputed_power_errors += 1


print(
    "\n--- CHECK 3: PHYSICAL CHARGER POWER ---"
)

print(
    "Minutes above physical charger limit:",
    charger_power_violations
)

print(
    "Highest charger utilisation:",
    round(
        maximum_actual_power_ratio
        *
        100,
        2
    ),
    "%"
)

print(
    "Approx. hours where power sharing was needed:",
    round(
        hours_with_power_sharing,
        2
    )
)

print(
    "Sessions inconsistent with sharing calculation:",
    recomputed_power_errors
)

print(
    "Largest saved/recomputed power difference:",
    round(
        largest_power_difference,
        3
    ),
    "kW"
)


# =========================================================
# FINAL RESULT
# =========================================================

total_errors = (
    mapping_errors
    +
    connector_overlap_count
    +
    len(
        connector_power_violations
    )
    +
    charger_power_violations
    +
    recomputed_power_errors
)


print(
    "\n========================================"
)

print(
    "FINAL VALIDATION RESULT"
)

print(
    "========================================"
)


if total_errors == 0:

    print(
        "PASS"
    )

    print(
        "All physical charging rules are satisfied."
    )

else:

    print(
        "FAIL"
    )

    print(
        "Total validation errors:",
        total_errors
    )