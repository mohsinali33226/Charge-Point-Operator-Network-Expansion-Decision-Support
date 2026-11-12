from pathlib import Path
import pandas as pd
import holidays


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HOURLY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_hourly_demand.csv"
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
    / "site_hourly_demand_with_holidays.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

hourly = pd.read_csv(HOURLY_FILE)
sites = pd.read_csv(SITE_FILE)

hourly["hour"] = pd.to_datetime(hourly["hour"])
hourly["date"] = hourly["hour"].dt.date


# ---------------------------------------------------------
# ADD STATE CODE
# ---------------------------------------------------------

hourly = hourly.merge(
    sites[
        [
            "site_id",
            "state_code"
        ]
    ],
    on="site_id",
    how="left"
)


# ---------------------------------------------------------
# CREATE HOLIDAY CALENDARS
# ---------------------------------------------------------

holiday_calendars = {}

for state_code in sites["state_code"].unique():

    holiday_calendars[state_code] = holidays.Germany(
        years=[2025],
        subdiv=state_code
    )


# ---------------------------------------------------------
# HOLIDAY CHECK
# ---------------------------------------------------------

def get_holiday_name(row):

    calendar = holiday_calendars[
        row["state_code"]
    ]

    return calendar.get(
        row["date"]
    )


hourly["holiday_name"] = hourly.apply(
    get_holiday_name,
    axis=1
)


hourly["is_public_holiday"] = (
    hourly["holiday_name"]
    .notna()
    .astype(int)
)


hourly["holiday_name"] = (
    hourly["holiday_name"]
    .fillna("None")
)


# ---------------------------------------------------------
# DAY BEFORE / AFTER HOLIDAY
# ---------------------------------------------------------

hourly["date_plus_1"] = (
    hourly["hour"]
    + pd.Timedelta(days=1)
).dt.date

hourly["date_minus_1"] = (
    hourly["hour"]
    - pd.Timedelta(days=1)
).dt.date


def check_date(calendar, date):

    return int(
        date in calendar
    )


hourly["day_before_holiday"] = hourly.apply(
    lambda row:
        check_date(
            holiday_calendars[
                row["state_code"]
            ],
            row["date_plus_1"]
        ),
    axis=1
)


hourly["day_after_holiday"] = hourly.apply(
    lambda row:
        check_date(
            holiday_calendars[
                row["state_code"]
            ],
            row["date_minus_1"]
        ),
    axis=1
)


# Remove helper columns
hourly = hourly.drop(
    columns=[
        "date_plus_1",
        "date_minus_1"
    ]
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

hourly.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print(
    "\n--- HOLIDAY FEATURES ADDED ---"
)

print(
    "Rows:",
    len(hourly)
)

print(
    "\nHoliday hours by site:"
)

print(
    hourly.groupby(
        [
            "site_id",
            "city"
        ]
    )["is_public_holiday"]
    .sum()
)


print(
    "\nDetected holidays:"
)

print(
    hourly[
        hourly["is_public_holiday"] == 1
    ][
        [
            "city",
            "date",
            "holiday_name"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "date",
            "city"
        ]
    )
    .to_string(index=False)
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)