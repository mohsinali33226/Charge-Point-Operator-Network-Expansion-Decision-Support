"""Link hourly P95 demand forecasts to configurable CPO expansion scenarios.

Engineering screening only: NOT a queue simulation, long-term forecast, grid
feasibility statement, or investment recommendation. Current demo forecasts
are June 2025 historical holdout predictions on synthetic data.

Run from project root:
    python src\build_forecast_capacity_screen.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FORECAST = DATA / "processed" / "calibrated_p95_forecast_results.csv"
SCENARIOS = DATA / "processed" / "registry_network_expansion_scenarios.csv"
SESSIONS = DATA / "raw" / "charging_sessions.csv"
HOURLY_OUT = DATA / "processed" / "forecast_capacity_hourly_screen.csv"
SUMMARY_OUT = DATA / "processed" / "forecast_capacity_scenario_summary.csv"


def load(path, columns):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, dtype={"site_id": str, "scenario_id": str})
    df.columns = df.columns.str.strip()
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if df.empty:
        raise ValueError(f"{path.name} is empty")
    return df


def number(df, column, strictly_positive=False):
    df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[column].isna().any() or (df[column] < 0).any():
        raise ValueError(f"Invalid or missing values in {column}")
    if strictly_positive and (df[column] == 0).any():
        raise ValueError(f"{column} must be greater than zero")


def main():
    forecasts = load(FORECAST, [
        "site_id", "hour", "calibrated_arrival_p95", "calibrated_energy_p95_kwh"
    ])
    scenarios = load(SCENARIOS, [
        "scenario_id", "site_id", "city", "scenario_type",
        "scenario_chargers", "scenario_simultaneous_positions", "scenario_grid_kw"
    ])
    sessions = load(SESSIONS, [
        "site_id", "start_time", "duration_minutes"
    ])

    forecasts["hour"] = pd.to_datetime(forecasts["hour"], errors="coerce")
    sessions["start_time"] = pd.to_datetime(sessions["start_time"], errors="coerce")
    if forecasts["hour"].isna().any() or sessions["start_time"].isna().any():
        raise ValueError("Invalid timestamps in forecasts or charging sessions")
    if forecasts.duplicated(["site_id", "hour"]).any():
        raise ValueError("Duplicate site-hour forecasts found")
    if scenarios.duplicated("scenario_id").any():
        raise ValueError("Duplicate scenario_id found")
    if scenarios["scenario_id"].isna().any():
        raise ValueError("Missing scenario_id")

    for column in ["calibrated_arrival_p95", "calibrated_energy_p95_kwh"]:
        number(forecasts, column)
    for column in ["scenario_chargers", "scenario_simultaneous_positions", "scenario_grid_kw"]:
        number(scenarios, column, strictly_positive=True)
    number(sessions, "duration_minutes", strictly_positive=True)

    # Use ONLY earlier charging sessions to estimate typical connector occupancy.
    # Do not use June holdout sessions to estimate June's duration assumption.
    first_forecast_hour = forecasts["hour"].min()
    prior_sessions = sessions[sessions["start_time"] < first_forecast_hour]
    durations = (
        prior_sessions.groupby("site_id", as_index=False)["duration_minutes"]
        .mean().rename(columns={"duration_minutes": "historical_avg_occupied_min"})
    )

    if set(scenarios["site_id"]) != set(forecasts["site_id"]):
        raise ValueError("Forecast sites and scenario sites differ; check site_id")
    if set(scenarios["site_id"]) - set(durations["site_id"]):
        raise ValueError("Some sites have no earlier sessions to estimate occupancy")

    current_counts = scenarios[scenarios["scenario_type"] == "CURRENT"].groupby("site_id").size()
    if (set(current_counts.index) != set(scenarios["site_id"])) or (current_counts != 1).any():
        raise ValueError("Exactly one CURRENT scenario is needed per site")

    # For each scenario, evaluate each predicted hour at that same site.
    hourly = scenarios[[
        "scenario_id", "site_id", "city", "scenario_type", "scenario_chargers",
        "scenario_simultaneous_positions", "scenario_grid_kw"
    ]].merge(
        forecasts[["site_id", "hour", "calibrated_arrival_p95", "calibrated_energy_p95_kwh"]],
        on="site_id", how="inner", validate="many_to_many"
    ).merge(durations, on="site_id", validate="many_to_one")

    if len(hourly) != sum(
        len(forecasts[forecasts.site_id == site]) * len(scenarios[scenarios.site_id == site])
        for site in scenarios.site_id.unique()
    ):
        raise ValueError("Forecast/scenario join did not preserve all expected rows")

    # Little's-law screening approximation: average simultaneous occupancy
    # ~= arrival rate (cars/hour) x mean occupied time (hours).
    # This is NOT a prediction of rejection probability or an Erlang/queue model.
    hourly["indicative_busy_hour_occupied_positions"] = (
        hourly["calibrated_arrival_p95"] * hourly["historical_avg_occupied_min"] / 60
    ).round(3)
    hourly["indicative_position_pressure_ratio"] = (
        hourly["indicative_busy_hour_occupied_positions"]
        / hourly["scenario_simultaneous_positions"]
    ).round(3)
    hourly["position_pressure_flag"] = (
        hourly["indicative_busy_hour_occupied_positions"]
        > hourly["scenario_simultaneous_positions"]
    )

    # Even if energy (kWh) over 1 hour exceeds the numeric AC grid limit (kW),
    # site-side feasibility requires a closer look. AC/DC losses, other site
    # loads, instantaneous peaks and time profiles are NOT handled here.
    hourly["grid_energy_lower_bound_flag"] = (
        hourly["calibrated_energy_p95_kwh"] > hourly["scenario_grid_kw"]
    )
    hourly["forecast_scope"] = "HISTORICAL_HOLDOUT_P95_SYNTHETIC"
    hourly["decision_status"] = "ENGINEERING_SCREEN_ONLY"

    summary = (
        hourly.groupby([
            "scenario_id", "site_id", "city", "scenario_type",
            "scenario_chargers", "scenario_simultaneous_positions", "scenario_grid_kw"
        ], as_index=False)
        .agg(
            forecast_hours=("hour", "size"),
            position_pressure_hours=("position_pressure_flag", "sum"),
            grid_energy_lower_bound_hours=("grid_energy_lower_bound_flag", "sum"),
            maximum_indicative_position_pressure_ratio=("indicative_position_pressure_ratio", "max"),
            historical_avg_occupied_min=("historical_avg_occupied_min", "first")
        )
    )
    summary["position_pressure_pct_of_hours"] = (
        100 * summary.position_pressure_hours / summary.forecast_hours
    ).round(2)

    baselines = summary[summary.scenario_type == "CURRENT"][
        ["site_id", "position_pressure_hours"]
    ].rename(columns={"position_pressure_hours": "current_position_pressure_hours"})
    summary = summary.merge(baselines, on="site_id", validate="many_to_one")
    summary["change_in_pressure_hours_vs_current"] = (
        summary.position_pressure_hours - summary.current_position_pressure_hours
    )
    summary["decision_status"] = "ENGINEERING_SCREEN_ONLY"
    summary = summary.sort_values(["site_id", "scenario_chargers", "scenario_id"])
    hourly = hourly.sort_values(["site_id", "scenario_id", "hour"])

    HOURLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(HOURLY_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n--- FORECAST-TO-CAPACITY SCREEN ---")
    print(f"Sites: {summary.site_id.nunique()} | Scenarios: {len(summary)} | "
          f"Hourly checks: {len(hourly):,}")
    print(summary[[
        "city", "scenario_chargers", "forecast_hours", "position_pressure_hours",
        "change_in_pressure_hours_vs_current", "grid_energy_lower_bound_hours"
    ]].to_string(index=False))
    print("\nInterpretation: Fewer flagged hours can indicate more connector headroom.")
    print("This is a rough capacity SCREEN, not simulated waiting/rejections.")
    print("June 2025 is historical holdout; this is NOT a multi-year demand forecast.")
    print("P95 arrival and energy forecasts are separate quantiles, not one joint scenario.")
    print("No investment, DSO approval or actual future performance is inferred.")
    print(f"\nSaved: {SUMMARY_OUT}\nSaved: {HOURLY_OUT}")


if __name__ == "__main__":
    main()
