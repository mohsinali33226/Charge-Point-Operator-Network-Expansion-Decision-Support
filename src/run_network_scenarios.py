"""Evaluate every configuration generated for a CPO charging network.

Reads:
  data/raw/site_master.csv
  data/raw/demand_attempts.csv
  data/raw/charging_sessions.csv
  data/processed/network_expansion_scenarios.csv

Writes:
  data/processed/network_scenario_simulation_results.csv
  data/processed/network_scenario_incremental_impact.csv

Run from the project folder:
  python src/run_network_scenarios.py

Uses the established charging physics from simulate_expansion.py instead of
creating a second, inconsistent charging model. No fixed charger counts or
network-size limit are imposed by this runner. Performance at very large
networks is NOT validated. For now, simulate_expansion.load_data still
requires exactly two simultaneous connectors per physical charger.

All results are synthetic screening estimates; not grid offers, true queues,
reliability forecasts, or investment recommendations.
"""

import argparse
from pathlib import Path

import pandas as pd

from simulate_expansion import load_data, simulate_site

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ROOT / "data" / "processed" / "network_expansion_scenarios.csv"
DEFAULT_RESULTS = ROOT / "data" / "processed" / "network_scenario_simulation_results.csv"
DEFAULT_IMPACT = ROOT / "data" / "processed" / "network_scenario_incremental_impact.csv"

REQUIRED_SCENARIO_COLUMNS = {
    "scenario_id", "site_id", "scenario_type", "current_chargers",
    "scenario_chargers", "scenario_connectors", "scenario_grid_kw",
    "connectors_per_charger", "charger_power_kw", "max_connector_power_kw"
}


def read_scenarios(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Scenario file not found: {path}\n"
            "First run: python src\\generate_network_scenarios.py"
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = REQUIRED_SCENARIO_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Scenario catalog is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("The scenario catalog is empty.")
    if df[list(REQUIRED_SCENARIO_COLUMNS)].isna().any().any():
        raise ValueError("The scenario catalog has missing required values.")
    if df["scenario_id"].duplicated().any():
        raise ValueError("The scenario catalog contains duplicate scenario IDs.")

    # Do not silently turn 4.5 chargers into 4 chargers.
    for field in ["current_chargers", "scenario_chargers", "scenario_connectors", "connectors_per_charger"]:
        values = pd.to_numeric(df[field], errors="raise")
        if (values <= 0).any() or (values != values.round()).any():
            raise ValueError(f"{field} must contain positive whole numbers.")
        df[field] = values.astype(int)

    for field in ["scenario_grid_kw", "charger_power_kw", "max_connector_power_kw"]:
        values = pd.to_numeric(df[field], errors="raise")
        if (values <= 0).any():
            raise ValueError(f"{field} must contain positive numbers.")
        df[field] = values.astype(float)

    if (df["scenario_connectors"] != df["scenario_chargers"] * df["connectors_per_charger"]).any():
        raise ValueError("One or more scenarios have inconsistent connector counts.")

    if (~df["scenario_type"].isin(["CURRENT", "SAME_GRID", "GRID_UPGRADE"])).any():
        raise ValueError("Unknown scenario_type found. Regenerate the scenario catalog.")

    counts = df[df["scenario_type"] == "CURRENT"].groupby("site_id").size()
    if not set(df["site_id"]).issubset(set(counts[counts == 1].index)):
        raise ValueError("Each scenario site must have exactly one CURRENT baseline scenario.")

    order = {"CURRENT": 0, "SAME_GRID": 1, "GRID_UPGRADE": 2}
    df["_order"] = df["scenario_type"].map(order)
    return df.sort_values(["site_id", "_order", "scenario_chargers", "scenario_grid_kw"])


def validate_against_master(case, site):
    site_id = str(case["site_id"])
    checks = [
        ("current_chargers", "charger_count"),
        ("connectors_per_charger", "connectors_per_charger"),
        ("charger_power_kw", "charger_power_kw"),
        ("max_connector_power_kw", "max_connector_power_kw"),
    ]
    for scenario_key, site_key in checks:
        if abs(float(case[scenario_key]) - float(site[site_key])) > 1e-6:
            raise ValueError(
                f"{site_id}: {scenario_key} in the scenario catalog does not "
                "match site_master.csv. Rerun generate_network_scenarios.py."
            )
    if case["scenario_type"] == "CURRENT":
        if abs(float(case["scenario_grid_kw"]) - float(site["grid_capacity_kw"])) > 1e-6:
            raise ValueError(f"{site_id}: baseline grid differs from site_master.csv.")


def summarize_incremental(results):
    passed = results[results["simulation_status"] == "SIMULATED"].copy()
    if passed.empty:
        return pd.DataFrame()
    baseline = passed[passed["scenario_type"] == "CURRENT"]
    comparison = passed.merge(
        baseline[[
            "site_id", "cars_rejected", "energy_delivered_kwh", "grid_capacity_kw",
            "scenario_chargers", "total_unserved_energy_kwh"
        ]].rename(columns={
            "cars_rejected": "baseline_cars_rejected",
            "energy_delivered_kwh": "baseline_energy_delivered_kwh",
            "grid_capacity_kw": "baseline_grid_kw",
            "scenario_chargers": "baseline_chargers",
            "total_unserved_energy_kwh": "baseline_unserved_energy_kwh"
        }),
        on="site_id", how="left", validate="many_to_one"
    )
    comparison["additional_chargers"] = (
        comparison["scenario_chargers"] - comparison["baseline_chargers"]
    )
    comparison["additional_grid_kw"] = (
        comparison["grid_capacity_kw"] - comparison["baseline_grid_kw"]
    ).round(2)
    comparison["additional_cars_served"] = (
        comparison["baseline_cars_rejected"] - comparison["cars_rejected"]
    )
    comparison["additional_energy_delivered_kwh"] = (
        comparison["energy_delivered_kwh"] - comparison["baseline_energy_delivered_kwh"]
    ).round(1)
    comparison["reduction_in_unserved_energy_kwh"] = (
        comparison["baseline_unserved_energy_kwh"] - comparison["total_unserved_energy_kwh"]
    ).round(1)
    return comparison[[
        "scenario_id", "site_id", "city", "scenario_type", "scenario_chargers",
        "scenario_connectors", "grid_capacity_kw", "additional_chargers",
        "additional_grid_kw", "additional_cars_served", "additional_energy_delivered_kwh",
        "reduction_in_unserved_energy_kwh", "cars_rejected", "rejected_pct",
        "minutes_at_grid_limit"
    ]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--impact", type=Path, default=DEFAULT_IMPACT)
    parser.add_argument("--site-id", default=None, help="Optional: simulate just one site for debugging")
    args = parser.parse_args()

    scenario_df = read_scenarios(args.scenarios)
    if args.site_id:
        scenario_df = scenario_df[scenario_df["site_id"] == args.site_id]
        if scenario_df.empty:
            raise ValueError(f"No scenarios for site ID {args.site_id!r}.")

    arrivals, sites = load_data()
    master = {str(site["site_id"]): site for _, site in sites.iterrows()}
    events_by_site = {str(k): group for k, group in arrivals.groupby("site_id", sort=False)}
    unknown = set(scenario_df["site_id"].astype(str)) - set(master)
    if unknown:
        raise ValueError(f"Unknown site IDs in scenario catalog: {sorted(unknown)}")

    results = []
    print("\n--- CONFIGURABLE NETWORK EXPANSION SIMULATION ---", flush=True)
    print(f"Sites: {scenario_df['site_id'].nunique()} | Scenarios: {len(scenario_df)}", flush=True)
    print("Each scenario replays the same arrivals for its site.\n", flush=True)

    for number, (_, case) in enumerate(scenario_df.iterrows(), 1):
        site_id = str(case["site_id"])
        site = master[site_id]
        validate_against_master(case, site)
        summary = {
            "scenario_id": case["scenario_id"], "site_id": site_id,
            "city": str(site["city"]), "scenario_type": case["scenario_type"],
            "scenario_chargers": int(case["scenario_chargers"]),
            "scenario_connectors": int(case["scenario_connectors"]),
            "grid_capacity_kw": float(case["scenario_grid_kw"])
        }
        print(
            f"[{number}/{len(scenario_df)}] {site['city']}: "
            f"{summary['scenario_chargers']} chargers, "
            f"{summary['grid_capacity_kw']:,.0f} kW ({case['scenario_type']})",
            flush=True
        )
        site_arrivals = events_by_site.get(site_id)
        if site_arrivals is None or site_arrivals.empty:
            summary["simulation_status"] = "NO_DEMAND_DATA"
            results.append(summary)
            continue

        simulated_site = site.copy()
        simulated_site["grid_capacity_kw"] = float(case["scenario_grid_kw"])
        outcome = simulate_site(site_arrivals, simulated_site, int(case["scenario_chargers"]))
        # Keep scenario metadata and computed outcome in a single record.
        summary.update(outcome)
        summary["simulation_status"] = "SIMULATED"
        summary["total_requested_energy_kwh"] = round(
            float(site_arrivals["requested_energy_kwh"].sum()), 1
        )
        summary["total_unserved_energy_kwh"] = round(
            max(0.0, summary["total_requested_energy_kwh"] - summary["energy_delivered_kwh"]), 1
        )
        results.append(summary)

    details = pd.DataFrame(results)
    impact = summarize_incremental(details)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.impact.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.results, index=False)
    impact.to_csv(args.impact, index=False)

    print("\n--- NETWORK SCENARIO IMPACT ---")
    if impact.empty:
        print("No simulated sites had demand data.")
    else:
        print(impact[[
            "city", "scenario_type", "scenario_chargers", "grid_capacity_kw",
            "additional_cars_served", "additional_energy_delivered_kwh",
            "minutes_at_grid_limit"
        ]].to_string(index=False))
    print(f"\nSaved detailed results: {args.results}")
    print(f"Saved comparison with baseline: {args.impact}")
    print("These are synthetic historical comparisons, NOT future-demand or financial forecasts.")
    print("Existing simulator assumes 2 simultaneous connectors per physical charger.")


if __name__ == "__main__":
    main()
