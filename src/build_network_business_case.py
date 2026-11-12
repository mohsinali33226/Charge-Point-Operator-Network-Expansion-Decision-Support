"""Evidence-gated CPO network expansion business case (Germany, Oct 2026).

Run from the project root:
    python src\build_network_business_case.py

Optional exploratory mode (explicitly NOT investment advice):
    python src\build_network_business_case.py --demo

This version replaces the old business-case implementation that treated
illustrative, global hardware costs/margins as if they supported a site-level
invest/no-invest recommendation.

Reads:
    data/processed/network_scenario_incremental_impact.csv
    data/processed/cpo_site_readiness.csv
    data/raw/demand_attempts.csv
    data/config/cpo_market_evidence_2026.json
    data/config/business_assumptions.json  (ONLY WITH --demo)

Writes:
    data/processed/network_expansion_business_cases.csv
    data/processed/network_expansion_priority.csv

Important:
- The current arrivals are synthetic, with simplified charging physics.
- Documented inputs are NOT automatically independently verified.
- Real investment decisions require operator data, validated demand, accurate
  CAPEX/OPEX and power topology; this script never claims investment readiness.
- No arbitrary charger-count, network-size or uniform tariff assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
SCENARIOS_FILE = PROCESSED / "network_scenario_incremental_impact.csv"
READINESS_FILE = PROCESSED / "cpo_site_readiness.csv"
DEMAND_FILE = DATA / "raw" / "demand_attempts.csv"
EVIDENCE_FILE = DATA / "config" / "cpo_market_evidence_2026.json"
DEMO_FILE = DATA / "config" / "business_assumptions.json"
OUT_CASES = PROCESSED / "network_expansion_business_cases.csv"
OUT_PRIORITY = PROCESSED / "network_expansion_priority.csv"

REQUIRED_IMPACT = {
    "scenario_id", "site_id", "city", "scenario_type", "scenario_chargers",
    "scenario_connectors", "grid_capacity_kw", "additional_chargers",
    "additional_grid_kw", "additional_cars_served",
    "additional_energy_delivered_kwh", "cars_rejected", "rejected_pct",
}
REQUIRED_READINESS = {
    "site_id", "documented_inputs", "missing_inputs", "investment_readiness",
    "demand_basis", "financial_model_basis",
}
DEMO_FIELDS = {
    "eur_contribution_margin_per_extra_kwh",
    "eur_new_charger_equipment_and_installation",
    "eur_fixed_site_expansion_works",
    "eur_grid_upgrade_fixed",
    "eur_grid_upgrade_per_added_kw",
    "eur_annual_maintenance_per_new_charger",
    "eur_annual_extra_site_overhead",
    "evaluation_years", "discount_rate",
}


def read_csv(path: Path, expected: set[str], name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    frame = pd.read_csv(path, dtype={"site_id": str})
    frame.columns = frame.columns.str.strip()
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    return frame


def demand_observation_days() -> pd.DataFrame:
    history = read_csv(DEMAND_FILE, {"site_id", "start_time"}, "demand_attempts.csv")
    history["start_time"] = pd.to_datetime(history["start_time"], errors="raise")
    if history["start_time"].isna().any():
        raise ValueError("Demand contains missing timestamps")
    span = history.groupby("site_id")["start_time"].agg(["min", "max"])
    span["observation_days"] = (
        (span["max"].dt.normalize() - span["min"].dt.normalize()).dt.days + 1
    )
    return span[["observation_days"]].reset_index()


def load_demo_settings() -> dict:
    if not DEMO_FILE.exists():
        raise FileNotFoundError(f"No demo config: {DEMO_FILE}")
    with DEMO_FILE.open(encoding="utf-8") as stream:
        settings = json.load(stream)
    missing = DEMO_FIELDS - settings.keys()
    if missing:
        raise ValueError(f"Demo assumptions missing: {sorted(missing)}")
    for key in DEMO_FIELDS:
        x = settings[key]
        if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
            raise ValueError(f"Non-numeric demo assumption: {key}")
        if x < 0:
            raise ValueError(f"Negative demo assumption: {key}")
    if not isinstance(settings["evaluation_years"], int) or settings["evaluation_years"] < 1:
        raise ValueError("evaluation_years must be a positive integer")
    if settings["discount_rate"] >= 1:
        raise ValueError("discount_rate must be below 1")
    return settings


def calculate_demo_values(cases: pd.DataFrame) -> pd.DataFrame:
    """Optional calculations; NEVER allowed to drive investment recommendations."""
    m = load_demo_settings()
    expansion = cases["additional_chargers"] > 0
    upgrade = cases["additional_grid_kw"] > 0
    cases["demo_capex_eur"] = (
        cases["additional_chargers"] * m["eur_new_charger_equipment_and_installation"]
        + expansion.astype(int) * m["eur_fixed_site_expansion_works"]
        + upgrade.astype(int) * m["eur_grid_upgrade_fixed"]
        + cases["additional_grid_kw"] * m["eur_grid_upgrade_per_added_kw"]
    )
    cases["demo_annual_contribution_eur"] = (
        cases["demo_annualized_extra_energy_kwh"]
        * m["eur_contribution_margin_per_extra_kwh"]
    )
    cases["demo_annual_additional_opex_eur"] = (
        cases["additional_chargers"] * m["eur_annual_maintenance_per_new_charger"]
        + expansion.astype(int) * m["eur_annual_extra_site_overhead"]
    )
    cases["demo_annual_cashflow_eur"] = (
        cases["demo_annual_contribution_eur"] - cases["demo_annual_additional_opex_eur"]
    )
    factor = sum(
        1.0 / (1.0 + m["discount_rate"]) ** year
        for year in range(1, m["evaluation_years"] + 1)
    )
    cases["demo_npv_eur"] = (
        cases["demo_annual_cashflow_eur"] * factor - cases["demo_capex_eur"]
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true",
        help="Also show illustrative economics from business_assumptions.json. NEVER investment advice.",
    )
    args = parser.parse_args()

    if not EVIDENCE_FILE.is_file():
        raise FileNotFoundError(f"Missing market evidence reference: {EVIDENCE_FILE}")
    with EVIDENCE_FILE.open(encoding="utf-8") as stream:
        market = json.load(stream)
    required_inputs = market.get("site_specific_inputs_required", {})
    if not isinstance(required_inputs, dict) or not required_inputs:
        raise ValueError("Missing site_specific_inputs_required in market evidence JSON")

    impact = read_csv(SCENARIOS_FILE, REQUIRED_IMPACT, "scenario impact")
    readiness = read_csv(READINESS_FILE, REQUIRED_READINESS, "market readiness")
    if impact["scenario_id"].isna().any() or impact["scenario_id"].duplicated().any():
        raise ValueError("Scenario IDs must be present and unique")
    if readiness["site_id"].duplicated().any():
        raise ValueError("Each site should have one readiness record")
    if set(impact["site_id"]) != set(readiness["site_id"]):
        raise ValueError(
            "Site IDs differ between scenario simulation and readiness audit. "
            "Rerun python src\\check_market_readiness.py and python src\\run_network_scenarios.py."
        )
    if not impact["scenario_type"].isin({"CURRENT", "SAME_GRID", "GRID_UPGRADE"}).all():
        raise ValueError("Unexpected scenario_type")
    baseline = impact[impact["scenario_type"] == "CURRENT"]
    if (baseline.groupby("site_id").size() != 1).any() or len(baseline) != impact["site_id"].nunique():
        raise ValueError("Expected exactly one CURRENT baseline per site")

    cases = impact.merge(
        readiness[[
            "site_id", "documented_inputs", "missing_inputs", "investment_readiness",
            "demand_basis", "financial_model_basis",
        ]], on="site_id", how="left", validate="many_to_one"
    ).merge(
        demand_observation_days(), on="site_id", how="left", validate="many_to_one"
    )
    if cases["observation_days"].isna().any():
        raise ValueError("Missing historical demand span for some sites")
    numeric = [
        "scenario_chargers", "additional_chargers", "additional_grid_kw",
        "additional_energy_delivered_kwh", "additional_cars_served",
        "documented_inputs", "missing_inputs",
    ]
    for column in numeric:
        cases[column] = pd.to_numeric(cases[column], errors="raise")
        if cases[column].isna().any():
            raise ValueError(f"Missing numeric values: {column}")
    if (cases["additional_chargers"] < 0).any() or (cases["additional_grid_kw"] < 0).any():
        raise ValueError("Only current and expansion scenarios are supported")

    # Annualization is a DEMONSTRATION extrapolation of one synthetic six-month
    # period, not a forecast of next year's charging energy or revenue.
    cases["demo_annualized_extra_energy_kwh"] = (
        cases["additional_energy_delivered_kwh"]
        * 365.25 / cases["observation_days"]
    ).round(1)
    cases["site_evidence_coverage_pct"] = (
        100 * cases["documented_inputs"] / len(required_inputs)
    ).round(1)
    cases["financial_decision_allowed"] = False
    cases["decision_status"] = "EVIDENCE_REQUIRED"
    cases["investment_recommendation"] = "NOT_ASSESSED"
    cases["forecast_validation"] = "SYNTHETIC_HISTORY_NOT_REAL_DEMAND_FORECAST"
    cases["model_limitations"] = "SIMPLIFIED_CHARGING_AND_SITE_POWER"
    cases["financial_calculation_mode"] = "NOT_COMPUTED"

    if args.demo:
        cases = calculate_demo_values(cases)
        cases["financial_calculation_mode"] = "ILLUSTRATIVE_DEMO_ONLY"

    # One row per site for the network dashboard. We intentionally do NOT rank
    # investment opportunities by synthetic NPV or pretend the best technical
    # upside is a justified expansion recommendation.
    priority_rows = []
    for site_id, group in cases.groupby("site_id", sort=True):
        base = group[group["scenario_type"] == "CURRENT"].iloc[0]
        alternatives = group[group["scenario_type"] != "CURRENT"]
        priority_rows.append({
            "site_id": site_id,
            "city": base["city"],
            "current_chargers": int(base["scenario_chargers"]),
            "evaluated_expansion_options": len(alternatives),
            "maximum_additional_cars_served_in_test_period": (
                int(alternatives["additional_cars_served"].max()) if len(alternatives) else 0
            ),
            "maximum_additional_energy_kwh_in_test_period": (
                round(float(alternatives["additional_energy_delivered_kwh"].max()), 1)
                if len(alternatives) else 0.0
            ),
            "documented_inputs": int(base["documented_inputs"]),
            "required_inputs": len(required_inputs),
            "missing_inputs": int(base["missing_inputs"]),
            "decision_status": "EVIDENCE_REQUIRED",
            "investment_recommendation": "NOT_ASSESSED",
            "financial_decision_allowed": False,
            "note": "Synthetic technical scenario comparisons only; site finances and demand unverified",
        })

    priority = pd.DataFrame(priority_rows).sort_values("city")
    cases = cases.sort_values(["city", "scenario_chargers", "additional_grid_kw"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUT_CASES, index=False, float_format="%.2f")
    priority.to_csv(OUT_PRIORITY, index=False, float_format="%.2f")

    print("\n--- EVIDENCE-GATED CPO NETWORK BUSINESS CASE ---")
    print(f"Sites: {len(priority)} | Scenarios: {len(cases)}")
    print(priority[[
        "city", "current_chargers", "evaluated_expansion_options",
        "maximum_additional_cars_served_in_test_period", "missing_inputs",
        "decision_status"
    ]].to_string(index=False))
    print("\n--- FINANCIAL DECISION STATUS ---")
    print("0 sites are investment-ready. All current financial recommendations are BLOCKED.")
    print("The scenario comparisons are useful for engineering screening only.")
    if args.demo:
        print("\n--- OPTIONAL DEMO ECONOMICS (NOT INVESTMENT ADVICE) ---")
        shown = cases[cases["scenario_type"] != "CURRENT"]
        print(shown[[
            "city", "scenario_chargers", "grid_capacity_kw", "demo_npv_eur",
            "investment_recommendation"
        ]].round({"demo_npv_eur": 0}).to_string(index=False))
        print("DEMO NPVs are illustrative; they cannot authorize an investment decision.")
    else:
        print("No unsourced euro margins, CAPEX, NPV or investment ranks were calculated.")
    print(f"\nSaved scenario decisions: {OUT_CASES}")
    print(f"Saved network decision screen: {OUT_PRIORITY}")
    print("Next: supply dated, site-specific operator data; validate real demand and charger hardware.")


if __name__ == "__main__":
    main()
