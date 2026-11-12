"""Select *technical* CPO expansion candidates from simulated demand scenarios.

Run from your CPO Network Development project root (Windows CMD):
    python src\build_technical_expansion_shortlist.py --max-rejected-pct 2

The 2% in this example is a USER-SELECTED SCREENING POLICY, not a German
regulatory or industry-wide service level. Set it to your own operator's
approved planning threshold. Results remain engineering demos because the
arrivals, assumed future growth, site hardware and grid data are not verified.

Inputs (already created in previous project steps):
    data/processed/future_expansion_decision_inputs.csv

Output (new file, no existing outputs overwritten):
    data/processed/cpo_technical_expansion_shortlist.csv

Selection: for each site and demand scenario, choose the current hardware if
its simulated rejection rate meets the selected threshold; otherwise choose
the tested expansion with the fewest physical chargers that meets the target.
When none meets it, report a test gap. Never issue financial or grid approval.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "processed" / "future_expansion_decision_inputs.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "cpo_technical_expansion_shortlist.csv"

REQUIRED = {
    "site_id", "city", "planning_year", "assumed_annual_arrival_growth_pct",
    "scenario_id", "scenario_type", "scenario_chargers", "scenario_arrivals",
    "cars_rejected", "additional_cars_served_vs_current",
    "additional_energy_delivered_kwh_vs_current",
}
NUMERIC = [
    "planning_year", "assumed_annual_arrival_growth_pct", "scenario_chargers",
    "scenario_arrivals", "cars_rejected", "additional_cars_served_vs_current",
    "additional_energy_delivered_kwh_vs_current",
]
GROUP = ["site_id", "planning_year", "assumed_annual_arrival_growth_pct"]


def read_scenarios(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing expansion dataset: {path}\n"
            "First run: python src\\build_future_expansion_decision_inputs.py"
        )
    df = pd.read_csv(path, dtype={"site_id": str, "scenario_id": str})
    df.columns = df.columns.str.strip()
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Expansion dataset has no rows")
    for column in ["site_id", "city", "scenario_id", "scenario_type"]:
        df[column] = df[column].astype("string").str.strip()
        if df[column].isna().any() or df[column].eq("").any():
            raise ValueError(f"Missing/blank {column}")
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any() or not df[column].map(math.isfinite).all():
            raise ValueError(f"Invalid numeric values in {column}")
    for column in ("planning_year", "scenario_chargers", "scenario_arrivals", "cars_rejected"):
        if (df[column] % 1 != 0).any():
            raise ValueError(f"Expected whole numbers for {column}")
        df[column] = df[column].astype(int)
    if ((df["scenario_chargers"] < 1) | (df["scenario_arrivals"] < 1)).any():
        raise ValueError("Charger and arrival counts must be positive")
    if ((df["cars_rejected"] < 0) | (df["cars_rejected"] > df["scenario_arrivals"])).any():
        raise ValueError("Rejected cars must be between 0 and total arrivals")
    if not df["scenario_type"].isin(["CURRENT", "SAME_GRID", "GRID_UPGRADE"]).all():
        raise ValueError("Unexpected scenario_type")
    if df.duplicated(GROUP + ["scenario_id"]).any():
        raise ValueError("Duplicate scenario_id within the same demand assumption")
    if "demand_outlook_classification" in df:
        df["demand_outlook_classification"] = df["demand_outlook_classification"].fillna("UNKNOWN")
    df["simulated_rejection_pct"] = df["cars_rejected"] / df["scenario_arrivals"] * 100
    return df


def technical_screen(df: pd.DataFrame, max_rejected_pct: float) -> pd.DataFrame:
    rows = []
    for (site_id, year, growth_pct), part in df.groupby(GROUP, sort=True, dropna=False):
        base = part[part["scenario_type"] == "CURRENT"]
        if len(base) != 1:
            raise ValueError(f"{site_id} / {year} / {growth_pct}: need exactly one CURRENT row")
        base = base.iloc[0]
        if part["scenario_arrivals"].nunique() != 1:
            raise ValueError(f"{site_id}: scenarios must use IDENTICAL customer demand")
        if part["city"].nunique() != 1:
            raise ValueError(f"{site_id}: conflicting city names")
        if (part["scenario_chargers"] < base["scenario_chargers"]).any():
            raise ValueError(f"{site_id}: an expansion option has fewer chargers than baseline")
        if (part["scenario_type"].ne("CURRENT") & part["scenario_chargers"].eq(base["scenario_chargers"])).any():
            raise ValueError(f"{site_id}: expansion must change charger count for this screening")

        options = part[part["scenario_type"] != "CURRENT"].copy()
        # Only use same-grid configurations here. Grid upgrades require
        # documented DSO proposals and technical connection reviews.
        options = options[options["scenario_type"] == "SAME_GRID"]
        qualifying = options[options["simulated_rejection_pct"] <= max_rejected_pct]
        qualifying = qualifying.sort_values(
            ["scenario_chargers", "simulated_rejection_pct", "scenario_id"], kind="stable"
        )

        baseline_pct = float(base["simulated_rejection_pct"])
        if baseline_pct <= max_rejected_pct:
            chosen = base
            result = "CURRENT_MEETS_SIMULATED_SERVICE_TARGET"
        elif not qualifying.empty:
            chosen = qualifying.iloc[0]
            result = "TESTED_EXPANSION_MEETS_SIMULATED_TARGET"
        else:
            chosen = None
            result = "NO_TESTED_SAME_GRID_OPTION_MEETS_TARGET"

        if chosen is None:
            chosen_chargers = None
            chosen_rejected = None
            after_pct = None
            extra_cars = None
            extra_energy = None
            chosen_id = None
            extra_chargers = None
        else:
            chosen_chargers = int(chosen["scenario_chargers"])
            chosen_rejected = int(chosen["cars_rejected"])
            after_pct = round(float(chosen["simulated_rejection_pct"]), 3)
            extra_cars = int(base["cars_rejected"] - chosen["cars_rejected"])
            extra_energy = round(float(chosen["additional_energy_delivered_kwh_vs_current"]), 1)
            extra_chargers = chosen_chargers - int(base["scenario_chargers"])
            chosen_id = str(chosen["scenario_id"])

        rows.append({
            "site_id": str(site_id),
            "city": str(base["city"]),
            "planning_year_label": int(year),
            "assumed_annual_arrival_growth_pct": float(growth_pct),
            "demand_status": str(base.get(
                "demand_outlook_classification",
                "CONDITIONAL_SYNTHETIC_STRESS_TEST_NOT_VALIDATED_FORECAST"
            )),
            "simulated_arrivals_in_test_period": int(base["scenario_arrivals"]),
            "current_chargers": int(base["scenario_chargers"]),
            "baseline_rejected_cars": int(base["cars_rejected"]),
            "baseline_rejection_pct": round(baseline_pct, 3),
            "operator_chosen_max_rejection_pct": max_rejected_pct,
            "same_grid_expansion_options_tested": len(options),
            "screening_status": result,
            "selected_scenario_id_for_screening": chosen_id,
            "screened_charger_count": chosen_chargers,
            "additional_chargers_in_screen": extra_chargers,
            "screened_rejected_cars": chosen_rejected,
            "screened_rejection_pct": after_pct,
            "avoided_rejections_vs_current": extra_cars,
            "additional_energy_kwh_in_test_period": extra_energy,
            "grid_connection_approval": "NOT_VALIDATED_WITH_DSO",
            "real_world_equipment_approval": "NOT_VERIFIED",
            "financial_investment_decision": "NOT_ASSESSED_EVIDENCE_REQUIRED",
            "screening_limitations": (
                "SYNTHETIC_DEMAND;NO_WAITING_QUEUE;NO_VERIFIED_EV_CHARGING_CURVES;"
                "NO_EQUIPMENT_RELIABILITY;NO_FINANCIAL_OR_SITE_GRID_APPROVAL"
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["assumed_annual_arrival_growth_pct", "site_id"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rejected-pct", type=float, required=True,
                        help="Operator-selected SCREENING threshold (e.g. 2); NOT a German standard")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not math.isfinite(args.max_rejected_pct) or not (0 <= args.max_rejected_pct <= 100):
        raise ValueError("--max-rejected-pct must be between 0 and 100")
    df = read_scenarios(args.input)
    short = technical_screen(df, args.max_rejected_pct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    short.to_csv(args.output, index=False)
    print("\n--- TECHNICAL CPO EXPANSION SHORTLIST ---")
    print(f"Sites: {df.site_id.nunique()} | Demand cases: {len(short)}")
    print(f"Screening policy: at most {args.max_rejected_pct:g}% rejected cars (chosen for demonstration, NOT an industry standard)")
    print(short[[
        "city", "assumed_annual_arrival_growth_pct", "current_chargers",
        "baseline_rejection_pct", "screened_charger_count",
        "screened_rejection_pct", "screening_status",
    ]].to_string(index=False))
    print("\nTechnical shortlist ONLY — no validated future demand, grid permission, or financial recommendation.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
