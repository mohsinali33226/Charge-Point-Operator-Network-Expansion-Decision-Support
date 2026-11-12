"""CPO site feasibility review: link expansion screening to hardware/grid evidence.

Run from the project root:
    python src\build_cpo_site_feasibility.py

Inputs already produced by the project:
    data/processed/cpo_technical_expansion_shortlist.csv
    data/processed/registry_network_expansion_scenarios.csv

Outputs (do not overwrite source files or older financial models):
    data/processed/cpo_site_feasibility_screen.csv
    data/processed/cpo_site_evidence_requests.csv

This script does NOT approve investments, grid connections or construction.
Even when the charging model meets a service target, real equipment, the
local distribution network operator (DSO), parking space, costs and operating
conditions still require verification. DC rated output is NOT the same as
AC grid import; the comparison is only a conservative review trigger.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_SHORTLIST = PROCESSED / "cpo_technical_expansion_shortlist.csv"
DEFAULT_SCENARIOS = PROCESSED / "registry_network_expansion_scenarios.csv"
DEFAULT_SCREEN = PROCESSED / "cpo_site_feasibility_screen.csv"
DEFAULT_REQUESTS = PROCESSED / "cpo_site_evidence_requests.csv"

SHORTLIST_FIELDS = {
    "site_id", "city", "planning_year_label", "assumed_annual_arrival_growth_pct",
    "current_chargers", "screening_status", "selected_scenario_id_for_screening",
    "screened_charger_count", "additional_chargers_in_screen",
}
SCENARIO_FIELDS = {
    "site_id", "scenario_id", "scenario_chargers", "scenario_connectors",
    "scenario_simultaneous_positions", "scenario_grid_kw", "grid_limit_source",
    "grid_evidence_status", "equipment_evidence_statuses", "other_site_load_kw",
    "sum_of_unit_ratings_dc_kw", "existing_shared_pool_output_kw",
}

# A request list, not a claim that every CPO must purchase every item.
# Verified evidence can be supplied later by the operator in its own systems.
REQUESTS = (
    ("DEMAND", "Verified site charging sessions, failed arrivals, and time trends"),
    ("SERVICE_POLICY", "CPO-approved rejection/queue/service target"),
    ("HARDWARE", "Actual charger, connector and shared-power-unit specifications"),
    ("GRID", "Contracted import limit, DSO connection conditions and written grid feasibility"),
    ("ELECTRICAL", "Transformer, switchgear, cabling, protection and spare electrical capacity"),
    ("LOAD_MANAGEMENT", "Site power-controller settings, priority rules and supporting OEM limits"),
    ("SITE", "Available parking bays, access, land rights and civil-works feasibility"),
    ("COMPLIANCE", "Applicable AFIR, payment, metering/Eichrecht and permitting review"),
    ("OPERATIONS", "Current availability, faults, charging curves, queue behaviour and maintenance"),
    ("COMMERCIAL", "Operator tariffs, net receipts, energy/network costs, roaming and site OPEX"),
    ("QUOTES", "Site-specific charger, installation and (if needed) grid-upgrade quotations"),
    ("COMPETITION", "Nearby competing charging supply and expected local demand changes"),
)


def read_csv(path: Path, fields: set[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    missing = fields - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{label} is empty")
    for key in ["site_id", "city"] if "city" in df.columns else ["site_id"]:
        df[key] = df[key].str.strip()
        if df[key].eq("").any():
            raise ValueError(f"{label}: empty {key}")
    return df


def optional_float(raw: str, name: str) -> float | None:
    raw = str(raw).strip()
    if not raw:
        return None
    try:
        result = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric {name}: {raw!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {name}")
    return result


def required_float(raw: str, name: str) -> float:
    result = optional_float(raw, name)
    if result is None:
        raise ValueError(f"Missing numeric {name}")
    return result


def whole(raw: str, name: str) -> int:
    number = required_float(raw, name)
    if number < 0 or not number.is_integer():
        raise ValueError(f"Expected nonnegative whole number for {name}")
    return int(number)


def make_screen(shortlist: pd.DataFrame, scenarios: pd.DataFrame):
    if scenarios["scenario_id"].duplicated().any():
        raise ValueError("Scenario IDs must be unique")
    if shortlist.duplicated(
        ["site_id", "planning_year_label", "assumed_annual_arrival_growth_pct"]
    ).any():
        raise ValueError("Repeated site / planning year / demand scenario")

    lookup = {str(r["scenario_id"]): r for _, r in scenarios.iterrows()}
    summary = []
    expansion_sites: dict[str, str] = {}

    for _, row in shortlist.iterrows():
        site = row["site_id"]
        city = row["city"]
        decision = row["screening_status"].strip()
        baseline_units = whole(row["current_chargers"], f"{site} current_chargers")
        selected_id = row["selected_scenario_id_for_screening"].strip()
        growth = required_float(row["assumed_annual_arrival_growth_pct"], f"{site} growth")
        year = whole(row["planning_year_label"], f"{site} planning year")

        selected = lookup.get(selected_id) if selected_id else None
        if selected_id and selected is None:
            raise ValueError(f"Unknown selected scenario {selected_id} for {site}")
        if selected is not None and selected["site_id"] != site:
            raise ValueError(f"Selected scenario {selected_id} belongs to another site")

        if decision == "NO_TESTED_SAME_GRID_OPTION_MEETS_TARGET":
            action = "MORE_CONFIGURATIONS_REQUIRED"
            engineering_gate = "NOT_ASSESSED"
            dc_installed = None
            grid_import = None
            connector_count = None
            simultaneous = None
            power_screen = "NO_QUALIFYING_SCENARIO"
            source = ""
            evidence = ""
            load_known = False
            additional = None
        else:
            if selected is None:
                raise ValueError(f"{site} missing scenario ID for {decision}")
            selected_units = whole(selected["scenario_chargers"], f"{site} chargers")
            if selected_units < baseline_units:
                raise ValueError(f"{site}: chosen scenario reduces physical chargers")
            additional = selected_units - baseline_units
            declared_add = whole(row["additional_chargers_in_screen"], f"{site} additions")
            if declared_add != additional:
                raise ValueError(f"{site}: additions differ between shortlist and registry")
            screened = whole(row["screened_charger_count"], f"{site} shortlist chargers")
            if selected_units != screened:
                raise ValueError(f"{site}: screened and registry chargers disagree")
            dc_installed = required_float(selected["sum_of_unit_ratings_dc_kw"], f"{site} DC nameplate")
            grid_import = required_float(selected["scenario_grid_kw"], f"{site} grid import")
            if dc_installed <= 0 or grid_import <= 0:
                raise ValueError(f"{site}: power limits must be positive")
            connector_count = whole(selected["scenario_connectors"], f"{site} connectors")
            simultaneous = whole(selected["scenario_simultaneous_positions"], f"{site} simultaneous positions")
            if not (0 < simultaneous <= connector_count):
                raise ValueError(f"{site}: invalid simultaneous positions")
            source = selected["grid_limit_source"].strip()
            evidence = selected["grid_evidence_status"].strip()
            load_known = optional_float(selected["other_site_load_kw"], "other site load") is not None
            # This is NOT a grid-upgrade requirement. Charger DC nameplate
            # can exceed grid import safely when load management is designed
            # and configured for it. AC conversion losses still apply.
            power_screen = (
                "DC_NAMEPLATE_ABOVE_SITE_IMPORT_REVIEW_LOAD_MANAGEMENT"
                if dc_installed > grid_import else
                "REVIEW_AC_LOSSES_OTHER_LOADS_AND_GRID_TERMS"
            )
            if additional == 0:
                action = "MONITOR_EXISTING_SITE_UNDER_TESTED_DEMAND"
                engineering_gate = "NO_EXPANSION_PROPOSED"
            else:
                action = "CHECK_SITE_EXPANSION_FEASIBILITY"
                engineering_gate = "EVIDENCE_REQUIRED_NOT_APPROVED"
                expansion_sites[site] = city

        summary.append({
            "site_id": site,
            "city": city,
            "planning_year_label": year,
            "assumed_annual_arrival_growth_pct": growth,
            "screening_status": decision,
            "selected_scenario_id": selected_id,
            "current_chargers": baseline_units,
            "additional_chargers_screened": additional,
            "selected_connectors": connector_count,
            "selected_simultaneous_positions": simultaneous,
            "site_import_limit_input_kw": grid_import,
            "installed_charger_dc_nameplate_kw": dc_installed,
            "grid_import_source": source,
            "grid_evidence_status": evidence,
            "other_site_load_documented": load_known,
            "power_review_flag": power_screen,
            "next_engineering_action": action,
            "feasibility_gate": engineering_gate,
            "financial_status": "NOT_ASSESSED_NO_VERIFIED_QUOTES",
            "evidence_warning": "SYNTHETIC_ARRIVALS_AND_GROWTH_SCENARIO_NOT_VERIFIED",
        })

    requests = []
    for site, city in sorted(expansion_sites.items()):
        for category, requirement in REQUESTS:
            requests.append({
                "site_id": site,
                "city": city,
                "category": category,
                "needed_evidence": requirement,
                "evidence_status": "NEEDS_OPERATOR_OR_SITE_VERIFICATION",
                "source_reference": "",
                "verified_date": "",
            })
    return pd.DataFrame(summary), pd.DataFrame(requests, columns=[
        "site_id", "city", "category", "needed_evidence",
        "evidence_status", "source_reference", "verified_date",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--screen-out", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--requests-out", type=Path, default=DEFAULT_REQUESTS)
    args = parser.parse_args()
    if args.screen_out.resolve() == args.requests_out.resolve():
        raise ValueError("Output paths must differ")
    short = read_csv(args.shortlist, SHORTLIST_FIELDS, "technical shortlist")
    options = read_csv(args.scenarios, SCENARIO_FIELDS, "equipment scenarios")
    screen, requests = make_screen(short, options)
    for frame, path in [(screen, args.screen_out), (requests, args.requests_out)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print("\n--- SITE ENGINEERING FEASIBILITY SCREEN ---")
    print(f"Sites: {screen.site_id.nunique()} | Demand cases: {len(screen)} | Evidence requests: {len(requests)}")
    print(screen[["city", "assumed_annual_arrival_growth_pct",
                  "additional_chargers_screened", "site_import_limit_input_kw",
                  "power_review_flag", "feasibility_gate"]].to_string(index=False))
    print("\nNOTES: DC charger ratings are not grid import; power-sharing can limit site draw.")
    print("Site permissions, actual grid/transformer limits, AC losses and costs remain unverified.")
    print("No grid-upgrade requirement, investment recommendation or real future demand was inferred.")
    print(f"Saved: {args.screen_out}")
    print(f"Saved: {args.requests_out}")


if __name__ == "__main__":
    main()
