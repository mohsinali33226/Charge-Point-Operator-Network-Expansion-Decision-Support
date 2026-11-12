"""Combine CPO capacity, site-feasibility and grid-value screens.

Run from the project's root (Windows CMD):
    python src\build_cpo_technical_decision_table.py

Inputs (existing project outputs):
    data/processed/cpo_technical_expansion_shortlist.csv
    data/processed/cpo_site_feasibility_screen.csv
    data/processed/cpo_grid_upgrade_value_screen.csv

Output (new file, does not overwrite previous analyses):
    data/processed/cpo_network_technical_decisions.csv

IMPORTANT: Technical engineering DEMO only. Synthetic arrivals, unverified
future-growth rates, equipment assumptions, site grid details and physical
constraints prevent real construction/investment approval. The grid-value
comparison is a DIAGNOSTIC, not a DSO grid connection offer.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SHORTLIST_FIELDS = {
    "site_id", "city", "planning_year_label", "assumed_annual_arrival_growth_pct",
    "current_chargers", "baseline_rejected_cars", "baseline_rejection_pct",
    "operator_chosen_max_rejection_pct", "screening_status",
    "selected_scenario_id_for_screening", "screened_charger_count",
    "additional_chargers_in_screen", "screened_rejected_cars",
    "screened_rejection_pct", "avoided_rejections_vs_current",
    "additional_energy_kwh_in_test_period", "demand_status",
}
FEASIBILITY_FIELDS = {
    "site_id", "city", "planning_year_label", "assumed_annual_arrival_growth_pct",
    "selected_scenario_id", "additional_chargers_screened",
    "site_import_limit_input_kw", "installed_charger_dc_nameplate_kw",
    "grid_import_source", "grid_evidence_status", "power_review_flag",
    "feasibility_gate", "next_engineering_action",
}
GRID_FIELDS = {
    "site_id", "city", "planning_year", "assumed_annual_arrival_growth_pct",
    "selected_scenario_id", "screened_added_chargers", "grid_value_screen",
    "extra_cars_served_if_grid_not_limiting",
    "extra_energy_if_grid_not_limiting_kwh", "grid_limit_minutes",
}


def read_checked(path: Path, expected: set[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}\nRun the corresponding earlier analysis first."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{label} has no rows")
    return df


def to_number(value: object, label: str, required: bool = False) -> float | None:
    raw = str(value).strip()
    if raw == "" or raw.lower() in {"nan", "none", "null"}:
        if required:
            raise ValueError(f"Missing {label}")
        return None
    try:
        number = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return number


def whole(value: object, label: str, required: bool = False) -> int | None:
    number = to_number(value, label, required=required)
    if number is None:
        return None
    if number < 0 or not number.is_integer():
        raise ValueError(f"Expected nonnegative whole number for {label}: {number}")
    return int(number)


def normalize_id(value: object) -> str:
    raw = str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null"} else raw


def row_key(row: pd.Series, label: str, year_field: str) -> tuple[str, int, float]:
    site_id = normalize_id(row["site_id"])
    if not site_id:
        raise ValueError(f"{label}: blank site_id")
    year = whole(row[year_field], f"{label} year", required=True)
    growth = to_number(
        row["assumed_annual_arrival_growth_pct"], f"{label} demand growth", required=True
    )
    assert year is not None and growth is not None
    return (site_id, year, round(growth, 8))


def as_lookup(
    df: pd.DataFrame, label: str, year_field: str
) -> dict[tuple[str, int, float], pd.Series]:
    result = {}
    for _, row in df.iterrows():
        key = row_key(row, label, year_field)
        if key in result:
            raise ValueError(f"Duplicate {label} row for {key}")
        result[key] = row
    return result


def build_decisions(shortlist: pd.DataFrame, feasibility: pd.DataFrame,
                    grid: pd.DataFrame) -> pd.DataFrame:
    f_lookup = as_lookup(feasibility, "feasibility", "planning_year_label")
    g_lookup = as_lookup(grid, "grid screen", "planning_year")
    s_lookup = as_lookup(shortlist, "shortlist", "planning_year_label")

    if set(s_lookup) != set(f_lookup) or set(s_lookup) != set(g_lookup):
        missing_feas = sorted(set(s_lookup) - set(f_lookup))
        missing_grid = sorted(set(s_lookup) - set(g_lookup))
        extra_feas = sorted(set(f_lookup) - set(s_lookup))
        extra_grid = sorted(set(g_lookup) - set(s_lookup))
        raise ValueError(
            "The three input files must cover the SAME sites, years and demand scenarios. "
            "Re-run the upstream scripts if they were generated at different times.\n"
            f"Missing feasibility: {missing_feas[:5]}; missing grid: {missing_grid[:5]}; "
            f"extra feasibility: {extra_feas[:5]}; extra grid: {extra_grid[:5]}"
        )

    records = []
    for key, s in sorted(s_lookup.items(), key=lambda pair: (pair[0][2], pair[0][0])):
        f = f_lookup[key]
        g = g_lookup[key]
        site, year, growth = key

        if len({s["city"].strip(), f["city"].strip(), g["city"].strip()}) != 1:
            raise ValueError(f"City labels disagree across sources for {key}")

        chosen = normalize_id(s["selected_scenario_id_for_screening"])
        feas_chosen = normalize_id(f["selected_scenario_id"])
        grid_chosen = normalize_id(g["selected_scenario_id"])
        if chosen != feas_chosen or chosen != grid_chosen:
            raise ValueError(
                f"Selected hardware scenario mismatch for {site}, {year}, {growth:g}%: "
                f"shortlist={chosen!r}, feasibility={feas_chosen!r}, grid={grid_chosen!r}. "
                "Re-run the technical shortlist, feasibility and grid-value steps."
            )

        current = whole(s["current_chargers"], f"{site} current chargers", required=True)
        added = whole(s["additional_chargers_in_screen"], f"{site} added chargers")
        feas_added = whole(f["additional_chargers_screened"], f"{site} feasibility chargers")
        grid_added = whole(g["screened_added_chargers"], f"{site} grid chargers")
        selected = whole(s["screened_charger_count"], f"{site} selected chargers")

        # A missing selection is permitted ONLY when no tested scenario met the target.
        status = s["screening_status"].strip()
        no_qualifying = status == "NO_TESTED_SAME_GRID_OPTION_MEETS_TARGET"
        if no_qualifying:
            if chosen or selected is not None or added is not None:
                raise ValueError(f"{site}: unsupported combination for unqualified shortlist")
            if (feas_added not in (None, 0)) or (grid_added not in (None, 0)):
                raise ValueError(f"{site}: expansion mismatch for unqualified shortlist")
        else:
            if not chosen or selected is None or added is None:
                raise ValueError(f"{site}: shortlisted result is missing an expansion selection")
            if selected != current + added:
                raise ValueError(f"{site}: selected chargers != current + added")
            if feas_added != added or grid_added != added:
                raise ValueError(f"{site}: number of added chargers differs between files")

        before_pct = to_number(s["baseline_rejection_pct"], "baseline rejection", required=True)
        after_pct = to_number(s["screened_rejection_pct"], "selected rejection")
        target = to_number(s["operator_chosen_max_rejection_pct"], "rejection target", required=True)
        extra_served = whole(s["avoided_rejections_vs_current"], "avoided rejections")
        grid_saved = whole(g["extra_cars_served_if_grid_not_limiting"], "grid extra cars")
        grid_energy = to_number(g["extra_energy_if_grid_not_limiting_kwh"], "grid extra kWh")
        limit_minutes = whole(g["grid_limit_minutes"], "minutes at grid limit")

        if not 0 <= target <= 100 or not 0 <= before_pct <= 100:
            raise ValueError(f"{site}: rejection percentages out of range")
        if after_pct is not None and not 0 <= after_pct <= 100:
            raise ValueError(f"{site}: selected rejection percentage out of range")
        if not no_qualifying and after_pct is not None and after_pct > target + 0.002:
            raise ValueError(f"{site}: shortlisted chargers do not meet the stated service target")

        if no_qualifying:
            action = "TEST_ADDITIONAL_CHARGER_OPTIONS"
        elif added == 0:
            action = "MONITOR_CURRENT_SITE"
        elif (grid_saved or 0) > 0 or (grid_energy or 0) > 0.11:
            action = "REVIEW_LOAD_MANAGEMENT_AND_GRID_LIMIT"
        else:
            action = "VERIFY_EXPANSION_SITE_AND_EQUIPMENT"

        records.append({
            "site_id": site,
            "city": s["city"].strip(),
            "planning_year": year,
            "assumed_annual_arrival_growth_pct": growth,
            "demand_evidence_status": s["demand_status"],
            "current_chargers": current,
            "shortlisted_chargers": selected,
            "additional_chargers": added,
            "selected_scenario_id": chosen,
            "max_rejected_pct_screening_target": target,
            "rejection_pct_current": before_pct,
            "rejection_pct_shortlisted": after_pct,
            "additional_cars_served_in_test_period": extra_served,
            "additional_energy_kwh_in_test_period": to_number(
                s["additional_energy_kwh_in_test_period"], "additional energy"
            ),
            "site_grid_import_input_kw": to_number(
                f["site_import_limit_input_kw"], "site grid power"
            ),
            "charger_dc_nameplate_kw": to_number(
                f["installed_charger_dc_nameplate_kw"], "installed DC power"
            ),
            "power_review_flag": f["power_review_flag"],
            "grid_limit_minutes_in_test": limit_minutes,
            "extra_cars_if_grid_bottleneck_removed": grid_saved,
            "extra_kwh_if_grid_bottleneck_removed": grid_energy,
            "grid_value_screen": g["grid_value_screen"],
            "grid_evidence_status": f["grid_evidence_status"],
            "site_feasibility_gate": f["feasibility_gate"],
            "technical_next_action": action,
            "investment_decision": "NOT_ASSESSED_EVIDENCE_REQUIRED",
            "model_status": "ENGINEERING_DEMO_UNVERIFIED_INPUTS",
        })

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, default=PROCESSED / "cpo_technical_expansion_shortlist.csv")
    parser.add_argument("--feasibility", type=Path, default=PROCESSED / "cpo_site_feasibility_screen.csv")
    parser.add_argument("--grid", type=Path, default=PROCESSED / "cpo_grid_upgrade_value_screen.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED / "cpo_network_technical_decisions.csv")
    args = parser.parse_args()

    shortlist = read_checked(args.shortlist, SHORTLIST_FIELDS, "technical shortlist")
    feasibility = read_checked(args.feasibility, FEASIBILITY_FIELDS, "site feasibility screen")
    grid = read_checked(args.grid, GRID_FIELDS, "grid value screen")
    result = build_decisions(shortlist, feasibility, grid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("\n--- CPO NETWORK TECHNICAL DECISION TABLE ---")
    print(f"Sites: {result.site_id.nunique()} | Demand cases: {len(result)}")
    short_actions = {
        "MONITOR_CURRENT_SITE": "MONITOR",
        "VERIFY_EXPANSION_SITE_AND_EQUIPMENT": "CHECK_SITE",
        "REVIEW_LOAD_MANAGEMENT_AND_GRID_LIMIT": "CHECK_GRID",
        "TEST_ADDITIONAL_CHARGER_OPTIONS": "TEST_MORE_OPTIONS",
    }
    display = pd.DataFrame({
        "city": result["city"],
        "growth_%": result["assumed_annual_arrival_growth_pct"].map(lambda n: f"{n:g}"),
        "chargers": [
            f"{cur} -> {new}" if pd.notna(new) else f"{cur} -> ?"
            for cur, new in zip(result["current_chargers"], result["shortlisted_chargers"])
        ],
        "rejected_%": [
            f"{before:.2f} -> {after:.2f}" if pd.notna(after) else f"{before:.2f} -> N/A"
            for before, after in zip(result["rejection_pct_current"], result["rejection_pct_shortlisted"])
        ],
        "extra_cars_if_more_grid": [
            str(int(n)) if pd.notna(n) else "N/A"
            for n in result["extra_cars_if_grid_bottleneck_removed"]
        ],
        "next_check": result["technical_next_action"].map(short_actions),
    })
    print(display.to_string(index=False))
    print("\nDECISION STATUS: ENGINEERING DEMO / INVESTMENT EVIDENCE REQUIRED")
    print("The grid comparison is diagnostic, not a proposal or DSO approval.")
    print("Future growth, charger details, other electrical loads and costs remain unverified.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
