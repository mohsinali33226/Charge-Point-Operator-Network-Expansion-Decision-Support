r"""Generate charger-group-aware CPO expansion options from the asset registry.

Replaces the old site_master-based generator.  Does NOT simulate energy, alter
existing simulations/financial decisions, or claim investment readiness.

Uses:
    data/config/cpo_charger_groups.csv
    data/config/cpo_shared_power_pools.csv
    data/config/cpo_grid_connections.csv
    data/raw/site_master.csv  (OPTIONAL: city display names only)

Writes two linked files:
    data/processed/registry_network_expansion_scenarios.csv
    data/processed/registry_scenario_charger_groups.csv

The previous network_expansion_scenarios.csv is deliberately left alone until
its legacy simulator is replaced with a group-aware simulator.

Usage:
    python src\generate_network_scenarios.py
    python src\generate_network_scenarios.py --growth-pcts 20,40 --min-additional 1

Growth percentages here are candidate EQUIPMENT quantity increments, not
predicted market demand growth. No grid upgrade is invented. Each option adds
units of ONE existing/planned equipment group; a mixed-site combination of
several added groups can be configured in a later development step.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data" / "config"
PROCESSED = ROOT / "data" / "processed"
CHARGERS = CONFIG / "cpo_charger_groups.csv"
POOLS = CONFIG / "cpo_shared_power_pools.csv"
GRIDS = CONFIG / "cpo_grid_connections.csv"
NAMES = ROOT / "data" / "raw" / "site_master.csv"
SCENARIOS_OUT = PROCESSED / "registry_network_expansion_scenarios.csv"
GROUPS_OUT = PROCESSED / "registry_scenario_charger_groups.csv"

GROUP_REQUIRED = {
    "site_id", "charger_group_id", "unit_count", "connectors_per_unit",
    "max_simultaneous_sessions_per_unit", "rated_unit_output_kw",
    "max_connector_output_kw", "architecture", "shared_power_pool_id",
    "evidence_status",
}
GRID_REQUIRED = {
    "site_id", "grid_connection_id", "modelled_site_import_limit_kw",
    "verified_contract_import_limit_kw", "verified_grid_connection_limit_kw",
    "other_site_load_kw", "evidence_status",
}
POOL_REQUIRED = {
    "site_id", "shared_power_pool_id", "total_output_capacity_kw",
    "evidence_status",
}


def read_csv(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


def numeric(raw: object, label: str, *, allow_blank: bool = False,
            allow_zero: bool = False, integer: bool = False) -> float | int | None:
    s = str(raw).strip()
    if s == "":
        if allow_blank:
            return None
        raise ValueError(f"Missing {label}")
    try:
        value = float(s)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric: {s!r}") from exc
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{label} must be {'nonnegative' if allow_zero else 'positive'}: {s!r}")
    if integer and not value.is_integer():
        raise ValueError(f"{label} must be a whole number: {s!r}")
    return int(value) if integer else value


def load_equipment(path: Path) -> pd.DataFrame:
    df = read_csv(path, GROUP_REQUIRED, "charger-group registry")
    if df.empty:
        raise ValueError("The charger-group registry has no equipment groups")
    if df[["site_id", "charger_group_id"]].eq("").any().any():
        raise ValueError("Equipment site_id and charger_group_id cannot be blank")
    if df.duplicated(["site_id", "charger_group_id"]).any():
        raise ValueError("Duplicate (site_id, charger_group_id) in the registry")

    # The input is deliberately read as text to preserve IDs and blank values.
    # Validate numeric fields first, then convert WHOLE columns at once. This
    # avoids pandas/pyarrow errors when assigning an int to a string column.
    parsed: dict[str, list[int | float]] = {
        "unit_count": [],
        "connectors_per_unit": [],
        "max_simultaneous_sessions_per_unit": [],
        "rated_unit_output_kw": [],
        "max_connector_output_kw": [],
    }
    for i, row in df.iterrows():
        prefix = f"row {i+2} / {row['site_id']} / {row['charger_group_id']}"
        units = numeric(row["unit_count"], f"{prefix} unit_count", allow_zero=True, integer=True)
        connectors = numeric(row["connectors_per_unit"], f"{prefix} connectors_per_unit", integer=True)
        simultaneous = numeric(
            row["max_simultaneous_sessions_per_unit"],
            f"{prefix} simultaneous sessions", integer=True
        )
        if simultaneous > connectors:
            raise ValueError(f"{prefix}: simultaneous sessions exceed connectors per unit")
        unit_kw = numeric(row["rated_unit_output_kw"], f"{prefix} unit output")
        connector_kw = numeric(row["max_connector_output_kw"], f"{prefix} connector output")
        if row["architecture"].upper() not in {"INTEGRATED", "DISTRIBUTED"}:
            raise ValueError(f"{prefix}: architecture must be INTEGRATED or DISTRIBUTED")
        if row["architecture"].upper() == "DISTRIBUTED" and not row["shared_power_pool_id"]:
            raise ValueError(f"{prefix}: distributed equipment needs a shared_power_pool_id")
        if not row["evidence_status"]:
            raise ValueError(f"{prefix}: evidence_status must be documented")

        parsed["unit_count"].append(units)
        parsed["connectors_per_unit"].append(connectors)
        parsed["max_simultaneous_sessions_per_unit"].append(simultaneous)
        parsed["rated_unit_output_kw"].append(unit_kw)
        parsed["max_connector_output_kw"].append(connector_kw)

    for col, values in parsed.items():
        df[col] = pd.Series(
            values, index=df.index,
            dtype="int64" if col in {
                "unit_count", "connectors_per_unit",
                "max_simultaneous_sessions_per_unit"
            } else "float64"
        )
    return df


def load_pools(path: Path) -> pd.DataFrame:
    df = read_csv(path, POOL_REQUIRED, "shared power pools")
    if not df.empty:
        if df[["site_id", "shared_power_pool_id"]].eq("").any().any():
            raise ValueError("Power pool site_id/pool ID cannot be blank")
        if df.duplicated(["site_id", "shared_power_pool_id"]).any():
            raise ValueError("Duplicate power pool within a site")
        capacities = [
            numeric(row["total_output_capacity_kw"],
                    f"power pool {row['shared_power_pool_id']} capacity")
            for _, row in df.iterrows()
        ]
        df["total_output_capacity_kw"] = pd.Series(
            capacities, index=df.index, dtype="float64"
        )
    return df


def load_grids(path: Path) -> dict[str, dict]:
    df = read_csv(path, GRID_REQUIRED, "grid registry")
    if df.empty:
        raise ValueError("Grid registry is empty")
    if df["site_id"].eq("").any() or df["site_id"].duplicated().any():
        raise ValueError("Exactly one site-level grid record per site is supported for now")
    grid_by_site = {}
    for _, row in df.iterrows():
        site_id = row["site_id"]
        contract = numeric(row["verified_contract_import_limit_kw"], f"{site_id}: contract import", allow_blank=True)
        connection = numeric(row["verified_grid_connection_limit_kw"], f"{site_id}: grid connection", allow_blank=True)
        modelled = numeric(row["modelled_site_import_limit_kw"], f"{site_id}: modelled import", allow_blank=True)
        site_load = numeric(row["other_site_load_kw"], f"{site_id}: other site load", allow_blank=True, allow_zero=True)
        documented = [v for v in (contract, connection) if v is not None]
        if documented:
            available = min(documented)
            source = "CONTRACT_OR_CONNECTION_LIMIT"
        elif modelled is not None:
            available = modelled
            source = "MODELLED_SITE_LIMIT"
        else:
            raise ValueError(f"{site_id}: no grid import limit available; do not invent one")
        if site_load is not None and site_load >= available:
            raise ValueError(f"{site_id}: other site load is greater than/equal to site import limit")
        grid_by_site[site_id] = {
            "site_grid_import_limit_kw": round(available, 4),
            "grid_limit_source": source,
            "grid_evidence_status": row["evidence_status"],
            "other_site_load_kw": "" if site_load is None else site_load,
            "grid_capacity_for_charging_kw": "" if site_load is None else round(available - site_load, 4),
        }
    return grid_by_site


def load_names(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    names = pd.read_csv(path, dtype=str)
    if not {"site_id", "city"}.issubset(names.columns):
        return {}
    return {str(row["site_id"]).strip(): str(row["city"]).strip()
            for _, row in names.iterrows() if pd.notna(row["site_id"]) and pd.notna(row["city"])}


def expand_scenarios(equipment: pd.DataFrame, pools: pd.DataFrame, grids: dict,
                     names: dict, growth: list[float], min_additional: int):
    site_ids = set(equipment["site_id"])
    if site_ids != set(grids):
        raise ValueError(
            f"Site mismatch: groups without grid={sorted(site_ids-set(grids))}; "
            f"grids without groups={sorted(set(grids)-site_ids)}")
    pool_keys = set(zip(pools["site_id"], pools["shared_power_pool_id"]))
    pool_kw = {(row["site_id"], row["shared_power_pool_id"]): float(row["total_output_capacity_kw"])
               for _, row in pools.iterrows()}
    for _, row in equipment.iterrows():
        if row["shared_power_pool_id"] and (row["site_id"], row["shared_power_pool_id"]) not in pool_keys:
            raise ValueError(f"Group {row['charger_group_id']} refers to missing shared pool {row['shared_power_pool_id']}")
    unreferenced_pools = pool_keys - {
        (row["site_id"], row["shared_power_pool_id"]) for _, row in equipment.iterrows()
        if row["shared_power_pool_id"]
    }
    if unreferenced_pools:
        raise ValueError(f"Power pools with no charger-group references: {sorted(unreferenced_pools)}")

    scenario_rows, group_rows = [], []
    for site_id, site_groups in equipment.groupby("site_id", sort=True):
        site_groups = site_groups.sort_values("charger_group_id").reset_index(drop=True)
        current_chargers = sum(int(v) for v in site_groups["unit_count"])
        current_connectors = sum(int(r["unit_count"]) * int(r["connectors_per_unit"]) for _, r in site_groups.iterrows())
        current_positions = sum(int(r["unit_count"]) * int(r["max_simultaneous_sessions_per_unit"]) for _, r in site_groups.iterrows())
        city = names.get(site_id, site_id)
        grid = grids[site_id]
        hw_statuses = sorted(set(site_groups["evidence_status"].tolist()))
        has_shared_pool = bool(site_groups["shared_power_pool_id"].ne("").any())

        # One baseline, followed by a limited set of add-units options per
        # equipment group. No Cartesian product of all group combinations.
        options: list[tuple[str, int, str, str]] = [("", 0, "CURRENT", f"{site_id}__CURRENT")]
        for _, group in site_groups.iterrows():
            count = int(group["unit_count"])
            proposed_additions = {max(min_additional, math.ceil(max(count, 1) * pct / 100)) for pct in growth}
            for added in sorted(proposed_additions):
                group_id = group["charger_group_id"]
                options.append((group_id, added, "SAME_GRID", f"{site_id}__ADD__{group_id}__{added}"))

        for selected_group_id, added_units, scenario_type, scenario_id in options:
            new_chargers = current_chargers + added_units
            selected = site_groups[site_groups["charger_group_id"] == selected_group_id]
            add_connectors = 0
            add_positions = 0
            if added_units:
                assert len(selected) == 1
                selected_row = selected.iloc[0]
                add_connectors = added_units * int(selected_row["connectors_per_unit"])
                add_positions = added_units * int(selected_row["max_simultaneous_sessions_per_unit"])

            total_rated_dc = 0.0
            shared_pool_ids = set()
            for _, group in site_groups.iterrows():
                group_id = group["charger_group_id"]
                n_current = int(group["unit_count"])
                added = added_units if group_id == selected_group_id else 0
                n_scenario = n_current + added
                unit_kw = float(group["rated_unit_output_kw"])
                total_rated_dc += n_scenario * unit_kw
                if group["shared_power_pool_id"]:
                    shared_pool_ids.add(group["shared_power_pool_id"])
                group_rows.append({
                    "scenario_id": scenario_id,
                    "site_id": site_id,
                    "charger_group_id": group_id,
                    "architecture": group["architecture"],
                    "manufacturer": group.get("manufacturer", ""),
                    "model": group.get("model", ""),
                    "baseline_units": n_current,
                    "added_units": added,
                    "scenario_units": n_scenario,
                    "connectors_per_unit": int(group["connectors_per_unit"]),
                    "max_simultaneous_sessions_per_unit": int(group["max_simultaneous_sessions_per_unit"]),
                    "rated_unit_output_kw": unit_kw,
                    "max_connector_output_kw": float(group["max_connector_output_kw"]),
                    "shared_power_pool_id": group["shared_power_pool_id"],
                    "evidence_status": group["evidence_status"],
                })
            shared_pool_power = sum(pool_kw[(site_id, p)] for p in shared_pool_ids)
            action = (
                "REVIEW_SHARED_POOL_AND_GRID_TOPOLOGY"
                if added_units and has_shared_pool else
                "GRID_FEASIBILITY_UNVERIFIED" if added_units else "BASELINE"
            )
            scenario_rows.append({
                "scenario_id": scenario_id,
                "site_id": site_id,
                "city": city,
                "scenario_type": scenario_type,
                "expanded_group_id": selected_group_id,
                "current_chargers": current_chargers,
                "scenario_chargers": new_chargers,
                "additional_chargers": added_units,
                "current_connectors": current_connectors,
                "scenario_connectors": current_connectors + add_connectors,
                "current_simultaneous_positions": current_positions,
                "scenario_simultaneous_positions": current_positions + add_positions,
                "scenario_grid_kw": grid["site_grid_import_limit_kw"],
                "grid_limit_source": grid["grid_limit_source"],
                "grid_evidence_status": grid["grid_evidence_status"],
                "other_site_load_kw": grid["other_site_load_kw"],
                "available_grid_for_charging_kw": grid["grid_capacity_for_charging_kw"],
                "sum_of_unit_ratings_dc_kw": round(total_rated_dc, 2),
                "existing_shared_pool_output_kw": round(shared_pool_power, 2),
                "equipment_evidence_statuses": ";".join(hw_statuses),
                "engineering_review_status": action,
                "decision_readiness": "CONFIGURATION_ONLY_NO_INVESTMENT_DECISION",
            })

    scenarios = pd.DataFrame(scenario_rows)
    details = pd.DataFrame(group_rows)
    if scenarios["scenario_id"].duplicated().any():
        raise AssertionError("Duplicate scenario IDs; please use unique group IDs")
    if details.duplicated(["scenario_id", "charger_group_id"]).any():
        raise AssertionError("Duplicate equipment configuration rows")
    checks = details.assign(
        total_connectors=details["scenario_units"] * details["connectors_per_unit"],
        simultaneous_positions=details["scenario_units"] * details["max_simultaneous_sessions_per_unit"]
    ).groupby("scenario_id")[['scenario_units', 'total_connectors', 'simultaneous_positions']].sum()
    for _, s in scenarios.iterrows():
        c = checks.loc[s["scenario_id"]]
        if (int(c["scenario_units"]) != s["scenario_chargers"] or
                int(c["total_connectors"]) != s["scenario_connectors"] or
                int(c["simultaneous_positions"]) != s["scenario_simultaneous_positions"]):
            raise AssertionError(f"Scenario hardware total mismatch: {s['scenario_id']}")
    return scenarios, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charger-groups", type=Path, default=CHARGERS)
    parser.add_argument("--shared-pools", type=Path, default=POOLS)
    parser.add_argument("--grids", type=Path, default=GRIDS)
    parser.add_argument("--site-names", type=Path, default=NAMES)
    parser.add_argument("--scenarios-output", type=Path, default=SCENARIOS_OUT)
    parser.add_argument("--groups-output", type=Path, default=GROUPS_OUT)
    parser.add_argument("--growth-pcts", default="25,50", help="Illustrative equipment addition options, e.g. 20,40")
    parser.add_argument("--min-additional", type=int, default=2)
    args = parser.parse_args()
    try:
        growth = sorted({float(s.strip()) for s in args.growth_pcts.split(",") if s.strip()})
    except ValueError as exc:
        parser.error(f"Invalid growth percentages: {exc}")
    if not growth or any(not math.isfinite(v) or v <= 0 for v in growth):
        parser.error("Growth percentages must be positive")
    if args.min_additional < 1:
        parser.error("--min-additional must be at least 1")
    if args.scenarios_output.resolve() == args.groups_output.resolve():
        parser.error("Scenario and group output paths cannot be identical")

    equipment = load_equipment(args.charger_groups)
    pools = load_pools(args.shared_pools)
    grids = load_grids(args.grids)
    names = load_names(args.site_names)
    scenarios, groups = expand_scenarios(equipment, pools, grids, names, growth, args.min_additional)

    args.scenarios_output.parent.mkdir(parents=True, exist_ok=True)
    args.groups_output.parent.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(args.scenarios_output, index=False)
    groups.to_csv(args.groups_output, index=False)

    print("\n--- REGISTRY-DRIVEN CPO NETWORK SCENARIOS ---")
    print(f"Sites: {scenarios['site_id'].nunique():,} | Current chargers: {int(equipment['unit_count'].sum()):,}")
    print(f"Equipment groups: {len(equipment):,} | Shared power pools: {len(pools):,}")
    print(f"Options generated: {len(scenarios):,} (including one baseline per site)")
    print(scenarios[["city", "scenario_type", "expanded_group_id", "scenario_chargers",
                     "scenario_connectors", "scenario_simultaneous_positions", "scenario_grid_kw",
                     "engineering_review_status"]].head(20).to_string(index=False))
    print(f"\nSaved scenarios: {args.scenarios_output}")
    print(f"Saved group configurations: {args.groups_output}")
    print("No grid-upgrade capacity was invented; changes require site-specific DSO evidence.")
    print("IMPORTANT: These new registry scenario files are NOT yet used by the old simulation runner.")
    print("NEXT: Connect group configurations to a group-aware charging simulator.")


if __name__ == "__main__":
    main()
