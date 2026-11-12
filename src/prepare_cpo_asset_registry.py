r"""Create editable, evidence-aware charging-site equipment and grid registries.

This is the next migration step from five synthetic site records toward a
configuration-driven German CPO asset model. It DOES NOT change simulations or
verify that any supplied equipment/grid value describes a real installation.

Outputs (created only if none already exists):
    data/config/cpo_charger_groups.csv
    data/config/cpo_shared_power_pools.csv
    data/config/cpo_grid_connections.csv

Run from the project directory:
    python src\prepare_cpo_asset_registry.py

For a different input path:
    python src\prepare_cpo_asset_registry.py --sites data/raw/site_master.csv

Design:
* Each site can have MANY charger groups with different manufacturers/models,
  quantities, connector counts, simultaneous session limits and power ratings.
* Distributed charging uses a shared_power_pool_id to reference a separately
  documented power cabinet in cpo_shared_power_pools.csv.
* A site's old 'grid_capacity_kw' is copied ONLY to a MODELLED field. Verified
  contract/import and transformer ratings stay blank until evidence is supplied.
* No investment-grade recommendation is made from these demo inputs.
* Existing user-entered inventory files are never overwritten.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITES = ROOT / "data" / "raw" / "site_master.csv"
DEFAULT_OUT = ROOT / "data" / "config"

CHARGER_COLUMNS = [
    "site_id", "charger_group_id", "charging_type", "architecture",
    "manufacturer", "model", "unit_count", "connectors_per_unit",
    "max_simultaneous_sessions_per_unit", "rated_unit_output_kw",
    "max_connector_output_kw", "shared_power_pool_id", "installation_date",
    "evidence_status", "source_reference", "source_date",
]
POWER_POOL_COLUMNS = [
    "site_id", "shared_power_pool_id", "power_unit_manufacturer",
    "power_unit_model", "installed_power_units", "total_output_capacity_kw",
    "allocation_strategy", "evidence_status", "source_reference", "source_date",
]
GRID_COLUMNS = [
    "site_id", "grid_connection_id", "modelled_site_import_limit_kw",
    "verified_contract_import_limit_kw", "verified_grid_connection_limit_kw",
    "transformer_capacity_kva", "other_site_load_kw", "grid_voltage_level",
    "distribution_grid_operator", "grid_connection_reference",
    "evidence_status", "source_reference", "source_date",
]


def whole_positive(value: object, label: str) -> int:
    try:
        numeric = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a positive integer, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return int(numeric)


def positive(value: object, label: str) -> float:
    try:
        numeric = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be positive, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return numeric


def create_initial_registries(sites: pd.DataFrame):
    sites.columns = sites.columns.str.strip()
    required = {
        "site_id", "city", "charger_count", "charger_power_kw",
        "max_connector_power_kw", "grid_capacity_kw",
    }
    missing = required - set(sites.columns)
    if missing:
        raise ValueError(f"site_master.csv missing: {sorted(missing)}")
    if sites.empty:
        raise ValueError("site_master.csv contains no sites")
    if sites["site_id"].isna().any():
        raise ValueError("site_id cannot be empty")
    ids = sites["site_id"].astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("site_id values must be nonempty and unique")

    charger_rows = []
    grid_rows = []
    for _, site in sites.iterrows():
        site_id = str(site["site_id"]).strip()
        units = whole_positive(site["charger_count"], f"{site_id}: charger_count")
        if "connectors_per_charger" in sites and pd.notna(site["connectors_per_charger"]):
            connectors_per_unit = whole_positive(
                site["connectors_per_charger"], f"{site_id}: connectors_per_charger"
            )
        elif "connector_count" in sites and pd.notna(site["connector_count"]):
            connectors = whole_positive(site["connector_count"], f"{site_id}: connector_count")
            if connectors % units:
                raise ValueError(f"{site_id}: connector_count not divisible by charger_count")
            connectors_per_unit = connectors // units
        else:
            raise ValueError(
                f"{site_id}: no connector layout found. Supply connectors_per_charger "
                "or connector_count in site_master.csv; hardware must not be guessed."
            )
        if "connector_count" in sites and pd.notna(site["connector_count"]):
            actual_total = whole_positive(site["connector_count"], f"{site_id}: connector_count")
            if actual_total != units * connectors_per_unit:
                raise ValueError(f"{site_id}: inconsistent connector_count")

        charger_output = positive(site["charger_power_kw"], f"{site_id}: charger_power_kw")
        connector_output = positive(
            site["max_connector_power_kw"], f"{site_id}: max_connector_power_kw"
        )
        grid_limit = positive(site["grid_capacity_kw"], f"{site_id}: grid_capacity_kw")
        if pd.isna(site["city"]) or not str(site["city"]).strip():
            raise ValueError(f"{site_id}: missing city")

        # This matches the CURRENT synthetic simulator. It does not claim that
        # two plugs can charge simultaneously on any real CPO's hardware.
        charger_rows.append({
            "site_id": site_id,
            "charger_group_id": f"{site_id}__DEMO_GROUP_01",
            "charging_type": "DC",
            "architecture": "INTEGRATED",
            "manufacturer": "",
            "model": "",
            "unit_count": units,
            "connectors_per_unit": connectors_per_unit,
            "max_simultaneous_sessions_per_unit": connectors_per_unit,
            "rated_unit_output_kw": charger_output,
            "max_connector_output_kw": connector_output,
            "shared_power_pool_id": "",
            "installation_date": "",
            "evidence_status": "SYNTHETIC_DEMO_NOT_VERIFIED",
            "source_reference": "data/raw/site_master.csv; original synthetic site configuration",
            "source_date": "",
        })
        grid_rows.append({
            "site_id": site_id,
            "grid_connection_id": f"{site_id}__DEMO_GRID_01",
            "modelled_site_import_limit_kw": grid_limit,
            "verified_contract_import_limit_kw": "",
            "verified_grid_connection_limit_kw": "",
            "transformer_capacity_kva": "",
            "other_site_load_kw": "",
            "grid_voltage_level": "",
            "distribution_grid_operator": "",
            "grid_connection_reference": "",
            "evidence_status": "SYNTHETIC_DEMO_NOT_VERIFIED",
            "source_reference": "data/raw/site_master.csv; original synthetic grid assumption",
            "source_date": "",
        })

    return (
        pd.DataFrame(charger_rows, columns=CHARGER_COLUMNS),
        pd.DataFrame(columns=POWER_POOL_COLUMNS),
        pd.DataFrame(grid_rows, columns=GRID_COLUMNS),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    filenames = [
        "cpo_charger_groups.csv", "cpo_shared_power_pools.csv", "cpo_grid_connections.csv"
    ]
    paths = [args.out_dir / name for name in filenames]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Stopped to protect existing CPO asset data. Already present: "
            + ", ".join(existing)
            + ". No files were changed."
        )

    if not args.sites.is_file():
        raise FileNotFoundError(f"Missing input: {args.sites}")
    sites = pd.read_csv(args.sites, dtype={"site_id": str})
    chargers, pools, grids = create_initial_registries(sites)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for frame, path in zip((chargers, pools, grids), paths):
        frame.to_csv(path, index=False)

    print("\n--- CONFIGURABLE CPO ASSET REGISTRY ---")
    print(f"Sites imported: {len(grids):,}")
    print(f"Demo physical chargers: {int(chargers['unit_count'].sum()):,}")
    print(f"Initial hardware groups: {len(chargers):,}")
    print("Independent groups, mixed models, and shared power pools can be represented.")
    print("All imported hardware and grid values remain SYNTHETIC_DEMO_NOT_VERIFIED.")
    print("No real contracted grid power, transformer size, or OEM model was invented.")
    for path in paths:
        print(f"Saved: {path}")
    print("\nNEXT: Make scenario/simulation engine read this flexible equipment registry.")
    print("Existing scenario and financial results have NOT been changed.")


if __name__ == "__main__":
    main()
