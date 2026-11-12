"""Find well-observed German DC charging equipment in the existing data.

Inputs:
  data/processed/german_occupancy_station_summary.csv  (ALL observed stations)
  data/external/Ladesaeulenregister_BNetzA*.csv        (official equipment)
  data/external/chargemyride*.csv                      (status observations)

Outputs, in data/processed:
  german_dc_occupancy_inventory.csv      All ID-matched, DC-confirmed equipment
  german_dc_station_quality.csv          Raw-quality checks on screened DC candidates
  german_dc_validation_candidates.csv    Quality-checked validation sample
  german_dc_hourly_occupancy.csv          Hourly snapshot profiles for selected stations

IMPORTANT: A 'Ladeeinrichtungs-ID' identifies a registered charging installation,
not necessarily an entire multi-charger charging park. DC connector presence is
established from the registered connector *type*, not power or 'Schnell' alone.
Occupancy snapshots do not measure waiting, rejected cars, delivered energy,
or EV arrivals. Equipment dates and observation dates need not coincide.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"
SUMMARY = PROCESSED / "german_occupancy_station_summary.csv"


def norm_id(values: pd.Series) -> pd.Series:
    return (
        values.astype("string").str.strip().str.strip('"')
        .str.replace(r"\.0+$", "", regex=True)
    )


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def discover_source(pattern: str) -> Path:
    paths = list(EXTERNAL.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Missing {pattern} in {EXTERNAL}")
    return max(paths, key=lambda p: p.stat().st_mtime)


def find_bnetza_header(path: Path) -> tuple[str, int, list[str]]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open(encoding=encoding, newline="") as fh:
                for i, line in enumerate(fh):
                    if i > 60:
                        break
                    cells = next(csv.reader([line], delimiter=";"))
                    cells = [s.strip().lstrip("\ufeff") for s in cells]
                    if cells and cells[0] == "Ladeeinrichtungs-ID":
                        return encoding, i, cells
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot find BNetzA header starting Ladeeinrichtungs-ID")


def load_all_observed_stations() -> pd.DataFrame:
    if not SUMMARY.exists():
        raise FileNotFoundError(
            f"{SUMMARY} not found. Run python src\\analyze_german_occupancy.py first."
        )
    data = pd.read_csv(SUMMARY, dtype={"station_id": "string"}, low_memory=False)
    data.columns = data.columns.str.strip()
    needed = ["station_id", "valid_status_observations", "observed_span_days"]
    missing = [c for c in needed if c not in data]
    if missing:
        raise ValueError(f"Occupancy summary missing columns: {missing}")
    data["station_id"] = norm_id(data["station_id"])
    data = data.dropna(subset=["station_id"])
    data = data.loc[data["station_id"].ne("")].copy()
    if data.station_id.duplicated().any():
        raise ValueError("Summary has duplicate station_id rows")
    data["valid_status_observations"] = numeric(data["valid_status_observations"])
    data["observed_span_days"] = numeric(data["observed_span_days"])
    return data


def read_dc_registry(path: Path, observed_ids: set[str], chunksize: int) -> tuple[pd.DataFrame, int]:
    encoding, header_i, header = find_bnetza_header(path)
    required = ["Ladeeinrichtungs-ID", "Art der Ladeeinrichtung",
                "Anzahl Ladepunkte", "Nennleistung Ladeeinrichtung [kW]"]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"BNetzA required headers missing: {missing}")
    connector_cols = [h for h in header if re.match(r"^Steckertypen\s*\d+$", h)]
    if not connector_cols:
        raise ValueError("Cannot locate BNetzA Steckertypen1... columns; cannot verify DC plugs")
    extras = ["Ort", "Bundesland", "Betreiber", "Status", "Inbetriebnahmedatum",
              "Breitengrad", "Längengrad"]
    cols = [h for h in header if h in set(required + extras + connector_cols)]

    # DC proof comes from connector type (CCS/CHAdeMO/DC), NOT only rated kW.
    dc_pattern = re.compile(r"ccs|combo|chademo|cha.?de.?mo|\bdc\b|gb.?t.?dc", re.I)
    found = []
    total = 0
    for part in pd.read_csv(
        path, sep=";", skiprows=header_i, encoding=encoding, dtype="string",
        usecols=cols, chunksize=chunksize, low_memory=False,
    ):
        total += len(part)
        ids = norm_id(part["Ladeeinrichtungs-ID"])
        part = part.loc[ids.isin(observed_ids)].copy()
        if part.empty:
            continue
        part["station_id"] = ids.loc[part.index]
        texts = part[connector_cols].fillna("").agg(" | ".join, axis=1)
        mask = texts.map(lambda s: bool(dc_pattern.search(s)))
        part = part.loc[mask].copy()
        if part.empty:
            continue
        part["dc_connector_types"] = texts.loc[part.index]
        found.append(part)
    if not found:
        raise ValueError(
            "No confirmed DC connector IDs matched occupancy stations. "
            "Check source files; do not fall back to power-only classification."
        )
    registry = pd.concat(found, ignore_index=True)
    registry["registered_charging_points"] = numeric(registry["Anzahl Ladepunkte"])
    registry["registered_equipment_power_kw"] = numeric(
        registry["Nennleistung Ladeeinrichtung [kW]"]
    )
    registry["registry_rows_for_id"] = registry.groupby("station_id")["station_id"].transform("size")
    # Keep duplicate ID records in inventory, but block them from validation sample.
    registry = registry.drop_duplicates("station_id", keep="first")
    registry = registry.rename(columns={
        "Art der Ladeeinrichtung": "registered_equipment_type",
        "Ort": "registered_city",
        "Bundesland": "registered_state",
        "Betreiber": "registered_operator",
        "Status": "registered_status",
    })
    return registry, total


def tier(power: float) -> str:
    if pd.isna(power):
        return "POWER_UNKNOWN"
    if power >= 300:
        return "DC_300_PLUS_KW"
    if power >= 150:
        return "DC_150_299_KW"
    if power >= 50:
        return "DC_50_149_KW"
    return "DC_BELOW_50_KW"


def diversify_candidates(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Round-robin sampling across power/point groups; avoid only busiest AC/DC IDs."""
    temp = df.copy()
    temp["point_group"] = np.select(
        [temp.registered_charging_points.le(1), temp.registered_charging_points.eq(2)],
        ["ONE_POINT", "TWO_POINTS"], default="THREE_PLUS_POINTS"
    )
    temp["sampling_group"] = temp.power_class + " / " + temp.point_group
    temp = temp.sort_values(
        ["valid_status_observations", "observed_span_days"], ascending=[False, False]
    )
    groups = {k: list(g.index) for k, g in temp.groupby("sampling_group", sort=True)}
    selected = []
    while len(selected) < limit and any(groups.values()):
        for name in sorted(groups):
            if groups[name]:
                selected.append(groups[name].pop(0))
                if len(selected) >= limit:
                    break
    return temp.loc[selected].copy().reset_index(drop=True)


def read_raw_quality(path: Path, ids: set[str], chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = ["station_id", "ts_unix", "available", "occupied", "total",
              "reserved", "out_of_service", "unknown"]
    batches = []
    for part in pd.read_csv(
        path, comment="#", encoding="utf-8-sig", dtype={"station_id": "string"},
        usecols=fields, chunksize=chunksize, low_memory=False,
    ):
        part["station_id"] = norm_id(part["station_id"])
        chosen = part.loc[part.station_id.isin(ids)].copy()
        if not chosen.empty:
            batches.append(chosen)
    if not batches:
        raise ValueError("No raw ChargeMyRide records found for screened DC IDs")
    all_rows = pd.concat(batches, ignore_index=True)
    for col in fields[1:]:
        all_rows[col] = numeric(all_rows[col])
    all_rows = all_rows.dropna(subset=["ts_unix"])
    all_rows = all_rows.drop_duplicates(["station_id", "ts_unix"], keep="last")
    all_rows["local_ts"] = pd.to_datetime(
        all_rows.ts_unix, unit="s", utc=True, errors="coerce"
    ).dt.tz_convert("Europe/Berlin")
    all_rows = all_rows.dropna(subset=["local_ts"])
    valid = (
        all_rows[["available", "occupied", "total"]].notna().all(axis=1)
        & all_rows.available.ge(0) & all_rows.occupied.ge(0)
        & all_rows.total.gt(0)
        & (all_rows.available + all_rows.occupied <= all_rows.total)
    )
    all_rows["valid_status"] = valid
    good = all_rows.loc[valid].copy()
    good["local_date"] = good.local_ts.dt.date
    good["local_hour"] = good.local_ts.dt.hour
    good["occupied_fraction"] = good.occupied / good.total
    good["no_available"] = good.available.eq(0).astype(float)

    if good.empty:
        raise ValueError("No selected DC stations have complete, consistent status snapshots")
    quality = good.groupby("station_id").agg(
        valid_snapshots=("ts_unix", "size"),
        observed_days=("local_date", "nunique"),
        observed_hours_of_day=("local_hour", "nunique"),
        mean_occupied_snapshot_pct=("occupied_fraction", "mean"),
        no_available_snapshot_pct=("no_available", "mean"),
        typical_reported_points=("total", lambda x: float(x.mode().iloc[0])),
    ).reset_index()
    quality["mean_occupied_snapshot_pct"] *= 100
    quality["no_available_snapshot_pct"] *= 100
    quality["invalid_or_incomplete_snapshots"] = (
        all_rows.groupby("station_id").size()
        .reindex(quality.station_id).to_numpy() - quality.valid_snapshots.to_numpy()
    )
    for col in ["mean_occupied_snapshot_pct", "no_available_snapshot_pct"]:
        quality[col] = quality[col].round(2)

    hourly = good.groupby(["station_id", "local_hour"]).agg(
        valid_snapshots=("ts_unix", "size"),
        mean_occupied_pct=("occupied_fraction", "mean"),
        no_available_snapshot_pct=("no_available", "mean"),
    ).reset_index()
    hourly[["mean_occupied_pct", "no_available_snapshot_pct"]] *= 100
    hourly[["mean_occupied_pct", "no_available_snapshot_pct"]] = (
        hourly[["mean_occupied_pct", "no_available_snapshot_pct"]].round(2)
    )
    return quality, hourly


def main() -> None:
    parser = argparse.ArgumentParser(description="Find real DC charging equipment with usable occupancy history")
    parser.add_argument("--min-power-kw", type=float, default=50.0,
                        help="Minimum registered equipment power, default 50 kW")
    parser.add_argument("--min-snapshots", type=int, default=200,
                        help="Minimum unique valid snapshots, default 200")
    parser.add_argument("--min-days", type=int, default=14,
                        help="Minimum observed days, default 14")
    parser.add_argument("--min-hours", type=int, default=18,
                        help="Distinct hours-of-day covered, default 18")
    parser.add_argument("--inspect-limit", type=int, default=450,
                        help="Max preliminary DC stations to check in raw data; 0 = all")
    parser.add_argument("--max-candidates", type=int, default=100,
                        help="Maximum selected validation sample; 0 = all")
    parser.add_argument("--chunksize", type=int, default=150_000)
    args = parser.parse_args()
    if args.min_power_kw < 0 or args.min_snapshots <= 0 or args.min_days <= 0 or args.min_hours not in range(1, 25) or args.inspect_limit < 0 or args.max_candidates < 0 or args.chunksize < 100:
        parser.error("Invalid screening settings")

    print("\n--- REAL GERMAN DC CHARGER DISCOVERY ---")
    observed = load_all_observed_stations()
    register_path = discover_source("Ladesaeulenregister_BNetzA*.csv")
    raw_path = discover_source("chargemyride*.csv")
    registry, registry_count = read_dc_registry(register_path, set(observed.station_id), args.chunksize)
    inventory = registry.merge(observed, on="station_id", how="left", validate="one_to_one")
    inventory["power_class"] = inventory.registered_equipment_power_kw.map(tier)
    inventory["preliminary_quality_ok"] = (
        inventory.registry_rows_for_id.eq(1)
        & inventory.registered_equipment_power_kw.ge(args.min_power_kw)
        & inventory.registered_charging_points.ge(1)
        & inventory.valid_status_observations.ge(args.min_snapshots)
        & inventory.observed_span_days.ge(args.min_days - 1)
    )
    output_inventory = PROCESSED / "german_dc_occupancy_inventory.csv"
    inventory.to_csv(output_inventory, index=False)
    eligible = inventory.loc[inventory.preliminary_quality_ok].copy()
    if eligible.empty:
        print("No stations meet the initial screening requirements; try lowering --min-snapshots or --min-days after inspecting the inventory.")
        print("Inventory saved:", output_inventory)
        return
    inspect_count = len(eligible) if args.inspect_limit == 0 else min(args.inspect_limit, len(eligible))
    sample = diversify_candidates(eligible, inspect_count)
    quality, hourly = read_raw_quality(raw_path, set(sample.station_id), args.chunksize)
    assessed = sample.merge(quality, on="station_id", how="left", validate="one_to_one")
    assessed["point_count_agrees"] = (
        assessed.registered_charging_points.notna()
        & assessed.typical_reported_points.notna()
        & assessed.registered_charging_points.eq(assessed.typical_reported_points)
    )
    assessed["passes_quality"] = (
        assessed.valid_snapshots.ge(args.min_snapshots)
        & assessed.observed_days.ge(args.min_days)
        & assessed.observed_hours_of_day.ge(args.min_hours)
        & assessed.point_count_agrees
    )
    assessed["quality_status"] = np.select(
        [~assessed.point_count_agrees,
         assessed.valid_snapshots.isna(),
         ~assessed.passes_quality],
        ["POINT_COUNT_MISMATCH", "NO_VALID_RAW_STATUSES", "INSUFFICIENT_VALID_COVERAGE"],
        default="READY_FOR_OCCUPANCY_COMPARISON",
    )
    selected = assessed.loc[assessed.passes_quality].copy()
    if args.max_candidates:
        selected = selected.head(args.max_candidates)

    details = PROCESSED / "german_dc_station_quality.csv"
    candidates = PROCESSED / "german_dc_validation_candidates.csv"
    hourly_path = PROCESSED / "german_dc_hourly_occupancy.csv"
    assessed.to_csv(details, index=False)
    selected.to_csv(candidates, index=False)
    hourly.loc[hourly.station_id.isin(set(selected.station_id))].to_csv(hourly_path, index=False)

    print(f"Occupancy station IDs available: {len(observed):,}")
    print(f"BNetzA register rows checked: {registry_count:,}")
    print(f"Confirmed DC equipment IDs matched: {len(inventory):,}")
    print(f"Of those, >=150 kW: {int(inventory.registered_equipment_power_kw.ge(150).sum()):,}")
    print(f"Eligible for detailed inspection: {len(eligible):,}")
    print(f"Inspected raw histories: {len(assessed):,}")
    print(f"Passed real-status and equipment-count checks: {int(assessed.passes_quality.sum()):,}")
    print(f"Selected validation sample: {len(selected):,}")
    if not selected.empty:
        print("\n--- SELECTED DC CHARGERS BY POWER ---")
        print(selected.power_class.value_counts().to_string())
        print("\n--- EXAMPLE MATCHED DC CHARGERS ---")
        print(selected[["station_id", "registered_city", "registered_equipment_power_kw",
                        "registered_charging_points", "valid_snapshots", "observed_days",
                        "mean_occupied_snapshot_pct", "no_available_snapshot_pct"]].head(10).to_string(index=False))
    print("\nSaved:")
    for p in (output_inventory, details, candidates, hourly_path):
        print(p)
    print("\nNOTE: These are registered charging installations, not necessarily whole sites.")
    print("Snapshot occupancy is NOT arrival demand, queues or charging-session energy.")
    print("No comparison with simulated rejection rates or investment decisions has been made.")


if __name__ == "__main__":
    main()
