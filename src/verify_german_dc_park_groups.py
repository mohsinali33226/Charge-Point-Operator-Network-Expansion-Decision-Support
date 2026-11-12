"""Review possible German DC charging parks and align real occupancy snapshots.

Place this complete file in the project's src/ folder and run:
    python src\\verify_german_dc_park_groups.py

Requires the outputs of expand_german_dc_comparables.py. Uses the official BNetzA
registry and the original ChargeMyRide CSV already placed in data/external/.

Important:
- Nearby installations are not confirmed charging parks.
- Observations from different installations are combined ONLY when they fall
  within the same short time window and have sufficiently close timestamps.
- Snapshot occupancy is NOT a time-weighted utilization, rejection rate,
  waiting time, or validated expansion forecast.
- Does not overwrite source datasets, previous models, or financial estimates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from expand_german_dc_comparables import (
    load_registry_addresses, distance_m, select_file, clean_id,
    to_number, power_band, prepare_synthetic,
)


def require(path: Path, cols: list[str], **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the preceding DC comparison step first.")
    df = pd.read_csv(path, low_memory=False, **kwargs)
    df.columns = df.columns.str.strip()
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return df


def members_of(text: object) -> list[str]:
    if pd.isna(text):
        return []
    return list(dict.fromkeys(x.strip() for x in str(text).split("|") if x.strip()))


def geography_check(member_ids: list[str], address_lookup: dict, max_distance_m: float):
    rows = [address_lookup.get(k) for k in member_ids]
    if any(x is None for x in rows):
        return "LOCATION_DATA_INCOMPLETE", np.nan, "Missing official address/coordinates"
    operators = [x.get("operator_normalized", "") for x in rows]
    postcodes = [x.get("postcode_normalized", "") for x in rows]
    streets = [x.get("street_normalized", "") for x in rows]
    cities = [x.get("registered_city_normalized", "") for x in rows]
    building_numbers = [x.get("building_number_normalized", "") for x in rows]
    coordinates = [(x["latitude"], x["longitude"]) for x in rows]
    if any(pd.isna(a) or pd.isna(b) for a,b in coordinates):
        return "LOCATION_DATA_INCOMPLETE", np.nan, "Missing official coordinates"
    if any(not (-90 <= a <= 90 and -180 <= b <= 180) for a,b in coordinates):
        return "LOCATION_DATA_INCOMPLETE", np.nan, "Coordinates outside valid ranges"
    span = max(distance_m(a,b,c,d) for i,(a,b) in enumerate(coordinates)
               for c,d in coordinates[i+1:])
    if len(set(operators)) != 1 or not operators[0]:
        return "LOCATION_CONFLICT", span, "Operator mismatch or missing"
    if len({v for v in postcodes if v}) > 1 or len({v for v in cities if v}) > 1:
        return "LOCATION_CONFLICT", span, "Different postcode or city"
    if span > max_distance_m:
        return "LOCATION_CONFLICT", span, "Group spans more than proximity threshold"
    address_agrees = (len({v for v in streets if v}) == 1 and bool(streets[0])
                      and len({v for v in building_numbers if v}) <= 1
                      and len({v for v in postcodes if v}) == 1 and bool(postcodes[0]))
    if address_agrees:
        return "STRONG_COLOCATION_CANDIDATE", span, "Same operator/address and close coordinates; manual check required"
    if span <= 12:
        return "NEARBY_CANDIDATE", span, "Same operator and very close coordinates; address review required"
    return "ADDRESS_REVIEW_REQUIRED", span, "Close coordinates; uncertain site boundary/address"


def read_aligned_status(csv_file: Path, station_ids: set[str], expected_points: dict,
                        chunk_size: int, bin_minutes: int, tolerance_minutes: int) -> pd.DataFrame:
    fields = ["station_id", "ts_unix", "available", "occupied", "total"]
    selected = []
    read_rows = 0
    for part in pd.read_csv(csv_file, encoding="utf-8-sig", comment="#",
                            dtype={"station_id": "string"}, usecols=fields,
                            chunksize=chunk_size, low_memory=False):
        read_rows += len(part)
        part["station_id"] = clean_id(part["station_id"])
        part = part.loc[part.station_id.isin(station_ids)].copy()
        if not part.empty:
            selected.append(part)
        if read_rows % 1500000 < chunk_size:
            print(f"  Read {read_rows:,} raw status rows ...")
    if not selected:
        return pd.DataFrame(columns=["station_id", "window_unix", "ts_unix", "occupied", "available", "total"])
    raw = pd.concat(selected, ignore_index=True)
    for c in fields[1:]:
        raw[c] = to_number(raw[c])
    raw = raw.dropna(subset=fields[1:]).copy()
    raw = raw.loc[
        raw.ts_unix.ge(0) & raw.occupied.ge(0) & raw.available.ge(0) & raw.total.gt(0)
        & (raw.occupied + raw.available <= raw.total)
    ].copy()
    raw["expected_points"] = raw.station_id.map(expected_points)
    # Reject records where the recorded total disagrees with the official register.
    raw = raw.loc[raw.total.eq(raw.expected_points)].copy()
    raw = raw.sort_values("ts_unix").drop_duplicates(["station_id", "ts_unix"], keep="last")
    interval = bin_minutes * 60
    max_offset = tolerance_minutes * 60
    raw["window_unix"] = ((raw.ts_unix + interval // 2) // interval * interval).astype("int64")
    raw["time_offset_seconds"] = (raw.ts_unix - raw.window_unix).abs()
    raw = raw.loc[raw.time_offset_seconds.le(max_offset)].copy()
    raw = raw.sort_values(["station_id", "window_unix", "time_offset_seconds", "ts_unix"],
                          ascending=[True, True, True, False])
    raw = raw.drop_duplicates(["station_id", "window_unix"], keep="first")
    return raw[["station_id", "window_unix", "ts_unix", "occupied", "available", "total"]]


def summarize_groups(parks: pd.DataFrame, quality: pd.DataFrame, addresses: pd.DataFrame,
                     observations: pd.DataFrame, args):
    address_lookup = {str(row["station_id"]): row for row in addresses.to_dict("records")}
    quality_lookup = {str(row["station_id"]): row for row in quality.to_dict("records")}
    records = []
    aligned = []
    for group in parks.to_dict("records"):
        members = members_of(group["member_ids"])
        park_id = str(group["park_candidate_id"])
        if len(members) < 2:
            raise ValueError(f"Invalid group {park_id}: fewer than two distinct installations")
        location_status, spread_m, reason = geography_check(members, address_lookup, args.max_park_span_m)
        qrows = [quality_lookup.get(k) for k in members]
        qualified = all(r is not None and r.get("passes_quality") in (True, 1, "True", "true") for r in qrows)
        # Restrict to measurements taken reasonably close together in time.
        subset = observations.loc[observations.station_id.isin(members)]
        if subset.empty:
            complete = pd.DataFrame()
        else:
            g = subset.groupby("window_unix").agg(
                members_observed=("station_id", "nunique"),
                occupied=("occupied", "sum"), available=("available", "sum"),
                total=("total", "sum"),
                earliest_ts=("ts_unix", "min"), latest_ts=("ts_unix", "max"),
            ).reset_index()
            complete = g.loc[(g.members_observed == len(members)) &
                             ((g.latest_ts - g.earliest_ts) <= 2 * args.tolerance_minutes * 60)].copy()
        if not complete.empty:
            complete["park_candidate_id"] = park_id
            complete["snapshot_occupancy_pct"] = 100 * complete.occupied / complete.total
            complete["no_free_reported"] = complete.available.eq(0).astype(int)
            complete["local_date"] = pd.to_datetime(complete.window_unix, unit="s", utc=True).dt.tz_convert("Europe/Berlin").dt.date
            complete["local_hour"] = pd.to_datetime(complete.window_unix, unit="s", utc=True).dt.tz_convert("Europe/Berlin").dt.hour
            aligned.append(complete[["park_candidate_id", "window_unix", "members_observed", "occupied", "available",
                                     "total", "earliest_ts", "latest_ts", "snapshot_occupancy_pct", "no_free_reported", "local_date", "local_hour"]])
        n = len(complete)
        days = int(complete.local_date.nunique()) if n else 0
        hours = int(complete.local_hour.nunique()) if n else 0
        sufficient = bool(n >= args.min_aligned_windows and days >= args.min_aligned_days and hours >= args.min_hours_of_day)
        if location_status in ("LOCATION_CONFLICT", "LOCATION_DATA_INCOMPLETE") or not qualified:
            review = "NOT_SUITABLE_FOR_GROUP_COMPARISON"
        elif not sufficient:
            review = "INSUFFICIENT_SYNCHRONIZED_OBSERVATIONS"
        else:
            review = "DESCRIPTIVE_COMPARISON_MANUAL_SITE_CHECK_REQUIRED"
        records.append({
            "park_candidate_id": park_id, "city": group.get("registered_city", ""),
            "operator": group.get("operator", ""), "member_ids": "|".join(members),
            "member_installations": len(members), "registered_points_combined": int(group["sum_registered_points"]),
            "member_power_bands": group["member_power_bands"], "max_pair_distance_m": round(spread_m, 1) if pd.notna(spread_m) else np.nan,
            "location_check": location_status, "location_note": reason,
            "registered_members_quality_pass": qualified,
            "synchronized_windows": n, "synchronized_days": days, "observed_hours_of_day": hours,
            "mean_aligned_occupancy_pct": round(float(complete.snapshot_occupancy_pct.mean()), 2) if n else np.nan,
            "no_free_aligned_snapshot_pct": round(float(100 * complete.no_free_reported.mean()), 2) if n else np.nan,
            "comparison_readiness": review,
            "park_identity": "POTENTIAL_GROUP_NOT_CONFIRMED",
            "manual_review": "Verify charger park boundary, ownership, hardware layout, shared electrical infrastructure and site coordinates",
        })
    summary = pd.DataFrame(records)
    if summary.empty:
        raise ValueError("No possible park groups found in preceding output")
    window_data = pd.concat(aligned, ignore_index=True) if aligned else pd.DataFrame(
        columns=["park_candidate_id", "window_unix", "members_observed", "occupied", "available", "total",
                 "earliest_ts", "latest_ts", "snapshot_occupancy_pct", "no_free_reported", "local_date", "local_hour"])
    return summary, window_data


def comparables(sites: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    out = []
    good = groups.loc[groups.comparison_readiness.eq("DESCRIPTIVE_COMPARISON_MANUAL_SITE_CHECK_REQUIRED")].copy()
    for site in sites.to_dict("records"):
        n = int(site["connector_count"])
        band = power_band(float(site["charger_power_kw"]))
        exact = good.loc[good.registered_points_combined.eq(n) & good.member_power_bands.eq(band)]
        near = good.loc[good.registered_points_combined.between(max(1, n-2), n+2) & good.member_power_bands.eq(band)]
        out.append({
            "site_id": site["site_id"], "city": site["city"], "simulated_connectors": n,
            "simulated_power_band": band,
            "usable_potential_park_groups_exact": len(exact),
            "usable_potential_park_groups_within_2_points": len(near),
            "median_real_snapshot_occupied_pct_exact": round(float(exact.mean_aligned_occupancy_pct.median()), 2) if len(exact) else np.nan,
            "comparison_status": "DESCRIPTIVE_ONLY_UNCONFIRMED_PARKS_DIFFERENT_PERIODS" if len(exact) >= 5
                                 else "SMALL_OR_NO_COMPARABLE_PARK_SAMPLE",
        })
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Check park co-location and time-aligned German DC charging occupancy")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--bin-minutes", type=int, default=30)
    p.add_argument("--tolerance-minutes", type=int, default=10)
    p.add_argument("--max-park-span-m", type=float, default=35)
    p.add_argument("--min-aligned-windows", type=int, default=24)
    p.add_argument("--min-aligned-days", type=int, default=7)
    p.add_argument("--min-hours-of-day", type=int, default=8)
    p.add_argument("--chunksize", type=int, default=200000)
    args = p.parse_args()
    if (args.bin_minutes < 5 or not 0 < args.tolerance_minutes < args.bin_minutes / 2 or
        not 0 < args.max_park_span_m <= 100 or args.min_aligned_windows < 1 or
        args.min_aligned_days < 1 or not 1 <= args.min_hours_of_day <= 24 or args.chunksize < 100):
        p.error("Invalid time, distance, or quality settings")

    root = args.root.resolve()
    processed = root / "data" / "processed"
    parks = require(processed / "german_dc_potential_charging_parks.csv",
                    ["park_candidate_id", "member_ids", "sum_registered_points", "member_power_bands"])
    quality = require(processed / "german_dc_extended_station_quality.csv",
                      ["station_id", "registered_charging_points", "passes_quality"], dtype={"station_id": "string"})
    quality["station_id"] = clean_id(quality["station_id"])
    if quality.station_id.duplicated().any():
        raise ValueError("Duplicate station IDs in station quality table")
    sites = prepare_synthetic(root / "data" / "raw" / "site_master.csv")
    all_ids = set().union(*(set(members_of(x)) for x in parks.member_ids))
    print("\n--- REAL DC CHARGING PARK VERIFICATION ---")
    print(f"Candidate groups: {len(parks):,} | Distinct member installations: {len(all_ids):,}")
    registry_csv = select_file(root / "data" / "external", "Ladesaeulenregister_BNetzA*.csv")
    occupancy_csv = select_file(root / "data" / "external", "chargemyride*.csv")
    addresses = load_registry_addresses(registry_csv, all_ids, args.chunksize)
    expected_points = quality.set_index("station_id")["registered_charging_points"].apply(pd.to_numeric, errors="coerce").to_dict()
    print("Reading real German occupancy snapshots in chunks ...")
    observations = read_aligned_status(occupancy_csv, all_ids, expected_points, args.chunksize,
                                       args.bin_minutes, args.tolerance_minutes)
    summary, aligned = summarize_groups(parks, quality, addresses, observations, args)
    comparison = comparables(sites, summary)
    processed.mkdir(parents=True, exist_ok=True)
    paths = {
        "review": processed / "german_dc_park_group_review.csv",
        "aligned": processed / "german_dc_park_time_aligned_occupancy.csv",
        "comparison": processed / "german_dc_park_simulation_comparables.csv",
    }
    summary.to_csv(paths["review"], index=False)
    aligned.to_csv(paths["aligned"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    print("\nLocation/group review:")
    print(summary.location_check.value_counts().to_string())
    print("\nTime-aligned data readiness:")
    print(summary.comparison_readiness.value_counts().to_string())
    print(f"Synchronized group snapshots: {len(aligned):,}")
    print("\n--- COMPARABLE PARK SCREEN ---")
    print(comparison[["city", "simulated_connectors", "simulated_power_band",
                      "usable_potential_park_groups_exact", "usable_potential_park_groups_within_2_points"]].to_string(index=False))
    print("\nSaved:")
    for path in paths.values():
        print(path)
    print("\nIMPORTANT: NO park is confirmed without external/manual site verification.")
    print("Combined occupancy uses ONLY closely time-aligned, valid observations.")
    print("Snapshot occupancy cannot validate arrivals, rejected cars, grid capacity or financial returns.")


if __name__ == "__main__":
    main()
