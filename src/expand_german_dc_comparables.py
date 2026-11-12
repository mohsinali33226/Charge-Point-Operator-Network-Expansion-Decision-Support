r"""Expand real German DC equipment comparisons beyond the first 100 stations.

Run from the CPO Network Development project root:
    python src\expand_german_dc_comparables.py

Inputs (already produced/downloaded in this project):
    data/processed/german_dc_occupancy_inventory.csv
    data/external/chargemyride*.csv
    data/external/Ladesaeulenregister_BNetzA*.csv
    data/raw/site_master.csv

Outputs (NEW files; previous results are not overwritten):
    data/processed/german_dc_extended_station_quality.csv
    data/processed/german_dc_extended_hourly_profiles.csv
    data/processed/german_dc_potential_charging_parks.csv
    data/processed/german_dc_similar_site_candidates.csv
    data/processed/german_dc_comparability_summary.csv

Caution: Status observations are irregular snapshots. They do NOT measure
arrivals, rejected drivers, kWh, or actual grid capacity. BNetzA IDs identify
installations rather than necessarily whole charging parks. Nearby installations
are *candidate groups*, NOT verified park boundaries. Their occupancy must not
be naively added across asynchronous timestamps.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    # Reuse the same careful data-quality logic as the prior step.
    from find_german_dc_validation_stations import read_raw_quality
except ImportError as exc:
    raise SystemExit(
        "Missing src/find_german_dc_validation_stations.py. "
        "Keep this script next to the earlier DC discovery script."
    ) from exc


def clean_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.strip('"').str.replace(r"\.0+$", "", regex=True)


def to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.strip().str.replace(",", ".", regex=False), errors="coerce")


def clean_text(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.strip().str.casefold()
            .str.replace(r"\s+", " ", regex=True))


def select_file(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        raise FileNotFoundError(f"Cannot find {pattern} in {directory}")
    return found[0]


def read_table(path: Path, required: list[str], **kw) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path, low_memory=False, **kw)
    df.columns = df.columns.str.strip()
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df


def bnetza_header(path: Path) -> tuple[str, int, list[str]]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open(encoding=encoding, newline="") as f:
                for n, line in enumerate(f):
                    if n > 80:
                        break
                    cells = [v.strip().lstrip("\ufeff") for v in next(csv.reader([line], delimiter=";"))]
                    if cells and cells[0] == "Ladeeinrichtungs-ID":
                        return encoding, n, cells
        except UnicodeDecodeError:
            pass
    raise ValueError("BNetzA: could not find Ladeeinrichtungs-ID header")


def load_registry_addresses(path: Path, wanted: set[str], chunksize: int) -> pd.DataFrame:
    encoding, skiprows, headers = bnetza_header(path)
    desired = ["Ladeeinrichtungs-ID", "Betreiber", "Ort", "Postleitzahl", "Straße", "Hausnummer", "Breitengrad", "Längengrad"]
    cols = [c for c in desired if c in headers]
    if "Ladeeinrichtungs-ID" not in cols or not {"Breitengrad", "Längengrad"}.issubset(cols):
        raise ValueError("BNetzA needs IDs and coordinates to identify potential same-park groups")
    batches = []
    for part in pd.read_csv(path, sep=";", skiprows=skiprows, encoding=encoding,
                            dtype="string", usecols=cols, chunksize=chunksize, low_memory=False):
        ids = clean_id(part["Ladeeinrichtungs-ID"])
        chosen = part.loc[ids.isin(wanted)].copy()
        if not chosen.empty:
            chosen["station_id"] = ids.loc[chosen.index]
            batches.append(chosen)
    if not batches:
        raise ValueError("No DC station IDs were found in the official register")
    addresses = pd.concat(batches, ignore_index=True)
    if addresses.station_id.duplicated().any():
        # An ambiguous registry ID must NOT be treated as a verified installation.
        duplicate = set(addresses.loc[addresses.station_id.duplicated(False), "station_id"])
        print(f"Warning: excluding {len(duplicate)} duplicate registry IDs from park grouping")
        addresses = addresses.loc[~addresses.station_id.isin(duplicate)].copy()
    rename = {"Betreiber": "operator", "Ort": "registered_city", "Postleitzahl": "postcode",
              "Straße": "street", "Hausnummer": "building_number", "Breitengrad": "latitude",
              "Längengrad": "longitude"}
    addresses = addresses.rename(columns=rename)
    for col in ["operator", "registered_city", "postcode", "street", "building_number"]:
        if col not in addresses:
            addresses[col] = ""
    addresses["latitude"] = to_number(addresses["latitude"])
    addresses["longitude"] = to_number(addresses["longitude"])
    for col in ["operator", "registered_city", "postcode", "street", "building_number"]:
        addresses[col + "_normalized"] = clean_text(addresses[col])
    return addresses


def distance_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    phi1, phi2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    t = math.sin(dp / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2)**2
    return 6371000 * 2 * math.atan2(math.sqrt(t), math.sqrt(max(0, 1-t)))


def possible_same_park(a: dict, b: dict, max_distance: float) -> tuple[bool, str]:
    if (not a["operator_normalized"] or a["operator_normalized"] != b["operator_normalized"]):
        return False, ""
    if any(pd.isna(x) for x in (a["latitude"], a["longitude"], b["latitude"], b["longitude"])):
        return False, ""
    if not (-90 <= a["latitude"] <= 90 and -90 <= b["latitude"] <= 90 and
            -180 <= a["longitude"] <= 180 and -180 <= b["longitude"] <= 180):
        return False, ""
    d = distance_m(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
    if d > max_distance:
        return False, ""
    postal1, postal2 = a["postcode_normalized"], b["postcode_normalized"]
    if postal1 and postal2 and postal1 != postal2:
        return False, ""
    street1, street2 = a["street_normalized"], b["street_normalized"]
    num1, num2 = a["building_number_normalized"], b["building_number_normalized"]
    if postal1 and postal1 == postal2 and street1 and street1 == street2:
        if num1 and num2 and num1 != num2:
            return False, ""
        return True, "SAME_OPERATOR_STREET_POSTCODE_WITHIN_DISTANCE"
    if (a["registered_city_normalized"] and
        a["registered_city_normalized"] == b["registered_city_normalized"] and d <= 12):
        return True, "SAME_OPERATOR_VERY_CLOSE_COORDINATES"
    return False, ""


def make_park_groups(quality: pd.DataFrame, addresses: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    # Important: group only installations with independently usable status data.
    # Both the DC inventory and the BNetzA address table can contain columns such
    # as latitude, longitude and registered_city. A plain merge adds _x/_y and
    # breaks access to the expected column names. For park proximity, explicitly
    # prefer the latest official registry coordinates and address metadata.
    registry_fields = [col for col in addresses.columns if col != "station_id"]
    overlapping_fields = [col for col in registry_fields if col in quality.columns]
    station_quality = quality.drop(columns=overlapping_fields)
    entries = station_quality.merge(addresses, on="station_id", how="left", validate="one_to_one")
    valid = entries.loc[
        entries["latitude"].notna() & entries["longitude"].notna() &
        entries["operator_normalized"].fillna("").ne("")
    ].copy().reset_index(drop=True)
    if valid.empty:
        return pd.DataFrame(columns=["park_candidate_id", "registered_city", "operator", "member_installations",
                                     "member_ids", "sum_registered_points", "sum_equipment_rated_kw_proxy",
                                     "min_valid_snapshots_per_member", "min_observed_days_per_member",
                                     "mean_member_snapshot_occupancy_pct", "co_location_evidence", "group_status"])
    # A separate registered installation may be a separate operational site:
    # grouping is only an investigative hint.
    rows = valid.to_dict("records")
    parent = list(range(len(rows)))
    rank = [0] * len(rows)
    evidence = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j, why):
        r1, r2 = find(i), find(j)
        if r1 == r2:
            return
        if rank[r1] < rank[r2]:
            r1, r2 = r2, r1
        parent[r2] = r1
        if rank[r1] == rank[r2]:
            rank[r1] += 1
        evidence.setdefault(r1, set()).update(evidence.pop(r2, set()))
        evidence[r1].add(why)

    # Pairwise only within same operator and city: no national-scale quadratic scan.
    valid["scope"] = valid.operator_normalized.astype(str) + "|" + valid.registered_city_normalized.astype(str)
    for _, loc in valid.groupby("scope", sort=False):
        indices = loc.index.tolist()
        for pos, i in enumerate(indices):
            for j in indices[pos+1:]:
                same, why = possible_same_park(rows[i], rows[j], max_distance)
                if same:
                    union(i, j, why)

    components: dict[int, list[int]] = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    output = []
    serial = 0
    for root, member_indices in sorted(components.items(), key=lambda kv: min(rows[i]["station_id"] for i in kv[1])):
        if len(member_indices) < 2:
            continue
        serial += 1
        members = [rows[i] for i in member_indices]
        powers = [float(m["registered_equipment_power_kw"]) for m in members]
        member_classes = sorted({power_band(x) for x in powers})
        output.append({
            "park_candidate_id": f"PARK_CANDIDATE_{serial:05d}",
            "registered_city": members[0].get("registered_city", ""),
            "operator": members[0].get("operator", ""),
            "member_installations": len(members),
            "member_ids": "|".join(sorted(str(m["station_id"]) for m in members)),
            "sum_registered_points": int(sum(m["registered_charging_points"] for m in members)),
            "sum_equipment_rated_kw_proxy": round(sum(powers), 1),
            "median_member_equipment_kw": float(np.median(powers)),
            "member_power_bands": "|".join(member_classes),
            "min_valid_snapshots_per_member": int(min(m["valid_snapshots"] for m in members)),
            "min_observed_days_per_member": int(min(m["observed_days"] for m in members)),
            "mean_member_snapshot_occupancy_pct": round(float(np.mean([m["mean_occupied_snapshot_pct"] for m in members])), 2),
            "co_location_evidence": "|".join(sorted(evidence.get(root, {"PROXIMITY"}))),
            "group_status": "POTENTIAL_SAME_PARK_MANUAL_REVIEW_REQUIRED",
        })
    return pd.DataFrame(output, columns=[
        "park_candidate_id", "registered_city", "operator", "member_installations", "member_ids",
        "sum_registered_points", "sum_equipment_rated_kw_proxy", "median_member_equipment_kw",
        "member_power_bands", "min_valid_snapshots_per_member", "min_observed_days_per_member",
        "mean_member_snapshot_occupancy_pct", "co_location_evidence", "group_status",
    ])


def power_band(x: float) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    if x >= 300:
        return "300_PLUS"
    if x >= 150:
        return "150_TO_299"
    if x >= 50:
        return "50_TO_149"
    return "BELOW_50"


def prepare_synthetic(site_file: Path) -> pd.DataFrame:
    sites = read_table(site_file, ["site_id", "city", "charger_count", "charger_power_kw"])
    if "connector_count" not in sites:
        if "connectors_per_charger" not in sites:
            raise ValueError("site_master.csv: connector_count or connectors_per_charger is required")
        sites["connector_count"] = to_number(sites.charger_count) * to_number(sites.connectors_per_charger)
    for col in ["connector_count", "charger_power_kw", "charger_count"]:
        sites[col] = to_number(sites[col])
    if (sites[["connector_count", "charger_power_kw", "charger_count"]].isna().any().any() or
            sites.connector_count.le(0).any() or sites.charger_power_kw.le(0).any()):
        raise ValueError("Invalid synthetic site connector or power values")
    sites["power_band"] = sites.charger_power_kw.map(power_band)
    return sites[["site_id", "city", "charger_count", "connector_count", "charger_power_kw", "power_band"]].copy()


def make_comparison(sites: pd.DataFrame, qualified: pd.DataFrame, parks: pd.DataFrame, max_rows_per_site: int):
    candidates = []
    summaries = []
    for site in sites.to_dict("records"):
        n = int(site["connector_count"])
        band = site["power_band"]
        real_exact = qualified.loc[
            qualified.registered_charging_points.eq(n) & qualified.equipment_power_band.eq(band)
        ].copy()
        real_approx = qualified.loc[
            qualified.registered_charging_points.between(max(1, n-2), n+2) &
            qualified.equipment_power_band.eq(band)
        ].copy()
        if parks.empty:
            park_exact = parks
            park_approx = parks
        else:
            fits_power = parks.member_power_bands.astype(str).str.split("|").map(lambda xs: set(xs) == {band})
            park_exact = parks.loc[parks.sum_registered_points.eq(n) & fits_power].copy()
            park_approx = parks.loc[
                parks.sum_registered_points.between(max(1,n-2),n+2) & fits_power
            ].copy()
        summaries.append({
            "site_id": site["site_id"], "city": site["city"],
            "synthetic_connector_count": n, "synthetic_equipment_power_kw": site["charger_power_kw"],
            "synthetic_power_band": band, "quality_checked_dc_installations": len(qualified),
            "exact_equipment_installation_matches": len(real_exact),
            "near_equipment_installation_matches": len(real_approx),
            "possible_same_park_groups_total": len(parks),
            "exact_point_count_park_candidates": len(park_exact),
            "near_point_count_park_candidates": len(park_approx),
            "interpretation": "DESCRIPTIVE_ONLY_DIFFERENT_SITES_PERIODS_AND_SYSTEM_BOUNDARIES",
        })
        for row in real_approx.sort_values("valid_snapshots", ascending=False).head(max_rows_per_site).to_dict("records"):
            candidates.append({
                "synthetic_site_id": site["site_id"], "synthetic_city": site["city"],
                "synthetic_connectors": n, "synthetic_power_band": band,
                "real_candidate_type": "SINGLE_REGISTERED_INSTALLATION",
                "real_candidate_id": row["station_id"], "real_registered_city": row.get("registered_city", ""),
                "real_charging_points": int(row["registered_charging_points"]),
                "real_power_band_member": row["equipment_power_band"],
                "valid_snapshots": int(row["valid_snapshots"]),
                "observed_days": int(row["observed_days"]),
                "real_snapshot_occupied_pct_descriptor": row["mean_occupied_snapshot_pct"],
                "point_count_match": "EXACT" if row["registered_charging_points"] == n else "NEAR",
                "comparison_warning": "REAL_SNAPSHOT_VS_SYNTHETIC_HOURLY_NOT_DIRECT_VALIDATION",
            })
        for row in park_approx.sort_values("min_valid_snapshots_per_member", ascending=False).head(max_rows_per_site).to_dict("records"):
            candidates.append({
                "synthetic_site_id": site["site_id"], "synthetic_city": site["city"],
                "synthetic_connectors": n, "synthetic_power_band": band,
                "real_candidate_type": "POTENTIAL_COLOCATED_INSTALLATIONS",
                "real_candidate_id": row["park_candidate_id"], "real_registered_city": row["registered_city"],
                "real_charging_points": int(row["sum_registered_points"]),
                "real_power_band_member": row["member_power_bands"],
                "valid_snapshots": int(row["min_valid_snapshots_per_member"]),
                "observed_days": int(row["min_observed_days_per_member"]),
                "real_snapshot_occupied_pct_descriptor": np.nan,
                "point_count_match": "EXACT" if row["sum_registered_points"] == n else "NEAR",
                "comparison_warning": "PARK_GROUP_UNVERIFIED_AND_MEMBER_OCCUPANCY_NOT_TIME_ALIGNED",
            })
    return pd.DataFrame(candidates), pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description="Expand German DC comparison sample, and identify possible same-site charging installations")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--min-snapshots", type=int, default=200)
    parser.add_argument("--min-days", type=int, default=14)
    parser.add_argument("--min-hours", type=int, default=18)
    parser.add_argument("--min-power-kw", type=float, default=50)
    parser.add_argument("--max-park-distance-m", type=float, default=35)
    parser.add_argument("--max-comparables-per-site", type=int, default=15)
    parser.add_argument("--chunksize", type=int, default=150000)
    args = parser.parse_args()
    if (args.min_snapshots < 1 or args.min_days < 1 or args.min_hours < 1 or args.min_hours > 24
        or args.min_power_kw < 0 or not 0 < args.max_park_distance_m <= 100
        or args.max_comparables_per_site < 1 or args.chunksize < 100):
        parser.error("Invalid screening settings")

    root = args.root.resolve()
    output = root / "data" / "processed"
    inventory_file = output / "german_dc_occupancy_inventory.csv"
    raw_file = select_file(root / "data" / "external", "chargemyride*.csv")
    registry_file = select_file(root / "data" / "external", "Ladesaeulenregister_BNetzA*.csv")
    sites_file = root / "data" / "raw" / "site_master.csv"
    inventory = read_table(inventory_file, ["station_id", "registered_charging_points", "registered_equipment_power_kw",
                                             "valid_status_observations", "observed_span_days"], dtype={"station_id": "string"})
    inventory["station_id"] = clean_id(inventory.station_id)
    if inventory.station_id.duplicated().any():
        raise ValueError("Duplicate station IDs in DC inventory")
    for col in ["registered_charging_points", "registered_equipment_power_kw", "valid_status_observations", "observed_span_days"]:
        inventory[col] = to_number(inventory[col])
    if "registry_rows_for_id" in inventory:
        inventory["registry_rows_for_id"] = to_number(inventory.registry_rows_for_id)
        unique_registry = inventory.registry_rows_for_id.eq(1)
    else:
        unique_registry = pd.Series(True, index=inventory.index)
        print("Warning: registry_rows_for_id missing; duplicate registry source cannot be independently ruled out")
    preliminary = inventory.loc[
        unique_registry & inventory.registered_charging_points.ge(1)
        & inventory.registered_equipment_power_kw.ge(args.min_power_kw)
        & inventory.valid_status_observations.ge(args.min_snapshots)
        & inventory.observed_span_days.ge(args.min_days - 1)
    ].copy()
    if preliminary.empty:
        raise ValueError("No DC installations passed initial checks; inspect the inventory and quality limits")
    print("\n--- EXPANDED REAL GERMAN DC COMPARABILITY ---")
    print(f"Matched DC installation IDs in inventory: {len(inventory):,}")
    print(f"Eligible for full raw-history check: {len(preliminary):,}")
    print("Rechecking real status history across ALL eligible installations...")
    quality, hourly = read_raw_quality(raw_file, set(preliminary.station_id), args.chunksize)
    evaluated = preliminary.merge(quality, on="station_id", how="left", validate="one_to_one")
    evaluated["point_count_matches"] = evaluated.registered_charging_points.eq(evaluated.typical_reported_points)
    evaluated["passes_quality"] = (
        evaluated.valid_snapshots.ge(args.min_snapshots)
        & evaluated.observed_days.ge(args.min_days)
        & evaluated.observed_hours_of_day.ge(args.min_hours)
        & evaluated.point_count_matches.fillna(False)
    )
    evaluated["quality_status"] = np.select(
        [evaluated.valid_snapshots.isna(), ~evaluated.point_count_matches.fillna(False), ~evaluated.passes_quality],
        ["NO_VALID_STATUS", "POINT_COUNT_MISMATCH", "INSUFFICIENT_COVERAGE"],
        default="DESCRIPTIVE_OCCUPANCY_READY",
    )
    qualified = evaluated.loc[evaluated.passes_quality].copy()
    qualified["equipment_power_band"] = qualified.registered_equipment_power_kw.map(power_band)
    if qualified.empty:
        raise ValueError("No DC installations passed strict raw-history checks")
    print(f"Qualified real DC installations: {len(qualified):,} / {len(evaluated):,}")
    addresses = load_registry_addresses(registry_file, set(qualified.station_id), args.chunksize)
    parks = make_park_groups(qualified, addresses, args.max_park_distance_m)
    sites = prepare_synthetic(sites_file)
    similar, summary = make_comparison(sites, qualified, parks, args.max_comparables_per_site)

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "quality": output / "german_dc_extended_station_quality.csv",
        "hourly": output / "german_dc_extended_hourly_profiles.csv",
        "parks": output / "german_dc_potential_charging_parks.csv",
        "similar": output / "german_dc_similar_site_candidates.csv",
        "summary": output / "german_dc_comparability_summary.csv",
    }
    evaluated.to_csv(paths["quality"], index=False)
    hourly.loc[hourly.station_id.isin(set(qualified.station_id))].to_csv(paths["hourly"], index=False)
    parks.to_csv(paths["parks"], index=False)
    similar.to_csv(paths["similar"], index=False)
    summary.to_csv(paths["summary"], index=False)

    print(f"Possible co-located installation groups (NOT confirmed parks): {len(parks):,}")
    print("\n--- COMPARABLE EQUIPMENT / POSSIBLE CHARGING PARKS ---")
    print(summary[["city", "synthetic_connector_count", "synthetic_power_band",
                   "exact_equipment_installation_matches", "exact_point_count_park_candidates",
                   "near_point_count_park_candidates"]].to_string(index=False))
    print("\nSaved:")
    for path in paths.values():
        print(path)
    print("\nIMPORTANT: Possible parks need manual verification of site boundaries.")
    print("Do NOT sum member occupancy from unsynchronised status snapshots.")
    print("2026 German measurements vs 2025 synthetic sites = descriptive comparison ONLY.")
    print("No observed car arrivals, rejections, charging kWh or grid capacity in this dataset.")


if __name__ == "__main__":
    main()
