"""Summarize the 2026 chargemyride German charging-point status observations.

Run from the CPO Network Development project root:
    python src\analyze_german_occupancy.py

Reads the large CSV in chunks. Does NOT claim to observe EV arrivals, queues,
rejections, actual energy sold, or time-weighted utilization. Empty status fields
remain missing rather than being turned into zero.

Source: https://chargemyride.live/datensatz (record download date in filename).
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FOLDER = ROOT / "data" / "external"
OUTPUT_FOLDER = ROOT / "data" / "processed"

SOURCE_COLUMNS = [
    "station_id", "ts_unix", "available", "occupied", "reserved", "total",
    "latitude", "longitude", "out_of_service", "unknown"
]
STATUS_COLUMNS = ["available", "occupied", "reserved", "total", "out_of_service", "unknown"]


def pick_source(source_argument):
    if source_argument:
        file = Path(source_argument)
        if not file.is_absolute():
            file = ROOT / file
        if not file.is_file():
            raise FileNotFoundError(f"CSV not found: {file}")
        return file

    candidates = sorted(
        SOURCE_FOLDER.glob("chargemyride*.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No chargemyride*.csv in {SOURCE_FOLDER}")
    return candidates[0]


def require_headers(path):
    headers = pd.read_csv(path, comment="#", nrows=0, encoding="utf-8-sig").columns.tolist()
    missing = sorted(set(SOURCE_COLUMNS).difference(headers))
    if missing:
        raise ValueError(f"Missing required CSV headers: {missing}. Found: {headers}")


def accumulate_station_counts(summary_dict, chunk):
    # CSV blanks remain NaN. The data provider explicitly warns blanks ≠ zero.
    chunk["station_id"] = chunk["station_id"].str.strip()
    usable = chunk["station_id"].notna() & (chunk["station_id"] != "")
    skipped = int((~usable).sum())
    chunk = chunk.loc[usable].copy()
    if chunk.empty:
        return skipped

    for column in ["ts_unix", *STATUS_COLUMNS, "latitude", "longitude"]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    available = chunk["available"]
    occupied = chunk["occupied"]
    known = available + occupied
    valid = available.ge(0) & occupied.ge(0)
    nonzero_known = valid & known.gt(0)

    # These percentages describe OBSERVATIONS, not a continuous time duration.
    occupied_pct = occupied.div(known.where(nonzero_known)).mul(100)
    total = chunk["total"]

    frame = pd.DataFrame({
        "station_id": chunk["station_id"],
        "valid_status": valid.astype("int8"),
        "occupied_pct_sum": occupied_pct.fillna(0),
        "occupied_pct_count": nonzero_known.astype("int8"),
        "zero_free": (valid & available.eq(0) & occupied.gt(0)).astype("int8"),
        "unknown_positive": chunk["unknown"].gt(0).astype("int8"),
        "unknown_missing": chunk["unknown"].isna().astype("int8"),
        "out_of_service_positive": chunk["out_of_service"].gt(0).astype("int8"),
        "reserved_positive": chunk["reserved"].gt(0).astype("int8"),
        "reported_count_exceeds_total": (
            valid & total.ge(0) & known.gt(total)
        ).astype("int8"),
        "ts_unix": chunk["ts_unix"],
        "latitude": chunk["latitude"],
        "longitude": chunk["longitude"],
    })

    groups = frame.groupby("station_id", sort=False, observed=True).agg(
        observations=("valid_status", "size"),
        valid_status_observations=("valid_status", "sum"),
        zero_free_observations=("zero_free", "sum"),
        pct_sum=("occupied_pct_sum", "sum"),
        pct_count=("occupied_pct_count", "sum"),
        unknown_positive_observations=("unknown_positive", "sum"),
        unknown_missing_observations=("unknown_missing", "sum"),
        outage_reported_observations=("out_of_service_positive", "sum"),
        reserved_reported_observations=("reserved_positive", "sum"),
        count_inconsistency_observations=("reported_count_exceeds_total", "sum"),
        first_ts=("ts_unix", "min"),
        last_ts=("ts_unix", "max"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )

    additive = [
        "observations", "valid_status_observations", "zero_free_observations",
        "pct_sum", "pct_count", "unknown_positive_observations",
        "unknown_missing_observations", "outage_reported_observations",
        "reserved_reported_observations", "count_inconsistency_observations",
    ]
    for station_id, record in groups.iterrows():
        if station_id not in summary_dict:
            summary_dict[station_id] = record.to_dict()
            continue
        result = summary_dict[station_id]
        for key in additive:
            result[key] += record[key]
        for key in ("first_ts",):
            if pd.notna(record[key]):
                result[key] = min(result[key], record[key]) if pd.notna(result[key]) else record[key]
        for key in ("last_ts",):
            if pd.notna(record[key]):
                result[key] = max(result[key], record[key]) if pd.notna(result[key]) else record[key]
        for key in ("latitude", "longitude"):
            if pd.isna(result[key]) and pd.notna(record[key]):
                result[key] = record[key]
    return skipped


def main():
    parser = argparse.ArgumentParser(description="Chunked summary of German occupancy observations")
    parser.add_argument("--input", default=None, help="Optional custom path to a chargemyride CSV")
    parser.add_argument("--chunksize", type=int, default=150_000)
    args = parser.parse_args()
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")

    file = pick_source(args.input)
    require_headers(file)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print("\n--- REAL GERMAN CHARGING OCCUPANCY IMPORT ---")
    print(f"Source: {file.name} ({file.stat().st_size / 1024**2:.1f} MB)")

    stations = {}
    row_count = 0
    skipped_count = 0
    for i, part in enumerate(pd.read_csv(
        file, comment="#", encoding="utf-8-sig", usecols=SOURCE_COLUMNS,
        dtype={"station_id": "string"}, chunksize=args.chunksize,
        low_memory=False,
    ), start=1):
        row_count += len(part)
        skipped_count += accumulate_station_counts(stations, part)
        if i % 10 == 0:
            print(f"Processed {row_count:,} observations...")

    if not stations:
        raise ValueError("No usable station records were found")

    summary = pd.DataFrame.from_dict(stations, orient="index")
    summary.index.name = "station_id"
    summary = summary.reset_index()
    summary["observed_occupied_pct"] = (
        100 * summary["zero_free_observations"] /
        summary["valid_status_observations"].replace(0, np.nan)
    ).round(2)
    # 'observed_occupied_pct' above = zero-free share, not occupied share.
    summary = summary.rename(columns={"observed_occupied_pct": "reported_no_free_snapshot_pct"})
    summary["mean_known_point_occupancy_snapshot_pct"] = (
        summary["pct_sum"] / summary["pct_count"].replace(0, np.nan)
    ).round(2)
    summary["observed_span_days"] = (
        (summary["last_ts"] - summary["first_ts"]) / 86400
    ).round(1)

    for field in ("first", "last"):
        summary[f"{field}_observed_local"] = pd.to_datetime(
            summary[f"{field}_ts"], unit="s", utc=True, errors="coerce"
        ).dt.tz_convert("Europe/Berlin").dt.strftime("%Y-%m-%d %H:%M")

    # Candidate ranking = data availability only, NOT "most congested".
    candidates = summary.loc[
        (summary["valid_status_observations"] >= 30) &
        (summary["observed_span_days"] >= 7)
    ].sort_values("valid_status_observations", ascending=False).head(50).copy()

    # Preserve counts and explicit missingness for auditing.
    keep = [
        "station_id", "observations", "valid_status_observations",
        "first_observed_local", "last_observed_local", "observed_span_days",
        "reported_no_free_snapshot_pct", "mean_known_point_occupancy_snapshot_pct",
        "zero_free_observations", "unknown_positive_observations",
        "unknown_missing_observations", "outage_reported_observations",
        "reserved_reported_observations", "count_inconsistency_observations",
        "latitude", "longitude",
    ]
    all_path = OUTPUT_FOLDER / "german_occupancy_station_summary.csv"
    candidates_path = OUTPUT_FOLDER / "german_occupancy_candidate_stations.csv"
    summary[keep].sort_values("valid_status_observations", ascending=False).to_csv(all_path, index=False)
    candidates[keep].to_csv(candidates_path, index=False)

    print(f"Records read: {row_count:,} | Station IDs: {len(summary):,}")
    print(f"Records with missing station ID: {skipped_count:,}")
    print(f"Candidate stations (30+ status records across 7+ days): {len(candidates)}")
    if not candidates.empty:
        print("\n--- EXAMPLE REAL STATIONS (TOP 5 BY SAMPLE COUNT) ---")
        print(candidates[[
            "station_id", "valid_status_observations", "observed_span_days",
            "reported_no_free_snapshot_pct",
        ]].head().to_string(index=False))
    print("\nSaved:")
    print(all_path)
    print(candidates_path)
    print("\nNOTE: Percentages count REPORTED SNAPSHOTS, not time-weighted occupancy.")
    print("No observed car arrivals, waiting times, rejected drivers or energy sales.")
    print("Missing reserved/outage/unknown values were NOT treated as zero.")
    print("Coverage is selective and irregular; this is an inspection step, not yet validation.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
