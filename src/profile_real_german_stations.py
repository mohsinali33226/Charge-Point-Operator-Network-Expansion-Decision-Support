"""Profile observed German charging-point occupancy for candidate stations.

Reads chargeMyRide status records in chunks, retaining only station IDs selected
by analyze_german_occupancy.py. Creates quality and hourly snapshot profiles.

IMPORTANT: Point-in-time occupancy is NOT car arrivals, rejection, queueing,
or time-weighted utilization. Missing status fields are not assumed to be zero.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"
CANDIDATES = PROCESSED / "german_occupancy_candidate_stations.csv"
PROFILE_OUT = PROCESSED / "german_real_station_hourly_profiles.csv"
QUALITY_OUT = PROCESSED / "german_real_station_quality_checks.csv"
NEEDED = [
    "station_id", "ts_unix", "available", "occupied", "total",
    "out_of_service", "reserved", "unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real German occupancy by local hour")
    parser.add_argument("--input", type=Path, default=None, help="Optional raw CSV path")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--max-stations", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser.parse_args()


def choose_source(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Raw CSV not found: {explicit}")
        return explicit
    paths = sorted(EXTERNAL.glob("chargemyride*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No chargemyride CSV in {EXTERNAL}")
    return paths[0]


def read_candidate_ids(path: Path, limit: int) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Run analyze_german_occupancy.py first: {path}")
    if limit < 1:
        raise ValueError("--max-stations must be positive")
    df = pd.read_csv(path)
    if "station_id" not in df.columns:
        raise ValueError("Candidate CSV needs a station_id column")
    ids = pd.to_numeric(df["station_id"], errors="coerce").dropna().astype("int64")
    picked = ids.drop_duplicates().head(limit)
    if picked.empty:
        raise ValueError("No valid candidate station IDs")
    return set(picked.tolist())


def collect_rows(source: Path, candidate_ids: set[int], chunk_size: int) -> pd.DataFrame:
    kept = []
    total = 0
    try:
        stream = pd.read_csv(
            source, comment="#", usecols=NEEDED, chunksize=chunk_size,
            low_memory=False, encoding="utf-8-sig",
        )
        for chunk in stream:
            total += len(chunk)
            ids = pd.to_numeric(chunk["station_id"], errors="coerce")
            selected = chunk.loc[ids.isin(candidate_ids)].copy()
            if not selected.empty:
                kept.append(selected)
            if total % 1_000_000 < chunk_size:
                print(f"Read {total:,} raw measurements...")
    except ValueError as exc:
        raise ValueError(f"Missing required column in raw data: {exc}") from exc
    if not kept:
        raise ValueError("No raw records found for candidate station IDs")
    data = pd.concat(kept, ignore_index=True)
    for name in NEEDED:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.dropna(subset=["station_id", "ts_unix"])
    data = data.drop_duplicates(["station_id", "ts_unix"], keep="last")
    data["station_id"] = data["station_id"].astype("int64")
    return data


def build_profiles(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Preserve missing fields as NaN. Only complete, internally consistent
    # available/occupied/total statuses contribute to occupancy measures.
    valid = (
        data[["available", "occupied", "total"]].notna().all(axis=1)
        & data["available"].ge(0)
        & data["occupied"].ge(0)
        & data["total"].gt(0)
        & (data["available"] + data["occupied"] <= data["total"])
    )
    data = data.copy()
    data["valid_status"] = valid
    # Epoch seconds are timezone-unambiguous (also on German DST change days).
    data["local_time"] = pd.to_datetime(
        data["ts_unix"], unit="s", utc=True, errors="coerce"
    ).dt.tz_convert("Europe/Berlin")
    data = data.dropna(subset=["local_time"])
    data["local_hour"] = data["local_time"].dt.hour
    data["local_date"] = data["local_time"].dt.date
    data["weekday"] = data["local_time"].dt.dayofweek

    data_valid = data.loc[data["valid_status"]].copy()
    if data_valid.empty:
        raise ValueError("None of the selected station rows has complete consistent statuses")
    data_valid["occupied_share"] = data_valid["occupied"] / data_valid["total"]
    data_valid["no_available"] = data_valid["available"].eq(0).astype(float)
    data_valid["all_occupied"] = data_valid["occupied"].eq(data_valid["total"]).astype(float)

    hourly = (
        data_valid.groupby(["station_id", "local_hour"], as_index=False)
        .agg(
            valid_snapshots=("no_available", "size"),
            mean_occupied_share=("occupied_share", "mean"),
            no_available_snapshot_fraction=("no_available", "mean"),
            all_occupied_snapshot_fraction=("all_occupied", "mean"),
        )
        .sort_values(["station_id", "local_hour"])
    )
    for name in [
        "mean_occupied_share", "no_available_snapshot_fraction", "all_occupied_snapshot_fraction"
    ]:
        hourly[name.replace("fraction", "pct") if "fraction" in name else "mean_occupied_pct"] = (
            hourly[name] * 100
        ).round(2)
    hourly = hourly.drop(columns=[
        "mean_occupied_share", "no_available_snapshot_fraction", "all_occupied_snapshot_fraction"
    ])
    hourly["low_sample_flag"] = hourly["valid_snapshots"] < 5

    stat = data.groupby("station_id").agg(
        raw_unique_snapshots=("ts_unix", "size"),
        total_reported_nonmissing=("total", "count"),
        outage_status_nonmissing=("out_of_service", "count"),
        unknown_status_nonmissing=("unknown", "count"),
        reserved_status_nonmissing=("reserved", "count"),
    )
    qual = data_valid.groupby("station_id").agg(
        valid_snapshots=("valid_status", "size"),
        observed_days=("local_date", "nunique"),
        observed_hours_of_day=("local_hour", "nunique"),
        first_observation=("local_time", "min"),
        last_observation=("local_time", "max"),
        typical_total_points=("total", lambda x: float(x.mode().iloc[0])),
        mean_occupied_pct=("occupied_share", lambda x: round(x.mean() * 100, 2)),
        no_available_snapshot_pct=("no_available", lambda x: round(x.mean() * 100, 2)),
        all_occupied_snapshot_pct=("all_occupied", lambda x: round(x.mean() * 100, 2)),
    )
    qual = stat.join(qual, how="left").reset_index()
    qual["valid_status_pct"] = (100 * qual["valid_snapshots"] / qual["raw_unique_snapshots"]).round(2)
    qual["outage_status_recorded_pct"] = (
        100 * qual["outage_status_nonmissing"] / qual["raw_unique_snapshots"]
    ).round(2)
    qual["validation_candidate"] = (
        qual["valid_snapshots"].ge(200)
        & qual["observed_days"].ge(7)
        & qual["observed_hours_of_day"].ge(20)
    )
    qual["validation_scope"] = "OCCUPANCY_SNAPSHOT_PATTERN_ONLY"
    qual["first_observation"] = qual["first_observation"].astype(str)
    qual["last_observation"] = qual["last_observation"].astype(str)
    return hourly, qual.sort_values("valid_snapshots", ascending=False)


def main() -> None:
    args = parse_args()
    source = choose_source(args.input)
    ids = read_candidate_ids(args.candidates, args.max_stations)
    print("\n--- REAL GERMAN STATION HOURLY OCCUPANCY ---")
    print(f"Raw source: {source.name}")
    print(f"Candidate station IDs: {len(ids)}")
    rows = collect_rows(source, ids, args.chunk_size)
    hourly, quality = build_profiles(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(PROFILE_OUT, index=False)
    quality.to_csv(QUALITY_OUT, index=False)
    print(f"Selected station snapshots after deduplication: {len(rows):,}")
    print(f"Stations with valid occupancy: {len(quality)}")
    print(f"Stations meeting basic observation checks: {int(quality['validation_candidate'].sum())}")
    cols = [
        "station_id", "valid_snapshots", "observed_days", "typical_total_points",
        "mean_occupied_pct", "no_available_snapshot_pct", "validation_candidate",
    ]
    print("\n--- FIRST 5 STATIONS ---")
    print(quality[cols].head(5).to_string(index=False))
    print("\nSaved:", PROFILE_OUT, "\nSaved:", QUALITY_OUT)
    print("\nNOTE: Snapshot occupancy is NOT waiting, refused cars, or time-weighted utilization.")
    print("A station's charger type and rated power still need infrastructure matching.")


if __name__ == "__main__":
    main()
