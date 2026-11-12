"""Leakage-safe, real-data DC charging occupancy backtest (Germany, 2026).

PURPOSE
  Establish whether a simple hour/weekend occupancy prediction trained on
  ACTUAL German DC status observations beats a station-average baseline on
  later, unseen observations. This is the first measurable validation gate
  before changing the project's synthetic arrival generator.

DATA LIMITATION
  ChargeMyRide reports irregular STATUS SNAPSHOTS, not car arrivals,
  charging sessions, rejected customers, charging energy or CPO grid import.
  Each station ID identifies a registered installation, not necessarily a
  charging park. No claim of validated arrivals or investment performance.

INPUTS (your existing project)
  data/external/chargemyride*.csv
  data/processed/german_dc_extended_station_quality.csv

OUTPUTS (new files; older models and datasets are untouched)
  data/processed/real_dc_2026_holdout_predictions.csv
  data/processed/real_dc_2026_station_backtest.csv
  data/processed/real_dc_2026_backtest_summary.csv
  data/processed/real_dc_2026_training_profiles.csv

Run from the project folder:
  python src\\validate_real_dc_forecasts.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED_RAW = ["station_id", "ts_unix", "available", "occupied", "total"]
DATA_STATUS = "REAL_OCCUPANCY_BACKTEST_ONLY_NOT_SESSION_OR_ARRIVAL_VALIDATION"


def choose_latest(directory: Path, pattern: str) -> Path:
    files = list(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def normalize_id(series: pd.Series) -> pd.Series:
    return (series.astype("string").str.strip().str.strip('"')
            .str.replace(r"\.0+$", "", regex=True))


def load_dc_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Run expand_german_dc_comparables.py first: {path}")
    df = pd.read_csv(path, dtype={"station_id": "string"}, low_memory=False)
    needed = {"station_id", "registered_equipment_power_kw", "registered_charging_points"}
    if missing := needed.difference(df.columns):
        raise ValueError(f"Real DC quality file missing {sorted(missing)}")
    df["station_id"] = normalize_id(df["station_id"])
    if "passes_quality" in df.columns:
        df = df.loc[df.passes_quality.astype("string").str.lower().eq("true")].copy()
    elif "quality_status" in df.columns:
        df = df.loc[df.quality_status.eq("DESCRIPTIVE_OCCUPANCY_READY")].copy()
    else:
        raise ValueError("Quality file must have passes_quality or quality_status")
    for col in ["registered_equipment_power_kw", "registered_charging_points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[
        df["registered_equipment_power_kw"].ge(50)
        & df["registered_charging_points"].ge(1)
        & df.station_id.notna()
    ].copy()
    if df.empty or df.station_id.duplicated().any():
        raise ValueError("No unique, quality-checked real DC installation IDs")
    df["power_band"] = np.select(
        [df.registered_equipment_power_kw.ge(300),
         df.registered_equipment_power_kw.ge(150)],
        ["300_PLUS", "150_TO_299"], default="50_TO_149"
    )
    df["registered_charging_points"] = df.registered_charging_points.astype(int)
    return df[["station_id", "registered_charging_points", "registered_equipment_power_kw", "power_band"]]


def load_real_hourly(raw_file: Path, registry: pd.DataFrame, chunksize: int) -> tuple[pd.DataFrame, int, int]:
    """Select real DC stations, validate snapshots, keep ONE observation per UTC hour."""
    desired_ids = set(registry.station_id)
    parts: list[pd.DataFrame] = []
    raw_read = 0
    filtered = 0
    registered_points = registry.set_index("station_id")["registered_charging_points"]
    print("Reading real DC observations in chunks ...")
    for i, chunk in enumerate(pd.read_csv(
        raw_file, comment="#", encoding="utf-8-sig", usecols=REQUIRED_RAW,
        dtype={"station_id": "string"}, chunksize=chunksize, low_memory=False
    ), start=1):
        raw_read += len(chunk)
        chunk["station_id"] = normalize_id(chunk.station_id)
        chunk = chunk.loc[chunk.station_id.isin(desired_ids)].copy()
        if chunk.empty:
            continue
        for field in ["ts_unix", "available", "occupied", "total"]:
            chunk[field] = pd.to_numeric(chunk[field], errors="coerce")
        chunk = chunk.dropna(subset=["ts_unix", "available", "occupied", "total"])
        if chunk.empty:
            continue
        # The official register provides the denominators. Do not call a
        # temporary equipment outage a new/smaller charging installation.
        actual_points = chunk.station_id.map(registered_points)
        valid = (
            chunk.ts_unix.between(1_500_000_000, 2_000_000_000)
            & chunk.total.eq(actual_points)
            & chunk.occupied.ge(0) & chunk.available.ge(0)
            & chunk.occupied.le(chunk.total)
            & (chunk.occupied + chunk.available).le(chunk.total)
        )
        chunk = chunk.loc[valid].copy()
        filtered += len(chunk)
        if not chunk.empty:
            chunk["utc_hour"] = np.floor(chunk.ts_unix / 3600).astype("int64")
            # Reduce the memory load inside each chunk, then dedupe globally.
            chunk = chunk.sort_values("ts_unix").drop_duplicates(
                ["station_id", "utc_hour"], keep="last"
            )
            parts.append(chunk[REQUIRED_RAW + ["utc_hour"]])
        if i % 10 == 0:
            print(f"  Scanned {raw_read:,} raw status records")
    if not parts:
        raise ValueError("No usable real DC occupancy observations found")
    hours = pd.concat(parts, ignore_index=True)
    hours = hours.sort_values("ts_unix").drop_duplicates(
        ["station_id", "utc_hour"], keep="last"
    )
    hours = hours.merge(registry, on="station_id", how="inner", validate="many_to_one")
    # Time zone is derived from Unix time, so autumn DST change is handled.
    hours["local_datetime"] = pd.to_datetime(
        hours.utc_hour * 3600, unit="s", utc=True
    ).dt.tz_convert("Europe/Berlin")
    hours["local_hour"] = hours.local_datetime.dt.hour.astype(int)
    hours["is_weekend"] = (hours.local_datetime.dt.dayofweek >= 5).astype(int)
    hours["local_date"] = hours.local_datetime.dt.date
    hours["occupancy_fraction"] = hours.occupied / hours.total
    # A definite full-occupancy label is available only if every point is
    # accounted for as occupied or available (no unknown/outage/reserved states).
    hours["full_occupancy"] = np.where(
        (hours.occupied + hours.available).eq(hours.total),
        hours.occupied.eq(hours.total).astype(float), np.nan,
    )
    return hours, raw_read, filtered


def estimate_profiles(training: pd.DataFrame, prior_strength: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on past only: station mean baseline and shrunk weekday/hour profile."""
    # Build equal-station-weighted regional equipment benchmark; otherwise
    # frequently reporting stations would overwhelm the hour profile.
    station_bucket = training.groupby(
        ["station_id", "power_band", "is_weekend", "local_hour"], as_index=False
    ).agg(
        occupancy_mean=("occupancy_fraction", "mean"),
        full_mean=("full_occupancy", "mean"),
    )
    global_bucket = station_bucket.groupby(
        ["power_band", "is_weekend", "local_hour"], as_index=False
    ).agg(
        peer_occupancy=("occupancy_mean", "mean"),
        peer_full=("full_mean", "mean"),
        peer_station_count=("station_id", "nunique"),
    )
    global_mean = training.groupby("power_band", as_index=False).agg(
        band_occupancy=("occupancy_fraction", "mean"),
        band_full=("full_occupancy", "mean"),
    )
    global_bucket = global_bucket.merge(global_mean, on="power_band", validate="many_to_one")
    # Stabilize an equipment/hour benchmark where very few peers are observed.
    n = global_bucket.peer_station_count.astype(float)
    global_bucket["peer_occupancy"] = (
        (n * global_bucket.peer_occupancy + 8 * global_bucket.band_occupancy) / (n + 8)
    )
    # Missing full-occupancy labels stay UNKNOWN; never replace with zero.
    # Only smooth using group observations where a full-status label exists.
    peer_full_known = global_bucket.peer_full.where(global_bucket.peer_full.notna(), global_bucket.band_full)
    global_bucket["peer_full"] = (
        (n * peer_full_known + 8 * global_bucket.band_full) / (n + 8)
    )
    station_means = training.groupby("station_id", as_index=False).agg(
        station_baseline=("occupancy_fraction", "mean"),
        station_full_baseline=("full_occupancy", "mean"),
        train_hourly_records=("occupancy_fraction", "size"),
        train_unique_days=("local_date", "nunique"),
    )
    station_hour = training.groupby(
        ["station_id", "power_band", "is_weekend", "local_hour"], as_index=False
    ).agg(
        site_mean=("occupancy_fraction", "mean"),
        site_full=("full_occupancy", "mean"),
        site_n=("occupancy_fraction", "size"),
        site_full_n=("full_occupancy", "count"),
    )
    station_hour = station_hour.merge(
        global_bucket[["power_band", "is_weekend", "local_hour", "peer_occupancy", "peer_full"]],
        on=["power_band", "is_weekend", "local_hour"], validate="many_to_one",
    )
    station_hour["predicted_occupancy"] = (
        station_hour.site_n * station_hour.site_mean
        + prior_strength * station_hour.peer_occupancy
    ) / (station_hour.site_n + prior_strength)
    station_hour["predicted_full"] = (
        station_hour.site_full_n * station_hour.site_full.fillna(0)
        + prior_strength * station_hour.peer_full
    ) / (station_hour.site_full_n + prior_strength)
    return station_means, (station_hour[["station_id", "power_band", "is_weekend", "local_hour",
                                        "predicted_occupancy", "predicted_full", "site_n"]]
                            .merge(global_bucket[["power_band", "is_weekend", "local_hour",
                                                  "peer_occupancy", "peer_full"]],
                                   on=["power_band", "is_weekend", "local_hour"], validate="many_to_one"))


def run_backtest(hours: pd.DataFrame, days: int, min_train_days: int,
                 min_test_days: int, min_train_hours: int, min_test_hours: int,
                 prior_strength: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest_hour = int(hours.utc_hour.max())
    cutoff = latest_hour - days * 24
    training = hours.loc[hours.utc_hour.lt(cutoff)].copy()
    testing = hours.loc[hours.utc_hour.ge(cutoff)].copy()
    if training.empty or testing.empty:
        raise ValueError("No chronological train/test split possible")
    # Do NOT use data from the holdout period for global means or site profiles.
    training_means, profiles = estimate_profiles(training, prior_strength)
    test = testing.merge(training_means, on="station_id", how="inner", validate="many_to_one")
    test = test.loc[
        test.train_hourly_records.ge(min_train_hours)
        & test.train_unique_days.ge(min_train_days)
    ].copy()
    test_days = test.groupby("station_id")["local_date"].nunique()
    test_hours = test.groupby("station_id")["utc_hour"].size()
    qualified = set(test_days.index[(test_days >= min_test_days) & (test_hours >= min_test_hours)])
    test = test.loc[test.station_id.isin(qualified)].copy()
    if test.empty:
        raise ValueError(
            "No stations have sufficient training and unseen test coverage. "
            "Try --min-train-days 7 --min-test-days 2 and inspect data coverage."
        )
    # Fallback for unseen (weekend/hour) bucket: station mean only. Never use
    # actual holdout occupancy to fill or adapt a forecast.
    test = test.merge(
        profiles[["station_id", "is_weekend", "local_hour", "predicted_occupancy", "predicted_full"]],
        on=["station_id", "is_weekend", "local_hour"], how="left", validate="many_to_one"
    )
    test["predicted_occupancy"] = test.predicted_occupancy.fillna(test.station_baseline)
    test["predicted_full"] = test.predicted_full.fillna(test.station_full_baseline)
    # For a station with no known full/available status, full-site metrics are
    # missing rather than falsely counting missing as a "not full" observation.
    test["predicted_full"] = test.predicted_full.clip(0, 1)
    test["predicted_occupancy"] = test.predicted_occupancy.clip(0, 1)
    test["baseline_abs_error_pp"] = 100 * abs(test.occupancy_fraction - test.station_baseline)
    test["profile_abs_error_pp"] = 100 * abs(test.occupancy_fraction - test.predicted_occupancy)
    test["baseline_full_brier"] = (test.full_occupancy - test.station_full_baseline) ** 2
    test["profile_full_brier"] = (test.full_occupancy - test.predicted_full) ** 2
    test["cutoff_utc"] = pd.to_datetime(cutoff * 3600, unit="s", utc=True).isoformat()
    station = test.groupby(["station_id", "power_band"], as_index=False).agg(
        train_hours=("train_hourly_records", "first"),
        train_days=("train_unique_days", "first"),
        test_hours=("utc_hour", "size"),
        test_days=("local_date", "nunique"),
        actual_occupancy_pct=("occupancy_fraction", lambda x: round(float(x.mean() * 100), 2)),
        stationary_baseline_mae_pp=("baseline_abs_error_pp", "mean"),
        hourly_profile_mae_pp=("profile_abs_error_pp", "mean"),
        fully_classified_test_hours=("full_occupancy", "count"),
        stationary_full_brier=("baseline_full_brier", "mean"),
        hourly_profile_full_brier=("profile_full_brier", "mean"),
    )
    station["mae_improvement_pp"] = station.stationary_baseline_mae_pp - station.hourly_profile_mae_pp
    station["hourly_profile_beats_baseline"] = station.mae_improvement_pp.gt(0)
    station["data_status"] = DATA_STATUS
    scores = []
    for power_band, frame in [("ALL_DC", station), *list(station.groupby("power_band", sort=True))]:
        scores.append({
            "power_band": power_band,
            "installations_tested": len(frame),
            "test_hourly_snapshots": int(frame.test_hours.sum()),
            "median_station_baseline_mae_pp": round(float(frame.stationary_baseline_mae_pp.median()), 3),
            "median_station_profile_mae_pp": round(float(frame.hourly_profile_mae_pp.median()), 3),
            "macro_average_baseline_mae_pp": round(float(frame.stationary_baseline_mae_pp.mean()), 3),
            "macro_average_profile_mae_pp": round(float(frame.hourly_profile_mae_pp.mean()), 3),
            "stations_profile_better_pct": round(100 * float(frame.hourly_profile_beats_baseline.mean()), 1),
            "macro_station_full_baseline_brier": round(float(frame.stationary_full_brier.mean()), 4),
            "macro_station_full_profile_brier": round(float(frame.hourly_profile_full_brier.mean()), 4),
            "validation_status": "PROFILE_IMPROVES_ON_HOLDOUT" if (
                frame.hourly_profile_mae_pp.mean() < frame.stationary_baseline_mae_pp.mean()
            ) else "PROFILE_NOT_BETTER_THAN_SIMPLE_BASELINE",
            "data_status": DATA_STATUS,
        })
    output_cols = ["station_id", "power_band", "local_datetime", "utc_hour", "local_hour", "is_weekend",
                   "total", "available", "occupied", "occupancy_fraction", "full_occupancy",
                   "station_baseline", "predicted_occupancy", "station_full_baseline", "predicted_full",
                   "baseline_abs_error_pp", "profile_abs_error_pp", "cutoff_utc"]
    result = test[output_cols].copy()
    result["local_datetime"] = result.local_datetime.astype(str)
    return result, station, pd.DataFrame(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real 2026 German DC occupancy out-of-time backtest")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--min-train-days", type=int, default=14)
    parser.add_argument("--min-test-days", type=int, default=3)
    parser.add_argument("--min-train-hours", type=int, default=80)
    parser.add_argument("--min-test-hours", type=int, default=20)
    parser.add_argument("--prior-strength", type=float, default=12.0)
    parser.add_argument("--chunksize", type=int, default=200000)
    args = parser.parse_args()
    if (args.test_days < 2 or args.min_train_days < 1 or args.min_test_days < 1 or
        args.min_train_hours < 1 or args.min_test_hours < 1 or
        args.prior_strength <= 0 or args.chunksize < 100):
        parser.error("Invalid validation settings")
    root = args.root.resolve()
    raw = choose_latest(root / "data" / "external", "chargemyride*.csv")
    registry = load_dc_registry(root / "data" / "processed" / "german_dc_extended_station_quality.csv")
    hours, n_raw, n_valid = load_real_hourly(raw, registry, args.chunksize)
    predictions, site_scores, summary = run_backtest(
        hours, args.test_days, args.min_train_days, args.min_test_days,
        args.min_train_hours, args.min_test_hours, args.prior_strength
    )
    output = root / "data" / "processed"
    output.mkdir(exist_ok=True, parents=True)
    paths = {
        "prediction": output / "real_dc_2026_holdout_predictions.csv",
        "station": output / "real_dc_2026_station_backtest.csv",
        "summary": output / "real_dc_2026_backtest_summary.csv",
    }
    predictions.to_csv(paths["prediction"], index=False)
    site_scores.to_csv(paths["station"], index=False)
    summary.to_csv(paths["summary"], index=False)
    print("\n--- REAL GERMAN DC FORECAST VALIDATION (UNSEEN HOLDOUT) ---")
    print(f"Raw status records scanned: {n_raw:,} | Valid DC snapshot rows: {n_valid:,}")
    print(f"Qualified registered DC installations: {len(registry):,}")
    print(f"Held-out installations evaluated: {len(site_scores):,}")
    print(f"Test period: last {args.test_days} days of available real data")
    print(f"Training/test boundary (UTC): {predictions.cutoff_utc.iloc[0]}")
    print(summary[["power_band", "installations_tested", "test_hourly_snapshots",
                   "macro_average_baseline_mae_pp", "macro_average_profile_mae_pp",
                   "stations_profile_better_pct", "validation_status"]].to_string(index=False))
    print("\nSaved:")
    for path in paths.values():
        print(path)
    print("\nLIMIT: Real OCCUPANCY snapshots only. No measured car arrivals, kWh, queues, or rejected drivers.")
    print("A model that fails to beat the baseline should NOT be used to justify expansion forecasts.")


if __name__ == "__main__":
    main()
