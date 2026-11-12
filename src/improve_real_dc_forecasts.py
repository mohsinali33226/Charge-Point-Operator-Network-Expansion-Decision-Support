"""Chronological real German DC occupancy model selection and honest holdout test.

Run from CPO Network Development project root:
    python src\\improve_real_dc_forecasts.py

Inputs (existing, unchanged):
    data/external/chargemyride*.csv
    data/processed/german_dc_extended_station_quality.csv
    src/validate_real_dc_forecasts.py  (reuses the tested import/quality logic)

Outputs (new files in data/processed):
    real_dc_2026_improved_model_selection.csv
    real_dc_2026_improved_holdout_predictions.csv
    real_dc_2026_improved_station_scores.csv
    real_dc_2026_improved_summary.csv

No future labels are used to select a model. The model is selected with an
intermediate validation window, refitted, then checked on a FINAL unseen window.
Targets are occupancy snapshots, never car arrivals, rejected EVs or energy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from validate_real_dc_forecasts import choose_latest, load_dc_registry, load_real_hourly

MODEL_NAMES = (
    "STATION_MEAN",
    "RECENT_STATION_MEAN",
    "HOUR_S8",
    "HOUR_S24",
    "WEEKEND_HOUR_S12",
    "WEEKEND_HOUR_S32",
    "RECENT_WEEKEND_HOUR_S16",
)


def eligibility(df: pd.DataFrame, min_days: int, min_hours: int) -> set[str]:
    if df.empty:
        return set()
    status = df.groupby("station_id").agg(
        observed_hours=("utc_hour", "nunique"),
        observed_days=("local_date", "nunique"),
    )
    return set(status.index[(status.observed_hours >= min_hours) & (status.observed_days >= min_days)])


def make_features(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Each prediction uses ONLY the earlier training snapshots."""
    result = target[["station_id", "power_band", "local_hour", "is_weekend", "utc_hour"]].copy()
    site_stats = train.groupby("station_id").agg(site_mean=("occupancy_fraction", "mean")).reset_index()
    result = result.merge(site_stats, on="station_id", how="left", validate="many_to_one")
    if result.site_mean.isna().any():
        raise ValueError("Scoring unseen station without sufficient historical observations")

    # Recent station mean can adapt to changes in site usage, using past observations only.
    recent_cutoff = int(train.utc_hour.max()) - 7 * 24
    recent = train.loc[train.utc_hour >= recent_cutoff].groupby("station_id").agg(
        recent_mean=("occupancy_fraction", "mean"),
        recent_hours=("utc_hour", "nunique"),
        recent_days=("local_date", "nunique"),
    ).reset_index()
    result = result.merge(recent, on="station_id", how="left", validate="many_to_one")
    enough_recent = result.recent_hours.ge(18) & result.recent_days.ge(4)
    result["recent_base"] = np.where(
        enough_recent, 0.6 * result.recent_mean.fillna(result.site_mean) + 0.4 * result.site_mean,
        result.site_mean,
    )
    result["STATION_MEAN"] = result.site_mean
    result["RECENT_STATION_MEAN"] = result.recent_base

    # Peer patterns get EQUAL station weighting, not more weight for high-frequency reporters.
    site_band = train[["station_id", "power_band"]].drop_duplicates()
    band_level = site_stats.merge(site_band, on="station_id", validate="one_to_one")
    band_average = band_level.groupby("power_band").site_mean.mean().rename("band_mean")

    for group_keys, configs in (
        (["local_hour"], [("HOUR_S8", 8.0), ("HOUR_S24", 24.0)]),
        (["is_weekend", "local_hour"], [
            ("WEEKEND_HOUR_S12", 12.0), ("WEEKEND_HOUR_S32", 32.0),
            ("RECENT_WEEKEND_HOUR_S16", 16.0),
        ]),
    ):
        station_bucket = train.groupby(
            ["station_id", "power_band"] + group_keys, as_index=False
        ).agg(bucket_mean=("occupancy_fraction", "mean"), bucket_n=("occupancy_fraction", "size"))
        peer = station_bucket.groupby(["power_band"] + group_keys, as_index=False).agg(
            peer_mean=("bucket_mean", "mean")
        )
        peer["band_mean"] = peer.power_band.map(band_average)
        peer["peer_delta"] = (peer.peer_mean - peer.band_mean).fillna(0)
        station_fields = ["station_id"] + group_keys
        group_frame = result[station_fields + ["power_band"]].merge(
            station_bucket[station_fields + ["bucket_mean", "bucket_n"]],
            on=station_fields, how="left", validate="many_to_one",
        ).merge(
            peer[["power_band"] + group_keys + ["peer_delta"]],
            on=["power_band"] + group_keys, how="left", validate="many_to_one",
        )
        n = group_frame.bucket_n.fillna(0).to_numpy(dtype=float)
        mean = group_frame.bucket_mean.fillna(0).to_numpy(dtype=float)
        delta = group_frame.peer_delta.fillna(0).to_numpy(dtype=float)
        for name, strength in configs:
            site_prior = (result.recent_base if name.startswith("RECENT_")
                          else result.site_mean).to_numpy(dtype=float)
            # A peer hour-pattern adjusts the individual station baseline. It
            # cannot replace site-specific history with a global occupancy level.
            prior = np.clip(site_prior + 0.65 * delta, 0, 1)
            result[name] = np.clip((n * mean + strength * prior) / (n + strength), 0, 1)
    return result[["station_id", "utc_hour", *MODEL_NAMES]]


def macro_scores(target: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    scored = target[["station_id", "utc_hour", "occupancy_fraction"]].merge(
        forecast, on=["station_id", "utc_hour"], how="inner", validate="one_to_one"
    )
    if len(scored) != len(target):
        raise ValueError("Prediction keys do not match the evaluation observations")
    results: dict[str, float] = {}
    for name in MODEL_NAMES:
        mae = (100 * (scored[name] - scored.occupancy_fraction).abs()).groupby(scored.station_id).mean()
        results[name] = float(mae.mean())
    return results


def run(train_val_test: pd.DataFrame, val_days: int, test_days: int,
        min_train_days: int, min_train_hours: int,
        min_val_days: int, min_val_hours: int,
        min_test_days: int, min_test_hours: int, min_gain_pp: float):
    last_hour = int(train_val_test.utc_hour.max())
    test_start = last_hour - test_days * 24
    validation_start = test_start - val_days * 24
    initial_train = train_val_test.loc[train_val_test.utc_hour < validation_start].copy()
    validation = train_val_test.loc[train_val_test.utc_hour.between(validation_start, test_start - 1)].copy()
    final_train = train_val_test.loc[train_val_test.utc_hour < test_start].copy()
    holdout = train_val_test.loc[train_val_test.utc_hour >= test_start].copy()

    val_qual = (eligibility(initial_train, min_train_days, min_train_hours)
                & eligibility(validation, min_val_days, min_val_hours))
    validation = validation[validation.station_id.isin(val_qual)].copy()
    if validation.empty:
        raise ValueError("Insufficient VALIDATION station history. Inspect source coverage before relaxing minimums")

    validation_forecasts = make_features(initial_train, validation)
    selection: list[dict] = []
    band_choice: dict[str, str] = {}
    for band, observations in validation.groupby("power_band", sort=True):
        forecasts = validation_forecasts.loc[
            validation_forecasts.station_id.isin(observations.station_id.unique())
        ]
        losses = macro_scores(observations, forecasts)
        best = min(losses, key=losses.get)
        gain = losses["STATION_MEAN"] - losses[best]
        chosen = best if gain >= min_gain_pp else "STATION_MEAN"
        band_choice[band] = chosen
        selection.append({
            "power_band": band, "validation_stations": observations.station_id.nunique(),
            "validation_snapshots": len(observations),
            "validation_baseline_mae_pp": round(losses["STATION_MEAN"], 3),
            "best_validation_candidate": best,
            "best_validation_mae_pp": round(losses[best], 3),
            "selected_model": chosen,
            "selected_validation_mae_pp": round(losses[chosen], 3),
            "selection_rule": f"MIN_GAIN_{min_gain_pp:g}_PERCENTAGE_POINTS_ON_VALIDATION",
        })
    choices = pd.DataFrame(selection)

    test_qual = (eligibility(final_train, min_train_days, min_train_hours)
                 & eligibility(holdout, min_test_days, min_test_hours))
    test = holdout[holdout.station_id.isin(test_qual)].copy()
    test = test[test.power_band.isin(band_choice)].copy()
    if test.empty:
        raise ValueError("Insufficient TEST history after quality checks; no accuracy claim possible")
    test_forecasts = make_features(final_train, test)
    test = test.merge(test_forecasts, on=["station_id", "utc_hour"], validate="one_to_one")
    test["selected_model"] = test.power_band.map(band_choice)
    test["predicted_occupancy"] = np.select(
        [test.selected_model.eq(name) for name in MODEL_NAMES],
        [test[name].to_numpy(dtype=float) for name in MODEL_NAMES],
        default=np.nan,
    )
    if test.predicted_occupancy.isna().any():
        raise ValueError("Selected prediction missing; do not silently substitute an observed value")
    test["baseline_error_pp"] = (100 * (test.occupancy_fraction - test.STATION_MEAN).abs())
    test["model_error_pp"] = (100 * (test.occupancy_fraction - test.predicted_occupancy).abs())
    station = test.groupby(["station_id", "power_band", "selected_model"], as_index=False).agg(
        test_snapshots=("utc_hour", "size"),
        test_days=("local_date", "nunique"),
        actual_occupancy_pct=("occupancy_fraction", lambda x: 100 * float(x.mean())),
        baseline_mae_pp=("baseline_error_pp", "mean"),
        model_mae_pp=("model_error_pp", "mean"),
    )
    station["gain_pp"] = station.baseline_mae_pp - station.model_mae_pp
    summary_rows = []
    for band, rows in [("ALL_DC", station), *list(station.groupby("power_band", sort=True))]:
        baseline = float(rows.baseline_mae_pp.mean())
        model = float(rows.model_mae_pp.mean())
        gain = baseline - model
        summary_rows.append({
            "power_band": band,
            "stations_tested": len(rows),
            "test_snapshots": int(rows.test_snapshots.sum()),
            "baseline_mae_pp": round(baseline, 3),
            "selected_model_mae_pp": round(model, 3),
            "mae_gain_pp": round(gain, 3),
            "stations_model_better_pct": round(100 * rows.gain_pp.gt(0).mean(), 1),
            "status": ("MEASURABLE_HOLDOUT_IMPROVEMENT" if gain >= min_gain_pp
                       else "NOT_YET_MATERIAL_OR_NOT_BETTER"),
        })
    summary = pd.DataFrame(summary_rows)
    forecast_export = test[["station_id", "power_band", "utc_hour", "local_datetime",
                            "local_hour", "is_weekend", "occupied", "available", "total",
                            "occupancy_fraction", "STATION_MEAN", "selected_model",
                            "predicted_occupancy", "baseline_error_pp", "model_error_pp"]].copy()
    forecast_export["local_datetime"] = forecast_export.local_datetime.astype(str)
    return choices, forecast_export, station, summary, validation_start, test_start, last_hour


def main() -> None:
    parser = argparse.ArgumentParser(description="Select occupancy models on one real period; test on a later unseen period")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--min-train-days", type=int, default=10)
    parser.add_argument("--min-train-hours", type=int, default=50)
    parser.add_argument("--min-validation-days", type=int, default=2)
    parser.add_argument("--min-validation-hours", type=int, default=12)
    parser.add_argument("--min-test-days", type=int, default=3)
    parser.add_argument("--min-test-hours", type=int, default=20)
    parser.add_argument("--min-improvement-pp", type=float, default=0.25)
    parser.add_argument("--chunksize", type=int, default=200000)
    args = parser.parse_args()
    if min(args.validation_days, args.test_days, args.min_train_days, args.min_train_hours,
           args.min_validation_days, args.min_validation_hours, args.min_test_days,
           args.min_test_hours, args.chunksize) <= 0 or args.min_improvement_pp < 0:
        parser.error("Positive duration/coverage settings required, nonnegative minimum improvement")
    root = args.root.resolve()
    registry = load_dc_registry(root / "data/processed/german_dc_extended_station_quality.csv")
    raw_path = choose_latest(root / "data/external", "chargemyride*.csv")
    observations, n_raw, n_valid = load_real_hourly(raw_path, registry, args.chunksize)
    choices, predictions, station, summary, val_start, test_start, last_hour = run(
        observations, args.validation_days, args.test_days,
        args.min_train_days, args.min_train_hours,
        args.min_validation_days, args.min_validation_hours,
        args.min_test_days, args.min_test_hours,
        args.min_improvement_pp,
    )
    output = root / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "selection": output / "real_dc_2026_improved_model_selection.csv",
        "predictions": output / "real_dc_2026_improved_holdout_predictions.csv",
        "station": output / "real_dc_2026_improved_station_scores.csv",
        "summary": output / "real_dc_2026_improved_summary.csv",
    }
    for label, frame in (("selection", choices), ("predictions", predictions),
                         ("station", station), ("summary", summary)):
        frame.to_csv(paths[label], index=False)
    utc = lambda h: pd.to_datetime(int(h) * 3600, unit="s", utc=True).isoformat()
    print("\n--- REAL DC OCCUPANCY FORECAST: TRAIN / VALIDATION / UNSEEN TEST ---")
    print(f"Raw status rows: {n_raw:,} | Valid selected snapshots: {n_valid:,} | Registry DC sites: {len(registry):,}")
    print(f"Model selection starts: {utc(val_start)} | Final unseen test starts: {utc(test_start)}")
    print(f"Latest observation: {utc(last_hour)} (actual data coverage, not download date)")
    print("\n--- MODELS CHOSEN USING VALIDATION ONLY ---")
    print(choices[["power_band", "validation_stations", "selected_model", "selected_validation_mae_pp"]].to_string(index=False))
    print("\n--- UNSEEN REAL DATA RESULTS ---")
    print(summary[["power_band", "stations_tested", "baseline_mae_pp", "selected_model_mae_pp",
                   "mae_gain_pp", "status"]].to_string(index=False))
    print("\nSaved:")
    for p in paths.values():
        print(p)
    print("\nNOTE: Occupancy snapshots only; never interpret as verified arrivals, queues, energy or CPO ROI.")
    print("A negative or small improvement means the forecasts still require better data/features.")


if __name__ == "__main__":
    main()
