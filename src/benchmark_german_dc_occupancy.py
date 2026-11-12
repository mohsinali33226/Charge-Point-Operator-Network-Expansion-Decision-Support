"""Descriptive comparison of real German DC occupancy and synthetic CPO sites.

Run from the project's root folder:
    python src\\benchmark_german_dc_occupancy.py

Reads the output of find_german_dc_validation_stations.py and the existing
synthetic site_hourly_demand.csv. Creates benchmarking CSVs without modifying
source files or claiming validation of rejected cars/queues/charging energy.

IMPORTANT:
- A BNetzA Ladeeinrichtungs-ID can describe a *single installation*, NOT a
  complete charging park with multiple registered installations.
- ChargeMyRide observations are irregular STATUS SNAPSHOTS, not time-weighted
  utilization, arrived cars, customer waiting or charging-session power.
- Real observations (2026) and synthetic histories (2025) are different sites
  and periods. The numbers below are DESCRIPTIVE sanity checks only.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
REAL_STATIONS = OUT / "german_dc_validation_candidates.csv"
REAL_HOURLY = OUT / "german_dc_hourly_occupancy.csv"
SYNTHETIC = OUT / "site_hourly_demand.csv"


def read_required(path: Path, required: list[str], *, id_dtype: dict | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}\nRun the appropriate previous importer first.")
    data = pd.read_csv(path, dtype=id_dtype, low_memory=False)
    data.columns = data.columns.str.strip()
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"{path.name} has missing columns: {missing}\nExisting columns: {data.columns.tolist()}")
    return data


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def classify_power(x: float) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    if x >= 300:
        return "300_PLUS"
    if x >= 150:
        return "150_TO_299"
    if x >= 50:
        return "50_TO_149"
    return "UNDER_50"


def point_group(x: float) -> str:
    if pd.isna(x) or x <= 0:
        return "UNKNOWN"
    if x <= 1:
        return "1"
    if x <= 2:
        return "2"
    if x <= 4:
        return "3_TO_4"
    if x <= 8:
        return "5_TO_8"
    return "9_PLUS"


def prepare_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    real = read_required(REAL_STATIONS, [
        "station_id", "registered_equipment_power_kw", "registered_charging_points",
        "valid_snapshots", "observed_days", "mean_occupied_snapshot_pct",
        "no_available_snapshot_pct"
    ], id_dtype={"station_id": "string"})
    real["station_id"] = real.station_id.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    if real.station_id.isna().any() or real.station_id.duplicated().any():
        raise ValueError("Selected real station IDs must be unique and nonmissing")
    for c in ["registered_equipment_power_kw", "registered_charging_points", "valid_snapshots",
              "observed_days", "mean_occupied_snapshot_pct", "no_available_snapshot_pct"]:
        real[c] = num(real[c])
    real = real.loc[
        real.registered_charging_points.gt(0)
        & real.registered_equipment_power_kw.ge(50)
        & real.mean_occupied_snapshot_pct.between(0, 100)
        & real.no_available_snapshot_pct.between(0, 100)
        & real.valid_snapshots.gt(0)
    ].copy()
    if real.empty:
        raise ValueError("No usable real DC station rows after column checks")
    real["power_band"] = real.registered_equipment_power_kw.map(classify_power)
    real["point_group"] = real.registered_charging_points.map(point_group)

    hourly = read_required(REAL_HOURLY, [
        "station_id", "local_hour", "valid_snapshots", "mean_occupied_pct",
        "no_available_snapshot_pct"
    ], id_dtype={"station_id": "string"})
    hourly["station_id"] = hourly.station_id.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    for c in ["local_hour", "valid_snapshots", "mean_occupied_pct", "no_available_snapshot_pct"]:
        hourly[c] = num(hourly[c])
    hourly = hourly.loc[
        hourly.local_hour.between(0, 23)
        & hourly.valid_snapshots.gt(0)
        & hourly.mean_occupied_pct.between(0, 100)
        & hourly.no_available_snapshot_pct.between(0, 100)
    ].copy()
    if hourly.empty:
        raise ValueError("No valid real station-hour observations")
    hourly = hourly.merge(real[["station_id", "power_band", "point_group"]],
                          on="station_id", how="inner", validate="many_to_one")
    if hourly.empty:
        raise ValueError("Selected station IDs are not present in the real hourly profile")
    return real, hourly


def build_benchmarks(real: pd.DataFrame, hourly: pd.DataFrame) -> None:
    # Medians are ACROSS STATIONS, so stations with more frequent status reports
    # do not automatically dominate the comparison.
    benchmark = real.groupby(["power_band", "point_group"], observed=True).agg(
        real_installations=("station_id", "nunique"),
        median_snapshot_occupancy_pct=("mean_occupied_snapshot_pct", "median"),
        p25_snapshot_occupancy_pct=("mean_occupied_snapshot_pct", lambda x: x.quantile(.25)),
        p75_snapshot_occupancy_pct=("mean_occupied_snapshot_pct", lambda x: x.quantile(.75)),
        median_no_free_snapshot_pct=("no_available_snapshot_pct", "median"),
        median_observed_days=("observed_days", "median"),
        median_snapshot_count=("valid_snapshots", "median"),
    ).reset_index().sort_values(["power_band", "point_group"])
    for c in benchmark.columns:
        if c not in ("power_band", "point_group", "real_installations"):
            benchmark[c] = benchmark[c].round(2)
    benchmark.to_csv(OUT / "german_dc_real_occupancy_benchmarks.csv", index=False)

    # Each real installation has equal weight when producing time-of-day curves.
    hourly = (hourly.groupby(["station_id", "power_band", "point_group", "local_hour"], observed=True)
              .agg(mean_occupied_pct=("mean_occupied_pct", "mean"),
                   no_available_snapshot_pct=("no_available_snapshot_pct", "mean"))
              .reset_index())
    grouped = hourly.groupby(["power_band", "point_group", "local_hour"], observed=True).agg(
        real_installations=("station_id", "nunique"),
        mean_station_hour_occupied_pct=("mean_occupied_pct", "mean"),
        mean_station_hour_no_free_pct=("no_available_snapshot_pct", "mean"),
    ).reset_index().sort_values(["power_band", "point_group", "local_hour"])
    grouped[["mean_station_hour_occupied_pct", "mean_station_hour_no_free_pct"]] = (
        grouped[["mean_station_hour_occupied_pct", "mean_station_hour_no_free_pct"]].round(2)
    )
    grouped.to_csv(OUT / "german_dc_real_hourly_benchmarks.csv", index=False)
    print("\n--- REAL DC OCCUPANCY BENCHMARKS ---")
    print(f"Real installations: {len(real)} | Hourly station records: {len(hourly):,}")
    print("By power band:")
    print(real.groupby("power_band").agg(
        installations=("station_id", "size"),
        median_occupied_pct=("mean_occupied_snapshot_pct", "median"),
        median_no_free_pct=("no_available_snapshot_pct", "median")
    ).round(1).to_string())

    # Compare only like-rated installed equipment and exact charging-position count.
    # If none or too few are comparable, flag it clearly instead of extrapolating.
    if not SYNTHETIC.exists():
        print(f"\nSynthetic hourly data not found: {SYNTHETIC.name}")
        print("Real benchmarks saved; no synthetic comparison was attempted.")
        return
    sim = read_required(SYNTHETIC, [
        "site_id", "city", "connector_count", "charger_power_kw",
        "average_connected_vehicles"
    ])
    for c in ["connector_count", "charger_power_kw", "average_connected_vehicles"]:
        sim[c] = num(sim[c])
    sim = sim.loc[sim.connector_count.gt(0) & sim.charger_power_kw.gt(0)
                  & sim.average_connected_vehicles.ge(0)].copy()
    if sim.empty:
        print("Synthetic hourly table has no usable rows; only real benchmarks saved.")
        return
    sim["mean_occupied_pct"] = (100 * sim.average_connected_vehicles / sim.connector_count).clip(0, 100)
    sim_summary = sim.groupby(["site_id", "city"], observed=True).agg(
        site_connector_count=("connector_count", "first"),
        assumed_charger_power_kw=("charger_power_kw", "first"),
        simulated_hours=("mean_occupied_pct", "size"),
        simulated_mean_hourly_occupancy_pct=("mean_occupied_pct", "mean"),
    ).reset_index()
    sim_summary["power_band"] = sim_summary.assumed_charger_power_kw.map(classify_power)
    matching = []
    for _, site in sim_summary.iterrows():
        # A BNetzA installation may be smaller than an entire multi-unit site.
        # Exact point-count matches are screenable examples, NOT proof of site equivalence.
        m = real.loc[
            real.power_band.eq(site.power_band)
            & real.registered_charging_points.eq(site.site_connector_count)
        ]
        n = len(m)
        matching.append({
            "site_id": site.site_id,
            "city": site.city,
            "simulated_connector_count": int(site.site_connector_count),
            "simulated_power_band": site.power_band,
            "synthetic_hours": int(site.simulated_hours),
            "simulated_mean_hourly_occupancy_pct": round(float(site.simulated_mean_hourly_occupancy_pct), 2),
            "real_comparable_installations_exact_points_and_power_band": n,
            "real_median_snapshot_occupancy_pct": round(float(m.mean_occupied_snapshot_pct.median()), 2) if n else np.nan,
            "comparison_status": "DESCRIPTIVE_ONLY_DIFFERENT_SITES_AND_TIME_PERIODS" if n >= 5 else (
                "LOW_REAL_SAMPLE_DESCRIPTIVE_ONLY" if n else "NO_DIRECT_EQUIPMENT_COUNT_MATCH"
            ),
        })
    comparison = pd.DataFrame(matching)
    comparison.to_csv(OUT / "cpo_real_vs_synthetic_occupancy_screen.csv", index=False)
    print("\n--- SYNTHETIC SITE COMPARABILITY SCREEN ---")
    print(comparison[["city", "simulated_connector_count", "simulated_power_band",
                      "simulated_mean_hourly_occupancy_pct",
                      "real_comparable_installations_exact_points_and_power_band",
                      "comparison_status"]].to_string(index=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real, hourly = prepare_real()
    build_benchmarks(real, hourly)
    print("\nSaved in data/processed:")
    for name in ("german_dc_real_occupancy_benchmarks.csv",
                 "german_dc_real_hourly_benchmarks.csv",
                 "cpo_real_vs_synthetic_occupancy_screen.csv"):
        file = OUT / name
        if file.exists():
            print(file)
    print("\nIMPORTANT: Real = irregular 2026 snapshots, synthetic = hourly 2025 averages.")
    print("This is a REALISM SCREEN, not validation of rejected cars, queues, energy or ROI.")
    print("Station ID may mean ONE installation, not an entire charging park.")


if __name__ == "__main__":
    main()
