"""Build traceable historical CCS session-behaviour inputs for CPO model correction.

Usage (from CPO Network Development project root):
    python src\build_mer_ccs_calibration.py

Read-only inputs:
    data/processed/mer_valid_sessions.csv
    data/raw/charging_sessions.csv

Creates new outputs (never overwrites existing simulator/demand files):
    data/processed/mer_ccs_empirical_behaviour_pool.csv
    data/processed/mer_ccs_vs_synthetic_audit.csv
    data/processed/mer_ccs_location_profiles.csv
    data/config/mer_ccs_behaviour_config.json

IMPORTANT:
- Mer reports DELIVERED energy and observed session duration, not a driver's
  requested energy, vehicle rated input power, or site arrivals incl. refusals.
- This is historical, selected operator data from 2020-2024, not a German-wide
  2026 calibration or direct evidence of an individual CPO site's demand.
- We retain energy + duration + observed average power TOGETHER. Future vehicle
  simulation must not independently sample them, or relabel average power as a
  measured vehicle maximum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL = ROOT / "data" / "processed" / "mer_valid_sessions.csv"
DEFAULT_SYNTH = ROOT / "data" / "raw" / "charging_sessions.csv"
DEFAULT_OUT = ROOT / "data" / "processed"
DEFAULT_CFG = ROOT / "data" / "config"
REQUIRED_REAL = {
    "source_row_number", "source_charger_type", "source_location_category",
    "start_time", "duration_min", "energy_kwh", "mean_charging_power_kw",
    "usable_for_behaviour_calibration",
}
REQUIRED_SYNTH = {"duration_minutes", "energy_kwh", "average_power_kw"}


def quantiles(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"p10": None, "p50": None, "p90": None, "mean": None}
    return {
        "p10": round(float(x.quantile(.10)), 3),
        "p50": round(float(x.quantile(.50)), 3),
        "p90": round(float(x.quantile(.90)), 3),
        "mean": round(float(x.mean()), 3),
    }


def numeric_filter(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    valid = pd.Series(True, index=df.index)
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        valid &= df[col].notna() & np.isfinite(df[col]) & (df[col] > 0)
    return valid


def brief_stats(label: str, data: pd.DataFrame, energy: str, duration: str, power: str) -> dict:
    return {
        "population": label,
        "sessions": int(len(data)),
        "energy_kwh_p10": quantiles(data[energy])["p10"],
        "energy_kwh_p50": quantiles(data[energy])["p50"],
        "energy_kwh_p90": quantiles(data[energy])["p90"],
        "duration_min_p10": quantiles(data[duration])["p10"],
        "duration_min_p50": quantiles(data[duration])["p50"],
        "duration_min_p90": quantiles(data[duration])["p90"],
        "average_power_kw_p50": quantiles(data[power])["p50"],
        "average_power_kw_p90": quantiles(data[power])["p90"],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare real CCS session behaviour for model calibration")
    parser.add_argument("--mer", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CFG)
    args = parser.parse_args()

    for filename in (args.mer, args.synthetic):
        if not filename.is_file():
            parser.error(f"Missing input file: {filename}")

    mer = pd.read_csv(args.mer, low_memory=False)
    synthetic = pd.read_csv(args.synthetic, low_memory=False)
    mer.columns = mer.columns.str.strip()
    synthetic.columns = synthetic.columns.str.strip()
    if REQUIRED_REAL - set(mer.columns):
        raise ValueError(f"Mer input missing: {sorted(REQUIRED_REAL-set(mer.columns))}")
    if REQUIRED_SYNTH - set(synthetic.columns):
        raise ValueError(f"Synthetic input missing: {sorted(REQUIRED_SYNTH-set(synthetic.columns))}")

    # The cleaned Mer file can contain CCS, Type 2 and CHAdeMO sessions.
    # Only CCS is suitable for our initial DC-HPC behaviour reference.
    ccs = mer[mer["source_charger_type"].astype("string").str.contains(
        r"\bCCS\b", case=False, na=False, regex=True
    )].copy()
    ccs["usable_for_behaviour_calibration"] = (
        ccs["usable_for_behaviour_calibration"].astype("string").str.strip().str.lower().eq("true")
    )
    ccs = ccs[ccs["usable_for_behaviour_calibration"]].copy()
    real_valid = numeric_filter(ccs, ["duration_min", "energy_kwh", "mean_charging_power_kw"])
    ccs = ccs[real_valid].copy()
    ccs["start_time"] = pd.to_datetime(ccs["start_time"], errors="coerce")
    ccs = ccs[ccs["start_time"].notna()].copy()
    if ccs.empty:
        raise ValueError("No valid Combo CCS sessions found. Inspect the imported Mer file.")

    ccs["source_location_category"] = (
        ccs["source_location_category"].fillna("UNSPECIFIED").astype(str)
        .str.strip().replace("", "UNSPECIFIED")
    )
    ccs["start_hour"] = ccs["start_time"].dt.hour
    ccs["start_weekday"] = ccs["start_time"].dt.dayofweek
    ccs["start_year"] = ccs["start_time"].dt.year
    ccs = ccs.sort_values(["start_time", "source_row_number"]).reset_index(drop=True)

    # Preserve paired observed measurements. The pool is NOT a file of
    # independently observed car energy requests or charger/vehicle maxima.
    pool = ccs.rename(columns={
        "energy_kwh": "observed_delivered_energy_kwh",
        "mean_charging_power_kw": "observed_average_power_kw",
        "duration_min": "observed_session_duration_min",
    })[
        ["source_row_number", "source_charger_type", "source_location_category",
         "start_time", "start_hour", "start_weekday", "start_year",
         "observed_session_duration_min", "observed_delivered_energy_kwh",
         "observed_average_power_kw"]
    ].copy()

    synthetic_valid = numeric_filter(synthetic, ["duration_minutes", "energy_kwh", "average_power_kw"])
    syn = synthetic[synthetic_valid].copy()
    if syn.empty:
        raise ValueError("No usable synthetic charging sessions found")

    audit = pd.DataFrame([
        brief_stats("Historical Mer Germany - Combo CCS served sessions", ccs,
                    "energy_kwh", "duration_min", "mean_charging_power_kw"),
        brief_stats("Existing 2025 synthetic served sessions", syn,
                    "energy_kwh", "duration_minutes", "average_power_kw"),
    ])
    segment_rows = []
    for category, subset in ccs.groupby("source_location_category", dropna=False):
        row = brief_stats(str(category), subset, "energy_kwh", "duration_min", "mean_charging_power_kw")
        row["location_category"] = str(category)
        segment_rows.append(row)
    segments = pd.DataFrame(segment_rows).sort_values("sessions", ascending=False)

    config = {
        "calibration_status": "HISTORICAL_BEHAVIOUR_PROXY_NOT_2026_SITE_VALIDATION",
        "source": "Mer_Dataset.csv -> mer_valid_sessions.csv",
        "source_data_kind": "observed_successful_charging_sessions",
        "charger_types": "Combo CCS only",
        "observed_start_date": str(ccs["start_time"].min()),
        "observed_end_date": str(ccs["start_time"].max()),
        "sample_size": len(pool),
        "behaviour_pool_csv": "data/processed/mer_ccs_empirical_behaviour_pool.csv",
        "observed_delivered_energy_kwh": quantiles(ccs["energy_kwh"]),
        "observed_session_duration_min": quantiles(ccs["duration_min"]),
        "observed_average_power_kw": quantiles(ccs["mean_charging_power_kw"]),
        "sampling_rule": "sample complete rows to preserve energy-duration-power relationship",
        "limitations": [
            "Observed delivered energy is not a measured vehicle energy request.",
            "Observed average power is not a measured vehicle maximum charging power.",
            "This dataset contains served sessions, not all arrivals or rejected drivers.",
            "Older operator/session sample is not representative of 2026 Germany by default.",
            "Source filters may exclude many sessions; report selection bias.",
            "Historical source location category is not a verified match to demo site IDs.",
            "No existing synthetic data, simulation parameters or P50/P95 models have been overwritten.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.out_dir / "mer_ccs_empirical_behaviour_pool.csv"
    audit_path = args.out_dir / "mer_ccs_vs_synthetic_audit.csv"
    segment_path = args.out_dir / "mer_ccs_location_profiles.csv"
    cfg_path = args.config_dir / "mer_ccs_behaviour_config.json"
    pool.to_csv(pool_path, index=False)
    audit.to_csv(audit_path, index=False)
    segments.to_csv(segment_path, index=False)
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n--- MER CCS VS SYNTHETIC BEHAVIOUR AUDIT ---")
    print(f"Mer Combo CCS sessions used: {len(ccs):,} (from {len(mer):,} imported usable sessions)")
    print(f"Synthetic sessions compared: {len(syn):,}")
    print(audit[["population", "sessions", "energy_kwh_p50", "duration_min_p50", "average_power_kw_p50"]].round(2).to_string(index=False))
    print("\n--- MER CCS LOCATION CATEGORIES ---")
    print(segments[["location_category", "sessions", "energy_kwh_p50", "duration_min_p50"]].head(10).round(2).to_string(index=False))
    print("\nSaved:")
    for path in (pool_path, audit_path, segment_path, cfg_path):
        print(path)
    print("\nIMPORTANT: Delivered energy != requested energy; observed average kW != EV maximum kW.")
    print("Existing simulator and synthetic files are unchanged. Next: controlled integration.")


if __name__ == "__main__":
    main()
