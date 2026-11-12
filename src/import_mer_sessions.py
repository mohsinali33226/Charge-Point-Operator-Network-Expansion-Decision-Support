"""Prepare historical Mer Germany DC charging sessions for model calibration.

Usage from the CPO Network Development project directory:
    python src/import_mer_sessions.py

This does NOT update existing synthetic arrivals or model parameters. All
quality thresholds are transparent command-line options, not market standards.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "external" / "Mer_Dataset.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
IMPORTER_VERSION = "2026-10-31-verified"


REQUIRED = ["Start Date", "End Date", "Total kwh", "Charger Type"]


def standardize_headers(data: pd.DataFrame) -> pd.DataFrame:
    """Accept source column capitalization such as 'Total kWh' or 'Total kwh'.

    Keep the downstream fields and output file structure unchanged.
    Reject duplicate names rather than silently selecting the wrong column.
    """
    canonical = {
        "start date": "Start Date",
        "end date": "End Date",
        "total kwh": "Total kwh",
        "start kwh": "Start kwh",
        "end kwh": "End kwh",
        "charger type": "Charger Type",
        "original location": "Original Location",
        "target_location": "target_location",
        "stop reason": "Stop Reason",
    }
    rename = {}
    used = set()
    for old_name in data.columns:
        normalized = re.sub(r"\s+", " ", str(old_name).strip()).casefold()
        output_name = canonical.get(normalized, str(old_name).strip())
        if output_name in used:
            raise ValueError(
                f"Duplicate column after normalizing headers: {output_name!r}. "
                "Check the source file for repeated headings."
            )
        used.add(output_name)
        rename[old_name] = output_name
    return data.rename(columns=rename)


def numeric_column(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    values = data[name].astype("string").str.strip()
    values = values.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    # German decimal comma, optionally with thousands periods.
    values = values.str.replace(r"\.(?=\d{3}(?:,|$))", "", regex=True)
    values = values.str.replace(",", ".", regex=False)
    return pd.to_numeric(values, errors="coerce")


def safe_text(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data.columns:
        return pd.Series(pd.NA, index=data.index, dtype="string")
    values = data[name].astype("string").str.strip()
    return values.mask(values == "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and import historical Mer charging sessions")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-duration-min", type=float, default=1.0,
                        help="Exploratory QA threshold; not an industry requirement")
    parser.add_argument("--max-duration-h", type=float, default=24.0,
                        help="Exploratory QA threshold; not an industry requirement")
    parser.add_argument("--min-energy-kwh", type=float, default=0.1,
                        help="Exploratory QA threshold; not an industry requirement")
    parser.add_argument("--max-average-power-kw", type=float, default=500.0,
                        help="Exploratory QA flag, not a charger nameplate assumption")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input CSV not found: {args.input}")
    if args.min_duration_min <= 0 or args.max_duration_h <= 0 or args.min_energy_kwh < 0 or args.max_average_power_kw <= 0:
        parser.error("Quality thresholds must be positive (energy threshold may be zero)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- MER GERMANY: REAL CHARGING SESSION AUDIT ---")
    print(f"Importer version: {IMPORTER_VERSION}")
    print(f"Source: {args.input.name} ({args.input.stat().st_size / 1048576:.1f} MB)")

    # 30 MB is small enough to read once; preserve all inputs as text for
    # correct handling of German decimals and unmodified source fields.
    try:
        raw = pd.read_csv(
            args.input, sep=";", encoding="utf-8-sig", dtype="string",
            low_memory=False, on_bad_lines="error",
        )
    except UnicodeDecodeError:
        raw = pd.read_csv(
            args.input, sep=";", encoding="cp1252", dtype="string",
            low_memory=False, on_bad_lines="error",
        )
    raw = standardize_headers(raw)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing source columns {missing}. Found: {raw.columns.tolist()}")

    sessions = pd.DataFrame(index=raw.index)
    sessions["source_row_number"] = raw.index + 2  # includes the header line
    sessions["source_charger_type"] = safe_text(raw, "Charger Type")
    sessions["source_location_label"] = safe_text(raw, "Original Location")
    sessions["source_location_category"] = safe_text(raw, "target_location")
    sessions["source_stop_reason"] = safe_text(raw, "Stop Reason")
    sessions["start_time"] = pd.to_datetime(raw["Start Date"], errors="coerce", format="mixed")
    sessions["end_time"] = pd.to_datetime(raw["End Date"], errors="coerce", format="mixed")
    sessions["duration_min"] = (sessions["end_time"] - sessions["start_time"]).dt.total_seconds() / 60.0
    sessions["energy_kwh"] = numeric_column(raw, "Total kwh")
    sessions["meter_start_kwh"] = numeric_column(raw, "Start kwh")
    sessions["meter_end_kwh"] = numeric_column(raw, "End kwh")
    sessions["meter_delta_kwh"] = sessions["meter_end_kwh"] - sessions["meter_start_kwh"]
    sessions["mean_charging_power_kw"] = np.where(
        sessions["duration_min"] > 0,
        60 * sessions["energy_kwh"] / sessions["duration_min"], np.nan,
    )

    flags = {
        "missing_timestamp": sessions["start_time"].isna() | sessions["end_time"].isna(),
        "invalid_duration": sessions["duration_min"].isna() | (sessions["duration_min"] <= 0),
        "very_short_session": sessions["duration_min"] < args.min_duration_min,
        "very_long_session": sessions["duration_min"] > args.max_duration_h * 60,
        "missing_or_nonpositive_energy": sessions["energy_kwh"].isna() | (sessions["energy_kwh"] <= 0),
        "tiny_energy": (sessions["energy_kwh"] > 0) & (sessions["energy_kwh"] < args.min_energy_kwh),
        "high_implied_power_review": sessions["mean_charging_power_kw"] > args.max_average_power_kw,
    }

    # The meter fields may be incomplete or use different meter conventions.
    # Report discrepancies, but do not automatically exclude rows on that basis.
    has_meters = sessions[["meter_start_kwh", "meter_end_kwh", "energy_kwh"]].notna().all(axis=1)
    mismatch = has_meters & (
        (sessions["meter_delta_kwh"] - sessions["energy_kwh"]).abs()
        > np.maximum(0.2, 0.02 * sessions["energy_kwh"].abs())
    )
    flags["meter_disagreement_review"] = mismatch
    for name, flag in flags.items():
        sessions[name] = flag.fillna(False).astype(bool)

    exclude_flags = [c for c in flags if c != "meter_disagreement_review"]
    sessions["usable_for_behaviour_calibration"] = ~sessions[exclude_flags].any(axis=1)
    sessions["quality_flag_names"] = [
        ";".join(k for k in flags if bool(sessions.at[i, k]))
        for i in sessions.index
    ]
    # Preserve full audit, not just usable records.
    valid = sessions.loc[sessions["usable_for_behaviour_calibration"]].copy()
    valid["start_hour"] = valid["start_time"].dt.hour
    valid["start_weekday"] = valid["start_time"].dt.dayofweek
    valid["start_month"] = valid["start_time"].dt.to_period("M").astype(str)

    audit_path = args.output_dir / "mer_session_quality_audit.csv"
    valid_path = args.output_dir / "mer_valid_sessions.csv"
    profile_path = args.output_dir / "mer_charging_behaviour_summary.csv"
    hourly_path = args.output_dir / "mer_session_start_hour_profile.csv"
    report_path = args.output_dir / "mer_import_quality_report.json"
    sessions.to_csv(audit_path, index=False)
    valid.to_csv(valid_path, index=False)

    if not valid.empty:
        by_type = (
            valid.groupby("source_charger_type", dropna=False)
            .agg(sessions=("energy_kwh", "size"),
                 median_energy_kwh=("energy_kwh", "median"),
                 median_duration_min=("duration_min", "median"),
                 median_session_average_power_kw=("mean_charging_power_kw", "median"),
                 first_session=("start_time", "min"),
                 last_session=("start_time", "max"))
            .reset_index()
        )
        hours = (
            valid.groupby(["source_charger_type", "start_weekday", "start_hour"], dropna=False)
            .size().rename("recorded_session_starts").reset_index()
        )
        hours["share_within_charger_type_pct"] = (
            100 * hours["recorded_session_starts"] /
            hours.groupby("source_charger_type", dropna=False)["recorded_session_starts"].transform("sum")
        ).round(3)
    else:
        by_type = pd.DataFrame(columns=["source_charger_type", "sessions", "median_energy_kwh", "median_duration_min", "median_session_average_power_kw", "first_session", "last_session"])
        hours = pd.DataFrame(columns=["source_charger_type", "start_weekday", "start_hour", "recorded_session_starts", "share_within_charger_type_pct"])
    by_type.to_csv(profile_path, index=False)
    hours.to_csv(hourly_path, index=False)

    date_min = valid["start_time"].min() if not valid.empty else pd.NaT
    date_max = valid["start_time"].max() if not valid.empty else pd.NaT
    summary = {
        "source": args.input.name,
        "source_type": "historical_real_sessions_not_current_2026_demand",
        "total_rows": int(len(sessions)),
        "usable_for_behaviour_calibration": int(len(valid)),
        "excluded_rows": int(len(sessions) - len(valid)),
        "flag_counts_not_mutually_exclusive": {k: int(v.sum()) for k, v in flags.items()},
        "first_usable_start": str(date_min) if pd.notna(date_min) else None,
        "last_usable_start": str(date_max) if pd.notna(date_max) else None,
        "unique_original_location_labels": int(valid["source_location_label"].nunique()),
        "configuration": {
            "minimum_duration_minutes": args.min_duration_min,
            "maximum_duration_hours": args.max_duration_h,
            "minimum_energy_kwh": args.min_energy_kwh,
            "maximum_implied_average_power_kw_review": args.max_average_power_kw,
        },
        "limitations": [
            "Location labels may not be unique station IDs; no site-level demand forecasts are derived.",
            "No guaranteed coverage of all arrivals or rejected drivers.",
            "Session start times should not be equated to total site arrivals without coverage checks.",
            "Historical session distributions do not establish Germany's current 2026 charging behaviour.",
            "No existing synthetic or simulation input files have been overwritten.",
        ],
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Records loaded: {summary['total_rows']:,}")
    print(f"Usable sessions: {summary['usable_for_behaviour_calibration']:,}")
    print(f"Flagged/excluded: {summary['excluded_rows']:,}")
    print(f"Usable date range: {summary['first_usable_start']} to {summary['last_usable_start']}")
    print("\nData-quality flags (may overlap):")
    for name, count in summary["flag_counts_not_mutually_exclusive"].items():
        print(f"  {name}: {count:,}")
    print("\n--- CHARGING BEHAVIOUR BY EQUIPMENT TYPE ---")
    if by_type.empty:
        print("No usable sessions; inspect quality report before proceeding.")
    else:
        print(by_type[["source_charger_type", "sessions", "median_energy_kwh", "median_duration_min", "median_session_average_power_kw"]].round(2).to_string(index=False))
    print("\nSaved:")
    for path in [valid_path, audit_path, profile_path, hourly_path, report_path]:
        print(f"  {path}")
    print("\nNOTE: These are historical real sessions. No existing forecasts or simulator assumptions were changed.")


if __name__ == "__main__":
    main()
