"""Audit Mer Germany CCS event outcomes without inventing car arrivals.

Reads the original Mer CSV and distinguishes observed charging sessions from
failed authorization, data-quality anomalies, and unknown outcomes.

Run from the project root:
    python src/audit_mer_event_outcomes.py

The screening thresholds below flag records for review; they are not industry
standards or specifications for individual EVs or charging equipment.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

DEFAULT_SOURCE = Path("data/external/Mer_Dataset.csv")
DEFAULT_OUTPUT = Path("data/processed")
MAX_ENERGY_KWH = 250.0
MAX_AVERAGE_POWER_KW = 500.0
MIN_DURATION_MIN = 2.0
MAX_DURATION_MIN = 240.0


def decimal_number(series: pd.Series) -> pd.Series:
    """Parse German decimal commas; never guess whether integer data mean Wh."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def require_columns(frame: pd.DataFrame, names: list[str]) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Mer source is missing columns: {missing}. Found: {list(frame.columns)}")


def create_audit(source: Path, output_dir: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Cannot find {source}. Check data/external/Mer_Dataset.csv")

    frame = pd.read_csv(source, sep=";", dtype="string", encoding="utf-8-sig", low_memory=False)
    frame.columns = frame.columns.str.strip()
    require_columns(
        frame,
        ["Charger Type", "Original Location", "Stop Reason", "Start Date", "End Date", "Total kWh"],
    )

    source_rows = len(frame)
    ccs = frame["Charger Type"].str.contains("CCS", case=False, na=False)
    frame = frame.loc[ccs].copy()
    frame["source_row_number"] = np.flatnonzero(ccs.to_numpy()) + 2

    frame["start_time"] = pd.to_datetime(frame["Start Date"], errors="coerce")
    frame["end_time"] = pd.to_datetime(frame["End Date"], errors="coerce")
    frame["duration_min"] = (frame["end_time"] - frame["start_time"]).dt.total_seconds() / 60.0
    frame["reported_energy_kwh"] = decimal_number(frame["Total kWh"])

    if "Start kWh" in frame.columns and "End kwh" in frame.columns:
        start_meter = decimal_number(frame["Start kWh"])
        end_meter = decimal_number(frame["End kwh"])
        frame["meter_change_kwh"] = end_meter - start_meter
    else:
        frame["meter_change_kwh"] = np.nan

    reason = frame["Stop Reason"].fillna("").astype(str)
    frame["authorization_failure_flag"] = reason.str.contains(
        r"Authorization Failure|Card not found|Invalid card|Access denied|No matching plan|not fully setup for billing",
        case=False,
        regex=True,
        na=False,
    )
    frame["technical_or_remote_stop_flag"] = reason.str.contains(
        r"Hard reset|Server stopped|Remote stop|Fault|Error",
        case=False,
        regex=True,
        na=False,
    )

    energy = frame["reported_energy_kwh"]
    dur = frame["duration_min"]
    frame["energy_out_of_review_range"] = energy.notna() & ((energy < 0) | (energy > MAX_ENERGY_KWH))
    frame["duration_out_of_review_range"] = dur.isna() | (dur < MIN_DURATION_MIN) | (dur > MAX_DURATION_MIN)
    frame["implied_mean_power_kw"] = energy * 60 / dur.where(dur > 0)
    frame["high_implied_power_review"] = frame["implied_mean_power_kw"] > MAX_AVERAGE_POWER_KW

    meter = frame["meter_change_kwh"]
    comparable_meter = meter.notna() & energy.notna() & meter.ge(0) & energy.ge(0)
    frame["meter_mismatch_review"] = comparable_meter & (
        (meter - energy).abs() > np.maximum(2.0, energy * 0.10)
    )

    energy_and_time_ok = (
        energy.gt(0) & energy.le(MAX_ENERGY_KWH)
        & dur.ge(MIN_DURATION_MIN) & dur.le(MAX_DURATION_MIN)
        & ~frame["high_implied_power_review"].fillna(False)
    ).fillna(False)
    auth = frame["authorization_failure_flag"].fillna(False)
    meter_bad = frame["meter_mismatch_review"].fillna(False)

    conditions = [
        energy_and_time_ok & auth,
        energy_and_time_ok & meter_bad,
        energy_and_time_ok & ~auth & ~meter_bad,
        auth,
        frame["energy_out_of_review_range"].fillna(False) | frame["high_implied_power_review"].fillna(False),
        frame["duration_out_of_review_range"].fillna(False),
        frame["technical_or_remote_stop_flag"].fillna(False),
    ]
    labels = [
        "POSITIVE_ENERGY_WITH_AUTH_FLAG_REVIEW",
        "POSITIVE_ENERGY_METER_DISAGREEMENT_REVIEW",
        "OBSERVED_CHARGING_SESSION",
        "AUTHORIZATION_FAILURE_NO_VERIFIED_CHARGE",
        "ENERGY_OR_POWER_VALUE_REVIEW",
        "DURATION_OR_TIMESTAMP_REVIEW",
        "TECHNICAL_OR_REMOTE_STOP_NO_VERIFIED_CHARGE",
    ]
    frame["event_class"] = np.select(conditions, labels, default="UNKNOWN_OR_NO_VERIFIED_CHARGE")

    frame["source_location_label"] = frame["Original Location"].fillna("UNKNOWN").astype(str).str.strip()
    frame["charging_start_observed"] = frame["event_class"].eq("OBSERVED_CHARGING_SESSION")

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "mer_ccs_event_outcomes.csv"
    location_path = output_dir / "mer_ccs_outcomes_by_location.csv"
    class_path = output_dir / "mer_ccs_event_class_summary.csv"

    columns = [
        "source_row_number", "source_location_label", "start_time", "end_time", "duration_min",
        "reported_energy_kwh", "meter_change_kwh", "implied_mean_power_kw", "event_class",
        "authorization_failure_flag", "technical_or_remote_stop_flag", "energy_out_of_review_range",
        "duration_out_of_review_range", "high_implied_power_review", "meter_mismatch_review",
        "charging_start_observed", "Stop Reason",
    ]
    frame[columns].to_csv(events_path, index=False)

    by_class = frame.groupby("event_class", dropna=False).size().reset_index(name="records")
    by_class = by_class.sort_values("records", ascending=False)
    by_class.to_csv(class_path, index=False)

    locations = frame.groupby("source_location_label", dropna=False).agg(
        total_ccs_logs=("event_class", "size"),
        observed_charging_sessions=("charging_start_observed", "sum"),
        auth_flagged_logs=("authorization_failure_flag", "sum"),
        energy_range_flagged=("energy_out_of_review_range", "sum"),
        meter_mismatch_flagged=("meter_mismatch_review", "sum"),
        first_log=("start_time", "min"),
        last_log=("start_time", "max"),
    ).reset_index().sort_values("total_ccs_logs", ascending=False)
    locations["other_or_review_logs"] = locations["total_ccs_logs"] - locations["observed_charging_sessions"]
    locations.to_csv(location_path, index=False)

    print("\n--- MER CCS EVENT OUTCOME / DATA QUALITY AUDIT ---")
    print(f"Original source rows: {source_rows:,} | CCS event logs: {len(frame):,}")
    print("Event classes (these are logs, NOT individual vehicle arrivals):")
    print(by_class.to_string(index=False))
    print("\nLocation check (largest 8 by event count):")
    print(locations[["source_location_label", "total_ccs_logs", "observed_charging_sessions",
                     "auth_flagged_logs", "other_or_review_logs"]].head(8).to_string(index=False))

    subset = locations[locations["source_location_label"].str.casefold().eq("ländlicher bereich 2".casefold())]
    if not subset.empty:
        print("\nSPECIAL CHECK: Ländlicher Bereich 2")
        print(subset[["total_ccs_logs", "observed_charging_sessions", "auth_flagged_logs",
                      "energy_range_flagged", "other_or_review_logs"]].to_string(index=False))

    print("\nSaved:")
    print(events_path)
    print(location_path)
    print(class_path)
    print("\nIMPORTANT: An authorization failure is NOT evidence of connector congestion.")
    print("Session start logs do NOT reveal lost arrivals or waiting queues.")
    print("Suspect energy values are flagged, never silently rescaled from kWh to Wh.")
    print("Source locations and site infrastructure remain UNVERIFIED. Existing simulations unchanged.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify Mer CCS operational records by evidence")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    create_audit(args.source, args.output_dir)


if __name__ == "__main__":
    main()
