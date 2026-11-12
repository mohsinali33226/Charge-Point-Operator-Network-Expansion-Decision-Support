"""Backtest reported successful charging starts from the STANDARD CPO session adapter.

Run from the CPO Network Development project root:
    python src/forecast_cpo_verified_sessions.py

Input (produced by prepare_cpo_real_session_adapter.py):
    data/processed/cpo_real_observed_sessions.csv
Optional documented source-hour evidence:
    data/config/cpo_hourly_reporting_coverage.csv
    Columns: source_system,site_id,source_location_label,hour_start,
             reporting_complete,site_operational,evidence_reference
    Use true/false and one row per hour; fill EITHER site_id OR source label.
    Never invent evidence or fill Mer's unknown site mapping.

IMPORTANT LIMITATIONS
- Target = REPORTED SUCCESSFUL CHARGING STARTS; not attempted arrivals, queues,
  rejected vehicles, or total site demand.
- With operator hourly reporting/uptime evidence, unknown hours are EXCLUDED
  (not silently treated as zero).
- Without evidence, zero-filling is allowed only for labelled research-only
  CONDITIONAL comparisons; this is not verified operational forecasting.
- Anonymous Mer source labels are kept separate from verified/operator site IDs.
- No real site is inferred from an anonymous location, and no source data is edited.
- A forecast NEVER becomes an invest/expand decision from this script.

Outputs (new, preserving legacy forecast and expansion files):
    data/processed/cpo_standard_starts_backtest_sites.csv
    data/processed/cpo_standard_starts_unseen_predictions.csv
    data/processed/cpo_standard_starts_data_readiness.csv
    data/processed/cpo_standard_starts_reporting_audit.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026-10-31-standard-session-forecast-v2-coverage-gated"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "cpo_real_observed_sessions.csv"
DEFAULT_OUT = ROOT / "data" / "processed"
DEFAULT_COVERAGE = ROOT / "data" / "config" / "cpo_hourly_reporting_coverage.csv"
MIN_VERIFIED_COVERAGE = 0.80  # Configurable demonstration gate, not an industry standard.
COVERAGE_COLUMNS = ["source_system", "site_id", "source_location_label", "hour_start",
                    "reporting_complete", "site_operational", "evidence_reference"]
TRAIN_DAYS = 56
VALIDATE_DAYS = 14
TEST_DAYS = 14
MIN_TRAIN_STARTS = 80
MIN_VALIDATION_STARTS = 8
MIN_TEST_STARTS = 8
MIN_TRAIN_ACTIVE_DAYS = 20
MIN_VALIDATION_ACTIVE_DAYS = 5
MIN_TEST_ACTIVE_DAYS = 5

REQUIRED = {
    "source_system", "source_event_id", "source_location_label", "site_id",
    "start_time_source", "site_identity_status", "timezone_status",
    "observed_record_type", "data_quality_status",
}
IDENTITIES_WITH_SITE = {
    "MAPPED_WITH_DOCUMENTED_EVIDENCE", "OPERATOR_REPORTED_SITE_ID",
}


def _clean_str(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame[col].fillna("").astype(str).str.strip()


def read_sessions(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. First run: python src\\prepare_cpo_real_session_adapter.py")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    df.columns = df.columns.str.strip()
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing standardized session columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Standardized session input is empty")
    for col in REQUIRED:
        df[col] = _clean_str(df, col)
    valid = (df["observed_record_type"].eq("SUCCESSFUL_CHARGING_START")
             & df["data_quality_status"].eq("VALID_OBSERVED_SESSION"))
    ignored = int((~valid).sum())
    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError("No valid successful charging starts in standardized input")
    if (df["source_event_id"] == "").any() or (df["source_system"] == "").any():
        raise ValueError("Missing source event ID or source system; cannot safely count sessions")
    # Do not sum duplicated session logs as two customer visits.
    duplicate = df.duplicated(["source_system", "source_event_id"], keep=False)
    if duplicate.any():
        example = df.loc[duplicate, ["source_system", "source_event_id"]].head(3).to_dict("records")
        raise ValueError(f"Duplicate (source system, event ID) detected: {example}. Fix source before forecasting")
    # No silent timezone adjustment: input must be source-local, naive wall clock.
    tz_aware = df["start_time_source"].str.contains(r"(?:Z$|[+-]\d{2}:?\d{2}$)", regex=True)
    if tz_aware.any():
        raise ValueError("Timezone-aware start_time_source encountered. Normalize with documented site timezone first")
    df["start_time"] = pd.to_datetime(df["start_time_source"], errors="coerce")
    if df["start_time"].isna().any():
        raise ValueError("Invalid start timestamp in accepted sessions. Review source timestamps")
    if df["start_time"].dt.tz is not None:
        raise ValueError("Timezone-aware datetime requires explicit site timezone conversion")
    missing_id = df["site_id"].eq("")
    mapped = df["site_identity_status"].isin(IDENTITIES_WITH_SITE) & ~missing_id
    # Detect contradictory identity claims, rather than silently falling back.
    bad_id = (~missing_id) & ~df["site_identity_status"].isin(IDENTITIES_WITH_SITE)
    if bad_id.any():
        raise ValueError("site_id populated without recognized mapping evidence/status. Review site_identity_status")
    wrong_mapped = missing_id & df["site_identity_status"].isin(IDENTITIES_WITH_SITE)
    if wrong_mapped.any():
        raise ValueError("Mapped/operator site identity status but site_id is missing")
    if (df.loc[~mapped, "source_location_label"] == "").any():
        raise ValueError("Anonymous records require non-empty source location label")
    df["forecast_unit"] = np.where(
        mapped, "SITE::" + df["source_system"] + "::" + df["site_id"],
        "LABEL::" + df["source_system"] + "::" + df["source_location_label"]
    )
    df["identity_level"] = np.where(mapped, df["site_identity_status"], "UNVERIFIED_SOURCE_LABEL")
    return df, {"rows_read": int(len(valid)), "excluded_rows": ignored, "accepted_events": len(df)}


def read_reporting_coverage(path: Path) -> tuple[pd.DataFrame, str]:
    """Read operator-supplied hour-by-hour proof; never fabricate coverage."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=COVERAGE_COLUMNS).to_csv(path, index=False)
        return pd.DataFrame(columns=COVERAGE_COLUMNS + ["forecast_unit", "usable_hour"]), "EMPTY_TEMPLATE_CREATED"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = frame.columns.str.strip()
    missing = set(COVERAGE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Reporting evidence missing columns: {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS + ["forecast_unit", "usable_hour"]), "EMPTY_TEMPLATE"
    for name in COVERAGE_COLUMNS:
        frame[name] = _clean_str(frame, name)
    has_site = frame.site_id.ne("")
    has_label = frame.source_location_label.ne("")
    if (has_site == has_label).any():
        raise ValueError("Each reporting-evidence row must identify EITHER site_id OR source_location_label")
    if frame.source_system.eq("").any() or frame.evidence_reference.eq("").any():
        raise ValueError("Missing source_system or documented evidence_reference in coverage file")
    frame["forecast_unit"] = np.where(
        has_site, "SITE::" + frame.source_system + "::" + frame.site_id,
        "LABEL::" + frame.source_system + "::" + frame.source_location_label)
    if frame.hour_start.str.contains(r"(?:Z$|[+-]\d{2}:?\d{2}$)", regex=True).any():
        raise ValueError("Coverage timestamps must use documented site-local wall time, without an offset")
    frame["hour_start"] = pd.to_datetime(frame.hour_start, errors="coerce", format="mixed")
    if frame.hour_start.isna().any() or (frame.hour_start.dt.floor("h") != frame.hour_start).any():
        raise ValueError("Coverage hour_start must be a valid timestamp at HH:00:00")
    if frame.duplicated(["forecast_unit", "hour_start"]).any():
        raise ValueError("Duplicate reporting-evidence hours for the same site")
    for name in ("reporting_complete", "site_operational"):
        value = frame[name].str.upper()
        if not value.isin({"TRUE", "FALSE", "YES", "NO", "1", "0"}).all():
            raise ValueError(f"Coverage {name} must be explicit true/false, not blank or guessed")
        frame[name] = value.isin({"TRUE", "YES", "1"})
    frame["usable_hour"] = frame.reporting_complete & frame.site_operational
    return frame, "OPERATOR_COVERAGE_ROWS_PRESENT"



def build_hourly_panel(events: pd.DataFrame, evidence: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    last_day = events["start_time"].max().normalize()
    if not evidence.empty:
        last_day = max(last_day, evidence["hour_start"].max().normalize())
    earliest = last_day - pd.Timedelta(days=TRAIN_DAYS + VALIDATE_DAYS + TEST_DAYS - 1)
    hours = pd.date_range(earliest, last_day + pd.Timedelta(hours=23), freq="h")
    counts = events.groupby(events["start_time"].dt.floor("h")).size()
    panel = pd.DataFrame({"hour_start": hours})
    panel["raw_recorded_starts"] = panel["hour_start"].map(counts).fillna(0).astype(int)
    has_coverage = not evidence.empty
    if has_coverage:
        panel = panel.merge(evidence[["hour_start", "reporting_complete", "site_operational", "usable_hour"]],
                            on="hour_start", how="left", validate="one_to_one")
        usable = panel["usable_hour"].fillna(False).astype(bool)
        conflict = panel["site_operational"].eq(False) & panel["raw_recorded_starts"].gt(0)
        if conflict.any():
            raise ValueError(f"{int(conflict.sum())} hours have charging starts but evidence says site not operational")
        # CRITICAL: unverified hours are MISSING, NOT ZERO.
        panel["recorded_successful_starts"] = panel["raw_recorded_starts"].where(usable, np.nan)
        panel["evidence_based_hour"] = usable
        panel["reporting_hour_status"] = np.select(
            [usable, panel["reporting_complete"].eq(False), panel["site_operational"].eq(False)],
            ["VERIFIED_REPORTING_AND_OPERATION", "REPORTING_INCOMPLETE", "SITE_NOT_OPERATIONAL"],
            default="UNKNOWN_NO_EVIDENCE")
    else:
        # Only retain old zero-fill assumption as visibly UNVERIFIED research diagnostics.
        panel["recorded_successful_starts"] = panel["raw_recorded_starts"].astype(float)
        panel["evidence_based_hour"] = False
        panel["reporting_hour_status"] = np.where(panel["raw_recorded_starts"].gt(0),
            "START_OBSERVED_REPORTING_UNKNOWN", "ZERO_ASSUMED_NO_COVERAGE_PROOF")
    panel["hour_of_day"] = panel["hour_start"].dt.hour
    panel["weekday"] = panel["hour_start"].dt.dayofweek
    panel["weekend"] = (panel["weekday"] >= 5).astype(int)
    val_start = earliest + pd.Timedelta(days=TRAIN_DAYS)
    test_start = val_start + pd.Timedelta(days=VALIDATE_DAYS)
    panel["split"] = np.select(
        [panel["hour_start"] < val_start, panel["hour_start"] < test_start],
        ["TRAIN", "VALIDATION"], default="UNSEEN_TEST")
    return panel, has_coverage


def forecast_methods(train: pd.DataFrame, target: pd.DataFrame) -> dict[str, np.ndarray]:
    y = train["recorded_successful_starts"]
    mu = float(y.mean())
    results = {"MEAN_HOURLY_BASELINE": np.full(len(target), mu)}
    for name, cols, prior in (
        ("HOUR_OF_DAY", ["hour_of_day"], 12.),
        ("HOUR_WEEKEND", ["weekend", "hour_of_day"], 8.),
        ("HOUR_OF_WEEK", ["weekday", "hour_of_day"], 4.),
    ):
        group = train.groupby(cols)["recorded_successful_starts"].agg(["sum", "count"])
        if len(cols) == 1:
            keys = pd.Index(target[cols[0]].to_numpy(), name=cols[0])
        else:
            keys = pd.MultiIndex.from_frame(target[cols])
        m = group.reindex(keys)
        results[name] = (m["sum"].fillna(0).to_numpy(dtype=float) + prior * mu) / (
            m["count"].fillna(0).to_numpy(dtype=float) + prior
        )
    return results


def error(y: pd.Series, pred: np.ndarray) -> float:
    return float(np.abs(y.to_numpy(dtype=float) - pred).mean())


def active_days(p: pd.DataFrame) -> int:
    nonzero = p.loc[p["recorded_successful_starts"].gt(0), "hour_start"]
    return int(nonzero.dt.normalize().nunique())


def score_unit(key: str, events: pd.DataFrame, all_evidence: pd.DataFrame,
               min_verified_coverage: float) -> tuple[dict, list[dict]]:
    ids = sorted({v for v in events["site_id"] if v})
    source_labels = sorted(set(events["source_location_label"]))
    systems = sorted(set(events["source_system"]))
    if len(ids) > 1 or len(systems) != 1:
        raise ValueError(f"Ambiguous identities/source systems in {key}")
    is_site = len(ids) == 1
    tz = sorted(set(events["timezone_status"]))
    time_verified = len(tz) == 1 and tz[0] in {"TIMEZONE_REPORTED", "TIMEZONE_VERIFIED"}
    evidence = all_evidence[all_evidence["forecast_unit"].eq(key)] if not all_evidence.empty else all_evidence
    p, has_coverage = build_hourly_panel(events, evidence)
    base = {
        "forecast_unit": key, "site_id": ids[0] if ids else "",
        "source_system": systems[0], "source_location_labels": ";".join(source_labels),
        "identity_level": ";".join(sorted(set(events["identity_level"]))),
        "timezone_status": ";".join(tz), "recorded_sessions_total": len(events),
        "training_days": TRAIN_DAYS, "validation_days": VALIDATE_DAYS,
        "unseen_test_days": TEST_DAYS,
        "target": "RECORDED_SUCCESSFUL_CHARGING_STARTS_ONLY",
        "reporting_evidence_rows": len(evidence),
        "reporting_evidence_available": has_coverage,
        "unknown_hours_treated_as_zero_assumption": not has_coverage,
        "site_level_forecast_available": is_site,
        "actual_vehicle_arrivals_known": False,
        "queues_and_lost_demand_known": False,
        "investment_ready": False,
        "status": "",
    }
    split_data = {}
    for split, prefix in (("TRAIN", "train"), ("VALIDATION", "validation"), ("UNSEEN_TEST", "unseen_test")):
        part = p[p["split"].eq(split)]
        valid = part[part["recorded_successful_starts"].notna()].copy()
        split_data[split] = valid
        base[f"{prefix}_hours_total"] = len(part)
        base[f"{prefix}_hours_usable"] = len(valid)
        base[f"{prefix}_coverage_pct"] = round(100 * len(valid) / len(part), 2) if has_coverage else np.nan
        base[f"{prefix}_recorded_starts"] = int(valid["recorded_successful_starts"].sum())
        base[f"{prefix}_active_days"] = active_days(valid)
        base[f"{prefix}_unknown_hours"] = len(part) - len(valid) if has_coverage else len(part)
        base[f"{prefix}_assumed_zero_hours"] = int(part["recorded_successful_starts"].eq(0).sum()) if not has_coverage else 0
    # An operator record of complete reporting is evidence for a SPECIFIC hour,
    # not for every hour in the whole time range.
    coverage_pct = [base[f"{pfx}_coverage_pct"] for pfx in ("train", "validation", "unseen_test")]
    base["logging_completeness_verified"] = bool(has_coverage and min(coverage_pct) == 100.0)
    base["observed_starts_in_unverified_hours"] = (
        int(p.loc[~p["evidence_based_hour"], "raw_recorded_starts"].sum()) if has_coverage else len(events))
    test = split_data["UNSEEN_TEST"]
    base["unseen_test_start"] = str(test["hour_start"].min().date()) if not test.empty else ""
    base["unseen_test_end"] = str(test["hour_start"].max().date()) if not test.empty else ""
    if has_coverage and min(coverage_pct) / 100.0 < min_verified_coverage:
        base["status"] = "BLOCKED_INSUFFICIENT_REPORTING_COVERAGE"
        return base, []
    enough = (base["train_recorded_starts"] >= MIN_TRAIN_STARTS
              and base["validation_recorded_starts"] >= MIN_VALIDATION_STARTS
              and base["unseen_test_recorded_starts"] >= MIN_TEST_STARTS
              and base["train_active_days"] >= MIN_TRAIN_ACTIVE_DAYS
              and base["validation_active_days"] >= MIN_VALIDATION_ACTIVE_DAYS
              and base["unseen_test_active_days"] >= MIN_TEST_ACTIVE_DAYS)
    if not enough:
        base["status"] = ("BLOCKED_INSUFFICIENT_VERIFIED_HISTORY" if has_coverage
                          else "INSUFFICIENT_RECORDED_HISTORY")
        return base, []
    train, val = split_data["TRAIN"], split_data["VALIDATION"]
    predicted_val = forecast_methods(train, val)
    scores = {name: error(val["recorded_successful_starts"], vals)
              for name, vals in predicted_val.items()}
    selected = min(scores, key=lambda n: (scores[n], n != "MEAN_HOURLY_BASELINE", n))
    train_val = pd.concat([train, val], ignore_index=True)
    predicted_test = forecast_methods(train_val, test)
    actual = test["recorded_successful_starts"].to_numpy(dtype=float)
    baseline, chosen = predicted_test["MEAN_HOURLY_BASELINE"], predicted_test[selected]
    base_mae = error(test["recorded_successful_starts"], baseline)
    chosen_mae = error(test["recorded_successful_starts"], chosen)
    base.update({
        "model_selected_on_validation": selected,
        "validation_selected_mae_starts_per_hour": round(scores[selected], 5),
        "unseen_baseline_mae_starts_per_hour": round(base_mae, 5),
        "unseen_selected_mae_starts_per_hour": round(chosen_mae, 5),
        "unseen_mae_improvement_pct": round((base_mae - chosen_mae) / base_mae * 100, 2) if base_mae else np.nan,
        "unseen_wape_pct": round(np.abs(actual - chosen).sum() / actual.sum() * 100, 2) if actual.sum() else np.nan,
        "unseen_predicted_starts": round(float(chosen.sum()), 3),
    })
    if not has_coverage:
        base["status"] = ("RESEARCH_ONLY_UNVERIFIED_LOCATION_AND_COVERAGE" if not is_site
                          else "CONDITIONAL_SITE_BACKTEST_REPORTING_UNVERIFIED")
    elif not is_site:
        base["status"] = "RESEARCH_ONLY_UNVERIFIED_PHYSICAL_SITE_ID"
    elif not time_verified:
        base["status"] = "SITE_BACKTEST_TIMEZONE_UNVERIFIED"
    else:
        base["status"] = "SITE_SUCCESSFUL_START_BACKTEST_WITH_REPORTING_EVIDENCE"
    details = test[["hour_start", "recorded_successful_starts", "reporting_hour_status", "evidence_based_hour"]].copy()
    details.insert(0, "forecast_unit", key)
    details.insert(1, "site_id", base["site_id"])
    details["selected_model"] = selected
    details["baseline_predicted_starts"] = np.round(baseline, 5)
    details["selected_predicted_starts"] = np.round(chosen, 5)
    details["forecast_status"] = base["status"]
    details["target"] = base["target"]
    return base, details.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE,
                        help="Optional operator hour-by-hour reporting and uptime evidence")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-verified-coverage", type=float, default=MIN_VERIFIED_COVERAGE,
                        help="Fraction of hours required in each split. Default 0.80, an example gate, not industry standard")
    args = parser.parse_args()
    if not 0 < args.min_verified_coverage <= 1:
        parser.error("--min-verified-coverage must be between 0 and 1 (exclusive of 0)")
    events, stats = read_sessions(args.input)
    evidence, evidence_status = read_reporting_coverage(args.coverage)
    summaries, details = [], []
    for key, group in events.groupby("forecast_unit", sort=True):
        row, preds = score_unit(key, group, evidence, args.min_verified_coverage)
        summaries.append(row)
        details.extend(preds)
    out = pd.DataFrame(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a = args.output_dir / "cpo_standard_starts_backtest_sites.csv"
    b = args.output_dir / "cpo_standard_starts_unseen_predictions.csv"
    c = args.output_dir / "cpo_standard_starts_data_readiness.csv"
    d = args.output_dir / "cpo_standard_starts_reporting_audit.csv"
    out.to_csv(a, index=False)
    pred_cols = ["forecast_unit", "site_id", "hour_start", "recorded_successful_starts",
                 "reporting_hour_status", "evidence_based_hour", "selected_model",
                 "baseline_predicted_starts", "selected_predicted_starts", "forecast_status", "target"]
    pd.DataFrame(details, columns=pred_cols).to_csv(b, index=False)
    ready_cols = ["forecast_unit", "site_id", "source_system", "source_location_labels",
                  "identity_level", "timezone_status", "site_level_forecast_available",
                  "reporting_evidence_available", "logging_completeness_verified",
                  "unknown_hours_treated_as_zero_assumption", "train_coverage_pct",
                  "validation_coverage_pct", "unseen_test_coverage_pct",
                  "observed_starts_in_unverified_hours", "actual_vehicle_arrivals_known",
                  "queues_and_lost_demand_known", "status", "investment_ready"]
    out[ready_cols].to_csv(c, index=False)
    audit_cols = ["forecast_unit", "site_id", "reporting_evidence_rows",
                  "train_hours_total", "train_hours_usable", "train_unknown_hours", "train_assumed_zero_hours",
                  "validation_hours_total", "validation_hours_usable", "validation_unknown_hours", "validation_assumed_zero_hours",
                  "unseen_test_hours_total", "unseen_test_hours_usable", "unseen_test_unknown_hours", "unseen_test_assumed_zero_hours",
                  "observed_starts_in_unverified_hours", "status"]
    out[audit_cols].to_csv(d, index=False)
    scored = (out["model_selected_on_validation"].notna() if "model_selected_on_validation" in out
              else pd.Series(False, index=out.index))
    proven = (scored & out.reporting_evidence_available & out.site_level_forecast_available &
              out.timezone_status.isin(["TIMEZONE_VERIFIED", "TIMEZONE_REPORTED"]))
    print("\n--- STANDARD CPO SESSIONS / REPORTING COVERAGE BACKTEST ---")
    print(f"Script version: {VERSION}")
    print(f"Valid observed charging sessions: {stats['accepted_events']:,}")
    print(f"Forecast groups: {len(out)} | Research/conditional backtests: {int(scored.sum() - proven.sum())} | Site backtests with reporting evidence: {int(proven.sum())}")
    print(f"Reporting evidence: {evidence_status} | Documented hourly rows: {len(evidence):,}")
    shown = out.copy()
    shown["site_or_label"] = np.where(shown.site_id.fillna("").ne(""), shown.site_id, shown.source_location_labels)
    for name in ("model_selected_on_validation", "unseen_selected_mae_starts_per_hour"):
        if name not in shown:
            shown[name] = np.nan
    print(shown[["site_or_label", "model_selected_on_validation", "unseen_selected_mae_starts_per_hour",
                 "train_coverage_pct", "unseen_test_coverage_pct", "status"]].to_string(index=False))
    print("\nIMPORTANT: Without hourly reporting proof, zeros are CONDITIONAL assumptions, not measured zero-demand hours.")
    print("With partial evidence, missing/closed hours are UNKNOWN and EXCLUDED; insufficient coverage blocks scoring.")
    print("This predicts successful charging starts, NOT arrivals, queues, rejected cars, energy, or ROI.")
    print(f"Saved: {a}\nSaved: {b}\nSaved: {c}\nSaved: {d}")
    print(f"Reporting coverage template/input: {args.coverage}")


if __name__ == "__main__":
    main()
