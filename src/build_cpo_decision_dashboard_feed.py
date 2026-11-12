"""Build one evidence-labelled CPO network decision feed for Power BI.

From the project root (Windows CMD):
    python src\build_cpo_decision_dashboard_feed.py

Inputs (already produced by previous project steps):
    data/processed/cpo_network_technical_decisions.csv
    data/processed/cpo_financial_whatif_results.csv
    data/processed/cpo_workflow_action_queue.csv
    data/processed/cpo_standard_starts_data_readiness.csv (optional, created by forecast_cpo_verified_sessions.py)
    data/processed/cpo_standard_starts_backtest_sites.csv (optional, created by forecast_cpo_verified_sessions.py)

Output:
    data/processed/cpo_decision_dashboard_feed.csv

One row per site + planning year + demand assumption. Illustrative financial
numbers NEVER count as an approved investment recommendation. Real charging-start
forecast evidence is joined by an EXACT documented site ID only. Anonymous Mer
labels are never mapped to synthetic sites. Successful starts cannot establish
all arriving drivers, lost demand, congestion, or investment feasibility.
No new growth, prices, capacities, or grid upgrade requirements are assumed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TECH_DEFAULT = DATA / "cpo_network_technical_decisions.csv"
FIN_DEFAULT = DATA / "cpo_financial_whatif_results.csv"
QUEUE_DEFAULT = DATA / "cpo_workflow_action_queue.csv"
OUTPUT_DEFAULT = DATA / "cpo_decision_dashboard_feed.csv"
FORECAST_READINESS_DEFAULT = DATA / "cpo_standard_starts_data_readiness.csv"
FORECAST_BACKTEST_DEFAULT = DATA / "cpo_standard_starts_backtest_sites.csv"
VERSION = "2026-10-31-evidence-gated-decision-feed-v2"

KEY = ["site_id", "planning_year", "assumed_annual_arrival_growth_pct"]
TECH_NEEDED = KEY + [
    "city", "selected_scenario_id", "current_chargers", "shortlisted_chargers",
    "additional_chargers", "rejection_pct_current", "rejection_pct_shortlisted",
    "additional_cars_served_in_test_period", "additional_energy_kwh_in_test_period",
    "max_rejected_pct_screening_target", "technical_next_action",
    "grid_value_screen", "grid_evidence_status", "site_feasibility_gate",
    "demand_evidence_status", "model_status",
]
FIN_NEEDED = KEY + [
    "selected_scenario_id", "financial_status", "input_evidence_class",
    "npv_eur_illustrative", "annual_contribution_eur_illustrative",
    "breakeven_contribution_eur_per_kwh_illustrative",
    "annualized_demo_energy_kwh",
]
QUEUE_NEEDED = KEY + [
    "next_workflow_step",
    "evidence_fields_missing", "evidence_fields_total",
]


def load_csv(path: Path, required: list[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}\n"
            "Run the corresponding earlier project script first."
        )
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    frame.columns = frame.columns.str.strip()
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{label} is empty")
    for field in ["site_id", "planning_year", "assumed_annual_arrival_growth_pct"]:
        if frame[field].isna().any() or frame[field].str.strip().eq("").any():
            raise ValueError(f"Missing {label} key: {field}")
    frame["site_id"] = frame["site_id"].str.strip()
    frame["planning_year"] = pd.to_numeric(frame["planning_year"], errors="raise")
    frame["assumed_annual_arrival_growth_pct"] = pd.to_numeric(
        frame["assumed_annual_arrival_growth_pct"], errors="raise"
    ).round(8)
    if frame[KEY[1:]].isna().any().any():
        raise ValueError(f"Non-numeric year/growth in {label}")
    if (frame["planning_year"] != frame["planning_year"].round(0)).any():
        raise ValueError(f"Non-integer planning year in {label}")
    frame["planning_year"] = frame["planning_year"].astype(int)
    if frame.duplicated(KEY).any():
        raise ValueError(f"Duplicate site/year/growth in {label}")
    return frame


def to_number(frame: pd.DataFrame, fields: list[str], context: str) -> None:
    for field in fields:
        frame[field] = pd.to_numeric(
            frame[field].replace({"": pd.NA}), errors="raise"
        )
        if field in {"current_chargers", "additional_chargers", "evidence_fields_missing", "evidence_fields_total"}:
            if frame[field].isna().any() or (frame[field] < 0).any():
                raise ValueError(f"Missing or negative {field} in {context}")



# These columns are from forecast_cpo_verified_sessions.py coverage-gated v2.
READINESS_NEEDED = [
    "forecast_unit", "site_id", "source_system", "identity_level", "timezone_status",
    "site_level_forecast_available", "reporting_evidence_available",
    "logging_completeness_verified", "unknown_hours_treated_as_zero_assumption",
    "train_coverage_pct", "validation_coverage_pct", "unseen_test_coverage_pct",
    "actual_vehicle_arrivals_known", "queues_and_lost_demand_known", "status",
]
BACKTEST_NEEDED = [
    "forecast_unit", "site_id", "source_system", "status", "target",
    "model_selected_on_validation", "unseen_baseline_mae_starts_per_hour",
    "unseen_selected_mae_starts_per_hour", "unseen_mae_improvement_pct",
]
TRUE_VALUES = {"true", "yes", "1"}
FALSE_VALUES = {"false", "no", "0"}
SUPPORTED_BACKTEST_STATUS = "SITE_SUCCESSFUL_START_BACKTEST_WITH_REPORTING_EVIDENCE"
VALID_IDENTITY = {"OPERATOR_REPORTED_SITE_ID", "MAPPED_WITH_DOCUMENTED_EVIDENCE"}
VALID_TIMEZONE = {"TIMEZONE_VERIFIED", "TIMEZONE_REPORTED"}


def _text(value: object) -> str:
    """One safe representation, including pandas missing values."""
    return "" if pd.isna(value) else str(value).strip()


def _truth(value: object) -> bool | None:
    cleaned = _text(value).lower()
    if cleaned in TRUE_VALUES:
        return True
    if cleaned in FALSE_VALUES:
        return False
    return None


def _numeric_or_nan(value: object, label: str, *, percent: bool = False) -> float:
    cleaned = _text(value)
    if not cleaned:
        return float("nan")
    try:
        number = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"{label}: invalid number {cleaned!r}") from exc
    if not np.isfinite(number) or (percent and not 0 <= number <= 100):
        raise ValueError(f"{label}: number outside accepted range: {cleaned!r}")
    return number


def _read_forecasts(readiness_path: Path, backtest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Missing BOTH inputs is safe; missing only one is a stale/incomplete pipeline."""
    if not readiness_path.exists() and not backtest_path.exists():
        return pd.DataFrame(), pd.DataFrame(), "FORECAST_FILES_NOT_YET_PRESENT"
    if not readiness_path.exists() or not backtest_path.exists():
        raise FileNotFoundError(
            "Only one of the two CPO forecasting output files is present. "
            "Re-run python src\\forecast_cpo_verified_sessions.py before creating the feed."
        )
    readiness = pd.read_csv(readiness_path, dtype="string", keep_default_na=False)
    backtests = pd.read_csv(backtest_path, dtype="string", keep_default_na=False)
    readiness.columns = readiness.columns.str.strip()
    backtests.columns = backtests.columns.str.strip()
    for label, frame, needed in (("forecast readiness", readiness, READINESS_NEEDED),
                                  ("forecast backtests", backtests, BACKTEST_NEEDED)):
        missing = set(needed) - set(frame.columns)
        if missing:
            raise ValueError(f"{label}: missing columns {sorted(missing)}. Re-run the coverage-gated forecast script")
        if frame["forecast_unit"].eq("").any() or frame["forecast_unit"].duplicated().any():
            raise ValueError(f"{label}: missing/duplicate forecast_unit; cannot match safely")
        frame["site_id"] = frame["site_id"].str.strip()
    if set(readiness["forecast_unit"]) != set(backtests["forecast_unit"]):
        raise ValueError("Forecast readiness and backtest outputs disagree on forecast units. Re-run forecasting")
    check = readiness[["forecast_unit", "site_id", "source_system", "status"]].merge(
        backtests[["forecast_unit", "site_id", "source_system", "status"]],
        on="forecast_unit", validate="one_to_one", suffixes=("_ready", "_backtest"))
    for col in ("site_id", "source_system", "status"):
        if (check[f"{col}_ready"].str.strip() != check[f"{col}_backtest"].str.strip()).any():
            raise ValueError(f"Forecast readiness/backtest disagree on {col}. Re-run forecasting")
    # Source labels are informative research evidence ONLY and never joined by city/name.
    return readiness, backtests, "FORECAST_FILES_READ"


def add_site_forecast_gates(result: pd.DataFrame, readiness: pd.DataFrame,
                            backtests: pd.DataFrame) -> pd.DataFrame:
    """Join by exact site ID; ambiguous site IDs fail closed; never approve investment."""
    joined = result.copy()
    evidence_fields = [
        "forecast_backtest_status", "forecast_evidence_gate", "forecast_source_system",
        "forecast_target", "forecast_selected_model", "forecast_identity_level",
        "forecast_reporting_evidence_available", "forecast_logging_completeness_verified",
        "forecast_unknown_hours_treated_as_zero", "forecast_train_coverage_pct",
        "forecast_validation_coverage_pct", "forecast_test_coverage_pct",
        "forecast_unseen_baseline_mae_starts_per_hour", "forecast_unseen_selected_mae_starts_per_hour",
        "forecast_unseen_mae_improvement_pct", "forecast_better_than_baseline",
        "forecast_matches_a_documented_site", "forecast_validated_for_expansion_decision",
        "forecast_actual_arrivals_known", "forecast_lost_demand_known",
    ]
    rows = []
    if readiness.empty:
        by_site = {}
        scores = {}
    else:
        by_site = {key: group for key, group in readiness.loc[readiness["site_id"].ne("")].groupby("site_id")}
        scores = backtests.set_index("forecast_unit", drop=False).to_dict("index")
    for site_id in joined["site_id"]:
        info = dict.fromkeys(evidence_fields, "")
        info.update({
            "forecast_backtest_status": "NO_SITE_BACKTEST",
            "forecast_evidence_gate": "NO_VERIFIED_SITE_FORECAST_EVIDENCE",
            "forecast_reporting_evidence_available": False,
            "forecast_logging_completeness_verified": False,
            "forecast_unknown_hours_treated_as_zero": "UNKNOWN",
            "forecast_better_than_baseline": False,
            "forecast_matches_a_documented_site": False,
            "forecast_validated_for_expansion_decision": False,
            "forecast_actual_arrivals_known": False,
            "forecast_lost_demand_known": False,
        })
        candidates = by_site.get(site_id)
        if candidates is None:
            rows.append(info)
            continue
        if len(candidates) != 1:
            info["forecast_evidence_gate"] = "AMBIGUOUS_MULTIPLE_SOURCE_BACKTESTS_REVIEW"
            rows.append(info)
            continue
        row = candidates.iloc[0]
        if (_truth(row["site_level_forecast_available"]) is not True
                or _text(row["identity_level"]) not in VALID_IDENTITY):
            info["forecast_evidence_gate"] = "SITE_ID_NOT_SUPPORTED_BY_DOCUMENTED_IDENTITY"
            rows.append(info)
            continue
        info["forecast_matches_a_documented_site"] = True
        score = scores.get(_text(row["forecast_unit"]))
        if score is None:
            info["forecast_evidence_gate"] = "MISSING_MATCHED_BACKTEST"
            rows.append(info)
            continue
        info.update({
            "forecast_backtest_status": _text(row["status"]),
            "forecast_source_system": _text(row["source_system"]),
            "forecast_target": _text(score["target"]),
            "forecast_selected_model": _text(score["model_selected_on_validation"]),
            "forecast_identity_level": _text(row["identity_level"]),
            "forecast_reporting_evidence_available": _truth(row["reporting_evidence_available"]) is True,
            "forecast_logging_completeness_verified": _truth(row["logging_completeness_verified"]) is True,
            "forecast_unknown_hours_treated_as_zero": _truth(row["unknown_hours_treated_as_zero_assumption"]),
            "forecast_actual_arrivals_known": _truth(row["actual_vehicle_arrivals_known"]) is True,
            "forecast_lost_demand_known": _truth(row["queues_and_lost_demand_known"]) is True,
        })
        for prefix, col in (("train", "train_coverage_pct"),
                            ("validation", "validation_coverage_pct"),
                            ("test", "unseen_test_coverage_pct")):
            info[f"forecast_{prefix}_coverage_pct"] = _numeric_or_nan(row[col], col, percent=True)
        for name, col in (("unseen_baseline_mae_starts_per_hour", "unseen_baseline_mae_starts_per_hour"),
                          ("unseen_selected_mae_starts_per_hour", "unseen_selected_mae_starts_per_hour"),
                          ("unseen_mae_improvement_pct", "unseen_mae_improvement_pct")):
            info[f"forecast_{name}"] = _numeric_or_nan(score[col], col)
        b = info["forecast_unseen_baseline_mae_starts_per_hour"]
        m = info["forecast_unseen_selected_mae_starts_per_hour"]
        better = bool(np.isfinite(b) and np.isfinite(m) and m <= b)
        info["forecast_better_than_baseline"] = better
        if _text(row["status"]) != SUPPORTED_BACKTEST_STATUS:
            info["forecast_evidence_gate"] = "SITE_FORECAST_NOT_EVIDENCE_GATED"
        elif _text(row["timezone_status"]) not in VALID_TIMEZONE:
            info["forecast_evidence_gate"] = "SITE_TIMEZONE_UNVERIFIED"
        elif not info["forecast_reporting_evidence_available"]:
            info["forecast_evidence_gate"] = "NO_HOURLY_REPORTING_EVIDENCE"
        elif not better:
            info["forecast_evidence_gate"] = "BACKTEST_DOES_NOT_BEAT_SIMPLE_BASELINE"
        elif not info["forecast_logging_completeness_verified"]:
            info["forecast_evidence_gate"] = "PARTIAL_HOURLY_COVERAGE_REVIEW_REQUIRED"
        else:
            # A genuine successful-starts backtest still cannot infer total driver arrivals.
            info["forecast_evidence_gate"] = "BACKTEST_SUPPORTED_FOR_SUCCESSFUL_STARTS_ONLY"
        rows.append(info)
    for name in evidence_fields:
        joined[name] = [r[name] for r in rows]
    # Real-backtest evidence never upgrades demo scenario, grid, or financial permissions.
    joined["forecast_validated_for_expansion_decision"] = False
    return joined


def assemble(tech: pd.DataFrame, fin: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    to_number(tech, [
        "current_chargers", "shortlisted_chargers", "additional_chargers",
        "rejection_pct_current", "rejection_pct_shortlisted",
        "additional_cars_served_in_test_period", "additional_energy_kwh_in_test_period",
        "max_rejected_pct_screening_target",
    ], "technical table")
    to_number(fin, [
        "npv_eur_illustrative", "annual_contribution_eur_illustrative",
        "breakeven_contribution_eur_per_kwh_illustrative", "annualized_demo_energy_kwh"
    ], "finance table")
    to_number(queue, ["evidence_fields_missing", "evidence_fields_total"], "action queue")

    tech_keys = set(tech[KEY].itertuples(index=False, name=None))
    queue_keys = set(queue[KEY].itertuples(index=False, name=None))
    expansion = tech.loc[tech["additional_chargers"] > 0].copy()
    expansion_keys = set(expansion[KEY].itertuples(index=False, name=None))
    finance_keys = set(fin[KEY].itertuples(index=False, name=None))
    if queue_keys != tech_keys:
        raise ValueError(
            "Operator action queue does not match technical cases. "
            "Re-run prepare_cpo_operator_evidence_intake.py."
        )
    if finance_keys != expansion_keys:
        raise ValueError(
            "Financial what-if results do not match technical expansion cases. "
            "Re-run run_cpo_financial_sensitivity.py --demo after the technical steps."
        )

    # Validate that the financial number belongs to exactly the chosen charger configuration.
    check = expansion[KEY + ["selected_scenario_id"]].merge(
        fin[KEY + ["selected_scenario_id"]], how="left", on=KEY,
        suffixes=("_technical", "_finance"), validate="one_to_one"
    )
    bad = (
        check["selected_scenario_id_technical"].str.strip()
        != check["selected_scenario_id_finance"].str.strip()
    )
    if bad.any():
        raise ValueError(
            "Different selected charger scenarios in financial and technical data. "
            "Re-run the financial what-if with the latest technical decisions."
        )

    if (tech["additional_chargers"] > 0).any() and (
        expansion["shortlisted_chargers"].isna().any()
        or (expansion["shortlisted_chargers"] != expansion["current_chargers"] + expansion["additional_chargers"]).any()
    ):
        raise ValueError("Shortlisted charger totals do not match current + additional")

    # Keep the full technical decision row, join the action queue and optional finance data.
    queue_columns = KEY + ["next_workflow_step", "evidence_fields_missing", "evidence_fields_total"]
    result = tech.merge(queue[queue_columns], on=KEY, how="left", validate="one_to_one")
    fin_columns = KEY + [
        "financial_status", "input_evidence_class", "npv_eur_illustrative",
        "annual_contribution_eur_illustrative",
        "breakeven_contribution_eur_per_kwh_illustrative", "annualized_demo_energy_kwh"
    ]
    result = result.merge(fin[fin_columns], on=KEY, how="left", validate="one_to_one")
    result["financial_status"] = result["financial_status"].fillna("NO_EXPANSION_TO_PRICE")
    result["input_evidence_class"] = result["input_evidence_class"].fillna("NOT_APPLICABLE")

    result["evidence_basis"] = "SYNTHETIC_DEMAND_AND_UNVERIFIED_SITE_INPUTS"
    result["decision_stage"] = "ENGINEERING_SCREENING_ONLY"
    result["investment_permission"] = "BLOCKED_VERIFIED_SITE_AND_FINANCIAL_DATA_REQUIRED"
    result["operator_decision"] = "NOT_ASSESSED"

    # Make important, curated fields easy to find for reporting, keep all technical context too.
    first = [
        "site_id", "city", "planning_year", "assumed_annual_arrival_growth_pct",
        "current_chargers", "shortlisted_chargers", "additional_chargers",
        "rejection_pct_current", "rejection_pct_shortlisted",
        "additional_cars_served_in_test_period", "additional_energy_kwh_in_test_period",
        "technical_next_action", "next_workflow_step", "grid_value_screen",
        "site_feasibility_gate", "evidence_fields_missing", "evidence_fields_total",
        "financial_status", "input_evidence_class", "npv_eur_illustrative",
        "breakeven_contribution_eur_per_kwh_illustrative", "annualized_demo_energy_kwh",
        "decision_stage", "investment_permission", "operator_decision", "evidence_basis",
    ]
    result = result[first + [c for c in result.columns if c not in first]]
    return result.sort_values(["assumed_annual_arrival_growth_pct", "site_id"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--technical", type=Path, default=TECH_DEFAULT)
    p.add_argument("--financial", type=Path, default=FIN_DEFAULT)
    p.add_argument("--queue", type=Path, default=QUEUE_DEFAULT)
    p.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--forecast-readiness", type=Path, default=FORECAST_READINESS_DEFAULT)
    p.add_argument("--forecast-backtests", type=Path, default=FORECAST_BACKTEST_DEFAULT)
    a = p.parse_args()

    tech = load_csv(a.technical, TECH_NEEDED, "technical decisions")
    fin = load_csv(a.financial, FIN_NEEDED, "financial what-if results")
    queue = load_csv(a.queue, QUEUE_NEEDED, "operator action queue")
    result = assemble(tech, fin, queue)
    readiness, backtests, forecast_file_status = _read_forecasts(a.forecast_readiness, a.forecast_backtests)
    result = add_site_forecast_gates(result, readiness, backtests)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)

    print("\n--- CPO NETWORK DECISION FEED / FORECAST EVIDENCE GATE ---")
    print(f"Script version: {VERSION}")
    print(f"Sites: {result.site_id.nunique()} | Demand cases: {len(result)} | Demo financial cases: "
          f"{(result['financial_status'] == 'ILLUSTRATIVE_WHAT_IF_ONLY').sum()}")
    show = result[[
        "city", "assumed_annual_arrival_growth_pct", "current_chargers",
        "shortlisted_chargers", "next_workflow_step", "evidence_fields_missing",
    ]].copy()
    show.columns = ["city", "growth_pct", "current", "screened", "next_action", "missing_inputs"]
    print(show.to_string(index=False))
    anonymous = int(readiness["site_id"].eq("").sum()) if not readiness.empty else 0
    matching = result.loc[result["forecast_matches_a_documented_site"], "site_id"].nunique()
    print(f"\nForecast inputs: {forecast_file_status} | Anonymous research forecast groups: {anonymous}")
    print(f"Site IDs with documented identity matching this demo network: {matching}")
    print("Forecast source gates:")
    print(result[["city", "assumed_annual_arrival_growth_pct", "forecast_evidence_gate"]].to_string(index=False))
    print("NOTE: No anonymous Mer location is mapped to a fictional site; no forecast replaces the synthetic scenarios.")
    print("Successful charging starts are NOT all vehicle arrivals or congestion observations.")
    print("\nFinancial NPV/break-even are ILLUSTRATIVE ONLY; not operator recommendations.")
    print("ALL investment decisions remain BLOCKED until data and feasibility are verified.")
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
