"""Audit the existing CPO decision pipeline before building the dashboard.

Run from CPO Network Development project root:
    python src\audit_cpo_end_to_end.py

This audits OUTPUT CONSISTENCY and evidence controls, not real-world accuracy.
It deliberately keeps the registry simulation (synthetic 2025 arrivals) separate
from the technical/financial screening (conditional 2027 demand assumptions).
It never maps anonymous Mer sites to fictional site IDs.

Inputs (existing files in data/processed):
  registry_network_simulation_results.csv
  registry_network_simulation_impact.csv
  cpo_network_technical_decisions.csv
  cpo_financial_whatif_results.csv
  cpo_workflow_action_queue.csv
  cpo_decision_dashboard_feed.csv
  cpo_standard_starts_data_readiness.csv
  cpo_standard_starts_backtest_sites.csv

Outputs (new files in data/processed):
  cpo_end_to_end_audit.csv
  cpo_end_to_end_audit_summary.json

Run after regenerating your latest simulation, forecast, and decision feed.
No source CSVs are changed. Exits with code 1 for any failed checks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026-10-31-end-to-end-audit-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed"
FILES = {
    "simulation": "registry_network_simulation_results.csv",
    "impact": "registry_network_simulation_impact.csv",
    "technical": "cpo_network_technical_decisions.csv",
    "finance": "cpo_financial_whatif_results.csv",
    "queue": "cpo_workflow_action_queue.csv",
    "feed": "cpo_decision_dashboard_feed.csv",
    "readiness": "cpo_standard_starts_data_readiness.csv",
    "backtests": "cpo_standard_starts_backtest_sites.csv",
}
CASE_KEY = ["site_id", "planning_year", "assumed_annual_arrival_growth_pct"]
SIM_KEY = ["site_id", "scenario_id"]


class Report:
    def __init__(self):
        self.rows = []

    def add(self, section: str, check: str, status: str, details: str):
        self.rows.append({"section": section, "check": check,
                          "status": status, "details": str(details)})

    def test(self, section: str, check: str, ok: bool, good: str, bad: str):
        self.add(section, check, "PASS" if bool(ok) else "FAIL", good if bool(ok) else bad)

    def warn(self, section: str, check: str, reason: str):
        self.add(section, check, "LIMITATION", reason)

    def count(self, status: str) -> int:
        return sum(r["status"] == status for r in self.rows)


def raw(path: Path, label: str, report: Report) -> pd.DataFrame | None:
    if not path.is_file():
        report.add("INPUT", label, "FAIL", f"Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        if df.empty:
            raise ValueError("CSV contains no records")
    except Exception as exc:
        report.add("INPUT", label, "FAIL", f"Cannot read: {exc}")
        return None
    report.add("INPUT", label, "PASS", f"{len(df):,} rows")
    return df


def columns(df: pd.DataFrame, section: str, names: list[str], report: Report) -> bool:
    missing = sorted(set(names) - set(df.columns))
    report.test(section, "required columns", not missing,
                "Required fields present", f"Missing: {missing}")
    return not missing


def keyset(df: pd.DataFrame, fields: list[str], section: str, report: Report):
    if not columns(df, section, fields, report):
        return None
    if df[fields].isna().any().any() or df[fields].astype(str).eq("").any().any():
        report.add(section, "nonempty keys", "FAIL", f"Blank key in {fields}")
        return None
    if df.duplicated(fields).any():
        report.add(section, "unique keys", "FAIL", f"Duplicated key in {fields}")
        return None
    report.add(section, "unique keys", "PASS", f"{len(df):,} unique rows")
    return set(df[fields].itertuples(index=False, name=None))


def nums(df: pd.DataFrame, *names: str) -> pd.DataFrame:
    return df.loc[:, list(names)].apply(pd.to_numeric, errors="coerce")


def equal_number(x, y, tolerance=0.001):
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    return np.isfinite(a).all() and np.isfinite(b).all() and bool(np.all(np.abs(a-b) <= tolerance))


def bool_col(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def check_simulation(sim: pd.DataFrame, impact: pd.DataFrame, r: Report):
    label = "SIMULATION"
    required = SIM_KEY + ["scenario_type", "scenario_chargers", "scenario_connectors",
       "scenario_simultaneous_positions", "grid_capacity_kw", "arrivals", "cars_served",
       "cars_rejected", "rejected_pct", "cars_fully_charged", "energy_requested_kwh",
       "energy_delivered_kwh", "energy_rejected_kwh", "energy_shortfall_served_kwh",
       "total_unserved_energy_kwh", "peak_site_grid_draw_kw", "minutes_at_grid_limit",
       "investment_readiness"]
    impact_required = SIM_KEY + ["additional_cars_served", "additional_energy_delivered_kwh",
                                     "change_in_total_unserved_energy_kwh", "minutes_at_grid_limit"]
    keys = keyset(sim, SIM_KEY, label, r)
    ikeys = keyset(impact, SIM_KEY, "SIMULATION IMPACT", r)
    if keys is None or ikeys is None or not columns(sim, label, required, r) or not columns(impact, "SIMULATION IMPACT", impact_required, r):
        return
    r.test(label, "simulation vs impact scenarios", keys == ikeys, "Scenario IDs match",
           f"Simulation-only: {len(keys-ikeys)}; impact-only: {len(ikeys-keys)}")
    x = sim.copy()
    nfields = [col for col in required if col not in SIM_KEY + ["scenario_type", "investment_readiness"]]
    x[nfields] = nums(x, *nfields)
    invalid = int((~np.isfinite(x[nfields].to_numpy(dtype=float))).sum())
    r.test(label, "numeric values", invalid == 0, "All core metrics numeric", f"{invalid} invalid values")
    if invalid:
        return
    baseline = x[x.scenario_type.eq("CURRENT")]
    counts = baseline.groupby("site_id").size()
    r.test(label, "one current scenario per site", len(counts) == x.site_id.nunique() and counts.eq(1).all(),
           f"{len(counts)} baseline scenarios", "Missing or duplicate CURRENT scenarios")
    valid_counts = ((x.arrivals == x.cars_served + x.cars_rejected)
                    & (x.cars_fully_charged <= x.cars_served)
                    & (x.scenario_chargers > 0)
                    & (x.scenario_connectors >= x.scenario_simultaneous_positions)
                    & (x.scenario_simultaneous_positions > 0)
                    & (x.cars_rejected >= 0))
    r.test(label, "vehicle and connector accounting", valid_counts.all(), "Vehicle counts and connector limits balanced",
           f"{(~valid_counts).sum()} rows inconsistent")
    expected = 100 * x.cars_rejected / x.arrivals.replace(0, np.nan)
    r.test(label, "reported rejection percentages", x.arrivals.gt(0).all() and (x.rejected_pct - expected).abs().le(0.025).all(),
           "Rejection rates reproduce", "Rejection percentages disagree with counts")
    p_ok = (x.peak_site_grid_draw_kw <= x.grid_capacity_kw + 0.05) & x.peak_site_grid_draw_kw.ge(-0.05)
    r.test(label, "site grid power cap", p_ok.all(), "All simulated peaks within supplied import limits",
           f"{(~p_ok).sum()} scenarios exceed site import")
    energy_ok = ((x.energy_requested_kwh - x.energy_delivered_kwh - x.total_unserved_energy_kwh).abs() <= 0.25)
    energy_ok &= ((x.energy_rejected_kwh + x.energy_shortfall_served_kwh - x.total_unserved_energy_kwh).abs() <= 0.25)
    energy_ok &= x.energy_delivered_kwh.ge(-0.05)
    r.test(label, "energy balance", energy_ok.all(), "Requested = delivered + unserved (within rounding)",
           f"{(~energy_ok).sum()} energy-balance mismatches")
    r.test(label, "simulated-only investment gate",
           x.investment_readiness.astype(str).str.contains("DEMO|EVIDENCE|BLOCKED|NOT_READY", case=False, regex=True).all(),
           "Synthetic scenarios are not investment-approved", "At least one simulation claims unsupported readiness")
    if len(counts) != x.site_id.nunique() or not counts.eq(1).all():
        return
    y = impact.merge(x[SIM_KEY + ["cars_served", "energy_delivered_kwh",
                                "total_unserved_energy_kwh", "minutes_at_grid_limit"]],
                     on=SIM_KEY, validate="one_to_one")
    baseline_by_site = baseline.set_index("site_id")
    expected_cars = y.cars_served - y.site_id.map(baseline_by_site.cars_served)
    expected_energy = y.energy_delivered_kwh - y.site_id.map(baseline_by_site.energy_delivered_kwh)
    expected_unserved = y.total_unserved_energy_kwh - y.site_id.map(baseline_by_site.total_unserved_energy_kwh)
    good = ((pd.to_numeric(y.additional_cars_served, errors="coerce") - expected_cars).abs().le(0.01)
            & (pd.to_numeric(y.additional_energy_delivered_kwh, errors="coerce") - expected_energy).abs().le(0.25)
            & (pd.to_numeric(y.change_in_total_unserved_energy_kwh, errors="coerce") - expected_unserved).abs().le(0.25)
            & (pd.to_numeric(y.minutes_at_grid_limit_x, errors="coerce") == y.minutes_at_grid_limit_y))
    r.test("SIMULATION IMPACT", "changes match baseline", good.all(),
           "All expansion impacts reproduce from baseline", f"{(~good).sum()} impact mismatches")


def normalize_case(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["site_id"] = d["site_id"].astype(str).str.strip()
    d["planning_year"] = pd.to_numeric(d["planning_year"], errors="coerce")
    d["assumed_annual_arrival_growth_pct"] = pd.to_numeric(d["assumed_annual_arrival_growth_pct"], errors="coerce").round(8)
    return d


def check_decision(tech: pd.DataFrame, finance: pd.DataFrame, queue: pd.DataFrame, feed: pd.DataFrame, r: Report):
    label = "DECISION FEED"
    needed = CASE_KEY + ["selected_scenario_id", "current_chargers", "shortlisted_chargers", "additional_chargers",
                         "rejection_pct_current", "rejection_pct_shortlisted", "additional_cars_served_in_test_period",
                         "additional_energy_kwh_in_test_period"]
    needed_feed = needed + ["investment_permission", "operator_decision", "financial_status", "input_evidence_class",
                           "evidence_fields_missing", "evidence_fields_total", "next_workflow_step",
                           "forecast_evidence_gate", "forecast_matches_a_documented_site",
                           "forecast_validated_for_expansion_decision", "forecast_actual_arrivals_known",
                           "forecast_lost_demand_known", "npv_eur_illustrative", "annual_contribution_eur_illustrative",
                           "breakeven_contribution_eur_per_kwh_illustrative", "annualized_demo_energy_kwh"]
    if not all([columns(tech, "TECHNICAL", needed, r),
                columns(feed, label, needed_feed, r),
                columns(finance, "FINANCIAL", CASE_KEY + ["selected_scenario_id", "financial_status", "input_evidence_class",
                    "npv_eur_illustrative", "annual_contribution_eur_illustrative",
                    "breakeven_contribution_eur_per_kwh_illustrative", "annualized_demo_energy_kwh"], r),
                columns(queue, "ACTION QUEUE", CASE_KEY + ["evidence_fields_missing", "evidence_fields_total", "next_workflow_step"], r)]):
        return
    tech, finance, queue, feed = [normalize_case(d) for d in (tech, finance, queue, feed)]
    tkeys = keyset(tech, CASE_KEY, "TECHNICAL", r)
    fkeys = keyset(feed, CASE_KEY, label, r)
    qkeys = keyset(queue, CASE_KEY, "ACTION QUEUE", r)
    bkeys = keyset(finance, CASE_KEY, "FINANCIAL", r)
    if None in (tkeys, fkeys, qkeys, bkeys):
        return
    r.test(label, "technical vs feed cases", tkeys == fkeys, "Same case IDs", "Dashboard feed differs from technical decisions")
    r.test(label, "action queue cases", tkeys == qkeys, "Action queue covers every demand case",
           "Action queue keys differ from technical cases")
    expansion = tech[pd.to_numeric(tech.additional_chargers, errors="coerce").gt(0)]
    expkeys = set(expansion[CASE_KEY].itertuples(index=False, name=None))
    r.test(label, "finance cases", expkeys == bkeys, "Financial what-if present for every selected expansion",
           f"Expansions: {len(expkeys)}, finance cases: {len(bkeys)} (IDs differ)")
    if tkeys != fkeys or tkeys != qkeys or expkeys != bkeys:
        return
    merged = tech.merge(feed, on=CASE_KEY, suffixes=("_tech", "_feed"), validate="one_to_one")
    numerics = ["current_chargers", "shortlisted_chargers", "additional_chargers", "rejection_pct_current",
                "rejection_pct_shortlisted", "additional_cars_served_in_test_period", "additional_energy_kwh_in_test_period"]
    n_ok = all(equal_number(merged[f"{col}_tech"], merged[f"{col}_feed"], 0.001) for col in numerics)
    s_ok = (merged["selected_scenario_id_tech"].astype(str).str.strip()
            == merged["selected_scenario_id_feed"].astype(str).str.strip()).all()
    r.test(label, "technical figures copied correctly", n_ok and s_ok,
           "All technical counts, metrics and scenario IDs agree", "Dashboard feed has stale or changed technical values")
    joined = queue.merge(feed[CASE_KEY + ["evidence_fields_missing", "evidence_fields_total", "next_workflow_step"]],
                         on=CASE_KEY, suffixes=("_queue", "_feed"), validate="one_to_one")
    q_ok = (equal_number(joined.evidence_fields_missing_queue, joined.evidence_fields_missing_feed) and
            equal_number(joined.evidence_fields_total_queue, joined.evidence_fields_total_feed) and
            (joined.next_workflow_step_queue.astype(str) == joined.next_workflow_step_feed.astype(str)).all())
    r.test(label, "operator evidence queue copied", q_ok, "Evidence counts and actions agree", "Feed and action queue differ")
    mismatch = merged[["site_id", "planning_year", "assumed_annual_arrival_growth_pct", "additional_chargers_feed",
                        "selected_scenario_id_feed"]].merge(finance,
                         on=CASE_KEY, how="left", suffixes=("_feed", "_finance"), validate="one_to_one")
    selected = mismatch[pd.to_numeric(mismatch.additional_chargers_feed, errors="coerce").gt(0)]
    f_ok = (selected.selected_scenario_id_feed.astype(str).str.strip()
            == selected["selected_scenario_id"].astype(str).str.strip()).all()
    r.test(label, "financial scenario identity", f_ok, "Every illustrative NPV belongs to selected scenario",
           "Financial case scenario differs from technical option")
    finance_match = finance.merge(feed[CASE_KEY + ["financial_status", "input_evidence_class",
        "npv_eur_illustrative", "annual_contribution_eur_illustrative",
        "breakeven_contribution_eur_per_kwh_illustrative", "annualized_demo_energy_kwh"]],
        on=CASE_KEY, how="left", suffixes=("_finance", "_feed"), validate="one_to_one")
    fin_metrics = ["npv_eur_illustrative", "annual_contribution_eur_illustrative",
                   "breakeven_contribution_eur_per_kwh_illustrative", "annualized_demo_energy_kwh"]
    finance_equal = all(equal_number(finance_match[f"{c}_finance"],
                                     finance_match[f"{c}_feed"], 0.001) for c in fin_metrics)
    finance_equal = finance_equal and all((finance_match[f"{c}_finance"].astype(str)
                                     == finance_match[f"{c}_feed"].astype(str)).all()
                                    for c in ("financial_status", "input_evidence_class"))
    r.test(label, "finance values copied unchanged", finance_equal,
           "Illustrative NPV, margins and evidence labels match finance source",
           "Dashboard feed financial values or evidence labels differ from source")
    investments = feed.investment_permission.fillna("").astype(str).str.upper()
    decisions = feed.operator_decision.fillna("").astype(str).str.upper()
    r.test(label, "investment approval blocked", investments.str.contains("BLOCKED").all()
           and decisions.eq("NOT_ASSESSED").all(), "No fictional site approved for investment",
           "At least one demo row appears to grant an investment decision")
    demo_finance = feed[feed.additional_chargers.astype(float).gt(0)]
    r.test(label, "financial numbers clearly illustrative", demo_finance.financial_status.astype(str).str.contains("ILLUSTRATIVE").all()
           and demo_finance.input_evidence_class.astype(str).str.contains("FICTIONAL|DEMO|ASSUM", case=False, regex=True).all(),
           "Illustrative cases clearly marked", "At least one expansion has unsupported financial evidence label")
    required_evidence = nums(feed, "evidence_fields_missing", "evidence_fields_total")
    ev = (required_evidence.evidence_fields_missing >= 0) & (required_evidence.evidence_fields_missing <= required_evidence.evidence_fields_total)
    r.test(label, "evidence gap counts", ev.all(), "Missing-input counts within valid bounds", "Invalid evidence gaps")
    r.test(label, "forecast claims not promoted", (~bool_col(feed.forecast_validated_for_expansion_decision)).all()
           and (~bool_col(feed.forecast_actual_arrivals_known)).all()
           and (~bool_col(feed.forecast_lost_demand_known)).all(),
           "Successful charging starts never promoted to total arrivals or investment readiness",
           "Feed incorrectly claims validated expansion or all arrival/queue observations")


def check_forecast(readiness: pd.DataFrame, backtest: pd.DataFrame, feed: pd.DataFrame, r: Report):
    section = "FORECAST EVIDENCE"
    needed = ["forecast_unit", "site_id", "status", "source_system"]
    if not columns(readiness, section, needed + ["site_level_forecast_available", "logging_completeness_verified"], r):
        return
    if not columns(backtest, section, needed, r):
        return
    rk = keyset(readiness, ["forecast_unit"], section + " READINESS", r)
    bk = keyset(backtest, ["forecast_unit"], section + " BACKTEST", r)
    if rk is None or bk is None:
        return
    r.test(section, "forecast file identity", rk == bk, "Readiness and backtest unit IDs agree",
           "Forecast readiness/backtest contain different units")
    if rk != bk:
        return
    both = readiness.merge(backtest, on="forecast_unit", suffixes=("_r", "_b"), validate="one_to_one")
    same = all((both[f"{col}_r"].fillna("").astype(str)
                == both[f"{col}_b"].fillna("").astype(str)).all()
               for col in ["site_id", "status", "source_system"])
    r.test(section, "forecast status consistency", same, "Site IDs, statuses and sources agree",
           "Readiness/backtest disagree about site identity or status")
    mapped_ids = set(readiness.site_id.dropna().astype(str).str.strip()) - {""}
    feed_ids = set(feed.site_id.dropna().astype(str).str.strip())
    linked_ids = mapped_ids & feed_ids
    claims = feed[bool_col(feed.forecast_matches_a_documented_site)]
    truth = set(claims.site_id.astype(str).str.strip()) <= linked_ids
    r.test(section, "site forecast joins", truth, "Any documented forecast joins by exact site ID only",
           "Feed claims site matching without forecast site identity")
    unnamed = int(readiness.site_id.isna().sum() + readiness.site_id.fillna("").astype(str).str.strip().eq("").sum())
    if unnamed:
        r.warn(section, "anonymous research histories", f"{int(readiness.site_id.fillna('').astype(str).str.strip().eq('').sum())} anonymous forecast groups, not joinable to site investments")
    logging = bool_col(readiness.logging_completeness_verified)
    if not logging.any():
        r.warn(section, "reporting completeness", "No forecast group has verified complete operator reporting")
    r.warn("SCOPE", "synthetic vs observed pipelines", "The 2025 synthetic scenario simulation and historical Mer backtests are distinct datasets; do not claim that Mer validates simulation rejection rates")
    r.warn("SCOPE", "grid and investment", "Registered charger ratings are not DSO-approved grid import; costs, waiting queues and lost arrivals are unverified")


def run(processed: Path) -> Report:
    r = Report()
    datasets = {label: raw(processed / file, label, r) for label, file in FILES.items()}
    for label, file in FILES.items():
        if datasets[label] is None:
            continue
        if len(datasets[label]) == 0:
            r.add("INPUT", label, "FAIL", "Empty required dataset")
    if all(datasets[k] is not None for k in ("simulation", "impact")):
        check_simulation(datasets["simulation"], datasets["impact"], r)
    if all(datasets[k] is not None for k in ("technical", "finance", "queue", "feed")):
        check_decision(datasets["technical"], datasets["finance"], datasets["queue"], datasets["feed"], r)
    if all(datasets[k] is not None for k in ("readiness", "backtests", "feed")):
        check_forecast(datasets["readiness"], datasets["backtests"], datasets["feed"], r)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, default=DEFAULT_DATA,
                   help="Location of existing pipeline CSV outputs")
    args = p.parse_args()
    result = run(args.processed)
    args.processed.mkdir(parents=True, exist_ok=True)
    checks_path = args.processed / "cpo_end_to_end_audit.csv"
    summary_path = args.processed / "cpo_end_to_end_audit_summary.json"
    checks = pd.DataFrame(result.rows, columns=["section", "check", "status", "details"])
    checks.to_csv(checks_path, index=False)
    summary = {"audit_version": VERSION, "pass": result.count("PASS"),
               "fail": result.count("FAIL"), "limitations": result.count("LIMITATION"),
               "status": "FAIL_FIX_BEFORE_DASHBOARD" if result.count("FAIL") else "PASS_WITH_DOCUMENTED_LIMITATIONS",
               "scope": "Internal output consistency only; NOT verified real-world prediction accuracy"}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n--- CPO END-TO-END RELIABILITY AUDIT ---")
    print(f"PASS: {summary['pass']} | FAIL: {summary['fail']} | Documented limitations: {summary['limitations']}")
    print("\nChecks requiring attention:")
    for row in result.rows:
        if row["status"] != "PASS":
            print(f"  [{row['status']}] {row['section']}: {row['check']} — {row['details']}")
    print(f"\nRESULT: {summary['status']}")
    print(f"Saved: {checks_path}\nSaved: {summary_path}")
    print("This verifies pipeline consistency, NOT real-world forecast/rejection accuracy.")
    if summary["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
