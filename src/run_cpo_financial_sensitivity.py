"""Configurable, evidence-gated CPO expansion FINANCIAL WHAT-IF.

Run from project root (CMD):
    python src\run_cpo_financial_sensitivity.py --demo

Reads prior combined technical scenarios. On first run creates an editable per-
expansion assumptions CSV. --demo fills missing assumptions with conspicuously
FICTIONAL values for code testing; it NEVER certifies market realism.

Does not overwrite the technical decision table or existing finance outputs.
All financial results are WHAT_IF_ONLY, never investment recommendations.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TECH = ROOT / "data" / "processed" / "cpo_network_technical_decisions.csv"
CONFIG = ROOT / "data" / "config" / "cpo_financial_whatif_inputs.csv"
OUT = ROOT / "data" / "processed" / "cpo_financial_whatif_results.csv"

KEY = ["site_id", "selected_scenario_id"]
ASSUMPTIONS = [
    "upfront_capex_eur",
    "annual_incremental_fixed_opex_eur",
    "net_incremental_contribution_eur_per_kwh",
    "discount_rate_pct",
    "evaluation_years",
]
CONFIG_COLUMNS = KEY + ["city", "scenario_chargers", "additional_chargers"] + ASSUMPTIONS + [
    "evidence_class", "evidence_reference", "notes"
]
NEEDED_TECH = KEY + [
    "city", "planning_year", "assumed_annual_arrival_growth_pct",
    "current_chargers", "shortlisted_chargers", "additional_chargers",
    "additional_energy_kwh_in_test_period", "demand_evidence_status"
]
# ONLY for testing the financial mathematics. NOT sourced German costs/tariffs.
FAKE_TEST = {
    "per_added_charger_eur": 90000.0,
    "fixed_site_works_eur": 30000.0,
    "annual_fixed_opex_eur": 5000.0,
    "net_contribution_eur_per_kwh": 0.20,
    "discount_rate_pct": 8.0,
    "evaluation_years": 7,
}
# The synthetic historical simulation covers 1 Jan through 30 Jun 2025.
OBSERVED_PATTERN_DAYS = 181.0


def required_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [col for col in cols if col not in df]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def number(value: object, label: str, allow_zero: bool = True) -> float:
    result = pd.to_numeric(value, errors="raise")
    result = float(result)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"Invalid {label}: {value!r}")
    return result


def read_technical() -> pd.DataFrame:
    if not TECH.is_file():
        raise FileNotFoundError(f"Missing {TECH}; run build_cpo_technical_decision_table.py first")
    d = pd.read_csv(TECH, dtype={"site_id": "string", "selected_scenario_id": "string"})
    d.columns = d.columns.str.strip()
    required_cols(d, NEEDED_TECH, "Technical decision input")
    if d.empty or d[KEY].isna().any().any():
        raise ValueError("Empty technical decisions or missing IDs")
    if d.duplicated(["site_id", "planning_year", "assumed_annual_arrival_growth_pct"]).any():
        raise ValueError("Duplicate site/year/demand assumption")
    for col in ["additional_chargers", "additional_energy_kwh_in_test_period"]:
        d[col] = pd.to_numeric(d[col], errors="raise")
        if d[col].isna().any() or (d[col] < 0).any():
            raise ValueError(f"Invalid {col} in technical decisions")
    return d


def build_config(d: pd.DataFrame, demo: bool) -> tuple[pd.DataFrame, int]:
    # One configuration per physical investment; demand assumptions may differ.
    expansions = d.loc[d["additional_chargers"] > 0, KEY + ["city", "shortlisted_chargers", "additional_chargers"]].copy()
    conflicts = expansions.groupby(KEY).agg(
        chargers=("shortlisted_chargers", "nunique"),
        added=("additional_chargers", "nunique"),
    )
    if (conflicts > 1).any().any():
        raise ValueError("One investment scenario has conflicting hardware counts")
    expansions = expansions.drop_duplicates(KEY)
    expansions = expansions.rename(columns={"shortlisted_chargers": "scenario_chargers"})
    rows = []
    for item in expansions.itertuples(index=False):
        record = {
            "site_id": str(item.site_id),
            "selected_scenario_id": str(item.selected_scenario_id),
            "city": str(item.city),
            "scenario_chargers": int(item.scenario_chargers),
            "additional_chargers": int(item.additional_chargers),
            "evidence_class": "MISSING",
            "evidence_reference": "",
            "notes": "Add a scenario-specific, dated operator/supplier quote; do not use retail prices as net margin",
        }
        for field in ASSUMPTIONS:
            record[field] = ""
        if demo:
            n = record["additional_chargers"]
            record.update({
                "upfront_capex_eur": FAKE_TEST["per_added_charger_eur"] * n + FAKE_TEST["fixed_site_works_eur"],
                "annual_incremental_fixed_opex_eur": FAKE_TEST["annual_fixed_opex_eur"],
                "net_incremental_contribution_eur_per_kwh": FAKE_TEST["net_contribution_eur_per_kwh"],
                "discount_rate_pct": FAKE_TEST["discount_rate_pct"],
                "evaluation_years": FAKE_TEST["evaluation_years"],
                "evidence_class": "FICTIONAL_MATH_TEST",
                "evidence_reference": "NONE - explicit demonstration values, NOT market evidence",
                "notes": "FOR TESTING ONLY. Costs and margin are not validated German CPO values",
            })
        rows.append(record)
    template = pd.DataFrame(rows, columns=CONFIG_COLUMNS)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG.exists():
        template.to_csv(CONFIG, index=False)
        return template, len(template)
    existing = pd.read_csv(CONFIG, dtype=str, keep_default_na=False)
    existing.columns = existing.columns.str.strip()
    required_cols(existing, CONFIG_COLUMNS, "Existing financial assumptions")
    if existing.duplicated(KEY).any():
        raise ValueError("Duplicate site_id + selected_scenario_id in assumptions CSV")
    template_keys = set(map(tuple, template[KEY].values.tolist()))
    existing_keys = set(map(tuple, existing[KEY].values.tolist()))
    added = template.loc[~template[KEY].apply(tuple, axis=1).isin(existing_keys)]
    if not added.empty:
        existing = pd.concat([existing, added], ignore_index=True)
        existing.to_csv(CONFIG, index=False)
    unused = existing_keys - template_keys
    if unused:
        print(f"NOTE: {len(unused)} stale assumption rows were preserved but not analyzed")
    return existing, len(added)


def pv_annuity_factor(rate: float, years: int) -> float:
    if rate == 0:
        return float(years)
    return (1 - (1 + rate) ** (-years)) / rate


def calculate(d: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    candidates = d.loc[d["additional_chargers"] > 0].copy()
    if candidates.empty:
        return pd.DataFrame()
    merged = candidates.merge(
        config[KEY + ASSUMPTIONS + ["evidence_class", "evidence_reference"]],
        how="left", on=KEY, validate="many_to_one", indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("No financial configuration for some expansion scenarios")
    rows = []
    for r in merged.to_dict("records"):
        result = {
            "site_id": r["site_id"], "city": r["city"],
            "planning_year": r["planning_year"],
            "assumed_annual_arrival_growth_pct": r["assumed_annual_arrival_growth_pct"],
            "selected_scenario_id": r["selected_scenario_id"],
            "current_chargers": r["current_chargers"],
            "shortlisted_chargers": r["shortlisted_chargers"],
            "additional_chargers": r["additional_chargers"],
            "incremental_energy_kwh_synthetic_181_days": round(float(r["additional_energy_kwh_in_test_period"]), 2),
            "annualized_demo_energy_kwh": None,
            "input_evidence_class": r["evidence_class"],
            "financial_status": "NOT_CALCULATED_MISSING_INPUTS",
            "npv_eur_illustrative": None,
            "annual_contribution_eur_illustrative": None,
            "breakeven_contribution_eur_per_kwh_illustrative": None,
            "investment_recommendation": "BLOCKED_UNVERIFIED_OPERATOR_AND_FUTURE_DEMAND_INPUTS",
        }
        vals = [r[x] for x in ASSUMPTIONS]
        if any(pd.isna(v) or str(v).strip() == "" for v in vals):
            rows.append(result)
            continue
        capex = number(r["upfront_capex_eur"], "upfront capex")
        opex = number(r["annual_incremental_fixed_opex_eur"], "annual fixed opex")
        margin = number(r["net_incremental_contribution_eur_per_kwh"], "contribution margin")
        rate = number(r["discount_rate_pct"], "discount rate") / 100.0
        years = number(r["evaluation_years"], "evaluation years", allow_zero=False)
        if not years.is_integer() or years > 50 or rate > 1:
            raise ValueError("Evaluation years must be an integer 1-50, discount rate <=100%")
        annual_energy = float(r["additional_energy_kwh_in_test_period"]) * (365.0 / OBSERVED_PATTERN_DAYS)
        annuity = pv_annuity_factor(rate, int(years))
        annual_cash = annual_energy * margin - opex
        npv = -capex + annual_cash * annuity
        break_even_margin = ((capex / annuity) + opex) / annual_energy if annual_energy > 0 else None
        result.update({
            "annualized_demo_energy_kwh": round(annual_energy, 1),
            "financial_status": "ILLUSTRATIVE_WHAT_IF_ONLY",
            "annual_contribution_eur_illustrative": round(annual_cash),
            "npv_eur_illustrative": round(npv),
            "breakeven_contribution_eur_per_kwh_illustrative": (
                round(break_even_margin, 3) if break_even_margin is not None else None
            ),
        })
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial what-if with explicit evidence gate")
    parser.add_argument("--demo", action="store_true", help="Fill *new* input rows with fictional test values")
    args = parser.parse_args()

    decisions = read_technical()
    config, added = build_config(decisions, args.demo)
    result = calculate(decisions, config)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print("\n--- CPO FINANCIAL WHAT-IF (NOT INVESTMENT ADVICE) ---")
    print(f"Sites: {decisions.site_id.nunique()} | Expansion/demand cases: {len(result)}")
    print(f"New scenario-specific assumptions added: {added}")
    if not result.empty:
        print(result[[
            "city", "assumed_annual_arrival_growth_pct", "additional_chargers",
            "annualized_demo_energy_kwh", "npv_eur_illustrative",
            "breakeven_contribution_eur_per_kwh_illustrative", "financial_status"
        ]].to_string(index=False))
    print("\nALL investment decisions BLOCKED: simulated demand, annualized half-year energy,")
    print("and fictional or unverified cost inputs cannot establish real commercial value.")
    print(f"Editable financial inputs: {CONFIG}")
    print(f"Saved what-if results: {OUT}")


if __name__ == "__main__":
    main()
