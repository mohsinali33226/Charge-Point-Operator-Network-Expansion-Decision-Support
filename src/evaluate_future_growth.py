r"""Growth-aware CPO expansion business cases from simulated charging demand.

Run from the existing CPO project folder:
    python src\evaluate_future_growth.py

Requires existing files from:
    generate_network_scenarios.py
    run_network_scenarios.py
    build_network_business_case.py

Reads historic synthetic arrivals, per-site infrastructure, expansion scenarios,
previous current-demand impacts, and editable financial assumptions. Evaluates
future *illustrative* arrival-growth scenarios by replaying each site and its
expansion with the SAME growing vehicle demand in each future year.

Creates:
  data/config/arrival_growth_scenarios.json (editable on first run)
  data/processed/network_growth_yearly_results.csv
  data/processed/network_growth_business_cases.csv
  data/processed/network_growth_priority.csv

Limitations: assumed repeated 2025 seasonal pattern; modelled annual arrival
growth, not measured or forecast EV adoption. Rejected-driver power/stay data
are imputed in the established charging simulator. No waiting queue, battery
curves, charger reliability, grid application lead times or actual supplier
quotes. Entire analysis is a synthetic decision-support demonstration.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_expansion import load_data, simulate_site
from run_network_scenarios import read_scenarios, validate_against_master
from build_network_business_case import get_assumptions, site_observation_periods

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CONFIG_FILE = ROOT / "data" / "config" / "arrival_growth_scenarios.json"
IMPACT_FILE = PROCESSED / "network_scenario_incremental_impact.csv"
SCENARIOS_FILE = PROCESSED / "network_expansion_scenarios.csv"
YEARLY_FILE = PROCESSED / "network_growth_yearly_results.csv"
CASES_FILE = PROCESSED / "network_growth_business_cases.csv"
PRIORITY_FILE = PROCESSED / "network_growth_priority.csv"

DEFAULT_GROWTH_CONFIG = {
    "annual_arrival_growth_rates_pct": [0, 15],
    "random_seed": 2026
}


def get_growth_settings():
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_GROWTH_CONFIG, indent=2) + "\n", encoding="utf-8")
        print(f"Created editable arrival-growth scenarios: {CONFIG_FILE}")
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    rates = config.get("annual_arrival_growth_rates_pct")
    if not isinstance(rates, list) or not rates:
        raise ValueError("annual_arrival_growth_rates_pct must be a nonempty list, e.g. [0, 15].")
    for r in rates:
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r < 0:
            raise ValueError(f"Invalid annual growth rate: {r!r}; use nonnegative percentages.")
    seed = config.get("random_seed", 2026)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("random_seed must be a nonnegative integer.")
    return sorted(set(float(r) for r in rates)), seed


def make_future_arrivals(base_arrivals, factor, site_id, rate, year_number, global_seed):
    """Preserve all base arrivals and add sampled historical-like arrivals.

    The same newly generated arrivals are used in the baseline AND every
    expansion option. Only the number of cars changes, not vehicle energies.
    Sampling with replacement is deliberately a synthetic growth assumption.
    """
    required = ["start_time", "requested_energy_kwh", "requested_power_kw", "max_stay_min"]
    base = base_arrivals[required].copy().reset_index(drop=True)
    if factor < 1 or base.empty:
        raise ValueError("Expected nonempty historic arrivals and growth factor >= 1.")
    additional = round(len(base) * (factor - 1))
    if additional == 0:
        return base

    id_string = f"{global_seed}|{site_id}|{rate}|{year_number}"
    seed = int.from_bytes(hashlib.sha256(id_string.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    extras = base.iloc[rng.integers(0, len(base), size=additional)].copy().reset_index(drop=True)
    # Avoid artificially stacking all repeated sessions at exactly one minute.
    # This is a simple jitter, NOT a stochastic arrival forecast.
    changes = rng.integers(-30, 31, size=additional)
    extras["start_time"] = pd.to_datetime(extras["start_time"]) + pd.to_timedelta(changes, unit="m")
    lower = pd.to_datetime(base["start_time"]).min()
    upper = pd.to_datetime(base["start_time"]).max()
    extras["start_time"] = extras["start_time"].clip(lower=lower, upper=upper)
    result = pd.concat([base, extras], ignore_index=True)
    return result.sort_values("start_time", kind="stable").reset_index(drop=True)


def cost_for(case, current, m):
    additional_chargers = int(case["scenario_chargers"]) - int(current["scenario_chargers"])
    additional_grid_kw = float(case["scenario_grid_kw"]) - float(current["scenario_grid_kw"])
    if additional_chargers < 0 or additional_grid_kw < -0.000001:
        raise ValueError("Scenario is a reduction, not an expansion; unsupported by this case.")
    new_chargers = additional_chargers > 0
    grid_upgrade = additional_grid_kw > 0.000001
    capex = (additional_chargers * m["eur_new_charger_equipment_and_installation"]
             + int(new_chargers) * m["eur_fixed_site_expansion_works"]
             + int(grid_upgrade) * m["eur_grid_upgrade_fixed"]
             + max(0, additional_grid_kw) * m["eur_grid_upgrade_per_added_kw"])
    annual_opex = (additional_chargers * m["eur_annual_maintenance_per_new_charger"]
                   + int(new_chargers) * m["eur_annual_extra_site_overhead"])
    return capex, annual_opex


def main():
    assumptions = get_assumptions()
    rates, seed = get_growth_settings()
    scenarios = read_scenarios(SCENARIOS_FILE)
    if not IMPACT_FILE.is_file():
        raise FileNotFoundError(f"Missing {IMPACT_FILE}; run network scenario simulations first.")
    existing = pd.read_csv(IMPACT_FILE)
    if existing["scenario_id"].duplicated().any():
        raise ValueError("Duplicated scenario IDs in existing impact table.")
    required_impact = {"scenario_id", "additional_cars_served", "additional_energy_delivered_kwh"}
    if required_impact - set(existing):
        raise ValueError("Existing scenario impact file lacks growth screening inputs.")
    impact_by_id = existing.set_index("scenario_id")
    unknown = set(scenarios["scenario_id"]) - set(impact_by_id.index)
    if unknown:
        raise ValueError("Missing historic impact scenarios: " + repr(sorted(unknown)[:3]))

    arrivals, sites = load_data()
    sites_by_id = {str(x["site_id"]): x for _, x in sites.iterrows()}
    days_table = site_observation_periods().set_index("site_id")
    by_site = {str(k): group.copy() for k, group in arrivals.groupby("site_id", sort=False)}
    baseline_by_site = {}
    for site_id, g in scenarios.groupby("site_id", sort=False):
        if site_id not in sites_by_id or site_id not in by_site:
            raise ValueError(f"Missing site metadata or arrivals for {site_id!r}.")
        bases = g[g["scenario_type"] == "CURRENT"]
        if len(bases) != 1:
            raise ValueError(f"Expected exactly one current configuration for {site_id}.")
        baseline_by_site[site_id] = bases.iloc[0]
        for _, case in g.iterrows():
            validate_against_master(case, sites_by_id[site_id])

    years = assumptions["evaluation_years"]
    margin = assumptions["eur_contribution_margin_per_extra_kwh"]
    discount_rate = assumptions["discount_rate"]
    rows = []
    print("\n--- GROWTH-AWARE NETWORK EXPANSION ---", flush=True)
    print(f"Sites: {len(baseline_by_site)} | Scenarios: {len(scenarios)} | Future years: {years}", flush=True)
    print(f"Illustrative annual arrival-growth rates: {rates}%", flush=True)

    for rate in rates:
        print(f"\nDemand growth: {rate:g}% per year", flush=True)
        for site_id, site_cases in scenarios.groupby("site_id", sort=False):
            current = baseline_by_site[site_id]
            site = sites_by_id[site_id]
            base_arrivals = by_site[site_id]
            annualize = 365.25 / int(days_table.loc[site_id, "observation_days"])
            count_cases = len(site_cases)
            print(f"  {site['city']}: {count_cases} configurations", flush=True)
            for year in range(1, years + 1):
                factor = (1 + rate / 100) ** year
                # For 0% growth, reuse the *same existing simulated impacts*,
                # not a new full year of identical calculation.
                if rate == 0:
                    outcomes = None
                else:
                    year_arrivals = make_future_arrivals(base_arrivals, factor, site_id, rate, year, seed)
                    site_for_baseline = site.copy()
                    site_for_baseline["grid_capacity_kw"] = float(current["scenario_grid_kw"])
                    baseline_result = simulate_site(
                        year_arrivals, site_for_baseline, int(current["scenario_chargers"])
                    )
                    outcomes = {}
                    for _, case in site_cases.iterrows():
                        if case["scenario_type"] == "CURRENT":
                            outcomes[case["scenario_id"]] = baseline_result
                        else:
                            expanded_site = site.copy()
                            expanded_site["grid_capacity_kw"] = float(case["scenario_grid_kw"])
                            outcomes[case["scenario_id"]] = simulate_site(
                                year_arrivals, expanded_site, int(case["scenario_chargers"])
                            )

                for _, case in site_cases.iterrows():
                    case_id = case["scenario_id"]
                    if rate == 0:
                        extra_energy = float(impact_by_id.at[case_id, "additional_energy_delivered_kwh"])
                        extra_cars = float(impact_by_id.at[case_id, "additional_cars_served"])
                    else:
                        outcome = outcomes[case_id]
                        extra_energy = outcome["energy_delivered_kwh"] - baseline_result["energy_delivered_kwh"]
                        extra_cars = baseline_result["cars_rejected"] - outcome["cars_rejected"]
                    extra_energy_ann = extra_energy * annualize
                    capex, annual_opex = cost_for(case, current, assumptions)
                    annual_cash = extra_energy_ann * margin - annual_opex
                    rows.append({
                        "site_id": site_id, "city": site["city"],
                        "scenario_id": case_id, "scenario_type": case["scenario_type"],
                        "scenario_chargers": int(case["scenario_chargers"]),
                        "grid_capacity_kw": float(case["scenario_grid_kw"]),
                        "annual_arrival_growth_pct": rate, "future_year": year,
                        "demand_multiplier_vs_history": round(factor, 4),
                        "additional_cars_served_annual_est": round(extra_cars * annualize, 1),
                        "additional_energy_delivered_annual_kwh": round(extra_energy_ann, 1),
                        "annual_energy_margin_eur": round(extra_energy_ann * margin, 2),
                        "annual_extra_operating_cost_eur": round(annual_opex, 2),
                        "incremental_annual_cashflow_eur": round(annual_cash, 2),
                        "discounted_annual_cashflow_eur": round(annual_cash / (1 + discount_rate) ** year, 2),
                        "upfront_investment_eur": round(capex, 2),
                    })

    yearly = pd.DataFrame(rows)
    summary = yearly.groupby([
        "site_id", "city", "scenario_id", "scenario_type", "scenario_chargers",
        "grid_capacity_kw", "annual_arrival_growth_pct"
    ], as_index=False).agg(
        upfront_investment_eur=("upfront_investment_eur", "first"),
        annual_impact_year1_kwh=("additional_energy_delivered_annual_kwh", "first"),
        total_incremental_energy_over_horizon_kwh=("additional_energy_delivered_annual_kwh", "sum"),
        total_incremental_cars_over_horizon=("additional_cars_served_annual_est", "sum"),
        pv_operating_benefit_eur=("discounted_annual_cashflow_eur", "sum"),
    )
    summary["growth_adjusted_npv_eur"] = (
        summary["pv_operating_benefit_eur"] - summary["upfront_investment_eur"]
    ).round(2)
    summary["evaluation_years"] = years
    summary["investment_screen"] = np.where(
        summary["growth_adjusted_npv_eur"] > 0,
        "POSITIVE_NPV_UNDER_ASSUMPTIONS", "NEGATIVE_NPV_UNDER_ASSUMPTIONS"
    )
    summary.loc[summary["scenario_type"] == "CURRENT", "investment_screen"] = "CURRENT"

    priority = []
    for (site_id, rate), g in summary.groupby(["site_id", "annual_arrival_growth_pct"], sort=False):
        options = g[g["scenario_type"] != "CURRENT"].sort_values(
            ["growth_adjusted_npv_eur", "upfront_investment_eur"], ascending=[False, True]
        )
        if options.empty:
            continue
        best = options.iloc[0]
        priority.append({
            "site_id": site_id, "city": best["city"],
            "annual_arrival_growth_pct": rate,
            "action": "EXPANSION_CANDIDATE_FOR_REVIEW" if best["growth_adjusted_npv_eur"] > 0
                     else "NO_EXPANSION_UNDER_ASSUMPTIONS",
            "best_scenario_id": best["scenario_id"],
            "best_chargers": int(best["scenario_chargers"]),
            "best_grid_kw": best["grid_capacity_kw"],
            "best_expansion_npv_eur": round(best["growth_adjusted_npv_eur"], 2),
        })
    ranked = pd.DataFrame(priority).sort_values(
        ["annual_arrival_growth_pct", "best_expansion_npv_eur"], ascending=[True, False]
    )
    for p in [YEARLY_FILE, CASES_FILE, PRIORITY_FILE]:
        p.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(YEARLY_FILE, index=False)
    summary.to_csv(CASES_FILE, index=False)
    ranked.to_csv(PRIORITY_FILE, index=False)
    print("\n--- GROWTH-AWARE INVESTMENT SCREEN ---")
    print(ranked[[
        "city", "annual_arrival_growth_pct", "best_chargers", "best_grid_kw",
        "best_expansion_npv_eur", "action"
    ]].to_string(index=False))
    print(f"\nSaved: {YEARLY_FILE}")
    print(f"Saved: {CASES_FILE}")
    print(f"Saved: {PRIORITY_FILE}")
    print("\nIMPORTANT: Growth rates are user-defined scenarios, NOT demand forecasts.")
    print("Historical six-month demand patterns are repeated and annualized.")
    print("Charging hardware, stay times and future growth assumptions need real-data validation.")


if __name__ == "__main__":
    main()
