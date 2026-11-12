"""CPO expansion: determine whether site grid limits materially affect service.

Uses already shortlisted, registry-configured charger expansions and reproduces
identical synthetic vehicle-arrival demand under two conditions:
  1. The modeled site import limit (with site-wide load management).
  2. A *diagnostic* import ceiling high enough that charger hardware, not the
     site grid connection, sets the charging-power limit.

The second condition is NOT a proposed grid connection, DSO offer, grid approval,
or capital-investment recommendation. It is a controlled simulation comparator.
All historical demand and growth assumptions remain synthetic/unverified.

Required upstream runs:
  simulate_growth_aware_network.py
  build_future_expansion_decision_inputs.py
  build_technical_expansion_shortlist.py
  generate_network_scenarios.py

Windows CMD, from project root:
  python src\\check_grid_upgrade_value.py

Output:
  data/processed/cpo_grid_upgrade_value_screen.csv

No original data or previous outputs are modified.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from simulate_growth_aware_network import (
    REFERENCE_YEAR,
    demand_for_year,
    load_growth_rates,
)
from simulate_registry_network import (
    CHARGING_EFFICIENCY,
    load_arrivals,
    load_scenario_inputs,
    simulate_scenario,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "data" / "config"
RAW = ROOT / "data" / "raw"


REQUIRED_SHORTLIST = {
    "site_id", "city", "planning_year_label",
    "assumed_annual_arrival_growth_pct", "selected_scenario_id_for_screening",
    "additional_chargers_in_screen", "screening_status",
}


def read_shortlist(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing technical shortlist: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    missing = REQUIRED_SHORTLIST - set(df.columns)
    if missing:
        raise ValueError(f"Shortlist is missing: {sorted(missing)}")
    if df.empty:
        raise ValueError("Technical shortlist is empty")
    for col in ["site_id", "city", "screening_status"]:
        if df[col].eq("").any():
            raise ValueError(f"Shortlist has blank {col}")
    if df.duplicated(["site_id", "planning_year_label", "assumed_annual_arrival_growth_pct"]).any():
        raise ValueError("Duplicate site / year / growth assumptions")
    return df


def number(value: object, label: str, *, allow_blank: bool = False) -> float:
    if allow_blank and str(value).strip() == "":
        return 0.0
    try:
        num = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(num):
        raise ValueError(f"Nonfinite {label}")
    return num


def equipment_ac_ceiling_kw(groups: list[dict], other_site_load_kw: float) -> float:
    """Enough theoretical AC import to eliminate *site grid* curtailment.

    Individual unit, simultaneous connector and central DC pool limits continue
    to apply. No particular transformer size or grid upgrade is being proposed.
    """
    max_possible_dc_kw = 0.0
    for g in groups:
        max_unit_dc = min(
            float(g["rated_kw"]),
            int(g["positions"]) * float(g["connector_kw"]),
        )
        max_possible_dc_kw += int(g["count"]) * max_unit_dc
    if max_possible_dc_kw <= 0:
        raise ValueError("Configured units have zero theoretical output")
    return math.ceil((max_possible_dc_kw / CHARGING_EFFICIENCY + other_site_load_kw) * 100) / 100


def find_matching_prior_run(previous: pd.DataFrame, site_id: str,
                            scenario_id: str, year: int, growth: float):
    if previous.empty:
        return None
    p = previous[
        (previous["site_id"] == site_id)
        & (previous["scenario_id"] == scenario_id)
        & (previous["planning_year"] == year)
        & ((previous["assumed_annual_arrival_growth_pct"] - growth).abs() < 1e-7)
    ]
    if len(p) != 1:
        raise ValueError(
            f"Earlier simulation result not uniquely found: {site_id}, {scenario_id}, "
            f"{year}, {growth:g}%. Re-run simulate_growth_aware_network.py."
        )
    return p.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-id", help="Optionally run only one site")
    parser.add_argument("--shortlist", type=Path, default=PROCESSED / "cpo_technical_expansion_shortlist.csv")
    parser.add_argument("--scenarios", type=Path, default=PROCESSED / "registry_network_expansion_scenarios.csv")
    parser.add_argument("--groups", type=Path, default=PROCESSED / "registry_scenario_charger_groups.csv")
    parser.add_argument("--pools", type=Path, default=CONFIG / "cpo_shared_power_pools.csv")
    parser.add_argument("--growth", type=Path, default=CONFIG / "arrival_growth_scenarios.json")
    parser.add_argument("--attempts", type=Path, default=RAW / "demand_attempts.csv")
    parser.add_argument("--sessions", type=Path, default=RAW / "charging_sessions.csv")
    parser.add_argument("--prior-results", type=Path, default=PROCESSED / "future_demand_network_results.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED / "cpo_grid_upgrade_value_screen.csv")
    args = parser.parse_args()

    shortlist = read_shortlist(args.shortlist)
    if args.site_id:
        shortlist = shortlist[shortlist["site_id"] == args.site_id]
        if shortlist.empty:
            raise ValueError(f"Unknown site_id: {args.site_id}")
    scenario_df, configurations, pool_caps = load_scenario_inputs(
        args.scenarios, args.groups, args.pools
    )
    scenario_by_id = {row["scenario_id"]: row for _, row in scenario_df.iterrows()}
    rates, seed = load_growth_rates(args.growth)
    arrivals = load_arrivals(args.attempts, args.sessions)
    arrivals_by_site = {site_id: rows.copy() for site_id, rows in arrivals.groupby("site_id")}

    prior = pd.DataFrame()
    if args.prior_results.is_file():
        prior = pd.read_csv(args.prior_results)
        needed = {"site_id", "scenario_id", "planning_year",
                  "assumed_annual_arrival_growth_pct", "cars_rejected", "energy_delivered_kwh"}
        if needed - set(prior.columns):
            raise ValueError(f"Previous results missing: {sorted(needed - set(prior.columns))}")
        prior["planning_year"] = pd.to_numeric(prior["planning_year"])
        prior["assumed_annual_arrival_growth_pct"] = pd.to_numeric(
            prior["assumed_annual_arrival_growth_pct"]
        )

    rows = []
    cache: dict[tuple[str, int, float], pd.DataFrame] = {}
    for _, row in shortlist.iterrows():
        site = row["site_id"]
        year_val = number(row["planning_year_label"], "planning year")
        if not year_val.is_integer() or year_val <= REFERENCE_YEAR:
            raise ValueError(f"Invalid planning year: {year_val}")
        year = int(year_val)
        rate = number(row["assumed_annual_arrival_growth_pct"], "annual growth")
        if not any(abs(rate - allowed) < 1e-7 for allowed in rates):
            raise ValueError(f"Growth scenario {rate:g}% is not listed in growth config")
        chosen_id = row["selected_scenario_id_for_screening"].strip()
        additions = number(row["additional_chargers_in_screen"], "additional chargers", allow_blank=True)
        if additions < 0 or not additions.is_integer():
            raise ValueError("Expected nonnegative integer additional chargers")

        base = {
            "site_id": site,
            "city": row["city"],
            "planning_year": year,
            "assumed_annual_arrival_growth_pct": rate,
            "screened_added_chargers": int(additions),
            "selected_scenario_id": chosen_id,
            "assumed_charging_efficiency_pct": round(CHARGING_EFFICIENCY * 100, 2),
            "evidence_status": "SYNTHETIC_DEMO_NOT_A_GRID_APPROVAL",
        }
        if not chosen_id:
            if row["screening_status"] == "NO_TESTED_SAME_GRID_OPTION_MEETS_TARGET":
                status = "MORE_CHARGER_CONFIGURATIONS_REQUIRED"
                reason = "No tested same-grid equipment option met the chosen congestion target"
            else:
                raise ValueError(f"{site}: no selected scenario despite status {row['screening_status']}")
            rows.append({**base, "grid_value_screen": status, "reason": reason})
            continue
        if int(additions) == 0:
            rows.append({**base, "grid_value_screen": "NO_EXPANSION_SELECTED", "reason":
                         "Existing chargers meet the selected simulated service target"})
            continue

        scenario = scenario_by_id.get(chosen_id)
        if scenario is None or scenario["site_id"] != site:
            raise ValueError(f"Selected scenario {chosen_id!r} is missing or belongs to another site")
        if scenario["scenario_type"] != "SAME_GRID":
            raise ValueError(f"{chosen_id}: expected same-grid scenario, got {scenario['scenario_type']}")
        groups = configurations[chosen_id]
        grid_kw = number(scenario["scenario_grid_kw"], "grid limit")
        other_kw = number(scenario["other_site_load_kw"], "other site loads", allow_blank=True)
        if grid_kw <= other_kw:
            raise ValueError(f"{site}: other site load >= grid limit")
        diagnostic_kw = max(grid_kw, equipment_ac_ceiling_kw(groups, other_kw))
        multiplier = (1.0 + rate / 100.0) ** (year - REFERENCE_YEAR)
        if not 0 < multiplier <= 5:
            raise ValueError(f"Demand multiplier {multiplier:.2f} exceeds safe replay test range")
        key = (site, year, rate)
        if key not in cache:
            if site not in arrivals_by_site:
                raise ValueError(f"{site}: historical arrival data not found")
            cache[key] = demand_for_year(
                arrivals_by_site[site], multiplier=multiplier,
                site_id=site, year=year, rate=rate, seed=seed
            )
        arrivals_case = cache[key]

        measured = simulate_scenario(arrivals_case, scenario, groups, pool_caps)
        prior_row = find_matching_prior_run(prior, site, chosen_id, year, rate)
        if prior_row is not None:
            if (measured["cars_rejected"] != int(prior_row["cars_rejected"])
                or abs(measured["energy_delivered_kwh"] - float(prior_row["energy_delivered_kwh"])) > 0.11):
                raise ValueError(
                    f"Results changed for {site}/{chosen_id}/{rate:g}% since earlier run. "
                    "Re-run simulate_growth_aware_network.py and update the shortlist."
                )

        if diagnostic_kw <= grid_kw + 0.005:
            comparison = measured
        else:
            diagnostic_scenario = scenario.copy()
            diagnostic_scenario["scenario_grid_kw"] = str(diagnostic_kw)
            comparison = simulate_scenario(arrivals_case, diagnostic_scenario, groups, pool_caps)

        extra_energy = round(comparison["energy_delivered_kwh"] - measured["energy_delivered_kwh"], 1)
        extra_cars = comparison["cars_served"] - measured["cars_served"]
        # A higher ceiling can only be interpreted as a theoretical maximum
        # for THIS simulation, not a business case or DSO-approved upgrade.
        if diagnostic_kw <= grid_kw + 0.005:
            status = "GRID_NOT_THE_INSTALLED_EQUIPMENT_LIMIT"
            reason = "Modeled grid already allows full configured output"
        elif extra_energy > 0.11 or extra_cars > 0:
            status = "GRID_MAY_AFFECT_SERVICE_REVIEW_REQUIRED"
            reason = "Removing the modeled site limit improves simulated service"
        else:
            status = "NO_MEASURABLE_GRID_UPGRADE_BENEFIT_IN_TEST"
            reason = "More site power did not improve this simulated demand case"

        rows.append({
            **base,
            "site_grid_limit_kw": round(grid_kw, 2),
            "diagnostic_no_grid_bottleneck_ceiling_kw": round(diagnostic_kw, 2),
            "arrivals_tested": int(measured["arrivals"]),
            "rejected_with_site_limit": int(measured["cars_rejected"]),
            "rejected_without_site_limit": int(comparison["cars_rejected"]),
            "extra_cars_served_if_grid_not_limiting": int(extra_cars),
            "energy_delivered_with_site_limit_kwh": measured["energy_delivered_kwh"],
            "energy_delivered_without_site_limit_kwh": comparison["energy_delivered_kwh"],
            "extra_energy_if_grid_not_limiting_kwh": extra_energy,
            "partial_charging_shortfall_with_site_limit_kwh": measured["energy_shortfall_served_kwh"],
            "partial_charging_shortfall_without_site_limit_kwh": comparison["energy_shortfall_served_kwh"],
            "grid_limit_minutes": int(measured["minutes_at_grid_limit"]),
            "fully_charged_pct_with_site_limit": measured["cars_fully_charged_pct"],
            "fully_charged_pct_without_site_limit": comparison["cars_fully_charged_pct"],
            "grid_value_screen": status,
            "reason": reason,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No scenarios to evaluate")
    result = result.sort_values(["assumed_annual_arrival_growth_pct", "site_id"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("\n--- LOAD MANAGEMENT / GRID VALUE SCREEN ---")
    print(f"Sites: {result.site_id.nunique()} | Demand cases: {len(result)}")
    print(result[["city", "assumed_annual_arrival_growth_pct", "screened_added_chargers",
                  "grid_limit_minutes", "extra_cars_served_if_grid_not_limiting",
                  "extra_energy_if_grid_not_limiting_kwh", "grid_value_screen"]].to_string(index=False))
    print("\nInterpretation: extra benefit = difference between SAME equipment at current vs diagnostic grid limit.")
    print("The diagnostic limit is calculated from hardware ratings; it is NOT a proposed grid upgrade.")
    print("Synthetic demand, simplified car stays, modeled 95% efficiency and unknown AC loads remain unverified.")
    print("No DSO approval, customer SLA, tariff, cost or investment decision is inferred.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
