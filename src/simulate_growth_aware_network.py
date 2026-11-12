"""CPO network: compare infrastructure under conditional future demand (DEMO).

This connects a site-level arrival growth *scenario* with the existing
occupancy-aware, group-aware charger simulator. These are NOT independently
validated forecasts for future years.

Inputs (already in this project):
    data/config/arrival_growth_scenarios.json
    data/processed/registry_network_expansion_scenarios.csv
    data/processed/registry_scenario_charger_groups.csv
    data/config/cpo_shared_power_pools.csv
    data/raw/demand_attempts.csv
    data/raw/charging_sessions.csv

Outputs (new files):
    data/processed/future_demand_network_results.csv
    data/processed/future_demand_network_impact.csv

Usage:
    python src\\simulate_growth_aware_network.py
    python src\\simulate_growth_aware_network.py --year 2028
    python src\\simulate_growth_aware_network.py --site-id DE_NBG_001

Method: Treat historic 2025 arrival times/vehicle profiles as a repeatable
reference demand pattern, and sample additional/reduced arrivals for each
specified year and annual change rate. Identical arrivals are then fed to
current and expanded infrastructure. At each minute, the simulator accounts
for cars still plugged in, new arrivals, departures, connector limits, per-
charger limits, shared DC pools and site-wide grid limits.

CAUTIONS:
* Changing a 2025 synthetic sample is NOT a true prediction for 2027/2028.
* 0/15 percent from the older JSON file are user-adjustable demo scenarios,
  NOT sourced German EV charging market growth rates.
* No actual queues, real battery charging curves, station failures, EV mix
  changes, site-specific growth evidence, equipment quotes or investment
  approvals are modeled.
* Never use this script's results as a real investment recommendation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_registry_network import (
    load_arrivals,
    load_scenario_inputs,
    simulate_scenario,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "data" / "config"
PROCESSED = ROOT / "data" / "processed"
REFERENCE_YEAR = 2025


def load_growth_rates(path: Path) -> tuple[list[float], int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing demand scenario file: {path}\n"
            "We will not invent a growth forecast. Supply an existing "
            "arrival_growth_scenarios.json with annual_arrival_growth_rates_pct "
            '(for example [0] for an unchanged-demand demonstration).'
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    rates = config.get("annual_arrival_growth_rates_pct")
    if not isinstance(rates, list) or not rates:
        raise ValueError("annual_arrival_growth_rates_pct must be a nonempty list")
    validated = []
    for rate in rates:
        if isinstance(rate, bool) or not isinstance(rate, (float, int)):
            raise ValueError(f"Growth rate must be numeric: {rate!r}")
        if not math.isfinite(rate) or rate <= -100:
            raise ValueError(f"Annual growth must be finite and above -100%: {rate}")
        validated.append(float(rate))
    seed = config.get("random_seed", 2026)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    return sorted(set(validated)), seed


def demand_for_year(source: pd.DataFrame, *, multiplier: float,
                    site_id: str, year: int, rate: float, seed: int) -> pd.DataFrame:
    """Replay same historic time pattern at an assumed demand multiplier.

    The replay remains on the original historical calendar. We use `year`
    only to calculate the multiplier, NOT as a fabricated timestamp forecast.
    """
    if source.empty or multiplier <= 0 or not math.isfinite(multiplier):
        raise ValueError("Need nonempty arrivals and positive finite multiplier")

    original = source.copy().reset_index(drop=True)
    original["start_time"] = pd.to_datetime(original["start_time"])
    n_base = len(original)
    n_target = max(1, round(n_base * multiplier))
    key = f"{seed}|{site_id}|{year}|{rate:.8f}"
    stable_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(stable_seed)

    if n_target == n_base:
        adjusted = original
    elif n_target < n_base:
        # Keep a representative sub-sample rather than changing session power.
        keep = np.sort(rng.choice(n_base, size=n_target, replace=False))
        adjusted = original.iloc[keep].copy()
    else:
        extras_count = n_target - n_base
        picked = rng.integers(0, n_base, size=extras_count)
        extras = original.iloc[picked].copy()
        # Slight randomized arrival-time shifts preserve approximate hourly
        # and daily patterns without placing every clone at the exact minute.
        shift_mins = rng.integers(-30, 31, size=extras_count)
        extras["start_time"] = (extras["start_time"] +
                                pd.to_timedelta(shift_mins, unit="m"))
        extras["start_time"] = extras["start_time"].clip(
            lower=original["start_time"].min(),
            upper=original["start_time"].max(),
        )
        adjusted = pd.concat([original, extras], ignore_index=True)

    adjusted = adjusted.sort_values("start_time", kind="stable").reset_index(drop=True)
    if len(adjusted) != n_target:
        raise AssertionError("Incorrect number of future-scenario arrivals")
    return adjusted


def run(args: argparse.Namespace) -> None:
    rates, seed = load_growth_rates(args.growth_config)
    if args.year <= REFERENCE_YEAR:
        raise ValueError(f"Choose a planning year after {REFERENCE_YEAR}")

    scenarios, configs, pools = load_scenario_inputs(
        args.scenarios, args.scenario_groups, args.pools
    )
    if args.site_id:
        scenarios = scenarios[scenarios["site_id"] == args.site_id].copy()
        if scenarios.empty:
            raise ValueError(f"Site {args.site_id!r} is absent from the scenario file")

    arrivals = load_arrivals(args.attempts, args.sessions)
    arrivals_by_site = {
        str(site_id): part.copy()
        for site_id, part in arrivals.groupby("site_id", sort=False)
    }

    for site_id, site_cases in scenarios.groupby("site_id"):
        current = site_cases[site_cases["scenario_type"] == "CURRENT"]
        if len(current) != 1:
            raise ValueError(f"{site_id}: expected exactly one CURRENT scenario")
        if site_id not in arrivals_by_site:
            raise ValueError(f"{site_id}: missing vehicle arrivals")

    total_runs = len(scenarios) * len(rates)
    print("\n--- FUTURE DEMAND / OCCUPANCY-AWARE EXPANSION TEST ---", flush=True)
    print(f"Sites: {scenarios.site_id.nunique()} | Hardware configurations: {len(scenarios)}", flush=True)
    print(f"Historical demand pattern: {REFERENCE_YEAR} | Planning year: {args.year}", flush=True)
    print(f"Assumed annual arrival changes: {rates}%", flush=True)
    print("These are scenario assumptions, NOT verified future forecasts.", flush=True)

    results = []
    impacts = []
    done = 0
    for rate in rates:
        multiplier = (1.0 + rate / 100.0) ** (args.year - REFERENCE_YEAR)
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Demand multiplier is invalid")
        # Avoid accidentally running massive scenarios on a laptop.
        if multiplier > args.max_multiplier:
            raise ValueError(
                f"{rate:g}% to {args.year} scales demand by {multiplier:.1f}x, "
                f"above the --max-multiplier safety threshold {args.max_multiplier}. "
                "Reduce the horizon/rates or consciously adjust that limit."
            )
        print(f"\nAssumption {rate:g}%/year -> demand multiplier {multiplier:.3f}x", flush=True)
        for site_id, site_cases in scenarios.groupby("site_id", sort=True):
            site_arrivals = demand_for_year(
                arrivals_by_site[site_id], multiplier=multiplier,
                site_id=site_id, year=args.year, rate=rate, seed=seed,
            )
            site_outcomes = []
            for _, scenario in site_cases.iterrows():
                done += 1
                sim = simulate_scenario(
                    site_arrivals, scenario,
                    configs[scenario["scenario_id"]], pools,
                )
                sim.update({
                    "planning_year": args.year,
                    "reference_year": REFERENCE_YEAR,
                    "assumed_annual_arrival_growth_pct": rate,
                    "assumed_arrival_multiplier": round(multiplier, 5),
                    "demand_basis": "REPLAYED_2025_SYNTHETIC_ARRIVALS",
                    "outlook_evidence_status": "UNVERIFIED_SCENARIO_NOT_FORECAST",
                    "investment_decision": "NOT_ASSESSED",
                })
                results.append(sim)
                site_outcomes.append(sim)
                print(
                    f"[{done}/{total_runs}] {sim['city']} | {sim['scenario_chargers']} chargers "
                    f"| {sim['arrivals']:,} arrivals | {sim['cars_rejected']:,} rejected",
                    flush=True,
                )

            baseline = [r for r in site_outcomes if r["scenario_type"] == "CURRENT"]
            if len(baseline) != 1:
                raise AssertionError("Baseline missing after simulation")
            baseline = baseline[0]
            for r in site_outcomes:
                impacts.append({
                    "site_id": site_id,
                    "city": r["city"],
                    "planning_year": args.year,
                    "assumed_annual_arrival_growth_pct": rate,
                    "assumed_arrival_multiplier": round(multiplier, 5),
                    "scenario_id": r["scenario_id"],
                    "scenario_type": r["scenario_type"],
                    "scenario_chargers": r["scenario_chargers"],
                    "scenario_connectors": r["scenario_connectors"],
                    "scenario_arrivals": r["arrivals"],
                    "cars_rejected": r["cars_rejected"],
                    "additional_cars_served_vs_current": (
                        r["cars_served"] - baseline["cars_served"]
                    ),
                    "additional_energy_delivered_kwh_vs_current": round(
                        r["energy_delivered_kwh"] - baseline["energy_delivered_kwh"], 1
                    ),
                    "change_in_unserved_energy_kwh_vs_current": round(
                        r["total_unserved_energy_kwh"] - baseline["total_unserved_energy_kwh"], 1
                    ),
                    "minutes_at_grid_limit": r["minutes_at_grid_limit"],
                    "demand_basis": r["demand_basis"],
                    "outlook_evidence_status": r["outlook_evidence_status"],
                    "investment_decision": "NOT_ASSESSED",
                })

    summary = pd.DataFrame(impacts).sort_values(
        ["assumed_annual_arrival_growth_pct", "site_id", "scenario_chargers", "scenario_id"]
    )
    details = pd.DataFrame(results).sort_values(
        ["assumed_annual_arrival_growth_pct", "site_id", "scenario_chargers", "scenario_id"]
    )
    if args.output.resolve() == args.impact_output.resolve():
        raise ValueError("Output filenames must differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.impact_output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output, index=False)
    summary.to_csv(args.impact_output, index=False)

    print("\n--- DEMAND-GROWTH / EXPANSION IMPACT ---")
    print(summary[[
        "city", "assumed_annual_arrival_growth_pct", "scenario_chargers",
        "cars_rejected", "additional_cars_served_vs_current",
        "additional_energy_delivered_kwh_vs_current"
    ]].to_string(index=False))
    print("\nNOTE: These are conditional stress tests using synthetic arrival patterns.")
    print("They are NOT actual 2027 demand forecasts or investment advice.")
    print(f"Saved: {args.output}")
    print(f"Saved: {args.impact_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2027,
                        help="Future scenario planning year (default: 2027)")
    parser.add_argument("--site-id", type=str,
                        help="Optional single-site debugging run")
    parser.add_argument("--growth-config", type=Path,
                        default=CONFIG / "arrival_growth_scenarios.json")
    parser.add_argument("--scenarios", type=Path,
                        default=PROCESSED / "registry_network_expansion_scenarios.csv")
    parser.add_argument("--scenario-groups", type=Path,
                        default=PROCESSED / "registry_scenario_charger_groups.csv")
    parser.add_argument("--pools", type=Path,
                        default=CONFIG / "cpo_shared_power_pools.csv")
    parser.add_argument("--attempts", type=Path,
                        default=RAW / "demand_attempts.csv")
    parser.add_argument("--sessions", type=Path,
                        default=RAW / "charging_sessions.csv")
    parser.add_argument("--output", type=Path,
                        default=PROCESSED / "future_demand_network_results.csv")
    parser.add_argument("--impact-output", type=Path,
                        default=PROCESSED / "future_demand_network_impact.csv")
    parser.add_argument("--max-multiplier", type=float, default=5.0,
                        help="Safety threshold for huge replay datasets, not a CPO network limit")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
