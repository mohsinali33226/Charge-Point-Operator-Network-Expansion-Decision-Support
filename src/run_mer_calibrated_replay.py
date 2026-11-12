"""Controlled Mer CCS behaviour replay through the EXISTING CPO charging simulator.

Run from CPO Network Development project root:
    python src\\run_mer_calibrated_replay.py --site-id DE_NBG_001
    python src\\run_mer_calibrated_replay.py

This does not overwrite synthetic raw data, forecasts, scenario configurations,
financial outputs, or the existing simulation. It substitutes paired historical
observed delivered energy, duration and average-power PROXIES for demand inputs,
keeping the original synthetic arrival TIMES and COUNT unchanged.

IMPORTANT DATA LIMITATIONS
--------------------------
- Mer 2020-2024 is an old, selected operator sample of SUCCESSFULLY served CCS
  sessions, NOT a representative 2026 German DC fleet or all visitor arrivals.
- An observed delivered kWh value is used as a hypothetical energy TARGET,
  NOT mistaken for a measured driver's requested kWh.
- Observed mean kW is used as an effective constant-rate PROXY in the existing
  simulator, NOT mistaken for the car's peak or charger connector rating.
- The 2025 synthetic arrival count/time pattern is unchanged and NOT validated.
- The existing simulator has simplified stay limits, efficiency and no queues,
  availability/failures or battery charging curves.
- Results are *sensitivity / model-correction diagnostics*, not confirmed future
  demand, congestion forecasts, revenue or expansion investments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "data" / "config"

MER_NEEDED = {
    "observed_delivered_energy_kwh",
    "observed_session_duration_min",
    "observed_average_power_kw",
    "start_hour",
    "start_weekday",
}
ATTEMPT_NEEDED = {"arrival_id", "site_id", "start_time", "requested_energy_kwh"}
MIN_BUCKET_SIZE = 25


def stable_seed(base: int, site: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{site}".encode()).digest()[:8], "big")


def check_inputs(args):
    for label, path in {
        "original arrivals": args.attempts,
        "existing synthetic sessions": args.sessions,
        "historical Mer CCS pool": args.mer_pool,
        "registry scenarios": args.scenarios,
        "registry charger groups": args.scenario_groups,
        "registered power pools": args.pools,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    for target in (args.output_dir, args.reference_results):
        if args.output_dir.resolve() == args.attempts.parent.resolve():
            raise ValueError("Output folder cannot be data/raw: use a separate processed output folder")


def read_mer_pool(path: Path, location: str | None):
    mer = pd.read_csv(path, low_memory=False)
    mer.columns = mer.columns.str.strip()
    missing = MER_NEEDED - set(mer.columns)
    if missing:
        raise ValueError(f"Mer CCS pool missing columns: {sorted(missing)}. Run build_mer_ccs_calibration.py")
    if location is not None:
        if "source_location_category" not in mer:
            raise ValueError("Location filter supplied but Mer pool has no source_location_category")
        mer = mer[mer["source_location_category"].astype(str).str.casefold() == location.casefold()].copy()
        if mer.empty:
            raise ValueError(f"No Mer sessions for category {location!r}")
    for name in ("observed_delivered_energy_kwh", "observed_session_duration_min", "observed_average_power_kw"):
        mer[name] = pd.to_numeric(mer[name], errors="coerce")
    mer["start_hour"] = pd.to_numeric(mer["start_hour"], errors="coerce")
    mer["start_weekday"] = pd.to_numeric(mer["start_weekday"], errors="coerce")
    valid = (
        mer[list(MER_NEEDED)].notna().all(axis=1)
        & mer["observed_delivered_energy_kwh"].gt(0)
        & mer["observed_session_duration_min"].gt(0)
        & mer["observed_average_power_kw"].gt(0)
        & mer["start_hour"].between(0, 23)
        & mer["start_weekday"].between(0, 6)
    )
    mer = mer[valid].copy().reset_index(drop=True)
    if len(mer) < MIN_BUCKET_SIZE:
        raise ValueError(f"Not enough valid historical CCS sessions: {len(mer)}")
    return mer


def empirical_profiles(attempts: pd.DataFrame, mer: pd.DataFrame, seed: int):
    """Assign complete Mer sessions by (weekday, hour), falling back to hour or all.

    The observed trio stays linked. Always sample WITH replacement so the sample
    size is driven by the synthetic arrivals, not by Mer source session counts.
    """
    complete_rows = []
    value_cols = ["observed_delivered_energy_kwh", "observed_session_duration_min", "observed_average_power_kw"]
    for site_id, site_att in attempts.groupby("site_id", sort=True):
        site_att = site_att.copy().reset_index(drop=True)
        ts = pd.to_datetime(site_att["start_time"], errors="raise")
        rng = np.random.default_rng(stable_seed(seed, site_id))
        chosen = np.empty(len(site_att), dtype=np.int64)
        choice_type = np.empty(len(site_att), dtype=object)
        target_h = ts.dt.hour.to_numpy()
        target_w = ts.dt.dayofweek.to_numpy()
        hours = mer["start_hour"].to_numpy(dtype=int)
        weekdays = mer["start_weekday"].to_numpy(dtype=int)
        for (w, h) in sorted(set(zip(target_w, target_h))):
            positions = np.flatnonzero((target_w == w) & (target_h == h))
            pool_idx = np.flatnonzero((weekdays == w) & (hours == h))
            label = "weekday_hour"
            if len(pool_idx) < MIN_BUCKET_SIZE:
                pool_idx = np.flatnonzero(hours == h)
                label = "hour_only"
            if len(pool_idx) < MIN_BUCKET_SIZE:
                pool_idx = np.arange(len(mer))
                label = "unstratified"
            chosen[positions] = rng.choice(pool_idx, size=len(positions), replace=True)
            choice_type[positions] = label
        sampled = mer.iloc[chosen][value_cols].reset_index(drop=True)
        block = pd.DataFrame({
            "arrival_id": site_att["arrival_id"].astype(str),
            "site_id": site_id,
            "start_time": ts.astype(str),
            "historical_source_row": mer.iloc[chosen]["source_row_number"].to_numpy() if "source_row_number" in mer else chosen,
            "matching_rule": choice_type,
            "requested_energy_kwh": sampled["observed_delivered_energy_kwh"].to_numpy(),
            "requested_power_kw": sampled["observed_average_power_kw"].to_numpy(),
            "duration_minutes": sampled["observed_session_duration_min"].to_numpy(),
        })
        complete_rows.append(block)
    result = pd.concat(complete_rows, ignore_index=True)
    if result["arrival_id"].duplicated().any() or len(result) != len(attempts):
        raise AssertionError("Incorrect mapping or duplicate synthetic arrival IDs")
    return result


def quantile(s):
    return round(float(pd.to_numeric(s).median()), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-id", help="One-site run before full-network run")
    parser.add_argument("--location", help="Optional historical Mer location category (e.g. Highway), exact label")
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--attempts", type=Path, default=RAW / "demand_attempts.csv")
    parser.add_argument("--sessions", type=Path, default=RAW / "charging_sessions.csv")
    parser.add_argument("--mer-pool", type=Path, default=PROCESSED / "mer_ccs_empirical_behaviour_pool.csv")
    parser.add_argument("--scenarios", type=Path, default=PROCESSED / "registry_network_expansion_scenarios.csv")
    parser.add_argument("--scenario-groups", type=Path, default=PROCESSED / "registry_scenario_charger_groups.csv")
    parser.add_argument("--pools", type=Path, default=CONFIG / "cpo_shared_power_pools.csv")
    parser.add_argument("--reference-results", type=Path, default=PROCESSED / "registry_network_simulation_results.csv")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED / "mer_calibrated_replay")
    args = parser.parse_args()
    check_inputs(args)
    mer = read_mer_pool(args.mer_pool, args.location)
    attempts = pd.read_csv(args.attempts)
    if missing := ATTEMPT_NEEDED - set(attempts.columns):
        raise ValueError(f"Synthetic arrivals missing columns: {sorted(missing)}")
    attempts["start_time"] = pd.to_datetime(attempts["start_time"], errors="raise")
    if attempts["arrival_id"].duplicated().any():
        raise ValueError("Synthetic arrival IDs are not unique")
    scenarios, configs, pool_caps = load_scenario_inputs(args.scenarios, args.scenario_groups, args.pools)
    if args.site_id:
        scenarios = scenarios[scenarios["site_id"] == args.site_id].copy()
        if scenarios.empty:
            raise ValueError(f"Site {args.site_id!r} not found in registry scenarios")
        attempts = attempts[attempts["site_id"] == args.site_id].copy()
    if attempts.empty:
        raise ValueError("No arrivals found for selected sites")
    if attempts["site_id"].isna().any():
        raise ValueError("Arrival site ID missing")
    required_sites = set(scenarios["site_id"])
    if set(attempts["site_id"]) != required_sites:
        raise ValueError("Sites in the synthetic arrival records and registry scenarios differ")

    new_inputs = empirical_profiles(attempts, mer, args.seed)
    print("\n--- MER CCS / CPO REPLAY ---")
    print(f"Historical CCS source sessions: {len(mer):,}")
    print(f"Synthetic arrival timestamps preserved: {len(new_inputs):,}")
    print(f"Sites: {len(required_sites)} | Infrastructure scenarios: {len(scenarios)}")
    print(f"Mer selection: {args.location or 'ALL historical CCS location categories'}")
    print("NOTE: observed delivered kWh and mean kW are DEMONSTRATION TARGET / RATE PROXIES, NOT customer requests or vehicle maximum charging power.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed_attempts = args.output_dir / "mer_proxy_demand_attempts.csv"
    processed_sessions = args.output_dir / "mer_proxy_session_inputs.csv"
    new_inputs[["arrival_id", "site_id", "start_time", "requested_energy_kwh"]].to_csv(processed_attempts, index=False)
    new_inputs[["arrival_id", "requested_power_kw", "duration_minutes"]].to_csv(processed_sessions, index=False)
    new_arrivals = load_arrivals(processed_attempts, processed_sessions)

    simulated = []
    for idx, (_, scenario) in enumerate(scenarios.iterrows(), 1):
        s = scenario["site_id"]
        subset = new_arrivals[new_arrivals["site_id"] == s]
        print(f"[{idx}/{len(scenarios)}] {scenario['city']} — {scenario['scenario_chargers']} chargers", flush=True)
        simulated.append(simulate_scenario(subset, scenario, configs[scenario["scenario_id"]], pool_caps))
    simulations = pd.DataFrame(simulated)
    results_path = args.output_dir / "mer_proxy_network_simulation_results.csv"
    simulations.to_csv(results_path, index=False)

    impact = []
    for site_id, grp in simulations.groupby("site_id"):
        base = grp[grp["scenario_type"] == "CURRENT"]
        if len(base) != 1:
            raise ValueError(f"{site_id}: expected one CURRENT scenario")
        for _, item in grp.iterrows():
            impact.append({
                "site_id": site_id,
                "city": item["city"],
                "scenario_id": item["scenario_id"],
                "scenario_chargers": int(item["scenario_chargers"]),
                "scenario_type": item["scenario_type"],
                "cars_rejected_mer_proxy": int(item["cars_rejected"]),
                "rejected_pct_mer_proxy": float(item["rejected_pct"]),
                "energy_delivered_mer_proxy_kwh": float(item["energy_delivered_kwh"]),
                "additional_cars_served_mer_proxy": int(item["cars_served"] - base.iloc[0]["cars_served"]),
                "additional_energy_delivered_mer_proxy_kwh": round(float(item["energy_delivered_kwh"] - base.iloc[0]["energy_delivered_kwh"]), 1),
            })
    impact = pd.DataFrame(impact)
    impact_path = args.output_dir / "mer_proxy_expansion_impact.csv"
    impact.to_csv(impact_path, index=False)

    reference = None
    if args.reference_results.is_file():
        candidate = pd.read_csv(args.reference_results)
        needed = {"scenario_id", "site_id", "arrivals", "scenario_chargers", "cars_rejected", "energy_delivered_kwh"}
        if not needed - set(candidate.columns) and not candidate["scenario_id"].duplicated().any():
            selected = candidate[candidate["scenario_id"].isin(simulations["scenario_id"])].copy()
            if (len(selected) == len(simulations)
                and selected.set_index("scenario_id")["site_id"].to_dict() == simulations.set_index("scenario_id")["site_id"].to_dict()
                and selected.set_index("scenario_id")["arrivals"].astype(int).to_dict() == simulations.set_index("scenario_id")["arrivals"].astype(int).to_dict()
                and selected.set_index("scenario_id")["scenario_chargers"].astype(int).to_dict() == simulations.set_index("scenario_id")["scenario_chargers"].astype(int).to_dict()):
                reference = selected
    comparison_cols = ["scenario_id", "site_id", "city", "scenario_type", "scenario_chargers", "arrivals", "cars_rejected", "rejected_pct", "energy_delivered_kwh"]
    comparison = simulations[comparison_cols].rename(columns={
        "cars_rejected": "mer_proxy_cars_rejected",
        "rejected_pct": "mer_proxy_rejected_pct",
        "energy_delivered_kwh": "mer_proxy_delivered_kwh",
    })
    if reference is not None:
        other = reference[["scenario_id", "cars_rejected", "rejected_pct", "energy_delivered_kwh"]].rename(columns={
            "cars_rejected": "old_synthetic_cars_rejected",
            "rejected_pct": "old_synthetic_rejected_pct",
            "energy_delivered_kwh": "old_synthetic_delivered_kwh",
        })
        comparison = comparison.merge(other, on="scenario_id", validate="one_to_one")
        comparison["change_in_rejected_cars"] = comparison["mer_proxy_cars_rejected"] - comparison["old_synthetic_cars_rejected"]
    comparison_path = args.output_dir / "mer_proxy_vs_previous_simulation.csv"
    comparison.to_csv(comparison_path, index=False)

    manifest = {
        "status": "HISTORICAL_BEHAVIOUR_PROXY_SENSITIVITY_ONLY",
        "mer_pool_file": str(args.mer_pool),
        "mer_source_sessions": len(mer),
        "selected_location_category": args.location or "ALL",
        "synthetic_arrival_count_and_timing": "UNCHANGED",
        "synthetic_arrivals_replayed": len(new_inputs),
        "random_seed": args.seed,
        "mer_sampling": "weekday_hour_with_hour_and_all_fallback; paired energy_duration_average_power",
        "mean_power_used_as": "effective_constant_power_proxy_NOT_vehicle_maximum",
        "delivered_energy_used_as": "hypothetical_target_NOT_observed_request",
        "reference_comparison": "AVAILABLE" if reference is not None else "NOT_COMPARABLE_OR_MISSING",
        "limitations": [
            "Mer historical served CCS sessions only; no rejected arrival observations.",
            "Mer 2020-2024 not a validated 2026 German equipment- or location-specific charging profile.",
            "Same previously generated synthetic arrival rate/time pattern used in replay.",
            "Historical mean kW is not actual EV peak charge capability or DC charging curve.",
            "Scenario calculation still uses demo stay cap, site efficiency, no queue, no outages, no SOC taper.",
            "Do not use these outputs as verified congestion, future forecast, or investment advice.",
        ],
    }
    (args.output_dir / "mer_proxy_replay_metadata.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n--- REAL SESSION BEHAVIOUR REPLAY ---")
    print(f"Sampled profiles: energy P50={quantile(new_inputs['requested_energy_kwh'])} kWh; "
          f"duration P50={quantile(new_inputs['duration_minutes'])} min; "
          f"rate proxy P50={quantile(new_inputs['requested_power_kw'])} kW")
    print(comparison[["city", "scenario_chargers", "mer_proxy_rejected_pct"] +
                     (["old_synthetic_rejected_pct"] if reference is not None else [])].to_string(index=False))
    print("\nSaved in:", args.output_dir)
    print("WARNING: Historical Mer average power is a proxy, not a validated EV max-power model. No source files modified.")


if __name__ == "__main__":
    main()
