"""Compare charger counts using the same arrivals and site-wide power sharing.

Illustrative planning simulation, not a validated financial/operational model.
"""

from pathlib import Path
from math import ceil
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "expansion_simulation_summary.csv"

SCENARIO_CHARGERS = (2, 4, 6)
CHARGING_EFFICIENCY = 0.95  # AC grid input -> DC energy at the car
MAX_STAY_MULTIPLIER = 1.5   # Assumption: drivers may extend their usual stay by 50%
MAX_STAY_MINUTES = 120      # Assumption: drivers leave after at most 2 hours
RANDOM_SEED = 42
EPS = 1e-8
ONE_MINUTE = pd.Timedelta(minutes=1)


def require_columns(frame, expected, file_name):
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"{file_name} is missing columns: {missing}")


def fair_allocation(caps, budget):
    """Share a power budget equally while giving unused shares to others."""
    if not caps:
        return []
    caps = [max(0.0, float(p)) for p in caps]
    if sum(caps) <= budget + EPS:
        return caps
    if budget <= 0:
        return [0.0] * len(caps)

    allocated = [0.0] * len(caps)
    remaining = set(range(len(caps)))
    available = float(budget)
    while remaining:
        fair_share = available / len(remaining)
        limited = [i for i in remaining if caps[i] <= fair_share + EPS]
        if limited:
            for i in limited:
                allocated[i] = caps[i]
                available -= caps[i]
                remaining.remove(i)
        else:
            for i in remaining:
                allocated[i] = fair_share
            break
    return allocated


def load_data():
    arrivals = pd.read_csv(RAW / "demand_attempts.csv")
    sessions = pd.read_csv(RAW / "charging_sessions.csv")
    sites = pd.read_csv(RAW / "site_master.csv")
    require_columns(arrivals, ["arrival_id", "site_id", "start_time", "requested_energy_kwh"], "demand_attempts.csv")
    require_columns(sessions, ["arrival_id", "requested_power_kw", "duration_minutes"], "charging_sessions.csv")
    require_columns(sites, ["site_id", "city", "charger_count", "charger_power_kw", "max_connector_power_kw", "grid_capacity_kw"], "site_master.csv")

    # Some versions of the site master store the total number of connectors
    # but do not include a separate connectors_per_charger column.
    # Reconstruct it when possible; otherwise use this project's agreed
    # hardware assumption: two simultaneous connectors per physical charger.
    if "connectors_per_charger" not in sites.columns:
        if "connector_count" in sites.columns:
            chargers = pd.to_numeric(sites["charger_count"], errors="coerce")
            connectors = pd.to_numeric(sites["connector_count"], errors="coerce")
            if chargers.isna().any() or connectors.isna().any() or (chargers <= 0).any():
                raise ValueError("Invalid charger_count or connector_count in site_master.csv.")
            ratios = connectors / chargers
            if ((ratios < 1) | (ratios != ratios.round())).any():
                raise ValueError("connector_count must be a positive whole multiple of charger_count.")
            sites["connectors_per_charger"] = ratios.astype(int)
            print("INFO: Calculated connectors_per_charger from connector_count / charger_count.")
        else:
            sites["connectors_per_charger"] = 2
            print("INFO: No connector-count column found; assuming 2 connectors per charger.")
    else:
        sites["connectors_per_charger"] = pd.to_numeric(
            sites["connectors_per_charger"], errors="raise"
        )

    if (sites["connectors_per_charger"] != 2).any():
        raise ValueError(
            "This project assumes exactly 2 simultaneous connectors per charger; "
            "please verify site_master.csv."
        )

    if arrivals["arrival_id"].duplicated().any() or sessions["arrival_id"].duplicated().any():
        raise ValueError("arrival_id must be unique in both input files.")
    if sites["site_id"].duplicated().any():
        raise ValueError("site_id must be unique in site_master.csv.")

    arrivals = arrivals.merge(
        sessions[["arrival_id", "requested_power_kw", "duration_minutes"]],
        on="arrival_id", how="left", validate="one_to_one"
    )
    arrivals["start_time"] = pd.to_datetime(arrivals["start_time"], errors="raise")
    arrivals["start_time"] = arrivals["start_time"].dt.floor("min")
    arrivals = arrivals.sort_values(["site_id", "start_time", "arrival_id"])

    if arrivals[["site_id", "start_time", "requested_energy_kwh"]].isna().any().any():
        raise ValueError("An arrival is missing its site, time or requested energy.")
    if (arrivals["requested_energy_kwh"] <= 0).any():
        raise ValueError("Each arrival must request a positive energy amount.")

    # Rejected vehicles have no recorded charging power. Give them a
    # reproducible power value sampled from accepted vehicles at that site.
    rng = np.random.default_rng(RANDOM_SEED)
    for site_id, group in arrivals.groupby("site_id", sort=True):
        observed = group["requested_power_kw"].dropna()
        observed = observed[observed > 0]
        if observed.empty:
            raise ValueError(f"No valid observed charging powers for {site_id}.")
        missing_idx = group.index[group["requested_power_kw"].isna()]
        if len(missing_idx):
            arrivals.loc[missing_idx, "requested_power_kw"] = rng.choice(
                observed.to_numpy(), size=len(missing_idx), replace=True
            )

    if (arrivals["requested_power_kw"] <= 0).any():
        raise ValueError("Requested charging power must be positive.")

    # Observed stays inform the normal planned stay. For originally refused
    # drivers, estimate that stay from energy / sampled requested power.
    missing_duration = arrivals["duration_minutes"].isna()
    estimated = (
        arrivals.loc[missing_duration, "requested_energy_kwh"] * 60
        / arrivals.loc[missing_duration, "requested_power_kw"]
    )
    arrivals.loc[missing_duration, "duration_minutes"] = estimated.clip(upper=60)
    arrivals["max_stay_min"] = np.minimum(
        MAX_STAY_MINUTES,
        np.maximum(1, np.ceil(arrivals["duration_minutes"] * MAX_STAY_MULTIPLIER))
    ).astype(int)
    return arrivals, sites


def simulate_site(site_arrivals, site, charger_count):
    connectors_per_charger = int(site["connectors_per_charger"])
    charger_kw = float(site["charger_power_kw"])
    connector_kw = float(site["max_connector_power_kw"])
    grid_kw = float(site["grid_capacity_kw"])
    grid_dc_budget_kw = grid_kw * CHARGING_EFFICIENCY

    if connectors_per_charger < 1 or min(charger_kw, connector_kw, grid_kw) <= 0:
        raise ValueError(f"Invalid equipment values for {site['site_id']}.")

    # The charger hardware, not simply the number of connectors, sets power.
    ports = [[None] * connectors_per_charger for _ in range(charger_count)]
    events = list(site_arrivals[["start_time", "requested_energy_kwh", "requested_power_kw", "max_stay_min"]].itertuples(index=False, name=None))
    total_arrivals = len(events)
    if not total_arrivals:
        raise ValueError(f"No arrivals found for {site['site_id']}.")

    accepted = rejected = completed = 0
    energy_delivered = requested_accepted = total_occupied_min = 0.0
    peak_dc_kw = peak_grid_kw = 0.0
    minute_count_at_grid_limit = 0
    index = 0
    now = events[0][0]
    active_count = 0

    while index < total_arrivals or active_count > 0:
        if active_count == 0 and index < total_arrivals:
            now = events[index][0]  # Skip empty minutes.

        # Every scenario processes exactly the same sequence of arrivals.
        while index < total_arrivals and events[index][0] <= now:
            _, energy, power, max_stay = events[index]
            index += 1

            # Prefer a completely free charger before using its 2nd connector.
            possible = []
            for ci, charger_ports in enumerate(ports):
                occupied = sum(p is not None for p in charger_ports)
                if occupied < connectors_per_charger:
                    possible.append((occupied, ci))
            if not possible:
                rejected += 1
                continue

            _, ci = min(possible)
            connector_index = ports[ci].index(None)
            ports[ci][connector_index] = {
                "remaining_kwh": float(energy),
                "requested_kw": float(power),
                "max_stay_min": int(max_stay),
                "occupied_min": 0,
            }
            accepted += 1
            active_count += 1
            requested_accepted += float(energy)

        # Step 1: Share each charger's rating between its active connectors.
        car_refs = []
        car_caps = []
        for charger_ports in ports:
            charging_cars = [p for p in charger_ports if p is not None]
            local_caps = [min(connector_kw, p["requested_kw"]) for p in charging_cars]
            local_allocated = fair_allocation(local_caps, charger_kw)
            car_refs.extend(charging_cars)
            car_caps.extend(local_allocated)

        # Step 2: If needed, share the total grid budget site-wide.
        power_to_cars = fair_allocation(car_caps, grid_dc_budget_kw)
        actual_dc_kw = 0.0
        for car, power in zip(car_refs, power_to_cars):
            delivered_now = min(car["remaining_kwh"], power / 60.0)
            car["remaining_kwh"] -= delivered_now
            car["occupied_min"] += 1
            energy_delivered += delivered_now
            actual_dc_kw += delivered_now * 60

        current_grid_kw = actual_dc_kw / CHARGING_EFFICIENCY
        peak_dc_kw = max(peak_dc_kw, actual_dc_kw)
        peak_grid_kw = max(peak_grid_kw, current_grid_kw)
        if current_grid_kw >= grid_kw - 0.01:
            minute_count_at_grid_limit += 1
        if current_grid_kw > grid_kw + 0.01:
            raise AssertionError("Site grid power limit exceeded.")

        # Cars leave when they receive requested energy OR reach max stay.
        for charger_ports in ports:
            for pi, car in enumerate(charger_ports):
                if car is None:
                    continue
                if car["remaining_kwh"] <= EPS or car["occupied_min"] >= car["max_stay_min"]:
                    completed += int(car["remaining_kwh"] <= EPS)
                    total_occupied_min += car["occupied_min"]
                    charger_ports[pi] = None
                    active_count -= 1

        now += ONE_MINUTE

    assert accepted + rejected == total_arrivals
    assert peak_grid_kw <= grid_kw + 0.01

    return {
        "site_id": site["site_id"],
        "city": site["city"],
        "current_chargers": int(site["charger_count"]),
        "scenario_chargers": charger_count,
        "scenario_connectors": charger_count * connectors_per_charger,
        "grid_capacity_kw": round(grid_kw, 2),
        "installed_charger_rating_kw": round(charger_count * charger_kw, 2),
        "arrivals": total_arrivals,
        "cars_served": accepted,
        "cars_rejected": rejected,
        "rejected_pct": round(100 * rejected / total_arrivals, 2),
        "cars_fully_charged_pct": round(100 * completed / accepted, 2) if accepted else 0.0,
        "energy_delivered_kwh": round(energy_delivered, 1),
        "energy_shortfall_served_kwh": round(max(0.0, requested_accepted - energy_delivered), 1),
        "average_charging_power_kw": round(60 * energy_delivered / total_occupied_min, 1) if total_occupied_min else 0.0,
        "average_occupied_minutes": round(total_occupied_min / accepted, 1) if accepted else 0.0,
        "peak_site_grid_draw_kw": round(peak_grid_kw, 1),
        "minutes_at_grid_limit": minute_count_at_grid_limit,
    }


def main():
    arrivals, sites = load_data()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\n--- DYNAMIC LOAD MANAGEMENT: SAME GRID, MORE CHARGERS ---")
    print("Two connectors per physical charger; no waiting queue; 95% efficiency.\n")

    for _, site in sites.iterrows():
        site_arrivals = arrivals[arrivals["site_id"] == site["site_id"]]
        if site_arrivals.empty:
            raise ValueError(f"No data for {site['site_id']}.")
        print(f"Simulating {site['city']}...")
        for count in SCENARIO_CHARGERS:
            rows.append(simulate_site(site_arrivals, site, count))

    results = pd.DataFrame(rows).sort_values(["city", "scenario_chargers"])
    results.to_csv(OUT, index=False)
    print("\n--- EXPANSION RESULTS ---")
    print(results[["city", "current_chargers", "scenario_chargers", "scenario_connectors", "cars_rejected", "rejected_pct", "cars_fully_charged_pct", "average_charging_power_kw", "energy_delivered_kwh", "peak_site_grid_draw_kw"]].to_string(index=False))
    print(f"\nSaved: {OUT}")
    print("\nNOTE: Scenario estimates are synthetic. No waiting queue is modeled.")
    print("Rejected cars' power needs are estimated from served cars at the same site.")
    print("Charging time can increase up to 1.5x the original stay, capped at 120 minutes.")


if __name__ == "__main__":
    main()
