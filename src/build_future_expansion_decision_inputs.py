"""Combine conditional expansion simulations with German market/site evidence.

Run from project root:
    python src\build_future_expansion_decision_inputs.py

Reads EXISTING CSVs:
    data/processed/future_demand_network_impact.csv
    data/processed/site_demand_growth_evidence.csv

Writes NEW CSVs:
    data/processed/future_expansion_decision_inputs.csv
    data/processed/future_expansion_network_summary.csv

This is engineering decision support, NOT a forecast or an investment approval.
It deliberately refuses to turn the growth of Germany's BEV fleet or a state's
public charging-point count into site-level arrival growth.

All outputs remain marked as synthetic/conditional until real operator data,
site-specific forecasts, grid feasibility and financial evidence are available.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed'

IMPACT_REQUIRED = {
    'site_id', 'city', 'planning_year', 'assumed_annual_arrival_growth_pct',
    'scenario_id', 'scenario_type', 'scenario_chargers',
    'scenario_arrivals', 'cars_rejected',
    'additional_cars_served_vs_current',
    'additional_energy_delivered_kwh_vs_current',
}
EVIDENCE_REQUIRED = {
    'site_id', 'state_code', 'state_public_points_change_pct',
    'verified_actual_months', 'site_growth_status',
    'future_site_forecast_status',
}


def load_csv(path: Path, required: set[str], description: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {description}: {path}\n'
            'Run the previous demand-evidence or future-demand simulation step.'
        )
    df = pd.read_csv(path, dtype={'site_id': str})
    df.columns = df.columns.str.strip()
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{description} is missing columns: {sorted(missing)}')
    if df.empty or df['site_id'].isna().any():
        raise ValueError(f'{description} must have rows with a site_id')
    df['site_id'] = df['site_id'].str.strip()
    if df['site_id'].eq('').any():
        raise ValueError(f'{description} has a blank site_id')
    return df


def numeric(df: pd.DataFrame, names: list[str], label: str) -> None:
    for name in names:
        df[name] = pd.to_numeric(df[name], errors='coerce')
        if df[name].isna().any():
            raise ValueError(f'{label}: invalid or missing numeric value in {name}')


def build(impact: pd.DataFrame, evidence: pd.DataFrame):
    if evidence['site_id'].duplicated().any():
        raise ValueError('Site evidence must have one row per site')
    if impact[['site_id', 'planning_year', 'assumed_annual_arrival_growth_pct', 'scenario_id']].duplicated().any():
        raise ValueError('Duplicate scenario for the same site, year and demand assumption')

    imp = impact.copy()
    ev = evidence.copy()
    numeric(imp, [
        'planning_year', 'assumed_annual_arrival_growth_pct',
        'scenario_chargers', 'scenario_arrivals', 'cars_rejected',
        'additional_cars_served_vs_current',
        'additional_energy_delivered_kwh_vs_current',
    ], 'Future-demand scenarios')
    numeric(ev, ['verified_actual_months'], 'Site evidence')
    if (imp['scenario_arrivals'] <= 0).any():
        raise ValueError('Simulation arrivals must be positive')
    if (imp['cars_rejected'] < 0).any() or (imp['cars_rejected'] > imp['scenario_arrivals']).any():
        raise ValueError('Rejected cars must be between zero and total arrivals')
    if (ev['verified_actual_months'] < 0).any():
        raise ValueError('Verified operator months cannot be negative')

    key = ['site_id', 'planning_year', 'assumed_annual_arrival_growth_pct']
    baselines = imp[imp['scenario_type'].eq('CURRENT')].copy()
    if baselines[key].duplicated().any() or (baselines.groupby(key).size() != 1).any():
        raise ValueError('Exactly one CURRENT configuration is required per site/year/demand assumption')
    if len(baselines) != imp.groupby(key).ngroups:
        raise ValueError('At least one demand scenario lacks a CURRENT baseline')
    baselines = baselines[key + ['scenario_chargers', 'cars_rejected', 'scenario_arrivals']].rename(
        columns={
            'scenario_chargers': 'current_chargers',
            'cars_rejected': 'baseline_cars_rejected',
            'scenario_arrivals': 'baseline_arrivals',
        }
    )

    joined = imp.merge(baselines, on=key, how='left', validate='many_to_one')
    if joined['baseline_arrivals'].isna().any():
        raise ValueError('Missing CURRENT baseline for a scenario')
    if (~joined['scenario_arrivals'].eq(joined['baseline_arrivals'])).any():
        raise ValueError('Current and expansion scenarios must use identical arrivals')
    joined = joined.merge(ev, on='site_id', how='left', validate='many_to_one',
                          suffixes=('', '_evidence'), indicator=True)
    if (~joined['_merge'].eq('both')).any():
        missing = joined.loc[~joined['_merge'].eq('both'), 'site_id'].unique().tolist()
        raise ValueError(f'Sites without market/site evidence: {missing}')
    joined = joined.drop(columns=['_merge'])

    joined['rejection_rate_pct'] = (100 * joined['cars_rejected'] / joined['scenario_arrivals']).round(2)
    joined['baseline_rejection_rate_pct'] = (
        100 * joined['baseline_cars_rejected'] / joined['baseline_arrivals']
    ).round(2)
    joined['avoided_rejections_vs_current'] = (
        joined['baseline_cars_rejected'] - joined['cars_rejected']
    ).astype(int)
    joined['additional_physical_chargers'] = (
        joined['scenario_chargers'] - joined['current_chargers']
    ).astype(int)

    # Important: even 24 verified historical months is not automatically a
    # validated prediction of the site's future arrivals.
    joined['demand_outlook_classification'] = 'CONDITIONAL_STRESS_TEST_NOT_SITE_FORECAST'
    joined['market_indicator_role'] = 'REGIONAL_CONTEXT_ONLY_NOT_SITE_ARRIVAL_GROWTH'
    joined['decision_readiness'] = 'ENGINEERING_DEMO_EVIDENCE_REQUIRED'
    joined['investment_recommendation'] = 'NOT_ASSESSED'

    # Per site & growth assumption: technical sensitivity, not an invest ranking.
    summary = []
    for (site, year, rate), group in joined.groupby(key, sort=True):
        base = group.loc[group['scenario_type'].eq('CURRENT')]
        options = group.loc[~group['scenario_type'].eq('CURRENT')]
        if len(base) != 1:
            raise ValueError(f'{site}: exactly one baseline expected')
        b = base.iloc[0]
        summary.append({
            'site_id': site,
            'city': b['city'],
            'state_code': b['state_code'],
            'planning_year': int(year),
            'assumed_annual_arrival_growth_pct': float(rate),
            'current_chargers': int(b['current_chargers']),
            'scenario_arrivals': int(b['baseline_arrivals']),
            'baseline_cars_rejected': int(b['baseline_cars_rejected']),
            'baseline_rejection_rate_pct': float(b['baseline_rejection_rate_pct']),
            'expansion_options_tested': len(options),
            'largest_avoided_rejections_among_tested_options': (
                int(options['avoided_rejections_vs_current'].max()) if len(options) else 0
            ),
            'lowest_rejected_cars_among_tested_options': (
                int(options['cars_rejected'].min()) if len(options) else int(b['cars_rejected'])
            ),
            'verified_actual_months': int(b['verified_actual_months']),
            'site_growth_status': b['site_growth_status'],
            'state_public_points_change_pct': b['state_public_points_change_pct'],
            'demand_outlook_classification': 'CONDITIONAL_STRESS_TEST_NOT_SITE_FORECAST',
            'decision_readiness': 'ENGINEERING_DEMO_EVIDENCE_REQUIRED',
            'investment_recommendation': 'NOT_ASSESSED',
        })
    network = pd.DataFrame(summary)
    joined = joined.sort_values(key + ['scenario_chargers', 'scenario_id']).reset_index(drop=True)
    network = network.sort_values(['assumed_annual_arrival_growth_pct', 'site_id']).reset_index(drop=True)
    return joined, network


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulation', type=Path,
                   default=PROCESSED / 'future_demand_network_impact.csv')
    p.add_argument('--evidence', type=Path,
                   default=PROCESSED / 'site_demand_growth_evidence.csv')
    p.add_argument('--scenario-output', type=Path,
                   default=PROCESSED / 'future_expansion_decision_inputs.csv')
    p.add_argument('--network-output', type=Path,
                   default=PROCESSED / 'future_expansion_network_summary.csv')
    args = p.parse_args()
    imp = load_csv(args.simulation, IMPACT_REQUIRED, 'growth-aware expansion results')
    ev = load_csv(args.evidence, EVIDENCE_REQUIRED, 'German market/site evidence')
    if args.scenario_output.resolve() == args.network_output.resolve():
        raise ValueError('Output paths must be different')
    scenarios, network = build(imp, ev)
    for path, df in [(args.scenario_output, scenarios), (args.network_output, network)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    print('\n--- EVIDENCE-LINKED FUTURE EXPANSION SCREEN ---')
    print(f"Sites: {network.site_id.nunique()} | Demand assumptions: {network.assumed_annual_arrival_growth_pct.nunique()} | Configurations: {len(scenarios)}")
    print(network[[
        'city', 'assumed_annual_arrival_growth_pct', 'baseline_rejection_rate_pct',
        'expansion_options_tested', 'largest_avoided_rejections_among_tested_options',
        'verified_actual_months', 'decision_readiness'
    ]].to_string(index=False))
    print('\nNOTE: Public charging-point growth is market context, NOT predicted site demand.')
    print('These are conditional scenarios using synthetic 2025 arrivals, NOT verified 2027 forecasts.')
    print('No grid approval, financial recommendation or site ranking is implied.')
    print(f'Saved: {args.scenario_output}')
    print(f'Saved: {args.network_output}')


if __name__ == '__main__':
    main()
