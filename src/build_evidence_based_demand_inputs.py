"""Build source-tracked German EV market context and site demand evidence.

Usage (from the CPO Network Development project root):
    python src\build_evidence_based_demand_inputs.py

Requirements: pandas (already used by this project).

This is a SOURCE-DOCUMENTED DATA PREPARATION step, not a demand prediction.
It does not infer that EV-fleet growth or charging-point growth equals the
future growth of arrivals at an individual CPO charging site.

Inputs:
    data/raw/site_master.csv                         [site ID, city, state]
    data/config/cpo_monthly_arrivals_verified.csv    [optional verified CPO data]

Outputs:
    data/processed/german_market_demand_evidence.csv
    data/processed/site_demand_growth_evidence.csv

If monthly actuals are absent, an EMPTY CSV template is created under data/config.
Existing scenario and financial files are never changed.

Research cutoff: October 24, 2026.
Public data sources:
  KBA BEV vehicle stock, 1 Jan 2026 and year-on-year change:
    https://www.kba.de/DE/Presse/Pressemitteilungen/Fahrzeugbestand/2026/pm09_fz_bestand_pm_komplett.html
  Bundesnetzagentur public charge points, 1 Sep 2026:
    https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/start.html
  State charging-point historical totals for reference years 2025, 2026:
    https://www.statistikportal.de/de/nachhaltigkeit/ergebnisse/ziel-9-industrie-innovation-und-infrastruktur/ladesaeuleninfrastruktur

Important: State charging-point counts and Sep 2026 national count have DIFFERENT
reporting snapshots. Never calculate a growth rate mixing them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW = DATA / 'raw'
CONFIG = DATA / 'config'
PROCESSED = DATA / 'processed'

SOURCE_KBA = 'https://www.kba.de/DE/Presse/Pressemitteilungen/Fahrzeugbestand/2026/pm09_fz_bestand_pm_komplett.html'
SOURCE_BNA = 'https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/start.html'
SOURCE_STATE = 'https://www.statistikportal.de/de/nachhaltigkeit/ergebnisse/ziel-9-industrie-innovation-und-infrastruktur/ladesaeuleninfrastruktur'
AS_OF = '2026-10-24'

# Source: Statistikportal.de. These are reported PUBLIC charging points, NOT
# individual physical chargers and NOT actual station utilization or arrivals.
# Avoid comparing these state year-labelled snapshots against the September
# 2026 national value (different reporting dates).
STATE_PUBLIC_POINTS = {
    'BW': ('Baden-Württemberg', 28792, 33324),
    'BY': ('Bayern', 32321, 37908),
    'BE': ('Berlin', 5617, 6854),
    'BB': ('Brandenburg', 4097, 4729),
    'HB': ('Bremen', 1208, 1530),
    'HH': ('Hamburg', 3467, 4928),
    'HE': ('Hessen', 13125, 15663),
    'MV': ('Mecklenburg-Vorpommern', 2218, 2927),
    'NI': ('Niedersachsen', 16356, 19531),
    'NW': ('Nordrhein-Westfalen', 32088, 39192),
    'RP': ('Rheinland-Pfalz', 6479, 7614),
    'SL': ('Saarland', 1256, 1434),
    'SN': ('Sachsen', 5936, 6928),
    'ST': ('Sachsen-Anhalt', 2855, 3506),
    'SH': ('Schleswig-Holstein', 6612, 7689),
    'TH': ('Thüringen', 3271, 3971),
}

# KBA: real stock of BEV passenger cars on Jan 1 2026. This is NOT the count
# of public charging sessions, nor a validated 2027 site-arrival growth rate.
GERMANY_BEV_STOCK_2026 = 2034260
GERMANY_BEV_YOY_PCT = 23.2
GERMANY_PUBLIC_POINTS_SEP_2026 = 157762 + 56572

TEMPLATE_COLUMNS = [
    'site_id', 'month', 'arrival_attempts',
    'source_reference', 'verified_as_of', 'evidence_type'
]


def create_operator_template(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(path, index=False)
    return True


def load_sites(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing site registry: {path}')
    sites = pd.read_csv(path, dtype=str).fillna('')
    sites.columns = sites.columns.str.strip()
    required = {'site_id', 'city', 'state_code'}
    missing = required - set(sites.columns)
    if missing:
        raise ValueError(f'{path.name} is missing {sorted(missing)}')
    sites = sites[['site_id', 'city', 'state_code']].copy()
    for name in sites.columns:
        sites[name] = sites[name].astype(str).str.strip()
    sites['state_code'] = sites['state_code'].str.upper()
    if sites.empty or sites['site_id'].eq('').any() or sites['site_id'].duplicated().any():
        raise ValueError('Site registry needs nonempty, unique site IDs')
    return sites


def load_operator_data(path: Path, valid_sites: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna('')
    df.columns = df.columns.str.strip()
    missing = set(TEMPLATE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Operator data file is missing {sorted(missing)}')
    if df.empty:
        return df
    for c in TEMPLATE_COLUMNS:
        df[c] = df[c].astype(str).str.strip()
    if (~df['site_id'].isin(valid_sites)).any():
        bad = df.loc[~df['site_id'].isin(valid_sites), 'site_id'].unique().tolist()
        raise ValueError(f'Unknown site IDs in operator evidence: {bad}')
    if (~df['evidence_type'].eq('VERIFIED_OPERATOR_ACTUAL')).any():
        raise ValueError("Only evidence_type=VERIFIED_OPERATOR_ACTUAL is accepted; do not mix synthetic rows")
    if df[['source_reference', 'verified_as_of']].eq('').any().any():
        raise ValueError('Each operator-data row needs a source reference and verification date')

    try:
        dt = pd.to_datetime(df['month'], format='%Y-%m', errors='raise')
        verified = pd.to_datetime(df['verified_as_of'], format='%Y-%m-%d', errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Use month YYYY-MM and verified_as_of YYYY-MM-DD') from exc
    if (verified > pd.Timestamp(AS_OF)).any():
        raise ValueError(f'Verification dates cannot be after project cutoff {AS_OF}')
    if (dt > pd.Timestamp(AS_OF)).any():
        raise ValueError('Historical monthly actuals cannot be in the future')
    numbers = pd.to_numeric(df['arrival_attempts'], errors='coerce')
    if numbers.isna().any() or (numbers < 0).any() or ((numbers % 1) != 0).any():
        raise ValueError('arrival_attempts must be nonnegative whole numbers')
    df['arrival_attempts'] = numbers.astype('int64')
    df['month_dt'] = dt
    if df.duplicated(['site_id', 'month_dt']).any():
        raise ValueError('At most one verified monthly count is allowed per site and month')
    return df


def observed_yoy_if_eligible(site_rows: pd.DataFrame) -> dict:
    """Compare matching 12-month periods, ONLY if 24 contiguous actual months exist.

    This detects observed past growth; it does NOT extrapolate to future years.
    """
    result = {
        'verified_actual_months': 0,
        'older_12m_arrivals': pd.NA,
        'recent_12m_arrivals': pd.NA,
        'observed_operator_yoy_change_pct': pd.NA,
        'latest_actual_month': '',
        'site_growth_status': 'NEED_24_MONTHS_VERIFIED_OPERATOR_ARRIVALS',
    }
    if site_rows.empty:
        return result
    sorted_rows = site_rows.sort_values('month_dt')
    result['verified_actual_months'] = len(sorted_rows)
    result['latest_actual_month'] = sorted_rows.iloc[-1]['month']
    last_24 = sorted_rows.tail(24)
    if len(last_24) != 24:
        return result
    months = pd.period_range(last_24.iloc[0]['month'], last_24.iloc[-1]['month'], freq='M')
    actual = pd.PeriodIndex(last_24['month'], freq='M')
    if len(months) != 24 or not months.equals(actual):
        result['site_growth_status'] = 'GAPS_IN_OPERATOR_MONTHLY_DATA'
        return result
    old = int(last_24.iloc[:12]['arrival_attempts'].sum())
    new = int(last_24.iloc[12:]['arrival_attempts'].sum())
    result['older_12m_arrivals'] = old
    result['recent_12m_arrivals'] = new
    if old == 0:
        result['site_growth_status'] = 'NO_BASELINE_ARRIVALS_TO_ESTIMATE_YOY'
        return result
    result['observed_operator_yoy_change_pct'] = round(100 * (new / old - 1), 2)
    result['site_growth_status'] = 'OBSERVED_SITE_GROWTH_NOT_FUTURE_FORECAST'
    return result


def build_market_ledger() -> pd.DataFrame:
    rows = [
        {
            'geography': 'Germany', 'indicator': 'BEV_REGISTERED_PASSENGER_CARS',
            'period': '2026-01-01', 'value': GERMANY_BEV_STOCK_2026, 'unit': 'vehicles',
            'source': SOURCE_KBA, 'as_of': AS_OF, 'use': 'MARKET_CONTEXT_ONLY',
        },
        {
            'geography': 'Germany', 'indicator': 'BEV_STOCK_PRIOR_YEAR_CHANGE',
            'period': '2025-01-01_to_2026-01-01', 'value': GERMANY_BEV_YOY_PCT, 'unit': '%',
            'source': SOURCE_KBA, 'as_of': AS_OF, 'use': 'NOT_SITE_ARRIVAL_GROWTH',
        },
        {
            'geography': 'Germany', 'indicator': 'PUBLIC_CHARGING_POINTS_IN_OPERATION',
            'period': '2026-09-01', 'value': GERMANY_PUBLIC_POINTS_SEP_2026, 'unit': 'points',
            'source': SOURCE_BNA, 'as_of': AS_OF, 'use': 'MARKET_CONTEXT_ONLY',
        },
    ]
    for code, (name, points25, points26) in STATE_PUBLIC_POINTS.items():
        for year, value in [(2025, points25), (2026, points26)]:
            rows.append({
                'geography': f'{name} ({code})', 'indicator': 'PUBLIC_CHARGING_POINTS_STATE',
                'period': str(year), 'value': value, 'unit': 'points',
                'source': SOURCE_STATE, 'as_of': AS_OF,
                'use': 'REGIONAL_CHARGING_SUPPLY_CONTEXT_NOT_SITE_ARRIVALS',
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sites', type=Path, default=RAW / 'site_master.csv')
    parser.add_argument('--operator-actuals', type=Path, default=CONFIG / 'cpo_monthly_arrivals_verified.csv')
    parser.add_argument('--market-output', type=Path, default=PROCESSED / 'german_market_demand_evidence.csv')
    parser.add_argument('--site-output', type=Path, default=PROCESSED / 'site_demand_growth_evidence.csv')
    args = parser.parse_args()

    sites = load_sites(args.sites)
    created = create_operator_template(args.operator_actuals)
    observed = load_operator_data(args.operator_actuals, set(sites['site_id']))
    results = []
    for _, site in sites.iterrows():
        state_code = site['state_code']
        benchmark = STATE_PUBLIC_POINTS.get(state_code)
        site_rows = observed[observed['site_id'] == site['site_id']] if not observed.empty else observed
        row = {
            'site_id': site['site_id'], 'city': site['city'], 'state_code': state_code,
            'national_bev_stock_yoy_pct_2025_2026': GERMANY_BEV_YOY_PCT,
            'state_public_points_2025': benchmark[1] if benchmark else pd.NA,
            'state_public_points_2026': benchmark[2] if benchmark else pd.NA,
            'state_public_points_change_pct': (
                round((benchmark[2] / benchmark[1] - 1) * 100, 2)
                if benchmark and benchmark[1] else pd.NA
            ),
            'regional_supply_source': SOURCE_STATE if benchmark else '',
            'national_fleet_source': SOURCE_KBA,
            'historical_demo_data_status': '2025_SYNTHETIC_NOT_VALID_FOR_YOY',
            'usable_future_site_growth_pct': pd.NA,
            'future_site_forecast_status': 'NOT_VALIDATED',
        }
        row.update(observed_yoy_if_eligible(site_rows))
        results.append(row)
    market = build_market_ledger()
    site_outlook = pd.DataFrame(results)
    for path in (args.market_output, args.site_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    market.to_csv(args.market_output, index=False)
    site_outlook.to_csv(args.site_output, index=False)

    print('\n--- GERMAN MARKET / SITE DEMAND EVIDENCE ---')
    print(f'Sites: {len(site_outlook)} | Verified market indicators: {len(market)}')
    print(f'National BEV fleet growth (2025 -> 2026): {GERMANY_BEV_YOY_PCT:.1f}% [KBA]')
    print('IMPORTANT: This is NOT a charging-site arrival growth forecast.')
    print(site_outlook[[
        'city', 'state_code', 'state_public_points_change_pct',
        'verified_actual_months', 'site_growth_status',
    ]].to_string(index=False))
    print('\n--- NEXT INPUT REQUIREMENT ---')
    print('Use 24 months of actual CPO monthly arrival-attempt counts to measure site growth.')
    print('Market adoption and public charger growth alone cannot predict site visits.')
    print('No 2027 growth rates have been invented or sent to the simulation.')
    if created:
        print(f'Created empty operator-data template: {args.operator_actuals}')
    else:
        print(f'Operator-data file retained: {args.operator_actuals}')
    print(f'Saved market ledger: {args.market_output}')
    print(f'Saved site evidence: {args.site_output}')


if __name__ == '__main__':
    main()
