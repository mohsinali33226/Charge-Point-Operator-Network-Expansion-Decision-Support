r"""Prepare REAL reported CCS charging-start records from the original Mer CSV.

Run from the CPO Network Development repository:
    python src\prepare_mer_observed_ccs_starts.py

Key rules:
- Read the original source rather than the heavily filtered behaviour-calibration file.
- Do not infer failed arrivals, queues or demand from successful session starts.
- Original Location is a source LABEL, not a verified physical charging site.
- Source timestamps have no confirmed timezone; do not merge with other sources
  on UTC without verifying their timestamp definitions.
- Does not overwrite existing simulator inputs or forecasts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VERSION = '2026-10-31-v1'
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'external' / 'Mer_Dataset.csv'
OUT = ROOT / 'data' / 'processed'
CONFIG = ROOT / 'data' / 'config'


def number_from_german(series: pd.Series) -> pd.Series:
    """Read German decimal commas, accepting missing cells as NaN."""
    text = series.astype('string').str.strip().str.replace('\u00a0', '', regex=False)
    text = text.str.replace(r'\.(?=\d{3}(?:,|$))', '', regex=True)
    return pd.to_numeric(text.str.replace(',', '.', regex=False), errors='coerce')


def read_mer(source: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(source, sep=';', encoding='utf-8-sig', dtype='string', low_memory=False)
    except UnicodeDecodeError:
        data = pd.read_csv(source, sep=';', encoding='cp1252', dtype='string', low_memory=False)
    data.columns = [str(c).strip() for c in data.columns]
    required = {'Charger Type', 'Original Location', 'Start Date', 'End Date', 'Total kWh'}
    # The published CSV has used variant spellings/capitalization of kWh.
    name_lookup = {c.casefold(): c for c in data.columns}
    missing = [c for c in required if c.casefold() not in name_lookup]
    if missing:
        raise ValueError(f'Mer source missing columns: {missing}. Found: {data.columns.tolist()}')
    return data.rename(columns={name_lookup[c.casefold()]: c for c in required})


def run(source: Path, out_dir: Path, config_dir: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f'Mer source file not found: {source}')
    raw = read_mer(source)
    original_rows = len(raw)

    # CCS ONLY: no filtering on duration/energy, because that selection would
    # discard many real reported starts and bias the arrival-time distribution.
    is_ccs = raw['Charger Type'].str.contains(r'\bCCS\b', case=False, na=False, regex=True)
    ccs = raw.loc[is_ccs].copy()
    ccs['source_row_number'] = ccs.index + 2  # header on line 1
    ccs['start_time'] = pd.to_datetime(ccs['Start Date'], errors='coerce')
    ccs['end_time'] = pd.to_datetime(ccs['End Date'], errors='coerce')
    ccs['energy_kwh'] = number_from_german(ccs['Total kWh'])
    ccs['duration_min'] = (
        (ccs['end_time'] - ccs['start_time']).dt.total_seconds() / 60
    )
    ccs['location_label'] = ccs['Original Location'].astype('string').str.strip()
    ccs['location_label'] = ccs['location_label'].mask(ccs['location_label'].eq(''))

    valid_start = ccs['start_time'].notna()
    valid_label = ccs['location_label'].notna()
    dropped_missing_time = int((~valid_start).sum())
    dropped_unknown_label = int((valid_start & ~valid_label).sum())
    ccs = ccs.loc[valid_start & valid_label].copy()
    if ccs.empty:
        raise ValueError('No CCS start records with usable timestamps and location labels.')

    # A weak plausibility check is NOT a test of vehicle demand or charger limits.
    # Keep all recorded start rows for audit, and distinguish records with
    # plausible delivered-energy evidence from aborted/invalid/unknown records.
    ccs['has_positive_charging_evidence'] = (
        ccs['energy_kwh'].gt(0)
        & ccs['duration_min'].between(1, 24 * 60, inclusive='both')
    )
    ccs['start_date'] = ccs['start_time'].dt.strftime('%Y-%m-%d')
    ccs['start_month'] = ccs['start_time'].dt.strftime('%Y-%m')
    ccs['start_hour'] = ccs['start_time'].dt.hour
    ccs['start_weekday'] = ccs['start_time'].dt.dayofweek
    ccs['start_year'] = ccs['start_time'].dt.year
    ccs['observation_type'] = 'REPORTED_CCS_SESSION_START'
    ccs['physical_site_verified'] = False
    ccs['timestamp_timezone_verified'] = False
    events = ccs.rename(columns={
        'location_label': 'source_location_label',
        'energy_kwh': 'reported_energy_kwh',
        'duration_min': 'reported_duration_min',
        'Charger Type': 'source_charger_type',
    })[
        ['source_row_number', 'source_location_label', 'source_charger_type',
         'start_time', 'end_time', 'start_date', 'start_month', 'start_year',
         'start_hour', 'start_weekday', 'reported_energy_kwh',
         'reported_duration_min', 'has_positive_charging_evidence',
         'observation_type', 'physical_site_verified',
         'timestamp_timezone_verified']
    ].sort_values(['start_time', 'source_row_number']).reset_index(drop=True)

    monthly = (events.groupby(['source_location_label', 'start_month'], sort=True)
               .agg(reported_session_starts=('start_time', 'size'),
                    positive_charge_records=('has_positive_charging_evidence', 'sum'),
                    days_with_recorded_starts=('start_date', 'nunique'))
               .reset_index())
    monthly['month_calendar_days'] = pd.to_datetime(monthly['start_month'] + '-01').dt.days_in_month
    monthly['coverage_note'] = 'DAYS_WITH_NO_RECORDS_CANNOT_BE_ASSUMED_ZERO_DEMAND'

    hours = (events.groupby(['source_location_label', 'start_weekday', 'start_hour'], sort=True)
             .agg(reported_session_starts=('start_time', 'size'),
                  positive_charge_records=('has_positive_charging_evidence', 'sum'))
             .reset_index())
    hours['share_of_all_reported_starts_in_label_pct'] = (
        100 * hours['reported_session_starts'] /
        hours.groupby('source_location_label')['reported_session_starts'].transform('sum')
    ).round(3)

    locations = (events.groupby('source_location_label', sort=False)
                 .agg(reported_ccs_starts=('start_time', 'size'),
                      positive_charge_records=('has_positive_charging_evidence', 'sum'),
                      recorded_start_days=('start_date', 'nunique'),
                      months_with_at_least_one_start=('start_month', 'nunique'),
                      first_recorded_start=('start_time', 'min'),
                      last_recorded_start=('start_time', 'max'))
                 .reset_index()
                 .sort_values('reported_ccs_starts', ascending=False))
    locations['records_without_positive_charging_evidence'] = (
        locations['reported_ccs_starts'] - locations['positive_charge_records']
    )
    locations['site_identity_status'] = 'UNVERIFIED_SOURCE_LABEL'
    locations['total_customer_arrivals_status'] = 'UNKNOWN_REJECTED_AND_WAITING_DRIVERS'
    locations['charging_hardware_status'] = 'UNVERIFIED_FOR_SOURCE_LABEL'
    locations['data_source'] = source.name

    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'observed starts': out_dir / 'mer_ccs_reported_start_events.csv',
        'monthly counts': out_dir / 'mer_ccs_monthly_reported_starts.csv',
        'hourly patterns': out_dir / 'mer_ccs_reported_start_hour_profiles.csv',
        'location audit': out_dir / 'mer_ccs_source_location_audit.csv',
    }
    events.to_csv(files['observed starts'], index=False)
    monthly.to_csv(files['monthly counts'], index=False)
    hours.to_csv(files['hourly patterns'], index=False)
    locations.to_csv(files['location audit'], index=False)

    # A separate mapping template makes the schema ready for a real CPO data
    # export. Unknown site identities and connector counts stay BLANK.
    mapping_file = config_dir / 'mer_source_location_site_mapping.csv'
    if not mapping_file.exists():
        pd.DataFrame({
            'source_location_label': locations['source_location_label'],
            'verified_site_id': '',
            'verified_physical_chargers': '',
            'verified_simultaneous_connectors': '',
            'verified_site_grid_limit_kw': '',
            'evidence_source': '',
            'verification_status': 'NOT_VERIFIED',
        }).to_csv(mapping_file, index=False)
        mapping_note = 'Created new blank mapping (no station identities invented)'
    else:
        mapping_note = 'Existing mapping file preserved unchanged'

    print('\n--- MER REAL CCS REPORTED STARTS / SITE IDENTITY AUDIT ---')
    print(f'Importer version: {VERSION}')
    print(f'Original Mer records: {original_rows:,}')
    print(f'CCS source records: {int(is_ccs.sum()):,}')
    print(f'CCS records with usable starts and location labels: {len(events):,}')
    print(f'  Of these, records with plausible positive charging evidence: {int(events.has_positive_charging_evidence.sum()):,}')
    print(f'  Start timestamps missing/invalid: {dropped_missing_time:,}')
    print(f'  Location labels missing after valid start: {dropped_unknown_label:,}')
    print(f'Distinct location labels: {len(locations)} (NOT verified site IDs)')
    print(f'Coverage: {events.start_time.min()} to {events.start_time.max()} (timezone unverified)')
    print('\n--- TOP SOURCE LOCATION LABELS ---')
    print(locations[['source_location_label', 'reported_ccs_starts',
                     'positive_charge_records', 'recorded_start_days',
                     'months_with_at_least_one_start']].head(12).to_string(index=False))
    print('\nIMPORTANT: A recorded CCS session start is NOT proof of every vehicle arrival.')
    print('Rejected cars, waiting queues, source label -> physical site mapping: UNKNOWN.')
    print('Do not substitute these location labels for verified grid/charger records.')
    for k, path in files.items():
        print(f'Saved {k}: {path}')
    print(f'Site mapping template: {mapping_file} ({mapping_note})')


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract all reported CCS start events from original Mer data')
    parser.add_argument('--input', type=Path, default=SOURCE)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    parser.add_argument('--config-dir', type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.input, args.out_dir, args.config_dir)


if __name__ == '__main__':
    main()
