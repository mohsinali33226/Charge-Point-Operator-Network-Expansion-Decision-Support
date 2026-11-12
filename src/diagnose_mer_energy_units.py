"""Investigate Mer CCS charging-energy units and meter consistency.

Run from the CPO Network Development root:
    python src/diagnose_mer_energy_units.py

This is a DATA QUALITY investigation, not an automatic data correction.
It never converts an observed energy value or treats authorization failures
as successful arrivals. Existing simulator inputs are not touched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'data' / 'external' / 'Mer_Dataset.csv'
DEFAULT_OUTPUT = ROOT / 'data' / 'processed'
MAX_SESSION_ENERGY_KWH = 250.0  # diagnostic review boundary, not a universal vehicle limit
MAX_AVG_POWER_KW = 500.0      # diagnostic review boundary, not a hardware specification
MIN_DURATION_MIN = 2.0
MAX_DURATION_MIN = 240.0


def numeric_german(values: pd.Series) -> pd.Series:
    """Parse comma-decimal strings; never infer Wh from an integer automatically."""
    s = values.astype('string').str.strip().str.replace('\u00a0', '', regex=False)
    # For values with a decimal comma, dots are grouping separators.
    has_comma = s.str.contains(',', regex=False, na=False)
    s = s.where(~has_comma, s.str.replace('.', '', regex=False))
    s = s.str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')


def matching_column(frame: pd.DataFrame, wanted: str) -> str | None:
    def normal(x: str) -> str:
        return re.sub(r'[^a-z0-9]', '', x.casefold())
    lookup = {normal(x): x for x in frame.columns}
    return lookup.get(normal(wanted))


def match_energy(reported: pd.Series, meter: pd.Series) -> pd.Series:
    """Agreement criterion is conservative and only for screening."""
    tolerance = np.maximum(1.0, 0.10 * reported.abs())
    return (reported.notna() & meter.notna() & meter.ge(0) &
            (reported - meter).abs().le(tolerance)).fillna(False)


def run(source: Path, out_dir: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f'Cannot find Mer file: {source}')
    try:
        df = pd.read_csv(source, sep=';', dtype='string', encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(source, sep=';', dtype='string', encoding='cp1252', low_memory=False)
    df.columns = df.columns.str.strip()

    names = {}
    for wanted in ('Charger Type', 'Original Location', 'Stop Reason', 'Start Date',
                   'End Date', 'Total kWh', 'Start kWh', 'End kWh'):
        actual = matching_column(df, wanted)
        if actual is None and wanted not in ('Start kWh', 'End kWh'):
            raise ValueError(f'Missing required column {wanted!r}. Found: {df.columns.tolist()}')
        names[wanted] = actual

    is_ccs = df[names['Charger Type']].str.contains(r'CCS', case=False, regex=True, na=False)
    d = df.loc[is_ccs].copy()
    d['source_row_number'] = d.index + 2
    d['source_location_label'] = d[names['Original Location']].fillna('UNKNOWN').str.strip()
    d['source_stop_reason'] = d[names['Stop Reason']].fillna('')
    reason = d['source_stop_reason']
    d['authorization_failure'] = reason.str.contains(
        r'Authorization Failure|Card not found|Invalid card|Access denied|No matching plan|not fully setup for billing',
        case=False, na=False, regex=True).fillna(False)
    start = pd.to_datetime(d[names['Start Date']], errors='coerce')
    end = pd.to_datetime(d[names['End Date']], errors='coerce')
    d['start_time'] = start
    d['duration_min'] = (end - start).dt.total_seconds() / 60.0
    d['raw_total_energy_string'] = d[names['Total kWh']]
    d['total_reported_number'] = numeric_german(d[names['Total kWh']])

    meter_available = names['Start kWh'] is not None and names['End kWh'] is not None
    if meter_available:
        a = numeric_german(d[names['Start kWh']])
        b = numeric_german(d[names['End kWh']])
        d['meter_delta_kwh'] = b - a
    else:
        d['meter_delta_kwh'] = np.nan

    total = d['total_reported_number']
    meter = d['meter_delta_kwh']
    direct_plausible = total.gt(0) & total.le(MAX_SESSION_ENERGY_KWH)
    as_wh = total / 1000.0
    wh_plausible = total.gt(MAX_SESSION_ENERGY_KWH) & as_wh.gt(0) & as_wh.le(MAX_SESSION_ENERGY_KWH)
    meter_plausible = meter.ge(0) & meter.le(MAX_SESSION_ENERGY_KWH)
    direct_agrees = match_energy(total, meter)
    wh_agrees = match_energy(as_wh, meter)
    meter_known_good = meter.notna() & meter_plausible

    conditions = [
        direct_plausible & direct_agrees,
        direct_plausible & meter_known_good & ~direct_agrees,
        direct_plausible,
        wh_plausible & wh_agrees,
        wh_plausible & meter_known_good & ~wh_agrees,
        wh_plausible,
        total.isna() | total.le(0),
    ]
    d['energy_unit_diagnosis'] = np.select(
        [c.fillna(False).to_numpy(dtype=bool) for c in conditions],
        [
            'REPORTED_KWH_METER_AGREES',
            'REPORTED_KWH_METER_DISAGREES_REVIEW',
            'REPORTED_KWH_NO_USABLE_METER_CHECK',
            'POSSIBLE_WH_SCALE_METER_SUPPORTS_REVIEW',
            'POSSIBLE_WH_SCALE_METER_DISAGREES_REVIEW',
            'POSSIBLE_WH_SCALE_NO_USABLE_METER_REVIEW',
            'MISSING_OR_NONPOSITIVE_ENERGY',
        ],
        default='OTHER_ENERGY_VALUE_REVIEW'
    )
    d['hypothetical_total_divided_by_1000_kwh'] = as_wh.where(wh_plausible)
    d['duration_plausible'] = d['duration_min'].between(MIN_DURATION_MIN, MAX_DURATION_MIN)
    # This classification is strict, and distinct from an observed customer arrival.
    usable_direct = (direct_plausible & ~d['authorization_failure'] &
                     d['duration_plausible'].fillna(False) &
                     ((total * 60 / d['duration_min']).le(MAX_AVG_POWER_KW)) &
                     ~d['energy_unit_diagnosis'].eq('REPORTED_KWH_METER_DISAGREES_REVIEW'))
    d['safe_reported_kwh_candidate'] = usable_direct.fillna(False)
    # A Wh-scale hypothesis is NOT accepted into charging-session input automatically.
    d['needs_source_unit_confirmation'] = wh_plausible

    summary = d.groupby('energy_unit_diagnosis', dropna=False).agg(
        records=('source_row_number', 'size'),
        auth_failures=('authorization_failure', 'sum'),
        plausible_duration=('duration_plausible', 'sum'),
        safe_direct_kwh_candidates=('safe_reported_kwh_candidate', 'sum'),
    ).reset_index().sort_values('records', ascending=False)
    summary['non_auth_records'] = summary['records'] - summary['auth_failures']

    by_location = d.groupby('source_location_label', dropna=False).agg(
        ccs_logs=('source_row_number', 'size'),
        auth_failures=('authorization_failure', 'sum'),
        safe_direct_kwh_candidates=('safe_reported_kwh_candidate', 'sum'),
        possible_wh_unit_rows=('needs_source_unit_confirmation', 'sum'),
        possible_wh_with_meter_support=('energy_unit_diagnosis', lambda x: int((x == 'POSSIBLE_WH_SCALE_METER_SUPPORTS_REVIEW').sum())),
    ).reset_index().sort_values('ccs_logs', ascending=False)

    columns = [
        'source_row_number','source_location_label','start_time','source_stop_reason',
        'authorization_failure','duration_min','raw_total_energy_string',
        'total_reported_number','meter_delta_kwh',
        'hypothetical_total_divided_by_1000_kwh',
        'energy_unit_diagnosis','safe_reported_kwh_candidate',
        'needs_source_unit_confirmation',
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = out_dir / 'mer_energy_unit_diagnostics.csv'
    summary_path = out_dir / 'mer_energy_unit_diagnosis_summary.csv'
    location_path = out_dir / 'mer_energy_unit_by_location.csv'
    examples_path = out_dir / 'mer_energy_unit_review_examples.csv'
    d[columns].to_csv(diagnostic_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_location.to_csv(location_path, index=False)
    reviews = d.loc[d['needs_source_unit_confirmation'] & ~d['authorization_failure']]
    reviews = reviews.sort_values('source_row_number').head(60)
    reviews[columns].to_csv(examples_path, index=False)

    print('\n--- MER CCS ENERGY-UNIT / METER DIAGNOSIS ---')
    print(f'CCS event logs checked: {len(d):,}')
    print(f'Authorization-failure logs: {int(d.authorization_failure.sum()):,} (NEVER counted as congestion)')
    print('\nEnergy interpretation (classes do not imply verified charging starts):')
    print(summary[['energy_unit_diagnosis', 'records', 'non_auth_records']].to_string(index=False))
    print('\nMost relevant checks:')
    print('Reported kWh candidate sessions (not full-site demand):', f'{int(d.safe_reported_kwh_candidate.sum()):,}')
    print('Possible /1000 scale, meter-consistent (still REVIEW):',
          f'{int((d.energy_unit_diagnosis == "POSSIBLE_WH_SCALE_METER_SUPPORTS_REVIEW").sum()):,}')
    print('Possible /1000 scale, NO meter proof or meter disagreement:',
          f'{int((wh_plausible & ~wh_agrees).sum()):,}')
    print('\nLargest source labels:')
    print(by_location.head(8).to_string(index=False))
    print('\nSaved:', summary_path, location_path, examples_path, diagnostic_path, sep='\n')
    print('\nIMPORTANT: /1000 is only a HYPOTHESIS; no source values were altered.')
    print('Meter agreement supports consistency, but is not independent proof of units.')
    print('Source location labels remain unverified physical-site identities.')
    print('Successful charging starts cannot reveal queues or lost customers.')


def main():
    ap = argparse.ArgumentParser(description='Audit possible Mer energy-unit mismatches')
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    run(args.source, args.output_dir)

if __name__ == '__main__':
    main()
