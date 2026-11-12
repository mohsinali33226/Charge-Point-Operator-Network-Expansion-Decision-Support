"""Build a conservative German CPO session-behaviour input from Mer audits.

Run from the CPO Network Development root:
    python src/build_mer_cpo_session_input.py

This script does not guess missing charging sessions, rescale suspect kWh,
change the model, or infer lost arrivals / queues / grid connection details.

Required predecessors (already run in this project):
    python src/audit_mer_event_outcomes.py
    python src/diagnose_mer_energy_units.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / 'data' / 'processed'
DEFAULT_OUT = ROOT / 'data' / 'processed'
AUDIT_NAME = 'mer_ccs_event_outcomes.csv'
UNIT_NAME = 'mer_energy_unit_diagnostics.csv'


def must_have(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f'{path.name} is missing columns: {missing}. Rerun its original audit script.')


def truth(values: pd.Series) -> pd.Series:
    return values.astype('string').str.strip().str.lower().isin(['true', '1', 'yes'])


def run(input_dir: Path, output_dir: Path) -> None:
    a_path, u_path = input_dir / AUDIT_NAME, input_dir / UNIT_NAME
    for path in (a_path, u_path):
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing {path}. Run audit_mer_event_outcomes.py and '
                'diagnose_mer_energy_units.py first.'
            )
    audit = pd.read_csv(a_path, low_memory=False)
    unit = pd.read_csv(u_path, low_memory=False)

    must_have(audit, [
        'source_row_number', 'source_location_label', 'start_time', 'end_time',
        'duration_min', 'reported_energy_kwh', 'implied_mean_power_kw',
        'event_class', 'charging_start_observed', 'authorization_failure_flag',
    ], a_path)
    must_have(unit, [
        'source_row_number', 'source_location_label', 'start_time',
        'duration_min', 'total_reported_number', 'energy_unit_diagnosis',
        'safe_reported_kwh_candidate', 'needs_source_unit_confirmation',
    ], u_path)

    for name, frame in ((a_path.name, audit), (u_path.name, unit)):
        frame['source_row_number'] = pd.to_numeric(frame['source_row_number'], errors='raise').astype('int64')
        if frame['source_row_number'].duplicated().any():
            raise ValueError(f'{name}: repeated source_row_number. Audit inputs cannot be merged safely.')

    joined = audit.merge(
        unit[[
            'source_row_number', 'source_location_label', 'start_time',
            'duration_min', 'total_reported_number', 'meter_delta_kwh',
            'energy_unit_diagnosis', 'safe_reported_kwh_candidate',
            'needs_source_unit_confirmation',
        ]],
        on='source_row_number', how='outer', validate='one_to_one',
        suffixes=('_audit', '_unit'), indicator=True,
    )
    if not joined['_merge'].eq('both').all():
        counts = joined['_merge'].value_counts().to_dict()
        raise ValueError(f'Two audits describe different source rows: {counts}. Rerun both on the same Mer file.')

    audit_time = pd.to_datetime(joined['start_time_audit'], errors='coerce')
    unit_time = pd.to_datetime(joined['start_time_unit'], errors='coerce')
    mismatch_location = (
        joined['source_location_label_audit'].fillna('') != joined['source_location_label_unit'].fillna('')
    )
    mismatch_time = (audit_time != unit_time) & ~(audit_time.isna() & unit_time.isna())
    audit_energy = pd.to_numeric(joined['reported_energy_kwh'], errors='coerce')
    unit_energy = pd.to_numeric(joined['total_reported_number'], errors='coerce')
    mismatch_energy = (
        ~(np.isclose(audit_energy.fillna(-999999), unit_energy.fillna(-999999), rtol=1e-7, atol=1e-4))
    )
    audit_duration = pd.to_numeric(joined['duration_min_audit'], errors='coerce')
    unit_duration = pd.to_numeric(joined['duration_min_unit'], errors='coerce')
    mismatch_duration = ~np.isclose(
        audit_duration.fillna(-999999), unit_duration.fillna(-999999), atol=0.01
    )
    mismatches = (mismatch_location | mismatch_time | mismatch_energy | mismatch_duration)
    if mismatches.any():
        sample = joined.loc[mismatches, 'source_row_number'].head(5).tolist()
        raise ValueError(
            f'Inconsistent audit rows ({int(mismatches.sum())}), examples: {sample}. '
            'Do not combine stale or differently parsed data. Rerun the two audits on the same original file.'
        )

    # A reported charging session is the intersection of TWO independent reviews.
    # A missing or inconsistent meter does not become a fabricated zero.
    from_event_audit = (
        joined['event_class'].eq('OBSERVED_CHARGING_SESSION')
        & truth(joined['charging_start_observed'])
        & ~truth(joined['authorization_failure_flag'])
    )
    from_energy_audit = (
        truth(joined['safe_reported_kwh_candidate'])
        & ~truth(joined['needs_source_unit_confirmation'])
        & joined['energy_unit_diagnosis'].isin([
            'REPORTED_KWH_METER_AGREES',
            'REPORTED_KWH_NO_USABLE_METER_CHECK',
        ])
    )
    safe = from_event_audit & from_energy_audit
    selected = joined.loc[safe].copy()
    if selected.empty:
        raise ValueError('No sessions passed both source audits. Check source parsing / filtering first.')

    selected['energy_delivered_kwh'] = audit_energy.loc[safe]
    selected['charging_duration_min'] = audit_duration.loc[safe]
    selected['observed_mean_power_kw'] = (
        selected['energy_delivered_kwh'] * 60.0 / selected['charging_duration_min']
    )
    selected['meter_check_status'] = np.where(
        selected['energy_unit_diagnosis'].eq('REPORTED_KWH_METER_AGREES'),
        'REPORTED_AND_METER_AGREE', 'REPORTED_KWH_NO_USABLE_METER_CHECK',
    )
    selected['record_type'] = 'OBSERVED_CHARGING_START_AND_DELIVERED_ENERGY'
    selected['source_dataset'] = 'Mer_Dataset.csv'
    selected['site_identity_status'] = 'UNVERIFIED_SOURCE_LOCATION_LABEL'
    selected['time_zone_status'] = 'SOURCE_TIME_ZONE_UNVERIFIED'
    selected['vehicle_arrival_status'] = 'UNOBSERVED_BEFORE_CHARGING_START'
    selected['intended_use'] = 'HISTORICAL_CCS_SESSION_BEHAVIOUR_CALIBRATION'
    selected['start_time'] = selected['start_time_audit']
    selected['end_time'] = selected['end_time']
    selected['source_location_label'] = selected['source_location_label_audit']
    selected = selected.sort_values(['start_time', 'source_row_number'])
    export_columns = [
        'source_row_number', 'source_dataset', 'source_location_label',
        'start_time', 'end_time', 'charging_duration_min',
        'energy_delivered_kwh', 'observed_mean_power_kw',
        'meter_check_status', 'record_type', 'site_identity_status',
        'time_zone_status', 'vehicle_arrival_status', 'intended_use',
    ]

    all_locations = joined['source_location_label_audit'].fillna('UNKNOWN').astype(str)
    all_summary = pd.DataFrame({
        'source_location_label': all_locations,
        'records': 1,
        'screened_sessions': safe.astype('int64'),
        'auth_flagged': truth(joined['authorization_failure_flag']).astype('int64'),
        'unresolved_unit_or_energy': (~from_energy_audit).astype('int64'),
    }).groupby('source_location_label', as_index=False).sum()
    all_summary['screened_share_pct'] = (
        all_summary['screened_sessions'] / all_summary['records'] * 100
    ).round(2)
    location_profiles = selected.groupby('source_location_label').agg(
        observed_days=('start_time', lambda s: pd.to_datetime(s).dt.date.nunique()),
        first_screened_start=('start_time', 'min'),
        last_screened_start=('start_time', 'max'),
        median_energy_kwh=('energy_delivered_kwh', 'median'),
        median_duration_min=('charging_duration_min', 'median'),
        median_mean_power_kw=('observed_mean_power_kw', 'median'),
        meter_agreed_sessions=('meter_check_status', lambda s: int((s == 'REPORTED_AND_METER_AGREE').sum())),
    ).reset_index()
    by_location = all_summary.merge(location_profiles, on='source_location_label', how='left', validate='one_to_one')
    by_location['use_limitation'] = 'NOT_VERIFIED_FULL_SITE_DEMAND_OR_LOST_ARRIVALS'

    categories = pd.Series('OTHER_OR_REVIEW', index=joined.index, dtype='object')
    categories.loc[~from_event_audit] = 'EVENT_OUTCOME_NOT_CONFIRMED_CHARGING'
    categories.loc[from_event_audit & ~from_energy_audit] = 'CHARGING_EVENT_BUT_ENERGY_NOT_SCREENED'
    categories.loc[safe] = 'SCREENED_OBSERVED_CCS_SESSION'
    # count the categories without interpreting rejects as congestion
    audit_summary = pd.DataFrame({
        'evidence_gate': categories, 'records': 1,
    }).groupby('evidence_gate', as_index=False).size()
    audit_summary = audit_summary.rename(columns={'size': 'records'})

    output_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = output_dir / 'mer_ccs_screened_charging_sessions.csv'
    locations_path = output_dir / 'mer_ccs_screened_location_coverage.csv'
    audit_path = output_dir / 'mer_ccs_session_input_evidence_summary.csv'
    selected[export_columns].to_csv(sessions_path, index=False)
    by_location.to_csv(locations_path, index=False)
    audit_summary.to_csv(audit_path, index=False)

    print('\n--- MER CCS / REAL-SESSION INPUT GATE ---')
    print(f'CCS source event rows checked: {len(joined):,}')
    print(f'Screened observed charging sessions: {len(selected):,}')
    print(f'With meter agreement: {(selected.meter_check_status == "REPORTED_AND_METER_AGREE").sum():,}')
    print(f'With reported kWh but no usable meter check: {(selected.meter_check_status == "REPORTED_KWH_NO_USABLE_METER_CHECK").sum():,}')
    print(f'Not accepted into calibration: {len(joined) - len(selected):,}')
    print('\nEvidence-gate outcomes:')
    print(audit_summary.to_string(index=False))
    print('\nLocation-label coverage (top 6):')
    print(by_location[['source_location_label', 'screened_sessions', 'observed_days',
                       'median_energy_kwh', 'median_duration_min',
                       'screened_share_pct']].sort_values(
                           'screened_sessions', ascending=False).head(6).to_string(index=False))
    print('\nSaved:')
    print(sessions_path)
    print(locations_path)
    print(audit_path)
    print('\nLIMIT: session starts are NOT all vehicle arrivals; site IDs are unverified.')
    print('No energy units were changed; missing grid/queues/lost cars remain UNKNOWN.')
    print('Existing simulation and financial results remain unchanged.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Screen Mer CCS charging records as model inputs')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_IN)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
