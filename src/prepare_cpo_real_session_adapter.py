"""Prepare a source-traceable observed-session table for the German CPO prototype.

Default input: quality-gated historical Mer CCS sessions.
No estimated vehicle arrivals, rejected cars, grid capacities, or site identities.

From project root:
    python src/prepare_cpo_real_session_adapter.py

Optional real operator input (canonical header template generated automatically):
    python src/prepare_cpo_real_session_adapter.py --operator-csv data/external/operator_sessions.csv

Existing project simulation, synthetic data, and forecasts are not changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MER = ROOT / 'data/processed/mer_ccs_screened_charging_sessions.csv'
DEFAULT_OUT = ROOT / 'data/processed'
DEFAULT_CONFIG = ROOT / 'data/config'
MAPPING_COLUMNS = ['source_location_label','verified_site_id','mapping_status','evidence_reference','site_timezone']
OPERATOR_TEMPLATE_COLUMNS = [
    'session_id','site_id','start_time','end_time','energy_delivered_kwh',
    'charger_id','connector_id','site_timezone','source_system',
]
CANONICAL_COLUMNS = [
    'source_system','source_event_id','source_location_label','site_id',
    'start_time_source','end_time_source','duration_min','delivered_energy_kwh',
    'session_average_power_kw','charger_id','connector_id','meter_check_status',
    'site_identity_status','timezone_status','observed_record_type',
    'includes_unsuccessful_arrivals','captures_waiting_queue','captures_lost_demand',
    'data_quality_status',
]


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f'{label}: missing columns {missing}. Do not guess their meaning.')


def preserve_mapping(path: Path, labels: pd.Series) -> pd.DataFrame:
    """Append only new labels; keep any evidence supplied by a CPO unchanged."""
    names = sorted(set(labels.dropna().astype(str).str.strip()) - {''})
    if path.is_file():
        m = pd.read_csv(path, dtype='string', keep_default_na=False)
        require_columns(m, MAPPING_COLUMNS, str(path))
    else:
        m = pd.DataFrame(columns=MAPPING_COLUMNS, dtype='string')
    if m['source_location_label'].duplicated().any():
        raise ValueError('Mer mapping has duplicate source_location_label entries.')
    present = set(m['source_location_label'].astype(str))
    additions = pd.DataFrame([
        {'source_location_label': name, 'verified_site_id': '',
         'mapping_status': 'UNVERIFIED', 'evidence_reference': '', 'site_timezone': ''}
        for name in names if name not in present
    ], columns=MAPPING_COLUMNS)
    if not additions.empty:
        m = pd.concat([m, additions], ignore_index=True)
    m.to_csv(path, index=False)
    return m


def read_mer(path: Path, config_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={'source_row_number':'string'}, low_memory=False)
    require_columns(d, [
        'source_row_number','source_dataset','source_location_label', 'start_time','end_time',
        'charging_duration_min','energy_delivered_kwh','observed_mean_power_kw',
        'meter_check_status','record_type','site_identity_status','time_zone_status',
    ], path.name)
    if d.empty:
        raise ValueError('Screened Mer sessions file contains no records.')
    if d['source_row_number'].isna().any() or d['source_row_number'].duplicated().any():
        raise ValueError('Mer screen contains missing/duplicate source row IDs.')
    mapping = preserve_mapping(config_dir / 'mer_verified_site_mapping.csv', d['source_location_label'])
    status = mapping['mapping_status'].str.upper().str.strip()
    approved = (
        status.eq('VERIFIED') & mapping['verified_site_id'].str.strip().ne('')
        & mapping['evidence_reference'].str.strip().ne('')
        & mapping['site_timezone'].str.strip().ne('')
    )
    mapping = mapping.assign(
        actual_site_id=mapping['verified_site_id'].where(approved, ''),
        accepted_mapping=approved,
    )
    merge = d.merge(
        mapping[['source_location_label','actual_site_id','accepted_mapping']],
        on='source_location_label', how='left', validate='many_to_one'
    )
    out = pd.DataFrame({
        'source_system': 'MER_GERMANY_HISTORICAL',
        'source_event_id': 'MER_ROW_' + merge['source_row_number'].astype(str),
        'source_location_label': merge['source_location_label'],
        'site_id': merge['actual_site_id'].fillna(''),
        'start_time_source': merge['start_time'],
        'end_time_source': merge['end_time'],
        'duration_min': merge['charging_duration_min'],
        'delivered_energy_kwh': merge['energy_delivered_kwh'],
        'session_average_power_kw': merge['observed_mean_power_kw'],
        'charger_id': '',
        'connector_id': '',
        'meter_check_status': merge['meter_check_status'],
        'site_identity_status': np.where(merge['accepted_mapping'].fillna(False),
            'MAPPED_WITH_DOCUMENTED_EVIDENCE','SOURCE_LABEL_NOT_VERIFIED_AS_SITE'),
        'timezone_status': merge['time_zone_status'],
        'observed_record_type': 'SUCCESSFUL_CHARGING_START',
        'includes_unsuccessful_arrivals': False,
        'captures_waiting_queue': False,
        'captures_lost_demand': False,
    })
    return out


def read_operator(path: Path) -> pd.DataFrame:
    """Operator-provided site IDs are retained, never independently invented."""
    d = pd.read_csv(path, dtype='string', keep_default_na=False)
    require_columns(d, OPERATOR_TEMPLATE_COLUMNS[:5], path.name)
    if d.empty:
        raise ValueError('Operator input is empty. See data/config/cpo_operator_sessions_template.csv.')
    if d['session_id'].str.strip().eq('').any() or d['session_id'].duplicated().any():
        raise ValueError('Each operator session requires a unique nonblank session_id.')
    if d['site_id'].str.strip().eq('').any():
        raise ValueError('Operator site_id is missing; do not invent one.')
    def optional(name: str) -> pd.Series:
        return d[name] if name in d.columns else pd.Series('', index=d.index)
    sources = optional('source_system').str.strip().replace('', 'CPO_OPERATOR_UPLOAD')
    tz = optional('site_timezone').str.strip()
    return pd.DataFrame({
        'source_system': sources,
        'source_event_id': d['session_id'],
        'source_location_label': d['site_id'],
        'site_id': d['site_id'],
        'start_time_source': d['start_time'],
        'end_time_source': d['end_time'],
        'duration_min': np.nan,
        'delivered_energy_kwh': d['energy_delivered_kwh'],
        'session_average_power_kw': np.nan,
        'charger_id': optional('charger_id'),
        'connector_id': optional('connector_id'),
        'meter_check_status': 'OPERATOR_REPORTED_UNVERIFIED_METER',
        'site_identity_status': 'OPERATOR_REPORTED_SITE_ID',
        'timezone_status': np.where(tz.ne(''), 'TIMEZONE_REPORTED', 'TIMEZONE_UNKNOWN'),
        'observed_record_type': 'SUCCESSFUL_CHARGING_START',
        'includes_unsuccessful_arrivals': False,
        'captures_waiting_queue': False,
        'captures_lost_demand': False,
    })


def validate_and_write(out: pd.DataFrame, out_dir: Path, mode: str) -> None:
    """Reject problematic rows, do not repair values with synthetic replacements."""
    start = pd.to_datetime(out['start_time_source'], errors='coerce', utc=False)
    end = pd.to_datetime(out['end_time_source'], errors='coerce', utc=False)
    recorded_duration = pd.to_numeric(out['duration_min'], errors='coerce')
    delta = (end - start).dt.total_seconds() / 60
    energy = pd.to_numeric(out['delivered_energy_kwh'], errors='coerce')
    duration = recorded_duration.where(recorded_duration.notna(), delta)
    implied = energy * 60 / duration
    valid = (
        start.notna() & end.notna() & start.lt(end)
        & energy.gt(0) & np.isfinite(energy)
        & duration.gt(0) & np.isfinite(duration)
        & delta.gt(0) & (recorded_duration.isna() | (delta - duration).abs().le(2))
    )
    out = out.copy()
    out['data_quality_status'] = np.where(valid, 'VALID_OBSERVED_SESSION', 'REVIEW_EXCLUDED')
    out['duration_min'] = duration
    out['delivered_energy_kwh'] = energy
    out['session_average_power_kw'] = implied
    out = out.loc[:, CANONICAL_COLUMNS]
    accepted = out[valid].copy()
    rejected = out[~valid].copy()
    if accepted.empty:
        raise ValueError('No sessions passed validity checks; review the source and timestamps.')
    # Session records are not an arrival-attempt log. We intentionally do not
    # infer occupancy queues, lost cars or demand from successful starts.
    accepted.sort_values(['source_location_label','start_time_source','source_event_id'], inplace=True)
    output = out_dir / 'cpo_real_observed_sessions.csv'
    rejected_output = out_dir / 'cpo_real_session_quality_exceptions.csv'
    summary_path = out_dir / 'cpo_real_session_import_readiness.csv'
    accepted.to_csv(output, index=False)
    rejected.to_csv(rejected_output, index=False)

    summary = accepted.groupby('source_location_label', dropna=False).agg(
        observed_charging_sessions=('source_event_id','size'),
        first_recorded_start=('start_time_source','min'),
        last_recorded_start=('start_time_source','max'),
        meter_qualified_count=('meter_check_status', lambda s: int(s.eq('REPORTED_AND_METER_AGREE').sum())),
        site_identity_status=('site_identity_status','first'),
        timezone_status=('timezone_status','first'),
    ).reset_index()
    summary['session_behaviour_available'] = True
    summary['full_vehicle_arrival_demand_known'] = False
    summary['queue_and_lost_customers_known'] = False
    summary['grid_connection_known'] = False
    summary['ready_for_real_investment_decision'] = False
    summary['next_missing_evidence'] = (
        'Verified complete site/charger mapping, observation completeness, '
        'grid and hardware data, arrivals/queues, and commercial costs'
    )
    summary.to_csv(summary_path, index=False)
    print('\n--- REAL CPO SESSION INPUT ADAPTER ---')
    print(f'Input type: {mode}')
    print(f'Valid observed charging sessions: {len(accepted):,}')
    print(f'Rows needing further quality review: {len(rejected):,}')
    print(f'Source location labels: {summary.shape[0]}')
    print(f'Sessions with a site ID supplied or documented: {int(accepted.site_id.fillna("").ne("").sum()):,}')
    print('Site ID status: Mer labels require documented mapping; operator IDs are operator-reported.')
    print('Full vehicle arrivals, queuing, rejected drivers: NOT OBSERVED')
    print('Investment decision: BLOCKED PENDING SITE / GRID / FINANCIAL EVIDENCE')
    print('\nSaved:')
    print(output)
    print(rejected_output)
    print(summary_path)
    print('No synthetic arrivals, site identities, charger power or financial figures were generated.')


def main() -> None:
    parser = argparse.ArgumentParser(description='CPO real charging-session intake without invented demand.')
    parser.add_argument('--mer-file', type=Path, default=DEFAULT_MER)
    parser.add_argument('--operator-csv', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--config-dir', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    operator_template = args.config_dir / 'cpo_operator_sessions_template.csv'
    if not operator_template.exists():
        pd.DataFrame(columns=OPERATOR_TEMPLATE_COLUMNS).to_csv(operator_template, index=False)
    if args.operator_csv:
        if not args.operator_csv.exists():
            raise FileNotFoundError(args.operator_csv)
        result = read_operator(args.operator_csv)
        mode = 'OPERATOR_SESSIONS'
    else:
        if not args.mer_file.exists():
            raise FileNotFoundError(f'{args.mer_file}. Run build_mer_cpo_session_input.py first.')
        result = read_mer(args.mer_file, args.config_dir)
        mode = 'SCREENED_MER_HISTORICAL_CCS'
    validate_and_write(result, args.output_dir, mode)


if __name__ == '__main__':
    main()
