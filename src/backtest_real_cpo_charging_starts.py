"""Backtest REAL observed charging-start counts, without claiming total arrivals.

Inputs: screened, standardized successful charging sessions from
    python src/prepare_cpo_real_session_adapter.py

Run from project root:
    python src/backtest_real_cpo_charging_starts.py

This backtest never fabricates failed arrivals, site identities, infrastructure,
operating uptime, or energy demand. Hourly predictions describe REPORTED
SUCCESSFUL STARTS, not all cars attempting to charge. Unobserved hours are
explicitly conditional on assumed complete session logging, NOT confirmed zeros.
The final 14 days are held out; models are selected using the preceding 14 days.

Outputs are new CSVs under data/processed. Existing simulation outputs stay intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'data' / 'processed' / 'cpo_real_observed_sessions.csv'
OUT = ROOT / 'data' / 'processed'
MIN_TRAIN_STARTS = 80
MIN_VALIDATE_STARTS = 8
MIN_TEST_STARTS = 8
MIN_ACTIVE_DAYS_TRAIN = 20
MIN_ACTIVE_DAYS_VALIDATION = 5
MIN_ACTIVE_DAYS_TEST = 5
TRAIN_DAYS = 56
VALIDATE_DAYS = 14
TEST_DAYS = 14


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Input not found: {path}\nRun prepare_cpo_real_session_adapter.py first.')
    d = pd.read_csv(path, low_memory=False)
    required = {'source_location_label', 'source_event_id', 'start_time_source',
                'observed_record_type', 'data_quality_status'}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if d.empty:
        raise ValueError('No observed charging records found.')
    good = (
        d['observed_record_type'].eq('SUCCESSFUL_CHARGING_START')
        & d['data_quality_status'].eq('VALID_OBSERVED_SESSION')
    )
    d = d.loc[good].copy()
    if not len(d):
        raise ValueError('No valid successful charging starts found.')
    if d['source_event_id'].isna().any() or d['source_event_id'].duplicated().any():
        raise ValueError('Missing or duplicated event IDs; cannot count unique starts reliably.')
    d['source_location_label'] = d['source_location_label'].astype('string').str.strip()
    if d['source_location_label'].isna().any() or d['source_location_label'].eq('').any():
        raise ValueError('Blank source labels found. Source identity cannot be invented.')
    # Historical Mer timestamps are local but their timezone and DST handling are
    # not validated; no timezone conversion or synthetic adjustment is performed.
    d['start_time'] = pd.to_datetime(d['start_time_source'], errors='coerce')
    if d['start_time'].isna().any():
        raise ValueError('Some accepted session starts have invalid timestamps.')
    if d['start_time'].dt.tz is not None:
        # UTC-aware sources require separate local timezone handling.
        raise ValueError('Timezone-aware timestamps require source-specific local time handling.')
    if 'site_id' not in d:
        d['site_id'] = ''
    d['site_id'] = d['site_id'].fillna('').astype(str).str.strip()
    if 'source_system' not in d:
        d['source_system'] = 'UNKNOWN'
    return d


def make_panel(events: pd.DataFrame) -> pd.DataFrame:
    latest = events['start_time'].max().normalize()
    start = latest - pd.Timedelta(days=TRAIN_DAYS + VALIDATE_DAYS + TEST_DAYS - 1)
    # Every hour is an *observed-event count*; zeros inside the window are not
    # confirmed zero demand because site operating/reporting completeness is unknown.
    grid = pd.date_range(start, latest + pd.Timedelta(hours=23), freq='h')
    panel = pd.DataFrame({'hour_start': grid})
    counts = events.groupby(events['start_time'].dt.floor('h')).size()
    panel['reported_starts'] = panel['hour_start'].map(counts).fillna(0).astype(int)
    panel['weekday'] = panel['hour_start'].dt.weekday
    panel['hour'] = panel['hour_start'].dt.hour
    panel['weekend'] = (panel['weekday'] >= 5).astype(int)
    end_train = start + pd.Timedelta(days=TRAIN_DAYS)
    end_validate = end_train + pd.Timedelta(days=VALIDATE_DAYS)
    panel['split'] = np.where(panel['hour_start'] < end_train, 'TRAIN',
                              np.where(panel['hour_start'] < end_validate, 'VALIDATION', 'UNSEEN_TEST'))
    return panel


def predictions(train: pd.DataFrame, target: pd.DataFrame) -> dict[str, np.ndarray]:
    mu = float(train['reported_starts'].mean())
    baseline = np.full(len(target), mu)
    # Regularized calendar profiles: shrink sparse bins to global hourly mean.
    # All means are computed exclusively using available past training data.
    hour_table = train.groupby('hour')['reported_starts'].agg(['sum', 'count'])
    weekend_table = train.groupby(['weekend', 'hour'])['reported_starts'].agg(['sum','count'])
    dow_table = train.groupby(['weekday', 'hour'])['reported_starts'].agg(['sum','count'])

    def lookup(table, keys, prior):
        idx = pd.MultiIndex.from_frame(target[keys]) if len(keys) > 1 else pd.Index(target[keys[0]])
        row = table.reindex(idx)
        return (row['sum'].fillna(0).to_numpy(dtype=float) + prior * mu) / (
            row['count'].fillna(0).to_numpy(dtype=float) + prior)

    return {
        'MEAN_HOURLY_BASELINE': baseline,
        'HOUR_OF_DAY': lookup(hour_table, ['hour'], 12),
        'HOUR_WEEKEND': lookup(weekend_table, ['weekend', 'hour'], 8),
        'HOUR_OF_WEEK': lookup(dow_table, ['weekday','hour'], 4),
    }


def mae(y, yp):
    return float(np.abs(np.asarray(y, dtype=float) - np.asarray(yp, dtype=float)).mean())


def summarize_split(p: pd.DataFrame, split: str) -> tuple[int, int, int]:
    s = p[p['split'] == split]
    total = int(s['reported_starts'].sum())
    active_days = int(s.loc[s['reported_starts'] > 0, 'hour_start'].dt.normalize().nunique())
    return total, active_days, len(s)


def backtest_one(label: str, events: pd.DataFrame):
    panel = make_panel(events)
    train = panel[panel['split'] == 'TRAIN'].copy()
    validate = panel[panel['split'] == 'VALIDATION'].copy()
    test = panel[panel['split'] == 'UNSEEN_TEST'].copy()
    tr, atr, _ = summarize_split(panel, 'TRAIN')
    va, ava, _ = summarize_split(panel, 'VALIDATION')
    te, ate, _ = summarize_split(panel, 'UNSEEN_TEST')
    row = {
        'source_location_label': label,
        'source_system': ';'.join(sorted(events['source_system'].dropna().astype(str).unique())),
        'site_id': ';'.join(sorted({x for x in events['site_id'].astype(str) if x})),
        'source_total_successful_sessions': len(events),
        'train_starts': tr, 'validation_starts': va, 'unseen_test_starts': te,
        'train_days_with_recorded_starts': atr,
        'validation_days_with_recorded_starts': ava,
        'unseen_test_days_with_recorded_starts': ate,
        'unseen_test_start_date': str(test['hour_start'].min().date()),
        'unseen_test_end_date': str(test['hour_start'].max().date()),
        'observation_completeness': 'UNKNOWN_NO_UPTIME_TELEMETRY',
        'target_definition': 'SUCCESSFUL_CHARGING_STARTS_ONLY',
        'validation_status': '',
    }
    sufficient = (tr >= MIN_TRAIN_STARTS and va >= MIN_VALIDATE_STARTS and te >= MIN_TEST_STARTS
                  and atr >= MIN_ACTIVE_DAYS_TRAIN and ava >= MIN_ACTIVE_DAYS_VALIDATION
                  and ate >= MIN_ACTIVE_DAYS_TEST)
    if not sufficient:
        row['validation_status'] = 'INSUFFICIENT_CONTINUOUS_RECORDED_HISTORY'
        return row, []
    pred_val = predictions(train, validate)
    scores = {name: mae(validate['reported_starts'], yp) for name, yp in pred_val.items()}
    chosen = min(scores, key=lambda name: (scores[name], name != 'MEAN_HOURLY_BASELINE'))
    # Validation period is now history before the untouched test period.
    train_plus_validation = panel[panel['split'] != 'UNSEEN_TEST'].copy()
    pred_test = predictions(train_plus_validation, test)
    baseline = pred_test['MEAN_HOURLY_BASELINE']
    selected = pred_test[chosen]
    actual = test['reported_starts'].to_numpy(dtype=float)
    base_mae = mae(actual, baseline)
    chosen_mae = mae(actual, selected)
    row.update({
        'selected_model_validation_only': chosen,
        'validation_selected_mae_starts_per_hour': round(scores[chosen], 4),
        'unseen_baseline_mae_starts_per_hour': round(base_mae, 4),
        'unseen_selected_mae_starts_per_hour': round(chosen_mae, 4),
        'unseen_mae_improvement_pct': round((base_mae - chosen_mae) / base_mae * 100, 2)
            if base_mae > 0 else np.nan,
        'unseen_selected_wape_pct': round(np.abs(actual - selected).sum() / max(actual.sum(), 1) * 100, 2),
        'unseen_reported_starts': int(actual.sum()),
        'unseen_predicted_reported_starts': round(float(selected.sum()), 2),
        'validation_status': 'CONDITIONAL_HOLDOUT_REPORTING_CONTINUITY_UNVERIFIED',
    })
    detail = test[['hour_start','weekday','hour','reported_starts']].copy()
    detail.insert(0, 'source_location_label', label)
    detail['selected_model'] = chosen
    detail['baseline_predicted_starts'] = baseline.round(5)
    detail['selected_predicted_starts'] = selected.round(5)
    detail['target_definition'] = 'SUCCESSFUL_CHARGING_STARTS_ONLY'
    detail['observation_completeness'] = 'UNKNOWN_NO_UPTIME_TELEMETRY'
    return row, detail.to_dict('records')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    events = load_events(args.input)
    summaries = []
    predictions_out = []
    for label, group in events.groupby('source_location_label', sort=True):
        row, preds = backtest_one(label, group)
        summaries.append(row)
        predictions_out.extend(preds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    detail = pd.DataFrame(predictions_out)
    spath = args.output_dir / 'cpo_real_starts_holdout_scores.csv'
    dpath = args.output_dir / 'cpo_real_starts_holdout_predictions.csv'
    summary.to_csv(spath, index=False)
    detail.to_csv(dpath, index=False)
    eligible = summary[summary['validation_status'].str.startswith('CONDITIONAL_')]
    print('\n--- REAL CPO SUCCESSFUL CHARGING-START BACKTEST ---')
    print(f'Observed valid successful sessions: {len(events):,}')
    print(f'Source labels: {summary.shape[0]} | Backtested labels: {len(eligible)}')
    print('Train: 56 days | Model selection: next 14 days | Unseen test: final 14 days per label')
    if not eligible.empty:
        cols = ['source_location_label','unseen_test_starts','selected_model_validation_only',
                'unseen_baseline_mae_starts_per_hour','unseen_selected_mae_starts_per_hour',
                'unseen_mae_improvement_pct']
        print(eligible[cols].to_string(index=False))
        print(f'Better than baseline: {(eligible.unseen_selected_mae_starts_per_hour < eligible.unseen_baseline_mae_starts_per_hour).sum()} / {len(eligible)} labels')
    if len(summary) > len(eligible):
        print(f'Insufficient recorded history: {len(summary) - len(eligible)} labels')
    print('\nLIMITATION: Targets are observed SUCCESSFUL charging starts, NOT all arrivals.')
    print('Zero observed starts may mean no customers OR missing reporting; uptime is not verified.')
    print('Anonymous Mer location labels cannot be linked to charger counts or grid capacity.')
    print('These metrics are CONDITIONAL and cannot justify actual CPO expansion decisions.')
    print(f'Saved: {spath}\nSaved: {dpath}')


if __name__ == '__main__':
    main()
