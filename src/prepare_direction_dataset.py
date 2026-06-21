from __future__ import annotations

import argparse
import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .forex import validate_forex_symbols, point_for_symbol, spread_points_to_pips
from .io_utils import ensure_dir, read_processed_csv, write_json
from .targets import generate_direction_targets
from .analytic_signals import ensure_analytic_signal_features
from .spread_risk_config import (
    apply_symbol_spread_profile_to_cfg,
    calculate_spread_profile,
    write_spread_risk_config,
)


_LABEL_KEYS_TO_RECORD = [
    'horizon_bars',
    'take_profit_pips',
    'stop_loss_pips',
    'spread_column',
    'default_spread_points',
    'spread_pips_per_point',
    'min_clean_win_net_pips',
    'min_side_edge_pips',
    'conservative_same_bar_hits',
    'use_live_bid_ask_simulation',
    'entry_on_next_bar_open',
    'same_bar_tp_sl_policy',
    'slippage_pips',
    'max_spread_pips',
    'positive_label_deduplication',
    'max_positive_setups_per_day',
    'discarded_positive_mode',
    'method',
    'label_method',
    'strong_setup',
]


def _configured_recent_years(cfg: dict, cli_recent_years: float | None) -> float | None:
    """Return optional recent-year window as a float.

    Fractional values are allowed, e.g. 0.5 for about six months. The training
    config remains the primary source so the preparer and trainer can share the
    same date window when desired.
    """
    if cli_recent_years is not None:
        return float(cli_recent_years)
    tcfg = cfg.get('training', {}) or {}
    dcfg = cfg.get('data', {}) or {}
    value = tcfg.get('recent_years', dcfg.get('recent_years'))
    return float(value) if value is not None else None


def _first_config_value(*dicts: dict, keys: list[str]) -> Any:
    for source in dicts:
        if not source:
            continue
        for key in keys:
            if key in source and source[key] not in (None, ''):
                return source[key]
    return None


def _configured_date_range(
    cfg: dict,
    cli_date_start: str | None,
    cli_date_end: str | None,
) -> tuple[str | None, str | None, str]:
    """Return optional start/end timestamps for pregeneration filtering.

    CLI values take precedence. Config lookup is deliberately permissive so this
    script can track older config names used by the trainer/backtester.
    """
    if cli_date_start or cli_date_end:
        return cli_date_start, cli_date_end, 'cli'

    tcfg = cfg.get('training', {}) or {}
    dcfg = cfg.get('data', {}) or {}
    vcfg = cfg.get('validation', {}) or {}

    start = _first_config_value(
        tcfg,
        dcfg,
        vcfg,
        keys=[
            'date_start_utc',
            'date_start',
            'start_date_utc',
            'start_date',
            'train_start_utc',
            'train_start_date',
            'training_start_utc',
            'training_start_date',
        ],
    )
    end = _first_config_value(
        tcfg,
        dcfg,
        vcfg,
        keys=[
            'date_end_utc',
            'date_end',
            'end_date_utc',
            'end_date',
            'train_end_utc',
            'train_end_date',
            'training_end_utc',
            'training_end_date',
        ],
    )
    return (str(start) if start is not None else None), (str(end) if end is not None else None), 'config'


def _safe_time_range(df: pd.DataFrame) -> dict[str, Any]:
    if 'time_utc' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['time_utc']):
        return {'has_time_utc': False}
    s = df['time_utc'].dropna()
    if s.empty:
        return {'has_time_utc': True, 'start': None, 'end': None}
    return {'has_time_utc': True, 'start': str(s.min()), 'end': str(s.max())}


def _ensure_time_utc_datetime(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Ensure time_utc is timezone-aware datetime for reliable filtering."""
    if 'time_utc' not in df.columns:
        return df, 'time_utc column is missing'
    if pd.api.types.is_datetime64_any_dtype(df['time_utc']):
        # Normalise to UTC. This handles both tz-aware and tz-naive dtypes.
        out = df.copy()
        out['time_utc'] = pd.to_datetime(out['time_utc'], utc=True, errors='coerce')
        return out, None
    out = df.copy()
    out['time_utc'] = pd.to_datetime(out['time_utc'], utc=True, errors='coerce')
    if out['time_utc'].notna().any():
        return out, 'time_utc was parsed to datetime during filtering'
    return df, 'time_utc is not datetime-like and could not be parsed'


def _parse_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == '':
        return None
    ts = pd.to_datetime(value, utc=True, errors='raise')
    if isinstance(ts, pd.DatetimeIndex):
        if len(ts) != 1:
            raise ValueError(f'Expected one timestamp, got {len(ts)} values for {value!r}')
        return pd.Timestamp(ts[0])
    return pd.Timestamp(ts)


def _filter_date_range(
    df: pd.DataFrame,
    date_start: str | None,
    date_end: str | None,
    source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicit date range using inclusive start and exclusive end."""
    if not date_start and not date_end:
        return df, {'filter_mode': None, 'date_filter_applied': False}

    original_rows = int(len(df))
    df, warning = _ensure_time_utc_datetime(df)
    if warning and 'could not' in warning:
        return df, {
            'filter_mode': 'date_range',
            'date_filter_applied': False,
            'warning': warning,
            'requested_date_start_utc': date_start,
            'requested_date_end_utc': date_end,
        }

    start_ts = _parse_utc_timestamp(date_start)
    end_ts = _parse_utc_timestamp(date_end)
    mask = pd.Series(True, index=df.index)
    if start_ts is not None:
        mask &= df['time_utc'] >= start_ts
    if end_ts is not None:
        mask &= df['time_utc'] < end_ts
    out = df.loc[mask].reset_index(drop=True)

    info = {
        'filter_mode': 'date_range',
        'date_filter_applied': True,
        'date_range_source': source,
        'date_start_utc': str(start_ts) if start_ts is not None else None,
        'date_end_utc': str(end_ts) if end_ts is not None else None,
        'date_end_is_exclusive': True,
        'rows_before_date_filter': original_rows,
        'rows_after_date_filter': int(len(out)),
    }
    if warning:
        info['warning'] = warning
    return out, info


def _filter_recent_years(df: pd.DataFrame, recent_years: float | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only the most recent N years by time_utc.

    recent_years may be fractional. For example, 0.5 keeps roughly the most
    recent six months. Non-positive values disable the filter.
    """
    if recent_years is None:
        return df, {'filter_mode': 'recent_years', 'recent_years': None, 'date_filter_applied': False}
    years = float(recent_years)
    if years <= 0:
        return df, {
            'filter_mode': 'recent_years',
            'recent_years': years,
            'date_filter_applied': False,
            'warning': 'recent_years <= 0 ignored',
        }

    original_rows = int(len(df))
    df, warning = _ensure_time_utc_datetime(df)
    if warning and 'could not' in warning:
        return df, {
            'filter_mode': 'recent_years',
            'recent_years': years,
            'date_filter_applied': False,
            'warning': warning,
        }

    max_time = df['time_utc'].dropna().max()
    if pd.isna(max_time):
        return df, {
            'filter_mode': 'recent_years',
            'recent_years': years,
            'date_filter_applied': False,
            'warning': 'time_utc contains no valid timestamps, so recent-year filtering was skipped',
        }
    cutoff = max_time - pd.Timedelta(days=365.25 * years)
    out = df[df['time_utc'] >= cutoff].reset_index(drop=True)
    info = {
        'filter_mode': 'recent_years',
        'recent_years': years,
        'date_filter_applied': True,
        'cutoff_time_utc': str(cutoff),
        'max_time_utc': str(max_time),
        'rows_before_date_filter': original_rows,
        'rows_after_date_filter': int(len(out)),
    }
    if warning:
        info['warning'] = warning
    return out, info


def _apply_date_filter(
    df: pd.DataFrame,
    *,
    recent_years: float | None,
    date_start: str | None,
    date_end: str | None,
    date_range_source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicit date range if present, otherwise recent_years."""
    if date_start or date_end:
        return _filter_date_range(df, date_start, date_end, date_range_source)
    return _filter_recent_years(df, recent_years)


def _configured_out_dir(cfg: dict, cli_out_dir: str | None) -> Path:
    if cli_out_dir:
        return Path(cli_out_dir)
    tcfg = cfg.get('training', {}) or {}
    dcfg = cfg.get('data', {}) or {}
    value = tcfg.get('direction_data_dir', dcfg.get('direction_data_dir', None))
    if value is None:
        value = 'data/direction'
    return Path(value)


def _configured_template(cfg: dict) -> str:
    tcfg = cfg.get('training', {}) or {}
    dcfg = cfg.get('data', {}) or {}
    value = tcfg.get('direction_data_template', dcfg.get('direction_data_template', None))
    if value is None:
        value = '{symbol}_{timeframe}_direction_training.csv'
    return str(value)

def _configured_workers(cfg: dict, cli_workers: int | None, symbol_count: int) -> int:
    """Return the number of symbol datasets to prepare concurrently.

    CLI takes precedence. Otherwise the config can provide one of:
      data.dataset_prepare_workers
      data.prepare_workers
      training.dataset_prepare_workers
      training.prepare_workers

    If none are set, use a conservative automatic default so a multi-symbol
    config starts preparing more than one dataset at a time without overloading
    smaller machines.
    """
    if symbol_count <= 1:
        return 1

    if cli_workers is not None:
        workers = int(cli_workers)
        if workers < 1:
            raise ValueError('--workers must be >= 1. Use 1 for serial preparation.')
        return min(workers, symbol_count)

    dcfg = cfg.get('data', {}) or {}
    tcfg = cfg.get('training', {}) or {}
    for source in (dcfg, tcfg):
        for key in ('dataset_prepare_workers', 'prepare_workers', 'dataset_workers'):
            if key in source and source[key] not in (None, ''):
                workers = int(source[key])
                if workers < 1:
                    raise ValueError(f'{key} must be >= 1. Use 1 for serial preparation.')
                return min(workers, symbol_count)

    cpu_count = os.cpu_count() or 1
    return max(1, min(symbol_count, cpu_count, 4))





def _symbol_map_value(mapping: Any, symbol: str) -> Any:
    """Return a per-symbol config value using common symbol key variants."""
    if not isinstance(mapping, dict):
        return None
    symbol_u = symbol.upper()
    for key in (symbol, symbol_u, symbol_u.lower()):
        if key in mapping and mapping[key] not in (None, ''):
            return mapping[key]
    return None


def _infer_pip_size(symbol: str, cfg: dict[str, Any]) -> float:
    """Infer pip size for spread/ATR conversion.

    Prefer explicit per-symbol config maps if present. Otherwise use standard CFD branch: prefer trading.pip_size_by_symbol; otherwise fallback to symbol point size.
    """
    for section_name in ('symbols', 'trading', 'data', 'labels', 'risk'):
        section = cfg.get(section_name, {}) or {}
        value = _symbol_map_value(section.get('pip_size_by_symbol'), symbol)
        if value is None:
            value = _symbol_map_value(section.get('pip_sizes'), symbol)
        if value is not None:
            return float(value)
    return 0.01 if 'JPY' in symbol.upper() else 0.0001


def _spread_pips_per_point(cfg: dict[str, Any]) -> float:
    """Return broker-point to pip conversion, defaulting to 0.1 pip/point."""
    for section_name in ('labels', 'risk', 'data', 'trading'):
        section = cfg.get(section_name, {}) or {}
        value = section.get('spread_pips_per_point')
        if value not in (None, ''):
            return float(value)
    value = cfg.get('spread_pips_per_point')
    return float(value) if value not in (None, '') else 0.1


def _fix_spread_atr_units(df: pd.DataFrame, symbol: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Fix spread_atr in already-processed CSVs before labels are generated.

    This keeps pregenerated direction datasets safe even if the processed feature
    CSV was produced before the prepare_mt5_data.py fix. Formula:
        spread_atr = spread_points * point_size / atr_14
    """
    if 'spread_points' not in df.columns or 'atr_14' not in df.columns:
        return df
    out = df.copy()
    spread_points = pd.to_numeric(out['spread_points'], errors='coerce')
    atr_price = pd.to_numeric(out['atr_14'], errors='coerce')
    spread_price = spread_points * point_for_symbol(symbol, cfg)
    ratio = spread_price / atr_price.where(atr_price > 0)
    out['spread_atr'] = ratio.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
    return out


def _max_spread_pips_for_symbol(cfg: dict[str, Any], symbol: str) -> float | None:
    """Return the spread cap used to mark rows as non-tradeable during labelling."""
    labels = cfg.get('labels', {}) or {}
    risk = cfg.get('risk', {}) or {}
    value = _symbol_map_value(labels.get('max_spread_pips_by_symbol'), symbol)
    if value is None:
        value = labels.get('max_spread_pips')
    if value not in (None, ''):
        return float(value)

    # Fall back to a points cap if only risk/labels points are configured.
    points = _symbol_map_value(labels.get('max_spread_points_by_symbol'), symbol)
    if points is None:
        points = _symbol_map_value(risk.get('max_spread_points_by_symbol'), symbol)
    if points is None:
        points = labels.get('max_spread_points', risk.get('max_spread_points'))
    if points not in (None, ''):
        return spread_points_to_pips(symbol, float(points), cfg)
    return None


def _force_high_spread_rows_to_no_trade(
    labelled: pd.DataFrame,
    symbol: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Make rows above the configured spread cap explicitly non-tradeable.

    This is a final safety layer after generate_direction_targets(). It prevents
    rows that live/demo would block on spread from being labelled as tradeable.
    Temporary auxiliary labels are cleared before the final direction dataset is saved.
    """
    max_spread_pips = _max_spread_pips_for_symbol(cfg, symbol)
    info: dict[str, Any] = {
        'enabled': max_spread_pips is not None,
        'max_spread_pips': max_spread_pips,
        'spread_pips_per_point': _spread_pips_per_point(cfg),
        'rows_above_max_spread': 0,
        'trade_rows_forced_to_no_trade': 0,
        'buy_tp_rows_removed': 0,
        'sell_tp_rows_removed': 0,
        'buy_quality_rows_removed': 0,
        'sell_quality_rows_removed': 0,
    }
    if max_spread_pips is None or 'spread_points' not in labelled.columns:
        return labelled, info

    spread_points = pd.to_numeric(labelled['spread_points'], errors='coerce')
    spread_pips = spread_points.map(lambda x: spread_points_to_pips(symbol, float(x), cfg))
    high_spread = spread_pips > float(max_spread_pips)
    n_high = int(high_spread.sum())
    info['rows_above_max_spread'] = n_high
    if n_high == 0:
        return labelled, info

    out = labelled.copy()
    if 'decision_target' in out.columns:
        decision_numeric = pd.to_numeric(out['decision_target'], errors='coerce').fillna(0)
        info['trade_rows_forced_to_no_trade'] = int((high_spread & (decision_numeric >= 0.5)).sum())
        out.loc[high_spread, 'decision_target'] = 0
    elif 'direction_target' in out.columns:
        direction_numeric = pd.to_numeric(out['direction_target'], errors='coerce').fillna(1)
        info['trade_rows_forced_to_no_trade'] = int((high_spread & (direction_numeric != 1)).sum())

    for col, info_key in (
        ('buy_tp_target', 'buy_tp_rows_removed'),
        ('sell_tp_target', 'sell_tp_rows_removed'),
        ('buy_quality_target', 'buy_quality_rows_removed'),
        ('sell_quality_target', 'sell_quality_rows_removed'),
    ):
        if col in out.columns:
            values = pd.to_numeric(out[col], errors='coerce').fillna(0)
            info[info_key] = int((high_spread & (values >= 0.5)).sum())
            out.loc[high_spread, col] = 0

    # Existing convention in the current datasets: direction 0=sell, 1=no trade,
    # 2=buy; outcome 1=no clean winner, 2=winner.
    if 'direction_target' in out.columns:
        # Event-based strong-setup datasets use -1 to mean IGNORE. Do not turn
        # ignored high-spread rows into supervised NO_TRADE endpoints; only force
        # rows that were already supervised labels.
        direction_numeric = pd.to_numeric(out['direction_target'], errors='coerce').fillna(-1)
        out.loc[high_spread & (direction_numeric >= 0), 'direction_target'] = 1
    for col in ('buy_setup_target', 'sell_setup_target'):
        if col in out.columns:
            side_numeric = pd.to_numeric(out[col], errors='coerce').fillna(-1)
            out.loc[high_spread & (side_numeric >= 0), col] = 0
    if 'outcome_target' in out.columns:
        out.loc[high_spread, 'outcome_target'] = 1

    for col in ('selected_side_target', 'side_target', 'trade_side_target'):
        if col in out.columns:
            # These columns may be numeric codes, object strings, pandas StringDtype,
            # or categoricals. Pandas StringDtype rejects integer assignment, which
            # caused the high-spread filter to fail on GER40. Only write the numeric
            # no-trade code for genuinely numeric columns; otherwise cast to object
            # and write the string label used by the rest of the project.
            if pd.api.types.is_numeric_dtype(out[col]):
                out.loc[high_spread, col] = 1
            else:
                out[col] = out[col].astype('object')
                out.loc[high_spread, col] = 'NO_TRADE'

    return out, info

def _label_config_snapshot(cfg: dict) -> dict[str, Any]:
    labels = cfg.get('labels', {}) or {}
    return {key: labels.get(key) for key in _LABEL_KEYS_TO_RECORD if key in labels}


def _counts(series: pd.Series) -> dict[int, int]:
    return {int(k): int(v) for k, v in series.value_counts().sort_index().to_dict().items()}


def _target_distribution(labelled: pd.DataFrame) -> dict[str, Any]:
    rows = int(len(labelled))
    direction_counts = _counts(labelled['direction_target']) if 'direction_target' in labelled else {}
    ignored_rows = int(direction_counts.get(-1, 0))
    sell_rows = int(direction_counts.get(0, 0))
    no_trade_rows = int(direction_counts.get(1, 0))
    buy_rows = int(direction_counts.get(2, 0))
    valid_rows = sell_rows + no_trade_rows + buy_rows
    trade_rows = buy_rows + sell_rows
    return {
        'rows': rows,
        'valid_label_rows': valid_rows,
        'ignored_rows': ignored_rows,
        'sell_rows': sell_rows,
        'no_trade_rows': no_trade_rows,
        'buy_rows': buy_rows,
        'trade_rows': trade_rows,
        'trade_rate': float(trade_rows / valid_rows) if valid_rows else 0.0,
        'raw_trade_rate_including_ignored': float(trade_rows / rows) if rows else 0.0,
        'ignored_rate': float(ignored_rows / rows) if rows else 0.0,
        'buy_sell_ratio': float(buy_rows / sell_rows) if sell_rows else None,
        'no_trade_to_trade_ratio': float(no_trade_rows / trade_rows) if trade_rows else None,
        'direction_counts': direction_counts,
    }


def _strip_unused_target_columns(labelled: pd.DataFrame) -> pd.DataFrame:
    """Keep only the single direction label used by the direction model.

    generate_direction_targets() is used because it already
    implements the live-style BUY/SELL barrier simulation. The temporary auxiliary labels are then removed so the saved dataset is
    a clean direction-policy dataset.
    """
    prefixes = ('buy_', 'sell_')
    explicit = {
        'decision_target', 'outcome_target', 'selected_side_target',
        'side_target', 'trade_side_target', 'candidate_strength_score',
        'label_filter_status',
    }
    keep_target_cols = {
        'direction_target',
        'buy_edge_pips_target', 'sell_edge_pips_target',
        'buy_setup_target', 'sell_setup_target',
        'buy_setup_quality_score_target', 'sell_setup_quality_score_target',
        'buy_setup_analytic_score', 'sell_setup_analytic_score',
    }
    drop_cols = [
        c for c in labelled.columns
        if (
            c not in keep_target_cols
            and (
                c.endswith('_target')
                or c in explicit
                or c.startswith(prefixes)
                or 'candidate_' in str(c).lower()
            )
        )
    ]
    return labelled.drop(columns=drop_cols, errors='ignore')

def _prepare_symbol_dataset(
    *,
    symbol: str,
    cfg: dict[str, Any],
    config_path: str,
    out_dir: Path,
    timeframe: str,
    template: str,
    recent_years: float | None,
    date_start: str | None,
    date_end: str | None,
    date_range_source: str,
    max_rows: int | None,
    fail_if_trade_rate_above: float | None,
) -> dict[str, Any]:
    """Prepare one symbol's dataset and metadata.

    This function is intentionally self-contained so it can safely run in a
    thread for each symbol. It deep-copies the config before applying the
    symbol-specific spread profile so one worker cannot mutate the config seen
    by another worker.
    """
    base_cfg = copy.deepcopy(cfg)
    df = read_processed_csv(symbol, base_cfg)
    df = _fix_spread_atr_units(df, symbol, base_cfg)
    df = ensure_analytic_signal_features(df, base_cfg)
    raw_rows = len(df)
    raw_time_range = _safe_time_range(df)
    df, date_filter_info = _apply_date_filter(
        df,
        recent_years=recent_years,
        date_start=date_start,
        date_end=date_end,
        date_range_source=date_range_source,
    )
    rows_after_date_filter = len(df)
    if max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    spread_profile = calculate_spread_profile(df, symbol, base_cfg)
    symbol_cfg = apply_symbol_spread_profile_to_cfg(copy.deepcopy(base_cfg), spread_profile)
    symbol_label_config = _label_config_snapshot(symbol_cfg)

    labelled = generate_direction_targets(df, symbol, symbol_cfg)
    positive_label_filters = dict(getattr(labelled, 'attrs', {}).get('positive_label_filters') or {})
    labelled, high_spread_no_trade_filter = _force_high_spread_rows_to_no_trade(labelled, symbol, symbol_cfg)
    labelled = _strip_unused_target_columns(labelled)
    distribution = _target_distribution(labelled)
    if fail_if_trade_rate_above is not None and distribution['trade_rate'] > float(fail_if_trade_rate_above):
        raise RuntimeError(
            f'{symbol}: generated trade_rate={distribution["trade_rate"]:.3f}, '
            f'above --fail-if-trade-rate-above={float(fail_if_trade_rate_above):.3f}. '
            'This usually means labels were not tightened as expected.'
        )

    path = Path(out_dir) / template.format(symbol=symbol, timeframe=timeframe)
    labelled.to_csv(path, index=False)
    meta = {
        'symbol': symbol,
        'timeframe': timeframe,
        'source_processed_rows': int(raw_rows),
        'source_time_range': raw_time_range,
        'date_filter': date_filter_info,
        'rows_after_date_filter_before_max_rows': int(rows_after_date_filter),
        'rows_written': int(len(labelled)),
        'recent_years': float(recent_years) if recent_years is not None else None,
        'date_start_utc': date_filter_info.get('date_start_utc'),
        'date_end_utc': date_filter_info.get('date_end_utc'),
        'max_rows': int(max_rows) if max_rows else None,
        'config': str(config_path),
        'path': str(path),
        'label_generation': {
            'target_function': 'generate_direction_targets',
            'label_method': (symbol_cfg.get('labels') or {}).get('method', (symbol_cfg.get('labels') or {}).get('label_method')),
            'labels_config': symbol_label_config,
            'spread_profile': spread_profile,
            'high_spread_no_trade_filter': high_spread_no_trade_filter,
            'positive_label_filters': positive_label_filters,
            'generated_by': 'prepare_direction_dataset.py',
        },
        'required_training_flags': {
            'training.use_pregenerated_direction_data': True,
            'training.direction_data_dir': str(out_dir),
        },
        'target_distribution': distribution,
        # Backward-compatible top-level fields used by older diagnostics.
        'direction_counts': distribution['direction_counts'],
        
    }
    write_json(path.with_suffix('.metadata.json'), meta)

    return {
        'symbol': symbol,
        'path': path,
        'metadata_path': path.with_suffix('.metadata.json'),
        'meta': meta,
        'distribution': distribution,
        'spread_profile': spread_profile,
        'symbol_label_config': symbol_label_config,
    }


def _print_symbol_result(result: dict[str, Any]) -> None:
    symbol = result['symbol']
    path = result['path']
    meta = result['meta']
    distribution = result['distribution']
    spread_profile = result['spread_profile']
    symbol_label_config = result['symbol_label_config']

    print(f'{symbol}: saved {meta["rows_written"]:,} rows to {path}')
    print('  metadata:', result['metadata_path'])
    print('  date filter:', meta['date_filter'])
    print('  label settings:', symbol_label_config)
    print('  spread profile p95:', {
        'median_points': spread_profile['median_spread_points'],
        'p95_points': spread_profile['p95_spread_points'],
        'p95_pips': spread_profile['p95_spread_pips'],
        'used_fallback': spread_profile['used_fallback'],
    })
    high_spread_filter = (meta.get('label_generation') or {}).get('high_spread_no_trade_filter') or {}
    if high_spread_filter.get('enabled'):
        print('  high-spread no-trade filter:', {
            'max_spread_pips': high_spread_filter.get('max_spread_pips'),
            'rows_above_max_spread': high_spread_filter.get('rows_above_max_spread'),
            'trade_rows_forced_to_no_trade': high_spread_filter.get('trade_rows_forced_to_no_trade'),
        })
    print('  direction counts:', meta['direction_counts'])
    print('  trade rate:', f'{distribution["trade_rate"]:.2%}', f'({distribution["trade_rows"]:,}/{distribution["rows"]:,})')
    positive_filters = (meta.get('label_generation') or {}).get('positive_label_filters') or {}
    if positive_filters:
        print('  positive label filters:', positive_filters)
    print('  direction rows:', {
        'IGNORED': distribution.get('ignored_rows', 0),
        'SELL': distribution.get('sell_rows', 0),
        'NO_TRADE': distribution.get('no_trade_rows', 0),
        'BUY': distribution.get('buy_rows', 0),
    })



def main() -> None:
    p = argparse.ArgumentParser(description='Create labelled direction datasets from processed M5 feature CSVs')
    p.add_argument('--config', default='config/direction_settings_generic_multisymbol_31_symbols.yaml')
    p.add_argument('--symbols', nargs='+', default=None)
    p.add_argument('--out-dir', default=None, help='Default: training.direction_data_dir, data.direction_data_dir, then data/direction')
    p.add_argument('--max-rows', type=int, default=None)
    p.add_argument('--recent-years', type=float, default=None, help='Use only the most recent N years of processed data before target generation; fractional values are allowed, e.g. 0.5')
    p.add_argument('--date-start', default=None, help='Inclusive UTC start, e.g. 2024-01-01 or 2024-01-01T00:00:00Z. Overrides recent-years.')
    p.add_argument('--date-end', default=None, help='Exclusive UTC end, e.g. 2025-09-01. Overrides recent-years.')
    p.add_argument('--fail-if-trade-rate-above', type=float, default=None, help='Optional safety check. Example: 0.35 fails if generated ALLOW rate is above 35%.')
    p.add_argument(
        '--workers',
        type=int,
        default=None,
        help=(
            'Number of symbols to prepare concurrently. Defaults to config '
            'data/training.prepare_workers or a conservative auto value up to 4. '
            'Use --workers 1 for serial preparation.'
        ),
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    symbols = validate_forex_symbols(args.symbols or ((cfg.get('trading') or {}).get('symbols') or ['US500']))
    out_dir = ensure_dir(_configured_out_dir(cfg, args.out_dir))
    timeframe = (cfg.get('trading') or {}).get('timeframe', 'M5')
    template = _configured_template(cfg)
    recent_years = _configured_recent_years(cfg, args.recent_years)
    date_start, date_end, date_range_source = _configured_date_range(cfg, args.date_start, args.date_end)
    workers = _configured_workers(cfg, args.workers, len(symbols))

    print(f'Preparing {len(symbols)} symbol dataset(s) with {workers} worker(s)')
    print(f'Output directory: {out_dir}')

    results: dict[str, dict[str, Any]] = {}

    common_kwargs = {
        'cfg': cfg,
        'config_path': str(args.config),
        'out_dir': out_dir,
        'timeframe': timeframe,
        'template': template,
        'recent_years': recent_years,
        'date_start': date_start,
        'date_end': date_end,
        'date_range_source': date_range_source,
        'max_rows': args.max_rows,
        'fail_if_trade_rate_above': args.fail_if_trade_rate_above,
    }

    if workers == 1:
        for symbol in symbols:
            result = _prepare_symbol_dataset(symbol=symbol, **common_kwargs)
            results[symbol] = result
            _print_symbol_result(result)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        future_to_symbol = {
            executor.submit(_prepare_symbol_dataset, symbol=symbol, **common_kwargs): symbol
            for symbol in symbols
        }
        try:
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                except Exception as exc:
                    for pending in future_to_symbol:
                        pending.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(f'{symbol}: dataset preparation failed: {exc}') from exc
                results[symbol] = result
                _print_symbol_result(result)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    # Preserve requested symbol order in the generated risk config. This also
    # guarantees every requested symbol completed successfully before the shared
    # p95 spread risk file is written.
    spread_profiles = {symbol: results[symbol]['spread_profile'] for symbol in symbols}
    risk_config_path = write_spread_risk_config(
        spread_profiles,
        cfg,
        source_config=args.config,
        timeframe=timeframe,
    )
    print(f'Generated p95 spread risk config: {risk_config_path}')


if __name__ == '__main__':
    main()
