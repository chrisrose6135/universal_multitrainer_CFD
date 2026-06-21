#!/usr/bin/env python3
"""Combine per-symbol direction-training CSVs into a universal training dataset.

The output remains compatible with the existing direction trainer, but adds:

* ``symbol`` and one-hot ``sym_*`` context features,
* ``universal_sequence_group`` so sequence windows cannot cross symbols,
* ``universal_split`` so training/validation splits are made per symbol.

This is intentionally an add-on script. Normal symbol-specific datasets and
configs are not changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config_with_optional_spread_risk
from .forex import validate_forex_symbols
from .universal_symbol_features import add_universal_symbol_features, append_universal_symbol_feature_columns


def _timeframe(cfg: dict[str, Any]) -> str:
    return str((cfg.get('trading') or {}).get('timeframe') or (cfg.get('project') or {}).get('timeframe') or 'M5').upper()


def _dataset_path(symbol: str, cfg: dict[str, Any], *, data_root: str | None, template: str | None) -> Path:
    data_cfg = cfg.get('data', {}) or {}
    train_cfg = cfg.get('training', {}) or {}
    root = Path(data_root or train_cfg.get('direction_data_dir') or data_cfg.get('direction_data_dir') or 'data/direction')
    tmpl = template or train_cfg.get('direction_data_template') or data_cfg.get('direction_data_template') or '{symbol}_{timeframe}_direction_training.csv'
    return root / str(tmpl).format(symbol=symbol, timeframe=_timeframe(cfg))


def _time_col(df: pd.DataFrame) -> str | None:
    for c in ('time_utc', 'time', 'datetime', 'date'):
        if c in df.columns:
            return c
    return None


def _filter_date(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if not start and not end:
        return df.reset_index(drop=True)
    col = _time_col(df)
    if col is None:
        return df.reset_index(drop=True)
    t = pd.to_datetime(df[col], utc=True, errors='coerce')
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= t >= pd.to_datetime(start, utc=True)
    if end:
        mask &= t < pd.to_datetime(end, utc=True)
    return df.loc[mask].reset_index(drop=True)


def _sort_time(df: pd.DataFrame) -> pd.DataFrame:
    col = _time_col(df)
    if col is None:
        return df.reset_index(drop=True)
    out = df.copy()
    out['_universal_sort_time'] = pd.to_datetime(out[col], utc=True, errors='coerce')
    out = out.sort_values('_universal_sort_time', kind='mergesort').drop(columns=['_universal_sort_time'])
    return out.reset_index(drop=True)


def _sample_per_symbol(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    # Keep chronological order after the random sample.
    return _sort_time(df.sample(n=int(max_rows), random_state=int(seed)))


def _assign_split(df: pd.DataFrame, *, cfg: dict[str, Any], val_fraction: float, min_validation_rows: int) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    ucfg = cfg.get('universal', {}) or {}
    split_col = str(ucfg.get('split_column', 'universal_split'))
    group_col = str(ucfg.get('sequence_group_column', 'universal_sequence_group'))
    train_label = str(ucfg.get('train_split_label', 'train'))
    val_label = str(ucfg.get('validation_split_label', 'validation'))
    n = len(out)
    val_rows = int(round(n * float(val_fraction)))
    if min_validation_rows > 0:
        val_rows = max(val_rows, int(min_validation_rows))
    val_rows = min(max(val_rows, 1), max(n - 1, 1)) if n > 1 else 0
    split = [train_label] * n
    if val_rows > 0:
        for i in range(n - val_rows, n):
            split[i] = val_label
    out[split_col] = split
    # Every symbol segment is one sequence group.  prepare_direction_arrays()
    # uses this to reject windows crossing symbol boundaries.
    sym = str(out['symbol'].iloc[0]) if n else ''
    out[group_col] = sym
    return out


def _target_counts(df: pd.DataFrame) -> dict[str, int]:
    if 'direction_target' not in df.columns:
        return {}
    vals = pd.to_numeric(df['direction_target'], errors='coerce').fillna(-999).astype(int)
    return {str(k): int(v) for k, v in vals.value_counts(dropna=False).sort_index().items()}


def _split_counts(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, int]:
    col = str((cfg.get('universal', {}) or {}).get('split_column', 'universal_split'))
    if col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].astype(str).value_counts().sort_index().items()}


def main() -> None:
    ap = argparse.ArgumentParser(description='Combine per-symbol direction datasets into one universal training CSV.')
    ap.add_argument('--config', default='config/direction_settings_universal_models.yaml')
    ap.add_argument('--symbols', nargs='*', default=None)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--template', default=None)
    ap.add_argument('--output', default=None, help='Combined CSV path. Defaults to universal.combined_dataset_path.')
    ap.add_argument('--date-start', default=None)
    ap.add_argument('--date-end', default=None)
    ap.add_argument('--max-rows-per-symbol', type=int, default=None)
    ap.add_argument('--validation-fraction', type=float, default=None, help='Per-symbol validation fraction. Defaults to universal.validation_fraction or training.val_fraction.')
    ap.add_argument('--min-validation-rows-per-symbol', type=int, default=None)
    ap.add_argument('--shuffle-symbol-order', action=argparse.BooleanOptionalAction, default=False, help='Shuffle symbol block order only. Rows within each symbol remain chronological.')
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    cfg = load_config_with_optional_spread_risk(args.config)
    cfg.setdefault('universal', {})['enabled'] = True
    cfg = append_universal_symbol_feature_columns(cfg)
    ucfg = cfg.get('universal', {}) or {}
    symbols = args.symbols or ucfg.get('symbols') or (cfg.get('trading') or {}).get('symbols') or []
    symbols = validate_forex_symbols(symbols)
    if args.shuffle_symbol_order:
        import random
        rng = random.Random(int(args.seed))
        symbols = list(symbols)
        rng.shuffle(symbols)

    val_fraction = float(args.validation_fraction if args.validation_fraction is not None else ucfg.get('validation_fraction', (cfg.get('training', {}) or {}).get('val_fraction', 0.2)))
    min_val = int(args.min_validation_rows_per_symbol if args.min_validation_rows_per_symbol is not None else ucfg.get('min_validation_rows_per_symbol', 500))
    output = Path(args.output or ucfg.get('combined_dataset_path') or f'data/universal/UNIVERSAL_{_timeframe(cfg)}_direction_training.csv')
    if output.exists() and not args.force:
        raise SystemExit(f'Output already exists: {output}. Pass --force to overwrite.')

    frames: list[pd.DataFrame] = []
    summary: dict[str, Any] = {
        'created_by': 'combine_universal_direction_datasets.py',
        'config': str(args.config),
        'timeframe': _timeframe(cfg),
        'symbols': symbols,
        'output': str(output),
        'validation_fraction': val_fraction,
        'min_validation_rows_per_symbol': min_val,
        'rows_by_symbol': {},
        'target_counts_by_symbol': {},
        'split_counts_by_symbol': {},
        'notes': [
            'Rows are kept chronological within each symbol.',
            'universal_sequence_group prevents sequence windows crossing symbol boundaries.',
            'universal_split provides per-symbol train/validation split labels.',
        ],
    }
    for i, symbol in enumerate(symbols):
        path = _dataset_path(symbol, cfg, data_root=args.data_root, template=args.template)
        if not path.exists():
            raise FileNotFoundError(f'Missing direction dataset for {symbol}: {path}')
        df = pd.read_csv(path)
        before = int(len(df))
        df = _filter_date(df, args.date_start, args.date_end)
        df = _sort_time(df)
        df = _sample_per_symbol(df, args.max_rows_per_symbol, args.seed + i)
        df['symbol'] = symbol
        df = _assign_split(df, cfg=cfg, val_fraction=val_fraction, min_validation_rows=min_val)
        df = add_universal_symbol_features(df, cfg, symbol=symbol)
        frames.append(df)
        summary['rows_by_symbol'][symbol] = {'raw': before, 'used': int(len(df))}
        summary['target_counts_by_symbol'][symbol] = _target_counts(df)
        summary['split_counts_by_symbol'][symbol] = _split_counts(df, cfg)
        print(f'{symbol}: loaded {before:,} row(s), using {len(df):,} row(s) from {path}')

    combined = pd.concat(frames, ignore_index=True, sort=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    summary['rows_total'] = int(len(combined))
    summary['columns_total'] = int(len(combined.columns))
    summary['target_counts_total'] = _target_counts(combined)
    summary['split_counts_total'] = _split_counts(combined, cfg)
    summary['symbol_feature_columns'] = [c for c in combined.columns if c.startswith(str(ucfg.get('symbol_feature_prefix', 'sym_')))]
    meta_path = output.with_suffix('.summary.json')
    meta_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Wrote combined universal dataset: {output} ({len(combined):,} rows)')
    print(f'Wrote dataset summary: {meta_path}')


if __name__ == '__main__':
    main()
