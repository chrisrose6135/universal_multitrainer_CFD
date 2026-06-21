#!/usr/bin/env python3
"""Replay universal models on each symbol and build a live manifest.

Each universal model is trained once on a pooled dataset, then replayed separately
for every symbol so performance can be judged per symbol and per architecture.
The output live manifest repeats the universal model artifact paths for each
symbol where the model passed the optional filters.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .config import load_config_with_optional_spread_risk
from .forex import validate_forex_symbols
from .io_utils import ensure_dir, write_json
from .test_saved_direction_policy import replay_symbol
from .train_direction_policy import _json_safe
from .universal_symbol_features import append_universal_symbol_feature_columns


def _read_structured(path: Path) -> Any:
    if path.suffix.lower() in {'.yaml', '.yml'}:
        return yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return json.loads(path.read_text(encoding='utf-8'))


def _timeframe(cfg: dict[str, Any]) -> str:
    return str((cfg.get('trading') or {}).get('timeframe') or (cfg.get('project') or {}).get('timeframe') or 'M5').upper()


def _side_allows(side: str, replay_cfg: dict[str, Any]) -> tuple[bool, bool]:
    side = str(side or 'both').lower()
    if side == 'buy':
        return True, False
    if side == 'sell':
        return False, True
    return True, True


def _passes_filters(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    reasons: list[str] = []
    def f(name: str, default: float = 0.0) -> float:
        try:
            return float(row.get(name, default) or 0.0)
        except Exception:
            return default
    if f('trades') < float(args.min_trades):
        reasons.append(f'trades<{args.min_trades:g}')
    if f('net_pips') < float(args.min_net_pips):
        reasons.append(f'net_pips<{args.min_net_pips:g}')
    if f('average_net_pips') < float(args.min_average_net_pips):
        reasons.append(f'average_net_pips<{args.min_average_net_pips:g}')
    if args.min_win_rate is not None and f('win_rate') < float(args.min_win_rate):
        reasons.append(f'win_rate<{args.min_win_rate:g}')
    if args.max_drawdown_pips is not None and f('max_drawdown_pips') > float(args.max_drawdown_pips):
        reasons.append(f'drawdown>{args.max_drawdown_pips:g}')
    if args.max_drawdown_to_net_ratio is not None:
        net = max(f('net_pips'), 1e-9)
        ratio = f('max_drawdown_pips') / net
        if ratio > float(args.max_drawdown_to_net_ratio):
            reasons.append(f'drawdown_to_net>{args.max_drawdown_to_net_ratio:g}')
    return not reasons, ';'.join(reasons)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description='Replay universal models per symbol and create a universal live manifest.')
    ap.add_argument('--config', default='config/direction_settings_universal_models.yaml')
    ap.add_argument('--manifest', default='models/universal/universal_model_manifest.json')
    ap.add_argument('--symbols', nargs='*', default=None)
    ap.add_argument('--replay-start', default=None)
    ap.add_argument('--replay-end', default=None)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--save-decisions', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--save-trades', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--min-trades', type=float, default=50)
    ap.add_argument('--min-net-pips', type=float, default=0.0)
    ap.add_argument('--min-average-net-pips', type=float, default=0.0)
    ap.add_argument('--min-win-rate', type=float, default=None)
    ap.add_argument('--max-drawdown-pips', type=float, default=None)
    ap.add_argument('--max-drawdown-to-net-ratio', type=float, default=None)
    args = ap.parse_args()

    cfg = load_config_with_optional_spread_risk(args.config)
    cfg = append_universal_symbol_feature_columns(cfg)
    ucfg = cfg.get('universal', {}) or {}
    manifest_path = Path(args.manifest)
    manifest = _read_structured(manifest_path)
    models = [m for m in manifest.get('models', []) if isinstance(m, dict)]
    symbols = validate_forex_symbols(args.symbols or ucfg.get('symbols') or (cfg.get('trading') or {}).get('symbols') or [])
    output_dir = Path(args.output_dir or ucfg.get('replay_output_dir') or 'logs/universal_replay')
    ensure_dir(output_dir)

    rows: list[dict[str, Any]] = []
    live_models: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []
    for model_entry in models:
        token = str(model_entry.get('model') or model_entry.get('architecture') or 'model')
        side = str(model_entry.get('side') or model_entry.get('train_side') or 'both').lower()
        for symbol in symbols:
            rcfg = copy.deepcopy(cfg)
            rcfg['_active_symbol'] = symbol
            rcfg.setdefault('trading', {})['symbols'] = [symbol]
            rcfg.setdefault('model', {})['architecture'] = model_entry.get('architecture') or rcfg.get('model', {}).get('architecture')
            allow_buy, allow_sell = _side_allows(side, rcfg.get('replay', {}) or {})
            rcfg.setdefault('replay', {})['allow_buy'] = allow_buy
            rcfg.setdefault('replay', {})['allow_sell'] = allow_sell
            rcfg.setdefault('training', {})['train_side'] = side
            prefix = output_dir / symbol / f'{symbol}_{_timeframe(rcfg)}_{token}_{side}_universal_replay'
            print(f'Replaying universal {token} {side} on {symbol}...')
            summary = replay_symbol(
                symbol,
                rcfg,
                model_path=Path(str(model_entry['model_path'])),
                scaler_path=Path(str(model_entry['scaler_path'])),
                features_path=Path(str(model_entry['features_path'])),
                eval_start=args.replay_start,
                eval_end=args.replay_end,
                output_prefix=str(prefix),
                device=args.device,
                verbose=False,
            )
            row = {
                'id': f'{symbol}_universal_{token}_{side}',
                'symbol': symbol,
                'model': token,
                'architecture': model_entry.get('architecture'),
                'side': side,
                'timeframe': _timeframe(rcfg),
                'trades': summary.get('trades'),
                'net_pips': summary.get('net_pips'),
                'win_rate': summary.get('win_rate'),
                'average_net_pips': summary.get('average_net_pips'),
                'max_drawdown_pips': summary.get('max_drawdown_pips'),
                'buy_trades': summary.get('buy_trades'),
                'buy_net_pips': summary.get('buy_net_pips'),
                'sell_trades': summary.get('sell_trades'),
                'sell_net_pips': summary.get('sell_net_pips'),
                'replay_score': summary.get('replay_score'),
                'summary_path': summary.get('summary_path'),
                'trades_path': summary.get('trades_path') if args.save_trades else None,
                'decisions_path': summary.get('decisions_path') if args.save_decisions else None,
                'model_path': model_entry.get('model_path'),
                'scaler_path': model_entry.get('scaler_path'),
                'features_path': model_entry.get('features_path'),
                'config_path': model_entry.get('config_path'),
            }
            keep, reason = _passes_filters(row, args)
            row['enabled'] = bool(keep)
            row['disabled_reason'] = reason
            rows.append(row)
            live_entry = {
                'id': row['id'],
                'symbol': symbol,
                'model': f'universal_{token}',
                'architecture': model_entry.get('architecture'),
                'side': side,
                'timeframe': _timeframe(rcfg),
                'config_path': model_entry.get('config_path'),
                'model_path': model_entry.get('model_path'),
                'scaler_path': model_entry.get('scaler_path'),
                'features_path': model_entry.get('features_path'),
                'trades': row['trades'],
                'net_pips': row['net_pips'],
                'win_rate': row['win_rate'],
                'average_net_pips': row['average_net_pips'],
                'max_drawdown_pips': row['max_drawdown_pips'],
                'enabled': bool(keep),
            }
            if keep:
                live_models.append(live_entry)
            else:
                disabled.append({**live_entry, 'disabled_reason': reason})

    _write_csv(output_dir / 'universal_replay_by_symbol_model.csv', rows)
    _write_csv(output_dir / 'universal_live_models_enabled.csv', live_models)
    _write_csv(output_dir / 'universal_live_models_disabled.csv', disabled)
    live_manifest = {
        'created_by': 'replay_universal_direction_models.py',
        'config_path': str(args.config),
        'source_universal_model_manifest': str(manifest_path),
        'live_root': str(output_dir / 'For Live Trading Universal'),
        'universal_config_path': str(args.config),
        'replay_start': args.replay_start,
        'replay_end': args.replay_end,
        'filter': {
            'min_trades': args.min_trades,
            'min_net_pips': args.min_net_pips,
            'min_average_net_pips': args.min_average_net_pips,
            'min_win_rate': args.min_win_rate,
            'max_drawdown_pips': args.max_drawdown_pips,
            'max_drawdown_to_net_ratio': args.max_drawdown_to_net_ratio,
        },
        'models': live_models,
        'disabled_models': disabled,
    }
    write_json(output_dir / 'universal_live_ensemble_manifest.json', _json_safe(live_manifest))
    print(f'Wrote replay table: {output_dir / "universal_replay_by_symbol_model.csv"}')
    print(f'Wrote universal live manifest: {output_dir / "universal_live_ensemble_manifest.json"}')
    print(f'Enabled universal model-symbol entries: {len(live_models)}; disabled: {len(disabled)}')


if __name__ == '__main__':
    main()
