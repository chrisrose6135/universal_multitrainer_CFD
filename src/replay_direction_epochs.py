from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config_with_optional_spread_risk
from .forex import validate_forex_symbols
from .io_utils import ensure_dir, write_json
from .test_saved_direction_policy import replay_symbol


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _timeframe(cfg: dict[str, Any]) -> str:
    return str((cfg.get('trading') or {}).get('timeframe') or (cfg.get('project') or {}).get('timeframe') or 'M5').upper()


def _model_dir(cfg: dict[str, Any]) -> Path:
    return Path((cfg.get('paths') or {}).get('model_dir', 'models'))


def _log_dir(cfg: dict[str, Any]) -> Path:
    return Path((cfg.get('paths') or {}).get('log_dir', 'logs'))


def _default_scaler_path(symbol: str, cfg: dict[str, Any]) -> Path:
    return _model_dir(cfg) / f'{symbol}_{_timeframe(cfg)}_direction_scaler.pkl'


def _default_features_path(symbol: str, cfg: dict[str, Any]) -> Path:
    return _model_dir(cfg) / f'{symbol}_{_timeframe(cfg)}_direction_features.json'


def _epoch_dir(symbol: str, cfg: dict[str, Any]) -> Path:
    tcfg = cfg.get('training', {}) or {}
    return _model_dir(cfg) / str(tcfg.get('epoch_model_dir', 'epoch_checkpoints')) / symbol


def _extract_epoch(path: Path) -> int:
    m = re.search(r'_epoch_(\d+)\.pt$', path.name)
    return int(m.group(1)) if m else -1


def _find_epoch_checkpoints(symbol: str, cfg: dict[str, Any]) -> list[Path]:
    tf = _timeframe(cfg)
    root = _epoch_dir(symbol, cfg)
    pattern = f'{symbol}_{tf}_direction_policy_epoch_*.pt'
    return sorted(root.glob(pattern), key=_extract_epoch)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, default=str, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def _best(rows: list[dict[str, Any]], key: str, *, min_trades: int = 0) -> dict[str, Any] | None:
    candidates = [r for r in rows if int(r.get('trades', 0) or 0) >= min_trades]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r.get(key, -1e18) or -1e18))


def replay_epochs_for_symbol(
    symbol: str,
    cfg: dict[str, Any],
    *,
    eval_start: str | None,
    eval_end: str | None,
    output_dir: Path,
    device: str | None,
    verbose: bool = False,
) -> dict[str, Any]:
    symbol_cfg = dict(cfg)
    symbol_cfg['_active_symbol'] = symbol
    checkpoints = _find_epoch_checkpoints(symbol, symbol_cfg)
    if not checkpoints:
        raise FileNotFoundError(f'No direction epoch checkpoints found for {symbol} in {_epoch_dir(symbol, symbol_cfg)}')

    scaler_path = _default_scaler_path(symbol, symbol_cfg)
    features_path = _default_features_path(symbol, symbol_cfg)
    rows: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        epoch = _extract_epoch(ckpt)
        prefix = output_dir / symbol / f'{symbol}_{_timeframe(symbol_cfg)}_epoch_{epoch:03d}_direction_replay'
        summary = replay_symbol(
            symbol,
            symbol_cfg,
            model_path=ckpt,
            scaler_path=scaler_path,
            features_path=features_path,
            eval_start=eval_start,
            eval_end=eval_end,
            output_prefix=str(prefix),
            device=device,
            verbose=False,
        )
        row = {
            'symbol': symbol,
            'epoch': epoch,
            'checkpoint_path': str(ckpt),
            'summary_path': summary.get('summary_path'),
            'trades_path': summary.get('trades_path'),
            'decisions_path': summary.get('decisions_path'),
            'model_type': summary.get('model_type'),
            'architecture': summary.get('architecture'),
            'trades': summary.get('trades'),
            'net_pips': summary.get('net_pips'),
            'win_rate': summary.get('win_rate'),
            'average_net_pips': summary.get('average_net_pips'),
            'max_drawdown_pips': summary.get('max_drawdown_pips'),
            'buy_trades': summary.get('buy_trades'),
            'buy_net_pips': summary.get('buy_net_pips'),
            'sell_trades': summary.get('sell_trades'),
            'sell_net_pips': summary.get('sell_net_pips'),
            'passes_model_gate': summary.get('passes_model_gate'),
            'passes_external_gate': summary.get('passes_external_gate'),
            'raw_direction_accuracy': summary.get('raw_direction_accuracy'),
            'replay_score': summary.get('replay_score'),
        }
        rows.append(row)
        if verbose:
            print(f"{symbol} epoch {epoch:03d}: trades={row['trades']} net={float(row['net_pips'] or 0):.1f} score={float(row['replay_score'] or 0):.2f}", flush=True)

    min_trades = int((cfg.get('replay', {}) or {}).get('min_trades_for_score', 50) or 0)
    combined_csv = output_dir / symbol / f'{symbol}_{_timeframe(cfg)}_direction_replay_by_epoch.csv'
    _write_csv(combined_csv, rows)
    result = {
        'symbol': symbol,
        'timeframe': _timeframe(cfg),
        'epoch_count': len(rows),
        'eval_start': eval_start,
        'eval_end': eval_end,
        'combined_csv': str(combined_csv),
        'best_replay_score': _best(rows, 'replay_score', min_trades=min_trades),
        'best_net_pips': _best(rows, 'net_pips', min_trades=min_trades),
        'best_average_net_pips': _best(rows, 'average_net_pips', min_trades=min_trades),
        'epochs': rows,
    }
    summary_path = output_dir / symbol / f'{symbol}_{_timeframe(cfg)}_direction_replay_all_epochs_summary.json'
    write_json(summary_path, _json_safe(result))
    result['summary_path'] = str(summary_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay all saved direction-policy epoch checkpoints')
    parser.add_argument('--config', default='config/direction_settings_generic_multisymbol_31_symbols.yaml')
    parser.add_argument('--symbol', default=None)
    parser.add_argument('--symbols', nargs='+', default=None)
    parser.add_argument('--eval-start', default=None)
    parser.add_argument('--eval-end', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    cfg = load_config_with_optional_spread_risk(args.config)
    symbols = args.symbols or ([args.symbol] if args.symbol else ((cfg.get('trading') or {}).get('symbols') or ['US500']))
    symbols = validate_forex_symbols(symbols)
    rcfg = cfg.get('replay', {}) or {}
    eval_start = args.eval_start if args.eval_start is not None else rcfg.get('eval_start')
    eval_end = args.eval_end if args.eval_end is not None else rcfg.get('eval_end')
    output_dir = Path(args.output_dir or rcfg.get('output_dir') or (_log_dir(cfg) / 'direction_replay_epochs'))
    ensure_dir(output_dir)

    summaries = []
    for symbol in symbols:
        summaries.append(replay_epochs_for_symbol(
            symbol,
            cfg,
            eval_start=eval_start,
            eval_end=eval_end,
            output_dir=output_dir,
            device=args.device,
            verbose=args.verbose,
        ))

    overall_path = output_dir / f'direction_replay_all_symbols_{_timeframe(cfg)}.json'
    write_json(overall_path, _json_safe({'symbols': symbols, 'summaries': summaries}))
    print(json.dumps(_json_safe({'summary_path': str(overall_path), 'symbols': symbols}), indent=2))


if __name__ == '__main__':
    main()
