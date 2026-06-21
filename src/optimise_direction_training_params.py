from __future__ import annotations

"""Grid-search training/model parameters for selectable neural direction models.

This script deliberately uses the current project training and replay entry
points rather than reimplementing the model. It creates one temporary config per
run, trains a model, replays it over a held-out date range, and writes a JSON/CSV
leaderboard ranked by replay/win performance.

Example:

    python -m src.optimise_direction_training_params \
        --config config/direction_settings_generic_multisymbol_31_symbols.yaml \
        --symbols US500 \
        --train-start 2024-01-01 --train-end 2024-10-01 \
        --replay-start 2024-10-01 --replay-end 2025-01-01 \
        --epochs 40 --max-runs 12

Set --smoke for a very small runtime check. Omit --smoke for the real search.
"""

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import load_config_with_optional_spread_risk
from .io_utils import ensure_dir, read_json, write_json


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except Exception:
        pass
    return value


def _set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split('.')
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def _default_grid(smoke: bool) -> list[dict[str, Any]]:
    """Return a small default grid that exercises each supported neural architecture.

    Larger architecture-specific sweeps should be supplied with --grid. The
    bundled config/direction_param_grid_model_architectures.yaml mirrors this
    default and is useful for a quick five-model comparison.
    """
    common = {
        'model.sequence_length': 64,
        'model.use_edge_pips_head': False,
        'training.use_edge_pips_loss': False,
        'training.edge_pips_loss_weight': 0.0,
        'training.use_analytic_signal_agreement_loss': False,
        'training.analytic_signal_agreement_loss_weight': 0.0,
        'training.gate_pos_weight': 0.35,
        'training.curriculum_sampler.start_ratios.no_trade': 3.0,
        'training.curriculum_sampler.end_ratios.no_trade': 8.0,
        'replay.min_trades_for_score': 20,
        'replay.min_trade_probability': 0.60,
        'replay.min_direction_probability': 0.40,
    }
    runs = [
        {
            'name': 'residual_mlp',
            **common,
            'model.architecture': 'residual_mlp_gate_direction_v1',
            'model.dropout': 0.05,
            'model.mlp_input_mode': 'last',
            'model.mlp_hidden_sizes': [256, 128, 64],
            'model.mlp_residual_blocks': 2,
        },
        {
            'name': 'tcn',
            **common,
            'model.architecture': 'hierarchical_tcn_edge_v1',
            'model.dropout': 0.03,
            'model.tcn_channels': 48,
            'model.tcn_blocks': 4,
            'model.tcn_kernel_size': 5,
            'model.tcn_dilations': [1, 2, 4, 8],
            'model.tcn_dropout': 0.03,
            'model.dense_hidden_sizes': [64, 32],
        },
        {
            'name': 'small_transformer',
            **common,
            'model.architecture': 'small_transformer_gate_direction_v1',
            'model.dropout': 0.05,
            'model.d_model': 64,
            'model.n_heads': 4,
            'model.num_layers': 2,
            'model.feedforward_dim': 128,
            'model.pooling': 'attention',
            'model.dense_hidden_sizes': [64, 32],
        },
        {
            'name': 'inception_time',
            **common,
            'model.architecture': 'inception_time_gate_direction_v1',
            'model.dropout': 0.05,
            'model.channels': 32,
            'model.blocks': 3,
            'model.kernel_sizes': [3, 5, 9, 17],
            'model.dense_hidden_sizes': [64, 32],
        },
        {
            'name': 'mixture_of_experts',
            **common,
            'model.architecture': 'mixture_of_experts_direction_v1',
            'model.dropout': 0.05,
            'model.num_experts': 4,
            'model.expert_hidden_size': 64,
            'model.expert_layers': 2,
            'model.expert_input_mode': 'last_mean',
            'model.router_hidden_size': 32,
            'model.router_inputs': [
                'sig_analytic_signal_class', 'sig_buy_signal_count', 'sig_sell_signal_count',
                'sig_net_signal_vote', 'sig_signal_conflict', 'sig_adx_strength', 'sig_atr_zscore',
            ],
            'model.dense_hidden_sizes': [64],
        },
    ]
    return runs[:2] if smoke else runs

def _load_grid(path: str | None, smoke: bool) -> list[dict[str, Any]]:
    if not path:
        return _default_grid(smoke)
    p = Path(path)
    payload = yaml.safe_load(p.read_text(encoding='utf-8'))
    if isinstance(payload, dict) and isinstance(payload.get('runs'), list):
        return [dict(x) for x in payload['runs']]
    if isinstance(payload, list):
        return [dict(x) for x in payload]
    raise ValueError('Parameter grid must be a list or a mapping with a runs list.')


def _run_subprocess(cmd: list[str], cwd: Path, threads: int) -> None:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(cwd)
    env['OMP_NUM_THREADS'] = str(threads)
    env['MKL_NUM_THREADS'] = str(threads)
    env['OPENBLAS_NUM_THREADS'] = str(threads)
    env['NUMEXPR_NUM_THREADS'] = str(threads)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _score(summary: dict[str, Any]) -> float:
    trades = float(summary.get('trades', 0) or 0)
    if trades <= 0:
        return -1e9
    win = float(summary.get('win_rate', 0.0) or 0.0)
    avg = float(summary.get('average_net_pips', 0.0) or 0.0)
    net = float(summary.get('net_pips', 0.0) or 0.0)
    dd = float(summary.get('max_drawdown_pips', 0.0) or 0.0)
    # Win-first but still penalise negative pips/drawdown.
    return win * 1000.0 + avg * 100.0 + net / 100.0 - dd / 100.0 + min(trades, 500.0) / 10.0


def _apply_run_params(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for k, v in run.items():
        if k == 'name':
            continue
        _set_nested(out, k, v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description='Optimise direction model architecture/training parameters by train+replay grid search.')
    p.add_argument('--config', default='config/direction_settings_generic_multisymbol_31_symbols.yaml')
    p.add_argument('--symbols', nargs='+', default=['US500'])
    p.add_argument('--grid', default=None, help='Optional YAML list of run parameter mappings.')
    p.add_argument('--out-dir', default='logs/direction_training_param_optimisation')
    p.add_argument('--train-start', default=None)
    p.add_argument('--train-end', default=None)
    p.add_argument('--replay-start', default=None)
    p.add_argument('--replay-end', default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--max-runs', type=int, default=None)
    p.add_argument('--threads', type=int, default=1, help='Use 1 on CPU to avoid OpenMP/MKL stalls during repeated training runs.')
    p.add_argument('--smoke', action='store_true', help='Use a tiny 2-run smoke grid.')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    cwd = Path.cwd()
    base_cfg = load_config_with_optional_spread_risk(args.config)
    runs = _load_grid(args.grid, args.smoke)
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    out_dir = ensure_dir(args.out_dir)
    run_root = ensure_dir(out_dir / 'runs')

    rows: list[dict[str, Any]] = []
    for i, run in enumerate(runs, start=1):
        name = str(run.get('name') or f'run_{i:03d}')
        safe_name = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name)
        this_dir = ensure_dir(run_root / safe_name)
        cfg = _apply_run_params(base_cfg, run)
        cfg.setdefault('trading', {})['symbols'] = list(args.symbols)
        cfg.setdefault('paths', {})['model_dir'] = str(this_dir / 'models')
        cfg.setdefault('paths', {})['log_dir'] = str(this_dir / 'logs')
        tcfg = cfg.setdefault('training', {})
        if args.train_start is not None:
            tcfg['date_start'] = args.train_start
        if args.train_end is not None:
            tcfg['date_end'] = args.train_end
        if args.epochs is not None:
            tcfg['epochs'] = int(args.epochs)
        if args.batch_size is not None:
            tcfg['batch_size'] = int(args.batch_size)
        if args.learning_rate is not None:
            tcfg['learning_rate'] = float(args.learning_rate)
        tcfg['replay_each_epoch'] = False
        tcfg.setdefault('curriculum_sampler', {})['enabled'] = True
        tcfg['use_curriculum_sampler'] = True
        # Keep side weights neutral; let gate/curriculum do most of the work.
        tcfg['side_direction_class_weights'] = {'sell': 1.0, 'buy': 1.0}
        rcfg = cfg.setdefault('replay', {})
        if args.replay_start is not None:
            rcfg['eval_start'] = args.replay_start
        if args.replay_end is not None:
            rcfg['eval_end'] = args.replay_end
        rcfg['output_dir'] = str(this_dir / 'replay')

        cfg_path = this_dir / 'config.yaml'
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        if args.dry_run:
            rows.append({'run': name, 'config_path': str(cfg_path), 'dry_run': True})
            continue

        print(f'[{i}/{len(runs)}] train {name}', flush=True)
        _run_subprocess([
            sys.executable, '-m', 'src.train_direction_policy',
            '--config', str(cfg_path),
            '--symbols', *list(args.symbols),
        ], cwd, args.threads)

        for symbol in args.symbols:
            print(f'[{i}/{len(runs)}] replay {name} {symbol}', flush=True)
            prefix = this_dir / 'replay' / symbol / f'{symbol}_M5_direction_replay'
            _run_subprocess([
                sys.executable, '-m', 'src.test_saved_direction_policy',
                '--config', str(cfg_path),
                '--symbol', symbol,
                '--eval-start', args.replay_start or str(rcfg.get('eval_start') or ''),
                '--eval-end', args.replay_end or str(rcfg.get('eval_end') or ''),
                '--output-prefix', str(prefix),
            ], cwd, args.threads)
            summary_path = Path(str(prefix) + '_summary.json')
            summary = read_json(summary_path)
            row = {
                'run': name,
                'symbol': symbol,
                'score': _score(summary),
                'trades': summary.get('trades'),
                'win_rate': summary.get('win_rate'),
                'net_pips': summary.get('net_pips'),
                'average_net_pips': summary.get('average_net_pips'),
                'max_drawdown_pips': summary.get('max_drawdown_pips'),
                'buy_trades': summary.get('buy_trades'),
                'buy_win_rate': summary.get('buy_win_rate'),
                'buy_net_pips': summary.get('buy_net_pips'),
                'sell_trades': summary.get('sell_trades'),
                'sell_win_rate': summary.get('sell_win_rate'),
                'sell_net_pips': summary.get('sell_net_pips'),
                'config_path': str(cfg_path),
                'summary_path': str(summary_path),
                'params': {k: v for k, v in run.items() if k != 'name'},
            }
            rows.append(row)
            leaderboard = pd.DataFrame(rows).sort_values('score', ascending=False)
            leaderboard.to_csv(out_dir / 'leaderboard.csv', index=False)
            write_json(out_dir / 'leaderboard.json', _json_safe({'rows': leaderboard.to_dict(orient='records')}))

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty and 'score' in leaderboard.columns:
        leaderboard = leaderboard.sort_values('score', ascending=False)
    leaderboard.to_csv(out_dir / 'leaderboard.csv', index=False)
    write_json(out_dir / 'leaderboard.json', _json_safe({'rows': leaderboard.to_dict(orient='records')}))
    print(f'Wrote {out_dir / "leaderboard.csv"}')


if __name__ == '__main__':
    main()
