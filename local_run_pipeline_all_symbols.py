from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYMBOLS = [
    'US500',
    'NAS100',
    'GER40',
    'XAUUSD',
    'XAGUSD',
]


DEFAULT_CONFIGS = [
    'config/direction_settings_residual_mlp.yaml',
    'config/direction_settings_tcn.yaml',
    'config/direction_settings_inception_time.yaml',
    'config/direction_settings_mixture_of_experts.yaml',
    'config/direction_settings_llm_transformer.yaml',
]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _run(cmd: list[str], *, dry_run: bool = False, check: bool = True) -> int:
    print('\n$ ' + ' '.join(map(str, cmd)), flush=True)
    if dry_run:
        return 0
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stdout:
        print(result.stdout, end='', flush=True)
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr, flush=True)
    if result.returncode != 0 and check:
        raise RuntimeError(f'Command failed with exit code {result.returncode}: ' + ' '.join(map(str, cmd)))
    return int(result.returncode)


def _add_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _normalise_symbols(symbols: list[str] | None) -> list[str]:
    """Return a clean symbol list.

    Passing no symbols, ALL, all, or * trains every configured default symbol.
    """
    if not symbols:
        return list(DEFAULT_SYMBOLS)

    cleaned = [str(s).upper().strip() for s in symbols if str(s).strip()]
    if not cleaned or any(s in {'ALL', '*'} for s in cleaned):
        return list(DEFAULT_SYMBOLS)

    # Preserve order while removing duplicates.
    out: list[str] = []
    seen: set[str] = set()
    for symbol in cleaned:
        if symbol not in seen:
            out.append(symbol)
            seen.add(symbol)
    return out


def _normalise_sides(sides: list[str] | None) -> list[str]:
    out: list[str] = []
    for side in sides or ['buy', 'sell']:
        side_l = str(side).strip().lower()
        if side_l in {'long'}:
            side_l = 'buy'
        if side_l in {'short'}:
            side_l = 'sell'
        if side_l == 'both':
            expanded = ['buy', 'sell']
        elif side_l in {'buy', 'sell'}:
            expanded = [side_l]
        else:
            raise ValueError(f'Unsupported side {side!r}; use buy, sell, or both.')
        for expanded_side in expanded:
            if expanded_side not in out:
                out.append(expanded_side)
    return out or ['buy', 'sell']


def _config_token(config_path: str | Path, cfg: dict[str, Any]) -> str:
    paths = cfg.get('paths', {}) or {}
    model_dir = str(paths.get('model_dir', '') or '')
    if model_dir:
        token = Path(model_dir).name
    else:
        token = Path(config_path).stem.replace('direction_settings_', '')
    token = token.strip().replace(' ', '_')
    return token or Path(config_path).stem


def _normalise_configs(configs: list[str] | None, *, skip_missing: bool = True) -> list[str]:
    """Clean config list and optionally skip unavailable default configs.

    This makes the default "all models" mode safe when a checkout does not yet
    contain a newer architecture config, while still failing for explicitly supplied
    missing configs if --no-skip-missing-configs is used.
    """
    raw = configs or list(DEFAULT_CONFIGS)
    if len(raw) == 1 and str(raw[0]).strip().lower() in {'all', '*'}:
        raw = list(DEFAULT_CONFIGS)

    out: list[str] = []
    missing: list[str] = []
    for cfg in raw:
        cfg_s = str(cfg).strip()
        if not cfg_s:
            continue
        if Path(cfg_s).exists():
            out.append(cfg_s)
        else:
            missing.append(cfg_s)

    if missing and not skip_missing:
        raise SystemExit('Missing config file(s): ' + ', '.join(missing))
    if missing and skip_missing:
        print('Skipping missing config file(s): ' + ', '.join(missing), flush=True)
    if not out:
        raise SystemExit('No model config files were found. Check --configs or run from the project root.')
    return out


def _make_side_config(base_config_path: str | Path, symbol: str, side: str, *, generated_root: Path, output_root: Path) -> Path:
    cfg = deepcopy(_load_yaml(base_config_path))
    symbol = symbol.upper()
    side = side.lower()
    token = _config_token(base_config_path, cfg)

    cfg.setdefault('project', {})['name'] = f"{cfg.get('project', {}).get('name', token)}_{symbol}_{side}"
    cfg.setdefault('trading', {})['symbols'] = [symbol]

    # Separate outputs per architecture/symbol/side so parallel jobs never write
    # the same checkpoint, scaler, feature list, report or summary JSON.
    model_dir = output_root / 'models' / token / symbol / side
    log_dir = output_root / 'logs' / token / symbol / side
    cfg.setdefault('paths', {})['model_dir'] = str(model_dir).replace('\\', '/')
    cfg.setdefault('paths', {})['log_dir'] = str(log_dir).replace('\\', '/')

    tcfg = cfg.setdefault('training', {})
    tcfg['side_setup_train_side'] = side
    tcfg['train_side'] = side
    tcfg['target_mode'] = 'side_setup_ranking'
    tcfg['use_pregenerated_direction_data'] = True
    tcfg['require_pregenerated_direction_data'] = True
    tcfg['buy_setup_loss_weight'] = 1.0 if side == 'buy' else 0.0
    tcfg['sell_setup_loss_weight'] = 1.0 if side == 'sell' else 0.0
    tcfg['replay_output_dir'] = str(log_dir / 'epoch_replay').replace('\\', '/')

    mcfg = cfg.setdefault('model', {})
    mcfg['use_side_setup_heads'] = True
    mcfg['decision_output_mode'] = 'side_setup'
    mcfg['use_setup_quality_head'] = bool(mcfg.get('use_setup_quality_head', True))

    rcfg = cfg.setdefault('replay', {})
    rcfg['threshold_mode'] = rcfg.get('threshold_mode', 'rolling_score_quantile')
    rcfg['allow_buy'] = side == 'buy'
    rcfg['allow_sell'] = side == 'sell'
    rcfg['output_dir'] = str(log_dir / 'replay').replace('\\', '/')

    out_path = generated_root / f'{Path(base_config_path).stem}_{symbol}_{side}.yaml'
    _write_yaml(out_path, cfg)
    return out_path


def _train_task(cmd: list[str], env_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('NUMEXPR_NUM_THREADS', '1')
    if env_overrides:
        env.update(env_overrides)
    print('\n$ ' + ' '.join(map(str, cmd)), flush=True)
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.stdout:
        print(result.stdout, end='', flush=True)
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr, flush=True)
    return {
        'cmd': cmd,
        'returncode': int(result.returncode),
        'stdout_tail': (result.stdout or '')[-4000:],
        'stderr_tail': (result.stderr or '')[-4000:],
    }


def main() -> None:
    p = argparse.ArgumentParser(description='CFD-only local raw-data -> features -> strong setup labels -> side-specific model training pipeline.')
    p.add_argument('--raw-input-dir', default=None, help='Directory containing per-symbol CSVs, for example ./raw_csvs')
    p.add_argument('--combined-csv', default=None, help='Optional combined raw CSV containing a symbol column.')
    p.add_argument('--symbols', nargs='+', default=['ALL'], help='Symbols to process. Default/all/* trains all CFD default symbols.')
    p.add_argument('--timeframe', default='M5')
    p.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='Model config files to train. Use all/* for the default model set.')
    p.add_argument('--sides', nargs='+', default=['buy', 'sell'], help='Sides to train separately: buy sell. Use both only for the old combined mode.')
    p.add_argument('--mode', choices=['prepare-only', 'train-side-all', 'train-all', 'train-one', 'train-combined-all'], default='train-side-all', help='train-side-all/train-all/train-one all build side-specific tasks. train-one is retained as a backwards-compatible alias.')
    p.add_argument('--train-start', default=None)
    p.add_argument('--train-end', default=None)
    p.add_argument('--replay-start', default=None)
    p.add_argument('--replay-end', default=None)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--device', default=None, help='cpu, cuda, or leave blank for script default.')
    p.add_argument('--parallel-jobs', type=int, default=2, help='Concurrent model-training subprocesses across model/symbol/side tasks. Use 1 for a single GPU unless you know you have memory headroom.')
    p.add_argument('--raw-max-rows-per-symbol', type=int, default=None)
    p.add_argument('--feature-max-rows', type=int, default=None)
    p.add_argument('--direction-max-rows', type=int, default=None)
    p.add_argument('--train-max-rows', type=int, default=None)
    p.add_argument('--prepare-workers', type=int, default=2)
    p.add_argument('--force-raw', action='store_true')
    p.add_argument('--force-features', action='store_true')
    p.add_argument('--skip-raw-copy', action='store_true')
    p.add_argument('--skip-feature-prep', action='store_true')
    p.add_argument('--skip-direction-prep', action='store_true')
    p.add_argument('--output-root', default='.', help='Root under which local models/, logs/ and generated configs are written.')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--continue-on-model-error', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--skip-missing-configs', action=argparse.BooleanOptionalAction, default=True, help='Skip default model config files that are not present. Use --no-skip-missing-configs to fail fast.')
    args = p.parse_args()

    if args.mode in {'train-all', 'train-one'}:
        args.mode = 'train-side-all'

    symbols = _normalise_symbols(args.symbols)
    sides = _normalise_sides(args.sides)
    args.configs = _normalise_configs(args.configs, skip_missing=args.skip_missing_configs)
    output_root = Path(args.output_root)
    generated_root = output_root / 'config' / 'generated_local'
    first_config = args.configs[0]

    print(
        f'Pipeline selection: {len(args.configs)} config(s) x {len(symbols)} symbol(s) x '
        f'{len(sides) if args.mode != "train-combined-all" else 1} side task(s)',
        flush=True,
    )
    print('Symbols: ' + ', '.join(symbols), flush=True)
    print('Configs: ' + ', '.join(args.configs), flush=True)
    if args.mode != 'train-combined-all':
        print('Sides: ' + ', '.join(sides), flush=True)

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    if not args.skip_raw_copy:
        if not args.raw_input_dir and not args.combined_csv:
            raise SystemExit('Provide --raw-input-dir or --combined-csv, or use --skip-raw-copy when data/raw already contains SYMBOL_M5.csv files.')
        cmd = [
            sys.executable, '-m', 'src.kaggle_prepare_raw_data',
            '--input-dir', args.raw_input_dir or str(Path(args.combined_csv).parent),
            '--output-dir', 'data/raw',
            '--timeframe', args.timeframe,
            '--symbols', *symbols,
        ]
        _add_optional(cmd, '--combined-csv', args.combined_csv)
        _add_optional(cmd, '--max-rows-per-symbol', args.raw_max_rows_per_symbol)
        if args.force_raw:
            cmd.append('--force')
        _run(cmd, dry_run=args.dry_run)

    if not args.skip_feature_prep:
        cmd = [
            sys.executable, '-m', 'src.prepare_mt5_data',
            '--config', first_config,
            '--symbols', *symbols,
            '--timeframe', args.timeframe,
            '--workers', str(args.prepare_workers),
        ]
        _add_optional(cmd, '--max-rows', args.feature_max_rows)
        if args.force_features:
            cmd.append('--force')
        _run(cmd, dry_run=args.dry_run)

    if not args.skip_direction_prep:
        cmd = [
            sys.executable, '-m', 'src.prepare_direction_dataset',
            '--config', first_config,
            '--symbols', *symbols,
            '--workers', str(args.prepare_workers),
        ]
        _add_optional(cmd, '--date-start', args.train_start)
        # Prepare through replay_end so replay has labelled rows. The training
        # loader still tail-drops horizon rows at date_end for the fitting split.
        _add_optional(cmd, '--date-end', args.replay_end or args.train_end)
        _add_optional(cmd, '--max-rows', args.direction_max_rows)
        _run(cmd, dry_run=args.dry_run)

    if args.mode == 'prepare-only':
        print('\nLocal preparation complete. Direction CSVs are under data/direction/.', flush=True)
        return

    tasks: list[tuple[str, str, str, list[str]]] = []
    if args.mode == 'train-combined-all':
        for config in args.configs:
            for symbol in symbols:
                cmd = [
                    sys.executable, '-m', 'src.train_direction_policy_replay_each_epoch',
                    '--config', config,
                    '--symbols', symbol,
                    '--epochs', str(args.epochs),
                    '--batch-size', str(args.batch_size),
                    '--model-selection-metric', 'replay_score',
                    '--train-side', 'both',
                ]
                _add_optional(cmd, '--date-start', args.train_start)
                _add_optional(cmd, '--date-end', args.train_end)
                _add_optional(cmd, '--replay-start', args.replay_start)
                _add_optional(cmd, '--replay-end', args.replay_end)
                _add_optional(cmd, '--max-rows', args.train_max_rows)
                _add_optional(cmd, '--learning-rate', args.learning_rate)
                _add_optional(cmd, '--device', args.device)
                tasks.append((config, symbol, 'both', cmd))
    else:
        for config in args.configs:
            for symbol in symbols:
                for side in sides:
                    cfg_path = _make_side_config(config, symbol, side, generated_root=generated_root, output_root=output_root)
                    cmd = [
                        sys.executable, '-m', 'src.train_direction_policy_replay_each_epoch',
                        '--config', str(cfg_path),
                        '--symbols', symbol,
                        '--epochs', str(args.epochs),
                        '--batch-size', str(args.batch_size),
                        '--model-selection-metric', 'replay_score',
                        '--train-side', side,
                    ]
                    _add_optional(cmd, '--date-start', args.train_start)
                    _add_optional(cmd, '--date-end', args.train_end)
                    _add_optional(cmd, '--replay-start', args.replay_start)
                    _add_optional(cmd, '--replay-end', args.replay_end)
                    _add_optional(cmd, '--max-rows', args.train_max_rows)
                    _add_optional(cmd, '--learning-rate', args.learning_rate)
                    _add_optional(cmd, '--device', args.device)
                    tasks.append((str(cfg_path), symbol, side, cmd))

    if args.dry_run:
        print(f'\nDry run: prepared {len(tasks)} training task(s).')
        for _, _, _, cmd in tasks:
            print('$ ' + ' '.join(map(str, cmd)))
        return

    max_workers = max(1, int(args.parallel_jobs or 1))
    print(f'\nStarting {len(tasks)} training task(s) with parallel_jobs={max_workers}', flush=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(_train_task, cmd): {'config': config, 'symbol': symbol, 'side': side, 'cmd': cmd}
            for config, symbol, side, cmd in tasks
        }
        for fut in cf.as_completed(future_map):
            meta = future_map[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {'cmd': meta['cmd'], 'returncode': -999, 'error': repr(exc)}
            res.update({k: v for k, v in meta.items() if k != 'cmd'})
            results.append(res)
            if int(res.get('returncode', 1)) != 0:
                failures.append(res)
                print(f"[FAILED] {meta['symbol']} {meta['side']} {meta['config']} rc={res.get('returncode')}", flush=True)
                if not args.continue_on_model_error:
                    raise SystemExit('Stopping because --no-continue-on-model-error was set.')
            else:
                print(f"[OK] {meta['symbol']} {meta['side']} {meta['config']}", flush=True)

    summary_path = output_root / 'logs' / 'local_side_setup_training_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'tasks': results, 'failures': failures}, f, indent=2)
    print(f'\nWrote local training summary: {summary_path}', flush=True)
    if failures:
        print(f'Completed with {len(failures)} failed task(s). See summary JSON for stdout/stderr tails.', flush=True)
    else:
        print('All local side-specific training tasks completed successfully.', flush=True)


if __name__ == '__main__':
    main()
