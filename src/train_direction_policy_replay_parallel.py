from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config_with_optional_spread_risk
from .forex import validate_forex_symbols
from .io_utils import ensure_dir, read_json, write_json
from .train_direction_policy import _json_safe, _timeframe, train_symbol


def _this_module_name() -> str:
    """Return the module path needed to re-launch this exact script via `python -m`.

    The parallel launcher must spawn children using this module, not the legacy
    train_direction_policy_replay_each_epoch module. The legacy module does not
    know about --parallel-symbols or --child-report-path, which causes child
    processes to fail with "unrecognized arguments".
    """
    spec_name = getattr(__spec__, 'name', None)
    if spec_name:
        return str(spec_name)
    package = __package__ or 'src'
    return f'{package}.{Path(__file__).stem}'


def _add_optional_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _add_optional_bool_arg(cmd: list[str], flag: str, value: bool | None) -> None:
    if value is True:
        cmd.append(flag)
    elif value is False:
        cmd.append(f'--no-{flag[2:]}')


def _child_command(symbol: str, args: argparse.Namespace, child_report_path: Path) -> list[str]:
    """Build a single-symbol child command.

    Each child trains exactly one symbol and writes a private report JSON. The
    parent then combines those JSONs into the usual multi-symbol summary.
    """
    cmd = [
        sys.executable,
        '-m',
        _this_module_name(),
        '--config',
        str(args.config),
        '--symbols',
        symbol,
        '--parallel-symbols',
        '1',
        '--child-report-path',
        str(child_report_path),
    ]
    _add_optional_arg(cmd, '--date-start', args.date_start)
    _add_optional_arg(cmd, '--date-end', args.date_end)
    _add_optional_arg(cmd, '--max-rows', args.max_rows)
    _add_optional_arg(cmd, '--epochs', args.epochs)
    _add_optional_arg(cmd, '--batch-size', args.batch_size)
    _add_optional_arg(cmd, '--learning-rate', args.learning_rate)
    _add_optional_arg(cmd, '--val-fraction', args.val_fraction)
    _add_optional_arg(cmd, '--device', args.device)
    _add_optional_arg(cmd, '--seed', args.seed)
    _add_optional_bool_arg(cmd, '--deterministic', args.deterministic)
    _add_optional_bool_arg(cmd, '--reseed-each-epoch', args.reseed_each_epoch)
    _add_optional_arg(cmd, '--epoch-seed-mode', args.epoch_seed_mode)
    _add_optional_arg(cmd, '--replay-start', args.replay_start)
    _add_optional_arg(cmd, '--replay-end', args.replay_end)
    _add_optional_arg(cmd, '--replay-output-dir', args.replay_output_dir)
    _add_optional_arg(cmd, '--model-selection-metric', args.model_selection_metric)
    return cmd


def _read_child_report(path: Path, symbol: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f'Parallel child for {symbol} finished but did not write report: {path}')
    payload = read_json(path)
    reports = payload.get('reports') if isinstance(payload, dict) else None
    if not isinstance(reports, list) or not reports:
        raise RuntimeError(f'Parallel child report for {symbol} has no reports list: {path}')
    return reports[0]


def _terminate_running(running: list[tuple[str, Path, subprocess.Popen]]) -> None:
    for _, _, proc in running:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 30.0
    for _, _, proc in running:
        if proc.poll() is None:
            try:
                proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()


def _train_symbols_in_parallel(symbols: list[str], cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Train several symbols concurrently by spawning one child process per symbol.

    This is intentionally process-based rather than trying to fit multiple
    models inside one Python loop. It keeps per-symbol checkpoint paths,
    replay logs, RNG seeds and exceptions isolated while allowing several
    training/replay jobs to share the GPU.
    """
    parallel = max(1, int(args.parallel_symbols or 1))
    if parallel <= 1 or len(symbols) <= 1:
        reports = []
        for symbol in symbols:
            symbol_cfg: dict[str, Any] = copy.deepcopy(cfg)
            symbol_cfg['_active_symbol'] = symbol
            reports.append(train_symbol(symbol, symbol_cfg, args))
        return reports

    log_dir = Path((cfg.get('paths') or {}).get('log_dir', 'logs'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    child_dir = log_dir / 'parallel_direction_training_replay_each_epoch' / f'{_timeframe(cfg)}_{stamp}_{os.getpid()}'
    ensure_dir(child_dir)

    pending = list(symbols)
    running: list[tuple[str, Path, subprocess.Popen]] = []
    completed: dict[str, dict[str, Any]] = {}

    print(
        f'Training {len(symbols)} symbols with up to {parallel} concurrent child processes.\n'
        f'Child reports: {child_dir}',
        flush=True,
    )

    while pending or running:
        while pending and len(running) < parallel:
            symbol = pending.pop(0)
            report_path = child_dir / f'{symbol}_child_report.json'
            cmd = _child_command(symbol, args, report_path)
            print(f'Starting {symbol}: {" ".join(cmd)}', flush=True)
            proc = subprocess.Popen(cmd)
            running.append((symbol, report_path, proc))

        time.sleep(2.0)
        still_running: list[tuple[str, Path, subprocess.Popen]] = []
        for symbol, report_path, proc in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((symbol, report_path, proc))
                continue
            if rc != 0:
                _terminate_running(still_running)
                raise RuntimeError(f'Parallel child for {symbol} failed with exit code {rc}')
            completed[symbol] = _read_child_report(report_path, symbol)
            print(f'Completed {symbol}; report: {report_path}', flush=True)
        running = still_running

    return [completed[symbol] for symbol in symbols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train direction-policy models and replay every saved epoch checkpoint'
    )
    parser.add_argument('--config', default='config/direction_settings_generic_multisymbol_31_symbols.yaml')
    parser.add_argument('--symbols', nargs='+', default=None)
    parser.add_argument('--date-start', default=None)
    parser.add_argument('--date-end', default=None)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--val-fraction', type=float, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=None, help='Base random seed. Overrides training.seed.')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable best-effort deterministic Torch behaviour.')
    parser.add_argument('--reseed-each-epoch', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable deterministic per-epoch reseeding.')
    parser.add_argument('--epoch-seed-mode', default=None, help='base_only, base_plus_epoch, base_plus_symbol, or base_plus_symbol_plus_epoch.')
    parser.add_argument('--replay-start', default=None)
    parser.add_argument('--replay-end', default=None)
    parser.add_argument('--replay-output-dir', default=None)
    parser.add_argument('--model-selection-metric', default=None, help='Default for this script is replay_score.')
    parser.add_argument('--parallel-symbols', type=int, default=1, help='Train this many symbols concurrently by launching one child process per symbol. Start with 2 on a single RTX 3090 Ti and increase cautiously.')
    parser.add_argument('--child-report-path', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = load_config_with_optional_spread_risk(args.config)
    cfg = copy.deepcopy(cfg)
    training = cfg.setdefault('training', {})
    training['replay_each_epoch'] = True
    training['save_epoch_models'] = True
    # In the train+replay route, checkpoint selection should be driven by replay,
    # not validation macro-F1. The user can override explicitly with the CLI.
    training['model_selection_metric'] = args.model_selection_metric or 'replay_score'
    if args.seed is not None:
        training['seed'] = int(args.seed)
    if args.deterministic is not None:
        training['deterministic'] = bool(args.deterministic)
    if args.reseed_each_epoch is not None:
        training['reseed_each_epoch'] = bool(args.reseed_each_epoch)
    if args.epoch_seed_mode is not None:
        training['epoch_seed_mode'] = args.epoch_seed_mode
    if args.replay_start is not None:
        training['replay_start'] = args.replay_start
    if args.replay_end is not None:
        training['replay_end'] = args.replay_end
    if args.replay_output_dir is not None:
        training['replay_output_dir'] = args.replay_output_dir

    symbols = validate_forex_symbols(args.symbols or ((cfg.get('trading') or {}).get('symbols') or ['US500']))
    reports = _train_symbols_in_parallel(symbols, cfg, args)

    payload = _json_safe({
        'symbols': symbols,
        'parallel_symbols': int(args.parallel_symbols or 1),
        'reports': reports,
    })

    if args.child_report_path:
        child_report_path = Path(args.child_report_path)
        ensure_dir(child_report_path.parent)
        write_json(child_report_path, payload)
        print(f'Wrote child training+replay report: {child_report_path}')
        return

    summary_path = Path((cfg.get('paths') or {}).get('log_dir', 'logs')) / f'direction_training_replay_each_epoch_summary_{_timeframe(cfg)}.json'
    ensure_dir(summary_path.parent)
    write_json(summary_path, payload)
    print(f'Wrote training+replay summary: {summary_path}')


if __name__ == '__main__':
    main()
