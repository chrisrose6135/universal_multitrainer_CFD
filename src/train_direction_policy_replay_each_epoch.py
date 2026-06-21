from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from .config import load_config_with_optional_spread_risk
from .forex import validate_forex_symbols
from .io_utils import ensure_dir, write_json
from .train_direction_policy import _json_safe, _timeframe, train_symbol


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
    parser.add_argument('--train-side', choices=['both', 'buy', 'sell'], default=None, help='Train both side-setup heads, or train a BUY-only/SELL-only setup model. Replay is side-filtered automatically for buy/sell.')
    parser.add_argument('--replay-start', default=None)
    parser.add_argument('--replay-end', default=None)
    parser.add_argument('--replay-output-dir', default=None)
    parser.add_argument('--model-selection-metric', default=None, help='Default for this script is replay_score.')
    args = parser.parse_args()

    cfg = load_config_with_optional_spread_risk(args.config)
    cfg = copy.deepcopy(cfg)
    cfg['_config_path'] = str(args.config)
    cfg.setdefault('_base_config_path', str(args.config))
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
    if args.train_side is not None:
        training['side_setup_train_side'] = args.train_side
        training['train_side'] = args.train_side

    symbols = validate_forex_symbols(args.symbols or ((cfg.get('trading') or {}).get('symbols') or ['US500']))
    reports = []
    for symbol in symbols:
        symbol_cfg: dict[str, Any] = copy.deepcopy(cfg)
        symbol_cfg['_active_symbol'] = symbol
        reports.append(train_symbol(symbol, symbol_cfg, args))

    summary_path = Path((cfg.get('paths') or {}).get('log_dir', 'logs')) / f'direction_training_replay_each_epoch_summary_{_timeframe(cfg)}.json'
    ensure_dir(summary_path.parent)
    write_json(summary_path, _json_safe({'symbols': symbols, 'reports': reports}))
    print(f'Wrote training+replay summary: {summary_path}')


if __name__ == '__main__':
    main()
