#!/usr/bin/env python3
"""Train universal BUY/SELL direction models on a combined multi-symbol dataset."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import load_config_with_optional_spread_risk, save_config
from .io_utils import ensure_dir, write_json
from .train_direction_policy import _json_safe, train_symbol
from .universal_symbol_features import append_universal_symbol_feature_columns


ARCHITECTURE_ALIASES: dict[str, str] = {
    'residual_mlp': 'residual_mlp_gate_direction_v1',
    'tcn': 'hierarchical_tcn_edge_v1',
    'inception_time': 'inception_time_gate_direction_v1',
    'mixture_of_experts': 'mixture_of_experts_direction_v1',
    'llm_transformer': 'llm_transformer_side_setup_v1',
    'small_transformer': 'small_transformer_gate_direction_v1',
    # New suggested universal architectures.
    'tsmixer': 'tsmixer_direction_v1',
    'patch_tst': 'patch_tst_direction_v1',
    'timesnet': 'timesnet_direction_v1',
}


def _timeframe(cfg: dict[str, Any]) -> str:
    return str((cfg.get('trading') or {}).get('timeframe') or (cfg.get('project') or {}).get('timeframe') or 'M5').upper()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _normalise_arch_token(token: str) -> tuple[str, str]:
    key = str(token).strip().lower()
    arch = ARCHITECTURE_ALIASES.get(key, token)
    # Use the friendly key as the output folder when available.
    friendly = key if key in ARCHITECTURE_ALIASES else str(token).replace('_direction_v1', '').replace('_gate_direction_v1', '')
    return friendly, arch


def _side_cfg(cfg: dict[str, Any], *, token: str, arch: str, side: str, combined_csv: Path, output_root: Path) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out = append_universal_symbol_feature_columns(out)
    ucfg = out.get('universal', {}) or {}
    arch_cfgs = ucfg.get('architecture_configs', {}) or {}
    model_overlay = arch_cfgs.get(token, {}) or arch_cfgs.get(arch, {}) or {}
    model = out.setdefault('model', {})
    model.update(copy.deepcopy(model_overlay))
    model['architecture'] = arch

    training = out.setdefault('training', {})
    training['use_pregenerated_direction_data'] = True
    training['require_pregenerated_direction_data'] = True
    training['pregenerated_direction_data_path'] = str(combined_csv)
    training['side_setup_train_side'] = side
    training['train_side'] = side
    # Universal replay is handled by replay_universal_direction_models.py by default.
    training['replay_each_epoch'] = bool(ucfg.get('replay_each_epoch_during_training', False))

    replay = out.setdefault('replay', {})
    if side == 'buy':
        replay['allow_buy'] = True
        replay['allow_sell'] = False
    elif side == 'sell':
        replay['allow_buy'] = False
        replay['allow_sell'] = True
    else:
        replay['allow_buy'] = True
        replay['allow_sell'] = True

    model_dir = output_root / 'models' / token / side
    log_dir = output_root / 'logs' / token / side
    out.setdefault('paths', {})['model_dir'] = str(model_dir)
    out.setdefault('paths', {})['log_dir'] = str(log_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Train universal direction-policy models using a combined multi-symbol dataset.')
    ap.add_argument('--config', default='config/direction_settings_universal_models.yaml')
    ap.add_argument('--combined-csv', default=None)
    ap.add_argument('--architectures', nargs='*', default=None, help='Architecture tokens. Defaults to universal.model_architectures.')
    ap.add_argument('--sides', nargs='*', default=None, choices=['buy', 'sell', 'both'])
    ap.add_argument('--output-root', default=None)
    ap.add_argument('--date-start', default=None)
    ap.add_argument('--date-end', default=None)
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--learning-rate', type=float, default=None)
    ap.add_argument('--val-fraction', type=float, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--reseed-each-epoch', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--epoch-seed-mode', default=None)
    args = ap.parse_args()

    base_cfg = load_config_with_optional_spread_risk(args.config)
    base_cfg['_config_path'] = str(args.config)
    base_cfg.setdefault('_base_config_path', str(args.config))
    ucfg = base_cfg.get('universal', {}) or {}
    combined_csv = Path(args.combined_csv or ucfg.get('combined_dataset_path') or f'data/universal/UNIVERSAL_{_timeframe(base_cfg)}_direction_training.csv')
    if not combined_csv.exists():
        raise SystemExit(f'Combined dataset not found: {combined_csv}. Run combine_universal_direction_datasets.py first.')
    arch_tokens = args.architectures or ucfg.get('model_architectures') or ['residual_mlp', 'tcn', 'inception_time', 'mixture_of_experts', 'llm_transformer', 'tsmixer', 'patch_tst', 'timesnet']
    sides = args.sides or ucfg.get('train_sides') or ['buy', 'sell']
    output_root = Path(args.output_root or ucfg.get('output_root') or 'models/universal')
    ensure_dir(output_root)

    reports: list[dict[str, Any]] = []
    manifest_models: list[dict[str, Any]] = []
    for token_in in arch_tokens:
        token, arch = _normalise_arch_token(token_in)
        for side in sides:
            cfg = _side_cfg(base_cfg, token=token, arch=arch, side=side, combined_csv=combined_csv, output_root=output_root)
            cfg_path = output_root / 'configs' / f'universal_{token}_{side}.yaml'
            save_config(cfg, cfg_path)
            train_args = SimpleNamespace(
                device=args.device,
                date_start=args.date_start,
                date_end=args.date_end,
                max_rows=args.max_rows,
                val_fraction=args.val_fraction,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                seed=args.seed,
                deterministic=args.deterministic,
                reseed_each_epoch=args.reseed_each_epoch,
                epoch_seed_mode=args.epoch_seed_mode,
            )
            universal_symbol = f'UNIVERSAL_{side.upper()}' if side != 'both' else 'UNIVERSAL'
            print(f'\n=== Training universal {token} {side} model ({arch}) ===')
            report = train_symbol(universal_symbol, cfg, train_args)
            report['universal_model_token'] = token
            report['universal_architecture'] = arch
            report['universal_train_side'] = side
            report['universal_config_path'] = str(cfg_path)
            reports.append(report)
            artifacts = report.get('artifacts', {}) or {}
            manifest_models.append({
                'id': f'universal_{token}_{side}',
                'model': token,
                'architecture': arch,
                'side': side,
                'train_side': side,
                'timeframe': _timeframe(cfg),
                'config_path': str(cfg_path),
                'model_path': artifacts.get('model_path'),
                'scaler_path': artifacts.get('scaler_path'),
                'features_path': artifacts.get('features_path'),
                'report_path': artifacts.get('report_path'),
                'symbols': list((cfg.get('trading') or {}).get('symbols') or []),
            })

    manifest = {
        'created_by': 'train_universal_direction_models.py',
        'config_path': str(args.config),
        'combined_dataset_path': str(combined_csv),
        'output_root': str(output_root),
        'timeframe': _timeframe(base_cfg),
        'models': manifest_models,
    }
    write_json(output_root / 'universal_model_manifest.json', _json_safe(manifest))
    write_json(output_root / 'universal_training_summary.json', _json_safe({'reports': reports, 'manifest': manifest}))
    print(f'\nWrote universal model manifest: {output_root / "universal_model_manifest.json"}')
    print(f'Wrote universal training summary: {output_root / "universal_training_summary.json"}')


if __name__ == '__main__':
    main()
