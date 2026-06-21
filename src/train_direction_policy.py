from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import load_config_with_optional_spread_risk
from .direction_dataset import DirectionDataset, prepare_direction_arrays
from .direction_model import DIRECTION_CLASS_NAMES, DirectionTradePolicyNet, direction_probabilities_from_outputs
from .analytic_signals import ensure_analytic_signal_features
from .forex import validate_forex_symbols
from .io_utils import ensure_dir, normalise_time_column, read_processed_csv, write_json
from .targets import generate_direction_targets
from .replay_decision_parameters import (
    config_snapshot,
    resolve_replay_decision_parameters,
    resolve_training_decision_parameters,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value




def _stable_symbol_seed_offset(symbol: str) -> int:
    """Deterministic small offset so each symbol has a different seed path."""
    return sum((i + 1) * ord(c) for i, c in enumerate(str(symbol).upper())) % 100_000


def _seed_cfg_value(cfg: dict[str, Any], args: argparse.Namespace, name: str, default: Any = None) -> Any:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    return (cfg.get('training', {}) or {}).get(name, default)


def _set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch. Deterministic mode is best-effort."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _seed_settings(symbol: str, cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    base_seed_raw = _seed_cfg_value(cfg, args, 'seed', None)
    deterministic = bool(_seed_cfg_value(cfg, args, 'deterministic', False))
    reseed_each_epoch = bool(_seed_cfg_value(cfg, args, 'reseed_each_epoch', False))
    epoch_seed_mode = str(_seed_cfg_value(cfg, args, 'epoch_seed_mode', 'base_plus_symbol_plus_epoch') or 'base_plus_symbol_plus_epoch')
    symbol_offset = _stable_symbol_seed_offset(symbol)
    if base_seed_raw in (None, ''):
        return {
            'enabled': False,
            'base_seed': None,
            'symbol_seed_offset': symbol_offset,
            'initial_seed': None,
            'deterministic': deterministic,
            'reseed_each_epoch': reseed_each_epoch,
            'epoch_seed_mode': epoch_seed_mode,
        }
    base_seed = int(base_seed_raw)
    if epoch_seed_mode in {'base_plus_symbol', 'base_plus_symbol_plus_epoch'}:
        initial_seed = base_seed + symbol_offset
    else:
        initial_seed = base_seed
    return {
        'enabled': True,
        'base_seed': base_seed,
        'symbol_seed_offset': symbol_offset,
        'initial_seed': int(initial_seed),
        'deterministic': deterministic,
        'reseed_each_epoch': reseed_each_epoch,
        'epoch_seed_mode': epoch_seed_mode,
    }


def _epoch_seed(seed_info: dict[str, Any], epoch: int) -> int | None:
    if not seed_info.get('enabled'):
        return None
    base_seed = int(seed_info['base_seed'])
    symbol_offset = int(seed_info.get('symbol_seed_offset') or 0)
    mode = str(seed_info.get('epoch_seed_mode') or 'base_plus_symbol_plus_epoch')
    if mode == 'base_plus_epoch':
        return base_seed + int(epoch)
    if mode == 'base_plus_symbol':
        return base_seed + symbol_offset
    if mode == 'base_plus_symbol_plus_epoch':
        return base_seed + symbol_offset + int(epoch)
    if mode == 'base_only':
        return base_seed
    raise ValueError(
        f"Unsupported training.epoch_seed_mode={mode!r}. "
        "Use base_only, base_plus_epoch, base_plus_symbol, or base_plus_symbol_plus_epoch."
    )


def _make_generator(seed: int | None, device: str = 'cpu') -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _normalise_torch_device(device: str | None) -> str:
    """Map user-friendly device aliases to valid torch device strings.

    Several project scripts historically accepted ``--device gpu``.  PyTorch
    expects ``cuda`` (or ``cuda:0``), so normalise it once here.
    """
    if device in (None, ''):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    d = str(device).strip().lower()
    if d in {'gpu', 'cuda_gpu'}:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if d == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return str(device)


def _timeframe(cfg: dict[str, Any]) -> str:
    return str((cfg.get('trading') or {}).get('timeframe') or (cfg.get('project') or {}).get('timeframe') or 'M5').upper()


def _pregenerated_path(symbol: str, cfg: dict[str, Any]) -> Path:
    tcfg = cfg.get('training', {}) or {}
    template = tcfg.get('direction_data_template', '{symbol}_{timeframe}_direction_training.csv')
    root = Path(tcfg.get('direction_data_dir', 'data/direction'))
    if tcfg.get('pregenerated_direction_data_path'):
        return Path(str(tcfg['pregenerated_direction_data_path']).format(symbol=symbol, timeframe=_timeframe(cfg)))
    return root / str(template).format(symbol=symbol, timeframe=_timeframe(cfg))


def _model_paths(symbol: str, cfg: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    paths = cfg.get('paths', {}) or {}
    model_dir = Path(paths.get('model_dir', 'models'))
    tf = _timeframe(cfg)
    model_path = model_dir / f'{symbol}_{tf}_direction_policy.pt'
    scaler_path = model_dir / f'{symbol}_{tf}_direction_scaler.pkl'
    features_path = model_dir / f'{symbol}_{tf}_direction_features.json'
    report_path = Path((cfg.get('paths') or {}).get('log_dir', 'logs')) / f'{symbol}_{tf}_direction_training_report.json'
    return model_path, scaler_path, features_path, report_path


def _epoch_artifact_paths(epoch_dir: Path, symbol: str, cfg: dict[str, Any], epoch: int) -> tuple[Path, Path, Path]:
    """Return model/scaler/features paths for a saved epoch checkpoint.

    Each epoch checkpoint is made self-contained by saving the scaler and
    feature metadata next to the epoch model. The scaler/feature contents are
    normally identical across epochs for a run, but storing them per checkpoint
    avoids accidentally replaying or deploying an epoch checkpoint with a scaler
    or feature list from another model, symbol, side, or later run.
    """
    tf = _timeframe(cfg)
    stem = f'{symbol}_{tf}_direction_policy_epoch_{int(epoch):03d}'
    checkpoint_path = epoch_dir / f'{stem}.pt'
    scaler_path = epoch_dir / f'{stem}_scaler.pkl'
    features_path = epoch_dir / f'{stem}_features.json'
    return checkpoint_path, scaler_path, features_path


def _feature_metadata(
    *,
    arr: Any,
    architecture_name: str,
    model: nn.Module,
    train_side: str | None,
    deployment_decision_parameters: dict[str, Any],
    resolved_config_snapshot: dict[str, Any],
    symbol: str,
    cfg: dict[str, Any],
    epoch: int | None = None,
    checkpoint_path: Path | None = None,
    scaler_path: Path | None = None,
) -> dict[str, Any]:
    meta = {
        'feature_columns': list(arr.feature_columns),
        'architecture': architecture_name,
        'model_details': getattr(model, 'model_details', {}),
        'symbol': symbol,
        'timeframe': _timeframe(cfg),
        'train_side': train_side,
        'deployment_decision_parameters': deployment_decision_parameters,
        'config_snapshot': {k: v for k, v in resolved_config_snapshot.items() if k != 'resolved_sections'},
    }
    if epoch is not None:
        meta['epoch'] = int(epoch)
    if checkpoint_path is not None:
        meta['checkpoint_model_path'] = str(checkpoint_path)
    if scaler_path is not None:
        meta['checkpoint_scaler_path'] = str(scaler_path)
    return meta


def _save_scaler_and_features(
    *,
    scaler: Any,
    scaler_path: Path,
    features_path: Path,
    feature_metadata: dict[str, Any],
) -> None:
    ensure_dir(scaler_path.parent)
    ensure_dir(features_path.parent)
    joblib.dump(scaler, scaler_path)
    write_json(features_path, _json_safe(feature_metadata))




def _universal_split_enabled(cfg: dict[str, Any], df: pd.DataFrame) -> bool:
    """Return True when a pooled universal dataset provides explicit split labels.

    The normal symbol-specific trainer uses a chronological split. Universal
    pooled datasets need per-symbol train/validation split labels so one symbol
    is not accidentally used mostly for validation just because of concat order.
    """
    ucfg = cfg.get('universal', {}) or {}
    if not bool(ucfg.get('enabled', False)):
        return False
    if not bool(ucfg.get('use_split_column', True)):
        return False
    col = str(ucfg.get('split_column', 'universal_split'))
    return col in df.columns


def _universal_split_column(cfg: dict[str, Any]) -> str:
    return str((cfg.get('universal', {}) or {}).get('split_column', 'universal_split'))


def _universal_split_labels(cfg: dict[str, Any]) -> tuple[str, str]:
    ucfg = cfg.get('universal', {}) or {}
    return (str(ucfg.get('train_split_label', 'train')).lower(), str(ucfg.get('validation_split_label', 'validation')).lower())


def _universal_train_scaler_df(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    col = _universal_split_column(cfg)
    train_label, val_label = _universal_split_labels(cfg)
    labels = df[col].astype(str).str.lower().str.strip()
    train_df = df.loc[labels == train_label].reset_index(drop=True)
    counts = {str(k): int(v) for k, v in labels.value_counts(dropna=False).sort_index().items()}
    return train_df, {
        'split_mode': 'universal_explicit_split_column',
        'split_column': col,
        'train_split_label': train_label,
        'validation_split_label': val_label,
        'split_label_row_counts': counts,
        'scaler_fit_scope': 'universal_train_split_rows_only',
    }


def _universal_sequence_split_indices(df: pd.DataFrame, arr: Any, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    col = _universal_split_column(cfg)
    train_label, val_label = _universal_split_labels(cfg)
    endpoint_labels = df.iloc[np.asarray(arr.row_indices, dtype=int)][col].astype(str).str.lower().str.strip().to_numpy()
    train_idx = np.where(endpoint_labels == train_label)[0].astype(int)
    val_idx = np.where(endpoint_labels == val_label)[0].astype(int)
    counts = {str(k): int(v) for k, v in pd.Series(endpoint_labels).value_counts(dropna=False).sort_index().items()}
    return train_idx, val_idx, {
        'split_mode': 'universal_explicit_split_column',
        'split_column': col,
        'train_split_label': train_label,
        'validation_split_label': val_label,
        'split_label_sequence_counts': counts,
    }


def _rebuild_universal_split_if_needed(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ensure date-filtered universal datasets still contain train and validation rows.

    The universal combiner writes a per-symbol ``universal_split`` column.  If a
    later training command applies ``--date-start/--date-end`` to the combined
    CSV, it can accidentally crop away the validation tail for every symbol,
    leaving only ``train`` labels.  Rebuild the split in-place, per symbol/group,
    when that happens.  Symbol-specific training is unaffected because this only
    runs when ``universal.enabled`` and the split column are present.
    """
    if not _universal_split_enabled(cfg, df):
        return df, {'universal_split_rebuilt': False, 'reason': 'universal split disabled or split column missing'}
    ucfg = cfg.get('universal', {}) or {}
    if not bool(ucfg.get('auto_rebuild_split_after_date_filter', True)):
        return df, {'universal_split_rebuilt': False, 'reason': 'auto rebuild disabled'}
    col = _universal_split_column(cfg)
    train_label, val_label = _universal_split_labels(cfg)
    labels = df[col].astype(str).str.lower().str.strip()
    counts_before = {str(k): int(v) for k, v in labels.value_counts(dropna=False).sort_index().items()}
    if counts_before.get(train_label, 0) > 0 and counts_before.get(val_label, 0) > 0:
        return df, {'universal_split_rebuilt': False, 'reason': 'train and validation labels already present', 'split_label_row_counts_before': counts_before}

    group_col = str(ucfg.get('sequence_group_column', 'universal_sequence_group'))
    min_val = int(ucfg.get('min_validation_rows_per_symbol', 500) or 0)
    min_train = int(ucfg.get('min_train_rows_per_symbol', 1000) or 0)
    out = df.copy().reset_index(drop=True)
    out[col] = train_label
    if group_col in out.columns:
        groups = [(str(k), v.index.to_numpy(dtype=int)) for k, v in out.groupby(group_col, sort=False)]
    elif 'symbol' in out.columns:
        groups = [(str(k), v.index.to_numpy(dtype=int)) for k, v in out.groupby('symbol', sort=False)]
    else:
        groups = [('ALL', np.arange(len(out), dtype=int))]

    group_rows: dict[str, dict[str, int]] = {}
    for group_name, idx in groups:
        n = int(len(idx))
        if n <= 1:
            group_rows[group_name] = {'rows': n, 'train': n, 'validation': 0, 'warning': 'too few rows'}
            continue
        val_rows = int(round(n * float(val_fraction)))
        if min_val > 0:
            val_rows = max(val_rows, min_val)
        max_val = max(1, n - max(1, min_train)) if n > min_train + 1 else max(1, n // 5)
        val_rows = min(max(val_rows, 1), max_val)
        val_idx = idx[-val_rows:]
        out.loc[val_idx, col] = val_label
        group_rows[group_name] = {'rows': n, 'train': int(n - val_rows), 'validation': int(val_rows)}

    labels_after = out[col].astype(str).str.lower().str.strip()
    counts_after = {str(k): int(v) for k, v in labels_after.value_counts(dropna=False).sort_index().items()}
    return out, {
        'universal_split_rebuilt': True,
        'reason': 'date/max-row filtering removed one split label' if counts_before.get(val_label, 0) == 0 else 'missing train or validation label',
        'split_label_row_counts_before': counts_before,
        'split_label_row_counts_after': counts_after,
        'validation_fraction': float(val_fraction),
        'min_validation_rows_per_symbol': int(min_val),
        'min_train_rows_per_symbol': int(min_train),
        'groups': group_rows,
    }

def _filter_date_range(df: pd.DataFrame, date_start: str | None, date_end: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not date_start and not date_end:
        return df, {'filter_mode': 'none', 'date_filter_applied': False, 'rows_before_date_filter': int(len(df)), 'rows_after_date_filter': int(len(df))}
    out = normalise_time_column(df)
    if 'time_utc' not in out.columns:
        return out, {'filter_mode': 'none', 'date_filter_applied': False, 'warning': 'time_utc missing', 'rows_before_date_filter': int(len(df)), 'rows_after_date_filter': int(len(out))}
    times = pd.to_datetime(out['time_utc'], utc=True, errors='coerce')
    mask = pd.Series(True, index=out.index)
    start = pd.to_datetime(date_start, utc=True) if date_start else None
    end = pd.to_datetime(date_end, utc=True) if date_end else None
    if start is not None:
        mask &= times >= start
    if end is not None:
        mask &= times < end
    filtered = out.loc[mask].reset_index(drop=True)
    return filtered, {
        'filter_mode': 'date_range',
        'date_filter_applied': True,
        'date_start_utc': str(start) if start is not None else None,
        'date_end_utc': str(end) if end is not None else None,
        'date_end_is_exclusive': True,
        'rows_before_date_filter': int(len(out)),
        'rows_after_date_filter': int(len(filtered)),
    }


def _load_direction_dataframe(symbol: str, cfg: dict[str, Any], *, date_start: str | None, date_end: str | None, max_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    tcfg = cfg.get('training', {}) or {}
    use_pre = bool(tcfg.get('use_pregenerated_direction_data', True))
    require_pre = bool(tcfg.get('require_pregenerated_direction_data', False))
    pre_path = _pregenerated_path(symbol, cfg)
    source = 'processed_csv_generated_direction_targets'
    if use_pre and pre_path.exists():
        df = pd.read_csv(pre_path)
        source = 'pregenerated_direction_csv'
    elif use_pre and require_pre:
        raise FileNotFoundError(f'Pregenerated direction dataset not found: {pre_path}')
    else:
        df = read_processed_csv(symbol, cfg)

    raw_rows = int(len(df))
    df, date_info = _filter_date_range(df, date_start, date_end)

    # If a pregenerated dataset is filtered by date, the final horizon_bars rows
    # can contain targets that were generated using bars beyond the requested
    # training end date. Drop that tail by default to keep the training window
    # causally closed. This is not needed when targets are generated after the
    # date filter, but it is safest for pregenerated target CSVs.
    tail_drop_info = {
        'enabled': False,
        'rows_dropped': 0,
        'reason': None,
    }
    horizon_bars = int((cfg.get('labels') or {}).get('horizon_bars', 0) or 0)
    drop_tail = bool(tcfg.get('drop_tail_after_date_filter', True))
    if (
        source == 'pregenerated_direction_csv'
        and drop_tail
        and date_info.get('date_filter_applied')
        and date_end not in (None, '')
        and horizon_bars > 0
    ):
        before_tail_drop = int(len(df))
        drop_n = min(horizon_bars, before_tail_drop)
        if drop_n > 0:
            df = df.iloc[:-drop_n].reset_index(drop=True) if drop_n < before_tail_drop else df.iloc[:0].reset_index(drop=True)
        tail_drop_info = {
            'enabled': True,
            'rows_dropped': int(drop_n),
            'horizon_bars': int(horizon_bars),
            'rows_before_tail_drop': int(before_tail_drop),
            'rows_after_tail_drop': int(len(df)),
            'reason': 'pregenerated labels may look forward beyond filtered date_end',
        }

    if max_rows:
        df = df.tail(int(max_rows)).reset_index(drop=True)

    targets_generated = False
    if 'direction_target' not in df.columns:
        df = generate_direction_targets(df, symbol, cfg)
        targets_generated = True

    df = ensure_analytic_signal_features(df, cfg)

    return df, {
        'source': source,
        'pregenerated_path': str(pre_path),
        'pregenerated_found': bool(pre_path.exists()),
        'raw_rows': raw_rows,
        'rows_after_filter': int(len(df)),
        'max_rows': max_rows,
        'date_filter': date_info,
        'tail_drop_after_date_filter': tail_drop_info,
        'targets_generated': targets_generated,
    }


def _class_counts(y: np.ndarray) -> dict[str, int]:
    return {DIRECTION_CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)}


def _class_weights(y: np.ndarray, cfg: dict[str, Any]) -> torch.Tensor | None:
    tcfg = cfg.get('training', {}) or {}
    if not bool(tcfg.get('use_class_weights', True)):
        return None
    counts = np.array([(y == i).sum() for i in range(3)], dtype=float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (3.0 * counts)
    max_weight = float(tcfg.get('max_class_weight', 10.0))
    weights = np.clip(weights, 1.0 / max_weight, max_weight)
    explicit = tcfg.get('direction_class_weights') or {}
    for name, idx in {'sell': 0, 'no_trade': 1, 'buy': 2}.items():
        if name in explicit and explicit[name] is not None:
            weights[idx] = float(explicit[name])
    return torch.tensor(weights, dtype=torch.float32)




_CLASS_NAME_TO_INDEX = {'sell': 0, 'no_trade': 1, 'buy': 2}
_INDEX_TO_CLASS_NAME = {0: 'sell', 1: 'no_trade', 2: 'buy'}


def _class_cfg_value(mapping: Any, class_name: str, default: Any = None) -> Any:
    """Read a class-specific config value using common key spellings."""
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        idx = _CLASS_NAME_TO_INDEX[class_name]
        keys = (
            class_name,
            class_name.upper(),
            DIRECTION_CLASS_NAMES[idx],
            DIRECTION_CLASS_NAMES[idx].lower(),
            str(idx),
            idx,
        )
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return default
    return mapping


def _class_float_dict(mapping: Any, defaults: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, default in defaults.items():
        value = _class_cfg_value(mapping, name, default)
        out[name] = float(default if value is None else value)
    return out


def _class_optional_int_dict(mapping: Any, defaults: dict[str, int | None]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for name, default in defaults.items():
        value = _class_cfg_value(mapping, name, default)
        if value is None:
            out[name] = default
        else:
            out[name] = int(value)
    return out


def _class_bool_dict(mapping: Any, defaults: dict[str, bool]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for name, default in defaults.items():
        value = _class_cfg_value(mapping, name, default)
        out[name] = bool(default if value is None else value)
    return out


def _curriculum_sampler_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return normalised training curriculum sampler settings.

    The curriculum changes only the training batches. Validation/replay remain
    untouched and realistic. Ratios are class exposure ratios, not loss weights.
    """
    tcfg = cfg.get('training', {}) or {}
    raw = tcfg.get('curriculum_sampler') or tcfg.get('class_curriculum_sampler') or {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('training.curriculum_sampler must be a mapping when provided.')
    enabled = bool(tcfg.get('use_curriculum_sampler', False) or raw.get('enabled', False))
    start_epoch = int(raw.get('start_epoch', 1) or 1)
    end_epoch = int(raw.get('end_epoch', max(start_epoch, int(tcfg.get('epochs', 1) or 1))) or start_epoch)
    if end_epoch < start_epoch:
        end_epoch = start_epoch
    schedule = str(raw.get('schedule', 'linear') or 'linear').lower()
    if schedule not in {'linear', 'cosine', 'constant', 'step'}:
        raise ValueError("training.curriculum_sampler.schedule must be linear, cosine, constant, or step.")
    start_ratios = _class_float_dict(
        raw.get('start_ratios'),
        {'sell': 1.0, 'no_trade': 2.0, 'buy': 1.0},
    )
    end_ratios = _class_float_dict(
        raw.get('end_ratios'),
        {'sell': 1.0, 'no_trade': 10.0, 'buy': 1.0},
    )
    for where, ratios in {'start_ratios': start_ratios, 'end_ratios': end_ratios}.items():
        for name, value in ratios.items():
            if value < 0:
                raise ValueError(f'training.curriculum_sampler.{where}.{name} must be >= 0.')
    min_class_samples = _class_optional_int_dict(
        raw.get('min_class_samples'),
        {'sell': None, 'no_trade': None, 'buy': None},
    )
    max_class_samples = _class_optional_int_dict(
        raw.get('max_class_samples'),
        {'sell': None, 'no_trade': None, 'buy': None},
    )
    with_replacement = _class_bool_dict(
        raw.get('with_replacement'),
        {'sell': True, 'no_trade': False, 'buy': True},
    )
    return {
        'enabled': bool(enabled),
        'schedule': schedule,
        'start_epoch': start_epoch,
        'end_epoch': end_epoch,
        'start_ratios': start_ratios,
        'end_ratios': end_ratios,
        'anchor_class': str(raw.get('anchor_class', 'max_positive') or 'max_positive').lower(),
        'anchor_count': None if raw.get('anchor_count') in (None, '') else int(raw.get('anchor_count')),
        'epoch_size_multiplier': float(raw.get('epoch_size_multiplier', 1.0) or 1.0),
        'min_class_samples': min_class_samples,
        'max_class_samples': max_class_samples,
        'with_replacement': with_replacement,
        'shuffle': bool(raw.get('shuffle', True)),
    }


def _curriculum_phase(settings: dict[str, Any], epoch: int) -> float:
    if not settings.get('enabled'):
        return 0.0
    start_epoch = int(settings.get('start_epoch', 1) or 1)
    end_epoch = int(settings.get('end_epoch', start_epoch) or start_epoch)
    if epoch <= start_epoch:
        raw_phase = 0.0
    elif epoch >= end_epoch:
        raw_phase = 1.0
    elif end_epoch == start_epoch:
        raw_phase = 1.0
    else:
        raw_phase = (float(epoch) - float(start_epoch)) / max(float(end_epoch - start_epoch), 1.0)
    schedule = str(settings.get('schedule', 'linear') or 'linear').lower()
    if schedule == 'cosine':
        return float(0.5 - 0.5 * math.cos(math.pi * raw_phase))
    if schedule == 'step':
        return 0.0 if raw_phase < 1.0 else 1.0
    if schedule == 'constant':
        return 0.0
    return float(raw_phase)


def _curriculum_ratios(settings: dict[str, Any], epoch: int) -> dict[str, float]:
    phase = _curriculum_phase(settings, epoch)
    start = settings.get('start_ratios') or {}
    end = settings.get('end_ratios') or {}
    ratios: dict[str, float] = {}
    for name in ('sell', 'no_trade', 'buy'):
        s = float(start.get(name, 0.0) or 0.0)
        e = float(end.get(name, s) or 0.0)
        ratios[name] = float(s + phase * (e - s))
    return ratios


def _curriculum_available_counts(y_train: np.ndarray) -> dict[str, int]:
    y_train = np.asarray(y_train, dtype=int)
    return {name: int((y_train == idx).sum()) for name, idx in _CLASS_NAME_TO_INDEX.items()}


def _curriculum_anchor_count(y_train: np.ndarray, settings: dict[str, Any]) -> int:
    available = _curriculum_available_counts(y_train)
    explicit = settings.get('anchor_count')
    if explicit not in (None, ''):
        return max(int(explicit), 1)
    anchor_class = str(settings.get('anchor_class', 'max_positive') or 'max_positive').lower()
    sell = int(available.get('sell', 0))
    buy = int(available.get('buy', 0))
    positives = [v for v in (sell, buy) if v > 0]
    if anchor_class in {'sell', 'no_trade', 'buy'}:
        return max(int(available.get(anchor_class, 0)), 1)
    if anchor_class in {'min_positive', 'min_trade'}:
        return max(min(positives) if positives else 1, 1)
    if anchor_class in {'mean_positive', 'mean_trade', 'avg_positive', 'average_positive'}:
        return max(int(round(float(sum(positives)) / max(len(positives), 1))) if positives else 1, 1)
    if anchor_class in {'total_positive', 'total_trade', 'trade_total'}:
        return max(sell + buy, 1)
    # Default: each BUY/SELL ratio unit corresponds to approximately the larger
    # available positive class count. This avoids extreme epoch sizes while using
    # nearly all positive examples each epoch.
    return max(max(positives) if positives else 1, 1)


def _curriculum_sample_counts(y_train: np.ndarray, settings: dict[str, Any], epoch: int) -> dict[str, int]:
    if not settings.get('enabled'):
        return _curriculum_available_counts(y_train)
    available = _curriculum_available_counts(y_train)
    ratios = _curriculum_ratios(settings, epoch)
    anchor = _curriculum_anchor_count(y_train, settings)
    multiplier = max(float(settings.get('epoch_size_multiplier', 1.0) or 1.0), 0.0)
    min_samples = settings.get('min_class_samples') or {}
    max_samples = settings.get('max_class_samples') or {}
    replacement = settings.get('with_replacement') or {}
    counts: dict[str, int] = {}
    for name in ('sell', 'no_trade', 'buy'):
        available_n = int(available.get(name, 0))
        if available_n <= 0 or float(ratios.get(name, 0.0) or 0.0) <= 0.0:
            counts[name] = 0
            continue
        n = int(round(float(anchor) * float(ratios[name]) * multiplier))
        if ratios[name] > 0 and n <= 0:
            n = 1
        min_n = min_samples.get(name)
        if min_n is not None:
            n = max(n, int(min_n))
        max_n = max_samples.get(name)
        if max_n is not None:
            n = min(n, int(max_n))
        if not bool(replacement.get(name, False)):
            n = min(n, available_n)
        counts[name] = max(int(n), 0)
    return counts


def _make_curriculum_epoch_indices(
    train_idx: np.ndarray,
    y_all: np.ndarray,
    settings: dict[str, Any],
    epoch: int,
    seed: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_idx = np.asarray(train_idx, dtype=int)
    y_train = np.asarray(y_all, dtype=int)[train_idx]
    available = _curriculum_available_counts(y_train)
    ratios = _curriculum_ratios(settings, epoch)
    sample_counts = _curriculum_sample_counts(y_train, settings, epoch)
    replacement = settings.get('with_replacement') or {}
    rng = np.random.default_rng(None if seed is None else int(seed) % (2**32 - 1))
    chosen_local: list[np.ndarray] = []
    for name in ('sell', 'no_trade', 'buy'):
        class_idx = _CLASS_NAME_TO_INDEX[name]
        local_positions = np.where(y_train == class_idx)[0]
        need = int(sample_counts.get(name, 0) or 0)
        if need <= 0 or len(local_positions) == 0:
            continue
        replace = bool(replacement.get(name, False))
        if not replace:
            need = min(need, int(len(local_positions)))
        selected = rng.choice(local_positions, size=need, replace=replace)
        chosen_local.append(np.asarray(selected, dtype=int))
    if chosen_local:
        local = np.concatenate(chosen_local).astype(int)
    else:
        local = np.arange(len(train_idx), dtype=int)
    if bool(settings.get('shuffle', True)) and len(local) > 1:
        rng.shuffle(local)
    epoch_indices = train_idx[local].astype(int)
    trade_samples = int(sample_counts.get('sell', 0) or 0) + int(sample_counts.get('buy', 0) or 0)
    no_trade_samples = int(sample_counts.get('no_trade', 0) or 0)
    plan = {
        'enabled': True,
        'epoch': int(epoch),
        'phase': float(_curriculum_phase(settings, epoch)),
        'schedule': str(settings.get('schedule', 'linear')),
        'ratios': {k: float(v) for k, v in ratios.items()},
        'anchor_class': str(settings.get('anchor_class', 'max_positive')),
        'anchor_count': int(_curriculum_anchor_count(y_train, settings)),
        'available_counts': {k: int(v) for k, v in available.items()},
        'sample_counts': {k: int(v) for k, v in sample_counts.items()},
        'with_replacement': {k: bool(v) for k, v in replacement.items()},
        'total_samples': int(len(epoch_indices)),
        'trade_samples': int(trade_samples),
        'no_trade_samples': int(no_trade_samples),
        'no_trade_to_trade_sample_ratio': float(no_trade_samples / trade_samples) if trade_samples > 0 else None,
    }
    return epoch_indices, plan


def _curriculum_static_report(y_train: np.ndarray, settings: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(settings.get('enabled', False))
    report = {
        'enabled': enabled,
        'available_counts': _curriculum_available_counts(np.asarray(y_train, dtype=int)),
    }
    if not enabled:
        return report
    start_epoch = int(settings.get('start_epoch', 1) or 1)
    end_epoch = int(settings.get('end_epoch', start_epoch) or start_epoch)
    report.update({
        'schedule': str(settings.get('schedule', 'linear')),
        'start_epoch': start_epoch,
        'end_epoch': end_epoch,
        'start_ratios': {k: float(v) for k, v in (settings.get('start_ratios') or {}).items()},
        'end_ratios': {k: float(v) for k, v in (settings.get('end_ratios') or {}).items()},
        'anchor_class': str(settings.get('anchor_class', 'max_positive')),
        'anchor_count': int(_curriculum_anchor_count(np.asarray(y_train, dtype=int), settings)),
        'with_replacement': {k: bool(v) for k, v in (settings.get('with_replacement') or {}).items()},
        'start_sample_counts': _curriculum_sample_counts(np.asarray(y_train, dtype=int), settings, start_epoch),
        'end_sample_counts': _curriculum_sample_counts(np.asarray(y_train, dtype=int), settings, end_epoch),
        'objective': 'Training-only class exposure curriculum; validation and replay are not resampled.',
    })
    return report


def _format_curriculum_plan(plan: dict[str, Any] | None) -> str:
    if not plan:
        return ''
    counts = plan.get('sample_counts') or {}
    ratios = plan.get('ratios') or {}
    return (
        ' curriculum='
        f"phase={float(plan.get('phase', 0.0)):.3f} "
        f"samples[sell={int(counts.get('sell', 0) or 0)} "
        f"no_trade={int(counts.get('no_trade', 0) or 0)} "
        f"buy={int(counts.get('buy', 0) or 0)}] "
        f"ratios[sell={float(ratios.get('sell', 0.0) or 0.0):.2f} "
        f"no_trade={float(ratios.get('no_trade', 0.0) or 0.0):.2f} "
        f"buy={float(ratios.get('buy', 0.0) or 0.0):.2f}]"
    )

def _average_decision_scores_by_actual_label(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Return mean model decision scores grouped by the true direction label.

    The rows are the actual labels in the validation data and the columns are
    the model's average softmax scores for each possible decision. This makes it
    easy to see, for example, how strongly the model scores BUY on rows that are
    actually labelled BUY, and how much probability it leaks into NO_TRADE or
    SELL for those same rows.
    """
    out: dict[str, Any] = {}
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    for actual_idx, actual_name in DIRECTION_CLASS_NAMES.items():
        actual_key = f'actual_{actual_name.lower()}'
        mask = y_true == int(actual_idx)
        count = int(mask.sum())
        row: dict[str, Any] = {'count': count}
        if count > 0:
            means = probs[mask].mean(axis=0)
            for decision_idx, decision_name in DIRECTION_CLASS_NAMES.items():
                row[f'average_{decision_name.lower()}_decision_score'] = float(means[int(decision_idx)])
        else:
            for _, decision_name in DIRECTION_CLASS_NAMES.items():
                row[f'average_{decision_name.lower()}_decision_score'] = None
        out[actual_key] = row
    return out




def _format_actual_label_decision_scores_for_terminal(metrics: dict[str, Any]) -> str:
    """Compact one-line summary of validation decision scores by actual label."""
    by_actual = metrics.get('average_decision_scores_by_actual_label') or {}
    if not isinstance(by_actual, dict) or not by_actual:
        return ''

    parts: list[str] = []
    # Keep a stable human-readable class order in the terminal output.
    for actual_name in ('sell', 'no_trade', 'buy'):
        row = by_actual.get(f'actual_{actual_name}') or {}
        if not isinstance(row, dict):
            continue
        count = int(row.get('count') or 0)
        score_text: list[str] = []
        for decision_name in ('sell', 'no_trade', 'buy'):
            value = row.get(f'average_{decision_name}_decision_score')
            if value is None:
                score_text.append(f'{decision_name}=NA')
            else:
                score_text.append(f'{decision_name}={float(value):.3f}')
        parts.append(f'actual_{actual_name}[n={count} ' + ' '.join(score_text) + ']')

    return ' decision_scores_by_actual=' + ' | '.join(parts) if parts else ''


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out['accuracy'] = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1s = []
    for idx, name in DIRECTION_CLASS_NAMES.items():
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        out[f'{name.lower()}_precision'] = float(precision)
        out[f'{name.lower()}_recall'] = float(recall)
        out[f'{name.lower()}_f1'] = float(f1)
        out[f'{name.lower()}_predicted'] = int((y_pred == idx).sum())
        out[f'{name.lower()}_true'] = int((y_true == idx).sum())
    out['macro_f1'] = float(np.mean(f1s)) if f1s else 0.0
    trade_mask = y_pred != 1
    out['predicted_trades'] = int(trade_mask.sum())
    if trade_mask.any():
        out['trade_direction_accuracy'] = float((y_true[trade_mask] == y_pred[trade_mask]).mean())
    else:
        out['trade_direction_accuracy'] = 0.0
    if probs is not None and len(probs):
        probs = np.asarray(probs, dtype=float)
        out['mean_selected_probability'] = float(np.max(probs, axis=1).mean())
        by_actual = _average_decision_scores_by_actual_label(y_true, probs)
        out['average_decision_scores_by_actual_label'] = by_actual
        # Flat aliases make the same values easy to grep, sort or use as a
        # model_selection_metric without having to parse the nested table.
        for actual_key, row in by_actual.items():
            for decision_idx, decision_name in DIRECTION_CLASS_NAMES.items():
                metric_key = f'{actual_key}_average_{decision_name.lower()}_decision_score'
                out[metric_key] = row.get(f'average_{decision_name.lower()}_decision_score')
    return out


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)




def _normalise_training_side(value: Any) -> str:
    """Return training side selector: both, buy, or sell."""
    if value is None:
        return 'both'
    side = str(value).strip().lower()
    aliases = {
        '': 'both',
        'all': 'both',
        'both_sides': 'both',
        'combined': 'both',
        'long': 'buy',
        'short': 'sell',
    }
    side = aliases.get(side, side)
    if side not in {'both', 'buy', 'sell'}:
        raise ValueError(f"Unsupported training side {value!r}; use both, buy, or sell.")
    return side


def _training_side(cfg: dict[str, Any]) -> str:
    tcfg = cfg.get('training', {}) or {}
    raw = cfg.get('_training_side', tcfg.get('side_setup_train_side', tcfg.get('train_side', 'both')))
    return _normalise_training_side(raw)


def _apply_training_side_to_config(cfg: dict[str, Any], side: str) -> dict[str, Any]:
    """Apply BUY-only/SELL-only training and replay filters to a config copy.

    Side-specific model files are normally separated by the local runner via
    generated per-task configs. This helper only changes the learning/replay
    semantics; it does not silently rewrite paths.
    """
    side = _normalise_training_side(side)
    cfg = copy.deepcopy(cfg)
    cfg['_training_side'] = side
    tcfg = cfg.setdefault('training', {})
    tcfg['side_setup_train_side'] = side
    tcfg['train_side'] = side
    if side in {'buy', 'sell'}:
        # Train only the selected setup head. The other head is still present in
        # the checkpoint for architecture compatibility, but its setup loss is
        # zero and replay is side-filtered below.
        tcfg['buy_setup_loss_weight'] = 1.0 if side == 'buy' else 0.0
        tcfg['sell_setup_loss_weight'] = 1.0 if side == 'sell' else 0.0
        rcfg = cfg.setdefault('replay', {})
        rcfg['allow_buy'] = side == 'buy'
        rcfg['allow_sell'] = side == 'sell'
    return cfg

def _branch_auxiliary_enabled(cfg: dict[str, Any]) -> bool:
    tcfg = cfg.get('training', {}) or {}
    explicit = tcfg.get('use_branch_auxiliary_loss')
    if explicit is not None:
        return _cfg_bool(explicit, False)
    return float(tcfg.get('branch_auxiliary_loss_weight', 0.0) or 0.0) > 0.0


def _branch_auxiliary_loss_weight(cfg: dict[str, Any]) -> float:
    tcfg = cfg.get('training', {}) or {}
    return max(0.0, float(tcfg.get('branch_auxiliary_loss_weight', 0.2) or 0.0))


def _branch_auxiliary_branch_weights(cfg: dict[str, Any]) -> torch.Tensor:
    tcfg = cfg.get('training', {}) or {}
    explicit = tcfg.get('branch_auxiliary_loss_weights') or {}
    if not isinstance(explicit, dict):
        explicit = {}
    values = []
    for name in ('sell', 'no_trade', 'buy'):
        values.append(float(explicit.get(name, explicit.get(name.upper(), 1.0)) or 0.0))
    weights = torch.tensor(values, dtype=torch.float32)
    # Avoid a silent divide-by-zero if all branch weights are configured as 0.
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones(3, dtype=torch.float32)
    return weights


def _branch_auxiliary_pos_weights(y: np.ndarray, cfg: dict[str, Any]) -> torch.Tensor | None:
    """Optional BCE positive weights for the three one-vs-rest branch losses."""
    tcfg = cfg.get('training', {}) or {}
    if not _cfg_bool(tcfg.get('use_branch_auxiliary_pos_weights'), True):
        return None

    y = np.asarray(y, dtype=int)
    n = max(int(len(y)), 1)
    mode = str(tcfg.get('branch_auxiliary_pos_weight_mode', 'inverse_frequency') or 'inverse_frequency').lower()
    if mode in {'none', 'off', 'false', 'disabled'}:
        return None
    if mode not in {'inverse_frequency', 'neg_over_pos', 'balanced'}:
        raise ValueError(
            f"Unsupported training.branch_auxiliary_pos_weight_mode={mode!r}. "
            "Use inverse_frequency/neg_over_pos/balanced or none."
        )

    min_weight = float(tcfg.get('branch_auxiliary_min_pos_weight', 0.25) or 0.0)
    max_weight = float(tcfg.get('branch_auxiliary_max_pos_weight', 5.0) or 0.0)
    if max_weight <= 0:
        max_weight = float('inf')
    explicit = tcfg.get('branch_auxiliary_pos_weights') or {}
    if not isinstance(explicit, dict):
        explicit = {}

    values: list[float] = []
    for idx, name in ((0, 'sell'), (1, 'no_trade'), (2, 'buy')):
        pos = max(float((y == idx).sum()), 1.0)
        neg = max(float(n) - pos, 1.0)
        value = neg / pos
        value = float(np.clip(value, min_weight, max_weight))
        if name in explicit and explicit[name] is not None:
            value = float(explicit[name])
        elif name.upper() in explicit and explicit[name.upper()] is not None:
            value = float(explicit[name.upper()])
        values.append(value)
    return torch.tensor(values, dtype=torch.float32)


def _branch_logits_from_outputs(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return raw logits in SELL, NO_TRADE, BUY order for branch BCE losses."""
    if all(key in outputs for key in ('sell_logit', 'no_trade_logit', 'buy_logit')):
        return torch.stack([
            outputs['sell_logit'].view(-1),
            outputs['no_trade_logit'].view(-1),
            outputs['buy_logit'].view(-1),
        ], dim=1)
    if 'branch_logits' in outputs:
        return outputs['branch_logits']
    return outputs['direction_logits']


def _branch_auxiliary_loss_components(
    outputs: dict[str, torch.Tensor],
    y: torch.Tensor,
    bce: nn.Module,
    branch_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute separate one-vs-rest BCE losses for SELL, NO_TRADE and BUY.

    The returned combined loss is the weighted mean of the three branch losses.
    It is intended to be added to the normal multiclass CrossEntropyLoss using
    training.branch_auxiliary_loss_weight.
    """
    branch_logits = _branch_logits_from_outputs(outputs)
    targets = F.one_hot(y.long(), num_classes=3).to(dtype=branch_logits.dtype, device=branch_logits.device)
    loss_matrix = bce(branch_logits, targets)
    per_branch = loss_matrix.mean(dim=0)
    if branch_weights is not None:
        weights = branch_weights.to(device=branch_logits.device, dtype=branch_logits.dtype)
        combined = (per_branch * weights).sum() / torch.clamp(weights.sum(), min=1e-12)
    else:
        combined = per_branch.mean()
    return combined, {
        'branch_auxiliary_loss': float(combined.detach().cpu().item()),
        'branch_auxiliary_sell_loss': float(per_branch[0].detach().cpu().item()),
        'branch_auxiliary_no_trade_loss': float(per_branch[1].detach().cpu().item()),
        'branch_auxiliary_buy_loss': float(per_branch[2].detach().cpu().item()),
    }


def _mean_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if vals:
            out[key] = float(np.mean(vals))
    return out




def _side_setup_training_mode(cfg: dict[str, Any]) -> bool:
    """Return True when the dataset/config should use side-specific setup heads.

    The strong-setup labelling path can be enabled from either training.target_mode
    or labels.method / labels.strong_setup.output_mode. This guard prevents Kaggle
    runs from silently falling back to the old 3-class hierarchical gate when a
    copied config is missing training.target_mode.
    """
    tcfg = cfg.get('training', {}) or {}
    mcfg = cfg.get('model', {}) or {}
    lcfg = cfg.get('labels', {}) or {}
    mode = str(tcfg.get('target_mode', mcfg.get('target_mode', 'direction')) or 'direction').strip().lower()
    if mode in {'side_setup', 'side_setup_ranking', 'setup_ranking', 'side_specific_setup'}:
        return True
    label_method = str(lcfg.get('method', lcfg.get('label_method', '')) or '').strip().lower()
    strong_cfg = lcfg.get('strong_setup', {}) or {}
    output_mode = str(strong_cfg.get('output_mode', '') or '').strip().lower() if isinstance(strong_cfg, dict) else ''
    return (
        label_method in {'strong_setup_v1', 'side_setup_v1', 'side_setup_ranking'}
        or output_mode in {'event_based', 'side_setup', 'side_setup_ranking', 'setup_ranking'}
    )


def _setup_loss_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    tcfg = cfg.get('training', {}) or {}
    mcfg = cfg.get('model', {}) or {}
    train_side = _training_side(cfg)
    buy_weight = float(tcfg.get('buy_setup_loss_weight', tcfg.get('setup_loss_weight', 1.0)) or 0.0)
    sell_weight = float(tcfg.get('sell_setup_loss_weight', tcfg.get('setup_loss_weight', 1.0)) or 0.0)
    if train_side == 'buy':
        sell_weight = 0.0
    elif train_side == 'sell':
        buy_weight = 0.0
    return {
        'train_side': train_side,
        'buy_setup_loss_weight': buy_weight,
        'sell_setup_loss_weight': sell_weight,
        'use_setup_quality_loss': _cfg_bool(tcfg.get('use_setup_quality_loss'), bool(mcfg.get('use_setup_quality_head', True))),
        'setup_quality_loss_weight': float(tcfg.get('setup_quality_loss_weight', 0.10) or 0.0),
        'setup_quality_scale': float(mcfg.get('setup_quality_scale', mcfg.get('edge_pips_scale', 12.0)) or 12.0),
        'buy_setup_pos_weight': tcfg.get('_buy_setup_pos_weight', tcfg.get('buy_setup_pos_weight', tcfg.get('setup_pos_weight', 'auto'))),
        'sell_setup_pos_weight': tcfg.get('_sell_setup_pos_weight', tcfg.get('sell_setup_pos_weight', tcfg.get('setup_pos_weight', 'auto'))),
    }


def _auto_setup_pos_weight(targets: np.ndarray | None, mask: np.ndarray | None, cfg: dict[str, Any], side: str) -> float:
    tcfg = cfg.get('training', {}) or {}
    explicit = tcfg.get(f'{side}_setup_pos_weight', tcfg.get('setup_pos_weight', 'auto'))
    if explicit not in (None, '', 'auto', 'balanced', 'inverse_frequency'):
        try:
            return max(float(explicit), 0.0)
        except Exception:
            pass
    if targets is None or mask is None:
        return 1.0
    target = np.asarray(targets, dtype=int)
    valid = np.asarray(mask, dtype=bool)
    pos = max(float(((target == 1) & valid).sum()), 1.0)
    neg = max(float(((target == 0) & valid).sum()), 1.0)
    value = neg / pos
    min_w = float(tcfg.get('min_setup_pos_weight', 0.25) or 0.25)
    max_w = float(tcfg.get('max_setup_pos_weight', 10.0) or 10.0)
    return float(np.clip(value, min_w, max_w))


def _side_setup_loss_components(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    settings = _setup_loss_settings(cfg)
    def _safe_float_setting(value: Any, default: float = 1.0) -> float:
        try:
            return float(value if value not in (None, '') else default)
        except Exception:
            return float(default)
    device = outputs['buy_setup_logit'].device
    zero = outputs['buy_setup_logit'].sum() * 0.0

    def _one_side(side: str) -> tuple[torch.Tensor, int, float, float, float]:
        logit = outputs[f'{side}_setup_logit'].view(-1)
        target = batch[f'{side}_setup_target'].to(device=device).float().view(-1)
        mask = batch.get(f'has_{side}_setup_target')
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)
        else:
            mask = mask.to(device=device).bool().view(-1)
        if not bool(mask.any()):
            return zero, 0, 0.0, 0.0, 0.0
        raw_pos_weight = settings.get(f'{side}_setup_pos_weight', 1.0)
        try:
            pos_weight_value = float(raw_pos_weight if raw_pos_weight not in (None, '') else 1.0)
        except Exception:
            pos_weight_value = 1.0
        pos_weight = torch.tensor([max(pos_weight_value, 0.0)], dtype=torch.float32, device=device)
        loss_vec = F.binary_cross_entropy_with_logits(logit[mask], target[mask], pos_weight=pos_weight, reduction='none')
        loss = loss_vec.mean()
        prob = torch.sigmoid(logit[mask])
        pred = prob >= 0.5
        truth = target[mask] >= 0.5
        tp = int((pred & truth).sum().detach().cpu().item())
        fp = int((pred & ~truth).sum().detach().cpu().item())
        fn = int((~pred & truth).sum().detach().cpu().item())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return loss, int(mask.sum().detach().cpu().item()), float(precision), float(recall), float(f1)

    buy_loss, buy_count, buy_precision, buy_recall, buy_f1 = _one_side('buy')
    sell_loss, sell_count, sell_precision, sell_recall, sell_f1 = _one_side('sell')

    quality_loss = zero
    quality_count = 0
    if (
        bool(settings.get('use_setup_quality_loss'))
        and float(settings.get('setup_quality_loss_weight', 0.0) or 0.0) > 0.0
        and 'setup_quality' in outputs
        and 'buy_setup_quality_score' in batch
        and 'sell_setup_quality_score' in batch
    ):
        qmask_buy = batch.get('has_buy_setup_target')
        qmask_sell = batch.get('has_sell_setup_target')
        if qmask_buy is None:
            qmask_buy = torch.ones_like(batch['buy_setup_quality_score'], dtype=torch.bool)
        else:
            qmask_buy = qmask_buy.to(device=device).bool()
        if qmask_sell is None:
            qmask_sell = torch.ones_like(batch['sell_setup_quality_score'], dtype=torch.bool)
        else:
            qmask_sell = qmask_sell.to(device=device).bool()
        target_q = torch.stack([
            batch['buy_setup_quality_score'].to(device=device, dtype=torch.float32),
            batch['sell_setup_quality_score'].to(device=device, dtype=torch.float32),
        ], dim=1)
        pred_q = outputs['setup_quality']
        mask_q = torch.stack([qmask_buy, qmask_sell], dim=1).bool()
        train_side = str(settings.get('train_side', 'both') or 'both').lower()
        if train_side == 'buy':
            mask_q[:, 1] = False
        elif train_side == 'sell':
            mask_q[:, 0] = False
        if bool(mask_q.any()):
            scale = max(float(settings.get('setup_quality_scale', 12.0) or 12.0), 1e-6)
            quality_loss = F.smooth_l1_loss(pred_q[mask_q] / scale, target_q[mask_q] / scale)
            quality_count = int(mask_q.sum().detach().cpu().item())

    total = (
        float(settings.get('buy_setup_loss_weight', 1.0)) * buy_loss
        + float(settings.get('sell_setup_loss_weight', 1.0)) * sell_loss
        + float(settings.get('setup_quality_loss_weight', 0.0)) * quality_loss
    )
    return total, {
        'loss': float(total.detach().cpu().item()),
        'setup_loss': float(total.detach().cpu().item()),
        'buy_setup_loss': float(buy_loss.detach().cpu().item()),
        'sell_setup_loss': float(sell_loss.detach().cpu().item()),
        'buy_setup_target_count': float(buy_count),
        'sell_setup_target_count': float(sell_count),
        'buy_setup_precision_05': float(buy_precision),
        'buy_setup_recall_05': float(buy_recall),
        'buy_setup_f1_05': float(buy_f1),
        'sell_setup_precision_05': float(sell_precision),
        'sell_setup_recall_05': float(sell_recall),
        'sell_setup_f1_05': float(sell_f1),
        'setup_quality_loss': float(quality_loss.detach().cpu().item()),
        'setup_quality_target_count': float(quality_count),
        'buy_setup_loss_weight': float(settings.get('buy_setup_loss_weight', 1.0)),
        'sell_setup_loss_weight': float(settings.get('sell_setup_loss_weight', 1.0)),
        'setup_quality_loss_weight': float(settings.get('setup_quality_loss_weight', 0.0)),
        'buy_setup_pos_weight': _safe_float_setting(settings.get('buy_setup_pos_weight', 1.0), 1.0),
        'sell_setup_pos_weight': _safe_float_setting(settings.get('sell_setup_pos_weight', 1.0), 1.0),
    }


def _hierarchical_loss_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    tcfg = cfg.get('training', {}) or {}
    edge_enabled = _cfg_bool(tcfg.get('use_edge_pips_loss', tcfg.get('use_edge_pips_head')), True)
    return {
        'gate_loss_weight': float(tcfg.get('gate_loss_weight', 1.0) or 0.0),
        'side_direction_loss_weight': float(tcfg.get('side_direction_loss_weight', tcfg.get('direction_loss_weight', 1.0)) or 0.0),
        'edge_pips_loss_weight': float(tcfg.get('edge_pips_loss_weight', 0.20) or 0.0),
        'use_edge_pips_loss': bool(edge_enabled),
        'edge_pips_scale': float((cfg.get('model') or {}).get('edge_pips_scale', (cfg.get('labels') or {}).get('take_profit_pips', 10.0)) or 10.0),
        'use_analytic_signal_agreement_loss': _cfg_bool(tcfg.get('use_analytic_signal_agreement_loss'), False),
        'analytic_signal_agreement_loss_weight': float(tcfg.get('analytic_signal_agreement_loss_weight', 0.0) or 0.0),
        'gate_pos_weight': tcfg.get('gate_pos_weight', 1.0),
    }


def _gate_pos_weight_tensor(y_train: np.ndarray, cfg: dict[str, Any], device: str) -> torch.Tensor | None:
    settings = _hierarchical_loss_settings(cfg)
    raw = settings.get('gate_pos_weight')
    if raw in (None, '', 'none', 'off', False):
        return None
    if isinstance(raw, str) and raw.strip().lower() in {'auto', 'balanced', 'inverse_frequency'}:
        y = np.asarray(y_train, dtype=int)
        pos = max(float((y != 1).sum()), 1.0)
        neg = max(float((y == 1).sum()), 1.0)
        value = neg / pos
        max_w = float((cfg.get('training') or {}).get('max_gate_pos_weight', 5.0) or 5.0)
        min_w = float((cfg.get('training') or {}).get('min_gate_pos_weight', 0.25) or 0.25)
        value = float(np.clip(value, min_w, max_w))
    else:
        value = float(raw)
    if value <= 0.0:
        return None
    return torch.tensor([value], dtype=torch.float32, device=device)


def _side_class_weight_tensor(y_train: np.ndarray, cfg: dict[str, Any], device: str) -> torch.Tensor | None:
    tcfg = cfg.get('training', {}) or {}
    raw = tcfg.get('side_direction_class_weights') or tcfg.get('side_class_weights')
    if raw in (None, '', False):
        return None
    sell_w = _class_cfg_value(raw, 'sell', 1.0)
    buy_w = _class_cfg_value(raw, 'buy', 1.0)
    values = torch.tensor([float(sell_w), float(buy_w)], dtype=torch.float32, device=device)
    if float(values.sum().item()) <= 0.0:
        return None
    return values


def _hierarchical_loss_components(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    gate_bce: nn.Module,
    side_ce: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    y = batch['direction'].long()
    device = y.device
    settings = _hierarchical_loss_settings(cfg)
    trade_target = (y != 1).to(dtype=torch.float32)
    gate_loss = gate_bce(outputs['trade_logit'].view(-1), trade_target)

    trade_mask = y != 1
    if bool(trade_mask.any()):
        # side target: SELL -> 0, BUY -> 1. NO_TRADE rows do not train the
        # direction head because direction is undefined when the gate is closed.
        side_target = (y[trade_mask] == 2).long()
        side_loss = side_ce(outputs['side_direction_logits'][trade_mask], side_target)
    else:
        side_loss = outputs['trade_logit'].sum() * 0.0

    edge_loss = outputs['trade_logit'].sum() * 0.0
    edge_count = 0
    if (
        bool(settings.get('use_edge_pips_loss'))
        and float(settings.get('edge_pips_loss_weight', 0.0) or 0.0) > 0.0
        and 'edge_pips' in outputs
        and 'buy_edge_pips' in batch
        and 'sell_edge_pips' in batch
    ):
        has_edge = batch.get('has_edge_targets')
        if has_edge is None:
            has_edge = torch.ones_like(y, dtype=torch.bool, device=device)
        else:
            has_edge = has_edge.to(device=device).bool()
        if bool(has_edge.any()):
            target = torch.stack([
                batch['buy_edge_pips'].to(device=device, dtype=torch.float32),
                batch['sell_edge_pips'].to(device=device, dtype=torch.float32),
            ], dim=1)
            scale = max(float(settings.get('edge_pips_scale', 10.0) or 10.0), 1e-6)
            edge_loss = F.smooth_l1_loss(outputs['edge_pips'][has_edge] / scale, target[has_edge] / scale)
            edge_count = int(has_edge.sum().detach().cpu().item())

    analytic_signal_loss = outputs['trade_logit'].sum() * 0.0
    analytic_signal_accuracy = 0.0
    analytic_signal_count = 0
    if (
        bool(settings.get('use_analytic_signal_agreement_loss'))
        and float(settings.get('analytic_signal_agreement_loss_weight', 0.0) or 0.0) > 0.0
        and 'analytic_signal_logits' in outputs
        and 'analytic_signal_class' in batch
    ):
        signal_target = batch['analytic_signal_class'].to(device=device).long().clamp(0, 2)
        analytic_signal_loss = F.cross_entropy(outputs['analytic_signal_logits'], signal_target)
        signal_pred = outputs['analytic_signal_logits'].argmax(dim=-1)
        analytic_signal_accuracy = float((signal_pred == signal_target).float().mean().detach().cpu().item())
        analytic_signal_count = int(signal_target.numel())

    total = (
        float(settings.get('gate_loss_weight', 1.0)) * gate_loss
        + float(settings.get('side_direction_loss_weight', 1.0)) * side_loss
        + float(settings.get('edge_pips_loss_weight', 0.0)) * edge_loss
        + float(settings.get('analytic_signal_agreement_loss_weight', 0.0)) * analytic_signal_loss
    )
    return total, {
        'loss': float(total.detach().cpu().item()),
        'gate_loss': float(gate_loss.detach().cpu().item()),
        'side_direction_loss': float(side_loss.detach().cpu().item()),
        'edge_pips_loss': float(edge_loss.detach().cpu().item()),
        'edge_pips_target_count': float(edge_count),
        'analytic_signal_agreement_loss': float(analytic_signal_loss.detach().cpu().item()),
        'analytic_signal_agreement_accuracy': float(analytic_signal_accuracy),
        'analytic_signal_agreement_count': float(analytic_signal_count),
        'trade_loss_weight': float(settings.get('gate_loss_weight', 1.0)),
        'side_direction_loss_weight': float(settings.get('side_direction_loss_weight', 1.0)),
        'edge_pips_loss_weight': float(settings.get('edge_pips_loss_weight', 0.0)),
        'analytic_signal_agreement_loss_weight': float(settings.get('analytic_signal_agreement_loss_weight', 0.0)),
    }


def _edge_metrics_from_outputs(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    if 'edge_pips' not in outputs or 'buy_edge_pips' not in batch or 'sell_edge_pips' not in batch:
        return {}
    y = batch['direction']
    has_edge = batch.get('has_edge_targets')
    if has_edge is None:
        has_edge = torch.ones_like(y, dtype=torch.bool)
    has_edge = has_edge.to(device=y.device).bool()
    if not bool(has_edge.any()):
        return {}
    target = torch.stack([
        batch['buy_edge_pips'].to(device=y.device, dtype=torch.float32),
        batch['sell_edge_pips'].to(device=y.device, dtype=torch.float32),
    ], dim=1)
    pred = outputs['edge_pips']
    err = torch.abs(pred[has_edge] - target[has_edge])
    out = {
        'edge_pips_mae': float(err.mean().detach().cpu().item()),
        'buy_edge_pips_mae': float(err[:, 0].mean().detach().cpu().item()),
        'sell_edge_pips_mae': float(err[:, 1].mean().detach().cpu().item()),
        'edge_pips_target_count': float(has_edge.sum().detach().cpu().item()),
        'pred_buy_edge_pips_mean': float(pred[has_edge, 0].mean().detach().cpu().item()),
        'pred_sell_edge_pips_mean': float(pred[has_edge, 1].mean().detach().cpu().item()),
        'target_buy_edge_pips_mean': float(target[has_edge, 0].mean().detach().cpu().item()),
        'target_sell_edge_pips_mean': float(target[has_edge, 1].mean().detach().cpu().item()),
    }
    return out


def _eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    *,
    branch_auxiliary_bce: nn.Module | None = None,
    branch_auxiliary_weight: float = 0.0,
    branch_auxiliary_branch_weights: torch.Tensor | None = None,
    gate_bce: nn.Module | None = None,
    side_ce: nn.Module | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    probs_out: list[np.ndarray] = []
    losses: list[float] = []
    direction_losses: list[float] = []
    total_losses: list[float] = []
    aux_rows: list[dict[str, float]] = []
    hier_rows: list[dict[str, float]] = []
    edge_rows: list[dict[str, float]] = []
    ce = nn.CrossEntropyLoss()
    aux_enabled = branch_auxiliary_bce is not None and branch_auxiliary_weight > 0.0
    hierarchical = bool(getattr(model, 'is_hierarchical', False))
    setup_training_mode = bool(cfg is not None and _side_setup_training_mode(cfg))
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['direction'].to(device)
            outputs = model(x)
            if setup_training_mode:
                if cfg is None:
                    raise RuntimeError('Side-setup evaluation requires cfg.')
                total_loss, smetrics = _side_setup_loss_components(
                    outputs,
                    {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()},
                    cfg,
                )
                direction_loss = torch.tensor(float(smetrics.get('setup_loss', 0.0)), device=device)
                probs = direction_probabilities_from_outputs(outputs)
                hier_rows.append(smetrics)
                edge_rows.append(_edge_metrics_from_outputs(outputs, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}))
            elif hierarchical:
                if gate_bce is None or side_ce is None or cfg is None:
                    raise RuntimeError('Hierarchical evaluation requires gate_bce, side_ce and cfg.')
                total_loss, hmetrics = _hierarchical_loss_components(outputs, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}, cfg, gate_bce, side_ce)
                direction_loss = torch.tensor(float(hmetrics.get('side_direction_loss', 0.0)), device=device)
                probs = direction_probabilities_from_outputs(outputs)
                hier_rows.append(hmetrics)
                edge_rows.append(_edge_metrics_from_outputs(outputs, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}))
            else:
                logits = outputs['direction_logits']
                direction_loss = ce(logits, y)
                total_loss = direction_loss
                if aux_enabled:
                    aux_loss, aux_metrics = _branch_auxiliary_loss_components(
                        outputs,
                        y,
                        branch_auxiliary_bce,
                        branch_auxiliary_branch_weights,
                    )
                    total_loss = direction_loss + float(branch_auxiliary_weight) * aux_loss
                    aux_rows.append(aux_metrics)
                probs = torch.softmax(logits, dim=-1)
            direction_losses.append(float(direction_loss.item()))
            total_losses.append(float(total_loss.item()))
            losses.append(float(total_loss.item()))
            ys.append(y.cpu().numpy())
            preds.append(probs.argmax(dim=-1).cpu().numpy())
            probs_out.append(probs.cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.asarray([], dtype=int)
    y_pred = np.concatenate(preds) if preds else np.asarray([], dtype=int)
    probs = np.concatenate(probs_out, axis=0) if probs_out else np.empty((0, 3), dtype=float)
    m = _metrics(y_true, y_pred, probs)
    m['loss'] = float(np.mean(losses)) if losses else 0.0
    m['direction_loss'] = float(np.mean(direction_losses)) if direction_losses else 0.0
    if setup_training_mode or hierarchical:
        m.update(_mean_metric_rows(hier_rows))
        m.update(_mean_metric_rows(edge_rows))
        # Extra gate-oriented metrics: these directly measure the decision to trade.
        true_trade = y_true != 1
        pred_trade = y_pred != 1
        tp = int((true_trade & pred_trade).sum())
        fp = int((~true_trade & pred_trade).sum())
        fn = int((true_trade & ~pred_trade).sum())
        gate_precision = tp / (tp + fp) if (tp + fp) else 0.0
        gate_recall = tp / (tp + fn) if (tp + fn) else 0.0
        m['trade_gate_precision'] = float(gate_precision)
        m['trade_gate_recall'] = float(gate_recall)
        m['trade_gate_f1'] = float(2 * gate_precision * gate_recall / (gate_precision + gate_recall)) if (gate_precision + gate_recall) else 0.0
    if aux_enabled:
        m['total_loss_with_branch_auxiliary'] = float(np.mean(total_losses)) if total_losses else 0.0
        m['branch_auxiliary_loss_weight'] = float(branch_auxiliary_weight)
        m.update(_mean_metric_rows(aux_rows))
    return m

def _replay_enabled(cfg: dict[str, Any]) -> bool:
    tcfg = cfg.get('training', {}) or {}
    return bool(tcfg.get('replay_each_epoch', False) or tcfg.get('replay_during_training', False))


def _replay_eval_window(cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    tcfg = cfg.get('training', {}) or {}
    rcfg = cfg.get('replay', {}) or {}
    start = tcfg.get('replay_start', rcfg.get('eval_start'))
    end = tcfg.get('replay_end', rcfg.get('eval_end'))
    return (str(start) if start not in (None, '') else None), (str(end) if end not in (None, '') else None)


def _replay_score_from_summary(summary: dict[str, Any], cfg: dict[str, Any]) -> float:
    rcfg = cfg.get('replay', {}) or {}
    min_trades = int(rcfg.get('min_trades_for_score', 50) or 0)
    trades = int(summary.get('trades', 0) or 0)
    if trades < min_trades:
        return -1_000_000_000.0 + trades
    return float(summary.get('replay_score', summary.get('net_pips', 0.0)) or 0.0)


def _select_epoch_score(val_metrics: dict[str, Any], replay_summary: dict[str, Any] | None, cfg: dict[str, Any]) -> float:
    metric = str((cfg.get('training', {}) or {}).get('model_selection_metric', 'macro_f1')).lower()
    if metric in {'replay_score', 'replay', 'replay_net_pips'} and replay_summary is not None:
        if metric == 'replay_net_pips':
            return float(replay_summary.get('net_pips', 0.0) or 0.0)
        return _replay_score_from_summary(replay_summary, cfg)
    return float(val_metrics.get(metric, val_metrics.get('macro_f1', 0.0)) or 0.0)


def train_symbol(symbol: str, cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cli_train_side = getattr(args, 'train_side', None)
    if cli_train_side is not None:
        cfg = _apply_training_side_to_config(cfg, cli_train_side)
    else:
        cfg = _apply_training_side_to_config(cfg, _training_side(cfg))
    train_side = _training_side(cfg)
    device = _normalise_torch_device(args.device)
    tcfg = cfg.get('training', {}) or {}
    seed_info = _seed_settings(symbol, cfg, args)
    if seed_info.get('enabled'):
        _set_global_seed(int(seed_info['initial_seed']), deterministic=bool(seed_info.get('deterministic')))
    date_start = args.date_start if args.date_start is not None else tcfg.get('date_start')
    date_end = args.date_end if args.date_end is not None else tcfg.get('date_end')
    max_rows = args.max_rows if args.max_rows is not None else tcfg.get('max_rows')
    df, load_info = _load_direction_dataframe(symbol, cfg, date_start=date_start, date_end=date_end, max_rows=max_rows)
    seq_len = int((cfg.get('model') or {}).get('sequence_length', 64))
    total_sequence_rows = int(len(df) - seq_len + 1)
    if total_sequence_rows < 100:
        raise RuntimeError(f'{symbol}: not enough raw sequence rows for direction training: {total_sequence_rows}')

    val_fraction = float(args.val_fraction if args.val_fraction is not None else tcfg.get('val_fraction', 0.2))
    if _universal_split_enabled(cfg, df):
        df, universal_rebuild_report = _rebuild_universal_split_if_needed(df, cfg, val_fraction=val_fraction)
    else:
        universal_rebuild_report = {'universal_split_rebuilt': False}
    val_start = int(total_sequence_rows * (1.0 - val_fraction))
    configured_embargo = int(tcfg.get('train_validation_embargo_bars', 0) or 0)
    horizon_bars = int((cfg.get('labels') or {}).get('horizon_bars', 0) or 0)
    auto_min_embargo = bool(tcfg.get('auto_min_train_validation_embargo', True))
    min_embargo = int(tcfg.get('min_train_validation_embargo_bars', 0) or 0)
    if auto_min_embargo:
        # The embargo is in raw sequence-index space. It covers both input
        # window overlap and forward-label horizon overlap at the split.
        min_embargo = max(min_embargo, seq_len + horizon_bars)
    embargo = max(configured_embargo, min_embargo)
    train_end = max(0, val_start - embargo)

    # Fit scaler on training rows only, then transform the full frame.
    # Symbol-specific runs use the existing chronological split. Universal pooled
    # datasets can opt into an explicit per-symbol split column generated by
    # combine_universal_direction_datasets.py.
    split_report: dict[str, Any]
    if _universal_split_enabled(cfg, df):
        train_scaler_df, split_report = _universal_train_scaler_df(df, cfg)
        train_raw_end_exclusive = int(len(train_scaler_df))
        if len(train_scaler_df) < seq_len:
            raise RuntimeError(
                f'{symbol}: universal train split has only {len(train_scaler_df)} rows, '
                f'need at least sequence_length={seq_len}'
            )
    else:
        train_raw_end_exclusive = int(train_end + seq_len - 1)
        train_scaler_df = df.iloc[:train_raw_end_exclusive].copy()
        split_report = {
            'split_mode': 'chronological_holdout',
            'val_start_sequence_index': int(val_start),
            'configured_train_validation_embargo_bars': int(configured_embargo),
            'effective_train_validation_embargo_bars': int(embargo),
            'auto_min_train_validation_embargo': bool(auto_min_embargo),
        }
    split_report['universal_split_rebuild_report'] = universal_rebuild_report

    train_arr_for_scaler = prepare_direction_arrays(train_scaler_df, cfg, fit_scaler=True)
    arr = prepare_direction_arrays(
        df,
        cfg,
        scaler=train_arr_for_scaler.scaler,
        feature_columns=train_arr_for_scaler.feature_columns,
        fit_scaler=False,
    )
    n = int(len(arr.X_seq))
    ignored_sequence_rows = int(total_sequence_rows - n)
    if n < 100:
        raise RuntimeError(
            f'{symbol}: not enough labelled sequence rows for direction training after IGNORE/group filtering: '
            f'{n} valid of {total_sequence_rows} raw sequences'
        )

    if _universal_split_enabled(cfg, df):
        train_idx, val_idx, sequence_split_report = _universal_sequence_split_indices(df, arr, cfg)
        split_report.update(sequence_split_report)
    else:
        sequence_indices = arr.row_indices.astype(int) - (seq_len - 1)
        train_idx = np.where(sequence_indices < train_end)[0].astype(int)
        val_idx = np.where(sequence_indices >= val_start)[0].astype(int)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(
            f'{symbol}: train/validation split produced empty labelled set. '
            f'total_sequence_rows={total_sequence_rows}, valid_sequence_rows={n}, '
            f'split_report={split_report}'
        )

    dataset = DirectionDataset(arr)
    batch_size = int(args.batch_size or tcfg.get('batch_size', 512))
    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    curriculum_settings = _curriculum_sampler_settings(cfg)
    curriculum_report = _curriculum_static_report(arr.y_direction[train_idx], curriculum_settings)
    setup_training_mode = _side_setup_training_mode(cfg)
    if setup_training_mode:
        # Store fixed, training-set-only positive weights in the copied config so
        # per-batch losses do not leak validation distribution information.
        cfg.setdefault('training', {})['_buy_setup_pos_weight'] = _auto_setup_pos_weight(
            None if arr.buy_setup_target is None else np.asarray(arr.buy_setup_target)[train_idx],
            None if arr.has_buy_setup_target is None else np.asarray(arr.has_buy_setup_target)[train_idx],
            cfg,
            'buy',
        )
        cfg.setdefault('training', {})['_sell_setup_pos_weight'] = _auto_setup_pos_weight(
            None if arr.sell_setup_target is None else np.asarray(arr.sell_setup_target)[train_idx],
            None if arr.has_sell_setup_target is None else np.asarray(arr.has_sell_setup_target)[train_idx],
            cfg,
            'sell',
        )
        cfg.setdefault('model', {})['use_side_setup_heads'] = True
        cfg.setdefault('model', {})['decision_output_mode'] = cfg.get('model', {}).get('decision_output_mode', 'side_setup')
        cfg.setdefault('model', {})['use_setup_quality_head'] = cfg.get('model', {}).get('use_setup_quality_head', True)

    def make_train_loader(epoch_seed: int | None = None, epoch: int | None = None) -> tuple[DataLoader, dict[str, Any] | None]:
        if curriculum_settings.get('enabled'):
            epoch_indices, plan = _make_curriculum_epoch_indices(
                train_idx,
                arr.y_direction,
                curriculum_settings,
                int(epoch or 1),
                epoch_seed,
            )
            epoch_subset = Subset(dataset, epoch_indices.tolist())
            return DataLoader(
                epoch_subset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            ), plan
        return DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=_make_generator(epoch_seed),
        ), None

    train_loader, _ = make_train_loader(seed_info.get('initial_seed') if seed_info.get('enabled') else None, epoch=1)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Provide the exact prepared feature order to architectures that use named feature subsets
    # internally, such as the mixture-of-experts router.
    cfg['_feature_columns'] = list(arr.feature_columns)
    model = DirectionTradePolicyNet(arr.X_seq.shape[-1], cfg).to(device)
    architecture_name = str(getattr(model, 'architecture', (cfg.get('model') or {}).get('architecture', 'hierarchical_tcn_edge_v1')))
    hierarchical_model = bool(getattr(model, 'is_hierarchical', False))
    weights = _class_weights(arr.y_direction[train_idx], cfg) if (not hierarchical_model and not setup_training_mode) else None
    ce = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    gate_pos_weight = _gate_pos_weight_tensor(arr.y_direction[train_idx], cfg, device) if hierarchical_model else None
    gate_bce = nn.BCEWithLogitsLoss(pos_weight=gate_pos_weight) if hierarchical_model else None
    side_weights = _side_class_weight_tensor(arr.y_direction[train_idx], cfg, device) if hierarchical_model else None
    side_ce = nn.CrossEntropyLoss(weight=side_weights) if hierarchical_model else None
    hierarchical_loss_config = _hierarchical_loss_settings(cfg) if hierarchical_model else None

    branch_auxiliary_enabled = (not hierarchical_model) and (not setup_training_mode) and _branch_auxiliary_enabled(cfg)
    branch_auxiliary_weight = _branch_auxiliary_loss_weight(cfg) if branch_auxiliary_enabled else 0.0
    branch_auxiliary_pos_weights = _branch_auxiliary_pos_weights(arr.y_direction[train_idx], cfg) if branch_auxiliary_enabled else None
    branch_auxiliary_branch_weights = _branch_auxiliary_branch_weights(cfg) if branch_auxiliary_enabled else None
    branch_auxiliary_bce = (
        nn.BCEWithLogitsLoss(
            pos_weight=branch_auxiliary_pos_weights.to(device) if branch_auxiliary_pos_weights is not None else None,
            reduction='none',
        )
        if branch_auxiliary_enabled and branch_auxiliary_weight > 0.0
        else None
    )
    branch_auxiliary_config = {
        'enabled': bool(branch_auxiliary_bce is not None),
        'loss_weight': float(branch_auxiliary_weight),
        'branch_loss_weights': None if branch_auxiliary_branch_weights is None else [float(x) for x in branch_auxiliary_branch_weights.cpu().numpy()],
        'pos_weights': None if branch_auxiliary_pos_weights is None else [float(x) for x in branch_auxiliary_pos_weights.cpu().numpy()],
        'target_order': ['SELL', 'NO_TRADE', 'BUY'],
    }

    lr = float(args.learning_rate or tcfg.get('learning_rate', tcfg.get('lr', 5e-4)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tcfg.get('weight_decay', 1e-3)))
    grad_clip = float(tcfg.get('gradient_clip_norm', 1.0) or 0.0)
    epochs = int(args.epochs or tcfg.get('epochs', 50))
    patience = int(tcfg.get('early_stopping_patience', epochs))
    min_delta = float(tcfg.get('min_delta', 1e-4))

    model_type_name = str((getattr(model, 'model_details', {}) or {}).get('model_type', architecture_name))
    replay_start_for_report, replay_end_for_report = _replay_eval_window(cfg)
    deployment_decision_parameters = resolve_replay_decision_parameters(
        cfg,
        train_side=train_side,
        eval_start=replay_start_for_report,
        eval_end=replay_end_for_report,
        symbol=symbol,
    )
    training_decision_parameters = resolve_training_decision_parameters(cfg, train_side=train_side, symbol=symbol)
    resolved_config_snapshot = config_snapshot(
        cfg,
        config_path=getattr(args, 'config', None) or cfg.get('_config_path'),
        base_config_path=cfg.get('_base_config_path'),
        include_resolved_sections=True,
    )

    model_path, scaler_path, features_path, report_path = _model_paths(symbol, cfg)
    ensure_dir(model_path.parent)
    ensure_dir(report_path.parent)
    # Save the run-level scaler/features before the epoch loop so optional
    # per-epoch replay can load the final/best checkpoint exactly as live will
    # use it. Per-epoch copies are also saved below beside every checkpoint.
    run_feature_metadata = _feature_metadata(
        arr=arr,
        architecture_name=architecture_name,
        model=model,
        train_side=train_side,
        deployment_decision_parameters=deployment_decision_parameters,
        resolved_config_snapshot=resolved_config_snapshot,
        symbol=symbol,
        cfg=cfg,
    )
    _save_scaler_and_features(
        scaler=arr.scaler,
        scaler_path=scaler_path,
        features_path=features_path,
        feature_metadata=run_feature_metadata,
    )
    epoch_dir = model_path.parent / str(tcfg.get('epoch_model_dir', 'epoch_checkpoints')) / symbol
    save_epoch_models = bool(tcfg.get('save_epoch_models', True))
    if save_epoch_models:
        ensure_dir(epoch_dir)

    history: list[dict[str, Any]] = []
    best_score = float('-inf')
    best_epoch = 0
    best_payload: dict[str, Any] | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        active_epoch_seed = _epoch_seed(seed_info, epoch) if seed_info.get('reseed_each_epoch') else None
        loader_epoch_seed = active_epoch_seed
        if loader_epoch_seed is None and curriculum_settings.get('enabled') and seed_info.get('enabled'):
            # Even when full model reseeding is disabled, keep curriculum sampling
            # deterministic and epoch-varying when a base seed is supplied.
            loader_epoch_seed = _epoch_seed(seed_info, epoch)
        curriculum_plan = None
        if active_epoch_seed is not None:
            _set_global_seed(active_epoch_seed, deterministic=bool(seed_info.get('deterministic')))
        if curriculum_settings.get('enabled') or active_epoch_seed is not None:
            train_loader, curriculum_plan = make_train_loader(loader_epoch_seed, epoch=epoch)
        model.train()
        losses: list[float] = []
        direction_losses: list[float] = []
        aux_metric_rows: list[dict[str, float]] = []
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['direction'].to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(x)
            if setup_training_mode:
                loss, smetrics = _side_setup_loss_components(
                    outputs,
                    {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()},
                    cfg,
                )
                direction_loss = torch.tensor(float(smetrics.get('setup_loss', 0.0)), device=device)
                aux_metric_rows.append(smetrics)
            elif hierarchical_model:
                if gate_bce is None or side_ce is None:
                    raise RuntimeError('Hierarchical model missing gate/side losses')
                loss, hmetrics = _hierarchical_loss_components(
                    outputs,
                    {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()},
                    cfg,
                    gate_bce,
                    side_ce,
                )
                direction_loss = torch.tensor(float(hmetrics.get('side_direction_loss', 0.0)), device=device)
                aux_metric_rows.append(hmetrics)
            else:
                logits = outputs['direction_logits']
                direction_loss = ce(logits, y)
                loss = direction_loss
                if branch_auxiliary_bce is not None:
                    aux_loss, aux_metrics = _branch_auxiliary_loss_components(
                        outputs,
                        y,
                        branch_auxiliary_bce,
                        branch_auxiliary_branch_weights,
                    )
                    loss = direction_loss + branch_auxiliary_weight * aux_loss
                    aux_metric_rows.append(aux_metrics)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.item()))
            direction_losses.append(float(direction_loss.item()))
        val_metrics = _eval_model(
            model,
            val_loader,
            device,
            branch_auxiliary_bce=branch_auxiliary_bce,
            branch_auxiliary_weight=branch_auxiliary_weight,
            branch_auxiliary_branch_weights=branch_auxiliary_branch_weights,
            gate_bce=gate_bce,
            side_ce=side_ce,
            cfg=cfg,
        )
        train_metrics = {
            'loss': float(np.mean(losses)) if losses else 0.0,
            'direction_loss': float(np.mean(direction_losses)) if direction_losses else 0.0,
        }
        if setup_training_mode or hierarchical_model:
            train_metrics.update(_mean_metric_rows(aux_metric_rows))
        elif branch_auxiliary_bce is not None:
            train_metrics['total_loss_with_branch_auxiliary'] = train_metrics['loss']
            train_metrics['branch_auxiliary_loss_weight'] = float(branch_auxiliary_weight)
            train_metrics.update(_mean_metric_rows(aux_metric_rows))
        if curriculum_plan is not None:
            train_metrics['curriculum_total_samples'] = int(curriculum_plan.get('total_samples', 0) or 0)
            train_metrics['curriculum_trade_samples'] = int(curriculum_plan.get('trade_samples', 0) or 0)
            train_metrics['curriculum_no_trade_samples'] = int(curriculum_plan.get('no_trade_samples', 0) or 0)
            train_metrics['curriculum_no_trade_to_trade_sample_ratio'] = curriculum_plan.get('no_trade_to_trade_sample_ratio')
        row = {
            'epoch': epoch,
            'train': train_metrics,
            'validation': val_metrics,
            'train_side': train_side,
            'training_decision_parameters': training_decision_parameters,
            'deployment_decision_parameters': deployment_decision_parameters,
        }
        if curriculum_plan is not None:
            row['curriculum'] = curriculum_plan
        if seed_info.get('enabled'):
            row['seed'] = int(active_epoch_seed if active_epoch_seed is not None else seed_info['initial_seed'])

        payload = {
            'model_state_dict': model.state_dict(),
            'architecture': architecture_name,
            'symbol': symbol,
            'timeframe': _timeframe(cfg),
            'epoch': epoch,
            'feature_columns': arr.feature_columns,
            'model_config': cfg.get('model', {}),
            'training_config': cfg.get('training', {}),
            'curriculum_sampler_config': curriculum_report,
            'branch_auxiliary_config': branch_auxiliary_config,
            'hierarchical_loss_config': hierarchical_loss_config,
            'setup_loss_config': _setup_loss_settings(cfg) if setup_training_mode else None,
            'model_details': getattr(model, 'model_details', {}),
            'seed_config': seed_info,
            'validation_metrics': val_metrics,
            'train_side': train_side,
            'deployment_decision_parameters': deployment_decision_parameters,
            'training_decision_parameters': training_decision_parameters,
            'config_snapshot': {k: v for k, v in resolved_config_snapshot.items() if k != 'resolved_sections'},
        }
        epoch_checkpoint_path, epoch_scaler_path, epoch_features_path = _epoch_artifact_paths(epoch_dir, symbol, cfg, epoch)
        epoch_feature_metadata = _feature_metadata(
            arr=arr,
            architecture_name=architecture_name,
            model=model,
            train_side=train_side,
            deployment_decision_parameters=deployment_decision_parameters,
            resolved_config_snapshot=resolved_config_snapshot,
            symbol=symbol,
            cfg=cfg,
            epoch=epoch,
            checkpoint_path=epoch_checkpoint_path,
            scaler_path=epoch_scaler_path,
        )
        row['checkpoint_artifacts'] = {
            'model_path': str(epoch_checkpoint_path),
            'scaler_path': str(epoch_scaler_path),
            'features_path': str(epoch_features_path),
        }
        payload['checkpoint_artifacts'] = row['checkpoint_artifacts']
        if save_epoch_models:
            torch.save(payload, epoch_checkpoint_path)
            _save_scaler_and_features(
                scaler=arr.scaler,
                scaler_path=epoch_scaler_path,
                features_path=epoch_features_path,
                feature_metadata=epoch_feature_metadata,
            )

        replay_summary = None
        if _replay_enabled(cfg) and save_epoch_models:
            try:
                from .test_saved_direction_policy import replay_symbol
                replay_start, replay_end = _replay_eval_window(cfg)
                replay_root = Path((tcfg.get('replay_output_dir') or (cfg.get('replay', {}) or {}).get('output_dir') or 'logs/training_replay_direction'))
                replay_prefix = replay_root / symbol / f'{symbol}_{_timeframe(cfg)}_epoch_{epoch:03d}_direction_replay'
                replay_summary = replay_symbol(
                    symbol,
                    cfg,
                    model_path=epoch_checkpoint_path,
                    scaler_path=epoch_scaler_path,
                    features_path=epoch_features_path,
                    eval_start=replay_start,
                    eval_end=replay_end,
                    output_prefix=str(replay_prefix),
                    device=tcfg.get('replay_device') or args.device,
                    verbose=False,
                )
                row['replay'] = {k: replay_summary.get(k) for k in (
                    'trades', 'net_pips', 'win_rate', 'average_net_pips',
                    'max_drawdown_pips',
                    'buy_trades', 'buy_net_pips', 'buy_win_rate',
                    'buy_average_net_pips', 'buy_loss_pips', 'buy_losing_trades',
                    'sell_trades', 'sell_net_pips', 'sell_win_rate',
                    'sell_average_net_pips', 'sell_loss_pips', 'sell_losing_trades',
                    'replay_score', 'summary_path', 'decisions_path', 'trades_path',
                    'threshold_mode', 'rolling_thresholds_used', 'allow_buy', 'allow_sell',
                    'min_direction_probability', 'min_trade_probability', 'min_edge_pips',
                    'decision_parameters', 'deployment_decision_parameters', 'config_snapshot',
                )}
                # Make the resolved settings explicit even when an older replay helper
                # returned only a partial summary. This is what the live registry
                # should copy for each symbol/model/side checkpoint.
                row['replay']['decision_parameters'] = replay_summary.get('decision_parameters') or deployment_decision_parameters
                row['replay']['deployment_decision_parameters'] = replay_summary.get('deployment_decision_parameters') or deployment_decision_parameters
                payload['replay_summary'] = row['replay']
                payload['model_selection_replay_score'] = replay_summary.get('replay_score')
                if save_epoch_models:
                    torch.save(payload, epoch_checkpoint_path)
            except Exception as exc:
                row['replay_error'] = str(exc)
                print(f'{symbol} epoch {epoch:03d}: replay failed: {exc}', flush=True)

        score = _select_epoch_score(val_metrics, replay_summary, cfg)
        row['model_selection_score'] = score
        history.append(row)
        replay_text = ''
        if replay_summary is not None:
            replay_text = (
                f" replay_net={float(replay_summary.get('net_pips', 0.0) or 0.0):.1f}"
                f" replay_buy={float(replay_summary.get('buy_net_pips', 0.0) or 0.0):.1f}"
                f" replay_sell={float(replay_summary.get('sell_net_pips', 0.0) or 0.0):.1f}"
                f" replay_trades={int(replay_summary.get('trades', 0) or 0)}"
                f" replay_buy_trades={int(replay_summary.get('buy_trades', 0) or 0)}"
                f" replay_sell_trades={int(replay_summary.get('sell_trades', 0) or 0)}"
                f" replay_buy_wr={float(replay_summary.get('buy_win_rate', 0.0) or 0.0):.3f}"
                f" replay_sell_wr={float(replay_summary.get('sell_win_rate', 0.0) or 0.0):.3f}"
                f" replay_buy_loss={float(replay_summary.get('buy_loss_pips', 0.0) or 0.0):.1f}"
                f" replay_sell_loss={float(replay_summary.get('sell_loss_pips', 0.0) or 0.0):.1f}"
            )
        aux_text = ''
        if setup_training_mode:
            aux_text = (
                f' train_buy_setup={train_metrics.get("buy_setup_loss", 0.0):.5f}'
                f' train_sell_setup={train_metrics.get("sell_setup_loss", 0.0):.5f}'
                f' train_quality={train_metrics.get("setup_quality_loss", 0.0):.5f}'
                f' val_buy_setup={val_metrics.get("buy_setup_loss", 0.0):.5f}'
                f' val_sell_setup={val_metrics.get("sell_setup_loss", 0.0):.5f}'
                f' val_quality={val_metrics.get("setup_quality_loss", 0.0):.5f}'
                f' val_buy_f1={val_metrics.get("buy_setup_f1_05", 0.0):.4f}'
                f' val_sell_f1={val_metrics.get("sell_setup_f1_05", 0.0):.4f}'
            )
        elif hierarchical_model:
            aux_text = (
                f' train_gate={train_metrics.get("gate_loss", 0.0):.5f}'
                f' train_side={train_metrics.get("side_direction_loss", 0.0):.5f}'
                f' train_edge={train_metrics.get("edge_pips_loss", 0.0):.5f}'
                f' val_gate={val_metrics.get("gate_loss", 0.0):.5f}'
                f' val_side={val_metrics.get("side_direction_loss", 0.0):.5f}'
                f' val_edge={val_metrics.get("edge_pips_loss", 0.0):.5f}'
                f' val_gate_f1={val_metrics.get("trade_gate_f1", 0.0):.4f}'
                f' val_edge_mae={val_metrics.get("edge_pips_mae", 0.0):.3f}'
                f' sig_aux={val_metrics.get("analytic_signal_agreement_loss", 0.0):.5f}'
                f' sig_acc={val_metrics.get("analytic_signal_agreement_accuracy", 0.0):.3f}'
            )
        elif branch_auxiliary_bce is not None:
            aux_text = (
                f' train_dir_loss={train_metrics.get("direction_loss", 0.0):.5f}'
                f' train_aux={train_metrics.get("branch_auxiliary_loss", 0.0):.5f}'
                f' val_dir_loss={val_metrics.get("direction_loss", 0.0):.5f}'
                f' val_aux={val_metrics.get("branch_auxiliary_loss", 0.0):.5f}'
                f' val_aux_sell={val_metrics.get("branch_auxiliary_sell_loss", 0.0):.5f}'
                f' val_aux_no_trade={val_metrics.get("branch_auxiliary_no_trade_loss", 0.0):.5f}'
                f' val_aux_buy={val_metrics.get("branch_auxiliary_buy_loss", 0.0):.5f}'
            )
        decision_score_text = _format_actual_label_decision_scores_for_terminal(val_metrics)
        curriculum_text = _format_curriculum_plan(curriculum_plan)
        print(
            f'{symbol} epoch {epoch:03d}: train_loss={train_metrics["loss"]:.5f} '
            f'val_loss={val_metrics["loss"]:.5f} val_macro_f1={val_metrics.get("macro_f1", 0.0):.4f} '
            f'pred_trades={val_metrics.get("predicted_trades", 0)} score={score:.4f}'
            f'{curriculum_text}{aux_text}{decision_score_text}{replay_text}',
            flush=True,
        )

        if best_payload is None or score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            # state_dict tensors are references to the live model parameters, so
            # keep a deep copy. Otherwise the final saved "best" model can
            # silently become the last epoch's weights.
            best_payload = copy.deepcopy(payload)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            print(f'{symbol}: early stopping at epoch {epoch}; best_epoch={best_epoch}', flush=True)
            break

    if best_payload is None:
        # Defensive fallback: this should now be rare because epoch 1 is always
        # accepted, but keep the trainer from killing a multi-model Kaggle run.
        if history:
            print(f'{symbol}: WARNING no best checkpoint was selected; using last epoch payload as fallback.', flush=True)
            best_epoch = int(history[-1].get('epoch', len(history)) or len(history))
            best_score = float(history[-1].get('model_selection_score', -1_000_000_000.0) or -1_000_000_000.0)
            best_payload = payload
        else:
            raise RuntimeError(f'{symbol}: training produced no epochs and no checkpoint')
    torch.save(best_payload, model_path)
    final_val = history[best_epoch - 1]['validation'] if 0 < best_epoch <= len(history) else history[-1]['validation']
    report = {
        'symbol': symbol,
        'timeframe': _timeframe(cfg),
        'model_type': (f'{model_type_name}_side_setup_ranking' if setup_training_mode else (model_type_name if hierarchical_model else 'direction_buy_sell_no_trade')),
        'class_mapping': DIRECTION_CLASS_NAMES,
        'data': load_info,
        'sequence_rows': int(n),
        'total_raw_sequence_rows': int(total_sequence_rows),
        'ignored_sequence_rows': int(ignored_sequence_rows),
        'feature_count': int(arr.X_seq.shape[-1]),
        'sequence_length': int(arr.X_seq.shape[1]),
        'train_rows': int(len(train_idx)),
        'validation_rows': int(len(val_idx)),
        'seed': seed_info,
        'train_side': train_side,
        'training_decision_parameters': training_decision_parameters,
        'deployment_decision_parameters': deployment_decision_parameters,
        'config_snapshot': resolved_config_snapshot,
        'split': {
            **split_report,
            'val_fraction': float(val_fraction),
            'sequence_length': int(seq_len),
            'horizon_bars': int(horizon_bars),
            'train_raw_end_exclusive': int(train_raw_end_exclusive),
            'scaler_fit_rows': int(len(train_scaler_df)),
            'total_raw_sequence_rows': int(total_sequence_rows),
            'valid_labelled_sequence_rows': int(n),
            'ignored_sequence_rows': int(ignored_sequence_rows),
        },
        'train_class_counts': _class_counts(arr.y_direction[train_idx]),
        'validation_class_counts': _class_counts(arr.y_direction[val_idx]),
        'class_weights': None if weights is None else [float(x) for x in weights.cpu().numpy()],
        'hierarchical_loss': hierarchical_loss_config,
        'setup_loss': _setup_loss_settings(cfg) if setup_training_mode else None,
        'setup_target_counts_train': {
            'buy_positive': int(((np.asarray(arr.buy_setup_target) == 1) & np.asarray(arr.has_buy_setup_target) & np.isin(np.arange(len(arr.y_direction)), train_idx)).sum()) if arr.buy_setup_target is not None else 0,
            'buy_negative': int(((np.asarray(arr.buy_setup_target) == 0) & np.asarray(arr.has_buy_setup_target) & np.isin(np.arange(len(arr.y_direction)), train_idx)).sum()) if arr.buy_setup_target is not None else 0,
            'sell_positive': int(((np.asarray(arr.sell_setup_target) == 1) & np.asarray(arr.has_sell_setup_target) & np.isin(np.arange(len(arr.y_direction)), train_idx)).sum()) if arr.sell_setup_target is not None else 0,
            'sell_negative': int(((np.asarray(arr.sell_setup_target) == 0) & np.asarray(arr.has_sell_setup_target) & np.isin(np.arange(len(arr.y_direction)), train_idx)).sum()) if arr.sell_setup_target is not None else 0,
        } if setup_training_mode else None,
        'setup_target_counts_validation': {
            'buy_positive': int(((np.asarray(arr.buy_setup_target) == 1) & np.asarray(arr.has_buy_setup_target) & np.isin(np.arange(len(arr.y_direction)), val_idx)).sum()) if arr.buy_setup_target is not None else 0,
            'buy_negative': int(((np.asarray(arr.buy_setup_target) == 0) & np.asarray(arr.has_buy_setup_target) & np.isin(np.arange(len(arr.y_direction)), val_idx)).sum()) if arr.buy_setup_target is not None else 0,
            'sell_positive': int(((np.asarray(arr.sell_setup_target) == 1) & np.asarray(arr.has_sell_setup_target) & np.isin(np.arange(len(arr.y_direction)), val_idx)).sum()) if arr.sell_setup_target is not None else 0,
            'sell_negative': int(((np.asarray(arr.sell_setup_target) == 0) & np.asarray(arr.has_sell_setup_target) & np.isin(np.arange(len(arr.y_direction)), val_idx)).sum()) if arr.sell_setup_target is not None else 0,
        } if setup_training_mode else None,
        'gate_pos_weight': None if gate_pos_weight is None else float(gate_pos_weight.detach().cpu().view(-1)[0].item()),
        'side_direction_class_weights': None if side_weights is None else [float(x) for x in side_weights.detach().cpu().numpy()],
        'edge_targets_available': bool(arr.has_edge_targets is not None and np.asarray(arr.has_edge_targets).any()),
        'edge_target_count_train': int(np.asarray(arr.has_edge_targets)[train_idx].sum()) if arr.has_edge_targets is not None else 0,
        'edge_target_count_validation': int(np.asarray(arr.has_edge_targets)[val_idx].sum()) if arr.has_edge_targets is not None else 0,
        'analytic_signal_class_available': bool(arr.analytic_signal_class is not None),
        'analytic_signal_class_counts_train': _class_counts(np.asarray(arr.analytic_signal_class)[train_idx]) if arr.analytic_signal_class is not None else None,
        'analytic_signal_class_counts_validation': _class_counts(np.asarray(arr.analytic_signal_class)[val_idx]) if arr.analytic_signal_class is not None else None,
        'curriculum_sampler': curriculum_report,
        'branch_auxiliary': branch_auxiliary_config,
        'best_epoch': int(best_epoch),
        'best_model_selection_score': float(best_score),
        'best_checkpoint_is_fallback': bool(best_score <= -999_999_999.0),
        'best_checkpoint_warning': (
            'No replay-qualified epoch was found; saved the least-bad/first available checkpoint so the multi-model pipeline can continue.'
            if best_score <= -999_999_999.0 else None
        ),
        'best_validation_metrics': final_val,
        'best_replay': (history[best_epoch - 1].get('replay') if 0 < best_epoch <= len(history) else None),
        'artifacts': {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'features_path': str(features_path),
            'report_path': str(report_path),
            'epoch_dir': str(epoch_dir) if save_epoch_models else None,
            'epoch_artifact_naming': (
                f'{symbol}_{_timeframe(cfg)}_direction_policy_epoch_###.pt / '
                f'{symbol}_{_timeframe(cfg)}_direction_policy_epoch_###_scaler.pkl / '
                f'{symbol}_{_timeframe(cfg)}_direction_policy_epoch_###_features.json'
            ) if save_epoch_models else None,
        },
        'history': history,
    }
    write_json(report_path, _json_safe(report))
    print(f'{symbol}: saved direction model to {model_path}')
    return report


def main() -> None:
    p = argparse.ArgumentParser(description='Train BUY/SELL/NO_TRADE direction policy models. Model type is selected with model.architecture in the config.')
    p.add_argument('--config', default='config/direction_settings_generic_multisymbol_31_symbols.yaml')
    p.add_argument('--symbols', nargs='+', default=None)
    p.add_argument('--date-start', default=None)
    p.add_argument('--date-end', default=None)
    p.add_argument('--max-rows', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--val-fraction', type=float, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=None, help='Base random seed. Overrides training.seed.')
    p.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable best-effort deterministic Torch behaviour.')
    p.add_argument('--reseed-each-epoch', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable deterministic per-epoch reseeding.')
    p.add_argument('--epoch-seed-mode', default=None, help='base_only, base_plus_epoch, base_plus_symbol, or base_plus_symbol_plus_epoch.')
    p.add_argument('--train-side', choices=['both', 'buy', 'sell'], default=None, help='Train both side-setup heads, or train a BUY-only/SELL-only setup model. Replay is side-filtered automatically for buy/sell.')
    args = p.parse_args()

    cfg = load_config_with_optional_spread_risk(args.config)
    cfg['_config_path'] = str(args.config)
    cfg.setdefault('_base_config_path', str(args.config))
    symbols = validate_forex_symbols(args.symbols or ((cfg.get('trading') or {}).get('symbols') or ['US500']))
    reports = []
    for symbol in symbols:
        symbol_cfg = dict(cfg)
        symbol_cfg['_active_symbol'] = symbol
        reports.append(train_symbol(symbol, symbol_cfg, args))
    summary_path = Path((cfg.get('paths') or {}).get('log_dir', 'logs')) / f'direction_training_summary_{_timeframe(cfg)}.json'
    ensure_dir(summary_path.parent)
    write_json(summary_path, _json_safe({'symbols': symbols, 'reports': reports}))
    print(f'Wrote summary: {summary_path}')


if __name__ == '__main__':
    main()
