from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    """Return JSON-serialisable built-in values for config/report snapshots."""
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy normally exists in this project
        np = None  # type: ignore

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (np.ndarray,)):
            return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _cfg_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value is False:
        return default
    if isinstance(value, str) and value.strip().lower() in {'', 'none', 'off'}:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _cfg_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value is False:
        return default
    if isinstance(value, str) and value.strip().lower() in {'', 'none', 'off'}:
        return default
    try:
        return int(value)
    except Exception:
        return default


def timeframe_from_config(cfg: dict[str, Any]) -> str:
    return str((cfg.get('trading') or {}).get('timeframe') or (cfg.get('project') or {}).get('timeframe') or 'M5').upper()


def training_side_from_config(cfg: dict[str, Any]) -> str:
    tcfg = cfg.get('training', {}) or {}
    side = str(cfg.get('_training_side', tcfg.get('side_setup_train_side', tcfg.get('train_side', 'both'))) or 'both').strip().lower()
    aliases = {
        '': 'both',
        'all': 'both',
        'both_sides': 'both',
        'combined': 'both',
        'long': 'buy',
        'short': 'sell',
    }
    return aliases.get(side, side)


def _side_cfg(cfg: dict[str, Any], side: str) -> dict[str, Any]:
    rcfg = cfg.get('replay', {}) or {}
    raw = rcfg.get(side.lower()) or rcfg.get(side.upper()) or {}
    return raw if isinstance(raw, dict) else {}


def _allow_side(cfg: dict[str, Any], side: str) -> bool:
    rcfg = cfg.get('replay', {}) or {}
    return _cfg_bool(rcfg.get(f'allow_{side.lower()}', True), True)


def _threshold_mode(cfg: dict[str, Any]) -> str:
    rcfg = cfg.get('replay', {}) or {}
    return str(rcfg.get('threshold_mode', rcfg.get('score_threshold_mode', 'fixed_probability')) or 'fixed_probability').strip().lower()


def _min_direction_probability(cfg: dict[str, Any]) -> float:
    bcfg = cfg.get('backtest', {}) or {}
    rcfg = cfg.get('replay', {}) or {}
    dcfg = cfg.get('direction_policy', {}) or {}
    return float(rcfg.get('min_direction_probability', bcfg.get('min_direction_probability', dcfg.get('min_direction_probability', 0.50))))


def _min_trade_probability(cfg: dict[str, Any]) -> float:
    bcfg = cfg.get('backtest', {}) or {}
    rcfg = cfg.get('replay', {}) or {}
    dcfg = cfg.get('direction_policy', {}) or {}
    return float(rcfg.get('min_trade_probability', bcfg.get('min_trade_probability', dcfg.get('min_trade_probability', 0.50))))


def _min_edge_pips(cfg: dict[str, Any]) -> float | None:
    rcfg = cfg.get('replay', {}) or {}
    dcfg = cfg.get('direction_policy', {}) or {}
    value = rcfg.get('min_edge_pips', dcfg.get('min_edge_pips', None))
    if value in (None, '', 'none', 'off', False):
        return None
    return float(value)


def _score_source_for_side(cfg: dict[str, Any], side: str) -> str:
    side_cfg = _side_cfg(cfg, side)
    rcfg = cfg.get('replay', {}) or {}
    return str(side_cfg.get('score_source', rcfg.get('score_source', f'{side.lower()}_side_score')) or f'{side.lower()}_side_score')


def _resolved_side_threshold_params(
    cfg: dict[str, Any],
    side: str,
    *,
    default_lookback: int,
    default_min_history: int,
    default_fallback: float,
) -> dict[str, Any]:
    rcfg = cfg.get('replay', {}) or {}
    scfg = _side_cfg(cfg, side)
    side_l = side.lower()
    default_quantile = float(rcfg.get(f'{side_l}_quantile', 0.985 if side_l == 'buy' else 0.990) or (0.985 if side_l == 'buy' else 0.990))
    fallback = _cfg_float(scfg.get('fallback_threshold', rcfg.get('fallback_threshold', default_fallback)), default_fallback)
    return {
        'allow': bool(_allow_side(cfg, side_l)),
        'score_source': _score_source_for_side(cfg, side_l),
        'lookback_bars': int(_cfg_int(scfg.get('lookback_bars', default_lookback), default_lookback) or default_lookback),
        'quantile': float(_cfg_float(scfg.get('quantile', default_quantile), default_quantile) or default_quantile),
        'fallback_threshold': float(fallback if fallback is not None else default_fallback),
        'min_history_bars': int(_cfg_int(scfg.get('min_history_bars', default_min_history), default_min_history) or default_min_history),
        'min_quality_score': _cfg_float(scfg.get('min_quality_score', None), None),
        'min_edge_pips': _cfg_float(scfg.get('min_edge_pips', None), None),
    }


def resolve_replay_decision_parameters(
    cfg: dict[str, Any],
    *,
    train_side: str | None = None,
    eval_start: str | None = None,
    eval_end: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Resolve the exact replay/deployment decision settings from a config.

    This is deliberately written as a pure resolver so training reports, replay
    summaries and live registries can all record the same fields. It records the
    resolved values after generated BUY-only/SELL-only configs have applied their
    allow_buy/allow_sell and loss-weight overrides.
    """
    rcfg = cfg.get('replay', {}) or {}
    bcfg = cfg.get('backtest', {}) or {}
    dcfg = cfg.get('direction_policy', {}) or {}
    tcfg = cfg.get('training', {}) or {}

    threshold_mode = _threshold_mode(cfg)
    rolling_thresholds = threshold_mode in {'rolling_score_quantile', 'rolling_quantile', 'score_quantile'}
    min_trade_prob = _min_trade_probability(cfg)
    default_lookback = int(_cfg_int(rcfg.get('lookback_bars', rcfg.get('rolling_lookback_bars', 4000)), 4000) or 4000)
    default_min_history = int(_cfg_int(rcfg.get('min_history_bars', max(200, min(default_lookback, 1000))), max(200, min(default_lookback, 1000))) or 1)
    default_fallback = float(_cfg_float(rcfg.get('fallback_threshold', min_trade_prob), min_trade_prob) or min_trade_prob)

    out: dict[str, Any] = {
        'symbol': symbol or cfg.get('_active_symbol'),
        'timeframe': timeframe_from_config(cfg),
        'train_side': train_side or training_side_from_config(cfg),
        'threshold_mode': threshold_mode,
        'rolling_thresholds_used': bool(rolling_thresholds),
        'allow_buy': bool(_allow_side(cfg, 'buy')),
        'allow_sell': bool(_allow_side(cfg, 'sell')),
        'min_direction_probability': float(_min_direction_probability(cfg)),
        'min_trade_probability': float(min_trade_prob),
        'min_edge_pips': _min_edge_pips(cfg),
        'min_gap_bars_between_same_side_trades': int(_cfg_int(rcfg.get('min_gap_bars_between_same_side_trades', rcfg.get('min_gap_bars', 0)), 0) or 0),
        'lookback_bars': int(default_lookback),
        'min_history_bars': int(default_min_history),
        'fallback_threshold': float(default_fallback),
        'buy': _resolved_side_threshold_params(
            cfg,
            'buy',
            default_lookback=default_lookback,
            default_min_history=default_min_history,
            default_fallback=default_fallback,
        ),
        'sell': _resolved_side_threshold_params(
            cfg,
            'sell',
            default_lookback=default_lookback,
            default_min_history=default_min_history,
            default_fallback=default_fallback,
        ),
        'replay_window': {
            'eval_start': eval_start if eval_start is not None else tcfg.get('replay_start', rcfg.get('eval_start')),
            'eval_end': eval_end if eval_end is not None else tcfg.get('replay_end', rcfg.get('eval_end')),
        },
        'score_settings': {
            'min_trades_for_score': int(_cfg_int(rcfg.get('min_trades_for_score', 50), 50) or 0),
            'score_net_pips_weight': float(_cfg_float(rcfg.get('score_net_pips_weight', 1.0), 1.0) or 0.0),
            'score_avg_pips_weight': float(_cfg_float(rcfg.get('score_avg_pips_weight', 50.0), 50.0) or 0.0),
            'score_win_rate_weight': float(_cfg_float(rcfg.get('score_win_rate_weight', 50.0), 50.0) or 0.0),
            'score_drawdown_weight': float(_cfg_float(rcfg.get('score_drawdown_weight', 0.10), 0.10) or 0.0),
            'score_side_underwater_weight': float(_cfg_float(rcfg.get('score_side_underwater_weight', 0.0), 0.0) or 0.0),
            'score_side_loss_pips_weight': float(_cfg_float(rcfg.get('score_side_loss_pips_weight', 0.0), 0.0) or 0.0),
        },
        'fixed_probability_fallbacks': {
            'backtest_min_direction_probability': bcfg.get('min_direction_probability'),
            'backtest_min_trade_probability': bcfg.get('min_trade_probability'),
            'direction_policy_min_direction_probability': dcfg.get('min_direction_probability'),
            'direction_policy_min_trade_probability': dcfg.get('min_trade_probability'),
        },
    }
    return _json_safe(out)


def resolve_training_decision_parameters(
    cfg: dict[str, Any],
    *,
    train_side: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Record training-side settings that affect side setup model behaviour."""
    tcfg = cfg.get('training', {}) or {}
    mcfg = cfg.get('model', {}) or {}
    out = {
        'symbol': symbol or cfg.get('_active_symbol'),
        'timeframe': timeframe_from_config(cfg),
        'train_side': train_side or training_side_from_config(cfg),
        'model_architecture': mcfg.get('architecture'),
        'decision_output_mode': mcfg.get('decision_output_mode'),
        'use_side_setup_heads': bool(mcfg.get('use_side_setup_heads', False)),
        'use_setup_quality_head': bool(mcfg.get('use_setup_quality_head', False)),
        'buy_setup_loss_weight': _cfg_float(tcfg.get('buy_setup_loss_weight', 1.0), 1.0),
        'sell_setup_loss_weight': _cfg_float(tcfg.get('sell_setup_loss_weight', 1.0), 1.0),
        'setup_quality_loss_weight': _cfg_float(tcfg.get('setup_quality_loss_weight', 0.0), 0.0),
        'buy_setup_pos_weight': tcfg.get('buy_setup_pos_weight', tcfg.get('_buy_setup_pos_weight')),
        'sell_setup_pos_weight': tcfg.get('sell_setup_pos_weight', tcfg.get('_sell_setup_pos_weight')),
        'model_selection_metric': tcfg.get('model_selection_metric'),
        'replay_each_epoch': bool(tcfg.get('replay_each_epoch', False)),
        'save_epoch_models': bool(tcfg.get('save_epoch_models', True)),
    }
    return _json_safe(out)


def config_snapshot(
    cfg: dict[str, Any],
    *,
    config_path: str | None = None,
    base_config_path: str | None = None,
    include_resolved_sections: bool = True,
) -> dict[str, Any]:
    """Return a stable hash and resolved section snapshot for reproducibility."""
    section_names = [
        'project', 'trading', 'paths', 'model', 'labels', 'training', 'replay',
        'backtest', 'direction_policy', 'external_trade_filter', 'risk', 'execution',
        'spread_control', 'features',
    ]
    resolved_sections = {name: copy.deepcopy(cfg.get(name)) for name in section_names if name in cfg}
    resolved_sections = _json_safe(resolved_sections)
    encoded = json.dumps(resolved_sections, sort_keys=True, default=str, separators=(',', ':')).encode('utf-8')
    snapshot = {
        'config_path': str(config_path or cfg.get('_config_path') or ''),
        'base_config_path': str(base_config_path or cfg.get('_base_config_path') or cfg.get('_config_path') or config_path or ''),
        'config_sha256': hashlib.sha256(encoded).hexdigest(),
        'contains_resolved_sections': bool(include_resolved_sections),
    }
    if include_resolved_sections:
        snapshot['resolved_sections'] = resolved_sections
    return snapshot
