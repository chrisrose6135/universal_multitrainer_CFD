from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .config import load_config_with_optional_spread_risk
from .direction_model import DIRECTION_CLASS_NAMES, DirectionTradePolicyNet, direction_probabilities_from_outputs
from .executor import send_order
from .external_trade_filter import external_trade_gate
from .forex import validate_forex_symbols
from .io_utils import ensure_dir, read_json
from .live_data import latest_processed_features
from .m5_bar_state import m5_bar_gate, mark_m5_bar_processed
from .mt5_client import shutdown_mt5
from .universal_symbol_features import add_universal_symbol_features


def _utc_now_iso() -> str:
    return pd.Timestamp.now('UTC').isoformat()


def _timeframe(cfg: dict[str, Any]) -> str:
    return str((cfg.get('trading') or {}).get('timeframe') or (cfg.get('project') or {}).get('timeframe') or 'M5').upper()


def _model_paths(symbol: str, cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = cfg.get('paths', {}) or {}
    model_dir = Path(paths.get('model_dir', 'models'))
    tf = _timeframe(cfg)
    live = cfg.get('live_direction_policy', {}) or {}
    model_path = Path(str(live.get('model_path', model_dir / f'{symbol}_{tf}_direction_policy.pt')).format(symbol=symbol, timeframe=tf))
    scaler_path = Path(str(live.get('scaler_path', model_dir / f'{symbol}_{tf}_direction_scaler.pkl')).format(symbol=symbol, timeframe=tf))
    features_path = Path(str(live.get('features_path', model_dir / f'{symbol}_{tf}_direction_features.json')).format(symbol=symbol, timeframe=tf))
    return model_path, scaler_path, features_path


def _read_feature_columns(path: Path) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        for key in ('feature_columns', 'features', 'columns'):
            if isinstance(payload.get(key), list):
                return [str(x) for x in payload[key]]
    raise ValueError(f'Could not read feature columns from {path}')


def _torch_load(path: Path, device: str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ('model_state_dict', 'model_state', 'state_dict'):
            value = payload.get(key)
            if isinstance(value, dict):
                return {str(k).replace('module.', ''): v for k, v in value.items()}
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return {str(k).replace('module.', ''): v for k, v in payload.items()}
    raise ValueError('Unsupported direction checkpoint format')


def load_direction_policy(symbol: str, cfg: dict[str, Any], device: str | None = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model_path, scaler_path, features_path = _model_paths(symbol, cfg)
    feature_columns = _read_feature_columns(features_path)
    scaler = joblib.load(scaler_path)
    payload = _torch_load(model_path, device)
    model_cfg = payload.get('model_config') if isinstance(payload, dict) and isinstance(payload.get('model_config'), dict) else None
    model_cfg_full = dict(cfg)
    if model_cfg is not None:
        model_cfg_full['model'] = model_cfg
    model_cfg_full['_feature_columns'] = list(feature_columns)
    model = DirectionTradePolicyNet(len(feature_columns), model_cfg_full).to(device)
    model.load_state_dict(_extract_state(payload), strict=True)
    model.eval()
    return model, scaler, feature_columns, device, {'model_path': str(model_path), 'scaler_path': str(scaler_path), 'features_path': str(features_path)}


def _live_sequence(df: pd.DataFrame, cfg: dict[str, Any], scaler: Any, feature_columns: list[str]) -> tuple[torch.Tensor, pd.Series]:
    seq_len = int((cfg.get('model') or {}).get('sequence_length', 64))
    fill = float((cfg.get('features') or {}).get('fillna_value', 0.0))
    df = add_universal_symbol_features(df, cfg)
    if len(df) < seq_len:
        raise RuntimeError(f'Not enough live feature rows: got {len(df)}, need sequence_length={seq_len}')
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f'Live feature frame is missing required columns: {missing[:20]}')
    X = (
        df[feature_columns]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill)
        .to_numpy(np.float32)
    )
    X = scaler.transform(X).astype(np.float32)
    seq = X[-seq_len:][None, :, :]
    return torch.tensor(seq, dtype=torch.float32), df.iloc[-1]


def _min_direction_probability(cfg: dict[str, Any]) -> float:
    dcfg = cfg.get('direction_policy', {}) or {}
    live = cfg.get('live_direction_policy', {}) or cfg.get('live', {}) or {}
    return float(live.get('min_direction_probability', dcfg.get('min_direction_probability', 0.50)))


def _min_trade_probability(cfg: dict[str, Any]) -> float:
    dcfg = cfg.get('direction_policy', {}) or {}
    live = cfg.get('live_direction_policy', {}) or cfg.get('live', {}) or {}
    return float(live.get('min_trade_probability', dcfg.get('min_trade_probability', 0.50)))


def _cfg_int(value: Any, default: int) -> int:
    try:
        if value in (None, ''):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _cfg_float(value: Any, default: float) -> float:
    try:
        if value in (None, ''):
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _replay_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get('replay', {}) or {}
    return raw if isinstance(raw, dict) else {}


def _side_threshold_cfg(cfg: dict[str, Any], side: str) -> dict[str, Any]:
    rcfg = _replay_cfg(cfg)
    raw = rcfg.get(side.lower()) or rcfg.get(side.upper()) or {}
    return raw if isinstance(raw, dict) else {}


def _allow_replay_side(cfg: dict[str, Any], side: str) -> bool:
    rcfg = _replay_cfg(cfg)
    value = rcfg.get(f'allow_{side.lower()}', True)
    if isinstance(value, str):
        return value.strip().lower() not in {'0', 'false', 'no', 'off', 'disabled'}
    return bool(value)


def _threshold_mode(cfg: dict[str, Any]) -> str:
    rcfg = _replay_cfg(cfg)
    return str(rcfg.get('threshold_mode', rcfg.get('score_threshold_mode', 'fixed_probability')) or 'fixed_probability').strip().lower()


def _uses_rolling_quantile_thresholds(cfg: dict[str, Any]) -> bool:
    return _threshold_mode(cfg) in {'rolling_score_quantile', 'rolling_quantile', 'score_quantile'}


def _default_rolling_lookback(cfg: dict[str, Any]) -> int:
    rcfg = _replay_cfg(cfg)
    return max(1, _cfg_int(rcfg.get('lookback_bars', rcfg.get('rolling_lookback_bars', 4000)), 4000))


def _default_rolling_min_history(cfg: dict[str, Any], default_lookback: int | None = None) -> int:
    rcfg = _replay_cfg(cfg)
    lb = int(default_lookback or _default_rolling_lookback(cfg))
    default = max(200, min(lb, 1000))
    return max(1, _cfg_int(rcfg.get('min_history_bars', default), default))


def _rolling_side_params(cfg: dict[str, Any], side: str) -> dict[str, Any]:
    rcfg = _replay_cfg(cfg)
    scfg = _side_threshold_cfg(cfg, side)
    side_l = side.lower()
    default_lookback = _default_rolling_lookback(cfg)
    default_min_history = _default_rolling_min_history(cfg, default_lookback)
    default_quantile = 0.985 if side_l == 'buy' else 0.990
    default_quantile = _cfg_float(rcfg.get(f'{side_l}_quantile', default_quantile), default_quantile)
    min_trade_prob = _min_trade_probability(cfg)
    fallback = _cfg_float(scfg.get('fallback_threshold', rcfg.get('fallback_threshold', min_trade_prob)), min_trade_prob)
    return {
        'allow': _allow_replay_side(cfg, side_l),
        'lookback_bars': max(1, _cfg_int(scfg.get('lookback_bars', default_lookback), default_lookback)),
        'quantile': _cfg_float(scfg.get('quantile', default_quantile), default_quantile),
        'fallback_threshold': fallback,
        'min_history_bars': max(1, _cfg_int(scfg.get('min_history_bars', default_min_history), default_min_history)),
    }


def _live_requested_bars(cfg: dict[str, Any]) -> int:
    """Return the MT5 raw-bar request count used by live inference.

    Live replay-equivalence needs enough history to compute the BUY/SELL rolling
    quantile thresholds. This deliberately prioritises replay.buy.lookback_bars
    and replay.sell.lookback_bars over the old live.bars setting. A small feature
    buffer is added because build_feature_frame drops warm-up rows for rolling
    indicators before sequences are created.
    """
    model_cfg = cfg.get('model', {}) or {}
    live = cfg.get('live_direction_policy', {}) or {}
    seq_len = max(1, _cfg_int(model_cfg.get('sequence_length', 64), 64))
    if _uses_rolling_quantile_thresholds(cfg):
        buy = _rolling_side_params(cfg, 'buy')
        sell = _rolling_side_params(cfg, 'sell')
        needed_scores = max(
            int(buy['lookback_bars']),
            int(sell['lookback_bars']),
            int(buy['min_history_bars']),
            int(sell['min_history_bars']),
        )
        feature_buffer = max(0, _cfg_int(live.get('feature_history_buffer_bars', live.get('rolling_feature_buffer_bars', 250)), 250))
        return int(needed_scores + seq_len + feature_buffer + 1)
    return max(seq_len, _cfg_int(live.get('bars', 800), 800))


def _rolling_quantile_threshold(scores: np.ndarray, *, lookback: int, quantile: float, fallback: float, min_history: int) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=float))
    # Match replay: the current closed bar must not help set its own threshold.
    th = s.shift(1).rolling(window=max(int(lookback), 1), min_periods=max(int(min_history), 1)).quantile(float(quantile))
    return th.fillna(float(fallback)).to_numpy(float)


def _live_feature_sequences(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    scaler: Any,
    feature_columns: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    seq_len = int((cfg.get('model') or {}).get('sequence_length', 64))
    fill = float((cfg.get('features') or {}).get('fillna_value', 0.0))
    df = add_universal_symbol_features(df, cfg)
    if len(df) < seq_len:
        raise RuntimeError(f'Not enough live feature rows: got {len(df)}, need sequence_length={seq_len}')
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f'Live feature frame is missing required columns: {missing[:20]}')
    X = (
        df[feature_columns]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill)
        .to_numpy(np.float32)
    )
    X = scaler.transform(X).astype(np.float32)
    sequences = np.stack([X[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(X))]).astype(np.float32)
    endpoint_rows = df.iloc[seq_len - 1:].reset_index(drop=True)
    return sequences, endpoint_rows


def _predict_live_sequences(model: torch.nn.Module, sequences: np.ndarray, device: str, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    live = cfg.get('live_direction_policy', {}) or {}
    rcfg = _replay_cfg(cfg)
    batch_size = max(1, _cfg_int(live.get('inference_batch_size', rcfg.get('batch_size', 1024)), 1024))
    collected: dict[str, list[np.ndarray]] = {
        'probabilities': [],
        'trade_probability': [],
        'side_sell_probability': [],
        'side_buy_probability': [],
        'buy_edge_pips': [],
        'sell_edge_pips': [],
        'buy_setup_probability': [],
        'sell_setup_probability': [],
        'buy_setup_quality_score': [],
        'sell_setup_quality_score': [],
    }
    n = int(len(sequences))
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = torch.tensor(sequences[start:start + batch_size], dtype=torch.float32, device=device)
            outputs = model(xb)
            probs = direction_probabilities_from_outputs(outputs).detach().cpu().numpy()
            collected['probabilities'].append(probs)
            fallback_trade = probs[:, 0] + probs[:, 2]
            for key in [
                'trade_probability', 'side_sell_probability', 'side_buy_probability',
                'buy_edge_pips', 'sell_edge_pips', 'buy_setup_probability',
                'sell_setup_probability', 'buy_setup_quality_score', 'sell_setup_quality_score',
            ]:
                value = outputs.get(key)
                if value is None:
                    if key == 'trade_probability':
                        arr = fallback_trade.astype(float)
                    else:
                        arr = np.full(len(probs), np.nan, dtype=float)
                else:
                    arr = value.detach().cpu().view(-1).numpy().astype(float)
                collected[key].append(arr)
    return {k: np.concatenate(v) if v else np.asarray([], dtype=float) for k, v in collected.items()}


def _side_score_arrays_from_predictions(pred: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    probs = pred['probabilities']
    buy_setup = pred.get('buy_setup_probability', np.full(len(probs), np.nan))
    sell_setup = pred.get('sell_setup_probability', np.full(len(probs), np.nan))
    buy_score = np.where(np.isfinite(buy_setup), buy_setup, probs[:, 2])
    sell_score = np.where(np.isfinite(sell_setup), sell_setup, probs[:, 0])
    return buy_score.astype(float), sell_score.astype(float)


def _latest_rolling_quantile_live_decision(pred: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, Any]:
    probs = pred['probabilities']
    if len(probs) == 0:
        raise RuntimeError('No live prediction rows available for rolling quantile decision')
    buy_params = _rolling_side_params(cfg, 'buy')
    sell_params = _rolling_side_params(cfg, 'sell')
    buy_scores, sell_scores = _side_score_arrays_from_predictions(pred)
    buy_thresholds = _rolling_quantile_threshold(
        buy_scores,
        lookback=int(buy_params['lookback_bars']),
        quantile=float(buy_params['quantile']),
        fallback=float(buy_params['fallback_threshold']),
        min_history=int(buy_params['min_history_bars']),
    )
    sell_thresholds = _rolling_quantile_threshold(
        sell_scores,
        lookback=int(sell_params['lookback_bars']),
        quantile=float(sell_params['quantile']),
        fallback=float(sell_params['fallback_threshold']),
        min_history=int(sell_params['min_history_bars']),
    )
    idx = -1
    buy_score = float(buy_scores[idx])
    sell_score = float(sell_scores[idx])
    buy_threshold = float(buy_thresholds[idx])
    sell_threshold = float(sell_thresholds[idx])
    buy_pass = bool(buy_params['allow']) and buy_score >= buy_threshold
    sell_pass = bool(sell_params['allow']) and sell_score >= sell_threshold
    if buy_pass and sell_pass:
        side = 'BUY' if (buy_score - buy_threshold) >= (sell_score - sell_threshold) else 'SELL'
    elif buy_pass:
        side = 'BUY'
    elif sell_pass:
        side = 'SELL'
    else:
        side = 'NO_TRADE'
    probs_last = probs[idx]
    selected_prob = buy_score if side == 'BUY' else sell_score if side == 'SELL' else max(buy_score, sell_score)
    trade_probability = selected_prob if side != 'NO_TRADE' else max(buy_score, sell_score)
    return {
        'side': side,
        'pred_class': {'SELL': 0, 'NO_TRADE': 1, 'BUY': 2}[side],
        'selected_probability': float(selected_prob),
        'trade_probability': float(trade_probability),
        'probabilities': probs_last,
        'buy_side_score': buy_score,
        'sell_side_score': sell_score,
        'buy_rolling_threshold': buy_threshold,
        'sell_rolling_threshold': sell_threshold,
        'buy_threshold_margin': float(buy_score - buy_threshold),
        'sell_threshold_margin': float(sell_score - sell_threshold),
        'buy_pass_rolling_quantile': buy_pass,
        'sell_pass_rolling_quantile': sell_pass,
        'threshold_mode': _threshold_mode(cfg),
        'rolling_thresholds_used': True,
        'buy_threshold_params': buy_params,
        'sell_threshold_params': sell_params,
        'rolling_prediction_rows': int(len(probs)),
    }


def _latest_fixed_threshold_live_decision(pred: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, Any]:
    probs = pred['probabilities'][-1]
    pred_class = int(np.argmax(probs))
    side = DIRECTION_CLASS_NAMES[pred_class]
    selected_prob = float(np.max(probs))
    trade_probability = float(pred.get('trade_probability', np.asarray([probs[0] + probs[2]], dtype=float))[-1])
    return {
        'side': side,
        'pred_class': pred_class,
        'selected_probability': selected_prob,
        'trade_probability': trade_probability,
        'probabilities': probs,
        'buy_side_score': None,
        'sell_side_score': None,
        'buy_rolling_threshold': None,
        'sell_rolling_threshold': None,
        'buy_threshold_margin': None,
        'sell_threshold_margin': None,
        'buy_pass_rolling_quantile': None,
        'sell_pass_rolling_quantile': None,
        'threshold_mode': _threshold_mode(cfg),
        'rolling_thresholds_used': False,
        'buy_threshold_params': _rolling_side_params(cfg, 'buy'),
        'sell_threshold_params': _rolling_side_params(cfg, 'sell'),
        'rolling_prediction_rows': int(len(pred['probabilities'])),
    }


def _min_edge_pips(cfg: dict[str, Any]) -> float | None:
    dcfg = cfg.get('direction_policy', {}) or {}
    live = cfg.get('live_direction_policy', {}) or cfg.get('live', {}) or {}
    value = live.get('min_edge_pips', dcfg.get('min_edge_pips', None))
    if value in (None, '', 'none', 'off', False):
        return None
    return float(value)


def _normalise_trade_mode(value: Any) -> str:
    if value is None:
        return 'buy_sell'
    mode = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        '': 'buy_sell',
        'all': 'buy_sell',
        'both': 'buy_sell',
        'buy_sell': 'buy_sell',
        'buy_and_sell': 'buy_sell',
        'both_sides': 'buy_sell',
        'long_short': 'buy_sell',
        'buy': 'buy_only',
        'long': 'buy_only',
        'buy_only': 'buy_only',
        'long_only': 'buy_only',
        'sell': 'sell_only',
        'short': 'sell_only',
        'sell_only': 'sell_only',
        'short_only': 'sell_only',
        'none': 'disabled',
        'off': 'disabled',
        'disabled': 'disabled',
        'block': 'disabled',
        'blocked': 'disabled',
        'no_trade': 'disabled',
    }
    return aliases.get(mode, f'invalid:{mode}')


def _symbol_trade_mode(symbol: str, cfg: dict[str, Any]) -> str:
    """Return the live-only trade mode for a symbol.

    Supported config forms:

      symbol_trade_modes:
        symbols:
          US500: buy_sell
          NAS100: buy_only

    and, for convenience:

      symbol_trade_modes:
        US500: buy_sell
        NAS100: buy_only

    This is intentionally applied only in live_direction_policy.py.
    Training and replay remain unrestricted unless patched separately.
    """
    block = cfg.get('symbol_trade_modes') or {}
    if not isinstance(block, dict):
        return 'buy_sell'
    modes = block.get('symbols') if isinstance(block.get('symbols'), dict) else block
    if not isinstance(modes, dict):
        return 'buy_sell'
    raw = None
    if symbol in modes:
        raw = modes.get(symbol)
    elif symbol.upper() in modes:
        raw = modes.get(symbol.upper())
    elif symbol.lower() in modes:
        raw = modes.get(symbol.lower())
    return _normalise_trade_mode(raw)


def _trade_mode_allows_side(trade_mode: str, side: str) -> bool:
    side = str(side).upper()
    if trade_mode == 'buy_sell':
        return side in {'BUY', 'SELL'}
    if trade_mode == 'buy_only':
        return side == 'BUY'
    if trade_mode == 'sell_only':
        return side == 'SELL'
    if trade_mode == 'disabled':
        return False
    if trade_mode.startswith('invalid:'):
        return False
    return False


def _trade_mode_block_reason(symbol: str, trade_mode: str, side: str) -> str:
    if trade_mode.startswith('invalid:'):
        return f'invalid_symbol_trade_mode:{trade_mode.split(":", 1)[1]}'
    return f'symbol_trade_mode_{trade_mode}_blocks_{str(side).lower()}'


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ''):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _daily_loss_window_utc(cfg: dict[str, Any]) -> tuple[datetime, datetime, str]:
    """Return the broker-day window expressed in UTC.

    If the config says to apply the broker server timezone offset, the "day"
    rolls over at broker midnight. Otherwise it rolls over at UTC midnight.
    """
    mt5_cfg = cfg.get('mt5', {}) or {}
    use_broker_offset = bool(mt5_cfg.get('apply_broker_server_timezone_offset', False))
    offset_hours = _safe_float(mt5_cfg.get('broker_server_utc_offset_hours'), 0.0) if use_broker_offset else 0.0
    offset_hours = float(offset_hours or 0.0)
    now_utc = datetime.now(timezone.utc)
    broker_now = now_utc + timedelta(hours=offset_hours)
    broker_day_start = broker_now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = broker_day_start - timedelta(hours=offset_hours)
    label = 'broker_day' if use_broker_offset else 'utc_day'
    return start_utc, now_utc, label


def _load_mt5() -> tuple[Any | None, str | None]:
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5, None
    except Exception as exc:  # pragma: no cover - depends on local MT5 install
        return None, str(exc)


def _load_mt5_for_daily_loss() -> tuple[Any | None, str | None]:
    return _load_mt5()


def _deal_profit_components(deal: Any) -> float:
    total = 0.0
    for name in ('profit', 'swap', 'commission', 'fee'):
        total += float(getattr(deal, name, 0.0) or 0.0)
    return total


def _account_daily_loss_gate(cfg: dict[str, Any], *, mode: str) -> tuple[bool, str, dict[str, Any]]:
    """Account-wide daily loss limiter for live/demo order creation.

    Config:
      risk:
        max_daily_loss_percent_of_balance: 2.0  # null/0 disables
        daily_loss_include_open_positions: true
        daily_loss_fail_closed: true
        daily_loss_check_in_paper: false

    The gate sums today's realised trade P/L from MT5 deals and, optionally,
    current floating P/L from open positions. If the loss reaches the configured
    percentage of current account balance, new orders are blocked.
    """
    risk_cfg = cfg.get('risk', {}) or {}
    max_pct = _safe_float(
        risk_cfg.get('max_daily_loss_percent_of_balance', risk_cfg.get('max_daily_loss_percent')),
        None,
    )
    include_open = bool(risk_cfg.get('daily_loss_include_open_positions', True))
    fail_closed = bool(risk_cfg.get('daily_loss_fail_closed', True))
    check_in_paper = bool(risk_cfg.get('daily_loss_check_in_paper', False))

    diagnostics: dict[str, Any] = {
        'enabled': bool(max_pct is not None and max_pct > 0.0),
        'mode': mode,
        'max_daily_loss_percent_of_balance': max_pct,
        'include_open_positions': include_open,
        'fail_closed': fail_closed,
        'check_in_paper': check_in_paper,
    }

    if max_pct is None or max_pct <= 0.0:
        diagnostics['status'] = 'disabled'
        return True, 'daily_loss_limit_disabled', diagnostics

    if mode not in {'demo', 'live'} and not check_in_paper:
        diagnostics['status'] = 'skipped_for_mode'
        return True, 'daily_loss_limit_skipped_for_mode', diagnostics

    mt5, import_error = _load_mt5_for_daily_loss()
    if mt5 is None:
        diagnostics.update({'status': 'error', 'error': f'MetaTrader5 import failed: {import_error}'})
        if fail_closed:
            return False, 'daily_loss_gate_error', diagnostics
        return True, 'daily_loss_gate_error_ignored', diagnostics

    try:
        account = mt5.account_info()
    except Exception as exc:
        diagnostics.update({'status': 'error', 'error': f'account_info failed: {exc}'})
        if fail_closed:
            return False, 'daily_loss_gate_error', diagnostics
        return True, 'daily_loss_gate_error_ignored', diagnostics

    if account is None:
        diagnostics.update({'status': 'error', 'error': 'MT5 account_info returned None'})
        if fail_closed:
            return False, 'daily_loss_no_account_info', diagnostics
        return True, 'daily_loss_no_account_info_ignored', diagnostics

    balance = _safe_float(getattr(account, 'balance', None), None)
    equity = _safe_float(getattr(account, 'equity', None), None)
    if balance is None or balance <= 0.0:
        diagnostics.update({'status': 'error', 'error': f'invalid account balance: {balance}'})
        if fail_closed:
            return False, 'daily_loss_invalid_balance', diagnostics
        return True, 'daily_loss_invalid_balance_ignored', diagnostics

    start_utc, now_utc, day_mode = _daily_loss_window_utc(cfg)
    closed_pnl = 0.0
    deal_count = 0
    skipped_deal_count = 0
    try:
        deals = mt5.history_deals_get(start_utc, now_utc)
        if deals is None:
            deals = []
        buy_type = getattr(mt5, 'DEAL_TYPE_BUY', None)
        sell_type = getattr(mt5, 'DEAL_TYPE_SELL', None)
        allowed_deal_types = {x for x in (buy_type, sell_type) if x is not None}
        for deal in deals:
            if allowed_deal_types:
                deal_type = getattr(deal, 'type', None)
                if deal_type not in allowed_deal_types:
                    skipped_deal_count += 1
                    continue
            closed_pnl += _deal_profit_components(deal)
            deal_count += 1
    except Exception as exc:
        diagnostics.update({'status': 'error', 'error': f'history_deals_get failed: {exc}'})
        if fail_closed:
            return False, 'daily_loss_gate_error', diagnostics
        return True, 'daily_loss_gate_error_ignored', diagnostics

    floating_pnl = 0.0
    position_count = 0
    if include_open:
        try:
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            for pos in positions:
                floating_pnl += float(getattr(pos, 'profit', 0.0) or 0.0)
                floating_pnl += float(getattr(pos, 'swap', 0.0) or 0.0)
                position_count += 1
        except Exception as exc:
            diagnostics.update({'status': 'error', 'error': f'positions_get failed: {exc}'})
            if fail_closed:
                return False, 'daily_loss_gate_error', diagnostics
            return True, 'daily_loss_gate_error_ignored', diagnostics

    daily_pnl = float(closed_pnl + (floating_pnl if include_open else 0.0))
    max_loss_amount = float(balance * (float(max_pct) / 100.0))
    loss_amount = max(0.0, -daily_pnl)
    loss_percent_of_balance = float((loss_amount / balance) * 100.0) if balance > 0 else 0.0
    allow = loss_amount < max_loss_amount

    diagnostics.update({
        'status': 'ok' if allow else 'limit_reached',
        'day_mode': day_mode,
        'window_start_utc': start_utc.isoformat(),
        'window_end_utc': now_utc.isoformat(),
        'account_balance': float(balance),
        'account_equity': None if equity is None else float(equity),
        'closed_trade_pnl_today': float(closed_pnl),
        'floating_open_pnl': float(floating_pnl),
        'daily_pnl_used': float(daily_pnl),
        'daily_loss_amount': float(loss_amount),
        'daily_loss_percent_of_balance': float(loss_percent_of_balance),
        'max_daily_loss_amount': float(max_loss_amount),
        'deals_counted_today': int(deal_count),
        'deals_skipped_today': int(skipped_deal_count),
        'open_positions_counted': int(position_count),
    })

    if not allow:
        return False, 'daily_loss_limit_reached', diagnostics
    return True, 'daily_loss_limit_ok', diagnostics


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    columns = list(row.keys())

    # If the CSV already exists, keep its original header. This avoids creating
    # malformed rows when the live logger gains extra diagnostic fields. To see
    # any newly added fields, rotate/delete the existing CSV before restarting.
    if exists and path.stat().st_size > 0:
        try:
            with path.open('r', encoding='utf-8', newline='') as rf:
                existing_header = next(csv.reader(rf), None)
            if existing_header:
                columns = [str(c) for c in existing_header]
        except Exception:
            columns = list(row.keys())

    serialised = {
        k: json.dumps(v, sort_keys=True, default=_json_default) if isinstance(v, (dict, list, tuple)) else v
        for k, v in row.items()
    }
    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(serialised)


def _write_csv_snapshot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a complete CSV snapshot so MT5 trade syncs do not duplicate rows."""
    ensure_dir(path.parent)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)
    if not columns:
        columns = [
            'snapshot_time_utc', 'status', 'symbol', 'side', 'trade_id', 'ticket',
            'position_id', 'open_time_utc', 'close_time_utc', 'volume', 'entry_price',
            'close_price', 'current_price', 'pips', 'pnl', 'profit', 'swap',
            'commission', 'fee', 'source', 'mode', 'data_source'
        ]
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: json.dumps(v, sort_keys=True, default=_json_default) if isinstance(v, (dict, list, tuple)) else v
                for k, v in row.items()
            })
    tmp.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding='utf-8')
    tmp.replace(path)


def _live_path(cfg: dict[str, Any], key: str, default: str) -> Path:
    live = cfg.get('live_direction_policy', {}) or {}
    return Path(str(live.get(key, default)))


def _pip_size(symbol: str) -> float:
    symbol = str(symbol).upper()
    return 0.01 if 'JPY' in symbol else 0.0001


def _mt5_time_to_iso(value: Any, *, milliseconds: bool = False) -> str | None:
    if value in (None, ''):
        return None
    try:
        unit = 'ms' if milliseconds else 's'
        return pd.to_datetime(float(value), unit=unit, utc=True).isoformat()
    except Exception:
        return None


def _object_attrs(obj: Any, names: list[str]) -> dict[str, Any]:
    return {name: getattr(obj, name, None) for name in names}


def _position_side(mt5: Any, pos: Any) -> str:
    pos_type = getattr(pos, 'type', None)
    if pos_type == getattr(mt5, 'POSITION_TYPE_BUY', 0):
        return 'BUY'
    if pos_type == getattr(mt5, 'POSITION_TYPE_SELL', 1):
        return 'SELL'
    return str(pos_type)


def _deal_type_side(mt5: Any, deal: Any) -> str:
    deal_type = getattr(deal, 'type', None)
    if deal_type == getattr(mt5, 'DEAL_TYPE_BUY', 0):
        return 'BUY'
    if deal_type == getattr(mt5, 'DEAL_TYPE_SELL', 1):
        return 'SELL'
    return str(deal_type)


def _deal_entry_name(mt5: Any, deal: Any) -> str:
    entry = getattr(deal, 'entry', None)
    names = {
        getattr(mt5, 'DEAL_ENTRY_IN', object()): 'IN',
        getattr(mt5, 'DEAL_ENTRY_OUT', object()): 'OUT',
        getattr(mt5, 'DEAL_ENTRY_INOUT', object()): 'INOUT',
        getattr(mt5, 'DEAL_ENTRY_OUT_BY', object()): 'OUT_BY',
    }
    return str(names.get(entry, entry))


def _closed_trade_rows_from_deals(mt5: Any, deals: list[Any], *, snapshot_time: str, mode: str, data_source: str) -> list[dict[str, Any]]:
    buy_type = getattr(mt5, 'DEAL_TYPE_BUY', None)
    sell_type = getattr(mt5, 'DEAL_TYPE_SELL', None)
    allowed_deal_types = {x for x in (buy_type, sell_type) if x is not None}
    close_entries = {
        getattr(mt5, 'DEAL_ENTRY_OUT', None),
        getattr(mt5, 'DEAL_ENTRY_INOUT', None),
        getattr(mt5, 'DEAL_ENTRY_OUT_BY', None),
    }
    close_entries = {x for x in close_entries if x is not None}
    entry_in_values = {getattr(mt5, 'DEAL_ENTRY_IN', None), getattr(mt5, 'DEAL_ENTRY_INOUT', None)}
    entry_in_values = {x for x in entry_in_values if x is not None}

    grouped: dict[str, list[Any]] = {}
    for deal in deals:
        if allowed_deal_types and getattr(deal, 'type', None) not in allowed_deal_types:
            continue
        position_id = getattr(deal, 'position_id', None) or getattr(deal, 'order', None) or getattr(deal, 'ticket', None)
        key = str(position_id)
        grouped.setdefault(key, []).append(deal)

    rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        group = sorted(group, key=lambda d: float(getattr(d, 'time_msc', None) or (getattr(d, 'time', 0) or 0) * 1000))
        close_deals = [d for d in group if getattr(d, 'entry', None) in close_entries]
        if close_entries and not close_deals:
            continue
        if not close_entries:
            close_deals = group

        entry_deals = [d for d in group if getattr(d, 'entry', None) in entry_in_values]
        first_entry = entry_deals[0] if entry_deals else None
        last_close = close_deals[-1]
        symbol = str(getattr(last_close, 'symbol', '') or (getattr(first_entry, 'symbol', '') if first_entry is not None else ''))

        # For a normal close deal, the close deal side is the opposite of the original position side.
        if first_entry is not None:
            side = _deal_type_side(mt5, first_entry)
        else:
            close_side = _deal_type_side(mt5, last_close)
            side = 'SELL' if close_side == 'BUY' else 'BUY' if close_side == 'SELL' else close_side

        entry_price = _safe_float(getattr(first_entry, 'price', None), None) if first_entry is not None else None
        close_price = _safe_float(getattr(last_close, 'price', None), None)
        pips = None
        if entry_price is not None and close_price is not None and symbol:
            pip = _pip_size(symbol)
            if side == 'BUY':
                pips = (close_price - entry_price) / pip
            elif side == 'SELL':
                pips = (entry_price - close_price) / pip

        profit = sum(float(getattr(d, 'profit', 0.0) or 0.0) for d in close_deals)
        swap = sum(float(getattr(d, 'swap', 0.0) or 0.0) for d in close_deals)
        commission = sum(float(getattr(d, 'commission', 0.0) or 0.0) for d in close_deals)
        fee = sum(float(getattr(d, 'fee', 0.0) or 0.0) for d in close_deals)
        pnl = float(profit + swap + commission + fee)
        volume = sum(float(getattr(d, 'volume', 0.0) or 0.0) for d in close_deals)

        rows.append({
            'snapshot_time_utc': snapshot_time,
            'status': 'CLOSED',
            'symbol': symbol,
            'side': side,
            'trade_id': key,
            'ticket': getattr(last_close, 'ticket', None),
            'position_id': getattr(last_close, 'position_id', None),
            'order': getattr(last_close, 'order', None),
            'open_time_utc': _mt5_time_to_iso(getattr(first_entry, 'time_msc', None), milliseconds=True) if first_entry is not None and getattr(first_entry, 'time_msc', None) else (_mt5_time_to_iso(getattr(first_entry, 'time', None)) if first_entry is not None else None),
            'close_time_utc': _mt5_time_to_iso(getattr(last_close, 'time_msc', None), milliseconds=True) if getattr(last_close, 'time_msc', None) else _mt5_time_to_iso(getattr(last_close, 'time', None)),
            'volume': float(volume),
            'entry_price': entry_price,
            'close_price': close_price,
            'current_price': None,
            'pips': None if pips is None else float(pips),
            'pnl': pnl,
            'profit': float(profit),
            'swap': float(swap),
            'commission': float(commission),
            'fee': float(fee),
            'deal_count': len(close_deals),
            'entry_type': _deal_entry_name(mt5, last_close),
            'source': 'mt5_history_deals',
            'mode': mode,
            'data_source': data_source,
        })
    return rows


def _open_trade_rows_from_positions(mt5: Any, positions: list[Any], *, snapshot_time: str, mode: str, data_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos in positions:
        symbol = str(getattr(pos, 'symbol', '') or '')
        side = _position_side(mt5, pos)
        entry_price = _safe_float(getattr(pos, 'price_open', None), None)
        current_price = _safe_float(getattr(pos, 'price_current', None), None)
        pips = None
        if entry_price is not None and current_price is not None and symbol:
            pip = _pip_size(symbol)
            if side == 'BUY':
                pips = (current_price - entry_price) / pip
            elif side == 'SELL':
                pips = (entry_price - current_price) / pip
        profit = float(getattr(pos, 'profit', 0.0) or 0.0)
        swap = float(getattr(pos, 'swap', 0.0) or 0.0)
        commission = float(getattr(pos, 'commission', 0.0) or 0.0)
        pnl = float(profit + swap + commission)
        rows.append({
            'snapshot_time_utc': snapshot_time,
            'status': 'OPEN',
            'symbol': symbol,
            'side': side,
            'trade_id': getattr(pos, 'ticket', None) or getattr(pos, 'identifier', None),
            'ticket': getattr(pos, 'ticket', None),
            'position_id': getattr(pos, 'identifier', None),
            'order': None,
            'open_time_utc': _mt5_time_to_iso(getattr(pos, 'time_msc', None), milliseconds=True) if getattr(pos, 'time_msc', None) else _mt5_time_to_iso(getattr(pos, 'time', None)),
            'close_time_utc': None,
            'volume': float(getattr(pos, 'volume', 0.0) or 0.0),
            'entry_price': entry_price,
            'close_price': None,
            'current_price': current_price,
            'sl': _safe_float(getattr(pos, 'sl', None), None),
            'tp': _safe_float(getattr(pos, 'tp', None), None),
            'pips': None if pips is None else float(pips),
            'pnl': pnl,
            'profit': float(profit),
            'swap': float(swap),
            'commission': float(commission),
            'fee': 0.0,
            'comment': getattr(pos, 'comment', None),
            'magic': getattr(pos, 'magic', None),
            'source': 'mt5_positions',
            'mode': mode,
            'data_source': data_source,
        })
    return rows


def _summarise_trade_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    open_rows = [r for r in rows if str(r.get('status')).upper() == 'OPEN']
    closed_rows = [r for r in rows if str(r.get('status')).upper() == 'CLOSED']
    closed_pnls = [float(r.get('pnl') or 0.0) for r in closed_rows]
    closed_pips = [float(r.get('pips') or 0.0) for r in closed_rows if r.get('pips') not in (None, '')]
    open_pnl = sum(float(r.get('pnl') or 0.0) for r in open_rows)
    open_pips = sum(float(r.get('pips') or 0.0) for r in open_rows if r.get('pips') not in (None, ''))
    wins = sum(1 for x in closed_pnls if x > 0.0)
    losses = sum(1 for x in closed_pnls if x < 0.0)
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.get('symbol') or '')
        if not symbol:
            continue
        item = by_symbol.setdefault(symbol, {'open_trades': 0, 'closed_trades': 0, 'open_pnl': 0.0, 'closed_pnl': 0.0, 'closed_pips': 0.0})
        pnl = float(row.get('pnl') or 0.0)
        pips = float(row.get('pips') or 0.0) if row.get('pips') not in (None, '') else 0.0
        if str(row.get('status')).upper() == 'OPEN':
            item['open_trades'] += 1
            item['open_pnl'] += pnl
        elif str(row.get('status')).upper() == 'CLOSED':
            item['closed_trades'] += 1
            item['closed_pnl'] += pnl
            item['closed_pips'] += pips
    return {
        'open_trades': len(open_rows),
        'closed_trades': len(closed_rows),
        'total_trades': len(rows),
        'open_pnl': float(open_pnl),
        'open_pips': float(open_pips),
        'closed_pnl': float(sum(closed_pnls)),
        'closed_pips': float(sum(closed_pips)),
        'overall_pnl': float(sum(closed_pnls) + open_pnl),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(wins / len(closed_rows)) if closed_rows else None,
        'avg_closed_pnl': float(sum(closed_pnls) / len(closed_rows)) if closed_rows else None,
        'avg_closed_pips': float(sum(closed_pips) / len(closed_pips)) if closed_pips else None,
        'by_symbol': by_symbol,
    }


def sync_trade_logs_and_summary(cfg: dict[str, Any], *, mode: str, data_source: str) -> dict[str, Any]:
    """Write dashboard-friendly live trade CSV and summary JSON snapshots.

    The signal CSV records model decisions. This function records the account/trade
    state the dashboard needs: active open positions, closed trades, and summary
    totals. The CSV is rewritten as a snapshot each cycle to avoid duplicated rows.
    """
    live = cfg.get('live_direction_policy', {}) or {}
    enabled = bool(live.get('sync_trade_logs', True))
    trades_csv = _live_path(cfg, 'trades_csv', 'logs/live_direction_trades.csv')
    summary_json = _live_path(cfg, 'summary_json', 'logs/live_direction_summary.json')
    open_trades_json = _live_path(cfg, 'open_trades_json', 'logs/live_direction_open_trades.json')
    trade_history_days = int(live.get('trade_history_days', 30) or 30)
    snapshot_time = _utc_now_iso()

    summary: dict[str, Any] = {
        'time_utc': snapshot_time,
        'enabled': enabled,
        'mode': mode,
        'data_source': data_source,
        'trades_csv': str(trades_csv),
        'summary_json': str(summary_json),
        'open_trades_json': str(open_trades_json),
        'trade_history_days': trade_history_days,
    }

    if not enabled:
        summary.update({'status': 'disabled'})
        _write_csv_snapshot(trades_csv, [])
        _write_json(summary_json, summary)
        _write_json(open_trades_json, {'time_utc': snapshot_time, 'open_trades': []})
        return summary

    if mode not in {'demo', 'live'} and not bool(live.get('sync_trade_logs_in_paper', False)):
        summary.update({'status': 'skipped_for_mode'})
        _write_csv_snapshot(trades_csv, [])
        _write_json(summary_json, summary | _summarise_trade_rows([]))
        _write_json(open_trades_json, {'time_utc': snapshot_time, 'open_trades': []})
        return summary

    mt5, import_error = _load_mt5()
    if mt5 is None:
        summary.update({'status': 'error', 'error': f'MetaTrader5 import failed: {import_error}'})
        _write_csv_snapshot(trades_csv, [])
        _write_json(summary_json, summary | _summarise_trade_rows([]))
        _write_json(open_trades_json, {'time_utc': snapshot_time, 'open_trades': []})
        return summary

    try:
        account = mt5.account_info()
    except Exception as exc:
        account = None
        summary.update({'account_error': f'account_info failed: {exc}'})

    if account is not None:
        summary.update({
            'account_balance': _safe_float(getattr(account, 'balance', None), None),
            'account_equity': _safe_float(getattr(account, 'equity', None), None),
            'account_profit': _safe_float(getattr(account, 'profit', None), None),
            'account_margin': _safe_float(getattr(account, 'margin', None), None),
            'account_free_margin': _safe_float(getattr(account, 'margin_free', None), None),
            'account_currency': getattr(account, 'currency', None),
        })

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=max(1, trade_history_days))

    try:
        positions = mt5.positions_get()
        if positions is None:
            positions = []
    except Exception as exc:
        positions = []
        summary.update({'positions_error': f'positions_get failed: {exc}'})

    try:
        deals = mt5.history_deals_get(start_utc, now_utc)
        if deals is None:
            deals = []
    except Exception as exc:
        deals = []
        summary.update({'deals_error': f'history_deals_get failed: {exc}'})

    open_rows = _open_trade_rows_from_positions(mt5, list(positions), snapshot_time=snapshot_time, mode=mode, data_source=data_source)
    closed_rows = _closed_trade_rows_from_deals(mt5, list(deals), snapshot_time=snapshot_time, mode=mode, data_source=data_source)
    rows = open_rows + closed_rows
    totals = _summarise_trade_rows(rows)
    summary.update({'status': 'ok', **totals})

    _write_csv_snapshot(trades_csv, rows)
    _write_json(summary_json, summary)
    _write_json(open_trades_json, {'time_utc': snapshot_time, 'open_trades': open_rows})
    return summary



def _cfg_int_from_blocks(cfg: dict[str, Any], key: str, default: int = 0) -> int:
    for block_name in ('live_direction_policy', 'risk', 'execution', 'trading'):
        block = cfg.get(block_name, {}) or {}
        if isinstance(block, dict) and key in block:
            try:
                value = int(block.get(key) or 0)
            except (TypeError, ValueError):
                value = default
            return max(0, value)
    return max(0, int(default))


def _cooldown_bars(cfg: dict[str, Any]) -> int:
    """Return the live per-symbol cooldown in closed bars.

    The common config key is risk.cooldown_bars, but this also accepts
    live_direction_policy.cooldown_bars, execution.cooldown_bars, or
    trading.cooldown_bars so existing configs keep working.
    """
    return _cfg_int_from_blocks(cfg, 'cooldown_bars', 0)


def _timeframe_seconds(cfg: dict[str, Any]) -> int:
    tf = _timeframe(cfg)
    units = {'S': 1, 'M': 60, 'H': 3600, 'D': 86400}
    if len(tf) >= 2 and tf[0] in units:
        try:
            return max(1, int(tf[1:]) * units[tf[0]])
        except ValueError:
            return 300
    # MetaTrader-style aliases commonly seen in configs.
    aliases = {'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800, 'H1': 3600, 'H4': 14400, 'D1': 86400}
    return int(aliases.get(tf, 300))


def _parse_bar_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ''):
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        return ts
    except Exception:
        return None


def _cooldown_state_path(cfg: dict[str, Any]) -> Path:
    return _live_path(cfg, 'cooldown_state_json', 'logs/live_direction_cooldown_state.json')


def _load_cooldown_state(cfg: dict[str, Any]) -> dict[str, Any]:
    path = _cooldown_state_path(cfg)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_cooldown_state(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    path = _cooldown_state_path(cfg)
    payload = dict(payload)
    payload['updated_at_utc'] = _utc_now_iso()
    _write_json(path, payload)


def _cooldown_key(symbol: str) -> str:
    return str(symbol).upper()


def _cooldown_gate(symbol: str, side: str, bar_time: str, cfg: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    cooldown_bars = _cooldown_bars(cfg)
    diagnostics: dict[str, Any] = {
        'enabled': cooldown_bars > 0,
        'cooldown_bars': cooldown_bars,
        'scope': 'symbol',
        'state_path': str(_cooldown_state_path(cfg)),
    }
    if cooldown_bars <= 0:
        diagnostics['status'] = 'disabled'
        return True, 'cooldown_disabled', diagnostics

    current_ts = _parse_bar_timestamp(bar_time)
    if current_ts is None:
        diagnostics.update({'status': 'invalid_current_bar_time', 'bar_time': bar_time})
        # Do not fail closed on parse problems; keep trading but expose diagnostics.
        return True, 'cooldown_invalid_current_bar_time_ignored', diagnostics

    state = _load_cooldown_state(cfg)
    symbol_state = (state.get('symbols') or {}).get(_cooldown_key(symbol), {})
    last_bar_time = symbol_state.get('last_order_bar_time') or symbol_state.get('last_allow_bar_time')
    last_ts = _parse_bar_timestamp(last_bar_time)
    if last_ts is None:
        diagnostics.update({'status': 'no_prior_trade', 'bar_time': current_ts.isoformat()})
        return True, 'cooldown_no_prior_trade', diagnostics

    tf_seconds = _timeframe_seconds(cfg)
    elapsed_seconds = float((current_ts - last_ts).total_seconds())
    elapsed_bars = int(elapsed_seconds // tf_seconds) if elapsed_seconds >= 0 else -1

    diagnostics.update({
        'status': 'ok',
        'bar_time': current_ts.isoformat(),
        'last_order_bar_time': last_ts.isoformat(),
        'last_order_side': symbol_state.get('last_order_side'),
        'last_order_time_utc': symbol_state.get('last_order_time_utc'),
        'timeframe_seconds': tf_seconds,
        'elapsed_bars': elapsed_bars,
    })

    # cooldown_bars means skip that many full bars after an accepted order.
    # Example: cooldown_bars=3 blocks the next 3 closed M5 bars and allows the 4th.
    if elapsed_bars <= cooldown_bars:
        remaining_bars = max(0, cooldown_bars - elapsed_bars + 1)
        diagnostics.update({'status': 'active', 'remaining_bars': remaining_bars})
        return False, f'cooldown_bars_active:{remaining_bars}_bars_remaining', diagnostics

    diagnostics.update({'status': 'passed', 'remaining_bars': 0})
    return True, 'cooldown_passed', diagnostics


def _record_cooldown_order(symbol: str, side: str, bar_time: str, cfg: dict[str, Any], *, mode: str, order_sent: bool, order_result: Any = None) -> None:
    cooldown_bars = _cooldown_bars(cfg)
    if cooldown_bars <= 0:
        return
    bar_ts = _parse_bar_timestamp(bar_time)
    if bar_ts is None:
        return
    state = _load_cooldown_state(cfg)
    symbols = state.setdefault('symbols', {})
    symbols[_cooldown_key(symbol)] = {
        'symbol': str(symbol).upper(),
        'last_order_bar_time': bar_ts.isoformat(),
        'last_order_time_utc': _utc_now_iso(),
        'last_order_side': str(side).upper(),
        'mode': mode,
        'order_sent': bool(order_sent),
        'order_result': str(order_result) if order_result is not None else None,
        'cooldown_bars': cooldown_bars,
    }
    _write_cooldown_state(cfg, state)

def evaluate_symbol(symbol: str, cfg: dict[str, Any], bundle: tuple, *, mode: str, data_source: str) -> dict[str, Any]:
    model, scaler, feature_columns, device, artifacts = bundle
    requested_bars = _live_requested_bars(cfg)
    feat, meta = latest_processed_features(symbol, cfg, bars=requested_bars)
    if len(feat) == 0:
        raise RuntimeError(f'No live feature rows returned for {symbol}')
    meta = dict(meta or {})
    meta['requested_bars'] = int(requested_bars)
    meta['requested_bars_source'] = 'replay_buy_sell_lookback_bars' if _uses_rolling_quantile_thresholds(cfg) else 'live_direction_policy.bars'

    # Do the M5 bar de-duplication before model inference and before CSV logging.
    # Once a symbol/bar has had a final decision, repeated polling of the same
    # closed candle should not bloat the signal CSV.
    latest_for_gate = feat.iloc[-1]
    bar_time = str(latest_for_gate.get('time_utc') or latest_for_gate.get('time') or '')
    ok_bar, bar_reason = m5_bar_gate(symbol, bar_time, cfg)
    if not ok_bar:
        return {
            'time_utc': _utc_now_iso(),
            'symbol': symbol,
            'mode': mode,
            'data_source': data_source,
            'bar_time': bar_time,
            'final_decision': 'SKIP',
            'reason': bar_reason,
            'direction': 'SKIP',
            'selected_probability': 0.0,
            'sell_probability': None,
            'no_trade_probability': None,
            'buy_probability': None,
            'threshold_mode': _threshold_mode(cfg),
            'rolling_thresholds_used': bool(_uses_rolling_quantile_thresholds(cfg)),
            'requested_history_bars': int(requested_bars),
            'min_direction_probability': _min_direction_probability(cfg),
            'min_trade_probability': _min_trade_probability(cfg),
            'min_edge_pips': _min_edge_pips(cfg),
            'symbol_trade_mode': _symbol_trade_mode(symbol, cfg),
            'symbol_trade_mode_allows_side': False,
            'daily_loss_allows_order': True,
            'daily_loss_gate': {},
            'cooldown_allows_order': True,
            'cooldown_gate': {},
            'spread_points': latest_for_gate.get('spread_points', None),
            'analytics_gate': {},
            'order_attempted': False,
            'order_sent': False,
            'order_result': None,
            'order_error': None,
            'feature_rows': len(feat),
            'live_meta': meta,
            'logged_to_csv': False,
            'skipped_existing_final_decision': True,
            **artifacts,
        }

    sequences, endpoint_rows = _live_feature_sequences(feat, cfg, scaler, feature_columns)
    latest_row = endpoint_rows.iloc[-1]
    pred = _predict_live_sequences(model, sequences, device, cfg)
    if _uses_rolling_quantile_thresholds(cfg):
        decision_info = _latest_rolling_quantile_live_decision(pred, cfg)
    else:
        decision_info = _latest_fixed_threshold_live_decision(pred, cfg)

    probs = decision_info['probabilities']
    pred_class = int(decision_info['pred_class'])
    side = str(decision_info['side'])
    selected_prob = float(decision_info['selected_probability'])
    trade_probability = float(decision_info['trade_probability'])
    side_sell_probability = float(pred['side_sell_probability'][-1]) if np.isfinite(pred['side_sell_probability'][-1]) else None
    side_buy_probability = float(pred['side_buy_probability'][-1]) if np.isfinite(pred['side_buy_probability'][-1]) else None
    buy_edge_pips = float(pred['buy_edge_pips'][-1]) if np.isfinite(pred['buy_edge_pips'][-1]) else None
    sell_edge_pips = float(pred['sell_edge_pips'][-1]) if np.isfinite(pred['sell_edge_pips'][-1]) else None
    buy_setup_probability = float(pred['buy_setup_probability'][-1]) if np.isfinite(pred['buy_setup_probability'][-1]) else None
    sell_setup_probability = float(pred['sell_setup_probability'][-1]) if np.isfinite(pred['sell_setup_probability'][-1]) else None
    buy_setup_quality_score = float(pred['buy_setup_quality_score'][-1]) if np.isfinite(pred['buy_setup_quality_score'][-1]) else None
    sell_setup_quality_score = float(pred['sell_setup_quality_score'][-1]) if np.isfinite(pred['sell_setup_quality_score'][-1]) else None
    selected_edge_pips = buy_edge_pips if side == 'BUY' else sell_edge_pips if side == 'SELL' else None
    min_prob = _min_direction_probability(cfg)
    min_trade_prob = _min_trade_probability(cfg)
    min_edge_pips = _min_edge_pips(cfg)
    symbol_trade_mode = _symbol_trade_mode(symbol, cfg)
    symbol_trade_mode_allows_side = _trade_mode_allows_side(symbol_trade_mode, side)


    final_decision = 'BLOCK'
    reason = ''
    analytics = {}
    order_attempted = False
    order_sent = False
    order_result = None
    order_error = None
    daily_loss_gate: dict[str, Any] = {}
    daily_loss_allows_order = True
    cooldown_gate: dict[str, Any] = {}
    cooldown_allows_order = True

    rolling_thresholds_used = bool(decision_info.get('rolling_thresholds_used', False))

    if side == 'NO_TRADE':
        reason = 'side_score_below_rolling_quantile' if rolling_thresholds_used else 'model_no_trade'
    elif not symbol_trade_mode_allows_side:
        reason = _trade_mode_block_reason(symbol, symbol_trade_mode, side)
    elif not rolling_thresholds_used and trade_probability < min_trade_prob:
        reason = 'trade_probability_low'
    elif not rolling_thresholds_used and selected_prob < min_prob:
        reason = 'direction_probability_low'
    elif min_edge_pips is not None and selected_edge_pips is not None and selected_edge_pips < float(min_edge_pips):
        reason = 'edge_pips_low'
    else:
        cooldown_allows_order, cooldown_reason, cooldown_gate = _cooldown_gate(symbol, side, bar_time, cfg)
        if not cooldown_allows_order:
            reason = cooldown_reason
            analytics = {'cooldown_gate': cooldown_gate}
        else:
            gate = external_trade_gate(symbol, side, latest_row, cfg)
            analytics = gate.diagnostics
            if not gate.allow:
                reason = '|'.join(gate.reasons) if gate.reasons else 'external_gate_blocked'
            else:
                daily_loss_allows_order, daily_loss_reason, daily_loss_gate = _account_daily_loss_gate(cfg, mode=mode)
                if not daily_loss_allows_order:
                    final_decision = 'BLOCK'
                    reason = daily_loss_reason
                else:
                    final_decision = 'ALLOW'
                    reason = 'ok'
                    if mode in {'demo', 'live'}:
                        order_attempted = True
                        try:
                            result = send_order(symbol, side, cfg)
                            order_result = str(result)
                            order_sent = True
                        except Exception as exc:
                            order_error = str(exc)
                            order_sent = False

    # Mark every completed evaluation as the final decision for this symbol/bar,
    # not only ALLOW decisions. This prevents repeated BLOCK / NO_TRADE rows for
    # the same closed candle on subsequent polling cycles.
    mark_m5_bar_processed(symbol, bar_time, cfg)

    # Start the per-symbol cooldown only after an accepted trade signal. In paper
    # mode there is no MT5 order to confirm, so ALLOW starts cooldown. In demo/live
    # it starts once send_order returned without raising an exception.
    if final_decision == 'ALLOW' and (mode == 'paper' or order_sent):
        _record_cooldown_order(symbol, side, bar_time, cfg, mode=mode, order_sent=order_sent, order_result=order_result)

    row = {
        'time_utc': _utc_now_iso(),
        'symbol': symbol,
        'mode': mode,
        'data_source': data_source,
        'bar_time': bar_time,
        'final_decision': final_decision,
        'reason': reason,
        'direction': side,
        'selected_probability': selected_prob,
        'sell_probability': float(probs[0]),
        'no_trade_probability': float(probs[1]),
        'buy_probability': float(probs[2]),
        'trade_probability': trade_probability,
        'side_sell_probability': side_sell_probability,
        'side_buy_probability': side_buy_probability,
        'buy_setup_probability': buy_setup_probability,
        'sell_setup_probability': sell_setup_probability,
        'buy_setup_quality_score': buy_setup_quality_score,
        'sell_setup_quality_score': sell_setup_quality_score,
        'buy_edge_pips': buy_edge_pips,
        'sell_edge_pips': sell_edge_pips,
        'selected_edge_pips': selected_edge_pips,
        'threshold_mode': decision_info.get('threshold_mode'),
        'rolling_thresholds_used': bool(decision_info.get('rolling_thresholds_used', False)),
        'rolling_prediction_rows': decision_info.get('rolling_prediction_rows'),
        'requested_history_bars': int(requested_bars),
        'buy_side_score': decision_info.get('buy_side_score'),
        'sell_side_score': decision_info.get('sell_side_score'),
        'buy_rolling_threshold': decision_info.get('buy_rolling_threshold'),
        'sell_rolling_threshold': decision_info.get('sell_rolling_threshold'),
        'buy_threshold_margin': decision_info.get('buy_threshold_margin'),
        'sell_threshold_margin': decision_info.get('sell_threshold_margin'),
        'buy_pass_rolling_quantile': decision_info.get('buy_pass_rolling_quantile'),
        'sell_pass_rolling_quantile': decision_info.get('sell_pass_rolling_quantile'),
        'buy_threshold_params': decision_info.get('buy_threshold_params'),
        'sell_threshold_params': decision_info.get('sell_threshold_params'),
        'min_direction_probability': min_prob,
        'min_trade_probability': min_trade_prob,
        'min_edge_pips': min_edge_pips,
        'symbol_trade_mode': symbol_trade_mode,
        'symbol_trade_mode_allows_side': bool(symbol_trade_mode_allows_side),
        'daily_loss_allows_order': bool(daily_loss_allows_order),
        'daily_loss_gate': daily_loss_gate,
        'cooldown_allows_order': bool(cooldown_allows_order),
        'cooldown_gate': cooldown_gate,
        'spread_points': latest_row.get('spread_points', None),
        'analytics_gate': analytics,
        'order_attempted': order_attempted,
        'order_sent': order_sent,
        'order_result': order_result,
        'order_error': order_error,
        'feature_rows': len(feat),
        'live_meta': meta,
        **artifacts,
    }
    log_path = Path((cfg.get('live_direction_policy') or {}).get('signals_csv', 'logs/live_direction_signals.csv'))
    _append_csv(log_path, row)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description='Live/demo runner for simple BUY/SELL/NO_TRADE direction policy')
    p.add_argument('--config', default='config/direction_settings_generic_multisymbol_31_symbols.yaml')
    p.add_argument('--symbols', nargs='+', default=None)
    p.add_argument('--mode', choices=['paper', 'demo', 'live'], default='paper')
    p.add_argument('--data-source', choices=['mt5'], default='mt5')
    p.add_argument('--poll-seconds', type=float, default=20.0)
    p.add_argument('--once', action='store_true')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    cfg = load_config_with_optional_spread_risk(args.config)
    symbols = validate_forex_symbols(args.symbols or ((cfg.get('trading') or {}).get('symbols') or ['US500']))
    bundles = {symbol: load_direction_policy(symbol, cfg, args.device) for symbol in symbols}
    try:
        while True:
            for symbol in symbols:
                try:
                    row = evaluate_symbol(symbol, cfg, bundles[symbol], mode=args.mode, data_source=args.data_source)
                    if row.get('skipped_existing_final_decision'):
                        print(f"{symbol}: SKIP already_decided bar={row.get('bar_time', '')}", flush=True)
                    else:
                        print(f"{symbol}: {row['final_decision']} {row['direction']} p={row['selected_probability']:.3f} reason={row['reason']}", flush=True)
                except Exception as exc:
                    print(f'{symbol}: ERROR {exc}', flush=True)
            try:
                summary = sync_trade_logs_and_summary(cfg, mode=args.mode, data_source=args.data_source)
                if summary.get('status') == 'ok':
                    print(
                        'TRADE_LOG_SYNC: '
                        f"open={summary.get('open_trades', 0)} "
                        f"closed={summary.get('closed_trades', 0)} "
                        f"overall_pnl={summary.get('overall_pnl', 0.0):.2f} "
                        f"closed_pips={summary.get('closed_pips', 0.0):.1f}",
                        flush=True,
                    )
                elif summary.get('status') not in {'skipped_for_mode', 'disabled'}:
                    print(f"TRADE_LOG_SYNC: {summary.get('status')} {summary.get('error', '')}", flush=True)
            except Exception as exc:
                print(f'TRADE_LOG_SYNC: ERROR {exc}', flush=True)
            if args.once:
                break
            time.sleep(float(args.poll_seconds))
    finally:
        shutdown_mt5()


if __name__ == '__main__':
    main()
