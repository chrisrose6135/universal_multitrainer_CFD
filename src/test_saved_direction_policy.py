from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .config import load_config_with_optional_spread_risk
from .direction_dataset import prepare_direction_arrays
from .direction_model import DIRECTION_CLASS_NAMES, DirectionTradePolicyNet, direction_probabilities_from_outputs
from .analytic_signals import ensure_analytic_signal_features
from .external_trade_filter import external_trade_gate
from .forex import validate_forex_symbols
from .io_utils import ensure_dir, normalise_time_column, read_json, read_processed_csv, write_json
from .simulation import simulate_trade_from_row, summarise_trades
from .targets import generate_direction_targets
from .replay_decision_parameters import config_snapshot, resolve_replay_decision_parameters


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


def _timeframe(cfg: dict[str, Any]) -> str:
    return str((cfg.get('trading') or {}).get('timeframe') or (cfg.get('project') or {}).get('timeframe') or 'M5').upper()


def _model_paths(symbol: str, cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = cfg.get('paths', {}) or {}
    model_dir = Path(paths.get('model_dir', 'models'))
    tf = _timeframe(cfg)
    return (
        model_dir / f'{symbol}_{tf}_direction_policy.pt',
        model_dir / f'{symbol}_{tf}_direction_scaler.pkl',
        model_dir / f'{symbol}_{tf}_direction_features.json',
    )


def _pregenerated_path(symbol: str, cfg: dict[str, Any]) -> Path:
    tcfg = cfg.get('training', {}) or {}
    root = Path(tcfg.get('direction_data_dir', 'data/direction'))
    template = tcfg.get('direction_data_template', '{symbol}_{timeframe}_direction_training.csv')
    if tcfg.get('pregenerated_direction_data_path'):
        return Path(str(tcfg['pregenerated_direction_data_path']).format(symbol=symbol, timeframe=_timeframe(cfg)))
    return root / str(template).format(symbol=symbol, timeframe=_timeframe(cfg))


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
            if isinstance(payload.get(key), dict):
                return {str(k).replace('module.', ''): v for k, v in payload[key].items()}
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return {str(k).replace('module.', ''): v for k, v in payload.items()}
    raise ValueError('Unsupported direction checkpoint format')


def _filter_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not start and not end:
        return df, {'date_filter_applied': False, 'rows_before_date_filter': int(len(df)), 'rows_after_date_filter': int(len(df))}
    out = normalise_time_column(df)
    times = pd.to_datetime(out['time_utc'], utc=True, errors='coerce')
    mask = pd.Series(True, index=out.index)
    start_ts = pd.to_datetime(start, utc=True) if start else None
    end_ts = pd.to_datetime(end, utc=True) if end else None
    if start_ts is not None:
        mask &= times >= start_ts
    if end_ts is not None:
        mask &= times < end_ts
    filtered = out.loc[mask].reset_index(drop=True)
    return filtered, {
        'date_filter_applied': True,
        'date_start_utc': str(start_ts) if start_ts is not None else None,
        'date_end_utc': str(end_ts) if end_ts is not None else None,
        'date_end_is_exclusive': True,
        'rows_before_date_filter': int(len(out)),
        'rows_after_date_filter': int(len(filtered)),
    }


def _load_dataframe(symbol: str, cfg: dict[str, Any], *, eval_start: str | None, eval_end: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    tcfg = cfg.get('training', {}) or {}
    use_pre = bool(tcfg.get('use_pregenerated_direction_data', True))
    pre_path = _pregenerated_path(symbol, cfg)
    if use_pre and pre_path.exists():
        df = pd.read_csv(pre_path)
        source = 'pregenerated_direction_csv'
    else:
        df = read_processed_csv(symbol, cfg)
        source = 'processed_csv_generated_direction_targets'
    raw_rows = int(len(df))
    df, date_info = _filter_date_range(df, eval_start, eval_end)
    if 'direction_target' not in df.columns:
        df = generate_direction_targets(df, symbol, cfg)
    df = ensure_analytic_signal_features(df, cfg)
    return df, {'source': source, 'pregenerated_path': str(pre_path), 'raw_rows': raw_rows, 'rows': int(len(df)), 'date_filter': date_info}


def _predict(model: DirectionTradePolicyNet, arr, device: str, batch_size: int) -> dict[str, np.ndarray]:
    probs: list[np.ndarray] = []
    trade_probs: list[np.ndarray] = []
    side_sell_probs: list[np.ndarray] = []
    side_buy_probs: list[np.ndarray] = []
    buy_edges: list[np.ndarray] = []
    sell_edges: list[np.ndarray] = []
    buy_setup_probs: list[np.ndarray] = []
    sell_setup_probs: list[np.ndarray] = []
    setup_trade_probs: list[np.ndarray] = []
    buy_setup_quality: list[np.ndarray] = []
    sell_setup_quality: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arr.X_seq), batch_size):
            x = torch.tensor(arr.X_seq[start:start + batch_size], dtype=torch.float32, device=device)
            outputs = model(x)
            p = direction_probabilities_from_outputs(outputs).cpu().numpy()
            probs.append(p)
            n = p.shape[0]
            trade_probs.append(outputs.get('trade_probability', torch.tensor(p[:, 0] + p[:, 2])).detach().cpu().numpy())
            side_sell_probs.append(outputs.get('side_sell_probability', torch.full((n,), np.nan, device=x.device)).detach().cpu().numpy())
            side_buy_probs.append(outputs.get('side_buy_probability', torch.full((n,), np.nan, device=x.device)).detach().cpu().numpy())
            buy_edges.append(outputs.get('buy_edge_pips', torch.full((n,), np.nan, device=x.device)).detach().cpu().numpy())
            sell_edges.append(outputs.get('sell_edge_pips', torch.full((n,), np.nan, device=x.device)).detach().cpu().numpy())
            buy_setup_probs.append(outputs.get('buy_setup_probability', torch.full((n,), np.nan, device=x.device)).detach().cpu().numpy())
            sell_setup_probs.append(outputs.get('sell_setup_probability', torch.full((n,), np.nan, device=x.device)).detach().cpu().numpy())
            setup_trade_probs.append(outputs.get('setup_trade_probability', torch.full((n,), np.nan, device=x.device)).detach().cpu().numpy())
            buy_setup_quality.append(outputs.get('buy_setup_quality_score', torch.full((n,), np.nan, device=x.device)).detach().cpu().numpy())
            sell_setup_quality.append(outputs.get('sell_setup_quality_score', torch.full((n,), np.nan, device=x.device)).detach().cpu().numpy())
    empty = np.empty((0, 3), dtype=float)
    return {
        'probabilities': np.concatenate(probs, axis=0) if probs else empty,
        'trade_probability': np.concatenate(trade_probs, axis=0) if trade_probs else np.asarray([], dtype=float),
        'side_sell_probability': np.concatenate(side_sell_probs, axis=0) if side_sell_probs else np.asarray([], dtype=float),
        'side_buy_probability': np.concatenate(side_buy_probs, axis=0) if side_buy_probs else np.asarray([], dtype=float),
        'buy_edge_pips': np.concatenate(buy_edges, axis=0) if buy_edges else np.asarray([], dtype=float),
        'sell_edge_pips': np.concatenate(sell_edges, axis=0) if sell_edges else np.asarray([], dtype=float),
        'buy_setup_probability': np.concatenate(buy_setup_probs, axis=0) if buy_setup_probs else np.asarray([], dtype=float),
        'sell_setup_probability': np.concatenate(sell_setup_probs, axis=0) if sell_setup_probs else np.asarray([], dtype=float),
        'setup_trade_probability': np.concatenate(setup_trade_probs, axis=0) if setup_trade_probs else np.asarray([], dtype=float),
        'buy_setup_quality_score': np.concatenate(buy_setup_quality, axis=0) if buy_setup_quality else np.asarray([], dtype=float),
        'sell_setup_quality_score': np.concatenate(sell_setup_quality, axis=0) if sell_setup_quality else np.asarray([], dtype=float),
    }


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
    bcfg = cfg.get('backtest', {}) or {}
    rcfg = cfg.get('replay', {}) or {}
    dcfg = cfg.get('direction_policy', {}) or {}
    value = rcfg.get('min_edge_pips', bcfg.get('min_edge_pips', dcfg.get('min_edge_pips', None)))
    if value in (None, '', 'none', 'off', False):
        return None
    return float(value)



def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _replay_threshold_mode(cfg: dict[str, Any]) -> str:
    rcfg = cfg.get('replay', {}) or {}
    return str(rcfg.get('threshold_mode', rcfg.get('score_threshold_mode', 'fixed_probability')) or 'fixed_probability').strip().lower()


def _side_cfg(cfg: dict[str, Any], side: str) -> dict[str, Any]:
    rcfg = cfg.get('replay', {}) or {}
    side_l = side.lower()
    raw = rcfg.get(side_l) or rcfg.get(side.upper()) or {}
    return raw if isinstance(raw, dict) else {}


def _allow_side(cfg: dict[str, Any], side: str) -> bool:
    rcfg = cfg.get('replay', {}) or {}
    key = f'allow_{side.lower()}'
    return _cfg_bool(rcfg.get(key, True), True)


def _rolling_quantile_threshold(scores: np.ndarray, *, lookback: int, quantile: float, fallback: float, min_history: int) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=float))
    # Use only previous rows so the current bar never helps set its own threshold.
    th = s.shift(1).rolling(window=max(int(lookback), 1), min_periods=max(int(min_history), 1)).quantile(float(quantile))
    return th.fillna(float(fallback)).to_numpy(float)


def _side_score_arrays(pred: dict[str, np.ndarray], probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    buy_setup = pred.get('buy_setup_probability', np.full(len(probs), np.nan))
    sell_setup = pred.get('sell_setup_probability', np.full(len(probs), np.nan))
    buy_score = np.where(np.isfinite(buy_setup), buy_setup, probs[:, 2])
    sell_score = np.where(np.isfinite(sell_setup), sell_setup, probs[:, 0])
    return buy_score.astype(float), sell_score.astype(float)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def _replay_score(summary: dict[str, Any], cfg: dict[str, Any]) -> float:
    rcfg = cfg.get('replay', {}) or {}
    min_trades = int(rcfg.get('min_trades_for_score', 50) or 0)
    trades = int(summary.get('trades', 0) or 0)
    if trades < min_trades:
        return -1_000_000_000.0 + trades

    net = float(summary.get('net_pips', 0.0) or 0.0)
    avg = float(summary.get('average_net_pips', 0.0) or 0.0)
    win = float(summary.get('win_rate', 0.0) or 0.0)
    dd = float(summary.get('max_drawdown_pips', 0.0) or 0.0)

    # Side-aware penalties prevent epoch selection from hiding a large losing
    # SELL book behind a profitable BUY book, or vice versa. The total net-pips
    # term already penalises side losses once; these optional terms add an extra
    # penalty when a side is underwater. This directly targets the case where
    # US500 learns high-confidence SELL signals but replay SELL net pips are
    # strongly negative.
    buy_underwater = max(0.0, -float(summary.get('buy_net_pips', 0.0) or 0.0))
    sell_underwater = max(0.0, -float(summary.get('sell_net_pips', 0.0) or 0.0))
    buy_loss_pips = float(summary.get('buy_loss_pips', 0.0) or 0.0)
    sell_loss_pips = float(summary.get('sell_loss_pips', 0.0) or 0.0)

    score = (
        net * float(rcfg.get('score_net_pips_weight', 1.0))
        + avg * float(rcfg.get('score_avg_pips_weight', 50.0))
        + win * float(rcfg.get('score_win_rate_weight', 50.0))
        - dd * float(rcfg.get('score_drawdown_weight', 0.10))
        - buy_underwater * float(rcfg.get('score_buy_underwater_weight', rcfg.get('score_side_underwater_weight', 0.0)) or 0.0)
        - sell_underwater * float(rcfg.get('score_sell_underwater_weight', rcfg.get('score_side_underwater_weight', 0.0)) or 0.0)
        - buy_loss_pips * float(rcfg.get('score_buy_loss_pips_weight', rcfg.get('score_side_loss_pips_weight', 0.0)) or 0.0)
        - sell_loss_pips * float(rcfg.get('score_sell_loss_pips_weight', rcfg.get('score_side_loss_pips_weight', 0.0)) or 0.0)
    )
    return float(score)


def replay_symbol(
    symbol: str,
    cfg: dict[str, Any],
    *,
    model_path: Path | None,
    scaler_path: Path | None,
    features_path: Path | None,
    eval_start: str | None,
    eval_end: str | None,
    output_prefix: str | None,
    device: str | None,
    verbose: bool,
) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg['_active_symbol'] = symbol
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    default_model, default_scaler, default_features = _model_paths(symbol, cfg)
    model_path = Path(model_path or default_model)
    scaler_path = Path(scaler_path or default_scaler)
    features_path = Path(features_path or default_features)

    feature_columns = _read_feature_columns(features_path)
    scaler = joblib.load(scaler_path)
    payload = _torch_load(model_path, device)
    state = _extract_state(payload)
    model_cfg = payload.get('model_config') if isinstance(payload, dict) and isinstance(payload.get('model_config'), dict) else None
    model_cfg_full = dict(cfg)
    if model_cfg is not None:
        model_cfg_full['model'] = model_cfg
    model_cfg_full['_feature_columns'] = list(feature_columns)
    model = DirectionTradePolicyNet(len(feature_columns), model_cfg_full).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    df, data_info = _load_dataframe(symbol, cfg, eval_start=eval_start, eval_end=eval_end)
    arr = prepare_direction_arrays(df, cfg, scaler=scaler, feature_columns=feature_columns, fit_scaler=False)
    pred = _predict(model, arr, device, int((cfg.get('backtest') or {}).get('batch_size', 2048)))
    probs = pred['probabilities']
    trade_probs = pred.get('trade_probability', probs[:, 0] + probs[:, 2])
    side_sell_probs = pred.get('side_sell_probability', np.full(len(probs), np.nan))
    side_buy_probs = pred.get('side_buy_probability', np.full(len(probs), np.nan))
    buy_edge_pips_arr = pred.get('buy_edge_pips', np.full(len(probs), np.nan))
    sell_edge_pips_arr = pred.get('sell_edge_pips', np.full(len(probs), np.nan))
    buy_setup_prob_arr = pred.get('buy_setup_probability', np.full(len(probs), np.nan))
    sell_setup_prob_arr = pred.get('sell_setup_probability', np.full(len(probs), np.nan))
    buy_setup_quality_arr = pred.get('buy_setup_quality_score', np.full(len(probs), np.nan))
    sell_setup_quality_arr = pred.get('sell_setup_quality_score', np.full(len(probs), np.nan))
    buy_side_scores, sell_side_scores = _side_score_arrays(pred, probs)
    min_prob = _min_direction_probability(cfg)
    min_trade_prob = _min_trade_probability(cfg)
    min_edge_pips = _min_edge_pips(cfg)
    threshold_mode = _replay_threshold_mode(cfg)
    rolling_thresholds = threshold_mode in {'rolling_score_quantile', 'rolling_quantile', 'score_quantile'}
    rcfg = cfg.get('replay', {}) or {}
    buy_cfg = _side_cfg(cfg, 'buy')
    sell_cfg = _side_cfg(cfg, 'sell')
    default_lookback = int(rcfg.get('lookback_bars', rcfg.get('rolling_lookback_bars', 4000)) or 4000)
    default_min_history = int(rcfg.get('min_history_bars', max(200, min(default_lookback, 1000))) or 1)
    buy_thresholds = np.full(len(probs), min_trade_prob, dtype=float)
    sell_thresholds = np.full(len(probs), min_trade_prob, dtype=float)
    if rolling_thresholds and len(probs):
        buy_thresholds = _rolling_quantile_threshold(
            buy_side_scores,
            lookback=int(buy_cfg.get('lookback_bars', default_lookback) or default_lookback),
            quantile=float(buy_cfg.get('quantile', rcfg.get('buy_quantile', 0.985)) or 0.985),
            fallback=float(buy_cfg.get('fallback_threshold', rcfg.get('fallback_threshold', min_trade_prob)) or min_trade_prob),
            min_history=int(buy_cfg.get('min_history_bars', default_min_history) or default_min_history),
        )
        sell_thresholds = _rolling_quantile_threshold(
            sell_side_scores,
            lookback=int(sell_cfg.get('lookback_bars', default_lookback) or default_lookback),
            quantile=float(sell_cfg.get('quantile', rcfg.get('sell_quantile', 0.990)) or 0.990),
            fallback=float(sell_cfg.get('fallback_threshold', rcfg.get('fallback_threshold', min_trade_prob)) or min_trade_prob),
            min_history=int(sell_cfg.get('min_history_bars', default_min_history) or default_min_history),
        )
    min_gap_bars = int(rcfg.get('min_gap_bars_between_same_side_trades', rcfg.get('min_gap_bars', 0)) or 0)
    decision_parameters = resolve_replay_decision_parameters(
        cfg,
        train_side=(cfg.get('training') or {}).get('train_side'),
        eval_start=eval_start,
        eval_end=eval_end,
        symbol=symbol,
    )
    last_trade_row_by_side: dict[str, int] = {'BUY': -10**12, 'SELL': -10**12}

    decisions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    block_counts: dict[str, int] = {}
    passes_model = 0
    passes_external = 0

    for i, row_idx in enumerate(arr.row_indices):
        row = df.iloc[int(row_idx)]
        p_sell, p_no, p_buy = [float(x) for x in probs[i]]
        buy_side_score = float(buy_side_scores[i])
        sell_side_score = float(sell_side_scores[i])
        buy_threshold = float(buy_thresholds[i])
        sell_threshold = float(sell_thresholds[i])
        if rolling_thresholds:
            buy_pass = _allow_side(cfg, 'buy') and buy_side_score >= buy_threshold
            sell_pass = _allow_side(cfg, 'sell') and sell_side_score >= sell_threshold
            if buy_pass and sell_pass:
                pred_side = 'BUY' if (buy_side_score - buy_threshold) >= (sell_side_score - sell_threshold) else 'SELL'
            elif buy_pass:
                pred_side = 'BUY'
            elif sell_pass:
                pred_side = 'SELL'
            else:
                pred_side = 'NO_TRADE'
            pred_class = {'SELL': 0, 'NO_TRADE': 1, 'BUY': 2}[pred_side]
            selected_prob = buy_side_score if pred_side == 'BUY' else sell_side_score if pred_side == 'SELL' else max(buy_side_score, sell_side_score)
            trade_probability = selected_prob if pred_side != 'NO_TRADE' else max(buy_side_score, sell_side_score)
        else:
            pred_class = int(np.argmax(probs[i]))
            pred_side = DIRECTION_CLASS_NAMES[pred_class]
            selected_prob = float(np.max(probs[i]))
            trade_probability = float(trade_probs[i]) if len(trade_probs) else float(p_sell + p_buy)
        side_sell_probability = float(side_sell_probs[i]) if len(side_sell_probs) else None
        side_buy_probability = float(side_buy_probs[i]) if len(side_buy_probs) else None
        buy_edge_pips = float(buy_edge_pips_arr[i]) if len(buy_edge_pips_arr) else float('nan')
        sell_edge_pips = float(sell_edge_pips_arr[i]) if len(sell_edge_pips_arr) else float('nan')
        buy_setup_probability = float(buy_setup_prob_arr[i]) if len(buy_setup_prob_arr) else float('nan')
        sell_setup_probability = float(sell_setup_prob_arr[i]) if len(sell_setup_prob_arr) else float('nan')
        buy_setup_quality = float(buy_setup_quality_arr[i]) if len(buy_setup_quality_arr) else float('nan')
        sell_setup_quality = float(sell_setup_quality_arr[i]) if len(sell_setup_quality_arr) else float('nan')
        selected_edge_pips = buy_edge_pips if pred_side == 'BUY' else sell_edge_pips if pred_side == 'SELL' else float('nan')
        decision = 'BLOCK'
        reason = ''
        gate_diag: dict[str, Any] = {}
        if pred_side == 'NO_TRADE':
            reason = 'side_score_below_rolling_quantile' if rolling_thresholds else 'model_no_trade'
        elif not _allow_side(cfg, pred_side):
            reason = f'{pred_side.lower()}_disabled'
        elif not rolling_thresholds and trade_probability < min_trade_prob:
            reason = 'trade_probability_low'
        elif not rolling_thresholds and selected_prob < min_prob:
            reason = 'direction_probability_low'
        elif min_edge_pips is not None and np.isfinite(selected_edge_pips) and selected_edge_pips < float(min_edge_pips):
            reason = 'edge_pips_low'
        elif min_gap_bars > 0 and int(row_idx) - int(last_trade_row_by_side.get(pred_side, -10**12)) < min_gap_bars:
            reason = 'same_side_min_gap'
        else:
            passes_model += 1
            gate = external_trade_gate(symbol, pred_side, row, cfg)
            gate_diag = gate.diagnostics
            if gate.allow:
                decision = 'ALLOW'
                reason = 'ok'
                passes_external += 1
                try:
                    trade = simulate_trade_from_row(df, int(row_idx), symbol, pred_side, cfg)
                    trade.update({
                        'symbol': symbol,
                        'row_index': int(row_idx),
                        'model_probability': selected_prob,
                        'sell_probability': p_sell,
                        'no_trade_probability': p_no,
                        'buy_probability': p_buy,
                        'trade_probability': trade_probability,
                        'side_sell_probability': side_sell_probability,
                        'side_buy_probability': side_buy_probability,
                        'buy_edge_pips': buy_edge_pips,
                        'sell_edge_pips': sell_edge_pips,
                        'selected_edge_pips': selected_edge_pips,
                        'buy_setup_probability': buy_setup_probability,
                        'sell_setup_probability': sell_setup_probability,
                        'buy_setup_quality_score': buy_setup_quality,
                        'sell_setup_quality_score': sell_setup_quality,
                        'buy_score_threshold': buy_threshold,
                        'sell_score_threshold': sell_threshold,
                        'threshold_mode': threshold_mode,
                        'min_trade_probability': min_trade_prob,
                        'min_direction_probability': min_prob,
                        'min_edge_pips': min_edge_pips,
                        'min_gap_bars_between_same_side_trades': min_gap_bars,
                        'allow_buy': decision_parameters.get('allow_buy'),
                        'allow_sell': decision_parameters.get('allow_sell'),
                        'buy_threshold_lookback_bars': (decision_parameters.get('buy') or {}).get('lookback_bars'),
                        'buy_threshold_quantile': (decision_parameters.get('buy') or {}).get('quantile'),
                        'buy_threshold_min_history_bars': (decision_parameters.get('buy') or {}).get('min_history_bars'),
                        'sell_threshold_lookback_bars': (decision_parameters.get('sell') or {}).get('lookback_bars'),
                        'sell_threshold_quantile': (decision_parameters.get('sell') or {}).get('quantile'),
                        'sell_threshold_min_history_bars': (decision_parameters.get('sell') or {}).get('min_history_bars'),
                    })
                    trades.append(trade)
                    last_trade_row_by_side[pred_side] = int(row_idx)
                except Exception as exc:
                    decision = 'BLOCK'
                    reason = f'simulation_error:{exc}'
            else:
                reason = '|'.join(gate.reasons) if gate.reasons else 'external_gate_blocked'
        if decision != 'ALLOW':
            for part in str(reason).split('|'):
                block_counts[part] = block_counts.get(part, 0) + 1
        true_class = int(pd.to_numeric(pd.Series([row.get('direction_target', 1)]), errors='coerce').fillna(1).iloc[0])
        decisions.append({
            'time_utc': str(row.get('time_utc', '')),
            'symbol': symbol,
            'row_index': int(row_idx),
            'decision': decision,
            'reason': reason,
            'predicted_direction': pred_side,
            'true_direction': DIRECTION_CLASS_NAMES.get(true_class, str(true_class)),
            'selected_probability': selected_prob,
            'sell_probability': p_sell,
            'no_trade_probability': p_no,
            'buy_probability': p_buy,
            'trade_probability': trade_probability,
            'side_sell_probability': side_sell_probability,
            'side_buy_probability': side_buy_probability,
            'buy_edge_pips': buy_edge_pips,
            'sell_edge_pips': sell_edge_pips,
            'selected_edge_pips': selected_edge_pips,
            'buy_setup_probability': buy_setup_probability,
            'sell_setup_probability': sell_setup_probability,
            'buy_setup_quality_score': buy_setup_quality,
            'sell_setup_quality_score': sell_setup_quality,
            'buy_side_score': buy_side_score,
            'sell_side_score': sell_side_score,
            'buy_score_threshold': buy_threshold,
            'sell_score_threshold': sell_threshold,
            'threshold_mode': threshold_mode,
            'min_trade_probability': min_trade_prob,
            'min_direction_probability': min_prob,
            'min_edge_pips': min_edge_pips,
            'min_gap_bars_between_same_side_trades': min_gap_bars,
            'allow_buy': decision_parameters.get('allow_buy'),
            'allow_sell': decision_parameters.get('allow_sell'),
            'buy_threshold_lookback_bars': (decision_parameters.get('buy') or {}).get('lookback_bars'),
            'buy_threshold_quantile': (decision_parameters.get('buy') or {}).get('quantile'),
            'buy_threshold_min_history_bars': (decision_parameters.get('buy') or {}).get('min_history_bars'),
            'sell_threshold_lookback_bars': (decision_parameters.get('sell') or {}).get('lookback_bars'),
            'sell_threshold_quantile': (decision_parameters.get('sell') or {}).get('quantile'),
            'sell_threshold_min_history_bars': (decision_parameters.get('sell') or {}).get('min_history_bars'),
            'spread_points': row.get('spread_points', None),
            'analytics_gate': gate_diag,
        })

    trade_summary = summarise_trades(trades)
    replay_score = _replay_score(trade_summary, cfg)
    y_true = arr.y_direction.astype(int)
    y_pred = probs.argmax(axis=1).astype(int)
    raw_accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    model_details = getattr(model, 'model_details', {}) or {}
    base_model_type = str(model_details.get('model_type', getattr(model, 'architecture', 'direction_policy')))
    if bool(model_details.get('use_side_setup_heads', False)):
        replay_model_type = f'{base_model_type}_side_setup_ranking'
    elif bool(getattr(model, 'is_hierarchical', False)):
        replay_model_type = base_model_type
    else:
        replay_model_type = 'direction_buy_sell_no_trade'
    summary = {
        'symbol': symbol,
        'model_type': replay_model_type,
        'architecture': str(getattr(model, 'architecture', 'unknown')),
        'model_details': model_details,
        'class_mapping': DIRECTION_CLASS_NAMES,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'features_path': str(features_path),
        'eval_start': eval_start,
        'eval_end': eval_end,
        'data': data_info,
        'sequence_rows': int(len(arr.X_seq)),
        'raw_direction_accuracy': raw_accuracy,
        'min_direction_probability': min_prob,
        'min_trade_probability': min_trade_prob,
        'min_edge_pips': min_edge_pips,
        'threshold_mode': threshold_mode,
        'rolling_thresholds_used': bool(rolling_thresholds),
        'allow_buy': bool(_allow_side(cfg, 'buy')),
        'allow_sell': bool(_allow_side(cfg, 'sell')),
        'decision_parameters': decision_parameters,
        'deployment_decision_parameters': decision_parameters,
        'config_snapshot': config_snapshot(cfg, config_path=cfg.get('_config_path'), base_config_path=cfg.get('_base_config_path'), include_resolved_sections=False),
        'passes_model_gate': int(passes_model),
        'passes_external_gate': int(passes_external),
        'block_counts': block_counts,
        **trade_summary,
        'replay_score': float(replay_score),
    }

    if output_prefix:
        prefix = Path(output_prefix)
    else:
        prefix = Path('logs') / f'{symbol}_{_timeframe(cfg)}_direction_replay'
    summary_path = Path(str(prefix)[:-5] + '_summary.json') if str(prefix).endswith('.json') else Path(str(prefix) + '_summary.json')
    decisions_path = Path(str(prefix) + '_decisions.csv')
    trades_path = Path(str(prefix) + '_trades.csv')
    ensure_dir(summary_path.parent)
    write_json(summary_path, _json_safe(summary))
    _write_csv(decisions_path, decisions)
    _write_csv(trades_path, trades)
    summary['summary_path'] = str(summary_path)
    summary['decisions_path'] = str(decisions_path)
    summary['trades_path'] = str(trades_path)
    if verbose:
        print(json.dumps(_json_safe(summary), indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description='Replay a saved BUY/SELL/NO_TRADE direction policy with external analytics gate. Architecture is restored from the checkpoint/config')
    p.add_argument('--config', default='config/direction_settings_generic_multisymbol_31_symbols.yaml')
    p.add_argument('--symbol', default=None)
    p.add_argument('--symbols', nargs='+', default=None)
    p.add_argument('--model-path', default=None)
    p.add_argument('--scaler-path', default=None)
    p.add_argument('--features-path', default=None)
    p.add_argument('--eval-start', default=None)
    p.add_argument('--eval-end', default=None)
    p.add_argument('--output-prefix', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    cfg = load_config_with_optional_spread_risk(args.config)
    cfg['_config_path'] = str(args.config)
    cfg.setdefault('_base_config_path', str(args.config))
    symbols = args.symbols or ([args.symbol] if args.symbol else ((cfg.get('trading') or {}).get('symbols') or ['US500']))
    symbols = validate_forex_symbols(symbols)
    summaries = []
    for symbol in symbols:
        prefix = args.output_prefix
        if prefix and len(symbols) > 1:
            prefix = str(Path(prefix) / symbol / f'{symbol}_{_timeframe(cfg)}_direction_replay')
        summaries.append(replay_symbol(
            symbol,
            cfg,
            model_path=Path(args.model_path) if args.model_path and len(symbols) == 1 else None,
            scaler_path=Path(args.scaler_path) if args.scaler_path and len(symbols) == 1 else None,
            features_path=Path(args.features_path) if args.features_path and len(symbols) == 1 else None,
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            output_prefix=prefix,
            device=args.device,
            verbose=args.verbose,
        ))
    print(json.dumps(_json_safe({'summaries': summaries}), indent=2))


if __name__ == '__main__':
    main()
