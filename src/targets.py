from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .forex import pips_from_price_delta, price_delta_from_pips, spread_points_to_pips, symbol_cfg_value
from .spread_risk_config import symbol_default_spread_points, symbol_max_spread_pips

# Legacy names are kept only for diagnostics/backwards-compatible CSV columns.
DIRECTION_NAMES = {0: 'SELL', 1: 'NO_TRADE', 2: 'BUY'}
DECISION_NAMES = {0: 'BLOCK', 1: 'ALLOW'}
OUTCOME_NAMES = {0: 'SL', 1: 'TIME_EXIT', 2: 'TP'}
SIDE_NAMES = ('BUY', 'SELL')


@dataclass
class TradeOutcome:
    pips: float
    outcome: int  # 0=SL, 1=TIME_EXIT, 2=TP
    bars_to_outcome: int
    ambiguous: bool = False


def _label_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get('labels', {}) or {}


def _spread_cost_pips(symbol: str, spread_points: float, cfg: dict[str, Any]) -> float:
    return spread_points_to_pips(symbol, float(spread_points), cfg)


def _spread_delta(symbol: str, spread_points: float, cfg: dict[str, Any]) -> float:
    return price_delta_from_pips(symbol, _spread_cost_pips(symbol, spread_points, cfg), cfg)


def _same_bar_policy(cfg: dict[str, Any]) -> str:
    lcfg = _label_cfg(cfg)
    policy = str(lcfg.get('same_bar_tp_sl_policy', '') or '').strip().lower()
    if policy in {'stop_first', 'sl_first', 'conservative'}:
        return 'stop_first'
    if policy in {'take_profit_first', 'tp_first', 'optimistic'}:
        return 'take_profit_first'
    if policy in {'discard', 'time_exit'}:
        return 'discard'
    # Backwards-compatible behaviour.
    return 'stop_first' if bool(lcfg.get('conservative_same_bar_hits', True)) else 'take_profit_first'


def _resolve_same_bar(tp_pips: float, sl_pips: float, bars_to_outcome: int, cfg: dict[str, Any]) -> TradeOutcome:
    policy = _same_bar_policy(cfg)
    if policy == 'take_profit_first':
        return TradeOutcome(tp_pips, 2, bars_to_outcome, ambiguous=True)
    if policy == 'discard':
        return TradeOutcome(0.0, 1, bars_to_outcome, ambiguous=True)
    return TradeOutcome(-sl_pips, 0, bars_to_outcome, ambiguous=True)


def _first_hit_outcome(
    symbol: str,
    entry: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    side: str,
    cfg: dict[str, Any],
) -> TradeOutcome:
    """Legacy close/high/low barrier simulation.

    This path is preserved for backwards compatibility when
    labels.use_live_bid_ask_simulation is false or missing.
    """
    lcfg = _label_cfg(cfg)
    tp_pips = float(symbol_cfg_value(cfg, 'labels', 'take_profit_pips', symbol, 7.0))
    sl_pips = float(symbol_cfg_value(cfg, 'labels', 'stop_loss_pips', symbol, 5.0))
    side = str(side).upper()

    if side == 'BUY':
        tp = entry + price_delta_from_pips(symbol, tp_pips, cfg)
        sl = entry - price_delta_from_pips(symbol, sl_pips, cfg)
        for j, (hi, lo) in enumerate(zip(highs, lows), start=1):
            hit_tp = hi >= tp
            hit_sl = lo <= sl
            if hit_tp and hit_sl:
                return _resolve_same_bar(tp_pips, sl_pips, j, cfg)
            if hit_tp:
                return TradeOutcome(tp_pips, 2, j)
            if hit_sl:
                return TradeOutcome(-sl_pips, 0, j)
        end = closes[-1] if len(closes) else entry
        return TradeOutcome(pips_from_price_delta(symbol, end - entry, cfg), 1, len(closes))

    if side == 'SELL':
        tp = entry - price_delta_from_pips(symbol, tp_pips, cfg)
        sl = entry + price_delta_from_pips(symbol, sl_pips, cfg)
        for j, (hi, lo) in enumerate(zip(highs, lows), start=1):
            hit_tp = lo <= tp
            hit_sl = hi >= sl
            if hit_tp and hit_sl:
                return _resolve_same_bar(tp_pips, sl_pips, j, cfg)
            if hit_tp:
                return TradeOutcome(tp_pips, 2, j)
            if hit_sl:
                return TradeOutcome(-sl_pips, 0, j)
        end = closes[-1] if len(closes) else entry
        return TradeOutcome(pips_from_price_delta(symbol, entry - end, cfg), 1, len(closes))

    raise ValueError(f'Unsupported side {side!r}; expected BUY or SELL')


def _first_hit_outcome_live_bidask(
    symbol: str,
    bid_entry: float,
    bid_highs: np.ndarray,
    bid_lows: np.ndarray,
    bid_closes: np.ndarray,
    spread_points: np.ndarray,
    side: str,
    cfg: dict[str, Any],
    *,
    entry_spread_points: float,
) -> TradeOutcome:
    """Live-style barrier simulation using bid OHLC plus spread.

    Assumption: MT5/rates OHLC are bid prices.

    BUY:
      - opens at ask = bid + spread + slippage
      - TP/SL are triggered by future bid high/low

    SELL:
      - opens at bid - slippage
      - TP/SL are triggered by future ask low/high = future bid + future spread
    """
    lcfg = _label_cfg(cfg)
    tp_pips = float(symbol_cfg_value(cfg, 'labels', 'take_profit_pips', symbol, 7.0))
    sl_pips = float(symbol_cfg_value(cfg, 'labels', 'stop_loss_pips', symbol, 5.0))
    slippage_pips = float(symbol_cfg_value(cfg, 'labels', 'slippage_pips', symbol, 0.0) or 0.0)
    slippage_delta = price_delta_from_pips(symbol, slippage_pips, cfg)
    side = str(side).upper()

    if len(bid_closes) == 0:
        return TradeOutcome(0.0, 1, 0)

    future_spreads = np.asarray(spread_points, dtype=float)
    if len(future_spreads) < len(bid_closes):
        pad = np.full(len(bid_closes) - len(future_spreads), float(entry_spread_points), dtype=float)
        future_spreads = np.concatenate([future_spreads, pad])

    if side == 'BUY':
        ask_entry = bid_entry + _spread_delta(symbol, entry_spread_points, cfg) + slippage_delta
        tp = ask_entry + price_delta_from_pips(symbol, tp_pips, cfg)
        sl = ask_entry - price_delta_from_pips(symbol, sl_pips, cfg)
        for j, (bid_hi, bid_lo) in enumerate(zip(bid_highs, bid_lows), start=1):
            hit_tp = bid_hi >= tp
            hit_sl = bid_lo <= sl
            if hit_tp and hit_sl:
                return _resolve_same_bar(tp_pips, sl_pips, j, cfg)
            if hit_tp:
                return TradeOutcome(tp_pips, 2, j)
            if hit_sl:
                return TradeOutcome(-sl_pips, 0, j)
        exit_bid = bid_closes[-1]
        return TradeOutcome(pips_from_price_delta(symbol, exit_bid - ask_entry, cfg), 1, len(bid_closes))

    if side == 'SELL':
        sell_entry = bid_entry - slippage_delta
        tp = sell_entry - price_delta_from_pips(symbol, tp_pips, cfg)
        sl = sell_entry + price_delta_from_pips(symbol, sl_pips, cfg)
        for j, (bid_hi, bid_lo, sp) in enumerate(zip(bid_highs, bid_lows, future_spreads), start=1):
            ask_hi = bid_hi + _spread_delta(symbol, sp, cfg)
            ask_lo = bid_lo + _spread_delta(symbol, sp, cfg)
            hit_tp = ask_lo <= tp
            hit_sl = ask_hi >= sl
            if hit_tp and hit_sl:
                return _resolve_same_bar(tp_pips, sl_pips, j, cfg)
            if hit_tp:
                return TradeOutcome(tp_pips, 2, j)
            if hit_sl:
                return TradeOutcome(-sl_pips, 0, j)
        exit_ask = bid_closes[-1] + _spread_delta(symbol, future_spreads[min(len(bid_closes) - 1, len(future_spreads) - 1)], cfg)
        return TradeOutcome(pips_from_price_delta(symbol, sell_entry - exit_ask, cfg), 1, len(bid_closes))

    raise ValueError(f'Unsupported side {side!r}; expected BUY or SELL')


def _required_ohlc_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    lower = {str(c).lower(): c for c in df.columns}
    close = lower.get('close') or lower.get('close_price') or lower.get('bid_close')
    high = lower.get('high') or lower.get('high_price') or lower.get('bid_high')
    low = lower.get('low') or lower.get('low_price') or lower.get('bid_low')
    open_ = lower.get('open') or lower.get('open_price') or lower.get('bid_open') or close
    if not close or not high or not low:
        raise ValueError('Processed CSV must contain close/high/low columns to generate BUY/SELL outcome targets.')
    return open_, high, low, close



def _positive_filter_cfg(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {'enabled': bool(value)}


def _discard_positive_rows(out: pd.DataFrame, indices: list[int], *, mode: str, reason: str) -> int:
    if not indices:
        return 0
    mode_norm = str(mode or 'no_trade').strip().lower()
    if mode_norm in {'ignore', 'ignored', 'exclude', 'skip'}:
        replacement = -1
    elif mode_norm in {'drop'}:
        # Keep row positions stable for sequence generation. A physical row drop
        # can create artificial sequence jumps, so treat drop as ignore.
        replacement = -1
    else:
        replacement = 1
    out.loc[indices, 'direction_target'] = int(replacement)
    # Keep side-specific setup targets consistent with positive deduplication /
    # daily caps. Rows discarded as IGNORE should not continue to train the BUY
    # or SELL setup heads as positives. Rows converted to NO_TRADE become
    # negatives for whichever side they were previously positive for.
    for side_col in ('buy_setup_target', 'sell_setup_target'):
        if side_col in out.columns:
            current = pd.to_numeric(out.loc[indices, side_col], errors='coerce').fillna(-1).astype(int)
            pos_idx = list(current[current == 1].index)
            if pos_idx:
                out.loc[pos_idx, side_col] = int(replacement if replacement < 0 else 0)
    if 'label_filter_status' in out.columns:
        out.loc[indices, 'label_filter_status'] = reason
    return int(len(indices))


def _side_strength_column(side: str) -> str:
    return f'{str(side).lower()}_candidate_strength_score'


def _best_index_by_strength(out: pd.DataFrame, indices: list[int], side: str) -> int:
    if len(indices) == 1:
        return int(indices[0])
    col = _side_strength_column(side)
    if col not in out.columns:
        return int(indices[0])
    values = pd.to_numeric(out.loc[indices, col], errors='coerce').fillna(-1.0e18)
    return int(values.idxmax())


def _apply_positive_deduplication(out: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only one strong positive label from each nearby setup cluster.

    A single market setup often creates many consecutive positive rows. Keeping
    all of them makes the side models learn broad regions rather than the
    strongest, most identifiable entry point. This filter keeps the strongest
    row in each side-specific cluster and turns the other positive rows into
    NO_TRADE or IGNORE according to discarded_positive_mode.
    """
    lcfg = _label_cfg(cfg)
    dcfg = _positive_filter_cfg(lcfg.get('positive_label_deduplication', {}))
    enabled = bool(dcfg.get('enabled', False))
    min_gap_bars = int(dcfg.get('min_gap_bars', 0) or 0)
    mode = str(dcfg.get('discarded_positive_mode', lcfg.get('discarded_positive_mode', 'no_trade')) or 'no_trade')
    info: dict[str, Any] = {
        'enabled': enabled,
        'mode': str(dcfg.get('mode', 'best_per_cluster')),
        'min_gap_bars': int(min_gap_bars),
        'discarded_positive_mode': mode,
        'buy_removed': 0,
        'sell_removed': 0,
        'buy_clusters': 0,
        'sell_clusters': 0,
    }
    if not enabled or min_gap_bars <= 0 or 'direction_target' not in out.columns:
        return out, info

    result = out.copy()
    cluster_days = None
    if 'time_utc' in result.columns:
        parsed_times = pd.to_datetime(result['time_utc'], utc=True, errors='coerce')
        if parsed_times.notna().any():
            cluster_days = parsed_times.dt.floor('D')

    for side, direction_idx in (('buy', 2), ('sell', 0)):
        positive_indices = [int(x) for x in result.index[pd.to_numeric(result['direction_target'], errors='coerce') == direction_idx].tolist()]
        if not positive_indices:
            continue
        clusters: list[list[int]] = []
        current = [positive_indices[0]]
        for idx in positive_indices[1:]:
            same_day = True
            if cluster_days is not None:
                same_day = bool(cluster_days.iloc[idx] == cluster_days.iloc[current[-1]])
            if same_day and idx - current[-1] <= min_gap_bars:
                current.append(idx)
            else:
                clusters.append(current)
                current = [idx]
        clusters.append(current)
        info[f'{side}_clusters'] = int(len(clusters))

        discard: list[int] = []
        for cluster in clusters:
            keep = _best_index_by_strength(result, cluster, side)
            discard.extend([int(x) for x in cluster if int(x) != keep])
        removed = _discard_positive_rows(result, discard, mode=mode, reason=f'{side}_deduplicated')
        info[f'{side}_removed'] = int(removed)
    return result, info


def _apply_daily_positive_cap(out: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limit retained positive labels to the strongest K setups per day.

    The cap is side-specific. It runs after de-duplication so a day with many
    overlapping BUY labels does not dominate the training set. Discarded labels
    can be converted to NO_TRADE or IGNORE.
    """
    lcfg = _label_cfg(cfg)
    ccfg = _positive_filter_cfg(lcfg.get('max_positive_setups_per_day', {}))
    enabled = bool(ccfg.get('enabled', False))
    mode = str(ccfg.get('discarded_positive_mode', lcfg.get('discarded_positive_mode', 'no_trade')) or 'no_trade')
    info: dict[str, Any] = {
        'enabled': enabled,
        'buy_cap': ccfg.get('buy', ccfg.get('per_side', ccfg.get('total'))),
        'sell_cap': ccfg.get('sell', ccfg.get('per_side', ccfg.get('total'))),
        'discarded_positive_mode': mode,
        'buy_removed': 0,
        'sell_removed': 0,
        'days_seen': 0,
        'warning': None,
    }
    if not enabled or 'direction_target' not in out.columns:
        return out, info
    if 'time_utc' not in out.columns:
        info['warning'] = 'time_utc missing; daily positive cap skipped'
        return out, info

    result = out.copy()
    times = pd.to_datetime(result['time_utc'], utc=True, errors='coerce')
    valid_days = times.dt.floor('D')
    if valid_days.isna().all():
        info['warning'] = 'time_utc could not be parsed; daily positive cap skipped'
        return result, info
    result['_label_day_utc'] = valid_days
    info['days_seen'] = int(result['_label_day_utc'].dropna().nunique())

    for side, direction_idx, cap_key in (('buy', 2, 'buy_cap'), ('sell', 0, 'sell_cap')):
        raw_cap = info.get(cap_key)
        if raw_cap in (None, '', 0, '0'):
            continue
        cap = int(raw_cap)
        if cap <= 0:
            continue
        strength_col = _side_strength_column(side)
        discard: list[int] = []
        side_mask = pd.to_numeric(result['direction_target'], errors='coerce') == direction_idx
        for _, group in result.loc[side_mask & result['_label_day_utc'].notna()].groupby('_label_day_utc', sort=False):
            if len(group) <= cap:
                continue
            if strength_col in group.columns:
                scores = pd.to_numeric(group[strength_col], errors='coerce').fillna(-1.0e18)
                keep = set(scores.nlargest(cap).index.astype(int).tolist())
            else:
                keep = set(group.index[:cap].astype(int).tolist())
            discard.extend([int(i) for i in group.index if int(i) not in keep])
        removed = _discard_positive_rows(result, discard, mode=mode, reason=f'{side}_daily_cap')
        info[f'{side}_removed'] = int(removed)

    result = result.drop(columns=['_label_day_utc'], errors='ignore')
    return result, info


def _candidate_strength(side_net: float, other_net: float, bars_to_outcome: int, horizon: int) -> float:
    """Rank clean candidates by strength, edge and speed.

    This score is only used for label filtering/ranking and is dropped before
    training data is saved, so it cannot leak future information into the model.
    """
    if not np.isfinite(side_net):
        return float('-inf')
    edge = max(0.0, float(side_net) - float(other_net)) if np.isfinite(other_net) else 0.0
    speed_bonus = 0.0
    if horizon > 0 and bars_to_outcome > 0:
        speed_bonus = max(0.0, float(horizon - bars_to_outcome + 1) / float(horizon))
    return float(side_net + edge + 0.25 * speed_bonus)


def _cfg_get_path(cfg: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur.get(key)
    return cur


def _cfg_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ''):
        return default
    return float(value)




def _symbol_map_value(mapping: Any, symbol: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    symbol_u = str(symbol).upper()
    for key in (symbol, symbol_u, symbol_u.lower()):
        if key in mapping and mapping[key] not in (None, ''):
            return mapping[key]
    return None


def _cfg_float_symbol(section: dict[str, Any], key: str, symbol: str, default: float | None = None) -> float | None:
    value = _symbol_map_value(section.get(f'{key}_by_symbol'), symbol)
    if value is None:
        value = _symbol_map_value(section.get(f'{key}s_by_symbol'), symbol)
    if value is None:
        value = section.get(key, default)
    return _cfg_float(value, default)

def _cfg_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ''):
        return default
    return int(value)


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _strong_setup_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    lcfg = _label_cfg(cfg)
    scfg = lcfg.get('strong_setup') or lcfg.get('strong_setup_labels') or {}
    return scfg if isinstance(scfg, dict) else {}


def _label_method(cfg: dict[str, Any]) -> str:
    lcfg = _label_cfg(cfg)
    method = lcfg.get('method', lcfg.get('label_method', lcfg.get('target_generation_mode', 'barrier_direction')))
    return str(method or 'barrier_direction').strip().lower()


def _is_strong_setup_method(cfg: dict[str, Any]) -> bool:
    method = _label_method(cfg)
    return method in {
        'strong_setup_v1',
        'clean_margin_v1',
        'event_rank_v1',
        'side_specific_event_rank_v1',
    }


def _side_cfg(scfg: dict[str, Any], side: str) -> dict[str, Any]:
    side = side.lower()
    pcfg = scfg.get('positive') or {}
    if isinstance(pcfg, dict) and isinstance(pcfg.get(side), dict):
        merged = dict(scfg.get('positive_defaults') or {})
        merged.update(pcfg.get(side) or {})
        return merged
    return dict(scfg.get(side) or {})


def _signal_count(out: pd.DataFrame, side: str) -> pd.Series:
    side = side.lower()
    col = f'sig_{side}_signal_count'
    if col in out.columns:
        return pd.to_numeric(out[col], errors='coerce').fillna(0.0)
    cols = [c for c in out.columns if str(c).startswith('sig_') and str(c).endswith(f'_{side}')]
    if not cols:
        return pd.Series(0.0, index=out.index, dtype='float64')
    return out[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1)


def _numeric_column_or_zero(out: pd.DataFrame, column: str) -> pd.Series:
    if column in out.columns:
        return pd.to_numeric(out[column], errors='coerce').fillna(0.0)
    return pd.Series(0.0, index=out.index, dtype='float64')


def _analytic_setup_score(out: pd.DataFrame, side: str) -> pd.Series:
    """Causal analytic setup score used for label selection only.

    The score is intentionally broad. It rewards same-side analytic votes and
    trend/momentum alignment and penalises opposing votes/conflict. It is not a
    future-derived value and may also be kept as a live feature via the existing
    sig_* columns.
    """
    side = side.lower()
    buy_count = _signal_count(out, 'buy')
    sell_count = _signal_count(out, 'sell')
    adx = _numeric_column_or_zero(out, 'sig_adx_strength')
    trend1 = _numeric_column_or_zero(out, 'sig_ema_fast_minus_mid_atr')
    trend2 = _numeric_column_or_zero(out, 'sig_ema_mid_minus_slow_atr')
    slope1 = _numeric_column_or_zero(out, 'sig_ema_fast_slope_atr')
    slope2 = _numeric_column_or_zero(out, 'sig_ema_mid_slope_atr')
    macd = _numeric_column_or_zero(out, 'sig_macd_hist_atr')
    macd_slope = _numeric_column_or_zero(out, 'sig_macd_hist_slope_atr')
    rsi = _numeric_column_or_zero(out, 'sig_rsi_centered')
    conflict = _numeric_column_or_zero(out, 'sig_signal_conflict')

    if side == 'buy':
        same = buy_count
        opp = sell_count
        directional = (
            trend1.clip(lower=0) + trend2.clip(lower=0)
            + 0.75 * slope1.clip(lower=0) + 0.5 * slope2.clip(lower=0)
            + 0.75 * macd.clip(lower=0) + 0.5 * macd_slope.clip(lower=0)
            + 0.25 * rsi.clip(lower=0)
        )
    else:
        same = sell_count
        opp = buy_count
        directional = (
            (-trend1).clip(lower=0) + (-trend2).clip(lower=0)
            + 0.75 * (-slope1).clip(lower=0) + 0.5 * (-slope2).clip(lower=0)
            + 0.75 * (-macd).clip(lower=0) + 0.5 * (-macd_slope).clip(lower=0)
            + 0.25 * (-rsi).clip(lower=0)
        )
    score = 2.0 * same - 1.5 * opp + 2.0 * directional + 1.0 * adx.clip(lower=0) - 1.0 * conflict
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _mfe_mae_live_bidask(
    symbol: str,
    bid_entry: float,
    bid_highs: np.ndarray,
    bid_lows: np.ndarray,
    spread_points: np.ndarray,
    side: str,
    cfg: dict[str, Any],
    *,
    entry_spread_points: float,
) -> tuple[float, float]:
    lcfg = _label_cfg(cfg)
    slippage_pips = float(symbol_cfg_value(cfg, 'labels', 'slippage_pips', symbol, 0.0) or 0.0)
    slippage_delta = price_delta_from_pips(symbol, slippage_pips, cfg)
    side = side.upper()
    if len(bid_highs) == 0 or len(bid_lows) == 0:
        return np.nan, np.nan
    future_spreads = np.asarray(spread_points, dtype=float)
    if len(future_spreads) < len(bid_highs):
        pad = np.full(len(bid_highs) - len(future_spreads), float(entry_spread_points), dtype=float)
        future_spreads = np.concatenate([future_spreads, pad])
    if side == 'BUY':
        ask_entry = bid_entry + _spread_delta(symbol, entry_spread_points, cfg) + slippage_delta
        mfe = pips_from_price_delta(symbol, float(np.nanmax(bid_highs)) - ask_entry, cfg)
        mae = pips_from_price_delta(symbol, ask_entry - float(np.nanmin(bid_lows)), cfg)
        return float(max(0.0, mfe)), float(max(0.0, mae))
    sell_entry = bid_entry - slippage_delta
    ask_highs = np.asarray(bid_highs, dtype=float) + np.array([_spread_delta(symbol, sp, cfg) for sp in future_spreads[:len(bid_highs)]])
    ask_lows = np.asarray(bid_lows, dtype=float) + np.array([_spread_delta(symbol, sp, cfg) for sp in future_spreads[:len(bid_lows)]])
    mfe = pips_from_price_delta(symbol, sell_entry - float(np.nanmin(ask_lows)), cfg)
    mae = pips_from_price_delta(symbol, float(np.nanmax(ask_highs)) - sell_entry, cfg)
    return float(max(0.0, mfe)), float(max(0.0, mae))


def _mfe_mae_legacy(symbol: str, entry: float, highs: np.ndarray, lows: np.ndarray, side: str, cfg: dict[str, Any]) -> tuple[float, float]:
    side = side.upper()
    if len(highs) == 0 or len(lows) == 0:
        return np.nan, np.nan
    if side == 'BUY':
        mfe = pips_from_price_delta(symbol, float(np.nanmax(highs)) - entry, cfg)
        mae = pips_from_price_delta(symbol, entry - float(np.nanmin(lows)), cfg)
    else:
        mfe = pips_from_price_delta(symbol, entry - float(np.nanmin(lows)), cfg)
        mae = pips_from_price_delta(symbol, float(np.nanmax(highs)) - entry, cfg)
    return float(max(0.0, mfe)), float(max(0.0, mae))


def _choose_random_indices(indices: np.ndarray, n: int, seed: int) -> list[int]:
    if n <= 0 or len(indices) == 0:
        return []
    n = min(int(n), int(len(indices)))
    rng = np.random.default_rng(int(seed))
    return [int(x) for x in rng.choice(indices.astype(int), size=n, replace=False).tolist()]


def _apply_strong_setup_labels(out: pd.DataFrame, cfg: dict[str, Any], symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert raw TP/SL outcomes into side-specific strong setup labels.

    This intentionally differs from every-bar 3-class labelling. It keeps clean,
    high-margin BUY/SELL setup endpoints as positives and can keep only selected
    hard/background negatives as NO_TRADE endpoints while marking the rest IGNORE.
    """
    scfg = _strong_setup_cfg(cfg)
    if not _is_strong_setup_method(cfg):
        return out, {'enabled': False, 'method': _label_method(cfg)}

    result = out.copy()
    n = len(result)
    buy_cfg = _side_cfg(scfg, 'buy')
    sell_cfg = _side_cfg(scfg, 'sell')
    defaults = dict(scfg.get('positive_defaults') or {})

    for col in (
        'buy_candidate_mfe_pips', 'buy_candidate_mae_pips',
        'sell_candidate_mfe_pips', 'sell_candidate_mae_pips',
    ):
        if col not in result.columns:
            result[col] = np.nan

    buy_net = pd.to_numeric(result.get('buy_candidate_net_pips'), errors='coerce')
    sell_net = pd.to_numeric(result.get('sell_candidate_net_pips'), errors='coerce')
    buy_outcome = pd.to_numeric(result.get('buy_candidate_outcome'), errors='coerce').fillna(-1).astype(int)
    sell_outcome = pd.to_numeric(result.get('sell_candidate_outcome'), errors='coerce').fillna(-1).astype(int)
    buy_mfe = pd.to_numeric(result.get('buy_candidate_mfe_pips'), errors='coerce')
    sell_mfe = pd.to_numeric(result.get('sell_candidate_mfe_pips'), errors='coerce')
    buy_mae = pd.to_numeric(result.get('buy_candidate_mae_pips'), errors='coerce')
    sell_mae = pd.to_numeric(result.get('sell_candidate_mae_pips'), errors='coerce')

    buy_score = _analytic_setup_score(result, 'buy')
    sell_score = _analytic_setup_score(result, 'sell')
    result['buy_analytic_setup_score'] = buy_score
    result['sell_analytic_setup_score'] = sell_score

    def _side_positive_mask(side: str, cfg_side: dict[str, Any]) -> pd.Series:
        side = side.lower()
        net = buy_net if side == 'buy' else sell_net
        other_net = sell_net if side == 'buy' else buy_net
        outcome = buy_outcome if side == 'buy' else sell_outcome
        mfe = buy_mfe if side == 'buy' else sell_mfe
        mae = buy_mae if side == 'buy' else sell_mae
        score = buy_score if side == 'buy' else sell_score
        min_net = _cfg_float_symbol(cfg_side, 'min_net_pips', symbol, _cfg_float_symbol(_label_cfg(cfg), 'min_clean_win_net_pips', symbol, _cfg_float(defaults.get('min_net_pips'), 0.0)))
        min_mfe = _cfg_float_symbol(cfg_side, 'min_mfe_pips', symbol, _cfg_float(defaults.get('min_mfe_pips'), None))
        max_mae = _cfg_float_symbol(cfg_side, 'max_mae_pips', symbol, _cfg_float(defaults.get('max_mae_pips'), None))
        min_edge = _cfg_float_symbol(cfg_side, 'min_side_edge_pips', symbol, _cfg_float_symbol(_label_cfg(cfg), 'min_side_edge_pips', symbol, _cfg_float(defaults.get('min_side_edge_pips'), 0.0)))
        min_score = _cfg_float_symbol(cfg_side, 'min_analytic_score', symbol, _cfg_float(defaults.get('min_analytic_score'), None))
        require_tp = _cfg_bool(cfg_side.get('require_tp_before_sl', defaults.get('require_tp_before_sl')), True)
        allow_clean_without_analytic = _cfg_bool(cfg_side.get('allow_clean_without_analytic', defaults.get('allow_clean_without_analytic')), True)

        mask = pd.Series(True, index=result.index)
        if require_tp:
            mask &= outcome == 2
        if min_net is not None:
            mask &= net >= float(min_net)
        if min_mfe is not None:
            mask &= mfe >= float(min_mfe)
        if max_mae is not None:
            mask &= mae <= float(max_mae)
        if min_edge is not None:
            mask &= (net - other_net) >= float(min_edge)
        if min_score is not None:
            analytic_ok = score >= float(min_score)
            if allow_clean_without_analytic:
                clean_ok = (outcome == 2) & (net >= float(min_net if min_net is not None else 0.0))
                if min_mfe is not None:
                    clean_ok &= mfe >= float(min_mfe)
                if max_mae is not None:
                    clean_ok &= mae <= float(max_mae)
                mask &= analytic_ok | clean_ok
            else:
                mask &= analytic_ok
        return mask.fillna(False)

    buy_pos = _side_positive_mask('buy', buy_cfg)
    sell_pos = _side_positive_mask('sell', sell_cfg)

    # Resolve rare rows where both sides satisfy criteria.
    buy_quality = (
        buy_net.fillna(-1e9)
        + (buy_net - sell_net).fillna(0.0).clip(lower=0)
        + 0.25 * buy_mfe.fillna(0.0)
        - 0.75 * buy_mae.fillna(0.0)
        + 0.5 * buy_score.fillna(0.0)
    )
    sell_quality = (
        sell_net.fillna(-1e9)
        + (sell_net - buy_net).fillna(0.0).clip(lower=0)
        + 0.25 * sell_mfe.fillna(0.0)
        - 0.75 * sell_mae.fillna(0.0)
        + 0.5 * sell_score.fillna(0.0)
    )
    both = buy_pos & sell_pos
    buy_pos = buy_pos & (~both | (buy_quality >= sell_quality))
    sell_pos = sell_pos & (~both | (sell_quality > buy_quality))

    output_mode = str(scfg.get('output_mode', scfg.get('sequence_endpoint_mode', 'event_based')) or 'event_based').lower()
    event_based = output_mode in {'event', 'events', 'event_based', 'event_rank', 'selected_endpoints'}
    if event_based:
        direction = np.full(n, -1, dtype=np.int64)
    else:
        direction = np.full(n, 1, dtype=np.int64)
    direction[sell_pos.to_numpy(bool)] = 0
    direction[buy_pos.to_numpy(bool)] = 2

    positive_indices = np.flatnonzero((direction == 0) | (direction == 2))
    hard_cfg = scfg.get('hard_negatives') or {}
    bg_cfg = scfg.get('background_no_trade') or {}
    # Side-specific failed setup masks are used both for selecting hard negatives
    # and for the new side-specific setup heads. Initialise them here so the
    # non-event mode still has well-defined targets.
    buy_failed = pd.Series(False, index=result.index)
    sell_failed = pd.Series(False, index=result.index)
    selected_negatives: set[int] = set()
    if event_based:
        neg_mask = pd.Series(False, index=result.index)
        if _cfg_bool(hard_cfg.get('enabled'), True):
            buy_min = _cfg_float(hard_cfg.get('buy_analytic_score_min', hard_cfg.get('analytic_score_min')), 6.0)
            sell_min = _cfg_float(hard_cfg.get('sell_analytic_score_min', hard_cfg.get('analytic_score_min')), 6.0)
            buy_failed = (buy_score >= float(buy_min)) & ~buy_pos & ((buy_outcome != 2) | (buy_net <= 0) | (sell_net > buy_net))
            sell_failed = (sell_score >= float(sell_min)) & ~sell_pos & ((sell_outcome != 2) | (sell_net <= 0) | (buy_net > sell_net))
            neg_mask |= buy_failed.fillna(False) | sell_failed.fillna(False)
            # Keep near-positive failures, because they are the most useful hard negatives.
            near_mult = _cfg_float(hard_cfg.get('near_positive_mfe_fraction'), 0.75)
            min_buy_mfe = _cfg_float_symbol(buy_cfg, 'min_mfe_pips', symbol, _cfg_float(defaults.get('min_mfe_pips'), 10.0))
            min_sell_mfe = _cfg_float_symbol(sell_cfg, 'min_mfe_pips', symbol, _cfg_float(defaults.get('min_mfe_pips'), 10.0))
            neg_mask |= (~buy_pos & (buy_mfe >= float(min_buy_mfe) * float(near_mult)) & (buy_net <= 0)).fillna(False)
            neg_mask |= (~sell_pos & (sell_mfe >= float(min_sell_mfe) * float(near_mult)) & (sell_net <= 0)).fillna(False)
        hard_idx = np.flatnonzero(neg_mask.to_numpy(bool) & (direction < 0))
        hard_ratio = _cfg_float(hard_cfg.get('max_ratio_to_positive'), 2.0)
        seed = int(_cfg_int(scfg.get('random_seed', _label_cfg(cfg).get('random_seed')), 43) or 43)
        if hard_ratio is not None and len(positive_indices) > 0:
            max_hard = int(round(float(hard_ratio) * len(positive_indices)))
            hard_keep = _choose_random_indices(hard_idx, max_hard, seed + 1001)
        else:
            hard_keep = [int(x) for x in hard_idx.tolist()]
        selected_negatives.update(hard_keep)

        if _cfg_bool(bg_cfg.get('enabled'), True):
            bg_ratio = _cfg_float(bg_cfg.get('ratio_to_positive'), 2.0)
            n_bg = int(round(float(bg_ratio or 0.0) * max(1, len(positive_indices))))
            all_no_trade = np.flatnonzero((direction < 0))
            if selected_negatives:
                all_no_trade = np.asarray([i for i in all_no_trade if int(i) not in selected_negatives], dtype=int)
            bg_keep = _choose_random_indices(all_no_trade, n_bg, seed + 2002)
            selected_negatives.update(bg_keep)
        if selected_negatives:
            direction[np.asarray(sorted(selected_negatives), dtype=int)] = 1

    result['direction_target'] = direction.astype(np.int64)

    # Side-specific setup targets for the new training mode:
    #   buy_setup_target/sell_setup_target: 1=clean setup, 0=side-specific failed/background negative, -1=ignore.
    # These solve the over-conservative 3-class gate problem by letting BUY and
    # SELL be learned as two independent setup-quality ranking problems.
    buy_setup_target = np.full(n, -1, dtype=np.int64) if event_based else np.zeros(n, dtype=np.int64)
    sell_setup_target = np.full(n, -1, dtype=np.int64) if event_based else np.zeros(n, dtype=np.int64)
    buy_setup_target[buy_pos.to_numpy(bool)] = 1
    sell_setup_target[buy_pos.to_numpy(bool)] = 0
    sell_setup_target[sell_pos.to_numpy(bool)] = 1
    buy_setup_target[sell_pos.to_numpy(bool)] = 0
    if event_based and selected_negatives:
        neg_idx = np.asarray(sorted(selected_negatives), dtype=int)
        buy_failed_arr = buy_failed.to_numpy(bool)
        sell_failed_arr = sell_failed.to_numpy(bool)
        buy_setup_target[neg_idx[buy_failed_arr[neg_idx]]] = 0
        sell_setup_target[neg_idx[sell_failed_arr[neg_idx]]] = 0
        background_idx = neg_idx[~(buy_failed_arr[neg_idx] | sell_failed_arr[neg_idx])]
        if len(background_idx):
            buy_setup_target[background_idx] = 0
            sell_setup_target[background_idx] = 0
    elif not event_based:
        # In every-bar mode all non-positive rows are negatives for both side heads.
        buy_setup_target[~buy_pos.to_numpy(bool)] = 0
        sell_setup_target[~sell_pos.to_numpy(bool)] = 0

    result['buy_setup_target'] = buy_setup_target.astype(np.int64)
    result['sell_setup_target'] = sell_setup_target.astype(np.int64)
    quality_clip = _cfg_float(scfg.get('quality_clip_pips', scfg.get('quality_clip')), 48.0)
    buy_quality_values = buy_quality.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    sell_quality_values = sell_quality.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    if quality_clip is not None and float(quality_clip) > 0:
        buy_quality_values = np.clip(buy_quality_values, -float(quality_clip), float(quality_clip))
        sell_quality_values = np.clip(sell_quality_values, -float(quality_clip), float(quality_clip))
    result['buy_setup_quality_score_target'] = buy_quality_values
    result['sell_setup_quality_score_target'] = sell_quality_values
    result['buy_setup_analytic_score'] = buy_score.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    result['sell_setup_analytic_score'] = sell_score.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)

    result['buy_candidate_strength_score'] = buy_quality.replace([np.inf, -np.inf], np.nan).to_numpy(float)
    result['sell_candidate_strength_score'] = sell_quality.replace([np.inf, -np.inf], np.nan).to_numpy(float)
    result['candidate_strength_score'] = np.where(
        result['direction_target'].to_numpy(int) == 2,
        result['buy_candidate_strength_score'].to_numpy(float),
        np.where(
            result['direction_target'].to_numpy(int) == 0,
            result['sell_candidate_strength_score'].to_numpy(float),
            np.nan,
        ),
    )
    result['label_filter_status'] = np.where(result['direction_target'] < 0, 'ignored_non_event', 'kept')
    result.loc[result['direction_target'] == 1, 'label_filter_status'] = 'kept_negative'
    result.loc[result['direction_target'] == 2, 'label_filter_status'] = 'kept_buy_positive'
    result.loc[result['direction_target'] == 0, 'label_filter_status'] = 'kept_sell_positive'

    info = {
        'enabled': True,
        'method': _label_method(cfg),
        'output_mode': output_mode,
        'buy_positive_rows': int((result['direction_target'] == 2).sum()),
        'sell_positive_rows': int((result['direction_target'] == 0).sum()),
        'hard_negative_rows': int(len(selected_negatives)),
        'buy_setup_positive_rows': int((result['buy_setup_target'] == 1).sum()),
        'buy_setup_negative_rows': int((result['buy_setup_target'] == 0).sum()),
        'buy_setup_ignored_rows': int((result['buy_setup_target'] < 0).sum()),
        'sell_setup_positive_rows': int((result['sell_setup_target'] == 1).sum()),
        'sell_setup_negative_rows': int((result['sell_setup_target'] == 0).sum()),
        'sell_setup_ignored_rows': int((result['sell_setup_target'] < 0).sum()),
        'ignored_rows': int((result['direction_target'] < 0).sum()),
        'both_side_positive_candidates': int(both.sum()),
        'buy_rules': buy_cfg,
        'sell_rules': sell_cfg,
        'hard_negative_config': hard_cfg,
        'background_no_trade_config': bg_cfg,
    }
    return result, info

def _generate_barrier_direction_targets(df: pd.DataFrame, symbol: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Generate BUY/SELL/NO_TRADE direction targets.

    For every bar, both candidate trades are simulated over the configured horizon.
    In legacy mode the simulation uses the row close and future bid-like high/low
    candles, then subtracts spread afterwards.

    When labels.use_live_bid_ask_simulation is true, labels are generated much
    closer to live execution:
      - the model decides from closed bar i
      - entry can be taken at the next bar open
      - BUY enters at ask and exits against bid prices
      - SELL enters at bid and exits against ask prices
      - spread/slippage affect barrier hits, not just final net pips

    Optional label-quality filters can then keep only the strongest, de-duplicated
    positive setup rows. Discarded positives may be converted to NO_TRADE or
    IGNORE (-1). The training array builder skips IGNORE rows as supervised
    labels while still allowing surrounding bars to exist in sequence context.
    """
    df = df.copy()
    lcfg = _label_cfg(cfg)
    horizon = int(lcfg.get('horizon_bars', 18))
    spread_col = str(lcfg.get('spread_column', 'spread_points'))
    default_spread_points = symbol_default_spread_points(cfg, symbol, default=2.0)
    min_clean_win_net_pips = float(lcfg.get('min_clean_win_net_pips', 0.0))
    min_side_edge_pips = float(lcfg.get('min_side_edge_pips', lcfg.get('min_ev_edge_pips', 0.0)))
    use_live_bid_ask = bool(lcfg.get('use_live_bid_ask_simulation', False))
    entry_on_next_bar_open = bool(lcfg.get('entry_on_next_bar_open', use_live_bid_ask))
    max_spread_pips = symbol_max_spread_pips(cfg, symbol)

    open_col, high_col, low_col, close_col = _required_ohlc_columns(df)
    opens = pd.to_numeric(df[open_col], errors='coerce').to_numpy(float)
    highs = pd.to_numeric(df[high_col], errors='coerce').to_numpy(float)
    lows = pd.to_numeric(df[low_col], errors='coerce').to_numpy(float)
    closes = pd.to_numeric(df[close_col], errors='coerce').to_numpy(float)
    if spread_col in df.columns:
        spreads = pd.to_numeric(df[spread_col], errors='coerce').fillna(default_spread_points).to_numpy(float)
    else:
        spreads = np.full(len(df), default_spread_points, dtype=float)

    n = len(df)

    # Final direction class: 0=SELL, 1=NO_TRADE, 2=BUY. -1=IGNORE when
    # optional strong-setup filters discard a clustered/marginal positive row.
    direction = np.full(n, 1, dtype=np.int64)
    buy_net = np.full(n, np.nan, dtype=float)
    sell_net = np.full(n, np.nan, dtype=float)
    buy_outcome = np.full(n, -1, dtype=np.int64)
    sell_outcome = np.full(n, -1, dtype=np.int64)
    buy_bars = np.full(n, 0, dtype=np.int64)
    sell_bars = np.full(n, 0, dtype=np.int64)
    buy_strength = np.full(n, np.nan, dtype=float)
    sell_strength = np.full(n, np.nan, dtype=float)
    buy_mfe = np.full(n, np.nan, dtype=float)
    buy_mae = np.full(n, np.nan, dtype=float)
    sell_mfe = np.full(n, np.nan, dtype=float)
    sell_mae = np.full(n, np.nan, dtype=float)

    last_i = max(0, n - horizon - 1)
    for i in range(last_i):
        entry_idx = i + 1 if entry_on_next_bar_open else i
        if entry_idx >= n:
            continue
        entry_spread_points = spreads[entry_idx] if use_live_bid_ask else spreads[i]
        if max_spread_pips is not None and _spread_cost_pips(symbol, entry_spread_points, cfg) > max_spread_pips:
            continue

        if use_live_bid_ask:
            entry_bid = opens[entry_idx]
            if not np.isfinite(entry_bid):
                continue
            # Include the entry bar in the future path: a live order placed at
            # next-bar open can hit TP/SL within that same candle.
            future_start = entry_idx
            future_end = min(n, future_start + horizon)
            future_hi = highs[future_start:future_end]
            future_lo = lows[future_start:future_end]
            future_close = closes[future_start:future_end]
            future_spreads = spreads[future_start:future_end]
            if len(future_close) == 0:
                continue
            buy = _first_hit_outcome_live_bidask(
                symbol,
                entry_bid,
                future_hi,
                future_lo,
                future_close,
                future_spreads,
                'BUY',
                cfg,
                entry_spread_points=entry_spread_points,
            )
            sell = _first_hit_outcome_live_bidask(
                symbol,
                entry_bid,
                future_hi,
                future_lo,
                future_close,
                future_spreads,
                'SELL',
                cfg,
                entry_spread_points=entry_spread_points,
            )
            # In live bid/ask mode spread and slippage are already embedded in
            # entry/exit prices and barrier reachability. Do not subtract spread
            # again.
            bnet = float(buy.pips)
            snet = float(sell.pips)
            bmfe, bmae = _mfe_mae_live_bidask(
                symbol, entry_bid, future_hi, future_lo, future_spreads, 'BUY', cfg,
                entry_spread_points=entry_spread_points,
            )
            smfe, smae = _mfe_mae_live_bidask(
                symbol, entry_bid, future_hi, future_lo, future_spreads, 'SELL', cfg,
                entry_spread_points=entry_spread_points,
            )
        else:
            entry = closes[i]
            if not np.isfinite(entry):
                continue
            future_hi = highs[i + 1:i + 1 + horizon]
            future_lo = lows[i + 1:i + 1 + horizon]
            future_close = closes[i + 1:i + 1 + horizon]
            if len(future_close) == 0:
                continue
            buy = _first_hit_outcome(symbol, entry, future_hi, future_lo, future_close, 'BUY', cfg)
            sell = _first_hit_outcome(symbol, entry, future_hi, future_lo, future_close, 'SELL', cfg)
            cost = _spread_cost_pips(symbol, spreads[i], cfg)
            bnet = float(buy.pips - cost)
            snet = float(sell.pips - cost)
            bmfe, bmae = _mfe_mae_legacy(symbol, entry, future_hi, future_lo, 'BUY', cfg)
            smfe, smae = _mfe_mae_legacy(symbol, entry, future_hi, future_lo, 'SELL', cfg)

        buy_net[i] = bnet
        sell_net[i] = snet
        buy_outcome[i] = int(buy.outcome)
        sell_outcome[i] = int(sell.outcome)
        buy_bars[i] = int(buy.bars_to_outcome)
        sell_bars[i] = int(sell.bars_to_outcome)
        buy_mfe[i] = float(bmfe)
        buy_mae[i] = float(bmae)
        sell_mfe[i] = float(smfe)
        sell_mae[i] = float(smae)

        buy_is_clean_win = buy.outcome == 2 and bnet >= min_clean_win_net_pips
        sell_is_clean_win = sell.outcome == 2 and snet >= min_clean_win_net_pips
        if buy_is_clean_win:
            buy_strength[i] = _candidate_strength(bnet, snet, buy.bars_to_outcome, horizon)
        if sell_is_clean_win:
            sell_strength[i] = _candidate_strength(snet, bnet, sell.bars_to_outcome, horizon)

        # Choose the clean winner with better realised net pips.
        if buy_is_clean_win and (not sell_is_clean_win or bnet >= snet + min_side_edge_pips):
            direction[i] = 2
        elif sell_is_clean_win and (not buy_is_clean_win or snet >= bnet + min_side_edge_pips):
            direction[i] = 0

    out = df.iloc[:last_i].copy().reset_index(drop=True)
    keep = len(out)
    out['direction_target'] = direction[:keep]
    out['buy_candidate_net_pips'] = buy_net[:keep]
    out['sell_candidate_net_pips'] = sell_net[:keep]
    # Optional regression targets for the hierarchical edge/pips head. These are
    # future-derived labels, not live features. The feature selector blocks all
    # *_target columns, so they cannot leak into model inputs.
    out['buy_edge_pips_target'] = buy_net[:keep]
    out['sell_edge_pips_target'] = sell_net[:keep]
    out['buy_candidate_outcome'] = buy_outcome[:keep]
    out['sell_candidate_outcome'] = sell_outcome[:keep]
    out['buy_candidate_bars_to_outcome'] = buy_bars[:keep]
    out['sell_candidate_bars_to_outcome'] = sell_bars[:keep]
    out['buy_candidate_mfe_pips'] = buy_mfe[:keep]
    out['buy_candidate_mae_pips'] = buy_mae[:keep]
    out['sell_candidate_mfe_pips'] = sell_mfe[:keep]
    out['sell_candidate_mae_pips'] = sell_mae[:keep]
    out['buy_candidate_strength_score'] = buy_strength[:keep]
    out['sell_candidate_strength_score'] = sell_strength[:keep]
    out['candidate_strength_score'] = np.where(
        out['direction_target'].to_numpy(dtype=int) == 2,
        out['buy_candidate_strength_score'].to_numpy(dtype=float),
        np.where(
            out['direction_target'].to_numpy(dtype=int) == 0,
            out['sell_candidate_strength_score'].to_numpy(dtype=float),
            np.nan,
        ),
    )
    out['label_filter_status'] = 'kept'

    out, strong_setup_info = _apply_strong_setup_labels(out, cfg, symbol)
    out, dedup_info = _apply_positive_deduplication(out, cfg)
    out, daily_cap_info = _apply_daily_positive_cap(out, cfg)
    positive_filter_info = {
        'strong_setup': strong_setup_info,
        'deduplication': dedup_info,
        'daily_cap': daily_cap_info,
        'ignored_rows_after_filters': int((pd.to_numeric(out['direction_target'], errors='coerce') < 0).sum()),
    }

    # These future-derived diagnostics are used only inside label generation to
    # rank strong setup candidates. Drop them before returning so on-the-fly
    # training cannot accidentally use them as model features.
    auxiliary_cols = [
        c for c in out.columns
        if (
            'candidate_' in str(c).lower()
            or str(c).lower() in {'candidate_strength_score', 'label_filter_status'}
        )
    ]
    out = out.drop(columns=auxiliary_cols, errors='ignore')
    out.attrs['positive_label_filters'] = positive_filter_info
    return out

def generate_direction_targets(df: pd.DataFrame, symbol: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Generate the BUY/SELL/NO_TRADE direction label used by the simple model.

    Internally this uses the same live-style barrier simulation that replay uses, then returns a single direction_target column.
    """
    return _generate_barrier_direction_targets(df, symbol, cfg)
