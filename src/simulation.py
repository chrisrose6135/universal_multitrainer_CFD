from __future__ import annotations

import numpy as np
import pandas as pd

from .forex import pips_from_price_delta, price_delta_from_pips, spread_points_to_pips, symbol_cfg_value
from .spread_risk_config import symbol_default_spread_points


def _normalise_trade_side(side) -> str:
    """Return canonical BUY/SELL side labels for replay summaries.

    Older patched inference code could pass numeric direction ids into the
    simulator because DIRECTION_NAMES was treated as an iterable of keys. This
    helper makes the simulator tolerant of both canonical strings and the
    project's numeric direction ids: 0=SELL, 1=NO_TRADE, 2=BUY.
    """
    try:
        # numpy integers should behave the same as Python ints here.
        if int(side) == side and not isinstance(side, bool):
            side_int = int(side)
            if side_int == 0:
                return 'SELL'
            if side_int == 2:
                return 'BUY'
            if side_int == 1:
                raise ValueError('NO_TRADE cannot be simulated as a trade side')
    except Exception:
        pass

    value = str(side).strip().upper()
    if value in {'BUY', 'LONG'}:
        return 'BUY'
    if value in {'SELL', 'SHORT'}:
        return 'SELL'
    if value == '2':
        return 'BUY'
    if value == '0':
        return 'SELL'
    if value in {'1', 'NO_TRADE', 'NONE', 'WAIT', 'BLOCK'}:
        raise ValueError(f'{side!r} cannot be simulated as a trade side')
    raise ValueError(f'Unsupported trade side: {side!r}')


def _label_cfg(cfg: dict) -> dict:
    return cfg.get('labels', {}) or {}


def _spread_cost_pips(symbol: str, spread_points: float, cfg: dict) -> float:
    return spread_points_to_pips(symbol, float(spread_points), cfg)


def _spread_delta(symbol: str, spread_points: float, cfg: dict) -> float:
    return price_delta_from_pips(symbol, _spread_cost_pips(symbol, spread_points, cfg), cfg)


def _same_bar_policy(cfg: dict) -> str:
    lcfg = _label_cfg(cfg)
    policy = str(lcfg.get('same_bar_tp_sl_policy', '') or '').strip().lower()
    if policy in {'stop_first', 'sl_first', 'conservative'}:
        return 'stop_first'
    if policy in {'take_profit_first', 'tp_first', 'optimistic'}:
        return 'take_profit_first'
    if policy in {'discard', 'time_exit'}:
        return 'discard'
    return 'stop_first' if bool(lcfg.get('conservative_same_bar_hits', True)) else 'take_profit_first'


def _required_ohlc_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    lower = {str(c).lower(): c for c in df.columns}
    close = lower.get('close') or lower.get('close_price') or lower.get('bid_close')
    high = lower.get('high') or lower.get('high_price') or lower.get('bid_high')
    low = lower.get('low') or lower.get('low_price') or lower.get('bid_low')
    open_ = lower.get('open') or lower.get('open_price') or lower.get('bid_open') or close
    if not close or not high or not low:
        raise ValueError('Backtest simulation requires close/high/low columns.')
    return open_, high, low, close


def _spread_points_array(df: pd.DataFrame, symbol: str, cfg: dict) -> np.ndarray:
    lcfg = _label_cfg(cfg)
    spread_col = str(lcfg.get('spread_column', 'spread_points'))
    default_spread_points = symbol_default_spread_points(cfg, symbol, default=float(lcfg.get('default_spread_points', 2.0)))
    if spread_col in df.columns:
        return pd.to_numeric(df[spread_col], errors='coerce').fillna(default_spread_points).to_numpy(float)
    return np.full(len(df), default_spread_points, dtype=float)


def _same_bar_result(tp_pips: float, sl_pips: float, cfg: dict) -> tuple[str, float]:
    policy = _same_bar_policy(cfg)
    if policy == 'take_profit_first':
        return 'TP', float(tp_pips)
    if policy == 'discard':
        return 'TIME_EXIT', 0.0
    return 'SL', -float(sl_pips)


def _simulate_live_bidask_trade_from_row(df: pd.DataFrame, row_pos: int, symbol: str, side: str, cfg: dict) -> dict:
    """Replay using the same live-style assumptions used by direction labels.

    The label generator uses labels.use_live_bid_ask_simulation=True to decide
    from closed bar i, enter at next bar open, and then evaluate BUY against bid
    prices and SELL against ask prices. The old replay path used current close
    and bid-only barriers, which could make SELL replay performance diverge from
    the SELL examples the model was trained on. This function keeps replay and
    labels aligned.
    """
    side = _normalise_trade_side(side)
    lcfg = _label_cfg(cfg)
    horizon = int(lcfg.get('horizon_bars', 36))
    tp_pips = float(symbol_cfg_value(cfg, 'labels', 'take_profit_pips', symbol, 6.0))
    sl_pips = float(symbol_cfg_value(cfg, 'labels', 'stop_loss_pips', symbol, 5.0))
    slippage_pips = float(symbol_cfg_value(cfg, 'labels', 'slippage_pips', symbol, 0.0) or 0.0)
    entry_on_next_bar_open = bool(lcfg.get('entry_on_next_bar_open', True))

    open_col, high_col, low_col, close_col = _required_ohlc_columns(df)
    spreads = _spread_points_array(df, symbol, cfg)
    entry_idx = int(row_pos) + 1 if entry_on_next_bar_open else int(row_pos)
    if entry_idx >= len(df):
        raise ValueError('Not enough future rows to simulate next-bar entry.')

    entry_bid = float(df.iloc[entry_idx][open_col])
    if not np.isfinite(entry_bid):
        raise ValueError('Entry price is not finite.')

    future_start = entry_idx
    future_end = min(len(df), future_start + horizon)
    if future_end <= future_start:
        raise ValueError('Not enough future rows to simulate trade horizon.')

    slippage_delta = price_delta_from_pips(symbol, slippage_pips, cfg)
    entry_spread_points = float(spreads[entry_idx])
    exit_i = future_end - 1
    outcome = 'TIME_EXIT'

    if side == 'BUY':
        # MT5/rates OHLC are treated as bid prices. A BUY enters at ask and exits
        # against future bid prices.
        entry_price = entry_bid + _spread_delta(symbol, entry_spread_points, cfg) + slippage_delta
        tp = entry_price + price_delta_from_pips(symbol, tp_pips, cfg)
        sl = entry_price - price_delta_from_pips(symbol, sl_pips, cfg)
        net_pips = pips_from_price_delta(symbol, float(df.iloc[exit_i][close_col]) - entry_price, cfg)
        for i in range(future_start, future_end):
            bid_hi = float(df.iloc[i][high_col])
            bid_lo = float(df.iloc[i][low_col])
            hit_tp = bid_hi >= tp
            hit_sl = bid_lo <= sl
            if hit_tp and hit_sl:
                outcome, net_pips = _same_bar_result(tp_pips, sl_pips, cfg)
                exit_i = i
                break
            if hit_tp:
                outcome = 'TP'
                net_pips = tp_pips
                exit_i = i
                break
            if hit_sl:
                outcome = 'SL'
                net_pips = -sl_pips
                exit_i = i
                break
    else:
        # A SELL enters at bid and exits against future ask prices. Future spread
        # therefore affects both TP and SL reachability, just as in labelling.
        entry_price = entry_bid - slippage_delta
        tp = entry_price - price_delta_from_pips(symbol, tp_pips, cfg)
        sl = entry_price + price_delta_from_pips(symbol, sl_pips, cfg)
        exit_ask = float(df.iloc[exit_i][close_col]) + _spread_delta(symbol, float(spreads[exit_i]), cfg)
        net_pips = pips_from_price_delta(symbol, entry_price - exit_ask, cfg)
        for i in range(future_start, future_end):
            spread_delta = _spread_delta(symbol, float(spreads[i]), cfg)
            ask_hi = float(df.iloc[i][high_col]) + spread_delta
            ask_lo = float(df.iloc[i][low_col]) + spread_delta
            hit_tp = ask_lo <= tp
            hit_sl = ask_hi >= sl
            if hit_tp and hit_sl:
                outcome, net_pips = _same_bar_result(tp_pips, sl_pips, cfg)
                exit_i = i
                break
            if hit_tp:
                outcome = 'TP'
                net_pips = tp_pips
                exit_i = i
                break
            if hit_sl:
                outcome = 'SL'
                net_pips = -sl_pips
                exit_i = i
                break

    return {
        'entry_time': str(df.iloc[entry_idx].get('time_utc', entry_idx)),
        'decision_time': str(df.iloc[row_pos].get('time_utc', row_pos)),
        'exit_time': str(df.iloc[exit_i].get('time_utc', exit_i)),
        'side': side,
        'entry_price': float(entry_price),
        'entry_index': int(entry_idx),
        'exit_index': int(exit_i),
        'gross_pips': float(net_pips),
        'cost_pips': 0.0,
        'net_pips': float(net_pips),
        'outcome': outcome,
        'simulation_mode': 'live_bid_ask_next_open' if entry_on_next_bar_open else 'live_bid_ask_same_row',
        'entry_spread_points': float(entry_spread_points),
        'spread_embedded_in_price_simulation': True,
    }


def _simulate_legacy_trade_from_row(df: pd.DataFrame, row_pos: int, symbol: str, side: str, cfg: dict) -> dict:
    side = _normalise_trade_side(side)
    lcfg = _label_cfg(cfg)
    horizon = int(lcfg.get('horizon_bars', 36))
    tp_pips = float(symbol_cfg_value(cfg, 'labels', 'take_profit_pips', symbol, 6.0))
    sl_pips = float(symbol_cfg_value(cfg, 'labels', 'stop_loss_pips', symbol, 5.0))
    conservative = bool(lcfg.get('conservative_same_bar_hits', True))
    _, high_col, low_col, close_col = _required_ohlc_columns(df)
    entry = float(df.iloc[row_pos][close_col])
    end_pos = min(len(df) - 1, row_pos + horizon)
    exit_price = float(df.iloc[end_pos][close_col])
    outcome = 'TIME_EXIT'
    gross_pips = 0.0
    exit_i = end_pos
    for i in range(row_pos + 1, end_pos + 1):
        hi = float(df.iloc[i][high_col])
        lo = float(df.iloc[i][low_col])
        if side == 'BUY':
            tp = entry + price_delta_from_pips(symbol, tp_pips, cfg)
            sl = entry - price_delta_from_pips(symbol, sl_pips, cfg)
            hit_tp, hit_sl = hi >= tp, lo <= sl
            if hit_tp and hit_sl:
                outcome = 'SL' if conservative else 'TP'
                gross_pips = -sl_pips if conservative else tp_pips
                exit_i = i
                break
            if hit_tp:
                outcome = 'TP'
                gross_pips = tp_pips
                exit_i = i
                break
            if hit_sl:
                outcome = 'SL'
                gross_pips = -sl_pips
                exit_i = i
                break
        else:
            tp = entry - price_delta_from_pips(symbol, tp_pips, cfg)
            sl = entry + price_delta_from_pips(symbol, sl_pips, cfg)
            hit_tp, hit_sl = lo <= tp, hi >= sl
            if hit_tp and hit_sl:
                outcome = 'SL' if conservative else 'TP'
                gross_pips = -sl_pips if conservative else tp_pips
                exit_i = i
                break
            if hit_tp:
                outcome = 'TP'
                gross_pips = tp_pips
                exit_i = i
                break
            if hit_sl:
                outcome = 'SL'
                gross_pips = -sl_pips
                exit_i = i
                break
    else:
        if side == 'BUY':
            gross_pips = pips_from_price_delta(symbol, exit_price - entry, cfg)
        else:
            gross_pips = pips_from_price_delta(symbol, entry - exit_price, cfg)
    spreads = _spread_points_array(df, symbol, cfg)
    cost_pips = _spread_cost_pips(symbol, float(spreads[row_pos]), cfg)
    net_pips = gross_pips - cost_pips
    return {
        'entry_time': str(df.iloc[row_pos].get('time_utc', row_pos)),
        'exit_time': str(df.iloc[exit_i].get('time_utc', exit_i)),
        'side': side,
        'entry_price': entry,
        'exit_index': int(exit_i),
        'gross_pips': float(gross_pips),
        'cost_pips': float(cost_pips),
        'net_pips': float(net_pips),
        'outcome': outcome,
        'simulation_mode': 'legacy_close_bid_ohlc',
    }


def simulate_trade_from_row(df: pd.DataFrame, row_pos: int, symbol: str, side: str, cfg: dict) -> dict:
    """Simulate a replay trade from the model decision row.

    When labels.use_live_bid_ask_simulation is enabled, this deliberately uses
    the same entry and bid/ask barrier mechanics as target generation. That is
    critical for checking whether BUY/SELL labels are producing reasonable live-
    equivalent examples.
    """
    if bool((_label_cfg(cfg)).get('use_live_bid_ask_simulation', False)):
        return _simulate_live_bidask_trade_from_row(df, row_pos, symbol, side, cfg)
    return _simulate_legacy_trade_from_row(df, row_pos, symbol, side, cfg)


def _side_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {
            'trades': 0,
            'net_pips': 0.0,
            'win_rate': 0.0,
            'average_net_pips': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'loss_pips': 0.0,
            'win_pips': 0.0,
            'worst_trade_pips': 0.0,
            'best_trade_pips': 0.0,
        }
    pips = np.array([float(t['net_pips']) for t in trades], dtype=float)
    return {
        'trades': int(len(trades)),
        'net_pips': float(pips.sum()),
        'win_rate': float((pips > 0).mean()),
        'average_net_pips': float(pips.mean()),
        'winning_trades': int((pips > 0).sum()),
        'losing_trades': int((pips < 0).sum()),
        'loss_pips': float((-pips[pips < 0]).sum()) if (pips < 0).any() else 0.0,
        'win_pips': float(pips[pips > 0].sum()) if (pips > 0).any() else 0.0,
        'worst_trade_pips': float(pips.min()),
        'best_trade_pips': float(pips.max()),
    }


def summarise_trades(trades: list[dict]) -> dict:
    if not trades:
        return {
            'trades': 0,
            'net_pips': 0.0,
            'win_rate': 0.0,
            'average_net_pips': 0.0,
            'max_drawdown_pips': 0.0,
            'buy_trades': 0,
            'sell_trades': 0,
            'buy_net_pips': 0.0,
            'sell_net_pips': 0.0,
            'buy_win_rate': 0.0,
            'sell_win_rate': 0.0,
            'buy_average_net_pips': 0.0,
            'sell_average_net_pips': 0.0,
            'buy_loss_pips': 0.0,
            'sell_loss_pips': 0.0,
            'buy_losing_trades': 0,
            'sell_losing_trades': 0,
        }
    pips = np.array([float(t['net_pips']) for t in trades], dtype=float)
    equity = np.cumsum(pips)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity

    def trade_side(t: dict) -> str | None:
        for key in ('side', 'direction', 'predicted_direction', 'raw_direction'):
            if key in t and t.get(key) is not None:
                try:
                    return _normalise_trade_side(t.get(key))
                except ValueError:
                    return None
        return None

    buys = [t for t in trades if trade_side(t) == 'BUY']
    sells = [t for t in trades if trade_side(t) == 'SELL']
    buy = _side_metrics(buys)
    sell = _side_metrics(sells)
    return {
        'trades': int(len(trades)),
        'net_pips': float(pips.sum()),
        'win_rate': float((pips > 0).mean()),
        'average_net_pips': float(pips.mean()),
        'max_drawdown_pips': float(dd.max() if len(dd) else 0.0),
        'buy_trades': buy['trades'],
        'sell_trades': sell['trades'],
        'buy_net_pips': buy['net_pips'],
        'sell_net_pips': sell['net_pips'],
        'buy_win_rate': buy['win_rate'],
        'sell_win_rate': sell['win_rate'],
        'buy_average_net_pips': buy['average_net_pips'],
        'sell_average_net_pips': sell['average_net_pips'],
        'buy_losing_trades': buy['losing_trades'],
        'sell_losing_trades': sell['losing_trades'],
        'buy_winning_trades': buy['winning_trades'],
        'sell_winning_trades': sell['winning_trades'],
        'buy_loss_pips': buy['loss_pips'],
        'sell_loss_pips': sell['loss_pips'],
        'buy_win_pips': buy['win_pips'],
        'sell_win_pips': sell['win_pips'],
        'buy_worst_trade_pips': buy['worst_trade_pips'],
        'sell_worst_trade_pips': sell['worst_trade_pips'],
        'buy_best_trade_pips': buy['best_trade_pips'],
        'sell_best_trade_pips': sell['best_trade_pips'],
    }
