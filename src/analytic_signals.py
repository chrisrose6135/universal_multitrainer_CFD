from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ANALYTIC_SIGNAL_CONTINUOUS_COLUMNS = [
    # Trend state / location
    'sig_ema_fast_minus_mid_atr',
    'sig_ema_mid_minus_slow_atr',
    'sig_ema_fast_slope_atr',
    'sig_ema_mid_slope_atr',
    'sig_price_above_ema_fast',
    'sig_price_above_ema_mid',
    'sig_price_above_ema_slow',
    'sig_plus_di_minus_di_norm',
    'sig_adx_strength',
    # Momentum / oscillator state
    'sig_rsi_centered',
    'sig_rsi_slope',
    'sig_macd_hist_atr',
    'sig_macd_hist_slope_atr',
    # Volatility / range state
    'sig_atr_zscore',
    'sig_bb_width_zscore',
    'sig_donchian_position',
    'sig_dist_prev_high_atr',
    'sig_dist_prev_low_atr',
]

ANALYTIC_SIGNAL_FLAG_COLUMNS = [
    'sig_ema_trend_buy',
    'sig_ema_trend_sell',
    'sig_macd_buy',
    'sig_macd_sell',
    'sig_rsi_reversal_buy',
    'sig_rsi_reversal_sell',
    'sig_bollinger_reversion_buy',
    'sig_bollinger_reversion_sell',
    'sig_breakout_buy',
    'sig_breakout_sell',
    'sig_support_bounce_buy',
    'sig_resistance_reject_sell',
]

ANALYTIC_SIGNAL_VOTE_COLUMNS = [
    'sig_buy_signal_count',
    'sig_sell_signal_count',
    'sig_net_signal_vote',
    'sig_signal_conflict',
    'sig_any_buy_signal',
    'sig_any_sell_signal',
    'sig_any_trade_signal',
    'sig_analytic_signal_class',  # 0=SELL vote, 1=neutral/conflict, 2=BUY vote
]

ANALYTIC_SIGNAL_FEATURE_COLUMNS = (
    ANALYTIC_SIGNAL_CONTINUOUS_COLUMNS
    + ANALYTIC_SIGNAL_FLAG_COLUMNS
    + ANALYTIC_SIGNAL_VOTE_COLUMNS
)


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def analytic_signals_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    fcfg = cfg.get('features', {}) or {}
    scfg = dict(fcfg.get('analytic_signals') or {})
    scfg.setdefault('enabled', False)
    scfg.setdefault('include_continuous_indicators', True)
    scfg.setdefault('include_signal_flags', True)
    scfg.setdefault('include_signal_votes', True)
    scfg.setdefault('ema_fast_column', 'ema_20')
    scfg.setdefault('ema_mid_column', 'ema_50')
    scfg.setdefault('ema_slow_column', 'ema_200')
    scfg.setdefault('slope_lookback_bars', 3)
    scfg.setdefault('adx_strong_level', 25.0)
    scfg.setdefault('rsi_reversal_buy_max', 35.0)
    scfg.setdefault('rsi_reversal_sell_min', 65.0)
    scfg.setdefault('rsi_reversal_exit_level_buy', 45.0)
    scfg.setdefault('rsi_reversal_exit_level_sell', 55.0)
    scfg.setdefault('bollinger_reversion_z', 1.5)
    scfg.setdefault('donchian_period', 48)
    scfg.setdefault('volatility_z_window', 288)
    scfg.setdefault('min_adx_for_trend_signal', 12.0)
    scfg.setdefault('require_adx_for_trend_signal', False)
    scfg.setdefault('require_session_for_signals', False)
    return scfg


def _safe_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors='coerce')
    return pd.Series(float(default), index=df.index, dtype='float64')


def _safe_div(num: pd.Series, den: pd.Series | float) -> pd.Series:
    if isinstance(den, pd.Series):
        den_safe = den.replace(0, np.nan)
    else:
        den_safe = np.nan if float(den) == 0.0 else float(den)
    return num / den_safe


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 2)
    mean = series.rolling(window, min_periods=max(10, min(window, 30))).mean()
    std = series.rolling(window, min_periods=max(10, min(window, 30))).std().replace(0, np.nan)
    return (series - mean) / std


def _zero_analytic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ANALYTIC_SIGNAL_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out


def add_analytic_signal_features(df: pd.DataFrame, cfg: dict[str, Any], *, force: bool = False) -> pd.DataFrame:
    """Add causal analytic trading-signal features.

    These are advisory input features only. They are calculated from the current
    closed candle and earlier candles, so they remain compatible with next-bar
    execution. They do not replace the future TP/SL direction labels.
    """
    scfg = analytic_signals_cfg(cfg)
    enabled = _cfg_bool(scfg.get('enabled'), False) or bool(force)
    if not enabled:
        return _zero_analytic_columns(df)

    out = df.copy()
    close = _safe_series(out, 'close')
    high = _safe_series(out, 'high')
    low = _safe_series(out, 'low')

    ema_fast = _safe_series(out, str(scfg.get('ema_fast_column') or 'ema_20'))
    ema_mid = _safe_series(out, str(scfg.get('ema_mid_column') or 'ema_50'))
    ema_slow = _safe_series(out, str(scfg.get('ema_slow_column') or 'ema_200'))
    atr = _safe_series(out, 'atr_14').replace(0, np.nan)
    rsi = _safe_series(out, 'rsi_14', default=50.0)
    adx = _safe_series(out, 'adx_14')
    macd_hist = _safe_series(out, 'macd_hist')
    bb_z = _safe_series(out, 'bb_zscore')
    bb_width = _safe_series(out, 'bb_width')
    close_position = _safe_series(out, 'close_position_in_range', default=0.5)

    slope_n = max(int(scfg.get('slope_lookback_bars', 3) or 3), 1)
    donchian_n = max(int(scfg.get('donchian_period', 48) or 48), 2)
    z_window = max(int(scfg.get('volatility_z_window', 288) or 288), 20)

    ema_fast_slope = ema_fast.diff(slope_n)
    ema_mid_slope = ema_mid.diff(slope_n)
    macd_hist_slope = macd_hist.diff(slope_n)

    # Directional movement components are re-created here so the signal layer
    # does not need to depend on any non-feature columns from the ADX helper.
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_for_di = atr.replace(0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_for_di
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_for_di

    prev_high = high.shift(1).rolling(donchian_n, min_periods=max(5, min(12, donchian_n))).max()
    prev_low = low.shift(1).rolling(donchian_n, min_periods=max(5, min(12, donchian_n))).min()
    prev_range = (prev_high - prev_low).replace(0, np.nan)

    include_cont = _cfg_bool(scfg.get('include_continuous_indicators'), True)
    if include_cont:
        out['sig_ema_fast_minus_mid_atr'] = _safe_div(ema_fast - ema_mid, atr)
        out['sig_ema_mid_minus_slow_atr'] = _safe_div(ema_mid - ema_slow, atr)
        out['sig_ema_fast_slope_atr'] = _safe_div(ema_fast_slope, atr)
        out['sig_ema_mid_slope_atr'] = _safe_div(ema_mid_slope, atr)
        out['sig_price_above_ema_fast'] = (close > ema_fast).astype(float)
        out['sig_price_above_ema_mid'] = (close > ema_mid).astype(float)
        out['sig_price_above_ema_slow'] = (close > ema_slow).astype(float)
        out['sig_plus_di_minus_di_norm'] = (plus_di - minus_di) / 100.0
        out['sig_adx_strength'] = adx / 100.0
        out['sig_rsi_centered'] = (rsi - 50.0) / 50.0
        out['sig_rsi_slope'] = rsi.diff(slope_n) / 100.0
        out['sig_macd_hist_atr'] = _safe_div(macd_hist, atr)
        out['sig_macd_hist_slope_atr'] = _safe_div(macd_hist_slope, atr)
        out['sig_atr_zscore'] = _rolling_z(atr, z_window)
        out['sig_bb_width_zscore'] = _rolling_z(bb_width, z_window)
        out['sig_donchian_position'] = (close - prev_low) / prev_range
        out['sig_dist_prev_high_atr'] = _safe_div(close - prev_high, atr)
        out['sig_dist_prev_low_atr'] = _safe_div(close - prev_low, atr)
    else:
        for col in ANALYTIC_SIGNAL_CONTINUOUS_COLUMNS:
            out[col] = 0.0

    min_adx = float(scfg.get('min_adx_for_trend_signal', 12.0) or 0.0)
    require_adx = _cfg_bool(scfg.get('require_adx_for_trend_signal'), False)
    trend_adx_ok = (adx >= min_adx) if require_adx else pd.Series(True, index=out.index)
    session_ok = pd.Series(True, index=out.index)
    if _cfg_bool(scfg.get('require_session_for_signals'), False):
        london = _safe_series(out, 'session_london') > 0.5
        ny = _safe_series(out, 'session_newyork') > 0.5
        session_ok = london | ny

    ema_buy = (ema_fast > ema_mid) & (ema_mid > ema_slow) & (ema_fast_slope > 0) & (close > ema_fast) & trend_adx_ok & session_ok
    ema_sell = (ema_fast < ema_mid) & (ema_mid < ema_slow) & (ema_fast_slope < 0) & (close < ema_fast) & trend_adx_ok & session_ok
    macd_buy = (macd_hist > 0) & (macd_hist_slope > 0) & session_ok
    macd_sell = (macd_hist < 0) & (macd_hist_slope < 0) & session_ok

    rsi_buy_max = float(scfg.get('rsi_reversal_buy_max', 35.0) or 35.0)
    rsi_sell_min = float(scfg.get('rsi_reversal_sell_min', 65.0) or 65.0)
    rsi_buy_exit = float(scfg.get('rsi_reversal_exit_level_buy', 45.0) or 45.0)
    rsi_sell_exit = float(scfg.get('rsi_reversal_exit_level_sell', 55.0) or 55.0)
    rsi_slope = rsi.diff(slope_n)
    rsi_rev_buy = ((rsi.shift(1) <= rsi_buy_max) & (rsi > rsi.shift(1)) & (rsi <= rsi_buy_exit) & (close_position > 0.45) & session_ok)
    rsi_rev_sell = ((rsi.shift(1) >= rsi_sell_min) & (rsi < rsi.shift(1)) & (rsi >= rsi_sell_exit) & (close_position < 0.55) & session_ok)

    bb_reversion_z = float(scfg.get('bollinger_reversion_z', 1.5) or 1.5)
    bb_buy = (bb_z <= -bb_reversion_z) & (rsi < 50.0) & (rsi_slope >= 0) & session_ok
    bb_sell = (bb_z >= bb_reversion_z) & (rsi > 50.0) & (rsi_slope <= 0) & session_ok

    breakout_buy = (close > prev_high) & (macd_hist >= 0) & (ema_fast_slope >= 0) & session_ok
    breakout_sell = (close < prev_low) & (macd_hist <= 0) & (ema_fast_slope <= 0) & session_ok
    near_support = out['near_support'].astype(bool) if 'near_support' in out.columns else pd.Series(False, index=out.index)
    near_resistance = out['near_resistance'].astype(bool) if 'near_resistance' in out.columns else pd.Series(False, index=out.index)
    support_buy = near_support & (rsi_slope >= 0) & (close_position > 0.35) & session_ok
    resistance_sell = near_resistance & (rsi_slope <= 0) & (close_position < 0.65) & session_ok

    flag_values = {
        'sig_ema_trend_buy': ema_buy,
        'sig_ema_trend_sell': ema_sell,
        'sig_macd_buy': macd_buy,
        'sig_macd_sell': macd_sell,
        'sig_rsi_reversal_buy': rsi_rev_buy,
        'sig_rsi_reversal_sell': rsi_rev_sell,
        'sig_bollinger_reversion_buy': bb_buy,
        'sig_bollinger_reversion_sell': bb_sell,
        'sig_breakout_buy': breakout_buy,
        'sig_breakout_sell': breakout_sell,
        'sig_support_bounce_buy': support_buy,
        'sig_resistance_reject_sell': resistance_sell,
    }
    include_flags = _cfg_bool(scfg.get('include_signal_flags'), True)
    if include_flags:
        for col, value in flag_values.items():
            out[col] = value.astype(float)
    else:
        for col in ANALYTIC_SIGNAL_FLAG_COLUMNS:
            out[col] = 0.0

    include_votes = _cfg_bool(scfg.get('include_signal_votes'), True)
    if include_votes:
        buy_cols = [c for c in ANALYTIC_SIGNAL_FLAG_COLUMNS if c.endswith('_buy')]
        sell_cols = [c for c in ANALYTIC_SIGNAL_FLAG_COLUMNS if c.endswith('_sell')]
        buy_count = out[buy_cols].sum(axis=1) if buy_cols else pd.Series(0.0, index=out.index)
        sell_count = out[sell_cols].sum(axis=1) if sell_cols else pd.Series(0.0, index=out.index)
        net_vote = buy_count - sell_count
        out['sig_buy_signal_count'] = buy_count.astype(float)
        out['sig_sell_signal_count'] = sell_count.astype(float)
        out['sig_net_signal_vote'] = net_vote.astype(float)
        out['sig_signal_conflict'] = ((buy_count > 0) & (sell_count > 0)).astype(float)
        out['sig_any_buy_signal'] = (buy_count > 0).astype(float)
        out['sig_any_sell_signal'] = (sell_count > 0).astype(float)
        out['sig_any_trade_signal'] = ((buy_count + sell_count) > 0).astype(float)
        cls = np.where(net_vote > 0, 2, np.where(net_vote < 0, 0, 1))
        # Treat direct buy/sell conflict as neutral unless one side has a larger vote.
        out['sig_analytic_signal_class'] = cls.astype(float)
    else:
        for col in ANALYTIC_SIGNAL_VOTE_COLUMNS:
            out[col] = 0.0

    out[ANALYTIC_SIGNAL_FEATURE_COLUMNS] = (
        out[ANALYTIC_SIGNAL_FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    return out


def ensure_analytic_signal_features(df: pd.DataFrame, cfg: dict[str, Any], *, force: bool = False) -> pd.DataFrame:
    """Ensure analytic-signal columns exist when configs request them.

    This lets old pregenerated direction CSVs be augmented at training/replay
    time as long as the OHLC and base indicator columns are present.
    """
    scfg = analytic_signals_cfg(cfg)
    enabled = _cfg_bool(scfg.get('enabled'), False) or bool(force)
    missing = [c for c in ANALYTIC_SIGNAL_FEATURE_COLUMNS if c not in df.columns]
    if enabled or missing:
        return add_analytic_signal_features(df, cfg, force=enabled)
    return df
