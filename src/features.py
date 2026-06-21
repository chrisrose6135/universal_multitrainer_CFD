from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .indicators import adx, atr, bollinger, ema, macd, rsi
from .spread_utils import apply_spread_fallback, infer_symbol_from_frame
from .analytic_signals import ANALYTIC_SIGNAL_FEATURE_COLUMNS, add_analytic_signal_features


BASE_FEATURE_COLUMNS = [
    # Returns / momentum
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48", "ret_96",
    "logret_1", "logret_3", "logret_12", "logret_24", "momentum_24", "momentum_48",

    # Candle anatomy
    "hl_range_norm", "body_norm", "upper_wick_norm", "lower_wick_norm",
    "hl_range_atr", "body_atr", "upper_wick_atr", "lower_wick_atr",
    "body_to_range", "close_position_in_range", "candle_bullish", "candle_bearish",

    # Moving averages / trend distances
    "ema_20_dist", "ema_50_dist", "ema_200_dist", "ema_20_50_diff", "ema_50_200_diff",
    "dist_ema20_atr", "dist_ema50_atr", "dist_ema200_atr",

    # Volatility
    "vol_6", "vol_12", "vol_24", "vol_48", "vol_96", "vol_ratio_6_24", "vol_ratio_24_96",
    "atr_14_norm", "atr_14_change",

    # Oscillators / momentum indicators
    "rsi_14_norm", "rsi_14_delta", "rsi_14_slope_3", "adx_14_norm",
    "macd_norm", "macd_signal_norm", "macd_hist_norm", "macd_hist_delta", "macd_hist_slope_3",

    # Bollinger / range location
    "bb_width", "bb_position", "bb_zscore", "bb_upper_dist", "bb_lower_dist",
    "dist_roll_high_24", "dist_roll_low_24", "rolling_range_24_norm",
    "dist_roll_high_48", "dist_roll_low_48", "rolling_range_48_norm",
    "dist_roll_high_96", "dist_roll_low_96", "rolling_range_96_norm",

    # Time/session features
    "hour_utc_sin", "hour_utc_cos", "dayofweek_sin", "dayofweek_cos",
    "session_asia", "session_london", "session_newyork", "session_london_newyork_overlap",

    # Spread / volume
    "spread_norm", "spread_atr", "spread_rolling_z", "tick_volume_norm", "tick_volume_z", "tick_volume_change",
]


FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ANALYTIC_SIGNAL_FEATURE_COLUMNS


def _safe_div(num, den):
    return num / pd.Series(den).replace(0, np.nan)


def _analytics_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    acfg = dict((cfg.get("analytics") or {}))
    acfg.setdefault("ema_fast", 36)
    acfg.setdefault("ema_mid", 96)
    acfg.setdefault("ema_slow", 288)
    acfg.setdefault("rsi_period", 14)
    acfg.setdefault("atr_period", 14)
    acfg.setdefault("adx_period", 14)
    acfg.setdefault("bb_period", 20)
    acfg.setdefault("bb_std", 2.0)
    acfg.setdefault("swing_lookback", 144)
    acfg.setdefault("support_resistance_tolerance_atr", 0.18)
    return acfg


def _normalise_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time" not in out.columns:
        for c in ["time_utc", "datetime", "DateTime", "timestamp"]:
            if c in out.columns:
                out["time"] = out[c]
                break
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
        out["time_utc"] = out["time"]
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Raw MT5/rates data is missing required OHLC columns: {missing}")
    for c in required + ["tick_volume", "spread", "real_volume", "spread_points"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "tick_volume" not in out.columns:
        out["tick_volume"] = 0.0
    if "spread" not in out.columns:
        out["spread"] = np.nan
    return out


def build_feature_frame(df: pd.DataFrame, cfg: dict[str, Any], symbol: str | None = None, drop_warmup: bool = True) -> pd.DataFrame:
    """Build live-safe processed features for train, backtest and MT5 live inference.

    All features are calculated from the current or previous closed candles only.
    No target/label/future outcome fields are created here.
    """
    df = _normalise_raw_columns(df)
    if symbol is not None:
        df["symbol"] = str(symbol).upper()
    df = df.sort_values("time").reset_index(drop=True)
    acfg = _analytics_cfg(cfg)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    safe_close = close.replace(0, np.nan)

    # Returns and momentum.
    for lag in (1, 3, 6, 12, 24, 48, 96):
        df[f"ret_{lag}"] = close.pct_change(lag)
    for lag in (1, 3, 12, 24):
        df[f"logret_{lag}"] = np.log(_safe_div(close, close.shift(lag)))
    df["momentum_24"] = _safe_div(close - close.shift(24), close.shift(24))
    df["momentum_48"] = _safe_div(close - close.shift(48), close.shift(48))

    # Candle anatomy.
    hl_range = high - low
    body = close - open_
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low

    df["hl_range_norm"] = hl_range / safe_close
    df["body_norm"] = body / safe_close
    df["upper_wick_norm"] = upper_wick / safe_close
    df["lower_wick_norm"] = lower_wick / safe_close
    df["body_to_range"] = _safe_div(body.abs(), hl_range)
    df["close_position_in_range"] = _safe_div(close - low, hl_range)
    df["candle_bullish"] = (close > open_).astype(float)
    df["candle_bearish"] = (close < open_).astype(float)

    # Trend / moving-average context.
    df["ema_20"] = ema(close, int(acfg["ema_fast"]))
    df["ema_50"] = ema(close, int(acfg["ema_mid"]))
    df["ema_200"] = ema(close, int(acfg["ema_slow"]))
    df["ema_20_dist"] = (close - df["ema_20"]) / df["ema_20"].replace(0, np.nan)
    df["ema_50_dist"] = (close - df["ema_50"]) / df["ema_50"].replace(0, np.nan)
    df["ema_200_dist"] = (close - df["ema_200"]) / df["ema_200"].replace(0, np.nan)
    df["ema_20_50_diff"] = (df["ema_20"] - df["ema_50"]) / df["ema_50"].replace(0, np.nan)
    df["ema_50_200_diff"] = (df["ema_50"] - df["ema_200"]) / df["ema_200"].replace(0, np.nan)

    # Volatility and ATR-normalised distances.
    df["vol_6"] = df["ret_1"].rolling(6).std()
    df["vol_12"] = df["ret_1"].rolling(12).std()
    df["vol_24"] = df["ret_1"].rolling(24).std()
    df["vol_48"] = df["ret_1"].rolling(48).std()
    df["vol_96"] = df["ret_1"].rolling(96).std()
    df["vol_ratio_6_24"] = _safe_div(df["vol_6"], df["vol_24"])
    df["vol_ratio_24_96"] = _safe_div(df["vol_24"], df["vol_96"])

    df["atr_14"] = atr(df, int(acfg["atr_period"]))
    atr_safe = df["atr_14"].replace(0, np.nan)
    df["atr_14_norm"] = df["atr_14"] / safe_close
    df["atr_14_change"] = df["atr_14"].pct_change(12)
    df["hl_range_atr"] = _safe_div(hl_range, atr_safe)
    df["body_atr"] = _safe_div(body, atr_safe)
    df["upper_wick_atr"] = _safe_div(upper_wick, atr_safe)
    df["lower_wick_atr"] = _safe_div(lower_wick, atr_safe)
    df["dist_ema20_atr"] = _safe_div(close - df["ema_20"], atr_safe)
    df["dist_ema50_atr"] = _safe_div(close - df["ema_50"], atr_safe)
    df["dist_ema200_atr"] = _safe_div(close - df["ema_200"], atr_safe)

    df["rsi_14"] = rsi(close, int(acfg["rsi_period"]))
    df["rsi_14_norm"] = df["rsi_14"] / 100.0
    df["rsi_14_delta"] = df["rsi_14"].diff(1) / 100.0
    df["rsi_14_slope_3"] = df["rsi_14"].diff(3) / 100.0
    df["adx_14"] = adx(df, int(acfg["adx_period"]))
    df["adx_14_norm"] = df["adx_14"] / 100.0

    # Convenience aliases read by the deterministic reasoning layer.
    df["rsi"] = df["rsi_14"]
    df["adx"] = df["adx_14"]

    macd_line, signal_line, hist = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["macd_norm"] = df["macd"] / safe_close
    df["macd_signal_norm"] = df["macd_signal"] / safe_close
    df["macd_hist_norm"] = df["macd_hist"] / safe_close
    df["macd_hist_delta"] = df["macd_hist"].diff(1) / safe_close
    df["macd_hist_slope_3"] = df["macd_hist"].diff(3) / safe_close

    bb_upper, bb_mid, bb_lower, bb_width = bollinger(close, period=int(acfg["bb_period"]), std_mult=float(acfg["bb_std"]))
    df["bb_upper"] = bb_upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width
    bb_span = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (close - bb_lower) / bb_span
    rolling_std = close.rolling(int(acfg["bb_period"])).std().replace(0, np.nan)
    df["bb_zscore"] = (close - bb_mid) / rolling_std
    df["bb_upper_dist"] = _safe_div(bb_upper - close, atr_safe)
    df["bb_lower_dist"] = _safe_div(close - bb_lower, atr_safe)

    # Rolling high/low position. Uses the current closed candle and previous candles only.
    for lookback in (24, 48, 96):
        roll_high = high.rolling(lookback).max()
        roll_low = low.rolling(lookback).min()
        roll_range = (roll_high - roll_low).replace(0, np.nan)
        df[f"dist_roll_high_{lookback}"] = _safe_div(close - roll_high, atr_safe)
        df[f"dist_roll_low_{lookback}"] = _safe_div(close - roll_low, atr_safe)
        df[f"rolling_range_{lookback}_norm"] = roll_range / safe_close

    # Time/session features.
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    hour = t.dt.hour.fillna(0).astype(int)
    dow = t.dt.dayofweek.fillna(0).astype(int)
    df["hour_utc_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_utc_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dow / 5)
    df["session_asia"] = ((hour >= 0) & (hour < 8)).astype(float)
    df["session_london"] = ((hour >= 7) & (hour < 16)).astype(float)
    df["session_newyork"] = ((hour >= 12) & (hour < 21)).astype(float)
    df["session_london_newyork_overlap"] = ((hour >= 12) & (hour < 16)).astype(float)

    # Spread and volume features. apply_spread_fallback produces spread_points.
    symbol_for_spread = symbol or infer_symbol_from_frame(df, default="US500")
    df = apply_spread_fallback(df, symbol_for_spread, cfg)
    df["spread_norm"] = df["spread_points"] / 100.0
    df["spread_atr"] = _safe_div(df["spread_points"], atr_safe)
    spread_mean = df["spread_points"].rolling(96).mean()
    spread_std_raw = df["spread_points"].rolling(96).std()
    spread_std = spread_std_raw.replace(0, np.nan)
    df["spread_rolling_z"] = ((df["spread_points"] - spread_mean) / spread_std).fillna(0.0)

    vol_mean = df["tick_volume"].rolling(100).mean()
    vol_std_raw = df["tick_volume"].rolling(100).std()
    vol_std = vol_std_raw.replace(0, np.nan)
    df["tick_volume_norm"] = df["tick_volume"] / vol_mean.replace(0, np.nan)
    df["tick_volume_z"] = ((df["tick_volume"] - vol_mean) / vol_std).fillna(0.0)
    df["tick_volume_change"] = df["tick_volume"].pct_change(12)

    # Support/resistance booleans used by reasoning. They are also live-safe and
    # may be selected as numeric/bool inputs when include_columns is empty.
    lookback = int(acfg["swing_lookback"])
    tol = float(acfg["support_resistance_tolerance_atr"])
    swing_low = low.rolling(lookback, min_periods=10).min()
    swing_high = high.rolling(lookback, min_periods=10).max()
    df["near_support"] = (close - swing_low).abs() <= (df["atr_14"] * tol)
    df["near_resistance"] = (close - swing_high).abs() <= (df["atr_14"] * tol)

    # Causal analytic trading-signal features. These are advisory model inputs
    # only; future TP/SL labels remain the source of truth for training.
    df = add_analytic_signal_features(df, cfg)

    if drop_warmup:
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["open", "high", "low", "close"]).reset_index(drop=True)
    else:
        df = df.replace([np.inf, -np.inf], np.nan)
    return df
