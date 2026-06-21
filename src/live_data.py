from __future__ import annotations

from typing import Any

import pandas as pd

from .features import build_feature_frame
from .live_spread_utils import inject_live_spread_into_bars, read_live_spread_points
from .m5_bar_state import closed_m5_bars_only
from .mt5_client import latest_rates, require_mt5
from .spread_utils import apply_spread_fallback


def _time_range_meta(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if df.empty or "time_utc" not in df.columns:
        return {f"{prefix}_start": None, f"{prefix}_end": None}
    t = pd.to_datetime(df["time_utc"], utc=True, errors="coerce").dropna()
    if t.empty:
        return {f"{prefix}_start": None, f"{prefix}_end": None}
    return {f"{prefix}_start": str(t.min()), f"{prefix}_end": str(t.max())}


def latest_processed_features(symbol: str, cfg: dict[str, Any], bars: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch latest MT5 bars and build the same feature frame used in training."""
    timeframe = str((cfg.get("trading", {}) or {}).get("timeframe", "M5")).upper()
    bars = int(bars or (cfg.get("live", {}) or {}).get("bars", 800))
    raw = latest_rates(symbol, count=bars, timeframe=timeframe, cfg=cfg)
    raw = closed_m5_bars_only(raw, cfg)
    meta: dict[str, Any] = {"timeframe": timeframe, "raw_rows": len(raw), **_time_range_meta(raw, "raw")}
    if not raw.empty:
        if "time_broker" in raw.columns:
            bt = pd.to_datetime(raw["time_broker"], utc=True, errors="coerce").dropna()
            meta["raw_broker_time_start"] = str(bt.min()) if not bt.empty else None
            meta["raw_broker_time_end"] = str(bt.max()) if not bt.empty else None
        if "mt5_time_adjustment_applied" in raw.columns:
            meta["mt5_time_adjustment_applied"] = bool(raw["mt5_time_adjustment_applied"].astype(bool).any())
        if "broker_server_utc_offset_hours" in raw.columns:
            vals = pd.to_numeric(raw["broker_server_utc_offset_hours"], errors="coerce").dropna()
            meta["broker_server_utc_offset_hours"] = float(vals.iloc[-1]) if not vals.empty else None

    live_spread = None
    live_spread_source = "not_read"
    try:
        mt5_mod = require_mt5()
        live_spread, live_spread_source = read_live_spread_points(mt5_mod, symbol, cfg)
        raw = inject_live_spread_into_bars(raw, live_spread, live_spread_source)
    except Exception as exc:
        meta["spread_warning"] = str(exc)

    raw["symbol"] = symbol.upper()
    raw = apply_spread_fallback(raw, symbol, cfg)
    feat = build_feature_frame(raw, cfg, symbol=symbol)
    meta.update({
        "feature_rows": len(feat),
        **_time_range_meta(feat, "feature"),
        "live_spread_points": live_spread,
        "live_spread_source": live_spread_source,
    })
    if not raw.empty and not feat.empty and "time_utc" in raw.columns and "time_utc" in feat.columns:
        raw_last = pd.to_datetime(raw["time_utc"], utc=True, errors="coerce").dropna().max()
        feat_last = pd.to_datetime(feat["time_utc"], utc=True, errors="coerce").dropna().max()
        if pd.notna(raw_last) and pd.notna(feat_last):
            meta["feature_lag_bars"] = int(((raw_last - feat_last).total_seconds()) // 300) if timeframe == "M5" else None
            meta["feature_lag_seconds"] = float((raw_last - feat_last).total_seconds())
    return feat, meta
