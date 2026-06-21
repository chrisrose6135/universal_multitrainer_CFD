from __future__ import annotations

from typing import Any

import pandas as pd

from .spread_utils import fallback_min_spread_points, normalise_spread_points


def read_live_spread_points(mt5_module: Any, symbol: str, cfg: dict[str, Any] | None = None) -> tuple[float, str]:
    info = mt5_module.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"MT5 symbol_info({symbol!r}) returned None")
    point = float(getattr(info, "point", 0.0) or 0.0)
    tick = mt5_module.symbol_info_tick(symbol)
    if tick is not None and point > 0:
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid > 0 and ask > 0 and ask >= bid:
            sp, src = normalise_spread_points((ask - bid) / point, symbol=symbol, cfg=cfg)
            return sp, "tick_bid_ask" if src == "raw_spread" else "fallback_min_from_zero_tick_bid_ask"
    raw_spread = getattr(info, "spread", None)
    if raw_spread is not None:
        sp, src = normalise_spread_points(raw_spread, symbol=symbol, cfg=cfg)
        return sp, "symbol_info_spread" if src == "raw_spread" else "fallback_min_from_zero_symbol_info_spread"
    return fallback_min_spread_points(symbol, cfg), "fallback_min_unavailable"


def inject_live_spread_into_bars(df: pd.DataFrame, spread_points: float, source: str = "live") -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    last_idx = out.index[-1]
    sp = float(spread_points)
    if "spread_points" not in out.columns:
        out["spread_points"] = 0.0
    out["spread_points"] = out["spread_points"].astype("float64", copy=False)
    out.loc[last_idx, "spread_points"] = sp
    if "spread_source" not in out.columns:
        out["spread_source"] = ""
    out["spread_source"] = out["spread_source"].astype("object", copy=False)
    out.loc[last_idx, "spread_source"] = str(source)
    if "spread" in out.columns:
        try:
            out.loc[last_idx, "spread"] = int(round(sp))
        except Exception:
            out["spread"] = out["spread"].astype("float64", copy=False)
            out.loc[last_idx, "spread"] = sp
    else:
        out["spread"] = int(round(sp))
    return out
