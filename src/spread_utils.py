from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FALLBACK_MIN_SPREAD_POINTS: dict[str, float] = {
    "US500": 40.0,
    "NAS100": 100.0,
    "GER40": 100.0,
    "XAUUSD": 10.0,
    "XAGUSD": 10.0,
}


def get_spread_control(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Return spread-control settings with safe defaults.

    MT5 history can return zero, negative or missing spreads. Letting those values
    into target generation or dry-run/live reasoning makes results too optimistic,
    so this mirrors the old project's effective-spread fallback behaviour.
    """
    cfg = cfg or {}
    scfg = dict(cfg.get("spread_control", {}) or {})
    scfg.setdefault("enabled", True)
    scfg.setdefault("treat_zero_as_missing", True)
    scfg.setdefault("fallback_min_spread_points", DEFAULT_FALLBACK_MIN_SPREAD_POINTS.copy())
    scfg.setdefault("default_fallback_min_spread_points", 10.0)
    return scfg


def fallback_min_spread_points(symbol: str, cfg: dict[str, Any] | None = None) -> float:
    scfg = get_spread_control(cfg)
    fallback = scfg.get("fallback_min_spread_points", {}) or {}
    symbol_u = str(symbol).upper()
    if symbol_u in fallback:
        return float(fallback[symbol_u])
    return float(scfg.get("default_fallback_min_spread_points", DEFAULT_FALLBACK_MIN_SPREAD_POINTS.get(symbol_u, 10.0)))


def normalise_spread_points(spread_points: Any, *, symbol: str, cfg: dict[str, Any] | None = None) -> tuple[float, str]:
    """Return an effective non-negative spread in MT5 points plus source label."""
    scfg = get_spread_control(cfg)
    enabled = bool(scfg.get("enabled", True))
    treat_zero = bool(scfg.get("treat_zero_as_missing", True))
    try:
        value = float(spread_points)
    except Exception:
        value = float("nan")

    if not enabled:
        if not np.isfinite(value) or value < 0:
            return 0.0, "invalid_zero"
        return value, "raw_spread"

    missing = (not np.isfinite(value)) or value < 0 or (treat_zero and value <= 0)
    if missing:
        return fallback_min_spread_points(symbol, cfg), "fallback_min"
    return value, "raw_spread"


def infer_symbol_from_frame(df: pd.DataFrame, default: str | None = None) -> str:
    if "symbol" in df.columns and len(df) > 0:
        vals = df["symbol"].dropna().astype(str).str.upper().unique()
        if len(vals) == 1:
            return str(vals[0])
    return str(default or "").upper()


def apply_spread_fallback(
    df: pd.DataFrame,
    symbol: str,
    cfg: dict[str, Any] | None = None,
    *,
    spread_column: str = "spread",
    output_column: str = "spread_points",
) -> pd.DataFrame:
    """Apply effective-spread fallback to every row.

    `spread_points` is kept as float for target generation/logging, while `spread`
    remains an integer-style MT5 compatibility column.
    """
    out = df.copy()
    symbol_u = str(symbol).upper()
    if spread_column in out.columns:
        raw = pd.to_numeric(out[spread_column], errors="coerce")
    elif output_column in out.columns:
        raw = pd.to_numeric(out[output_column], errors="coerce")
    else:
        raw = pd.Series(np.nan, index=out.index, dtype="float64")

    values: list[float] = []
    sources: list[str] = []
    for v in raw.to_numpy():
        sp, src = normalise_spread_points(v, symbol=symbol_u, cfg=cfg)
        values.append(float(sp))
        sources.append(src)

    out[output_column] = pd.Series(values, index=out.index, dtype="float64")
    out["spread_source"] = pd.Series(sources, index=out.index, dtype="object")
    out[spread_column] = np.rint(out[output_column].to_numpy(dtype="float64")).astype("int64")
    return out


def effective_spread_points_from_row(row: pd.Series | dict[str, Any], symbol: str, cfg: dict[str, Any] | None = None) -> float:
    raw = row.get("spread_points", row.get("spread", np.nan))
    return float(normalise_spread_points(raw, symbol=symbol, cfg=cfg)[0])
