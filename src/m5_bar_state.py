from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def closed_m5_bars_only(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Remove the currently forming MT5 M5 candle unless disabled."""
    mcfg = cfg.get("m5_execution", {}) or {}
    if not bool(mcfg.get("use_closed_candle_only", mcfg.get("use_closed_candle", True))):
        return df
    if len(df) <= 1:
        return df
    return df.iloc[:-1].copy().reset_index(drop=True)


def _state_file(symbol: str, cfg: dict[str, Any] | None = None) -> Path:
    base = Path("logs") / "m5_bar_state"
    if cfg is not None:
        base = Path((cfg.get("m5_execution", {}) or {}).get("state_dir", base))
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{symbol}_last_processed_bar.txt"


def get_last_processed_m5_bar(symbol: str, cfg: dict[str, Any] | None = None) -> str | None:
    path = _state_file(symbol, cfg)
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def set_last_processed_m5_bar(symbol: str, bar_time: str, cfg: dict[str, Any] | None = None) -> None:
    _state_file(symbol, cfg).write_text(str(bar_time), encoding="utf-8")


def m5_bar_gate(symbol: str, bar_time: str, cfg: dict[str, Any]) -> tuple[bool, str]:
    mcfg = cfg.get("m5_execution", {}) or {}
    gate_enabled = bool(mcfg.get("one_decision_per_symbol_per_m5_bar", True) or mcfg.get("one_trade_per_symbol_per_m5_bar", False))
    if not gate_enabled:
        return True, "M5 bar gate disabled"
    last = get_last_processed_m5_bar(symbol, cfg)
    if last == str(bar_time):
        return False, f"{symbol}: already processed closed M5 bar {bar_time}"
    return True, "New closed M5 bar"


def mark_m5_bar_processed(symbol: str, bar_time: str, cfg: dict[str, Any]) -> None:
    mcfg = cfg.get("m5_execution", {}) or {}
    if bool(mcfg.get("one_decision_per_symbol_per_m5_bar", True) or mcfg.get("one_trade_per_symbol_per_m5_bar", False)):
        set_last_processed_m5_bar(symbol, str(bar_time), cfg)
