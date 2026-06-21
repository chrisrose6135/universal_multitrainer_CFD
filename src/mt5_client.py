from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # MetaTrader5 is Windows-only and optional until MT5 commands are used.
    mt5 = None

from .forex import validate_forex_symbols
from .spread_utils import normalise_spread_points


class MT5Error(RuntimeError):
    pass


def require_mt5():
    if mt5 is None:
        raise MT5Error(
            "MetaTrader5 is not installed or is unavailable on this platform. "
            "Install/run this part on the Windows MT5 machine with: pip install MetaTrader5"
        )
    return mt5


def initialize_mt5(login: int | None = None, password: str | None = None, server: str | None = None, path: str | None = None) -> None:
    mt5_mod = require_mt5()
    kwargs: dict[str, Any] = {}
    if login is not None:
        kwargs["login"] = int(login)
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server
    if path:
        kwargs["path"] = path
    ok = mt5_mod.initialize(**kwargs) if kwargs else mt5_mod.initialize()
    if not ok:
        raise MT5Error(f"MT5 initialize failed: {mt5_mod.last_error()}")


def initialize_from_config(cfg: dict[str, Any] | None = None) -> None:
    mt5_cfg = (cfg or {}).get("mt5", {}) or {}
    initialize_mt5(
        login=mt5_cfg.get("login"),
        password=mt5_cfg.get("password"),
        server=mt5_cfg.get("server"),
        path=mt5_cfg.get("terminal_path"),
    )


def shutdown_mt5() -> None:
    if mt5 is not None:
        try:
            mt5.shutdown()
        except Exception:
            pass


def timeframe_mt5(timeframe: str = "M5"):
    mt5_mod = require_mt5()
    tf = str(timeframe or "M5").upper()
    attr = f"TIMEFRAME_{tf}"
    if not hasattr(mt5_mod, attr):
        raise MT5Error(f"Unsupported MT5 timeframe {tf!r}; MetaTrader5 has no {attr}")
    return getattr(mt5_mod, attr)


def utc_date(date_text: str | datetime) -> datetime:
    if isinstance(date_text, datetime):
        dt = date_text
    else:
        dt = datetime.fromisoformat(str(date_text).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ensure_symbol_available(symbol: str) -> None:
    mt5_mod = require_mt5()
    info = mt5_mod.symbol_info(symbol)
    if info is None:
        raise MT5Error(f"Symbol not found in MT5 Market Watch/broker feed: {symbol}")
    if not info.visible and not mt5_mod.symbol_select(symbol, True):
        raise MT5Error(f"Could not select symbol in MT5 Market Watch: {symbol}")


def _mt5_time_cfg(cfg: dict[str, Any] | None = None) -> tuple[bool, float]:
    """Return whether MT5 bar timestamps should be shifted from broker/server time to UTC.

    MetaTrader5 Python normally returns timestamps as Unix epoch seconds, which should be
    interpreted as UTC. Some broker/terminal combinations can still appear broker-time
    shifted in downstream logs. For that reason the offset is OFF by default and must be
    explicitly enabled in config after running the diagnostic.

    Config keys:
        mt5.apply_broker_server_timezone_offset: bool
        mt5.convert_broker_server_time_to_utc: bool  # backwards-compatible alias
        mt5.broker_server_utc_offset_hours: float
    """
    mt5_cfg = (cfg or {}).get("mt5", {}) or {}
    apply = bool(
        mt5_cfg.get(
            "apply_broker_server_timezone_offset",
            mt5_cfg.get("convert_broker_server_time_to_utc", False),
        )
    )
    try:
        offset_hours = float(mt5_cfg.get("broker_server_utc_offset_hours", 0.0) or 0.0)
    except (TypeError, ValueError):
        offset_hours = 0.0
    return apply, offset_hours


def _apply_mt5_time_policy(reported_time_utc: pd.Series, cfg: dict[str, Any] | None = None) -> tuple[pd.Series, bool, float]:
    """Apply optional broker/server-time correction to MT5 timestamps.

    ``reported_time_utc`` is the direct conversion of MT5's epoch seconds using
    ``pd.to_datetime(..., unit="s", utc=True)``. If the configured correction is enabled,
    the broker offset is subtracted to produce canonical UTC for all downstream feature
    generation.
    """
    apply_offset, offset_hours = _mt5_time_cfg(cfg)
    if apply_offset and offset_hours:
        corrected = reported_time_utc - pd.to_timedelta(offset_hours, unit="h")
        return corrected, True, float(offset_hours)
    return reported_time_utc, False, 0.0


def _rates_to_frame(rates: Any, symbol: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)

    # Direct interpretation of MT5's raw epoch seconds.
    reported_time_utc = pd.to_datetime(df["time"], unit="s", utc=True)
    canonical_time_utc, adjustment_applied, offset_hours = _apply_mt5_time_policy(reported_time_utc, cfg)

    # Keep both timestamps so live/preflight/dashboard diagnostics can show whether
    # a broker/server-time correction was actually applied. Downstream feature code
    # should use ``time``/``time_utc`` only.
    df["time_broker"] = reported_time_utc
    df["time_reported"] = reported_time_utc
    df["time"] = canonical_time_utc
    df["time_utc"] = canonical_time_utc
    df["mt5_time_adjustment_applied"] = bool(adjustment_applied)
    df["broker_server_utc_offset_hours"] = float(offset_hours)
    df["symbol"] = symbol.upper()
    return df


def get_rates(symbol: str, start_utc: datetime, end_utc: datetime, timeframe: str = "M5", cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    validate_forex_symbols([symbol])
    initialize_from_config(cfg)
    ensure_symbol_available(symbol)
    mt5_mod = require_mt5()
    tf = str(timeframe or "M5").upper()
    rates = mt5_mod.copy_rates_range(symbol, timeframe_mt5(tf), start_utc, end_utc)
    df = _rates_to_frame(rates, symbol, cfg=cfg)
    if df.empty:
        raise MT5Error(f"No {tf} rates returned for {symbol}: {mt5_mod.last_error()}")
    return df


def latest_rates(symbol: str, count: int = 500, timeframe: str = "M5", cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    validate_forex_symbols([symbol])
    initialize_from_config(cfg)
    ensure_symbol_available(symbol)
    mt5_mod = require_mt5()
    tf = str(timeframe or "M5").upper()
    rates = mt5_mod.copy_rates_from_pos(symbol, timeframe_mt5(tf), 0, int(count))
    df = _rates_to_frame(rates, symbol, cfg=cfg)
    if df.empty:
        raise MT5Error(f"No latest {tf} rates returned for {symbol}: {mt5_mod.last_error()}")
    return df


def get_m5_rates(symbol: str, start_utc: datetime, end_utc: datetime, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    return get_rates(symbol, start_utc, end_utc, timeframe="M5", cfg=cfg)


def latest_m5_rates(symbol: str, count: int = 500, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    return latest_rates(symbol, count=count, timeframe="M5", cfg=cfg)


def diagnose_mt5_time(symbol: str, count: int = 10, timeframe: str = "M5", cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return diagnostics to decide whether the broker/server offset should be enabled.

    Run this on the MT5 machine. The key check is whether the newest MT5-reported bar
    timestamp is close to the current UTC time. For M5, a latest bar within roughly
    +/-10 minutes of current UTC is normally fine. If it is about +120 minutes for a
    GMT+2 broker, then enable ``mt5.apply_broker_server_timezone_offset``.

    This function does not place trades and does not modify config.
    """
    validate_forex_symbols([symbol])
    initialize_from_config(cfg)
    ensure_symbol_available(symbol)

    mt5_mod = require_mt5()
    tf = str(timeframe or "M5").upper()
    rates = mt5_mod.copy_rates_from_pos(symbol, timeframe_mt5(tf), 0, int(count))
    if rates is None or len(rates) == 0:
        raise MT5Error(f"No latest {tf} rates returned for {symbol}: {mt5_mod.last_error()}")

    raw = pd.DataFrame(rates)
    reported = pd.to_datetime(raw["time"], unit="s", utc=True)
    canonical, adjustment_applied, offset_hours = _apply_mt5_time_policy(reported, cfg)

    now_utc = pd.Timestamp.now(tz="UTC")
    newest_reported = reported.max()
    newest_canonical = canonical.max()

    tick = mt5_mod.symbol_info_tick(symbol)
    tick_time_utc = None
    tick_time_msc_utc = None
    if tick is not None:
        tick_time = getattr(tick, "time", None)
        tick_time_msc = getattr(tick, "time_msc", None)
        if tick_time:
            tick_time_utc = pd.to_datetime(tick_time, unit="s", utc=True)
        if tick_time_msc:
            tick_time_msc_utc = pd.to_datetime(tick_time_msc, unit="ms", utc=True)

    newest_reported_minus_now_min = float((newest_reported - now_utc).total_seconds() / 60.0)
    newest_canonical_minus_now_min = float((newest_canonical - now_utc).total_seconds() / 60.0)

    recommendation = "leave_offset_disabled"
    if newest_reported_minus_now_min > 30:
        recommendation = "reported_bar_time_is_in_future_check_broker_offset"
    if adjustment_applied and abs(newest_canonical_minus_now_min) <= 15:
        recommendation = "offset_enabled_and_canonical_time_looks_plausible"
    elif adjustment_applied and abs(newest_canonical_minus_now_min) > 30:
        recommendation = "offset_enabled_but_canonical_time_still_suspicious"

    return {
        "symbol": symbol.upper(),
        "timeframe": tf,
        "count": int(count),
        "now_utc": now_utc.isoformat(),
        "newest_reported_time_utc": newest_reported.isoformat(),
        "newest_canonical_time_utc": newest_canonical.isoformat(),
        "newest_reported_minus_now_minutes": newest_reported_minus_now_min,
        "newest_canonical_minus_now_minutes": newest_canonical_minus_now_min,
        "mt5_time_adjustment_applied": bool(adjustment_applied),
        "broker_server_utc_offset_hours": float(offset_hours),
        "tick_time_utc": tick_time_utc.isoformat() if tick_time_utc is not None else None,
        "tick_time_msc_utc": tick_time_msc_utc.isoformat() if tick_time_msc_utc is not None else None,
        "oldest_reported_time_utc": reported.min().isoformat(),
        "oldest_canonical_time_utc": canonical.min().isoformat(),
        "recommendation": recommendation,
    }


def get_tick(symbol: str, cfg: dict[str, Any] | None = None):
    validate_forex_symbols([symbol])
    initialize_from_config(cfg)
    ensure_symbol_available(symbol)
    tick = require_mt5().symbol_info_tick(symbol)
    if tick is None:
        raise MT5Error(f"No tick for symbol {symbol}: {require_mt5().last_error()}")
    return tick


def get_symbol_info(symbol: str, cfg: dict[str, Any] | None = None):
    validate_forex_symbols([symbol])
    initialize_from_config(cfg)
    ensure_symbol_available(symbol)
    info = require_mt5().symbol_info(symbol)
    if info is None:
        raise MT5Error(f"No symbol info for {symbol}: {require_mt5().last_error()}")
    return info


def live_spread_points(symbol: str, cfg: dict[str, Any] | None = None) -> tuple[float, str]:
    mt5_mod = require_mt5()
    initialize_from_config(cfg)
    ensure_symbol_available(symbol)
    info = mt5_mod.symbol_info(symbol)
    tick = mt5_mod.symbol_info_tick(symbol)
    point = float(getattr(info, "point", 0.0) or 0.0) if info is not None else 0.0
    if tick is not None and point > 0:
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if ask > 0 and bid > 0 and ask >= bid:
            sp, src = normalise_spread_points((ask - bid) / point, symbol=symbol, cfg=cfg)
            return sp, "tick_bid_ask" if src == "raw_spread" else "fallback_min_from_tick"
    if info is not None:
        sp, src = normalise_spread_points(getattr(info, "spread", 0.0), symbol=symbol, cfg=cfg)
        return sp, "symbol_info_spread" if src == "raw_spread" else "fallback_min_from_symbol_info"
    sp, _ = normalise_spread_points(None, symbol=symbol, cfg=cfg)
    return sp, "fallback_min_unavailable"
