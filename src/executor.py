from __future__ import annotations

from typing import Any

from .forex import pip_size, symbol_cfg_value
from .mt5_client import ensure_symbol_available, get_symbol_info, get_tick, initialize_from_config, require_mt5


def _filling_mode(mt5, text: str):
    text = str(text or "IOC").upper()
    if text == "FOK":
        return mt5.ORDER_FILLING_FOK
    if text == "RETURN":
        return mt5.ORDER_FILLING_RETURN
    return mt5.ORDER_FILLING_IOC


def calculate_lot(symbol: str, cfg: dict[str, Any]) -> float:
    risk = cfg.get("risk", {}) or {}
    execution = cfg.get("execution", {}) or {}
    base = risk.get("fixed_lot", execution.get("lot_size", 0.01))
    lot = max(0.0, float(base))
    min_lot = float(execution.get("min_lot", 0.01))
    max_lot = float(execution.get("max_lot", 100.0))
    step = float(execution.get("lot_step", 0.01))
    lot = min(max(lot, min_lot), max_lot)
    if step > 0:
        lot = round(round(lot / step) * step, 2)
    return lot


def build_order_request(symbol: str, direction: str, cfg: dict[str, Any]) -> dict[str, Any]:
    mt5 = require_mt5()
    initialize_from_config(cfg)
    ensure_symbol_available(symbol)
    tick = get_tick(symbol, cfg)
    info = get_symbol_info(symbol, cfg)

    orders = cfg.get("orders", {}) or {}
    execution = cfg.get("execution", {}) or {}
    pip = pip_size(symbol, cfg)
    volume = calculate_lot(symbol, cfg)

    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = float(tick.ask)
        sl = price - float(symbol_cfg_value(cfg, "orders", "stop_loss_pips", symbol, symbol_cfg_value(cfg, "labels", "stop_loss_pips", symbol, 5.0))) * pip
        tp = price + float(symbol_cfg_value(cfg, "orders", "take_profit_pips", symbol, symbol_cfg_value(cfg, "labels", "take_profit_pips", symbol, 6.0))) * pip
    elif direction == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)
        sl = price + float(symbol_cfg_value(cfg, "orders", "stop_loss_pips", symbol, symbol_cfg_value(cfg, "labels", "stop_loss_pips", symbol, 5.0))) * pip
        tp = price - float(symbol_cfg_value(cfg, "orders", "take_profit_pips", symbol, symbol_cfg_value(cfg, "labels", "take_profit_pips", symbol, 6.0))) * pip
    else:
        raise ValueError("direction must be BUY or SELL")

    digits = int(getattr(info, "digits", 5))
    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": round(price, digits),
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": int(execution.get("deviation", execution.get("deviation_points", 20))),
        "magic": int(execution.get("magic", (cfg.get("trading", {}) or {}).get("magic_number", 550120))),
        "comment": str(execution.get("comment", (cfg.get("trading", {}) or {}).get("comment", "direction_policy"))),
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(mt5, execution.get("order_filling", "IOC")),
    }


def send_order(symbol: str, direction: str, cfg: dict[str, Any]):
    mt5 = require_mt5()
    request = build_order_request(symbol, direction, cfg)
    return mt5.order_send(request)
