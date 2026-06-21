#!/usr/bin/env python3
"""Live/demo runner for staged direction-policy model ensembles.

This script consumes the folder produced by select_live_trading_models.py. It can
run every staged model-symbol-side combination, or a filtered subset, and then
optionally places at most one order per symbol/bar using the strongest model
signal.

Place this file in src/ and run, for example:

    python -m src.live_direction_policy_ensemble \
      --live-root "For Live Trading" \
      --mode paper \
      --data-source mt5 \
      --poll-seconds 20

For demo/live trading, change --mode to demo or live. By default, the script
allows only one order per symbol per closed bar, even if several models fire.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
import yaml

from .config import load_config_with_optional_spread_risk
from .executor import send_order
from .external_trade_filter import external_trade_gate
from .forex import validate_forex_symbols
from .live_data import latest_processed_features
from .live_direction_policy import (
    DIRECTION_CLASS_NAMES,
    _account_daily_loss_gate,
    _append_csv,
    _cooldown_gate,
    _json_default,
    _latest_fixed_threshold_live_decision,
    _latest_rolling_quantile_live_decision,
    _live_feature_sequences,
    _live_requested_bars,
    _min_direction_probability,
    _min_edge_pips,
    _min_trade_probability,
    _predict_live_sequences,
    _record_cooldown_order,
    _symbol_trade_mode,
    _threshold_mode,
    _trade_mode_allows_side,
    _trade_mode_block_reason,
    _uses_rolling_quantile_thresholds,
    _utc_now_iso,
    load_direction_policy,
    sync_trade_logs_and_summary,
)
from .mt5_client import shutdown_mt5


def _read_structured(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _resolve_path(value: Any, *, live_root: Path, manifest_dir: Path) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    for candidate in (Path.cwd() / raw, live_root / raw, manifest_dir / raw):
        if candidate.exists():
            return candidate
    # Return a sensible cwd-relative path even if it does not exist yet.
    return Path.cwd() / raw


def _normalise_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"long", "buy_only"}:
        return "buy"
    if side in {"short", "sell_only"}:
        return "sell"
    return side


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base and return a new dict."""
    out = deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _side_specific_overlay(side_cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep only model/side-specific sections from a staged side config.

    The universal generic config is the source of shared live behaviour: risk,
    execution, spread limits, analytic/external gates, MT5 settings and feature
    engineering settings. This overlay keeps only the sections that identify the
    staged model, symbol, side and artifact paths.
    """
    out: dict[str, Any] = {}
    for key in ("project", "trading", "paths", "model", "live_model_selection", "tp_profit_protection_exit"):
        if isinstance(side_cfg.get(key), dict):
            out[key] = deepcopy(side_cfg[key])

    replay = side_cfg.get("replay")
    if isinstance(replay, dict):
        out["replay"] = {}
        for key in ("allow_buy", "allow_sell"):
            if key in replay:
                out["replay"][key] = deepcopy(replay[key])

    training = side_cfg.get("training")
    if isinstance(training, dict):
        allowed_training = {
            "train_side",
            "side_setup_train_side",
            "buy_setup_loss_weight",
            "sell_setup_loss_weight",
        }
        kept = {k: deepcopy(v) for k, v in training.items() if k in allowed_training}
        if kept:
            out["training"] = kept

    if isinstance(side_cfg.get("symbol_trade_modes"), dict):
        out["symbol_trade_modes"] = deepcopy(side_cfg["symbol_trade_modes"])

    live = side_cfg.get("live_direction_policy")
    if isinstance(live, dict):
        allowed_live = {
            "model_path",
            "scaler_path",
            "features_path",
            "cooldown_state_json",
        }
        kept = {k: deepcopy(v) for k, v in live.items() if k in allowed_live}
        if kept:
            out["live_direction_policy"] = kept

    return out


def _resolve_universal_config_path(
    *,
    cli_value: str | None,
    manifest_payload: dict[str, Any],
    live_root: Path,
    manifest_dir: Path,
) -> Path | None:
    candidates: list[Path] = []
    for value in (cli_value, manifest_payload.get("universal_config_path")):
        if value:
            raw = Path(str(value))
            if raw.is_absolute():
                candidates.append(raw)
            else:
                candidates.extend([Path.cwd() / raw, live_root / raw, manifest_dir / raw])
    for name in (
        "direction_settings_generic_multisymbol_31_symbols.yaml",
        "direction_settings_generic_multisymbol_31_symbols.yml",
    ):
        candidates.extend([live_root / name, live_root / "config" / name, live_root / "configs" / name])
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _load_universal_config(path: Path | None, *, require: bool = True) -> dict[str, Any]:
    if path is None:
        if require:
            raise SystemExit(
                "No universal live config found. Copy direction_settings_generic_multisymbol_31_symbols.yaml "
                "into the live trading folder, or pass --universal-config."
            )
        return {}
    return load_config_with_optional_spread_risk(str(path))


def _ensure_live_defaults(cfg: dict[str, Any], *, live_root: Path) -> None:
    """Fill shared live log paths if the universal config did not define them."""
    live = cfg.setdefault("live_direction_policy", {})
    defaults = {
        "signals_csv": live_root / "logs" / "live_direction_ensemble_signals.csv",
        "trades_csv": live_root / "logs" / "live_direction_trades.csv",
        "summary_json": live_root / "logs" / "live_direction_summary.json",
        "open_trades_json": live_root / "logs" / "live_direction_open_trades.json",
    }
    for key, value in defaults.items():
        live.setdefault(key, str(value).replace("\\", "/"))
    live.setdefault("sync_trade_logs", True)


def _load_manifest(live_root: Path, manifest_path: Path | None = None) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    if manifest_path is None:
        for name in ("live_ensemble_manifest.json", "live_ensemble_manifest.yaml", "live_ensemble_manifest.yml"):
            p = live_root / name
            if p.exists():
                manifest_path = p
                break
    if manifest_path is None:
        configs = sorted((live_root / "configs").glob("*_live_ensemble.y*ml"))
        entries: list[dict[str, Any]] = []
        manifest_payload: dict[str, Any] = {}
        for cfg_path in configs:
            payload = _read_structured(cfg_path)
            if isinstance(payload, dict):
                manifest_payload.setdefault("universal_config_path", payload.get("universal_config_path"))
                for entry in payload.get("models") or []:
                    entries.append(dict(entry))
        if entries:
            return entries, live_root / "configs", manifest_payload
        raise SystemExit(f"No live ensemble manifest found under {live_root}")

    payload = _read_structured(manifest_path)
    if isinstance(payload, dict) and isinstance(payload.get("models"), list):
        return [dict(x) for x in payload["models"] if isinstance(x, dict)], manifest_path.parent, payload
    raise SystemExit(f"Manifest does not contain a models list: {manifest_path}")

def _filter_entries(entries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    symbols = {s.upper() for s in args.symbols} if args.symbols else None
    models = {m.lower() for m in args.models} if args.models else None
    sides = {_normalise_side(s) for s in args.sides} if args.sides else None
    out: list[dict[str, Any]] = []
    for e in entries:
        symbol = str(e.get("symbol") or "").upper()
        model = str(e.get("model") or "").lower()
        side = _normalise_side(e.get("side"))
        if symbols and symbol not in symbols:
            continue
        if models and model not in models:
            continue
        if sides and side not in sides:
            continue
        if not symbol or side not in {"buy", "sell"}:
            continue
        out.append(e)
    return out


def _signals_csv_path(args: argparse.Namespace, live_root: Path) -> Path:
    if args.signals_csv:
        return Path(args.signals_csv)
    return live_root / "logs" / "live_direction_ensemble_signals.csv"


def _bar_state_path(args: argparse.Namespace, live_root: Path) -> Path:
    if args.bar_state_json:
        return Path(args.bar_state_json)
    return live_root / "state" / "live_direction_ensemble_bar_state.json"


def _load_bar_state(args: argparse.Namespace, live_root: Path) -> dict[str, Any]:
    path = _bar_state_path(args, live_root)
    if not path.exists():
        return {"models": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"models": {}}
    except Exception:
        return {"models": {}}


def _save_bar_state(args: argparse.Namespace, live_root: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at_utc"] = _utc_now_iso()
    _write_json(_bar_state_path(args, live_root), payload)


def _combo_processed(state: dict[str, Any], combo_id: str, bar_time: str) -> bool:
    item = ((state.get("models") or {}).get(combo_id) or {})
    return bool(bar_time and item.get("last_bar_time") == bar_time)


def _mark_combo_processed(state: dict[str, Any], combo_id: str, row: dict[str, Any]) -> None:
    models = state.setdefault("models", {})
    models[combo_id] = {
        "last_bar_time": row.get("bar_time"),
        "last_seen_utc": _utc_now_iso(),
        "last_model_decision": row.get("model_decision"),
        "last_direction": row.get("direction"),
        "last_reason": row.get("reason"),
    }


def _load_combo(
    entry: dict[str, Any],
    *,
    live_root: Path,
    manifest_dir: Path,
    universal_cfg: dict[str, Any],
    universal_config_path: Path | None,
    device: str | None,
) -> dict[str, Any]:
    config_path = _resolve_path(entry["config_path"], live_root=live_root, manifest_dir=manifest_dir)
    side_cfg_full = load_config_with_optional_spread_risk(str(config_path))
    side_overlay = _side_specific_overlay(side_cfg_full)
    cfg = _deep_merge(universal_cfg, side_overlay)
    cfg["_universal_config_path"] = str(universal_config_path) if universal_config_path is not None else None
    cfg["_side_config_path"] = str(config_path)
    _ensure_live_defaults(cfg, live_root=live_root)

    # Resolve artifacts to absolute-ish paths for robust loading even when the
    # manifest was created on another machine or from a relative live root.
    live_cfg = cfg.setdefault("live_direction_policy", {})
    for key in ("model_path", "scaler_path", "features_path"):
        value = entry.get(key) or live_cfg.get(key)
        if value:
            live_cfg[key] = str(_resolve_path(value, live_root=live_root, manifest_dir=manifest_dir))

    symbol = str(entry.get("symbol") or ((cfg.get("trading") or {}).get("symbols") or [""])[0]).upper()
    # Force the staged combo symbol after merging so a multi-symbol universal
    # config cannot accidentally make a side model evaluate the wrong symbol.
    cfg.setdefault("trading", {})["symbols"] = [symbol]
    bundle = load_direction_policy(symbol, cfg, device)
    combo_id = str(entry.get("id") or f"{symbol}_{entry.get('model','model')}_{entry.get('side','side')}_epoch_{entry.get('epoch','unknown')}")
    return {
        "id": combo_id,
        "entry": entry,
        "symbol": symbol,
        "model": str(entry.get("model") or "model"),
        "side": _normalise_side(entry.get("side")),
        "epoch": entry.get("epoch"),
        "config_path": str(config_path),
        "universal_config_path": str(universal_config_path) if universal_config_path is not None else None,
        "cfg": cfg,
        "bundle": bundle,
    }

def _as_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _signal_rank_score(row: dict[str, Any]) -> float:
    side = str(row.get("direction") or "").upper()
    margin = row.get("buy_threshold_margin") if side == "BUY" else row.get("sell_threshold_margin") if side == "SELL" else None
    margin_f = _as_float_or_none(margin)
    prob = _as_float_or_none(row.get("selected_probability")) or 0.0
    deployment = _as_float_or_none(row.get("deployment_score")) or 0.0
    avg = _as_float_or_none(row.get("selected_model_average_net_pips")) or 0.0
    if margin_f is not None:
        return float(margin_f * 100.0 + prob + deployment * 0.001 + avg)
    return float(prob + deployment * 0.001 + avg)


def _tp_profit_protection_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the TP-relative profit-protection config block.

    The preferred location is a top-level ``tp_profit_protection_exit`` block in
    the universal live config. A nested ``live_direction_policy`` block is also
    accepted for backwards-compatible live-only configuration.
    """
    block = cfg.get("tp_profit_protection_exit")
    if isinstance(block, dict):
        return block
    live_block = (cfg.get("live_direction_policy") or {}).get("tp_profit_protection_exit")
    if isinstance(live_block, dict):
        return live_block
    return {}


def _tp_profit_protection_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_tp_profit_protection_cfg(cfg).get("enabled", False))


def _get_mt5_module():
    """Import MetaTrader5 lazily so paper mode can still run without MT5."""
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on deployment host
        raise RuntimeError(f"MetaTrader5 package is required for live position management: {exc}") from exc
    return mt5


def _mt5_last_error(mt5: Any) -> Any:
    try:
        return mt5.last_error()
    except Exception:
        return None


def _mt5_order_filling(mt5: Any, cfg: dict[str, Any]) -> Any | None:
    raw = str(((cfg.get("execution") or {}).get("order_filling") or "IOC")).strip().upper()
    mapping = {
        "FOK": getattr(mt5, "ORDER_FILLING_FOK", None),
        "IOC": getattr(mt5, "ORDER_FILLING_IOC", None),
        "RETURN": getattr(mt5, "ORDER_FILLING_RETURN", None),
    }
    return mapping.get(raw)


def _position_side(mt5: Any, position: Any) -> str:
    ptype = int(getattr(position, "type", -1))
    if ptype == int(getattr(mt5, "POSITION_TYPE_BUY")):
        return "BUY"
    if ptype == int(getattr(mt5, "POSITION_TYPE_SELL")):
        return "SELL"
    return "UNKNOWN"


def _position_matches_management_filter(
    position: Any,
    *,
    cfg: dict[str, Any],
    managed_symbols: set[str],
) -> tuple[bool, str]:
    """Only manage positions that look like they belong to this bot.

    By default this requires the position symbol to be part of the loaded
    ensemble and the position magic number to match ``execution.magic``. This is
    deliberately conservative so the script does not close manual trades.
    """
    cfg_block = _tp_profit_protection_cfg(cfg)
    symbol = str(getattr(position, "symbol", "")).upper()
    if symbol not in managed_symbols:
        return False, "symbol_not_in_loaded_ensemble"

    manage_only_matching_magic = bool(cfg_block.get("manage_only_matching_magic", True))
    require_magic_when_configured = bool(cfg_block.get("require_magic_when_configured", True))
    configured_magic = (cfg.get("execution") or {}).get("magic")
    position_magic = getattr(position, "magic", None)

    if manage_only_matching_magic and configured_magic not in (None, ""):
        try:
            if int(position_magic) != int(configured_magic):
                return False, f"magic_mismatch:{position_magic}!={configured_magic}"
        except Exception:
            return False, f"magic_mismatch:{position_magic}!={configured_magic}"
    elif manage_only_matching_magic and require_magic_when_configured:
        return False, "management_requires_execution_magic"

    return True, "ok"


def _expected_profit_at_position_tp(mt5: Any, position: Any) -> float | None:
    """Expected account-currency profit if the open position closes at its TP."""
    symbol = str(getattr(position, "symbol", ""))
    volume = _as_float_or_none(getattr(position, "volume", None))
    open_price = _as_float_or_none(getattr(position, "price_open", None))
    tp_price = _as_float_or_none(getattr(position, "tp", None))
    if not symbol or volume is None or open_price is None or tp_price is None:
        return None
    if volume <= 0 or open_price <= 0 or tp_price <= 0:
        return None

    side = _position_side(mt5, position)
    if side == "BUY":
        order_type = getattr(mt5, "ORDER_TYPE_BUY")
    elif side == "SELL":
        order_type = getattr(mt5, "ORDER_TYPE_SELL")
    else:
        return None

    profit = mt5.order_calc_profit(order_type, symbol, float(volume), float(open_price), float(tp_price))
    profit_f = _as_float_or_none(profit)
    if profit_f is None or profit_f <= 0:
        return None
    return float(profit_f)


def _latest_completed_bar_time_for_symbol(symbol: str, cfg: dict[str, Any]) -> str:
    """Return the latest processed bar timestamp for candle-based exit state."""
    bars = int((_tp_profit_protection_cfg(cfg).get("bar_time_lookup_bars") or 300))
    try:
        feat, _ = latest_processed_features(symbol, cfg, bars=bars)
        if len(feat) == 0:
            return ""
        latest = feat.iloc[-1]
        return str(latest.get("time_utc") or latest.get("time") or "")
    except Exception as exc:
        return f"bar_time_error:{exc}"


def _close_mt5_position(mt5: Any, position: Any, cfg: dict[str, Any], *, reason: str) -> Any:
    """Close one MT5 position by sending the opposite market deal."""
    symbol = str(getattr(position, "symbol", ""))
    volume = float(getattr(position, "volume", 0.0) or 0.0)
    ticket = int(getattr(position, "ticket", 0) or 0)
    if not symbol or volume <= 0 or ticket <= 0:
        raise RuntimeError(f"Invalid position for close: symbol={symbol!r} volume={volume} ticket={ticket}")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No MT5 tick available for {symbol}; last_error={_mt5_last_error(mt5)}")

    side = _position_side(mt5, position)
    if side == "BUY":
        close_type = getattr(mt5, "ORDER_TYPE_SELL")
        price = float(tick.bid)
    elif side == "SELL":
        close_type = getattr(mt5, "ORDER_TYPE_BUY")
        price = float(tick.ask)
    else:
        raise RuntimeError(f"Unsupported MT5 position type for ticket {ticket}: {getattr(position, 'type', None)}")

    execution = cfg.get("execution") or {}
    request: dict[str, Any] = {
        "action": getattr(mt5, "TRADE_ACTION_DEAL"),
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": int(execution.get("deviation", 20) or 20),
        "comment": str(execution.get("close_comment") or reason)[:31],
        "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
    }
    if execution.get("magic") not in (None, ""):
        request["magic"] = int(execution.get("magic"))
    filling = _mt5_order_filling(mt5, cfg)
    if filling is not None:
        request["type_filling"] = filling

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"MT5 close order_send returned None; last_error={_mt5_last_error(mt5)} request={request}")
    retcode = getattr(result, "retcode", None)
    success_codes = {
        getattr(mt5, "TRADE_RETCODE_DONE", None),
        getattr(mt5, "TRADE_RETCODE_PLACED", None),
        getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", None),
    }
    if retcode not in success_codes:
        raise RuntimeError(f"MT5 close failed retcode={retcode} result={result} request={request}")
    return result


def _append_position_manager_row(rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    row = {
        "time_utc": _utc_now_iso(),
        "symbol": payload.get("symbol"),
        "model": "position_manager",
        "side_model": "tp_profit_protection_exit",
        "epoch": None,
        "combo_id": f"position_manager:{payload.get('symbol')}:{payload.get('ticket')}",
        "mode": payload.get("mode"),
        "data_source": payload.get("data_source"),
        "bar_time": payload.get("bar_time"),
        "model_decision": payload.get("decision", "POSITION_MANAGEMENT"),
        "final_decision": payload.get("decision", "POSITION_MANAGEMENT"),
        "reason": payload.get("reason", ""),
        "direction": payload.get("side", ""),
        "selected_probability": 0.0,
        "order_selected_by_ensemble": False,
        "order_attempted": bool(payload.get("close_attempted", False)),
        "order_sent": bool(payload.get("close_sent", False)),
        "order_result": payload.get("close_result"),
        "order_error": payload.get("close_error"),
        "daily_loss_allows_order": True,
        "daily_loss_gate": {},
        "ticket": payload.get("ticket"),
        "current_profit": payload.get("current_profit"),
        "tp_profit": payload.get("tp_profit"),
        "trigger_profit": payload.get("trigger_profit"),
        "giveback_close_profit": payload.get("giveback_close_profit"),
        "candles_at_or_above_trigger": payload.get("candles_at_or_above_trigger"),
        "reached_trigger_profit": payload.get("reached_trigger_profit"),
    }
    rows.append(row)


def _manage_tp_profit_protection_exits(
    combos: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    mode: str,
    data_source: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Apply TP-relative active-position exit rules before new signals/orders.

    Rules implemented:
      1. If floating profit is >= 50% of expected TP profit for more than 10
         completed candles, close the position.
      2. Once floating profit has reached 50% of expected TP profit, close the
         position if it later falls to <= 30% of expected TP profit.

    The exact fractions/candle count are configurable. State is stored under the
    existing bar-state JSON so it survives process restarts.
    """
    rows: list[dict[str, Any]] = []
    closed_symbols: set[str] = set()
    if mode not in {"demo", "live"} or not combos:
        return rows, closed_symbols

    cfg_by_symbol: dict[str, dict[str, Any]] = {}
    for combo in combos:
        cfg_by_symbol.setdefault(str(combo.get("symbol") or "").upper(), combo["cfg"])
    managed_symbols = set(cfg_by_symbol)
    enabled_symbols = {symbol for symbol, cfg in cfg_by_symbol.items() if _tp_profit_protection_enabled(cfg)}
    if not enabled_symbols:
        return rows, closed_symbols

    mt5 = _get_mt5_module()
    positions = mt5.positions_get()
    if positions is None:
        _append_position_manager_row(rows, {
            "mode": mode,
            "data_source": data_source,
            "decision": "ERROR",
            "reason": f"positions_get_failed:{_mt5_last_error(mt5)}",
        })
        return rows, closed_symbols

    state_root = state.setdefault("tp_profit_protection_exit", {})
    position_states = state_root.setdefault("positions", {})
    open_tickets = {str(getattr(pos, "ticket", "")) for pos in positions}

    # Prune state for positions that are no longer open.
    for ticket in list(position_states.keys()):
        if ticket not in open_tickets:
            position_states.pop(ticket, None)

    for position in positions:
        ticket = str(getattr(position, "ticket", ""))
        symbol = str(getattr(position, "symbol", "")).upper()
        if symbol not in enabled_symbols:
            continue
        cfg = cfg_by_symbol[symbol]
        cfg_block = _tp_profit_protection_cfg(cfg)

        allowed, filter_reason = _position_matches_management_filter(position, cfg=cfg, managed_symbols=managed_symbols)
        if not allowed:
            if bool(cfg_block.get("log_filtered_positions", False)):
                _append_position_manager_row(rows, {
                    "mode": mode,
                    "data_source": data_source,
                    "symbol": symbol,
                    "ticket": ticket,
                    "side": _position_side(mt5, position),
                    "decision": "SKIP",
                    "reason": filter_reason,
                })
            continue

        tp_profit = _expected_profit_at_position_tp(mt5, position)
        if tp_profit is None:
            if bool(cfg_block.get("log_skipped_positions", True)):
                _append_position_manager_row(rows, {
                    "mode": mode,
                    "data_source": data_source,
                    "symbol": symbol,
                    "ticket": ticket,
                    "side": _position_side(mt5, position),
                    "decision": "SKIP",
                    "reason": "tp_profit_protection_skipped_no_valid_tp",
                })
            continue

        trigger_fraction = float(cfg_block.get("trigger_fraction_of_tp_profit", 0.50))
        sustained_candles = int(cfg_block.get("sustained_profit_candles", 10))
        giveback_close_fraction = float(cfg_block.get("giveback_close_fraction_of_tp_profit", 0.30))
        reentry_block_after_close = bool(cfg_block.get("block_new_entry_same_cycle_after_close", True))
        trigger_profit = trigger_fraction * tp_profit
        giveback_close_profit = giveback_close_fraction * tp_profit
        current_profit = float(getattr(position, "profit", 0.0) or 0.0)
        side = _position_side(mt5, position)
        bar_time = _latest_completed_bar_time_for_symbol(symbol, cfg)

        item = position_states.setdefault(ticket, {})
        identity = {
            "symbol": symbol,
            "side": side,
            "price_open": _as_float_or_none(getattr(position, "price_open", None)),
            "volume": _as_float_or_none(getattr(position, "volume", None)),
        }
        # If a broker reuses a ticket or the state is malformed, reset it.
        if item.get("symbol") and any(item.get(k) != v for k, v in identity.items() if v is not None):
            item.clear()
        item.update(identity)
        item["ticket"] = ticket
        item.setdefault("first_seen_utc", _utc_now_iso())
        item["last_seen_utc"] = _utc_now_iso()
        item["tp"] = _as_float_or_none(getattr(position, "tp", None))
        item["tp_profit"] = tp_profit
        item["trigger_profit"] = trigger_profit
        item["giveback_close_profit"] = giveback_close_profit
        item["current_profit"] = current_profit
        item["max_profit_seen"] = max(float(item.get("max_profit_seen") or 0.0), current_profit)

        # Arm the giveback rule as soon as half-TP profit is reached on any poll.
        if current_profit >= trigger_profit:
            item["reached_trigger_profit"] = True
        item.setdefault("reached_trigger_profit", False)
        item.setdefault("candles_at_or_above_trigger", 0)

        # Count sustained profit once per completed candle. The giveback rule is
        # still evaluated every polling cycle so profit decay can close quickly.
        if bar_time and not str(bar_time).startswith("bar_time_error:") and item.get("last_checked_bar_time") != bar_time:
            item["last_checked_bar_time"] = bar_time
            if current_profit >= trigger_profit:
                item["candles_at_or_above_trigger"] = int(item.get("candles_at_or_above_trigger") or 0) + 1
            else:
                item["candles_at_or_above_trigger"] = 0

        should_close = False
        close_reason = ""
        if bool(item.get("reached_trigger_profit")) and current_profit <= giveback_close_profit:
            should_close = True
            close_reason = (
                "tp_profit_giveback_exit:"
                f"current_profit={current_profit:.2f};"
                f"tp_profit={tp_profit:.2f};"
                f"trigger_profit={trigger_profit:.2f};"
                f"giveback_close_profit={giveback_close_profit:.2f}"
            )
        elif int(item.get("candles_at_or_above_trigger") or 0) > sustained_candles:
            should_close = True
            close_reason = (
                "tp_profit_sustained_exit:"
                f"current_profit={current_profit:.2f};"
                f"tp_profit={tp_profit:.2f};"
                f"trigger_profit={trigger_profit:.2f};"
                f"candles_at_or_above_trigger={item.get('candles_at_or_above_trigger')}"
            )

        if not should_close:
            if bool(cfg_block.get("log_position_state", False)):
                _append_position_manager_row(rows, {
                    "mode": mode,
                    "data_source": data_source,
                    "symbol": symbol,
                    "ticket": ticket,
                    "side": side,
                    "bar_time": bar_time,
                    "decision": "HOLD",
                    "reason": "tp_profit_protection_hold",
                    "current_profit": current_profit,
                    "tp_profit": tp_profit,
                    "trigger_profit": trigger_profit,
                    "giveback_close_profit": giveback_close_profit,
                    "candles_at_or_above_trigger": item.get("candles_at_or_above_trigger"),
                    "reached_trigger_profit": item.get("reached_trigger_profit"),
                })
            continue

        payload = {
            "mode": mode,
            "data_source": data_source,
            "symbol": symbol,
            "ticket": ticket,
            "side": side,
            "bar_time": bar_time,
            "decision": "CLOSE",
            "reason": close_reason,
            "current_profit": current_profit,
            "tp_profit": tp_profit,
            "trigger_profit": trigger_profit,
            "giveback_close_profit": giveback_close_profit,
            "candles_at_or_above_trigger": item.get("candles_at_or_above_trigger"),
            "reached_trigger_profit": item.get("reached_trigger_profit"),
            "close_attempted": True,
        }
        try:
            result = _close_mt5_position(mt5, position, cfg, reason="tp_profit_protection_exit")
            payload["close_result"] = str(result)
            payload["close_sent"] = True
            item["last_close_reason"] = close_reason
            if reentry_block_after_close:
                closed_symbols.add(symbol)
        except Exception as exc:
            payload["decision"] = "ERROR"
            payload["close_error"] = str(exc)
            payload["close_sent"] = False
            item["last_close_error"] = str(exc)
        _append_position_manager_row(rows, payload)

    return rows, closed_symbols


def evaluate_combo_signal(combo: dict[str, Any], *, args: argparse.Namespace, live_root: Path, state: dict[str, Any], mode: str, data_source: str) -> dict[str, Any]:
    symbol = combo["symbol"]
    cfg = combo["cfg"]
    model, scaler, feature_columns, device, artifacts = combo["bundle"]
    requested_bars = _live_requested_bars(cfg)
    feat, meta = latest_processed_features(symbol, cfg, bars=requested_bars)
    if len(feat) == 0:
        raise RuntimeError(f"No live feature rows returned for {symbol}")
    meta = dict(meta or {})
    meta["requested_bars"] = int(requested_bars)
    meta["requested_bars_source"] = "replay_buy_sell_lookback_bars" if _uses_rolling_quantile_thresholds(cfg) else "live_direction_policy.bars"

    latest_for_gate = feat.iloc[-1]
    bar_time = str(latest_for_gate.get("time_utc") or latest_for_gate.get("time") or "")
    if _combo_processed(state, combo["id"], bar_time):
        return {
            "time_utc": _utc_now_iso(),
            "symbol": symbol,
            "model": combo["model"],
            "side_model": combo["side"],
            "epoch": combo.get("epoch"),
            "combo_id": combo["id"],
            "mode": mode,
            "data_source": data_source,
            "bar_time": bar_time,
            "model_decision": "SKIP",
            "final_decision": "SKIP",
            "reason": "already_processed_model_bar",
            "direction": "SKIP",
            "selected_probability": 0.0,
            "order_selected_by_ensemble": False,
            "order_attempted": False,
            "order_sent": False,
            "live_meta": meta,
            **artifacts,
        }

    sequences, endpoint_rows = _live_feature_sequences(feat, cfg, scaler, feature_columns)
    latest_row = endpoint_rows.iloc[-1]
    pred = _predict_live_sequences(model, sequences, device, cfg)
    if _uses_rolling_quantile_thresholds(cfg):
        decision_info = _latest_rolling_quantile_live_decision(pred, cfg)
    else:
        decision_info = _latest_fixed_threshold_live_decision(pred, cfg)

    probs = decision_info["probabilities"]
    side = str(decision_info["side"])
    selected_prob = float(decision_info["selected_probability"])
    trade_probability = float(decision_info["trade_probability"])
    side_sell_probability = _as_float_or_none(pred["side_sell_probability"][-1])
    side_buy_probability = _as_float_or_none(pred["side_buy_probability"][-1])
    buy_edge_pips = _as_float_or_none(pred["buy_edge_pips"][-1])
    sell_edge_pips = _as_float_or_none(pred["sell_edge_pips"][-1])
    buy_setup_probability = _as_float_or_none(pred["buy_setup_probability"][-1])
    sell_setup_probability = _as_float_or_none(pred["sell_setup_probability"][-1])
    buy_setup_quality_score = _as_float_or_none(pred["buy_setup_quality_score"][-1])
    sell_setup_quality_score = _as_float_or_none(pred["sell_setup_quality_score"][-1])
    selected_edge_pips = buy_edge_pips if side == "BUY" else sell_edge_pips if side == "SELL" else None

    min_prob = _min_direction_probability(cfg)
    min_trade_prob = _min_trade_probability(cfg)
    min_edge_pips = _min_edge_pips(cfg)
    symbol_trade_mode = _symbol_trade_mode(symbol, cfg)
    symbol_trade_mode_allows_side = _trade_mode_allows_side(symbol_trade_mode, side)

    model_decision = "BLOCK"
    reason = ""
    analytics: dict[str, Any] = {}
    cooldown_gate: dict[str, Any] = {}
    cooldown_allows_order = True
    rolling_thresholds_used = bool(decision_info.get("rolling_thresholds_used", False))

    if side == "NO_TRADE":
        reason = "side_score_below_rolling_quantile" if rolling_thresholds_used else "model_no_trade"
    elif not symbol_trade_mode_allows_side:
        reason = _trade_mode_block_reason(symbol, symbol_trade_mode, side)
    elif not rolling_thresholds_used and trade_probability < min_trade_prob:
        reason = "trade_probability_low"
    elif not rolling_thresholds_used and selected_prob < min_prob:
        reason = "direction_probability_low"
    elif min_edge_pips is not None and selected_edge_pips is not None and selected_edge_pips < float(min_edge_pips):
        reason = "edge_pips_low"
    else:
        cooldown_allows_order, cooldown_reason, cooldown_gate = _cooldown_gate(symbol, side, bar_time, cfg)
        if not cooldown_allows_order:
            reason = cooldown_reason
            analytics = {"cooldown_gate": cooldown_gate}
        else:
            gate = external_trade_gate(symbol, side, latest_row, cfg)
            analytics = gate.diagnostics
            if not gate.allow:
                reason = "|".join(gate.reasons) if gate.reasons else "external_gate_blocked"
            else:
                model_decision = "ALLOW"
                reason = "ok"

    row = {
        "time_utc": _utc_now_iso(),
        "symbol": symbol,
        "model": combo["model"],
        "side_model": combo["side"],
        "epoch": combo.get("epoch"),
        "combo_id": combo["id"],
        "mode": mode,
        "data_source": data_source,
        "bar_time": bar_time,
        "model_decision": model_decision,
        "final_decision": model_decision,
        "reason": reason,
        "direction": side,
        "selected_probability": selected_prob,
        "sell_probability": float(probs[0]),
        "no_trade_probability": float(probs[1]),
        "buy_probability": float(probs[2]),
        "trade_probability": trade_probability,
        "side_sell_probability": side_sell_probability,
        "side_buy_probability": side_buy_probability,
        "buy_setup_probability": buy_setup_probability,
        "sell_setup_probability": sell_setup_probability,
        "buy_setup_quality_score": buy_setup_quality_score,
        "sell_setup_quality_score": sell_setup_quality_score,
        "buy_edge_pips": buy_edge_pips,
        "sell_edge_pips": sell_edge_pips,
        "selected_edge_pips": selected_edge_pips,
        "threshold_mode": decision_info.get("threshold_mode"),
        "rolling_thresholds_used": bool(decision_info.get("rolling_thresholds_used", False)),
        "rolling_prediction_rows": decision_info.get("rolling_prediction_rows"),
        "requested_history_bars": int(requested_bars),
        "buy_side_score": decision_info.get("buy_side_score"),
        "sell_side_score": decision_info.get("sell_side_score"),
        "buy_rolling_threshold": decision_info.get("buy_rolling_threshold"),
        "sell_rolling_threshold": decision_info.get("sell_rolling_threshold"),
        "buy_threshold_margin": decision_info.get("buy_threshold_margin"),
        "sell_threshold_margin": decision_info.get("sell_threshold_margin"),
        "buy_pass_rolling_quantile": decision_info.get("buy_pass_rolling_quantile"),
        "sell_pass_rolling_quantile": decision_info.get("sell_pass_rolling_quantile"),
        "buy_threshold_params": decision_info.get("buy_threshold_params"),
        "sell_threshold_params": decision_info.get("sell_threshold_params"),
        "min_direction_probability": min_prob,
        "min_trade_probability": min_trade_prob,
        "min_edge_pips": min_edge_pips,
        "symbol_trade_mode": symbol_trade_mode,
        "symbol_trade_mode_allows_side": bool(symbol_trade_mode_allows_side),
        "cooldown_allows_order": bool(cooldown_allows_order),
        "cooldown_gate": cooldown_gate,
        "spread_points": latest_row.get("spread_points", None),
        "analytics_gate": analytics,
        "order_selected_by_ensemble": False,
        "order_attempted": False,
        "order_sent": False,
        "order_result": None,
        "order_error": None,
        "daily_loss_allows_order": True,
        "daily_loss_gate": {},
        "feature_rows": len(feat),
        "live_meta": meta,
        "deployment_score": combo["entry"].get("deployment_score"),
        "selected_model_net_pips": combo["entry"].get("net_pips"),
        "selected_model_average_net_pips": combo["entry"].get("average_net_pips"),
        "selected_model_win_rate": combo["entry"].get("win_rate"),
        "selected_model_max_drawdown_pips": combo["entry"].get("max_drawdown_pips"),
        "config_path": combo["config_path"],
        **artifacts,
    }
    row["signal_rank_score"] = _signal_rank_score(row)
    _mark_combo_processed(state, combo["id"], row)
    return row


def _select_order_candidates(rows: list[dict[str, Any]], *, one_order_per_symbol_bar: bool) -> set[str]:
    candidates = [r for r in rows if r.get("model_decision") == "ALLOW" and str(r.get("direction")) in {"BUY", "SELL"}]
    if not one_order_per_symbol_bar:
        return {str(r.get("combo_id")) for r in candidates}
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in candidates:
        by_key.setdefault((str(r.get("symbol")), str(r.get("bar_time"))), []).append(r)
    selected: set[str] = set()
    for _, group in by_key.items():
        best = sorted(group, key=lambda x: _signal_rank_score(x), reverse=True)[0]
        selected.add(str(best.get("combo_id")))
    return selected


def _apply_orders(rows: list[dict[str, Any]], combos_by_id: dict[str, dict[str, Any]], *, mode: str, one_order_per_symbol_bar: bool, blocked_symbols: set[str] | None = None) -> None:
    selected_ids = _select_order_candidates(rows, one_order_per_symbol_bar=one_order_per_symbol_bar)
    for row in rows:
        combo_id = str(row.get("combo_id"))
        combo = combos_by_id.get(combo_id)
        if not combo or row.get("model_decision") != "ALLOW":
            continue
        if combo_id not in selected_ids:
            row["final_decision"] = "BLOCK"
            row["reason"] = "ensemble_not_selected_best_signal_for_symbol_bar"
            row["order_selected_by_ensemble"] = False
            continue

        cfg = combo["cfg"]
        symbol = combo["symbol"]
        side = str(row.get("direction"))
        if blocked_symbols and symbol in blocked_symbols:
            row["final_decision"] = "BLOCK"
            row["reason"] = "position_manager_closed_symbol_same_cycle"
            row["order_selected_by_ensemble"] = False
            continue
        row["order_selected_by_ensemble"] = True
        if mode in {"demo", "live"}:
            allow_daily, daily_reason, daily_gate = _account_daily_loss_gate(cfg, mode=mode)
            row["daily_loss_allows_order"] = bool(allow_daily)
            row["daily_loss_gate"] = daily_gate
            if not allow_daily:
                row["final_decision"] = "BLOCK"
                row["reason"] = daily_reason
                continue
            row["order_attempted"] = True
            try:
                result = send_order(symbol, side, cfg)
                row["order_result"] = str(result)
                row["order_sent"] = True
                row["final_decision"] = "ALLOW"
                row["reason"] = "ok"
            except Exception as exc:
                row["order_error"] = str(exc)
                row["order_sent"] = False
                row["final_decision"] = "BLOCK"
                row["reason"] = f"order_error:{exc}"
        else:
            row["final_decision"] = "ALLOW"
            row["reason"] = "paper_ok"

        if row["final_decision"] == "ALLOW" and (mode == "paper" or row.get("order_sent")):
            _record_cooldown_order(symbol, side, str(row.get("bar_time") or ""), cfg, mode=mode, order_sent=bool(row.get("order_sent")), order_result=row.get("order_result"))


def _print_cycle_summary(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        if r.get("model") == "position_manager":
            print(
                f"{r.get('symbol')} position_manager/{r.get('side_model')}: "
                f"{r.get('final_decision')} ticket={r.get('ticket')} "
                f"profit={float(r.get('current_profit') or 0.0):.2f} "
                f"tp_profit={float(r.get('tp_profit') or 0.0):.2f} "
                f"reason={r.get('reason')}",
                flush=True,
            )
        elif r.get("model_decision") == "SKIP":
            print(f"{r.get('symbol')} {r.get('model')}/{r.get('side_model')}: SKIP already_decided bar={r.get('bar_time','')}", flush=True)
        else:
            print(
                f"{r.get('symbol')} {r.get('model')}/{r.get('side_model')} e{r.get('epoch')}: "
                f"model={r.get('model_decision')} final={r.get('final_decision')} "
                f"{r.get('direction')} p={float(r.get('selected_probability') or 0.0):.3f} "
                f"rank={float(r.get('signal_rank_score') or 0.0):.3f} reason={r.get('reason')}",
                flush=True,
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Live/demo ensemble runner for staged direction-policy models.")
    p.add_argument("--live-root", default="For Live Trading", help="Folder created by select_live_trading_models.py")
    p.add_argument("--manifest", default=None, help="Optional manifest JSON/YAML or per-symbol config YAML.")
    p.add_argument("--universal-config", default=None, help="Universal generic live config. Defaults to manifest universal_config_path or direction_settings_generic_multisymbol_31_symbols.yaml under live-root.")
    p.add_argument("--no-require-universal-config", dest="require_universal_config", action="store_false", help="Allow fallback to side configs only if the universal config is missing.")
    p.set_defaults(require_universal_config=True)
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--sides", nargs="+", default=None)
    p.add_argument("--mode", choices=["paper", "demo", "live"], default="paper")
    p.add_argument("--data-source", choices=["mt5"], default="mt5")
    p.add_argument("--poll-seconds", type=float, default=20.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--signals-csv", default=None)
    p.add_argument("--bar-state-json", default=None)
    p.add_argument("--one-order-per-symbol-bar", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    live_root = Path(args.live_root)
    entries, manifest_dir, manifest_payload = _load_manifest(live_root, Path(args.manifest) if args.manifest else None)
    universal_config_path = _resolve_universal_config_path(
        cli_value=args.universal_config,
        manifest_payload=manifest_payload,
        live_root=live_root,
        manifest_dir=manifest_dir,
    )
    universal_cfg = _load_universal_config(universal_config_path, require=bool(args.require_universal_config))
    entries = _filter_entries(entries, args)
    if not entries:
        raise SystemExit("No staged live model entries matched the requested filters.")

    # Validate symbols up front to catch obvious typos before loading every model.
    validate_forex_symbols(sorted({str(e.get("symbol")).upper() for e in entries}))

    print(f"Using universal live config: {universal_config_path}", flush=True)
    print(f"Loading {len(entries)} staged live model(s)...", flush=True)
    combos = [
        _load_combo(
            e,
            live_root=live_root,
            manifest_dir=manifest_dir,
            universal_cfg=universal_cfg,
            universal_config_path=universal_config_path,
            device=args.device,
        )
        for e in entries
    ]
    combos_by_id = {c["id"]: c for c in combos}
    print("Loaded:", ", ".join(f"{c['symbol']}:{c['model']}:{c['side']}:e{c.get('epoch')}" for c in combos), flush=True)

    signals_csv = _signals_csv_path(args, live_root)
    try:
        while True:
            state = _load_bar_state(args, live_root)
            rows: list[dict[str, Any]] = []
            blocked_symbols: set[str] = set()
            try:
                manager_rows, blocked_symbols = _manage_tp_profit_protection_exits(
                    combos,
                    state=state,
                    mode=args.mode,
                    data_source=args.data_source,
                )
                rows.extend(manager_rows)
            except Exception as exc:
                rows.append({
                    "time_utc": _utc_now_iso(),
                    "symbol": None,
                    "model": "position_manager",
                    "side_model": "tp_profit_protection_exit",
                    "combo_id": "position_manager:error",
                    "mode": args.mode,
                    "data_source": args.data_source,
                    "model_decision": "ERROR",
                    "final_decision": "ERROR",
                    "reason": str(exc),
                    "direction": "ERROR",
                    "order_attempted": False,
                    "order_sent": False,
                })
            for combo in combos:
                try:
                    rows.append(evaluate_combo_signal(combo, args=args, live_root=live_root, state=state, mode=args.mode, data_source=args.data_source))
                except Exception as exc:
                    rows.append({
                        "time_utc": _utc_now_iso(),
                        "symbol": combo.get("symbol"),
                        "model": combo.get("model"),
                        "side_model": combo.get("side"),
                        "epoch": combo.get("epoch"),
                        "combo_id": combo.get("id"),
                        "mode": args.mode,
                        "data_source": args.data_source,
                        "model_decision": "ERROR",
                        "final_decision": "ERROR",
                        "reason": str(exc),
                        "direction": "ERROR",
                        "order_attempted": False,
                        "order_sent": False,
                        "config_path": combo.get("config_path"),
                    })

            _apply_orders(rows, combos_by_id, mode=args.mode, one_order_per_symbol_bar=bool(args.one_order_per_symbol_bar), blocked_symbols=blocked_symbols)
            for row in rows:
                _append_csv(signals_csv, row)
            _save_bar_state(args, live_root, state)
            _print_cycle_summary(rows)

            # Reuse the original trade sync with the first config. The staged
            # configs all point at the same live-root summary/trade files.
            try:
                summary = sync_trade_logs_and_summary(combos[0]["cfg"], mode=args.mode, data_source=args.data_source)
                if summary.get("status") == "ok":
                    print(
                        "TRADE_LOG_SYNC: "
                        f"open={summary.get('open_trades', 0)} "
                        f"closed={summary.get('closed_trades', 0)} "
                        f"overall_pnl={summary.get('overall_pnl', 0.0):.2f} "
                        f"closed_pips={summary.get('closed_pips', 0.0):.1f}",
                        flush=True,
                    )
                elif summary.get("status") not in {"skipped_for_mode", "disabled"}:
                    print(f"TRADE_LOG_SYNC: {summary.get('status')} {summary.get('error', '')}", flush=True)
            except Exception as exc:
                print(f"TRADE_LOG_SYNC: ERROR {exc}", flush=True)

            if args.once:
                break
            time.sleep(float(args.poll_seconds))
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
