#!/usr/bin/env python3
"""Replay staged live-trading direction models over a rolling-quantile sweep.

This utility is designed to test the models staged by select_live_trading_models.py
under the same folder structure consumed by live_direction_policy_ensemble.py.

It reads the staged manifest under "For Live Trading", loads the universal live
configuration plus each model/side overlay, runs historical inference for each
model-symbol-side combination, sweeps one or more rolling score quantiles, and
writes deployment/replay tables that can be used to choose tighter or looser
live thresholds.

Typical use from the project root:

    python replay_live_trading_quantile_sweep.py \
      --live-root "For Live Trading" \
      --data-root data/direction \
      --replay-start 2025-01-01 \
      --replay-end 2026-01-01 \
      --quantiles 0.975 0.980 0.985 0.990 0.995 \
      --device cuda

The script intentionally applies rolling thresholds using prior scores only:
current-bar score is never used to set its own threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

# The script can be run either from project root or copied into src/.
try:
    from src.config import load_config_with_optional_spread_risk
    from src.direction_model import DirectionTradePolicyNet, direction_probabilities_from_outputs
    from src.forex import validate_forex_symbols
    from src.io_utils import read_json
    from src.live_direction_policy import _extract_state, _torch_load
    from src.test_saved_direction_policy import replay_symbol
    from src.forex import pip_size as instrument_pip_size, symbol_cfg_value
except ImportError:  # pragma: no cover - supports python -m src.<script> after copying into src
    from .config import load_config_with_optional_spread_risk
    from .direction_model import DirectionTradePolicyNet, direction_probabilities_from_outputs
    from .forex import validate_forex_symbols
    from .io_utils import read_json
    from .live_direction_policy import _extract_state, _torch_load
    from .test_saved_direction_policy import replay_symbol
    from .forex import pip_size as instrument_pip_size, symbol_cfg_value


SIDE_TO_CLASS = {"sell": 0, "buy": 2}
CLASS_TO_SIDE = {0: "sell", 2: "buy"}
SIDE_TO_LIVE = {"buy": "BUY", "sell": "SELL"}

UNIVERSAL_OVERRIDE_SECTIONS = {
    # These are intended to be universal/live-policy settings. They should come
    # from the copied generic config unless a side overlay explicitly needs a
    # model-specific value through an allowed overlay section below.
    "risk",
    "execution",
    "external_trade_filter",
    "analytics",
    "analytic_signals",
    "spread_control",
    "spread_risk",
    "mt5",
    "broker",
    "account",
}

SIDE_OVERLAY_ALLOWED_TOP_LEVEL = {
    "project",
    "trading",
    "paths",
    "model",
    "replay",
    "training",
    "symbol_trade_modes",
    "live_direction_policy",
    "live_model_selection",
}


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


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except Exception:
            return default


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _side_overlay_only(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep model/side-specific sections and drop universal live-policy blocks."""
    out: dict[str, Any] = {}
    for key in SIDE_OVERLAY_ALLOWED_TOP_LEVEL:
        if key in cfg:
            out[key] = cfg[key]
    for key in UNIVERSAL_OVERRIDE_SECTIONS:
        out.pop(key, None)
    return out


def _resolve_path(value: Any, *, live_root: Path, manifest_dir: Path, project_root: Path | None = None) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    candidates = [Path.cwd() / raw, live_root / raw, manifest_dir / raw]
    if project_root is not None:
        candidates.append(project_root / raw)
    # If the manifest already starts with the live root name, avoid duplicating it.
    parts = raw.parts
    if parts and parts[0] == live_root.name:
        candidates.append(live_root.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_manifest(live_root: Path, manifest_path: Path | None) -> tuple[dict[str, Any], Path]:
    if manifest_path is None:
        for name in ("live_ensemble_manifest.json", "live_ensemble_manifest.yaml", "live_ensemble_manifest.yml"):
            p = live_root / name
            if p.exists():
                manifest_path = p
                break
    if manifest_path is None:
        raise SystemExit(f"No live ensemble manifest found under {live_root}")
    payload = _read_structured(manifest_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise SystemExit(f"Manifest does not contain a models list: {manifest_path}")
    return payload, manifest_path.parent


def _normalise_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"long", "buy_only"}:
        return "buy"
    if side in {"short", "sell_only"}:
        return "sell"
    return side


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


def _timeframe_from_cfg_or_entry(cfg: dict[str, Any], entry: dict[str, Any]) -> str:
    return str(
        entry.get("timeframe")
        or (cfg.get("trading") or {}).get("timeframe")
        or (cfg.get("project") or {}).get("timeframe")
        or "M5"
    ).upper()


def _load_universal_cfg(args: argparse.Namespace, manifest: dict[str, Any], live_root: Path, manifest_dir: Path) -> dict[str, Any]:
    explicit = Path(args.universal_config) if args.universal_config else None
    manifest_value = manifest.get("universal_config_path") or manifest.get("generic_config_path")
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if manifest_value:
        candidates.append(_resolve_path(manifest_value, live_root=live_root, manifest_dir=manifest_dir, project_root=Path(args.project_root)))
    candidates.append(live_root / "direction_settings_generic_multisymbol_31_symbols.yaml")
    candidates.append(Path(args.project_root) / "config" / "direction_settings_generic_multisymbol_31_symbols.yaml")

    for p in candidates:
        if p and p.exists():
            return load_config_with_optional_spread_risk(str(p))
    if args.require_universal_config:
        raise SystemExit("Universal config was not found. Pass --universal-config or place direction_settings_generic_multisymbol_31_symbols.yaml under the live root.")
    return {}


def _load_combo_cfg(entry: dict[str, Any], *, universal_cfg: dict[str, Any], live_root: Path, manifest_dir: Path, project_root: Path) -> dict[str, Any]:
    config_path = _resolve_path(entry["config_path"], live_root=live_root, manifest_dir=manifest_dir, project_root=project_root)
    side_cfg_raw = load_config_with_optional_spread_risk(str(config_path))
    side_overlay = _side_overlay_only(side_cfg_raw)
    cfg = _deep_merge(universal_cfg, side_overlay)

    live_cfg = cfg.setdefault("live_direction_policy", {})
    for key in ("model_path", "scaler_path", "features_path"):
        value = entry.get(key) or live_cfg.get(key)
        if value:
            live_cfg[key] = str(_resolve_path(value, live_root=live_root, manifest_dir=manifest_dir, project_root=project_root))

    symbol = str(entry.get("symbol") or "").upper()
    side = _normalise_side(entry.get("side"))
    timeframe = _timeframe_from_cfg_or_entry(cfg, entry)
    cfg.setdefault("trading", {})["symbols"] = [symbol]
    cfg["trading"]["timeframe"] = timeframe
    cfg.setdefault("project", {})["timeframe"] = timeframe
    cfg.setdefault("replay", {})["allow_buy"] = side == "buy"
    cfg.setdefault("replay", {})["allow_sell"] = side == "sell"
    cfg["symbol_trade_modes"] = {"symbols": {symbol: "buy_only" if side == "buy" else "sell_only"}}
    return cfg


def _read_feature_columns(path: Path) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        for key in ("feature_columns", "features", "columns"):
            if isinstance(payload.get(key), list):
                return [str(x) for x in payload[key]]
    raise ValueError(f"Could not read feature columns from {path}")


@dataclass
class Combo:
    id: str
    symbol: str
    model_token: str
    side: str
    epoch: int
    entry: dict[str, Any]
    cfg: dict[str, Any]
    model: torch.nn.Module
    scaler: Any
    feature_columns: list[str]
    device: str


def _load_model_for_combo(entry: dict[str, Any], cfg: dict[str, Any], args: argparse.Namespace) -> Combo:
    symbol = str(entry.get("symbol") or "").upper()
    side = _normalise_side(entry.get("side"))
    model_token = str(entry.get("model") or "model")
    epoch = _safe_int(entry.get("epoch"), -1)
    combo_id = str(entry.get("id") or f"{symbol}_{model_token}_{side}_epoch_{epoch:03d}")

    live = cfg.get("live_direction_policy") or {}
    model_path = Path(str(live.get("model_path") or entry.get("model_path")))
    scaler_path = Path(str(live.get("scaler_path") or entry.get("scaler_path")))
    features_path = Path(str(live.get("features_path") or entry.get("features_path")))
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found for {combo_id}: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found for {combo_id}: {scaler_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature list not found for {combo_id}: {features_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_columns = _read_feature_columns(features_path)
    scaler = joblib.load(scaler_path)
    payload = _torch_load(model_path, device)
    model_cfg = payload.get("model_config") if isinstance(payload, dict) and isinstance(payload.get("model_config"), dict) else None
    model_cfg_full = dict(cfg)
    if model_cfg is not None:
        model_cfg_full["model"] = model_cfg
    model_cfg_full["_feature_columns"] = list(feature_columns)
    model = DirectionTradePolicyNet(len(feature_columns), model_cfg_full).to(device)
    model.load_state_dict(_extract_state(payload), strict=True)
    model.eval()

    return Combo(combo_id, symbol, model_token, side, epoch, entry, cfg, model, scaler, feature_columns, device)


def _data_candidates(symbol: str, timeframe: str, args: argparse.Namespace, cfg: dict[str, Any]) -> list[Path]:
    data_root = Path(args.data_root)
    templates = list(args.data_template or [])
    if not templates:
        templates = [
            "{symbol}_{timeframe}_direction_training.csv",
            "{symbol}_{timeframe}_direction.csv",
            "{symbol}_{timeframe}_multitask_training.csv",
            "{symbol}_{timeframe}_processed.csv",
            "{symbol}_{timeframe}.csv",
            "{symbol}.csv",
        ]
    candidates: list[Path] = []
    for tmpl in templates:
        candidates.append(data_root / tmpl.format(symbol=symbol, timeframe=timeframe))
    # Common project fallbacks.
    candidates.extend([
        Path("data/direction") / f"{symbol}_{timeframe}_direction_training.csv",
        Path("data/direction") / f"{symbol}_{timeframe}_direction.csv",
        Path("data/multitask") / f"{symbol}_{timeframe}_multitask_training.csv",
        Path("data/processed_m5") / f"{symbol}_{timeframe}_processed.csv",
        Path("data/processed") / f"{symbol}_{timeframe}_processed.csv",
    ])
    # Config-driven processed dirs.
    paths = cfg.get("paths") or {}
    for key in ("direction_data_dir", "multitask_data_dir", "processed_dir", "data_dir"):
        if paths.get(key):
            p = Path(str(paths[key]))
            candidates.extend([
                p / f"{symbol}_{timeframe}_direction_training.csv",
                p / f"{symbol}_{timeframe}_direction.csv",
                p / f"{symbol}_{timeframe}_multitask_training.csv",
                p / f"{symbol}_{timeframe}_processed.csv",
                p / f"{symbol}_{timeframe}.csv",
            ])
    out: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            out.append(c)
            seen.add(key)
    return out


def _load_symbol_data(symbol: str, timeframe: str, args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    for path in _data_candidates(symbol, timeframe, args, cfg):
        if path.exists():
            df = pd.read_csv(path)
            return df, path
    msg = "\n".join(str(p) for p in _data_candidates(symbol, timeframe, args, cfg)[:15])
    raise FileNotFoundError(f"No replay data file found for {symbol} {timeframe}. Checked:\n{msg}")


def _find_time_column(df: pd.DataFrame) -> str | None:
    for col in ("time_utc", "time", "datetime", "timestamp", "date", "Date", "Time"):
        if col in df.columns:
            return col
    return None


def _filter_replay_window(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    col = _find_time_column(df)
    if col is None or (not args.replay_start and not args.replay_end):
        return df.reset_index(drop=True)
    ts = pd.to_datetime(df[col], errors="coerce", utc=True)
    mask = ts.notna()
    if args.replay_start:
        mask &= ts >= pd.Timestamp(args.replay_start, tz="UTC")
    if args.replay_end:
        mask &= ts < pd.Timestamp(args.replay_end, tz="UTC")
    out = df.loc[mask].copy().reset_index(drop=True)
    out["_replay_time_utc"] = ts.loc[mask].dt.tz_convert("UTC").astype(str).to_numpy()
    return out


def _prepare_sequences(df: pd.DataFrame, combo: Combo) -> tuple[np.ndarray, pd.DataFrame]:
    seq_len = _safe_int((combo.cfg.get("model") or {}).get("sequence_length"), 64)
    fill = _safe_float((combo.cfg.get("features") or {}).get("fillna_value"), 0.0)
    if len(df) < seq_len:
        raise RuntimeError(f"{combo.id}: not enough replay rows: got {len(df)}, need sequence_length={seq_len}")
    missing = [c for c in combo.feature_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"{combo.id}: replay data missing required feature columns: {missing[:20]}")
    X = (
        df[combo.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill)
        .to_numpy(np.float32)
    )
    X = combo.scaler.transform(X).astype(np.float32)
    sequences = np.stack([X[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(X))]).astype(np.float32)
    endpoints = df.iloc[seq_len - 1:].reset_index(drop=True)
    return sequences, endpoints


def _predict_sequences(combo: Combo, sequences: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {
        "probabilities": [],
        "buy_setup_probability": [],
        "sell_setup_probability": [],
        "buy_setup_quality_score": [],
        "sell_setup_quality_score": [],
        "buy_edge_pips": [],
        "sell_edge_pips": [],
    }
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            xb = torch.tensor(sequences[start:start + batch_size], dtype=torch.float32, device=combo.device)
            outputs = combo.model(xb)
            probs = direction_probabilities_from_outputs(outputs).detach().cpu().numpy()
            collected["probabilities"].append(probs)
            for key in [
                "buy_setup_probability", "sell_setup_probability",
                "buy_setup_quality_score", "sell_setup_quality_score",
                "buy_edge_pips", "sell_edge_pips",
            ]:
                value = outputs.get(key)
                if value is None:
                    arr = np.full(len(probs), np.nan, dtype=float)
                else:
                    arr = value.detach().cpu().view(-1).numpy().astype(float)
                collected[key].append(arr)
    return {k: np.concatenate(v) if v else np.asarray([], dtype=float) for k, v in collected.items()}


def _side_scores(pred: dict[str, np.ndarray], side: str) -> np.ndarray:
    probs = pred["probabilities"]
    if side == "buy":
        setup = pred.get("buy_setup_probability", np.full(len(probs), np.nan))
        return np.where(np.isfinite(setup), setup, probs[:, 2]).astype(float)
    setup = pred.get("sell_setup_probability", np.full(len(probs), np.nan))
    return np.where(np.isfinite(setup), setup, probs[:, 0]).astype(float)


def _rolling_quantile_threshold(scores: np.ndarray, *, lookback: int, quantile: float, fallback: float, min_history: int) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=float))
    th = s.shift(1).rolling(window=max(int(lookback), 1), min_periods=max(int(min_history), 1)).quantile(float(quantile))
    return th.fillna(float(fallback)).to_numpy(float)


def _cfg_replay(combo: Combo) -> dict[str, Any]:
    raw = combo.cfg.get("replay") or {}
    return raw if isinstance(raw, dict) else {}


def _side_threshold_params(combo: Combo, side: str, quantile: float | None, args: argparse.Namespace) -> dict[str, Any]:
    rcfg = _cfg_replay(combo)
    scfg = rcfg.get(side) or rcfg.get(side.upper()) or {}
    if not isinstance(scfg, dict):
        scfg = {}
    lookback_default = _safe_int(rcfg.get("lookback_bars", rcfg.get("rolling_lookback_bars")), args.lookback_bars)
    lookback = _safe_int(scfg.get("lookback_bars"), lookback_default)
    min_history_default = max(200, min(max(lookback, 1), 1000))
    min_history = _safe_int(scfg.get("min_history_bars", rcfg.get("min_history_bars")), args.min_history_bars or min_history_default)
    fallback = _safe_float(scfg.get("fallback_threshold", rcfg.get("fallback_threshold")), args.fallback_threshold)
    q = float(quantile if quantile is not None else scfg.get("quantile", rcfg.get(f"{side}_quantile", args.quantiles[0])))
    return {"lookback_bars": max(1, lookback), "min_history_bars": max(1, min_history), "fallback_threshold": fallback, "quantile": q}


def _pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def _col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _spread_pips(row: pd.Series, cfg: dict[str, Any], symbol: str) -> float:
    labels = cfg.get("labels") or {}
    spread_col = str(labels.get("spread_column", "spread_points"))
    if spread_col in row.index:
        points = _safe_float(row.get(spread_col), _safe_float(labels.get("default_spread_points"), 2.0))
        return points * _safe_float(labels.get("spread_pips_per_point"), 0.1)
    for col in ("spread_pips", "spread"):
        if col in row.index:
            return _safe_float(row.get(col), 0.0)
    return _safe_float(labels.get("default_spread_points"), 2.0) * _safe_float(labels.get("spread_pips_per_point"), 0.1)


def _ohlc_column_map(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "open": _col(df, ["open", "Open", "mid_open", "open_mid", "close", "bid_open"]),
        "high": _col(df, ["high", "High", "mid_high", "high_mid", "bid_high"]),
        "low": _col(df, ["low", "Low", "mid_low", "low_mid", "bid_low"]),
        "close": _col(df, ["close", "Close", "mid_close", "close_mid", "bid_close"]),
        "bid_open": _col(df, ["bid_open", "open_bid"]),
        "bid_high": _col(df, ["bid_high", "high_bid"]),
        "bid_low": _col(df, ["bid_low", "low_bid"]),
        "ask_open": _col(df, ["ask_open", "open_ask"]),
        "ask_high": _col(df, ["ask_high", "high_ask"]),
        "ask_low": _col(df, ["ask_low", "low_ask"]),
    }


def _has_ohlc(df: pd.DataFrame) -> bool:
    m = _ohlc_column_map(df)
    return bool(m["open"] and m["high"] and m["low"])


def _entry_price(row: pd.Series, side: str, cols: dict[str, str | None], cfg: dict[str, Any], symbol: str) -> float:
    pip = _pip_size(symbol)
    spread = _spread_pips(row, cfg, symbol) * pip
    if side == "buy" and cols.get("ask_open"):
        return _safe_float(row[cols["ask_open"]])
    if side == "sell" and cols.get("bid_open"):
        return _safe_float(row[cols["bid_open"]])
    mid = _safe_float(row[cols["open"]])
    return mid + spread / 2.0 if side == "buy" else mid - spread / 2.0


def _future_high_low(row_future: pd.Series, side: str, cols: dict[str, str | None], cfg: dict[str, Any], symbol: str) -> tuple[float, float]:
    pip = _pip_size(symbol)
    spread = _spread_pips(row_future, cfg, symbol) * pip
    if side == "buy":
        # BUY exits at bid; prefer bid high/low when available.
        high_col = cols.get("bid_high") or cols.get("high")
        low_col = cols.get("bid_low") or cols.get("low")
        high = _safe_float(row_future[high_col])
        low = _safe_float(row_future[low_col])
        if not cols.get("bid_high"):
            high -= spread / 2.0
        if not cols.get("bid_low"):
            low -= spread / 2.0
        return high, low
    # SELL exits at ask; prefer ask high/low when available.
    high_col = cols.get("ask_high") or cols.get("high")
    low_col = cols.get("ask_low") or cols.get("low")
    high = _safe_float(row_future[high_col])
    low = _safe_float(row_future[low_col])
    if not cols.get("ask_high"):
        high += spread / 2.0
    if not cols.get("ask_low"):
        low += spread / 2.0
    return high, low


def _barrier_trade_pips(df: pd.DataFrame, entry_index: int, side: str, cfg: dict[str, Any], symbol: str, cols: dict[str, str | None]) -> float:
    labels = cfg.get("labels") or {}
    horizon = _safe_int(labels.get("horizon_bars"), _safe_int((cfg.get("replay") or {}).get("horizon_bars"), 24))
    tp = _safe_float(labels.get("take_profit_pips"), 8.0)
    sl = _safe_float(labels.get("stop_loss_pips"), 5.0)
    slippage = _safe_float(labels.get("slippage_pips"), 0.0)
    same_bar_policy = str(labels.get("same_bar_tp_sl_policy", "stop_first")).lower()
    entry_next = bool(labels.get("entry_on_next_bar_open", True))

    pip = _pip_size(symbol)
    idx = entry_index + (1 if entry_next else 0)
    if idx >= len(df):
        return 0.0
    entry = _entry_price(df.iloc[idx], side, cols, cfg, symbol)
    tp_price = entry + tp * pip if side == "buy" else entry - tp * pip
    sl_price = entry - sl * pip if side == "buy" else entry + sl * pip

    last_idx = min(len(df) - 1, idx + max(horizon, 1) - 1)
    for j in range(idx, last_idx + 1):
        high, low = _future_high_low(df.iloc[j], side, cols, cfg, symbol)
        if side == "buy":
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
        else:
            hit_tp = low <= tp_price
            hit_sl = high >= sl_price
        if hit_tp and hit_sl:
            if same_bar_policy in {"target_first", "tp_first", "take_profit_first"}:
                return tp - slippage
            return -sl - slippage
        if hit_tp:
            return tp - slippage
        if hit_sl:
            return -sl - slippage

    # Time exit at final close/mid close if no barrier hit.
    close_col = cols.get("close") or cols.get("open")
    close_mid = _safe_float(df.iloc[last_idx][close_col])
    spread = _spread_pips(df.iloc[last_idx], cfg, symbol) * pip
    if side == "buy":
        exit_price = close_mid - spread / 2.0
        return ((exit_price - entry) / pip) - slippage
    exit_price = close_mid + spread / 2.0
    return ((entry - exit_price) / pip) - slippage


def _label_pips_from_row(row: pd.Series, side: str) -> float | None:
    candidates = [
        f"{side}_net_pips",
        f"future_{side}_net_pips",
        f"{side}_clean_net_pips",
        f"{side}_outcome_pips",
        f"{side}_pips",
    ]
    for c in candidates:
        if c in row.index:
            v = _safe_float(row.get(c), float("nan"))
            if math.isfinite(v):
                return v
    return None


def _trade_pips(df: pd.DataFrame, signal_index: int, side: str, cfg: dict[str, Any], symbol: str, cols: dict[str, str | None]) -> float:
    if _has_ohlc(df):
        return _barrier_trade_pips(df, signal_index, side, cfg, symbol, cols)
    label_pips = _label_pips_from_row(df.iloc[signal_index], side)
    if label_pips is not None:
        return label_pips
    raise RuntimeError(
        f"No OHLC columns and no side net-pips label available for replay simulation. "
        f"Need open/high/low columns or {side}_net_pips/future_{side}_net_pips."
    )


def _equity_stats(pips: list[float]) -> dict[str, float]:
    if not pips:
        return {"net_pips": 0.0, "gross_profit_pips": 0.0, "gross_loss_pips": 0.0, "max_drawdown_pips": 0.0, "profit_factor": 0.0}
    arr = np.asarray(pips, dtype=float)
    equity = np.cumsum(arr)
    running_max = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = running_max - equity
    gross_profit = float(arr[arr > 0].sum())
    gross_loss = float(-arr[arr < 0].sum())
    return {
        "net_pips": float(arr.sum()),
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
        "max_drawdown_pips": float(dd.max()) if len(dd) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
    }


def _signals_for_quantile(combo: Combo, df: pd.DataFrame, pred: dict[str, np.ndarray], q: float, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    side = combo.side
    scores = _side_scores(pred, side)
    params = _side_threshold_params(combo, side, q, args)
    thresholds = _rolling_quantile_threshold(
        scores,
        lookback=params["lookback_bars"],
        quantile=q,
        fallback=params["fallback_threshold"],
        min_history=params["min_history_bars"],
    )
    pass_mask = scores >= thresholds

    probs = pred["probabilities"]
    side_prob = probs[:, SIDE_TO_CLASS[side]]
    if args.min_side_probability is not None:
        pass_mask &= side_prob >= float(args.min_side_probability)

    if args.max_spread_pips is not None:
        spreads = np.array([_spread_pips(row, combo.cfg, combo.symbol) for _, row in df.iterrows()], dtype=float)
        pass_mask &= spreads <= float(args.max_spread_pips)
    else:
        spreads = np.array([_spread_pips(row, combo.cfg, combo.symbol) for _, row in df.iterrows()], dtype=float)

    gap_bars = _safe_int((_cfg_replay(combo)).get("min_gap_bars_between_same_side_trades"), args.min_gap_bars)
    cooldown_until = -1
    selected_indices: list[int] = []
    for i, ok in enumerate(pass_mask):
        if not ok:
            continue
        if i <= cooldown_until:
            continue
        selected_indices.append(i)
        cooldown_until = i + max(0, gap_bars)

    cols = _ohlc_column_map(df)
    pips: list[float] = []
    rows: list[dict[str, Any]] = []
    time_col = "_replay_time_utc" if "_replay_time_utc" in df.columns else _find_time_column(df)
    for i in selected_indices:
        trade_pips = _trade_pips(df, i, side, combo.cfg, combo.symbol, cols)
        pips.append(float(trade_pips))
        rows.append({
            "combo_id": combo.id,
            "symbol": combo.symbol,
            "model": combo.model_token,
            "side": side,
            "epoch": combo.epoch,
            "quantile": float(q),
            "row_index": int(i),
            "time": str(df.iloc[i].get(time_col, "")) if time_col else "",
            "score": float(scores[i]),
            "threshold": float(thresholds[i]),
            "margin": float(scores[i] - thresholds[i]),
            "side_probability": float(side_prob[i]),
            "spread_pips": float(spreads[i]),
            "pips": float(trade_pips),
        })
    trades = len(pips)
    wins = sum(1 for x in pips if x > 0)
    stats = _equity_stats(pips)
    metrics = {
        "combo_id": combo.id,
        "symbol": combo.symbol,
        "model": combo.model_token,
        "side": side,
        "epoch": combo.epoch,
        "quantile": float(q),
        "lookback_bars": int(params["lookback_bars"]),
        "min_history_bars": int(params["min_history_bars"]),
        "fallback_threshold": float(params["fallback_threshold"]),
        "signals": int(np.count_nonzero(pass_mask)),
        "trades": int(trades),
        "wins": int(wins),
        "losses": int(sum(1 for x in pips if x < 0)),
        "win_rate": float(wins / trades) if trades else 0.0,
        "average_net_pips": float(stats["net_pips"] / trades) if trades else 0.0,
        **stats,
        "deployment_score": _deployment_score_from_metrics(stats["net_pips"], trades, wins / trades if trades else 0.0, stats["net_pips"] / trades if trades else 0.0, stats["max_drawdown_pips"]),
        "source_manifest_score": _safe_float(combo.entry.get("deployment_score")),
        "source_manifest_net_pips": _safe_float(combo.entry.get("net_pips")),
        "source_manifest_trades": _safe_float(combo.entry.get("trades")),
    }
    return pd.DataFrame(rows), metrics


def _deployment_score_from_metrics(net: float, trades: int, win_rate: float, avg: float, dd: float) -> float:
    # Same broad shape as the model selector defaults.
    score = net + 50.0 * avg + 50.0 * win_rate + 0.02 * min(trades, 300) - 0.35 * dd
    if net > 0 and dd > 0:
        ratio = dd / max(net, 1e-9)
        if ratio > 1.0:
            score -= 40.0 * (ratio - 1.0)
    if avg <= 0:
        score -= 250.0
    if net <= 0:
        score -= 500.0
    return float(score)


def _aggregate_quantile_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    grouped = []
    for q, g in rows.groupby("quantile", sort=True):
        trades = int(g["trades"].sum())
        net = float(g["net_pips"].sum())
        wins = int(g["wins"].sum())
        max_dd = float(g["max_drawdown_pips"].max())
        grouped.append({
            "quantile": float(q),
            "models": int(g["combo_id"].nunique()),
            "symbols": int(g["symbol"].nunique()),
            "trades": trades,
            "net_pips": net,
            "win_rate": float(wins / trades) if trades else 0.0,
            "average_net_pips": float(net / trades) if trades else 0.0,
            "max_model_drawdown_pips": max_dd,
            "mean_model_drawdown_pips": float(g["max_drawdown_pips"].mean()),
            "profitable_models": int((g["net_pips"] > 0).sum()),
            "negative_models": int((g["net_pips"] <= 0).sum()),
            "total_deployment_score": float(g["deployment_score"].sum()),
        })
    return pd.DataFrame(grouped).sort_values(["total_deployment_score", "net_pips"], ascending=False)


def _aggregate_symbol_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    grouped = []
    for (symbol, q), g in rows.groupby(["symbol", "quantile"], sort=True):
        trades = int(g["trades"].sum())
        net = float(g["net_pips"].sum())
        wins = int(g["wins"].sum())
        grouped.append({
            "symbol": symbol,
            "quantile": float(q),
            "models": int(g["combo_id"].nunique()),
            "buy_models": int((g["side"] == "buy").sum()),
            "sell_models": int((g["side"] == "sell").sum()),
            "trades": trades,
            "net_pips": net,
            "win_rate": float(wins / trades) if trades else 0.0,
            "average_net_pips": float(net / trades) if trades else 0.0,
            "max_side_drawdown_pips": float(g["max_drawdown_pips"].max()),
            "deployment_score": float(g["deployment_score"].sum()),
        })
    return pd.DataFrame(grouped).sort_values(["deployment_score", "net_pips"], ascending=False)


def _best_by_model(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    rows_sorted = rows.sort_values(["combo_id", "deployment_score", "net_pips", "average_net_pips"], ascending=[True, False, False, False])
    return rows_sorted.groupby("combo_id", as_index=False).head(1).reset_index(drop=True)


def _best_by_symbol(symbol_summary: pd.DataFrame) -> pd.DataFrame:
    if symbol_summary.empty:
        return pd.DataFrame()
    rows_sorted = symbol_summary.sort_values(["symbol", "deployment_score", "net_pips", "average_net_pips"], ascending=[True, False, False, False])
    return rows_sorted.groupby("symbol", as_index=False).head(1).reset_index(drop=True)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def _q_token(q: float) -> str:
    return f"q{float(q):.6f}".replace(".", "p").rstrip("0").rstrip("p")


def _set_combo_quantile(cfg: dict[str, Any], side: str, q: float) -> None:
    """Set the replay quantile in the same config fields used by test_saved_direction_policy."""
    rcfg = cfg.setdefault("replay", {})
    rcfg["threshold_mode"] = rcfg.get("threshold_mode", "rolling_score_quantile")
    rcfg[f"{side}_quantile"] = float(q)
    scfg = rcfg.setdefault(side, {})
    if isinstance(scfg, dict):
        scfg["quantile"] = float(q)
    rcfg["allow_buy"] = side == "buy"
    rcfg["allow_sell"] = side == "sell"


def _apply_data_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Allow the sweep CLI to override the data source while leaving replay logic unchanged."""
    tcfg = cfg.setdefault("training", {})
    if args.data_root:
        tcfg["direction_data_dir"] = str(args.data_root)
    if args.data_template:
        tcfg["direction_data_template"] = str(args.data_template[0])
    cfg.setdefault("backtest", {})["batch_size"] = int(max(1, args.batch_size))


def _metric_from_replay_summary(
    *,
    entry: dict[str, Any],
    cfg: dict[str, Any],
    summary: dict[str, Any],
    combo_id: str,
    q: float,
) -> dict[str, Any]:
    symbol = str(entry.get("symbol") or summary.get("symbol") or "").upper()
    side = _normalise_side(entry.get("side"))
    model_token = str(entry.get("model") or "model")
    epoch = _safe_int(entry.get("epoch"), -1)
    rcfg = cfg.get("replay") or {}
    scfg = rcfg.get(side) or {}
    if not isinstance(scfg, dict):
        scfg = {}

    trades = _safe_int(summary.get("trades"), 0)
    wins = _safe_int(summary.get("wins"), 0)
    if wins == 0 and trades and summary.get("win_rate") is not None:
        wins = int(round(float(summary.get("win_rate") or 0.0) * trades))
    losses = _safe_int(summary.get("losses"), max(0, trades - wins))
    net = _safe_float(summary.get("net_pips"), 0.0)
    avg = _safe_float(summary.get("average_net_pips"), net / trades if trades else 0.0)
    win_rate = _safe_float(summary.get("win_rate"), wins / trades if trades else 0.0)
    dd = _safe_float(summary.get("max_drawdown_pips"), 0.0)

    labels = cfg.get("labels") or {}
    return {
        "combo_id": combo_id,
        "symbol": symbol,
        "model": model_token,
        "side": side,
        "epoch": epoch,
        "quantile": float(q),
        "lookback_bars": _safe_int(scfg.get("lookback_bars", rcfg.get("lookback_bars", rcfg.get("rolling_lookback_bars"))), 0),
        "min_history_bars": _safe_int(scfg.get("min_history_bars", rcfg.get("min_history_bars")), 0),
        "fallback_threshold": _safe_float(scfg.get("fallback_threshold", rcfg.get("fallback_threshold")), 0.0),
        "pip_size": _safe_float(instrument_pip_size(symbol, cfg), 0.0),
        "take_profit_pips": _safe_float(symbol_cfg_value(cfg, "labels", "take_profit_pips", symbol, labels.get("take_profit_pips", 0.0)), 0.0),
        "stop_loss_pips": _safe_float(symbol_cfg_value(cfg, "labels", "stop_loss_pips", symbol, labels.get("stop_loss_pips", 0.0)), 0.0),
        "slippage_pips": _safe_float(symbol_cfg_value(cfg, "labels", "slippage_pips", symbol, labels.get("slippage_pips", 0.0)), 0.0),
        "signals": _safe_int(summary.get("passes_model_gate"), 0),
        "passes_external_gate": _safe_int(summary.get("passes_external_gate"), trades),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "average_net_pips": avg,
        "net_pips": net,
        "gross_profit_pips": _safe_float(summary.get("gross_profit_pips"), 0.0),
        "gross_loss_pips": _safe_float(summary.get("gross_loss_pips"), 0.0),
        "max_drawdown_pips": dd,
        "profit_factor": _safe_float(summary.get("profit_factor"), 0.0),
        "deployment_score": _safe_float(summary.get("replay_score"), _deployment_score_from_metrics(net, trades, win_rate, avg, dd)),
        "source_manifest_score": _safe_float(entry.get("deployment_score")),
        "source_manifest_net_pips": _safe_float(entry.get("net_pips")),
        "source_manifest_trades": _safe_float(entry.get("trades")),
        "summary_path": str(summary.get("summary_path", "")),
        "decisions_path": str(summary.get("decisions_path", "")),
        "trades_path": str(summary.get("trades_path", "")),
        "replay_start": summary.get("eval_start"),
        "replay_end": summary.get("eval_end"),
    }


def _read_exact_replay_trades(summary: dict[str, Any], entry: dict[str, Any], combo_id: str, q: float) -> pd.DataFrame:
    path_value = summary.get("trades_path")
    if not path_value:
        return pd.DataFrame()
    path = Path(str(path_value))
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df

    # replay_symbol() already writes some metadata columns such as ``symbol``
    # in its per-trade CSV.  The sweep adds/overwrites a consistent metadata
    # block without using DataFrame.insert(), because insert() raises
    # ``ValueError: cannot insert <column>, already exists`` when exact replay
    # has already included the column.
    metadata = {
        "combo_id": combo_id,
        "symbol": str(entry.get("symbol") or ""),
        "model": str(entry.get("model") or ""),
        "side": _normalise_side(entry.get("side")),
        "epoch": _safe_int(entry.get("epoch"), -1),
        "quantile": float(q),
    }
    for col, value in metadata.items():
        df[col] = value
    metadata_cols = list(metadata)
    remaining_cols = [c for c in df.columns if c not in metadata_cols]
    return df[metadata_cols + remaining_cols]


def main() -> None:
    p = argparse.ArgumentParser(description="Replay staged live-trading models across rolling quantile thresholds.")
    p.add_argument("--live-root", default="For Live Trading", help="Folder created by select_live_trading_models.py")
    p.add_argument("--manifest", default=None, help="Optional manifest JSON/YAML path.")
    p.add_argument("--universal-config", default=None, help="Optional universal generic config override.")
    p.add_argument("--require-universal-config", action="store_true", help="Fail if the universal config is not found.")
    p.add_argument("--project-root", default=".")
    p.add_argument("--data-root", default="data/direction", help="Historical feature/direction CSV root.")
    p.add_argument("--data-template", nargs="*", default=None, help="One or more CSV filename templates, e.g. {symbol}_{timeframe}_direction_training.csv")
    p.add_argument("--output-dir", default=None, help="Default: <live-root>/quantile_sweep_replay")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--sides", nargs="+", default=None)
    p.add_argument("--replay-start", default=None)
    p.add_argument("--replay-end", default=None)
    p.add_argument("--quantiles", nargs="+", type=float, default=[0.975, 0.980, 0.985, 0.990, 0.995])
    p.add_argument("--lookback-bars", type=int, default=4000, help="Fallback rolling lookback if config does not specify one.")
    p.add_argument("--min-history-bars", type=int, default=None, help="Fallback min history. Default derives from lookback.")
    p.add_argument("--fallback-threshold", type=float, default=0.5)
    p.add_argument("--min-gap-bars", type=int, default=12)
    p.add_argument("--min-side-probability", type=float, default=None)
    p.add_argument("--max-spread-pips", type=float, default=None, help="Optional replay spread filter.")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default=None)
    p.add_argument("--save-trades", action="store_true", help="Write per-trade rows. Can be large.")
    p.add_argument("--max-models", type=int, default=None, help="Debug/smoke-test limit.")
    args = p.parse_args()

    live_root = Path(args.live_root)
    output_dir = Path(args.output_dir) if args.output_dir else live_root / "quantile_sweep_replay"
    manifest, manifest_dir = _load_manifest(live_root, Path(args.manifest) if args.manifest else None)
    entries = _filter_entries([dict(x) for x in manifest["models"] if isinstance(x, dict)], args)
    if args.max_models is not None:
        entries = entries[: max(0, args.max_models)]
    if not entries:
        raise SystemExit("No staged live model entries matched the requested filters.")

    validate_forex_symbols(sorted({str(e.get("symbol")).upper() for e in entries}))
    universal_cfg = _load_universal_cfg(args, manifest, live_root, manifest_dir)

    model_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    print(
        f"Replaying {len(entries)} staged model(s) over quantiles with EXACT replay logic: "
        f"{', '.join(f'{q:.4f}' for q in args.quantiles)}",
        flush=True,
    )
    for idx, entry in enumerate(entries, 1):
        combo_id = str(entry.get("id") or f"{entry.get('symbol')}_{entry.get('model')}_{entry.get('side')}_epoch_{_safe_int(entry.get('epoch'), -1):03d}")
        try:
            base_cfg = _load_combo_cfg(
                entry,
                universal_cfg=universal_cfg,
                live_root=live_root,
                manifest_dir=manifest_dir,
                project_root=Path(args.project_root),
            )
            _apply_data_overrides(base_cfg, args)
            side = _normalise_side(entry.get("side"))
            model_path = _resolve_path(entry["model_path"], live_root=live_root, manifest_dir=manifest_dir, project_root=Path(args.project_root))
            scaler_path = _resolve_path(entry["scaler_path"], live_root=live_root, manifest_dir=manifest_dir, project_root=Path(args.project_root))
            features_path = _resolve_path(entry["features_path"], live_root=live_root, manifest_dir=manifest_dir, project_root=Path(args.project_root))
            print(f"[{idx}/{len(entries)}] {combo_id}: exact replay", flush=True)
            for q in args.quantiles:
                cfg = _deep_merge({}, base_cfg)
                _set_combo_quantile(cfg, side, float(q))
                q_token = _q_token(float(q))
                output_prefix = output_dir / "exact_replay" / combo_id / f"{combo_id}_{q_token}"
                summary = replay_symbol(
                    str(entry.get("symbol")).upper(),
                    cfg,
                    model_path=model_path,
                    scaler_path=scaler_path,
                    features_path=features_path,
                    eval_start=args.replay_start,
                    eval_end=args.replay_end,
                    output_prefix=str(output_prefix),
                    device=args.device,
                    verbose=False,
                )
                metrics = _metric_from_replay_summary(entry=entry, cfg=cfg, summary=summary, combo_id=combo_id, q=float(q))
                model_rows.append(metrics)
                if args.save_trades:
                    trades_df = _read_exact_replay_trades(summary, entry, combo_id, float(q))
                    if not trades_df.empty:
                        trade_frames.append(trades_df)
        except Exception as exc:
            failures.append({
                "id": entry.get("id"),
                "symbol": entry.get("symbol"),
                "model": entry.get("model"),
                "side": entry.get("side"),
                "error": str(exc),
            })
            print(f"[WARN] Failed {combo_id}: {exc}", file=sys.stderr, flush=True)

    model_results = pd.DataFrame(model_rows)
    quantile_summary = _aggregate_quantile_summary(model_results)
    symbol_summary = _aggregate_symbol_summary(model_results)
    best_model = _best_by_model(model_results)
    best_symbol = _best_by_symbol(symbol_summary)

    _write_csv(output_dir / "quantile_sweep_model_results.csv", model_results)
    _write_csv(output_dir / "quantile_sweep_quantile_summary.csv", quantile_summary)
    _write_csv(output_dir / "quantile_sweep_symbol_summary.csv", symbol_summary)
    _write_csv(output_dir / "quantile_sweep_best_by_model.csv", best_model)
    _write_csv(output_dir / "quantile_sweep_best_by_symbol.csv", best_symbol)
    if args.save_trades:
        trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        _write_csv(output_dir / "quantile_sweep_trades.csv", trades_all)
    _write_json(output_dir / "quantile_sweep_failures.json", failures)

    summary_payload = {
        "live_root": str(live_root),
        "output_dir": str(output_dir),
        "models_requested": len(entries),
        "models_completed": int(model_results["combo_id"].nunique()) if not model_results.empty else 0,
        "failures": failures,
        "quantiles": [float(q) for q in args.quantiles],
        "best_overall_quantile": quantile_summary.head(1).to_dict(orient="records") if not quantile_summary.empty else [],
    }
    _write_json(output_dir / "quantile_sweep_summary.json", summary_payload)

    print(f"\nWrote model results: {output_dir / 'quantile_sweep_model_results.csv'}")
    print(f"Wrote quantile summary: {output_dir / 'quantile_sweep_quantile_summary.csv'}")
    print(f"Wrote symbol summary: {output_dir / 'quantile_sweep_symbol_summary.csv'}")
    print(f"Wrote best-by-model: {output_dir / 'quantile_sweep_best_by_model.csv'}")
    print(f"Wrote best-by-symbol: {output_dir / 'quantile_sweep_best_by_symbol.csv'}")
    if failures:
        print(f"Completed with {len(failures)} failed model(s). See {output_dir / 'quantile_sweep_failures.json'}")
    if not quantile_summary.empty:
        best = quantile_summary.iloc[0]
        print(
            f"Best aggregate quantile={best['quantile']:.4f} trades={int(best['trades'])} "
            f"net={best['net_pips']:.1f} avg={best['average_net_pips']:.3f} "
            f"wr={best['win_rate']:.3f} score={best['total_deployment_score']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
