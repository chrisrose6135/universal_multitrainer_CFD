from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import math

import pandas as pd
import yaml

from .forex import spread_points_to_pips

SPREAD_RISK_FILENAME = "generated_spread_risk.yaml"


def _project_config_dir(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path("config")
    path = Path(config_path)
    if path.suffix:
        return path.parent if str(path.parent) not in {"", "."} else Path("config")
    return path


def resolve_spread_risk_config_path(cfg: dict[str, Any] | None, config_path: str | Path | None = None) -> Path:
    """Return the generated spread-risk config path.

    The default is a file in the same folder as the main config:
        config/generated_spread_risk.yaml

    Operators can override the path with any of these keys:
        spread_risk_config_path
        spread_risk.config_path
        risk.spread_risk_config_path
    Relative paths are resolved relative to the main config folder.
    """
    cfg = cfg or {}
    raw = (
        cfg.get("spread_risk_config_path")
        or ((cfg.get("spread_risk") or {}).get("config_path") if isinstance(cfg.get("spread_risk"), dict) else None)
        or ((cfg.get("risk") or {}).get("spread_risk_config_path") if isinstance(cfg.get("risk"), dict) else None)
        or SPREAD_RISK_FILENAME
    )
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return _project_config_dir(config_path) / path


def _finite_positive(values: pd.Series, *, min_valid: float, max_valid: float | None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[numeric.notna()]
    numeric = numeric[numeric.map(lambda x: math.isfinite(float(x)))]
    numeric = numeric[numeric > float(min_valid)]
    if max_valid is not None:
        numeric = numeric[numeric <= float(max_valid)]
    return numeric.astype(float)


def calculate_spread_profile(df: pd.DataFrame, symbol: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Calculate symbol spread statistics from the training dataframe.

    Only the p95 threshold is used for trading/risk gating for now, but the
    metadata records a few extra descriptive statistics for auditing.
    """
    labels = cfg.get("labels", {}) or {}
    profile_cfg = cfg.get("spread_risk", {}) or {}
    spread_col = str(labels.get("spread_column", "spread_points"))
    fallback_points = float(labels.get("default_spread_points", 10.0))

    min_valid = float(profile_cfg.get("min_valid_spread_points", 0.0))
    max_valid_raw = profile_cfg.get("max_valid_spread_points", None)
    max_valid = float(max_valid_raw) if max_valid_raw not in (None, "") else None
    percentile = float(profile_cfg.get("percentile", 95.0))

    if spread_col in df.columns:
        spreads = _finite_positive(df[spread_col], min_valid=min_valid, max_valid=max_valid)
    else:
        spreads = pd.Series(dtype=float)

    used_fallback = bool(spreads.empty)
    if used_fallback:
        spreads = pd.Series([fallback_points], dtype=float)

    q = max(0.0, min(100.0, percentile)) / 100.0
    median_points = float(spreads.quantile(0.50))
    p95_points = float(spreads.quantile(q))
    mean_points = float(spreads.mean())
    max_points = float(spreads.max())
    min_points = float(spreads.min())

    return {
        "symbol": str(symbol).upper(),
        "spread_column": spread_col,
        "spread_unit_conversion": "symbol_point_to_cfd_unit",
        "percentile": percentile,
        "valid_spread_count": int(len(spreads)),
        "used_fallback": used_fallback,
        "fallback_default_spread_points": fallback_points,
        "min_valid_spread_points": min_valid,
        "max_valid_spread_points": max_valid,
        "min_spread_points": round(min_points, 6),
        "median_spread_points": round(median_points, 6),
        "mean_spread_points": round(mean_points, 6),
        "p95_spread_points": round(p95_points, 6),
        "max_spread_points": round(max_points, 6),
        "median_spread_pips": round(spread_points_to_pips(symbol, median_points, cfg), 6),
        "p95_spread_pips": round(spread_points_to_pips(symbol, p95_points, cfg), 6),
        "selected_default_spread_points": round(median_points, 6),
        "selected_default_spread_pips": round(spread_points_to_pips(symbol, median_points, cfg), 6),
        "selected_max_spread_points": round(p95_points, 6),
        "selected_max_spread_pips": round(spread_points_to_pips(symbol, p95_points, cfg), 6),
    }


def apply_symbol_spread_profile_to_cfg(cfg: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Return a cfg copy using this symbol's median and p95 spread settings."""
    out = deepcopy(cfg)
    labels = out.setdefault("labels", {})
    labels["default_spread_points"] = float(profile["selected_default_spread_points"])
    labels["max_spread_pips"] = float(profile["selected_max_spread_pips"])
    out.setdefault("risk", {})["max_spread_points"] = float(profile["selected_max_spread_points"])
    out["_active_symbol"] = str(profile["symbol"]).upper()
    return out


def build_spread_risk_config(
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    *,
    source_config: str | Path,
    timeframe: str,
) -> dict[str, Any]:
    symbols = sorted(str(s).upper() for s in profiles)
    max_spread_points_by_symbol = {s: float(profiles[s]["selected_max_spread_points"]) for s in symbols}
    max_spread_pips_by_symbol = {s: float(profiles[s]["selected_max_spread_pips"]) for s in symbols}
    default_spread_points_by_symbol = {s: float(profiles[s]["selected_default_spread_points"]) for s in symbols}
    default_spread_pips_by_symbol = {s: float(profiles[s]["selected_default_spread_pips"]) for s in symbols}

    labels = cfg.get("labels", {}) or {}
    return {
        "generated_spread_risk": {
            "version": 1,
            "generated_by": "src.prepare_direction_dataset",
            "source_config": str(source_config),
            "timeframe": str(timeframe),
            "threshold_stat": "p95",
            "spread_column": str(labels.get("spread_column", "spread_points")),
            "spread_unit_conversion": "symbol_point_to_cfd_unit",
            "symbols": symbols,
        },
        "risk": {
            "max_spread_points_by_symbol": max_spread_points_by_symbol,
        },
        "labels": {
            "default_spread_points_by_symbol": default_spread_points_by_symbol,
            "default_spread_pips_by_symbol": default_spread_pips_by_symbol,
            "max_spread_pips_by_symbol": max_spread_pips_by_symbol,
        },
        "spread_profile": {s: profiles[s] for s in symbols},
    }


def write_spread_risk_config(
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    *,
    source_config: str | Path,
    timeframe: str,
) -> Path:
    path = resolve_spread_risk_config_path(cfg, source_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_spread_risk_config(profiles, cfg, source_config=source_config, timeframe=timeframe)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def _merge_section_dict(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_section_dict(base[key], value)
        else:
            base[key] = value


def load_spread_risk_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or "generated_spread_risk" not in data:
        raise ValueError(f"{path} is not a generated spread risk config")
    return data


def apply_spread_risk_config(
    cfg: dict[str, Any],
    *,
    config_path: str | Path | None,
    require: bool = True,
) -> dict[str, Any]:
    """Merge generated p95 spread risk config into the main cfg.

    Training, grid search and live/demo runners should call this with
    require=True. Data preparation should not, because it creates the file.
    """
    out = deepcopy(cfg)
    risk_path = resolve_spread_risk_config_path(out, config_path)
    if not risk_path.exists():
        if require:
            raise FileNotFoundError(
                f"Required generated spread risk config was not found: {risk_path}. "
                "Run src.prepare_direction_dataset first so it can calculate per-symbol p95 spread thresholds."
            )
        out["_spread_risk_config_path"] = str(risk_path)
        out["_spread_risk_config_loaded"] = False
        return out

    spread_cfg = load_spread_risk_config(risk_path)
    for section in ("risk", "labels"):
        if isinstance(spread_cfg.get(section), dict):
            _merge_section_dict(out.setdefault(section, {}), spread_cfg[section])
    out["spread_profile"] = spread_cfg.get("spread_profile", {})
    out["generated_spread_risk"] = spread_cfg.get("generated_spread_risk", {})
    out["_spread_risk_config_path"] = str(risk_path)
    out["_spread_risk_config_loaded"] = True
    return out


def symbol_max_spread_points(cfg: dict[str, Any], symbol: str | None = None, default: float = 30.0) -> float:
    risk = cfg.get("risk", {}) or {}
    symbol_u = str(symbol or cfg.get("_active_symbol") or "").upper()
    by_symbol = risk.get("max_spread_points_by_symbol") or {}
    if isinstance(by_symbol, dict) and symbol_u:
        for key, value in by_symbol.items():
            if str(key).upper() == symbol_u:
                return float(value)
    return float(risk.get("max_spread_points", default))


def symbol_default_spread_points(cfg: dict[str, Any], symbol: str | None = None, default: float = 2.0) -> float:
    labels = cfg.get("labels", {}) or {}
    symbol_u = str(symbol or cfg.get("_active_symbol") or "").upper()
    by_symbol = labels.get("default_spread_points_by_symbol") or {}
    if isinstance(by_symbol, dict) and symbol_u:
        for key, value in by_symbol.items():
            if str(key).upper() == symbol_u:
                return float(value)
    return float(labels.get("default_spread_points", default))


def symbol_max_spread_pips(cfg: dict[str, Any], symbol: str | None = None) -> float | None:
    labels = cfg.get("labels", {}) or {}
    symbol_u = str(symbol or cfg.get("_active_symbol") or "").upper()
    by_symbol = labels.get("max_spread_pips_by_symbol") or {}
    if isinstance(by_symbol, dict) and symbol_u:
        for key, value in by_symbol.items():
            if str(key).upper() == symbol_u:
                return float(value)
    value = labels.get("max_spread_pips", None)
    return float(value) if value not in (None, "") else None
