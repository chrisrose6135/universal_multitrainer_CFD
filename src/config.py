from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, *, require_spread_risk: bool = False) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if require_spread_risk:
        from .spread_risk_config import apply_spread_risk_config

        data = apply_spread_risk_config(data, config_path=path, require=True)
    return data


def load_config_with_spread_risk(path: str | Path) -> dict[str, Any]:
    """Load the main config and require the generated p95 spread-risk config.

    Training, grid search and live/demo runners should use this so they cannot
    accidentally run with stale hard-coded spread limits.
    """
    return load_config(path, require_spread_risk=True)



def load_config_with_optional_spread_risk(path: str | Path) -> dict[str, Any]:
    """Load main config and merge generated spread-risk settings if present.

    Unlike load_config_with_spread_risk(), this does not fail before the first
    dataset generation run has created config/generated_spread_risk.yaml.
    """
    data = load_config(path, require_spread_risk=False)
    from .spread_risk_config import apply_spread_risk_config

    return apply_spread_risk_config(data, config_path=path, require=False)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def deep_get(d: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(d: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    out = deepcopy(d)
    cur = out
    parts = dotted.split('.')
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
    return out
