from __future__ import annotations

from typing import Any

import pandas as pd

TARGET_PREFIXES = (
    'reason_target_',
    'label_',
    'target_',
)

KNOWN_TARGETS = {
    'direction_target',
    'decision_target',
    'outcome_target',
    'selected_side_target',
    'side_target',
    'trade_side_target',
    'target',
    'label',
    # Old auxiliary target names are kept here only to prevent accidental
    # feature leakage if older CSVs are inspected.
    'buy_tp_target',
    'buy_sl_target',
    'sell_tp_target',
    'sell_sl_target',
    'buy_quality_target',
    'sell_quality_target',
    'buy_net_pips_target',
    'sell_net_pips_target',
}

NON_LIVE_INPUT_COLUMNS = {
    'time',
    'time_utc',
    'datetime',
    'timestamp',
    'date',
    'symbol',
    'spread_source',
}


def is_leakage_column(column: str) -> bool:
    c = str(column).strip().lower()
    if c in KNOWN_TARGETS:
        return True
    if any(c.startswith(prefix) for prefix in TARGET_PREFIXES):
        return True
    if c.endswith('_target'):
        return True
    return False


def assert_no_feature_leakage(feature_columns: list[str]) -> None:
    bad = [str(c) for c in feature_columns if is_leakage_column(str(c))]
    if bad:
        raise ValueError(
            'Feature leakage detected. These future-derived target/label columns '
            f'were included as model inputs: {bad}. Remove them and retrain.'
        )


def choose_feature_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    fcfg = cfg.get('features', {}) or {}
    include = fcfg.get('include_columns') or []
    if include:
        requested = [str(c) for c in include if c in df.columns]
        assert_no_feature_leakage(requested)
        return requested

    exclude = {str(c).lower() for c in (fcfg.get('exclude_columns') or [])}
    exclude |= {c.lower() for c in KNOWN_TARGETS}
    exclude |= {c.lower() for c in NON_LIVE_INPUT_COLUMNS}

    cols: list[str] = []
    for c in df.columns:
        c_lower = str(c).lower()
        if c_lower in exclude or is_leakage_column(c_lower):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool:
            cols.append(c)

    assert_no_feature_leakage(cols)
    if not cols:
        raise ValueError('No numeric feature columns found. Set features.include_columns in the config.')
    return cols
