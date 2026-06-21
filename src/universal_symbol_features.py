from __future__ import annotations

from typing import Any

import pandas as pd


def universal_symbols_from_cfg(cfg: dict[str, Any]) -> list[str]:
    ucfg = cfg.get('universal', {}) or {}
    symbols = ucfg.get('symbols') or (cfg.get('trading', {}) or {}).get('symbols') or []
    return [str(s).upper().strip() for s in symbols if str(s).strip()]


def universal_symbol_feature_prefix(cfg: dict[str, Any]) -> str:
    return str((cfg.get('universal', {}) or {}).get('symbol_feature_prefix', 'sym_'))


def add_universal_symbol_features(
    df: pd.DataFrame,
    cfg: dict[str, Any] | None,
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Add deterministic universal-model symbol context features.

    Universal models are trained on a pooled dataset, so they need a small amount
    of symbol context. The default is one-hot columns such as ``sym_US500``.
    This helper is intentionally safe to call during training, replay and live
    inference: if universal features are disabled it returns the original frame,
    and if columns already exist it overwrites/fills them consistently.
    """
    cfg = cfg or {}
    ucfg = cfg.get('universal', {}) or {}
    # Important: symbol one-hot features are for universal pooled models only.
    # Keep this strictly opt-in so existing per-symbol training/replay/live
    # pipelines are not changed just because a config has trading.symbols.
    if not bool(ucfg.get('enabled', False)):
        return df
    if not bool(ucfg.get('add_symbol_onehot_features', True)):
        return df
    symbols = universal_symbols_from_cfg(cfg)
    if not symbols:
        return df
    out = df.copy()
    prefix = universal_symbol_feature_prefix(cfg)
    symbol_col = str(ucfg.get('symbol_column', 'symbol'))
    active_symbol = str(symbol or cfg.get('_active_symbol') or ((cfg.get('trading', {}) or {}).get('symbols') or [''])[0] or '').upper().strip()
    if symbol_col in out.columns:
        row_symbols = out[symbol_col].astype(str).str.upper().str.strip()
    else:
        row_symbols = pd.Series(active_symbol, index=out.index)
        out[symbol_col] = active_symbol
    for sym in symbols:
        col = f'{prefix}{sym}'
        out[col] = (row_symbols == sym).astype('float32')
    if bool(ucfg.get('add_symbol_id_feature', False)):
        denom = max(len(symbols) - 1, 1)
        index_map = {sym: i / denom for i, sym in enumerate(symbols)}
        out[str(ucfg.get('symbol_id_feature_name', 'symbol_id_norm'))] = row_symbols.map(index_map).fillna(0.0).astype('float32')
    return out


def append_universal_symbol_feature_columns(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure config.features.include_columns includes universal symbol columns."""
    ucfg = cfg.get('universal', {}) or {}
    # Opt-in only; do not mutate symbol-specific configs.
    if not bool(ucfg.get('enabled', False)):
        return cfg
    if not bool(ucfg.get('add_symbol_onehot_features', True)):
        return cfg
    symbols = universal_symbols_from_cfg(cfg)
    if not symbols:
        return cfg
    features = cfg.setdefault('features', {})
    include = list(features.get('include_columns') or [])
    prefix = universal_symbol_feature_prefix(cfg)
    for sym in symbols:
        col = f'{prefix}{sym}'
        if col not in include:
            include.append(col)
    if bool(ucfg.get('add_symbol_id_feature', False)):
        col = str(ucfg.get('symbol_id_feature_name', 'symbol_id_norm'))
        if col not in include:
            include.append(col)
    features['include_columns'] = include
    return cfg
