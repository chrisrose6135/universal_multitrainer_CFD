from __future__ import annotations

from typing import Any, Iterable

# CFD-only tradable universe for the Pepperstone branch.
# The old module name is kept because many existing scripts import
# validate_forex_symbols(), pip_size(), etc.  In this branch those helpers now
# validate and convert CFD/index/metal instruments rather than FX pairs.
DEFAULT_CFD_SYMBOLS = (
    'US500',   # S&P 500 index CFD
    'NAS100',  # Nasdaq 100 index CFD
    'GER40',   # Germany 40 / DAX index CFD
    'XAUUSD',  # Spot gold CFD
    'XAGUSD',  # Spot silver CFD
)

# Backwards-compatible aliases used by old imports/scripts.
DEFAULT_FOREX_SYMBOLS = DEFAULT_CFD_SYMBOLS
VALID_CFD_SYMBOLS = set(DEFAULT_CFD_SYMBOLS)
VALID_FOREX = VALID_CFD_SYMBOLS

# Approximate Pepperstone/MT5 broker point sizes in price units.  If your broker
# suffixes symbols or uses different digits, override these in trading.point_overrides.
DEFAULT_POINT_OVERRIDES: dict[str, float] = {
    'US500': 0.01,
    'NAS100': 0.01,
    'GER40': 0.01,
    'XAUUSD': 0.01,
    'XAGUSD': 0.001,
}

# Model/replay reporting unit.  Existing code still calls this a "pip", but in
# the CFD branch it means the practical instrument unit used for TP/SL/replay:
# index points for indices, $0.10 for gold, and $0.01 for silver.
DEFAULT_PIP_SIZE_OVERRIDES: dict[str, float] = {
    'US500': 1.0,
    'NAS100': 1.0,
    'GER40': 1.0,
    'XAUUSD': 0.10,
    'XAGUSD': 0.01,
}

ALIASES: dict[str, str] = {
    'SPX500': 'US500',
    'SP500': 'US500',
    'S&P500': 'US500',
    'S&P_500': 'US500',
    'US500.CASH': 'US500',
    'US500CASH': 'US500',
    'USTEC': 'NAS100',
    'US100': 'NAS100',
    'NASDAQ': 'NAS100',
    'NAS100.CASH': 'NAS100',
    'NAS100CASH': 'NAS100',
    'DE40': 'GER40',
    'GERMANY40': 'GER40',
    'DAX40': 'GER40',
    'GER40.CASH': 'GER40',
    'GER40CASH': 'GER40',
    'GOLD': 'XAUUSD',
    'SILVER': 'XAGUSD',
}


def normalise_symbol(symbol: str) -> str:
    clean = str(symbol).upper().strip().replace(' ', '')
    return ALIASES.get(clean, clean)


def validate_forex_symbols(symbols: Iterable[str]) -> list[str]:
    """Validate CFD symbols.

    The function name is intentionally preserved for compatibility with the
    existing training/live modules.  This CFD-only branch rejects FX symbols and
    accepts only DEFAULT_CFD_SYMBOLS unless additional symbols are added to the
    config/code.
    """
    clean = [normalise_symbol(s) for s in symbols]
    bad = [s for s in clean if s not in VALID_CFD_SYMBOLS]
    if bad:
        raise ValueError(
            f'Unsupported/unknown CFD symbols: {bad}. '
            f'This CFD-only branch supports: {list(DEFAULT_CFD_SYMBOLS)}'
        )
    return clean


def _map_value(mapping: Any, symbol: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    symbol_u = normalise_symbol(symbol)
    for key in (symbol, symbol_u, symbol_u.lower()):
        if key in mapping and mapping[key] not in (None, ''):
            return mapping[key]
    return None


def _first_config_value(cfg: dict[str, Any] | None, symbol: str, names: tuple[str, ...]) -> Any:
    cfg = cfg or {}
    for section_name in ('trading', 'symbols', 'labels', 'risk', 'data', 'orders'):
        section = cfg.get(section_name, {}) or {}
        for name in names:
            value = _map_value(section.get(name), symbol)
            if value is not None:
                return value
    return None


def point_for_symbol(symbol: str, cfg: dict | None = None) -> float:
    """Return the MT5 broker point size in price units."""
    symbol_u = normalise_symbol(symbol)
    value = _first_config_value(cfg, symbol_u, ('point_overrides', 'point_size_by_symbol', 'point_sizes'))
    if value is not None:
        return float(value)
    return float(DEFAULT_POINT_OVERRIDES.get(symbol_u, 0.01))


def pip_size(symbol: str, cfg: dict | None = None) -> float:
    """Return the model/replay unit size in price units.

    Historical names are kept: "pip" now means CFD points/units.  Set
    trading.pip_size_by_symbol to override per instrument.
    """
    symbol_u = normalise_symbol(symbol)
    value = _first_config_value(
        cfg,
        symbol_u,
        ('pip_size_by_symbol', 'pip_sizes', 'price_unit_by_symbol', 'price_units_by_symbol'),
    )
    if value is not None:
        return float(value)
    return float(DEFAULT_PIP_SIZE_OVERRIDES.get(symbol_u, point_for_symbol(symbol_u, cfg)))


def pips_from_price_delta(symbol: str, price_delta: float, cfg: dict | None = None) -> float:
    unit = pip_size(symbol, cfg)
    if unit <= 0:
        raise ValueError(f'Invalid CFD price unit for {symbol}: {unit}')
    return float(price_delta) / float(unit)


def price_delta_from_pips(symbol: str, pips: float, cfg: dict | None = None) -> float:
    return float(pips) * pip_size(symbol, cfg)


def spread_points_to_pips(symbol: str, spread_points: float, cfg: dict | None = None) -> float:
    """Convert MT5 spread points to the model/replay unit."""
    spread_price_delta = float(spread_points) * point_for_symbol(symbol, cfg)
    return pips_from_price_delta(symbol, spread_price_delta, cfg)


def symbol_cfg_value(cfg: dict[str, Any] | None, section_name: str, base_key: str, symbol: str, default: Any = None) -> Any:
    """Return section[base_key_by_symbol][symbol] else section[base_key]."""
    section = (cfg or {}).get(section_name, {}) or {}
    value = _map_value(section.get(f'{base_key}_by_symbol'), symbol)
    if value is not None:
        return value
    value = _map_value(section.get(f'{base_key}s_by_symbol'), symbol)
    if value is not None:
        return value
    return section.get(base_key, default)
