from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .spread_risk_config import symbol_max_spread_points


@dataclass
class GateResult:
    allow: bool
    reasons: list[str]
    diagnostics: dict[str, Any]


def _section(cfg: dict[str, Any]) -> dict[str, Any]:
    return (cfg.get('external_trade_filter') or cfg.get('analytics_gate') or {}) or {}


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_float(row: dict[str, Any] | pd.Series, *names: str) -> float | None:
    for name in names:
        if name in row and row.get(name) not in (None, ''):
            value = _float(row.get(name), None)
            if value is not None:
                return value
    return None


def _symbol_currencies(symbol: str) -> set[str]:
    text = str(symbol).upper().replace('/', '').replace('_', '')
    if len(text) >= 6:
        return {text[:3], text[3:6]}
    return {text}


@lru_cache(maxsize=8)
def _load_news_events(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(dict(row))
    return out


def _news_gate(symbol: str, row: dict[str, Any] | pd.Series, cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    gate_cfg = _section(cfg)
    news_cfg = gate_cfg.get('news') or {}
    if not _truthy(news_cfg.get('enabled'), False):
        return [], {'news_enabled': False}
    path = news_cfg.get('path') or news_cfg.get('calendar_csv')
    if not path:
        return [], {'news_enabled': True, 'news_path': None, 'news_events_loaded': 0}
    events = _load_news_events(str(path))
    ts = pd.to_datetime(row.get('time_utc') or row.get('time'), utc=True, errors='coerce')
    if pd.isna(ts):
        return [], {'news_enabled': True, 'news_path': str(path), 'news_events_loaded': len(events), 'warning': 'row time missing'}

    default_before = int(news_cfg.get('minutes_before', 30))
    default_after = int(news_cfg.get('minutes_after', 30))
    min_impact = str(news_cfg.get('min_impact', 'high')).lower()
    impact_rank = {'low': 1, 'medium': 2, 'med': 2, 'high': 3}
    min_rank = impact_rank.get(min_impact, 3)
    currencies = _symbol_currencies(symbol)
    blocked: list[dict[str, Any]] = []

    for event in events:
        raw_time = event.get('time_utc') or event.get('time') or event.get('datetime')
        ev_time = pd.to_datetime(raw_time, utc=True, errors='coerce')
        if pd.isna(ev_time):
            continue
        ev_symbol = str(event.get('symbol', '')).upper().strip()
        ev_currency = str(event.get('currency', '')).upper().strip()
        if ev_symbol and ev_symbol != str(symbol).upper():
            continue
        if ev_currency and ev_currency not in currencies:
            continue
        if not ev_symbol and not ev_currency:
            continue
        impact = str(event.get('impact', event.get('importance', 'high'))).lower()
        if impact_rank.get(impact, 3) < min_rank:
            continue
        before = int(float(event.get('minutes_before') or default_before))
        after = int(float(event.get('minutes_after') or default_after))
        if ev_time - pd.Timedelta(minutes=before) <= ts <= ev_time + pd.Timedelta(minutes=after):
            blocked.append({'time_utc': str(ev_time), 'currency': ev_currency, 'symbol': ev_symbol, 'impact': impact})

    diagnostics = {'news_enabled': True, 'news_path': str(path), 'news_events_loaded': len(events), 'blocking_events': blocked[:5]}
    return (['news_window'] if blocked else []), diagnostics


def external_trade_gate(symbol: str, side: str, row: dict[str, Any] | pd.Series, cfg: dict[str, Any]) -> GateResult:
    """Traditional analytics/news/risk gate for the simple direction model.

    This deliberately blocks trades; it never changes BUY into SELL or vice versa.
    """
    gate_cfg = _section(cfg)
    if not _truthy(gate_cfg.get('enabled'), True):
        return GateResult(True, [], {'enabled': False})

    side = str(side).upper().strip()
    reasons: list[str] = []
    diag: dict[str, Any] = {'enabled': True, 'side': side}

    spread_points = _row_float(row, 'spread_points', 'spread')
    max_spread = _float(gate_cfg.get('max_spread_points'), None)
    if max_spread is None:
        try:
            max_spread = symbol_max_spread_points(cfg, symbol, default=30.0)
        except Exception:
            max_spread = None
    diag['spread_points'] = spread_points
    diag['max_spread_points'] = max_spread
    if spread_points is not None and max_spread is not None and spread_points > max_spread:
        reasons.append('spread_high')

    if _truthy(gate_cfg.get('use_ema_alignment'), True):
        ema20 = _row_float(row, 'ema_20')
        ema50 = _row_float(row, 'ema_50')
        ema200 = _row_float(row, 'ema_200')
        if ema20 is not None and ema50 is not None:
            diag['ema20_minus_ema50'] = ema20 - ema50
            if side == 'BUY' and ema20 < ema50:
                reasons.append('ema_alignment_not_bullish')
            if side == 'SELL' and ema20 > ema50:
                reasons.append('ema_alignment_not_bearish')
        if _truthy(gate_cfg.get('use_ema200_filter'), False) and ema20 is not None and ema200 is not None:
            diag['ema20_minus_ema200'] = ema20 - ema200
            if side == 'BUY' and ema20 < ema200:
                reasons.append('ema200_not_bullish')
            if side == 'SELL' and ema20 > ema200:
                reasons.append('ema200_not_bearish')

    min_adx = _float(gate_cfg.get('min_adx'), None)
    adx = _row_float(row, 'adx_14', 'adx')
    diag['adx'] = adx
    if min_adx is not None and adx is not None and adx < min_adx:
        reasons.append('adx_low')

    rsi = _row_float(row, 'rsi_14', 'rsi')
    diag['rsi'] = rsi
    buy_max_rsi = _float(gate_cfg.get('buy_max_rsi'), 70.0)
    sell_min_rsi = _float(gate_cfg.get('sell_min_rsi'), 30.0)
    if rsi is not None:
        if side == 'BUY' and buy_max_rsi is not None and rsi > buy_max_rsi:
            reasons.append('buy_rsi_overbought')
        if side == 'SELL' and sell_min_rsi is not None and rsi < sell_min_rsi:
            reasons.append('sell_rsi_oversold')

    if _truthy(gate_cfg.get('use_macd_hist_confirmation'), False):
        macd_hist = _row_float(row, 'macd_hist', 'macd_hist_norm')
        diag['macd_hist'] = macd_hist
        if macd_hist is not None:
            if side == 'BUY' and macd_hist < 0:
                reasons.append('macd_hist_not_bullish')
            if side == 'SELL' and macd_hist > 0:
                reasons.append('macd_hist_not_bearish')

    min_atr_norm = _float(gate_cfg.get('min_atr_14_norm'), None)
    max_atr_norm = _float(gate_cfg.get('max_atr_14_norm'), None)
    atr_norm = _row_float(row, 'atr_14_norm')
    diag['atr_14_norm'] = atr_norm
    if atr_norm is not None:
        if min_atr_norm is not None and atr_norm < min_atr_norm:
            reasons.append('atr_low')
        if max_atr_norm is not None and atr_norm > max_atr_norm:
            reasons.append('atr_high')

    allowed_sessions = gate_cfg.get('allowed_sessions') or []
    if isinstance(allowed_sessions, str):
        allowed_sessions = [allowed_sessions]
    allowed_sessions = [str(x).lower().strip() for x in allowed_sessions if str(x).strip()]
    if allowed_sessions:
        session_ok = False
        for session in allowed_sessions:
            col = f'session_{session}'
            if bool(row.get(col, False)):
                session_ok = True
                break
        diag['allowed_sessions'] = allowed_sessions
        if not session_ok:
            reasons.append('session_not_allowed')

    news_reasons, news_diag = _news_gate(symbol, row, cfg)
    reasons.extend(news_reasons)
    diag.update(news_diag)

    return GateResult(not reasons, reasons, diag)
