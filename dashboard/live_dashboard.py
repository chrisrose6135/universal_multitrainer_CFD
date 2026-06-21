#!/usr/bin/env python3
"""
Read-only live trading dashboard for the direction-policy / CFD model logs.

The dashboard does not control trading. It reads CSV/JSON log files produced by
live/paper/demo runners and exposes authenticated monitoring views for:
- overall profit/loss
- active open trades
- total closed trades
- per-symbol / per-model trade performance
- staged ensemble models and replay-selected checkpoint metrics
- TP-relative profit-protection / position-manager events
- recent signals and raw files
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required. Install with: pip install -r requirements-dashboard.txt") from exc

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware


APP_TITLE = "AI CFD Live Dashboard"
DEFAULT_CONFIG = {
    "server": {
        "host": "127.0.0.1",
        "port": 8050,
        "refresh_seconds": 10,
        "cache_ttl_seconds": 2,
    },
    "security": {
        "username_env": "DASHBOARD_USER",
        "password_env": "DASHBOARD_PASSWORD",
        "allow_config_credentials": False,
        "username": "",
        "password": "",
    },
    "paths": {
        "project_root": ".",
        "signal_logs": [
            "logs/live_direction_ensemble_signals*.csv",
            "For Live Trading/logs/live_direction_ensemble_signals*.csv",
            "logs/live_direction_signals*.csv",
            "logs/live_signals*.csv",
            "logs/*signal*.csv",
            "logs/*signals*.csv",
        ],
        "trade_logs": [
            "logs/live_direction_trades*.csv",
            "For Live Trading/logs/live_direction_trades*.csv",
            "logs/live_trades*.csv",
            "logs/paper_trades*.csv",
            "logs/*trade*.csv",
        ],
        "paper_state_files": [
            "logs/live_direction_open_trades.json",
            "For Live Trading/logs/live_direction_open_trades.json",
            "logs/paper_trading_state.json",
            "logs/paper_state.json",
            "logs/paper_trades.json",
        ],
        "summary_files": [
            "logs/live_direction_summary.json",
            "For Live Trading/logs/live_direction_summary.json",
            "logs/live_summary.json",
            "logs/saved_model_replay_report.json",
            "logs/saved_model_replay_all_epochs_summary.json",
        ],
        "model_selection_files": [
            "live_model_selection_table.csv",
            "For Live Trading/live_model_selection_table.csv",
        ],
        "manifest_files": [
            "live_ensemble_manifest.json",
            "For Live Trading/live_ensemble_manifest.json",
            "live_ensemble_manifest.yaml",
            "For Live Trading/live_ensemble_manifest.yaml",
        ],
        "bar_state_files": [
            "state/live_direction_ensemble_bar_state.json",
            "For Live Trading/state/live_direction_ensemble_bar_state.json",
        ],
        "exclude_patterns": [
            "*saved_model_replay*decisions.csv",
            "*saved_model_replay*trades.csv",
            "*training_replay*decisions.csv",
            "*training_replay*trades.csv",
            "*direction_replay*decisions.csv",
            "*direction_replay*trades.csv",
            "*replay*decisions.csv",
            "*replay*trades.csv",
            "*threshold_sweep*",
        ],
    },
    "display": {
        "default_limit": 100,
        "max_rows": 5000,
        "symbol": "US500",
        "timezone_label": "UTC/broker-corrected",
    },
}


security = HTTPBasic()
app = FastAPI(title=APP_TITLE, version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=[],
)


@dataclass
class CacheEntry:
    timestamp: float
    value: Any


class DashboardState:
    def __init__(self) -> None:
        self.config: Dict[str, Any] = deepcopy(DEFAULT_CONFIG)
        self.config_path: Optional[Path] = None
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl_seconds = 2.0


STATE = DashboardState()


# ------------------------- config / file discovery -------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dashboard config not found: {path}")
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config")
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = deep_merge(DEFAULT_CONFIG, loaded)
        STATE.config_path = path
    STATE.config = cfg
    STATE.cache_ttl_seconds = float((cfg.get("server") or {}).get("cache_ttl_seconds", 2.0))
    return cfg


def get_project_root() -> Path:
    root = (STATE.config.get("paths") or {}).get("project_root", ".")
    return Path(root).expanduser().resolve()


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def _match_any(path: Path, patterns: List[str]) -> bool:
    text = str(path)
    name = path.name
    for pat in patterns:
        if Path(text).match(pat) or Path(name).match(pat):
            return True
        if glob.fnmatch.fnmatch(text, pat) or glob.fnmatch.fnmatch(name, pat):
            return True
    return False


def find_files(patterns: List[str], excludes: Optional[List[str]] = None) -> List[Path]:
    root = get_project_root()
    found: List[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        full_pattern = str(p if p.is_absolute() else root / pattern)
        for item in glob.glob(full_pattern, recursive=True):
            path = Path(item).resolve()
            if path.is_file():
                found.append(path)
    unique = sorted(set(found), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    excludes = excludes or []
    if excludes:
        unique = [p for p in unique if not _match_any(p, excludes)]
    return unique


def cached(key: str, loader):
    now = time.time()
    entry = STATE.cache.get(key)
    if entry and (now - entry.timestamp) < STATE.cache_ttl_seconds:
        return entry.value
    value = loader()
    STATE.cache[key] = CacheEntry(timestamp=now, value=value)
    return value


# ------------------------------ data helpers ------------------------------

def _jsonify_cell(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _safe_read_csv(path: Path, *, add_source_columns: bool = False) -> pd.DataFrame:
    """Read dashboard CSV files without dtype inference failures.

    Live signal/trade logs can contain mixed values in the same column across
    runs: booleans, floats, empty cells, JSON-like diagnostics and path strings.
    Pandas' default dtype inference can raise/emit DtypeWarning on those files.
    Read as strings first, then let the dashboard convert numeric columns with
    ``pd.to_numeric(..., errors="coerce")`` where needed.
    """
    read_kwargs = {
        "dtype": str,
        "keep_default_na": False,
        "na_values": [],
        "low_memory": False,
    }
    try:
        df = pd.read_csv(path, **read_kwargs)
    except Exception as exc1:
        fallback_kwargs = dict(read_kwargs)
        fallback_kwargs.pop("low_memory", None)  # unsupported by python engine
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", **fallback_kwargs)
        df["_read_warning"] = f"strict CSV parse failed; skipped malformed rows: {exc1}"
    if add_source_columns:
        df["_source_file"] = path.name
        df["_source_mtime"] = str(path.stat().st_mtime)
    return df


def read_csvs(paths: List[Path], max_rows: int) -> pd.DataFrame:
    """Read CSV logs robustly.

    Existing live CSVs can change schema after a logging patch. If pandas strict
    parsing fails, retry with the python engine and skip malformed legacy rows
    instead of breaking the dashboard.
    """
    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(_safe_read_csv(path, add_source_columns=True))
        except Exception as exc:
            frames.append(pd.DataFrame([{"_source_file": path.name, "_read_error": str(exc)}]))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if len(out) > max_rows:
        out = out.tail(max_rows).copy()
    return out


def read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return None
    if "\n" in text and not text.startswith("{") and not text.startswith("["):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "bar_time", "time", "time_utc", "timestamp", "entry_time", "open_time",
        "exit_time", "close_time", "created_at", "datetime", "date"
    ]
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def dataframe_to_records(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    time_col = detect_time_column(df)
    if time_col:
        try:
            df = df.copy()
            df["_sort_time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
            df = df.sort_values("_sort_time")
            df = df.drop(columns=["_sort_time"], errors="ignore")
        except Exception:
            pass
    df = df.tail(limit)
    df = df.where(pd.notnull(df), None)
    return [{str(k): _jsonify_cell(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def first_existing_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


PIPS_COLUMNS = [
    "net_pips", "pips", "profit_pips", "pnl_pips",
    "realised_pips", "realized_pips", "closed_pips",
]
MONEY_COLUMNS = [
    "pnl", "profit", "net_profit", "realised_profit", "realized_profit",
    "closed_pnl", "account_profit",
]
FLOATING_MONEY_COLUMNS = ["floating_profit", "current_profit", "profit", "pnl"]


def _position_manager_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    model_col = first_existing_col(df, ["model", "model_label"])
    side_model_col = first_existing_col(df, ["side_model"])
    combo_col = first_existing_col(df, ["combo_id"])
    mask = pd.Series([False] * len(df), index=df.index)
    if model_col:
        mask = mask | df[model_col].astype(str).str.lower().eq("position_manager")
    if side_model_col:
        mask = mask | df[side_model_col].astype(str).str.lower().eq("tp_profit_protection_exit")
    if combo_col:
        mask = mask | df[combo_col].astype(str).str.lower().str.startswith("position_manager")
    return mask


def _normalise_side_text(value: Any) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "BUY_ONLY"}:
        return "BUY"
    if side in {"SHORT", "SELL_ONLY"}:
        return "SELL"
    return side


def _truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def count_values(df: pd.DataFrame, col_names: List[str]) -> Dict[str, int]:
    col = first_existing_col(df, col_names)
    if not col or df.empty:
        return {}
    return {str(k): int(v) for k, v in df[col].fillna("<missing>").value_counts().to_dict().items()}


def numeric_series(df: pd.DataFrame, col_names: List[str]) -> pd.Series:
    col = first_existing_col(df, col_names)
    if not col or df.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def numeric_sum(df: pd.DataFrame, col_names: List[str]) -> float:
    s = numeric_series(df, col_names)
    if s.empty:
        return 0.0
    return float(s.fillna(0.0).sum())


def numeric_mean(df: pd.DataFrame, col_names: List[str]) -> Optional[float]:
    s = numeric_series(df, col_names).dropna()
    if s.empty:
        return None
    return float(s.mean())


def bool_count(df: pd.DataFrame, col_names: List[str]) -> int:
    col = first_existing_col(df, col_names)
    if not col or df.empty:
        return 0
    return int(pd.Series(df[col]).astype(str).str.lower().isin(["true", "1", "yes", "y"]).sum())


def win_rate_from_trades(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    pnl_col = first_existing_col(df, ["net_pips", "pips", "profit_pips", "pnl_pips", "realised_pips", "realized_pips", "profit"])
    if pnl_col:
        pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
        if not pnl.empty:
            return float((pnl > 0).mean())
    result_col = first_existing_col(df, ["result", "outcome", "status", "close_reason"])
    if result_col:
        vals = df[result_col].astype(str).str.lower()
        wins = vals.str.contains("win|tp|profit|closed_win", regex=True).sum()
        losses = vals.str.contains("loss|sl|lose|closed_loss", regex=True).sum()
        total = wins + losses
        if total:
            return float(wins / total)
    return None


def max_drawdown_from_pips(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    pnl_col = first_existing_col(df, ["net_pips", "pips", "profit_pips", "pnl_pips", "realised_pips", "realized_pips", "profit"])
    if not pnl_col:
        return None
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    if pnl.empty:
        return None
    equity = pnl.cumsum()
    dd = equity.cummax() - equity
    return float(dd.max())


def infer_model_label(row: Dict[str, Any]) -> str:
    """Return a stable deployment-model label.

    The ensemble runner logs several side-specific models for the same symbol and
    architecture. Grouping by the old ``model`` column alone collapses them into
    one row, so prefer ``combo_id`` or symbol/model/side/epoch details. Position
    manager rows are kept separate from model signals.
    """
    combo_id = row.get("combo_id")
    if combo_id not in (None, ""):
        return str(combo_id)

    model = row.get("model") or row.get("model_name")
    side_model = row.get("side_model") or row.get("side") or row.get("direction")
    symbol = row.get("symbol") or row.get("Symbol")
    epoch = row.get("epoch")
    if str(model or "").lower() == "position_manager":
        return "position_manager:tp_profit_protection_exit"
    if symbol not in (None, "") and model not in (None, ""):
        label = f"{symbol}:{model}"
        if side_model not in (None, "", "SKIP", "ERROR"):
            label += f":{str(side_model).lower()}"
        if epoch not in (None, "", "unknown"):
            label += f":e{epoch}"
        return label

    for key in ("model_label", "model_name", "model_path", "checkpoint_path"):
        value = row.get(key)
        if value not in (None, ""):
            text = str(value)
            if key.endswith("path"):
                return Path(text).name
            return text
    if symbol not in (None, ""):
        return str(symbol)
    source = row.get("_source_file")
    return str(source or "unknown")


def add_model_label(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    records = out.where(pd.notnull(out), None).to_dict(orient="records")
    out["model_label"] = [infer_model_label(r) for r in records]
    return out


# ---------------------- trade status / open/closed split -------------------

def _status_series(df: pd.DataFrame) -> pd.Series:
    col = first_existing_col(df, ["status", "trade_status", "state", "position_status", "final_status"])
    if col:
        return df[col].astype(str).str.lower()
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def split_open_closed_trades(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    status = _status_series(df)
    status_has_value = status.str.strip() != ""
    exit_col = first_existing_col(df, ["exit_time", "close_time", "closed_time", "time_closed", "close_time_utc"])
    realised_col = first_existing_col(df, [
        "net_pips", "realised_pips", "realized_pips", "profit_pips",
        "pnl_pips", "closed_pips", "realised_profit", "realized_profit",
    ])

    explicit_open = status.str.contains("open|active|live|running", regex=True, na=False)
    explicit_closed = status.str.contains("closed|filled_exit|exit|tp|sl|win|loss", regex=True, na=False)
    open_mask = explicit_open.copy()
    closed_mask = explicit_closed.copy()

    if exit_col:
        exit_has_value = df[exit_col].notna() & (df[exit_col].astype(str).str.strip() != "")
        closed_mask = closed_mask | exit_has_value
        # A snapshot row with a blank close_time and no explicit CLOSED status is open.
        open_mask = open_mask | (~exit_has_value & ~explicit_closed)

    # Live open-position snapshots contain floating ``pnl``/``profit``. Do not
    # treat those as realised P/L. Only infer closed from a realised column where
    # the row has not explicitly said it is open.
    if realised_col:
        realised_has_value = df[realised_col].notna() & (df[realised_col].astype(str).str.strip() != "")
        closed_mask = closed_mask | (realised_has_value & ~explicit_open)

    # If there is no explicit status/exit/realised-P/L signal, assume CSV trade
    # rows are closed rather than falsely reporting all of them as active.
    if not status_has_value.any() and not exit_col and not realised_col:
        closed_mask = pd.Series([True] * len(df), index=df.index)
        open_mask = pd.Series([False] * len(df), index=df.index)

    open_df = df[open_mask & ~closed_mask].copy()
    closed_df = df[closed_mask].copy()
    return open_df, closed_df


def _rows_from_possible_collection(value: Any, source: str, forced_status: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault("_source_file", source)
                if forced_status and "status" not in {str(k).lower() for k in row.keys()}:
                    row["status"] = forced_status
                rows.append(row)
    return rows


def extract_state_trades(payload: Any, source_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    open_rows: List[Dict[str, Any]] = []
    closed_rows: List[Dict[str, Any]] = []
    if payload is None:
        return open_rows, closed_rows
    if isinstance(payload, list):
        # Treat a bare list as trade rows and let status inference handle them.
        rows = _rows_from_possible_collection(payload, source_name)
        df = pd.DataFrame(rows)
        o, c = split_open_closed_trades(df)
        return o.to_dict(orient="records"), c.to_dict(orient="records")
    if not isinstance(payload, dict):
        return open_rows, closed_rows

    open_keys = ["open_trades", "open_positions", "active_trades", "positions", "open"]
    closed_keys = ["closed_trades", "closed_positions", "history", "completed_trades", "closed"]
    for key in open_keys:
        if key in payload:
            open_rows.extend(_rows_from_possible_collection(payload[key], source_name, forced_status="open"))
    for key in closed_keys:
        if key in payload:
            closed_rows.extend(_rows_from_possible_collection(payload[key], source_name, forced_status="closed"))
    if not open_rows and not closed_rows and any(k in payload for k in ("symbol", "side", "direction", "ticket", "order_id")):
        rows = _rows_from_possible_collection(payload, source_name)
        df = pd.DataFrame(rows)
        o, c = split_open_closed_trades(df)
        open_rows.extend(o.to_dict(orient="records"))
        closed_rows.extend(c.to_dict(orient="records"))
    return open_rows, closed_rows


# ------------------------------- summaries --------------------------------

def summarize_position_manager(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "events": 0,
            "close_attempted": 0,
            "close_sent": 0,
            "giveback_exits": 0,
            "sustained_exits": 0,
            "reason_counts": {},
            "latest_event": None,
        }
    pm = df[_position_manager_mask(df)].copy()
    if pm.empty:
        return {
            "events": 0,
            "close_attempted": 0,
            "close_sent": 0,
            "giveback_exits": 0,
            "sustained_exits": 0,
            "reason_counts": {},
            "latest_event": None,
        }
    reason_col = first_existing_col(pm, ["reason"])
    reasons = pm[reason_col].astype(str) if reason_col else pd.Series([""] * len(pm), index=pm.index)
    records = dataframe_to_records(pm, 1)
    return {
        "events": int(len(pm)),
        "close_attempted": bool_count(pm, ["close_attempted", "order_attempted"]),
        "close_sent": bool_count(pm, ["close_sent", "order_sent"]),
        "giveback_exits": int(reasons.str.contains("tp_profit_giveback_exit", regex=False, na=False).sum()),
        "sustained_exits": int(reasons.str.contains("tp_profit_sustained_exit", regex=False, na=False).sum()),
        "reason_counts": count_values(pm, ["reason"]),
        "latest_event": records[-1] if records else None,
    }


def summarize_signals(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "signals": 0,
            "model_signals": 0,
            "latest_signal": None,
            "direction_counts": {},
            "final_decision_counts": {},
            "block_reason_counts": {},
            "orders_sent": 0,
            "paper_opened": 0,
            "avg_selected_probability": None,
            "avg_selected_ev": None,
            "avg_selected_tp_probability": None,
        }
    model_df = df[~_position_manager_mask(df)].copy()
    records = dataframe_to_records(model_df if not model_df.empty else df, 1)
    latest = records[-1] if records else None
    return {
        "signals": int(len(df)),
        "model_signals": int(len(model_df)),
        "latest_signal": latest,
        "direction_counts": count_values(model_df, ["direction", "raw_direction", "selected_side", "model_direction"]),
        "final_decision_counts": count_values(model_df, ["final_decision", "model_decision", "decision"]),
        "block_reason_counts": count_values(model_df, ["block_reason", "risk_reason", "reason", "gate_reason"]),
        "orders_sent": bool_count(model_df, ["order_sent"]),
        "paper_opened": bool_count(model_df, ["paper_opened"]),
        "avg_selected_probability": numeric_mean(model_df, ["selected_probability", "selected_prob", "probability"]),
        "avg_selected_ev": numeric_mean(model_df, ["selected_expected_value_pips", "expected_value_pips", "selected_ev_pips"]),
        "avg_selected_tp_probability": numeric_mean(model_df, ["selected_tp_probability", "tp_probability"]),
    }


def _group_sum(df: pd.DataFrame, group_col: Optional[str], value_col: Optional[str]) -> Dict[str, float]:
    if df.empty or not group_col or not value_col:
        return {}
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0.0)
    return {str(k): float(v) for k, v in tmp.groupby(group_col)[value_col].sum().to_dict().items()}


def max_drawdown_from_money(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    pnl_col = first_existing_col(df, MONEY_COLUMNS)
    if not pnl_col:
        return None
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    if pnl.empty:
        return None
    equity = pnl.cumsum()
    dd = equity.cummax() - equity
    return float(dd.max())


def summarize_trades(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "trades": 0,
            "net_pips": 0.0,
            "net_profit": 0.0,
            "average_pips": None,
            "average_profit": None,
            "win_rate": None,
            "max_drawdown_pips": None,
            "max_drawdown_profit": None,
            "side_counts": {},
            "side_pips": {},
            "side_profit": {},
            "symbol_counts": {},
            "symbol_pips": {},
            "symbol_profit": {},
            "latest_trade": None,
        }
    pips_col = first_existing_col(df, PIPS_COLUMNS)
    money_col = first_existing_col(df, MONEY_COLUMNS)
    side_col = first_existing_col(df, ["side", "direction", "trade_side"])
    symbol_col = first_existing_col(df, ["symbol", "ticker", "instrument"])
    records = dataframe_to_records(df, 1)
    return {
        "trades": int(len(df)),
        "net_pips": numeric_sum(df, PIPS_COLUMNS),
        "net_profit": numeric_sum(df, MONEY_COLUMNS),
        "average_pips": numeric_mean(df, PIPS_COLUMNS),
        "average_profit": numeric_mean(df, MONEY_COLUMNS),
        "win_rate": win_rate_from_trades(df),
        "max_drawdown_pips": max_drawdown_from_pips(df),
        "max_drawdown_profit": max_drawdown_from_money(df),
        "side_counts": count_values(df, ["side", "direction", "trade_side"]),
        "side_pips": _group_sum(df, side_col, pips_col),
        "side_profit": _group_sum(df, side_col, money_col),
        "symbol_counts": count_values(df, ["symbol", "ticker", "instrument"]),
        "symbol_pips": _group_sum(df, symbol_col, pips_col),
        "symbol_profit": _group_sum(df, symbol_col, money_col),
        "latest_trade": records[-1] if records else None,
    }


def _maybe_attach_staged_metrics(item: Dict[str, Any], staged_by_id: Dict[str, Dict[str, Any]]) -> None:
    key = str(item.get("model") or "")
    staged = staged_by_id.get(key)
    if not staged:
        return
    for field in (
        "symbol", "side", "timeframe", "epoch", "deployment_score", "deployment_usable",
        "deployment_reason", "trades", "net_pips", "win_rate", "average_net_pips",
        "max_drawdown_pips",
    ):
        if field in staged:
            item[f"selected_{field}"] = staged[field]


def summarize_models(
    signal_df: pd.DataFrame,
    closed_trade_df: pd.DataFrame,
    open_trade_df: pd.DataFrame,
    staged_models: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    staged_models = staged_models or []
    staged_by_id = {str(x.get("id") or x.get("combo_id") or ""): x for x in staged_models}

    if not signal_df.empty:
        sig = add_model_label(signal_df[~_position_manager_mask(signal_df)].copy())
        symbol_col = first_existing_col(sig, ["symbol", "ticker", "instrument"])
        decision_col = first_existing_col(sig, ["final_decision", "model_decision", "decision"])
        selected_col = first_existing_col(sig, ["order_selected_by_ensemble"])
        for model, group in sig.groupby("model_label", dropna=False):
            key = str(model)
            item = rows.setdefault(key, {"model": key})
            item["signals"] = int(len(group))
            item["orders_sent"] = bool_count(group, ["order_sent"])
            item["paper_opened"] = bool_count(group, ["paper_opened"])
            if selected_col:
                item["ensemble_selected"] = int(_truthy_series(group[selected_col]).sum())
            if decision_col:
                vals = group[decision_col].astype(str).str.upper()
                item["allow_signals"] = int((vals == "ALLOW").sum())
                item["block_signals"] = int((vals == "BLOCK").sum())
                item["skip_signals"] = int((vals == "SKIP").sum())
                item["error_signals"] = int((vals == "ERROR").sum())
            if symbol_col:
                item["symbols"] = ", ".join(sorted({str(x) for x in group[symbol_col].dropna().unique()}))
            item["avg_selected_probability"] = numeric_mean(group, ["selected_probability", "selected_prob", "probability"])
            item["avg_signal_rank_score"] = numeric_mean(group, ["signal_rank_score"])

    for source_df, kind in ((closed_trade_df, "closed"), (open_trade_df, "open")):
        if source_df.empty:
            continue
        trades = add_model_label(source_df)
        for model, group in trades.groupby("model_label", dropna=False):
            key = str(model)
            item = rows.setdefault(key, {"model": key})
            if kind == "closed":
                s = summarize_trades(group)
                item["closed_trades"] = s["trades"]
                item["net_pips"] = s["net_pips"]
                item["net_profit"] = s["net_profit"]
                item["average_pips"] = s["average_pips"]
                item["average_profit"] = s["average_profit"]
                item["win_rate"] = s["win_rate"]
                item["max_drawdown_pips"] = s["max_drawdown_pips"]
                item["side_counts"] = s["side_counts"]
                item["side_pips"] = s["side_pips"]
            else:
                item["open_trades"] = int(len(group))
                item["open_profit"] = numeric_sum(group, FLOATING_MONEY_COLUMNS)
                item["open_pips"] = numeric_sum(group, PIPS_COLUMNS)

    # Surface staged models even before they have generated signals.
    for staged in staged_models:
        key = str(staged.get("id") or staged.get("combo_id") or "")
        if not key:
            symbol = staged.get("symbol", "")
            model = staged.get("model", "")
            side = staged.get("side", "")
            epoch = staged.get("epoch", "")
            key = f"{symbol}_{model}_{side}_epoch_{epoch}"
        item = rows.setdefault(key, {"model": key})
        _maybe_attach_staged_metrics(item, {key: staged})

    for item in rows.values():
        item.setdefault("signals", 0)
        item.setdefault("allow_signals", 0)
        item.setdefault("block_signals", 0)
        item.setdefault("skip_signals", 0)
        item.setdefault("error_signals", 0)
        item.setdefault("orders_sent", 0)
        item.setdefault("paper_opened", 0)
        item.setdefault("ensemble_selected", 0)
        item.setdefault("closed_trades", 0)
        item.setdefault("open_trades", 0)
        item.setdefault("open_profit", 0.0)
        item.setdefault("open_pips", 0.0)
        item.setdefault("net_pips", 0.0)
        item.setdefault("net_profit", 0.0)
        item.setdefault("average_pips", None)
        item.setdefault("average_profit", None)
        item.setdefault("win_rate", None)
        item.setdefault("max_drawdown_pips", None)
        item.setdefault("symbols", "")

    return sorted(rows.values(), key=lambda r: (float(r.get("net_pips") or 0.0), float(r.get("selected_deployment_score") or 0.0)), reverse=True)


# -------------------------------- loaders ----------------------------------


def load_signals() -> Tuple[pd.DataFrame, List[str]]:
    paths_cfg = STATE.config.get("paths") or {}
    files = find_files(_as_list(paths_cfg.get("signal_logs")), excludes=_as_list(paths_cfg.get("exclude_patterns")))
    max_rows = int((STATE.config.get("display") or {}).get("max_rows", 5000))
    df = read_csvs(files, max_rows=max_rows)
    return df, [str(p) for p in files]


def load_trades() -> Tuple[pd.DataFrame, List[str]]:
    paths_cfg = STATE.config.get("paths") or {}
    files = find_files(_as_list(paths_cfg.get("trade_logs")), excludes=_as_list(paths_cfg.get("exclude_patterns")))
    max_rows = int((STATE.config.get("display") or {}).get("max_rows", 5000))
    df = read_csvs(files, max_rows=max_rows)
    return df, [str(p) for p in files]


def load_state_trades() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    paths_cfg = STATE.config.get("paths") or {}
    files = find_files(_as_list(paths_cfg.get("paper_state_files")), excludes=[])
    open_rows: List[Dict[str, Any]] = []
    closed_rows: List[Dict[str, Any]] = []
    for path in files:
        try:
            payload = read_json_file(path)
            o, c = extract_state_trades(payload, path.name)
            open_rows.extend(o)
            closed_rows.extend(c)
        except Exception as exc:
            open_rows.append({"_source_file": path.name, "_read_error": str(exc), "status": "error"})
    return pd.DataFrame(open_rows), pd.DataFrame(closed_rows), [str(p) for p in files]


def load_json_summaries() -> Dict[str, Any]:
    paths_cfg = STATE.config.get("paths") or {}
    files = find_files(_as_list(paths_cfg.get("summary_files")), excludes=[])
    out: Dict[str, Any] = {}
    for path in files[:10]:
        try:
            out[path.name] = read_json_file(path)
        except Exception as exc:
            out[path.name] = {"error": str(exc)}
    return out


def load_staged_models() -> Tuple[List[Dict[str, Any]], List[str]]:
    paths_cfg = STATE.config.get("paths") or {}
    files: List[Path] = []
    files.extend(find_files(_as_list(paths_cfg.get("manifest_files")), excludes=[]))
    files.extend(find_files(_as_list(paths_cfg.get("model_selection_files")), excludes=[]))

    rows: Dict[str, Dict[str, Any]] = {}
    for path in files:
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = _safe_read_csv(path)
                for rec in df.where(pd.notnull(df), None).to_dict(orient="records"):
                    symbol = str(rec.get("symbol") or "").upper()
                    model = str(rec.get("model") or "model")
                    side = str(rec.get("side") or "").lower()
                    epoch = rec.get("epoch")
                    combo_id = rec.get("id") or rec.get("combo_id")
                    if not combo_id:
                        try:
                            combo_id = f"{symbol}_{model}_{side}_epoch_{int(float(epoch)):03d}" if epoch not in (None, "") else f"{symbol}_{model}_{side}"
                        except Exception:
                            combo_id = f"{symbol}_{model}_{side}_epoch_{epoch}"
                    combo_id = str(combo_id)
                    rec["id"] = combo_id
                    rec.setdefault("_source_file", path.name)
                    rows[combo_id] = rec
            else:
                if suffix in {".yaml", ".yml"}:
                    if yaml is None:
                        raise RuntimeError("PyYAML is required to read YAML ensemble manifests")
                    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                else:
                    payload = read_json_file(path)
                if isinstance(payload, dict):
                    for rec in payload.get("models") or []:
                        if not isinstance(rec, dict):
                            continue
                        combo_id = str(rec.get("id") or rec.get("combo_id") or "")
                        if not combo_id:
                            symbol = str(rec.get("symbol") or "").upper()
                            model = str(rec.get("model") or "model")
                            side = str(rec.get("side") or "").lower()
                            epoch = rec.get("epoch", "")
                            combo_id = f"{symbol}_{model}_{side}_epoch_{epoch}"
                        item = dict(rec)
                        item["id"] = combo_id
                        item.setdefault("_source_file", path.name)
                        rows[combo_id] = {**rows.get(combo_id, {}), **item}
        except Exception as exc:
            key = f"{path.name}:error"
            rows[key] = {"id": key, "_source_file": path.name, "error": str(exc)}

    out = list(rows.values())
    def _sort_key(r: Dict[str, Any]) -> Tuple[float, str]:
        try:
            score = float(r.get("deployment_score") or 0.0)
        except Exception:
            score = 0.0
        return (-score, str(r.get("id") or ""))
    return sorted(out, key=_sort_key), [str(p) for p in files]


def load_bar_state_files() -> Dict[str, Any]:
    paths_cfg = STATE.config.get("paths") or {}
    files = find_files(_as_list(paths_cfg.get("bar_state_files")), excludes=[])
    out: Dict[str, Any] = {}
    for path in files[:5]:
        try:
            out[path.name] = read_json_file(path)
        except Exception as exc:
            out[path.name] = {"error": str(exc)}
    return out


def load_dashboard_data() -> Dict[str, Any]:
    signal_df, signal_files = load_signals()
    trade_df, trade_files = load_trades()
    state_open_df, state_closed_df, state_files = load_state_trades()
    staged_models, staged_files = load_staged_models()

    csv_open_df, csv_closed_df = split_open_closed_trades(trade_df)
    open_df = pd.concat([csv_open_df, state_open_df], ignore_index=True, sort=False) if not state_open_df.empty or not csv_open_df.empty else pd.DataFrame()
    closed_df = pd.concat([csv_closed_df, state_closed_df], ignore_index=True, sort=False) if not state_closed_df.empty or not csv_closed_df.empty else pd.DataFrame()

    summaries = load_json_summaries()
    bar_state = load_bar_state_files()
    signal_summary = summarize_signals(signal_df)
    position_manager_summary = summarize_position_manager(signal_df)
    closed_summary = summarize_trades(closed_df)
    open_summary = summarize_trades(open_df)
    models = summarize_models(signal_df, closed_df, open_df, staged_models=staged_models)

    return {
        "health": {
            "status": "ok",
            "server_time": pd.Timestamp.now('UTC').isoformat(),
            "project_root": str(get_project_root()),
        },
        "files": {
            "signals": signal_files,
            "trades": trade_files,
            "state": state_files,
            "staged_models": staged_files,
            "summaries": list(summaries.keys()),
            "bar_state": list(bar_state.keys()),
        },
        "signals": signal_summary,
        "position_manager": position_manager_summary,
        "trades": {
            **closed_summary,
            "total_closed_trades": int(len(closed_df)),
            "active_open_trades": int(len(open_df)),
            "open_summary": open_summary,
        },
        "models": models,
        "staged_models": staged_models,
        "open_trades": dataframe_to_records(open_df, 1000),
        "closed_trades": dataframe_to_records(closed_df, 1000),
        "all_trade_rows": dataframe_to_records(trade_df, 1000),
        "position_manager_events": dataframe_to_records(signal_df[_position_manager_mask(signal_df)].copy() if not signal_df.empty else pd.DataFrame(), 1000),
        "json_summaries": summaries,
        "bar_state": bar_state,
    }


# ----------------------------------- auth ----------------------------------

def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    sec_cfg = STATE.config.get("security") or {}
    username_env = str(sec_cfg.get("username_env", "DASHBOARD_USER"))
    password_env = str(sec_cfg.get("password_env", "DASHBOARD_PASSWORD"))
    expected_user = os.environ.get(username_env, "")
    expected_password = os.environ.get(password_env, "")

    if not expected_user or not expected_password:
        if bool(sec_cfg.get("allow_config_credentials", False)):
            expected_user = str(sec_cfg.get("username", ""))
            expected_password = str(sec_cfg.get("password", ""))

    if not expected_user or not expected_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Dashboard credentials are not configured. Set DASHBOARD_USER and "
                "DASHBOARD_PASSWORD environment variables before starting the server."
            ),
        )

    ok_user = secrets.compare_digest(credentials.username, expected_user)
    ok_pw = secrets.compare_digest(credentials.password, expected_password)
    if not (ok_user and ok_pw):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid dashboard credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# -------------------------------- endpoints --------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def index(_: str = Depends(require_auth)) -> HTMLResponse:
    refresh = int((STATE.config.get("server") or {}).get("refresh_seconds", 10))
    tz_label = (STATE.config.get("display") or {}).get("timezone_label", "UTC")
    html = DASHBOARD_HTML.replace("__REFRESH_SECONDS__", str(refresh)).replace("__TIMEZONE_LABEL__", str(tz_label))
    return HTMLResponse(html)


@app.get("/api/health")
@app.get("/dashboard/api/health")
def health(_: str = Depends(require_auth)) -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_TITLE,
        "config_path": str(STATE.config_path) if STATE.config_path else None,
        "project_root": str(get_project_root()),
        "server_time": pd.Timestamp.now('UTC').isoformat(),
    }


@app.get("/api/summary")
@app.get("/dashboard/api/summary")
def api_summary(_: str = Depends(require_auth)) -> Dict[str, Any]:
    return cached("summary", load_dashboard_data)


@app.get("/api/signals")
@app.get("/dashboard/api/signals")
def api_signals(limit: int = 100, _: str = Depends(require_auth)) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 1000))
    signal_df, files = load_signals()
    return {"files": files, "rows": dataframe_to_records(signal_df, limit)}


@app.get("/api/trades")
@app.get("/dashboard/api/trades")
def api_trades(limit: int = 100, _: str = Depends(require_auth)) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 1000))
    data = load_dashboard_data()
    return {"files": data["files"], "rows": data.get("all_trade_rows", [])[-limit:]}


@app.get("/api/open-trades")
@app.get("/dashboard/api/open-trades")
def api_open_trades(limit: int = 100, _: str = Depends(require_auth)) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 1000))
    data = load_dashboard_data()
    return {"rows": data.get("open_trades", [])[-limit:]}


@app.get("/api/closed-trades")
@app.get("/dashboard/api/closed-trades")
def api_closed_trades(limit: int = 100, _: str = Depends(require_auth)) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 1000))
    data = load_dashboard_data()
    return {"rows": data.get("closed_trades", [])[-limit:]}


@app.get("/api/models")
@app.get("/dashboard/api/models")
def api_models(_: str = Depends(require_auth)) -> Dict[str, Any]:
    data = load_dashboard_data()
    return {"rows": data.get("models", [])}


@app.get("/api/staged-models")
@app.get("/dashboard/api/staged-models")
def api_staged_models(_: str = Depends(require_auth)) -> Dict[str, Any]:
    data = load_dashboard_data()
    return {"rows": data.get("staged_models", [])}


@app.get("/api/position-manager")
@app.get("/dashboard/api/position-manager")
def api_position_manager(limit: int = 100, _: str = Depends(require_auth)) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 1000))
    data = load_dashboard_data()
    return {"summary": data.get("position_manager", {}), "rows": data.get("position_manager_events", [])[-limit:]}


@app.get("/api/config")
@app.get("/dashboard/api/config")
def api_config(_: str = Depends(require_auth)) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(STATE.config))
    sec = cfg.get("security") or {}
    if "password" in sec:
        sec["password"] = "***"
    return cfg


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI CFD Live Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1220; --panel:#121c30; --panel2:#0f1728; --muted:#92a0b6; --text:#e9eef8; --good:#42d392; --bad:#ff6b6b; --warn:#ffd166; --line:#23304a; --tab:#1b2944; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }
    header { padding: 20px 28px; border-bottom:1px solid var(--line); display:flex; gap:20px; justify-content:space-between; align-items:center; }
    h1 { margin:0; font-size: 22px; letter-spacing:.2px; }
    h2 { font-size:18px; margin:0 0 12px; }
    .sub { color:var(--muted); font-size:13px; }
    main { padding:24px 28px 40px; max-width:1650px; margin:0 auto; }
    .grid { display:grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap:14px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px; box-shadow:0 10px 25px rgba(0,0,0,.15); }
    .label { color:var(--muted); font-size:13px; margin-bottom:6px; }
    .value { font-size:26px; font-weight:700; }
    .good { color:var(--good); } .bad { color:var(--bad); } .warn { color:var(--warn); }
    .section { margin-top:18px; }
    .two { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
    .three { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:16px; }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 18px; }
    .tab-btn { cursor:pointer; border:1px solid var(--line); background:var(--tab); color:var(--text); padding:9px 12px; border-radius:999px; font-weight:600; }
    .tab-btn.active { background:#2a3c63; border-color:#49608e; }
    .tab { display:none; }
    .tab.active { display:block; }
    table { width:100%; border-collapse: collapse; font-size:13px; }
    th, td { border-bottom:1px solid var(--line); padding:8px 7px; text-align:left; white-space:nowrap; max-width:300px; overflow:hidden; text-overflow:ellipsis; }
    th { color:var(--muted); font-weight:600; position:sticky; top:0; background:var(--panel); z-index:1; }
    .table-wrap { max-height:540px; overflow:auto; border:1px solid var(--line); border-radius:12px; }
    .pill { display:inline-block; margin:2px; padding:3px 8px; border-radius:999px; background:#1d2a45; border:1px solid var(--line); color:var(--muted); font-size:12px; }
    .files { color:var(--muted); font-size:12px; line-height:1.45; word-break:break-all; }
    pre { white-space:pre-wrap; word-break:break-word; background:#0a1020; border:1px solid var(--line); padding:12px; border-radius:12px; color:#b9c4d8; max-height:340px; overflow:auto; }
    .bar { height:8px; background:#0a1020; border-radius:999px; overflow:hidden; border:1px solid var(--line); margin-top:8px; }
    .bar > div { height:100%; background:var(--good); width:0%; }
    @media (max-width:1200px) { .grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } .three { grid-template-columns:1fr; } }
    @media (max-width:900px) { .two { grid-template-columns:1fr; } }
    @media (max-width:650px) { .grid { grid-template-columns: 1fr; } header { align-items:flex-start; flex-direction:column; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AI CFD Live Dashboard</h1>
      <div class="sub">Read-only live/paper monitoring · times: __TIMEZONE_LABEL__</div>
    </div>
    <div class="sub">Auto-refresh: <span id="refresh">__REFRESH_SECONDS__</span>s · Last update: <span id="updated">never</span></div>
  </header>
  <main>
    <div class="tabs">
      <button class="tab-btn active" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="models">Trades per model</button>
      <button class="tab-btn" data-tab="staged">Staged models</button>
      <button class="tab-btn" data-tab="positionManager">TP protection</button>
      <button class="tab-btn" data-tab="open">Active open trades</button>
      <button class="tab-btn" data-tab="closed">Closed trades</button>
      <button class="tab-btn" data-tab="signals">Signals</button>
      <button class="tab-btn" data-tab="filesTab">Files</button>
    </div>

    <section id="overview" class="tab active">
      <div class="grid">
        <div class="card"><div class="label">Overall P/L</div><div id="netpips" class="value">–</div></div>
        <div class="card"><div class="label">Active open trades</div><div id="openTrades" class="value">–</div></div>
        <div class="card"><div class="label">Total closed trades</div><div id="closedTrades" class="value">–</div></div>
        <div class="card"><div class="label">Win rate</div><div id="winrate" class="value">–</div></div>
        <div class="card"><div class="label">Average pips/trade</div><div id="avgpips" class="value">–</div></div>
        <div class="card"><div class="label">Max drawdown</div><div id="drawdown" class="value">–</div></div>
        <div class="card"><div class="label">Signals</div><div id="signalsCount" class="value">–</div></div>
        <div class="card"><div class="label">Orders sent</div><div id="orders" class="value">–</div></div>
        <div class="card"><div class="label">Paper opened</div><div id="paper" class="value">–</div></div>
        <div class="card"><div class="label">Avg selected prob.</div><div id="avgProb" class="value">–</div></div>
        <div class="card"><div class="label">Avg selected EV</div><div id="avgEv" class="value">–</div></div>
        <div class="card"><div class="label">Trade log rows</div><div id="tradeRows" class="value">–</div></div>
        <div class="card"><div class="label">Open P/L</div><div id="openPnl" class="value">–</div></div>
        <div class="card"><div class="label">TP-protection closes</div><div id="tpCloses" class="value">–</div></div>
        <div class="card"><div class="label">Staged models</div><div id="stagedCount" class="value">–</div></div>
      </div>

      <div class="section three">
        <div class="card"><h2>Closed trade breakdown</h2><div id="tradeBreakdown"></div></div>
        <div class="card"><h2>Signal breakdown</h2><div id="signalBreakdown"></div></div>
        <div class="card"><h2>Latest signal</h2><pre id="latestSignal">–</pre></div>
      </div>
    </section>

    <section id="models" class="tab">
      <div class="card">
        <h2>Trades per model / symbol</h2>
        <div class="sub">Ensemble rows are grouped by combo_id so buy/sell side models and replay-selected epochs stay separate.</div><br>
        <div class="table-wrap"><table id="modelsTable"></table></div>
      </div>
    </section>

    <section id="staged" class="tab">
      <div class="card">
        <h2>Staged live models</h2>
        <div class="sub">Read from live_ensemble_manifest and live_model_selection_table. These are the replay-selected candidates available to the ensemble runner.</div><br>
        <div class="table-wrap"><table id="stagedTable"></table></div>
      </div>
    </section>

    <section id="positionManager" class="tab">
      <div class="card">
        <h2>TP-relative profit protection</h2>
        <div class="sub">Rows produced by the live position manager: sustained 50% TP-profit exits, 50%→30% giveback exits, skipped positions, and close errors.</div><br>
        <div id="positionManagerSummary"></div><br>
        <div class="table-wrap"><table id="positionManagerTable"></table></div>
      </div>
    </section>

    <section id="open" class="tab">
      <div class="card">
        <h2>Active open trades</h2>
        <div class="sub">Read from live trade logs and paper state files where an open/active status can be inferred.</div><br>
        <div class="table-wrap"><table id="openTable"></table></div>
      </div>
    </section>

    <section id="closed" class="tab">
      <div class="card">
        <h2>Total closed trades</h2>
        <div class="table-wrap"><table id="closedTable"></table></div>
      </div>
    </section>

    <section id="signals" class="tab">
      <div class="card">
        <h2>Recent signals</h2>
        <div class="table-wrap"><table id="signalsTable"></table></div>
      </div>
    </section>

    <section id="filesTab" class="tab">
      <div class="card">
        <h2>Files being read</h2>
        <div id="files" class="files">–</div>
      </div>
      <div class="section card">
        <h2>Latest trade</h2>
        <pre id="latestTrade">–</pre>
      </div>
    </section>
  </main>

<script>
const REFRESH_SECONDS = Number("__REFRESH_SECONDS__");
function normalisePath(path) {
  let p = String(path || "/");
  p = p.replace(/\/+$/, "");
  if (p.endsWith("/index.html")) p = p.slice(0, -"/index.html".length);
  return p || "/";
}
function cleanApiBase(path) {
  let p = String(path || "");
  p = p.replace(/\/+$/, "");
  if (!p.startsWith("/")) p = "/" + p;
  return p.replace(/\/{2,}/g, "/");
}
function unique(items) {
  const out = [];
  const seen = new Set();
  for (const item of items) {
    if (!item || seen.has(item)) continue;
    seen.add(item);
    out.push(item);
  }
  return out;
}
const PAGE_PATH = normalisePath(window.location.pathname);
const DASHBOARD_INDEX = PAGE_PATH.indexOf("/dashboard");
const PREFIX = DASHBOARD_INDEX >= 0 ? PAGE_PATH.slice(0, DASHBOARD_INDEX) : (PAGE_PATH === "/" ? "" : PAGE_PATH);
const API_BASE_CANDIDATES = unique([
  cleanApiBase(`${PAGE_PATH === "/" ? "" : PAGE_PATH}/api`),
  cleanApiBase(`${PREFIX}/dashboard/api`),
  "/dashboard/api",
  cleanApiBase(`${PREFIX}/api`),
  "/api",
]);
let ACTIVE_API_BASE = null;
async function getJson(endpoint) {
  const suffix = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  const bases = ACTIVE_API_BASE ? [ACTIVE_API_BASE, ...API_BASE_CANDIDATES.filter(b => b !== ACTIVE_API_BASE)] : API_BASE_CANDIDATES;
  const failures = [];
  for (const base of bases) {
    const url = `${base}${suffix}`;
    try {
      const response = await fetch(url, {cache: "no-store", credentials: "same-origin"});
      const text = await response.text();
      if (!response.ok) {
        failures.push(`${url} -> HTTP ${response.status}: ${text.slice(0, 180)}`);
        continue;
      }
      try {
        const data = JSON.parse(text || "{}");
        ACTIVE_API_BASE = base;
        return data;
      } catch (jsonErr) {
        failures.push(`${url} -> invalid JSON: ${text.slice(0, 180)}`);
      }
    } catch (err) {
      failures.push(`${url} -> network error: ${err && err.message ? err.message : err}`);
    }
  }
  throw new Error(failures.join(" | "));
}
const tabs = document.querySelectorAll('.tab-btn');
tabs.forEach(btn => btn.addEventListener('click', () => {
  tabs.forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(btn.dataset.tab).classList.add('active');
}));

function fmt(n, digits=1) { return (n === null || n === undefined || Number.isNaN(Number(n))) ? "–" : Number(n).toFixed(digits); }
function pct(n) { return (n === null || n === undefined || Number.isNaN(Number(n))) ? "–" : (Number(n)*100).toFixed(1) + "%"; }
function setText(id, text, klass="") { const el=document.getElementById(id); if(!el) return; el.textContent=text; el.className = "value " + klass; }
function escapeHtml(s) { return String(s).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#039;','"':'&quot;'}[c])); }
function objList(obj, digits=null) {
  if (!obj || Object.keys(obj).length === 0) return "<span class='sub'>No data</span>";
  return Object.entries(obj).map(([k,v]) => {
    const value = (digits !== null && !Number.isNaN(Number(v))) ? Number(v).toFixed(digits) : String(v);
    return `<span class="pill">${escapeHtml(k)}: ${escapeHtml(value)}</span>`;
  }).join(" ");
}
function table(elId, rows, preferred) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!rows || rows.length === 0) { el.innerHTML = "<tr><td>No rows found</td></tr>"; return; }
  const pref = preferred || ["time_utc","bar_time","entry_time","exit_time","symbol","model","model_label","side","direction","status","final_decision","reason","net_pips","pips","profit","selected_probability","order_sent","ticket","_source_file"];
  const keys = Array.from(new Set([...pref.filter(k => rows.some(r => r[k] !== undefined)), ...Object.keys(rows[rows.length-1]).slice(0, 16)])).slice(0, 22);
  el.innerHTML = `<thead><tr>${keys.map(k => `<th>${escapeHtml(k)}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.slice().reverse().map(r => `<tr>${keys.map(k => `<td title="${escapeHtml(r[k] ?? '')}">${escapeHtml(r[k] ?? '')}</td>`).join("")}</tr>`).join("")}</tbody>`;
}
function modelTable(rows) {
  table('modelsTable', rows || [], ['model','symbols','selected_side','selected_epoch','selected_deployment_score','selected_net_pips','selected_trades','selected_win_rate','selected_average_net_pips','selected_max_drawdown_pips','open_trades','closed_trades','net_pips','net_profit','win_rate','average_pips','max_drawdown_pips','signals','allow_signals','block_signals','skip_signals','ensemble_selected','orders_sent','avg_selected_probability','avg_signal_rank_score']);
}
function stagedTable(rows) {
  table('stagedTable', rows || [], ['id','symbol','model','side','timeframe','epoch','deployment_score','deployment_usable','deployment_reason','net_pips','trades','win_rate','average_net_pips','max_drawdown_pips','model_path','config_path','_source_file']);
}
function positionManagerTable(rows) {
  table('positionManagerTable', rows || [], ['time_utc','bar_time','symbol','side','ticket','final_decision','reason','current_profit','tp_profit','trigger_profit','giveback_close_profit','candles_at_or_above_trigger','close_sent','order_sent','close_error','order_error','_source_file']);
}

async function load() {
  try {
    const summary = await getJson('/summary');
    const signals = await getJson('/signals?limit=150');
    const openTrades = await getJson('/open-trades?limit=500');
    const closedTrades = await getJson('/closed-trades?limit=500');
    const models = await getJson('/models');
    const staged = await getJson('/staged-models');
    const pmEvents = await getJson('/position-manager?limit=500');

    const t = summary.trades || {};
    const s = summary.signals || {};
    const pm = summary.position_manager || {};
    const openSummary = t.open_summary || {};
    setText('netpips', fmt(t.net_pips, 1), (t.net_pips || 0) >= 0 ? 'good' : 'bad');
    setText('openTrades', t.active_open_trades ?? 0, (t.active_open_trades || 0) > 0 ? 'warn' : '');
    setText('closedTrades', t.total_closed_trades ?? 0);
    setText('winrate', pct(t.win_rate), (t.win_rate || 0) >= .55 ? 'good' : ((t.win_rate || 0) >= .48 ? 'warn' : 'bad'));
    setText('avgpips', fmt(t.average_pips, 2), (t.average_pips || 0) >= 0 ? 'good' : 'bad');
    setText('drawdown', fmt(t.max_drawdown_pips, 1));
    setText('signalsCount', s.signals ?? 0);
    setText('orders', s.orders_sent ?? 0);
    setText('paper', s.paper_opened ?? 0);
    setText('avgProb', pct(s.avg_selected_probability));
    setText('avgEv', fmt(s.avg_selected_ev, 2));
    setText('tradeRows', (summary.all_trade_rows || []).length);
    setText('openPnl', fmt(openSummary.net_profit, 2), (openSummary.net_profit || 0) >= 0 ? 'good' : 'bad');
    setText('tpCloses', pm.close_sent ?? 0, (pm.close_sent || 0) > 0 ? 'warn' : '');
    setText('stagedCount', (summary.staged_models || []).length);

    document.getElementById('tradeBreakdown').innerHTML = `<div class='label'>Side counts</div>${objList(t.side_counts)}<br><br><div class='label'>Side pips</div>${objList(t.side_pips, 1)}<br><br><div class='label'>Symbol pips</div>${objList(t.symbol_pips, 1)}`;
    document.getElementById('signalBreakdown').innerHTML = `<div class='label'>Model decisions</div>${objList(s.final_decision_counts)}<br><br><div class='label'>Directions</div>${objList(s.direction_counts)}<br><br><div class='label'>Block reasons</div>${objList(s.block_reason_counts)}<br><br><div class='label'>Position manager</div>${objList({events: pm.events || 0, close_sent: pm.close_sent || 0, giveback: pm.giveback_exits || 0, sustained: pm.sustained_exits || 0})}`;
    document.getElementById('latestSignal').textContent = JSON.stringify(s.latest_signal ?? {}, null, 2);
    document.getElementById('latestTrade').textContent = JSON.stringify(t.latest_trade ?? {}, null, 2);
    document.getElementById('files').innerHTML = `<b>Signal logs</b><br>${(summary.files.signals||[]).map(escapeHtml).join('<br>') || 'None'}<br><br><b>Trade logs</b><br>${(summary.files.trades||[]).map(escapeHtml).join('<br>') || 'None'}<br><br><b>State files</b><br>${(summary.files.state||[]).map(escapeHtml).join('<br>') || 'None'}<br><br><b>Staged model files</b><br>${(summary.files.staged_models||[]).map(escapeHtml).join('<br>') || 'None'}<br><br><b>Bar state files</b><br>${(summary.files.bar_state||[]).map(escapeHtml).join('<br>') || 'None'}<br><br><b>Summary JSON</b><br>${(summary.files.summaries||[]).map(escapeHtml).join('<br>') || 'None'}`;

    modelTable(models.rows || []);
    stagedTable(staged.rows || summary.staged_models || []);
    document.getElementById('positionManagerSummary').innerHTML = `<div class='label'>Events</div>${objList({events: pm.events || 0, close_attempted: pm.close_attempted || 0, close_sent: pm.close_sent || 0, giveback: pm.giveback_exits || 0, sustained: pm.sustained_exits || 0})}<br><br><div class='label'>Reasons</div>${objList(pm.reason_counts || {})}`;
    positionManagerTable(pmEvents.rows || summary.position_manager_events || []);
    table('openTable', openTrades.rows || [], ['snapshot_time_utc','open_time_utc','entry_time','open_time','time_utc','symbol','model_label','side','direction','status','entry_price','current_price','pips','pnl','profit','tp','sl','ticket','magic','comment','_source_file']);
    table('closedTable', closedTrades.rows || [], ['close_time_utc','exit_time','close_time','open_time_utc','entry_time','symbol','model_label','side','direction','status','pips','pnl','profit','ticket','position_id','order','comment','_source_file']);
    table('signalsTable', signals.rows || [], ['bar_time','time_utc','symbol','model','side_model','epoch','combo_id','direction','model_decision','final_decision','reason','selected_probability','signal_rank_score','order_selected_by_ensemble','order_sent','spread_points','buy_threshold_margin','sell_threshold_margin','tp_profit','current_profit','model_path','_source_file']);
    document.getElementById('updated').textContent = new Date().toLocaleString();
  } catch (err) {
    console.error('Dashboard update failed', err);
    const msg = err && err.message ? err.message : String(err);
    document.getElementById('updated').textContent = 'update error: ' + msg;
  }
}
load();
setInterval(load, REFRESH_SECONDS * 1000);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the AI CFD live dashboard")
    parser.add_argument("--config", default="dashboard_config.yaml", help="Dashboard YAML config path")
    parser.add_argument("--host", default=None, help="Override bind host")
    parser.add_argument("--port", type=int, default=None, help="Override port")
    args = parser.parse_args()

    cfg = load_config(args.config)
    server = cfg.get("server") or {}
    host = args.host or str(server.get("host", "127.0.0.1"))
    port = int(args.port or server.get("port", 8050))

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
