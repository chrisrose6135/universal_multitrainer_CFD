from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_processed_csv(symbol: str, cfg: dict) -> pd.DataFrame:
    processed_dir = Path((cfg.get('paths') or {}).get('processed_dir', 'data/processed_m5'))
    timeframe = (cfg.get('trading') or {}).get('timeframe', (cfg.get('project') or {}).get('timeframe', 'M5'))
    candidates = [
        processed_dir / f'{symbol}_{timeframe}_deep_features.csv',
        processed_dir / f'{symbol}_{timeframe}_features.csv',
        processed_dir / f'{symbol}_{timeframe}.csv',
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            return normalise_time_column(df)
    raise FileNotFoundError(f'No processed CSV found for {symbol}. Tried: {candidates}')


def normalise_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['time_utc', 'time', 'datetime', 'DateTime', 'timestamp']:
        if col in df.columns:
            df['time_utc'] = pd.to_datetime(df[col], utc=True, errors='coerce')
            return df
    df['time_utc'] = pd.RangeIndex(len(df))
    return df


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
