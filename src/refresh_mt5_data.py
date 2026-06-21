from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .forex import validate_forex_symbols
from .mt5_client import get_rates, shutdown_mt5, utc_date
from .prepare_mt5_data import prepare_symbol, raw_path_for


def refresh_symbol_raw_data(symbol: str, cfg: dict[str, Any], timeframe: str | None = None, overlap_days: int = 7) -> Path:
    timeframe = str(timeframe or (cfg.get("trading", {}) or {}).get("timeframe", "M5")).upper()
    raw_path = raw_path_for(symbol, cfg, timeframe)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {}) or {}
    configured_start = utc_date(data_cfg.get("start_date_utc", "2021-01-01"))
    end = datetime.now(timezone.utc)

    existing = None
    if raw_path.exists():
        existing = pd.read_csv(raw_path, parse_dates=["time"])
        if len(existing):
            last_time = pd.to_datetime(existing["time"], utc=True).max().to_pydatetime()
            start = max(configured_start, last_time - timedelta(days=int(overlap_days)))
        else:
            start = configured_start
    else:
        start = configured_start

    fresh = get_rates(symbol, start, end, timeframe=timeframe, cfg=cfg)
    if existing is not None and len(existing):
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined["time"] = pd.to_datetime(combined["time"], utc=True)
        combined = combined.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
    else:
        combined = fresh.sort_values("time").reset_index(drop=True)
    combined.to_csv(raw_path, index=False)
    return raw_path


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh local MT5 raw data and rebuild processed feature CSVs")
    p.add_argument("--config", default="config/direction_settings_generic_multisymbol_31_symbols.yaml")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--overlap-days", type=int, default=7)
    p.add_argument("--skip-prepare", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    symbols = validate_forex_symbols(args.symbols or ((cfg.get("trading") or {}).get("symbols") or ["US500"]))
    try:
        for symbol in symbols:
            raw = refresh_symbol_raw_data(symbol, cfg, timeframe=args.timeframe, overlap_days=args.overlap_days)
            print(f"{symbol}: refreshed raw data at {raw}")
            if not args.skip_prepare:
                out = prepare_symbol(symbol, cfg, timeframe=args.timeframe)
                print(f"{symbol}: rebuilt processed data at {out}")
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
