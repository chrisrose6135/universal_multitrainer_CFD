from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .forex import validate_forex_symbols
from .mt5_client import get_rates, shutdown_mt5, utc_date


def _data_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    data = dict(cfg.get("data", {}) or {})
    paths = cfg.get("paths", {}) or {}
    data.setdefault("raw_dir", "data/raw")
    data.setdefault("processed_dir", paths.get("processed_dir", "data/processed_m5"))
    return data


def download_symbol(symbol: str, cfg: dict[str, Any], start_utc: datetime, end_utc: datetime, timeframe: str | None = None) -> Path:
    timeframe = str(timeframe or (cfg.get("trading", {}) or {}).get("timeframe", "M5")).upper()
    data = _data_cfg(cfg)
    raw_dir = Path(data.get("raw_dir", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = get_rates(symbol, start_utc, end_utc, timeframe=timeframe, cfg=cfg)
    out = raw_dir / f"{symbol}_{timeframe}.csv"
    df.to_csv(out, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Download historical OHLCV candles from MetaTrader 5")
    p.add_argument("--config", default="config/direction_settings_generic_multisymbol_31_symbols.yaml")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--start", default=None, help="UTC start datetime/date, e.g. 2021-01-01")
    p.add_argument("--end", default=None, help="UTC end datetime/date. Defaults to now.")
    p.add_argument("--timeframe", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    symbols = validate_forex_symbols(args.symbols or ((cfg.get("trading") or {}).get("symbols") or ["US500"]))
    data = _data_cfg(cfg)
    start = utc_date(args.start or data.get("start_date_utc", "2021-01-01"))
    end = utc_date(args.end) if args.end else datetime.now(timezone.utc)
    try:
        for symbol in symbols:
            print(f"Downloading {symbol} {str(args.timeframe or (cfg.get('trading', {}) or {}).get('timeframe', 'M5')).upper()} from {start} to {end}...")
            out = download_symbol(symbol, cfg, start, end, timeframe=args.timeframe)
            print(f"Saved to {out}")
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
