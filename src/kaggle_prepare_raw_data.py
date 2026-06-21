from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


_TIME_ALIASES = [
    'time', 'time_utc', 'datetime', 'date_time', 'date', 'timestamp',
    'gmt_time', 'utc_time', 'Date', 'DateTime', 'Timestamp', 'Gmt time', 'Local time'
]
_COLUMN_ALIASES = {
    'open': ['open', 'Open', 'OPEN', '<OPEN>'],
    'high': ['high', 'High', 'HIGH', '<HIGH>'],
    'low': ['low', 'Low', 'LOW', '<LOW>'],
    'close': ['close', 'Close', 'CLOSE', '<CLOSE>'],
    'tick_volume': ['tick_volume', 'tickvol', 'tick_volume_utc', 'volume', 'Volume', 'tick_volume_', '<TICKVOL>'],
    'spread': ['spread', 'Spread', 'SPREAD', '<SPREAD>'],
    'spread_points': ['spread_points', 'spread_point', 'broker_spread_points'],
    'real_volume': ['real_volume', 'real_volume_', 'RealVolume', '<VOL>'],
    'symbol': ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker', 'pair', 'Pair'],
}


def _normalise_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(name).strip().lower())


def _find_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    cols = list(columns)
    normalised = {_normalise_name(c): c for c in cols}
    for alias in aliases:
        if alias in cols:
            return alias
        hit = normalised.get(_normalise_name(alias))
        if hit is not None:
            return hit
    return None


def _standardise_raw_frame(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    rename: dict[str, str] = {}
    time_col = _find_column(df.columns, _TIME_ALIASES)
    if time_col is not None:
        rename[time_col] = 'time'
    for target, aliases in _COLUMN_ALIASES.items():
        col = _find_column(df.columns, aliases)
        if col is not None:
            rename[col] = target
    out = df.rename(columns=rename).copy()
    if symbol is not None:
        out['symbol'] = symbol.upper()
    if 'time' in out.columns:
        out['time'] = pd.to_datetime(out['time'], utc=True, errors='coerce')
        out = out.loc[out['time'].notna()].copy()
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f'Raw CSV is missing required OHLC columns after alias mapping: {missing}; columns={list(df.columns)}')
    for col in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'spread_points', 'real_volume']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    if 'tick_volume' not in out.columns:
        out['tick_volume'] = 0.0
    if 'spread' not in out.columns and 'spread_points' in out.columns:
        out['spread'] = out['spread_points']
    keep = [c for c in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'spread_points', 'real_volume', 'symbol'] if c in out.columns]
    out = out[keep].dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    return out


def _csv_paths(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob('*.csv') if p.is_file())


def _infer_symbol_from_path(path: Path, symbols: list[str], timeframe: str) -> str | None:
    name = path.stem.upper().replace('-', '_').replace(' ', '_')
    timeframe_u = timeframe.upper()
    for sym in symbols:
        sym_u = sym.upper()
        patterns = [
            f'{sym_u}_{timeframe_u}',
            f'{sym_u}{timeframe_u}',
            sym_u,
        ]
        if any(p in name for p in patterns):
            return sym_u
    return None


def _looks_combined_csv(path: Path) -> bool:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    return _find_column(header.columns, _COLUMN_ALIASES['symbol']) is not None


def _split_combined_csv(path: Path, symbols: list[str], out_dir: Path, timeframe: str, *, max_rows_per_symbol: int | None, force: bool) -> list[Path]:
    df = pd.read_csv(path)
    sym_col = _find_column(df.columns, _COLUMN_ALIASES['symbol'])
    if sym_col is None:
        raise ValueError(f'Combined CSV has no symbol column: {path}')
    df[sym_col] = df[sym_col].astype(str).str.upper().str.strip()
    written: list[Path] = []
    for sym in symbols:
        part = df.loc[df[sym_col] == sym.upper()].copy()
        if part.empty:
            continue
        part = _standardise_raw_frame(part, symbol=sym)
        if max_rows_per_symbol is not None and max_rows_per_symbol > 0:
            part = part.tail(max_rows_per_symbol).reset_index(drop=True)
        out_path = out_dir / f'{sym.upper()}_{timeframe.upper()}.csv'
        if out_path.exists() and not force:
            print(f'Skipped existing raw file: {out_path}')
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            part.to_csv(out_path, index=False)
            print(f'Wrote {len(part):,} rows: {out_path}')
        written.append(out_path)
    return written


def _copy_or_standardise_per_symbol_files(input_dir: Path, symbols: list[str], out_dir: Path, timeframe: str, *, max_rows_per_symbol: int | None, force: bool) -> list[Path]:
    csvs = _csv_paths(input_dir)
    written: list[Path] = []
    for sym in symbols:
        candidates = [p for p in csvs if _infer_symbol_from_path(p, [sym], timeframe) == sym.upper()]
        if not candidates:
            print(f'WARNING: no raw CSV found for {sym} under {input_dir}')
            continue
        # Prefer exact SYMBOL_TIMEFRAME-like names over looser symbol-only matches.
        candidates.sort(key=lambda p: (f'{sym.upper()}_{timeframe.upper()}' not in p.stem.upper(), len(str(p))))
        source = candidates[0]
        out_path = out_dir / f'{sym.upper()}_{timeframe.upper()}.csv'
        if out_path.exists() and not force:
            print(f'Skipped existing raw file: {out_path}')
            written.append(out_path)
            continue
        df = pd.read_csv(source)
        df = _standardise_raw_frame(df, symbol=sym)
        if max_rows_per_symbol is not None and max_rows_per_symbol > 0:
            df = df.tail(max_rows_per_symbol).reset_index(drop=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f'Wrote {len(df):,} rows from {source.name}: {out_path}')
        written.append(out_path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description='Prepare Kaggle raw CFD CSVs into the project data/raw/SYMBOL_TIMEFRAME.csv format.'
    )
    p.add_argument('--input-dir', required=True, help='Kaggle raw dataset directory, e.g. /kaggle/input/my-CFD-raw-data')
    p.add_argument('--output-dir', default='data/raw')
    p.add_argument('--symbols', nargs='+', required=True)
    p.add_argument('--timeframe', default='M5')
    p.add_argument('--combined-csv', default=None, help='Optional combined CSV containing a symbol column. If omitted, per-symbol files are auto-discovered.')
    p.add_argument('--max-rows-per-symbol', type=int, default=None, help='Optional tail row limit per symbol for quick tests.')
    p.add_argument('--force', action='store_true', help='Overwrite existing data/raw files.')
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    symbols = [s.upper() for s in args.symbols]
    timeframe = args.timeframe.upper()

    if args.combined_csv:
        written = _split_combined_csv(Path(args.combined_csv), symbols, out_dir, timeframe, max_rows_per_symbol=args.max_rows_per_symbol, force=bool(args.force))
    else:
        combined_candidates = [p for p in _csv_paths(input_dir) if _looks_combined_csv(p)]
        if len(combined_candidates) == 1 and not any(_infer_symbol_from_path(p, symbols, timeframe) for p in _csv_paths(input_dir)):
            print(f'Auto-detected combined CSV: {combined_candidates[0]}')
            written = _split_combined_csv(combined_candidates[0], symbols, out_dir, timeframe, max_rows_per_symbol=args.max_rows_per_symbol, force=bool(args.force))
        else:
            written = _copy_or_standardise_per_symbol_files(input_dir, symbols, out_dir, timeframe, max_rows_per_symbol=args.max_rows_per_symbol, force=bool(args.force))

    missing = [s for s in symbols if not (out_dir / f'{s}_{timeframe}.csv').exists()]
    if missing:
        raise RuntimeError(f'Missing raw outputs for {missing}. Check file names or pass --combined-csv explicitly.')
    print(f'Prepared {len(written)} raw file(s) in {out_dir.resolve()}')


if __name__ == '__main__':
    main()
