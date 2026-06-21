from __future__ import annotations

import argparse
import copy
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .features import build_feature_frame
from .forex import validate_forex_symbols, point_for_symbol, spread_points_to_pips
from .io_utils import ensure_dir
from .spread_utils import apply_spread_fallback


def _data_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    data = dict(cfg.get("data", {}) or {})
    paths = cfg.get("paths", {}) or {}
    data.setdefault("raw_dir", "data/raw")
    data.setdefault("processed_dir", paths.get("processed_dir", "data/processed_m5"))
    return data


def raw_path_for(symbol: str, cfg: dict[str, Any], timeframe: str | None = None) -> Path:
    timeframe = str(timeframe or (cfg.get("trading", {}) or {}).get("timeframe", "M5")).upper()
    return Path(_data_cfg(cfg).get("raw_dir", "data/raw")) / f"{symbol}_{timeframe}.csv"


def processed_path_for(symbol: str, cfg: dict[str, Any], timeframe: str | None = None) -> Path:
    timeframe = str(timeframe or (cfg.get("trading", {}) or {}).get("timeframe", "M5")).upper()
    return Path(_data_cfg(cfg).get("processed_dir", "data/processed_m5")) / f"{symbol}_{timeframe}_features.csv"


def compatibility_path_for(symbol: str, cfg: dict[str, Any], timeframe: str | None = None) -> Path:
    """Compatibility filename for older scripts and read_processed_csv search order."""
    timeframe = str(timeframe or (cfg.get("trading", {}) or {}).get("timeframe", "M5")).upper()
    out = processed_path_for(symbol, cfg, timeframe)
    return out.parent / f"{symbol}_{timeframe}_deep_features.csv"





def _symbol_map_value(mapping: Any, symbol: str) -> Any:
    """Return a per-symbol config value using common symbol key variants."""
    if not isinstance(mapping, dict):
        return None
    symbol_u = symbol.upper()
    for key in (symbol, symbol_u, symbol_u.lower()):
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _infer_pip_size(symbol: str, cfg: dict[str, Any]) -> float:
    """Infer pip size for spread/ATR unit conversion.

    Prefer explicit per-symbol config maps if present, otherwise use the normal
    CFD branch: prefer trading.pip_size_by_symbol; otherwise fallback to symbol point size.
    """
    for section_name in ("symbols", "trading", "data", "labels", "risk"):
        section = cfg.get(section_name, {}) or {}
        value = _symbol_map_value(section.get("pip_size_by_symbol"), symbol)
        if value is None:
            value = _symbol_map_value(section.get("pip_sizes"), symbol)
        if value is not None:
            return float(value)
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def _spread_pips_per_point(cfg: dict[str, Any]) -> float:
    """Return broker-point to pip conversion, defaulting to 0.1 pip/point."""
    for section_name in ("labels", "risk", "data", "trading"):
        section = cfg.get(section_name, {}) or {}
        value = section.get("spread_pips_per_point")
        if value not in (None, ""):
            return float(value)
    value = cfg.get("spread_pips_per_point")
    return float(value) if value not in (None, "") else 0.1


def fix_spread_atr_units(df: pd.DataFrame, symbol: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Fix spread_atr so it is dimensionless and comparable across all pairs.

    Previous feature generation could leave spread_atr in broker points / price
    ATR units. For CFDs this must use the MT5 broker point size rather than FX pip conventions.

    Correct formula:
        spread_atr = spread_points * point_size / atr_14
    """
    if "spread_points" not in df.columns or "atr_14" not in df.columns:
        return df
    out = df.copy()
    spread_points = pd.to_numeric(out["spread_points"], errors="coerce")
    atr_price = pd.to_numeric(out["atr_14"], errors="coerce")
    spread_price = spread_points * point_for_symbol(symbol, cfg)
    ratio = spread_price / atr_price.where(atr_price > 0)
    out["spread_atr"] = ratio.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    return out

def _csv_data_row_count(path: Path) -> int | None:
    """Fast row count for status reporting; returns None if the file cannot be counted."""
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = sum(1 for _ in f)
        return max(0, rows - 1)
    except Exception:
        return None


def _should_skip_existing_outputs(
    symbol: str,
    cfg: dict[str, Any],
    timeframe: str | None,
    *,
    force: bool,
    write_compat_copy: bool = True,
) -> tuple[bool, Path, Path, str]:
    """Return whether the symbol can be skipped because an output already exists.

    The primary output is <symbol>_<timeframe>_features.csv. If only the older
    compatibility file exists, it is copied to the primary name and the symbol is
    still skipped. If the primary exists but the compatibility file is missing,
    the compatibility file is copied from the primary so older code paths remain
    usable without regenerating the dataset.
    """
    out = processed_path_for(symbol, cfg, timeframe)
    compat = compatibility_path_for(symbol, cfg, timeframe)
    if force:
        return False, out, compat, "force_overwrite"

    if out.exists():
        if write_compat_copy and compat != out and not compat.exists():
            ensure_dir(compat.parent)
            shutil.copy2(out, compat)
            return True, out, compat, "processed_exists_compat_copied_from_primary"
        if compat.exists():
            return True, out, compat, "processed_exists_compat_exists"
        return True, out, compat, "processed_exists_no_compat_copy_requested"

    if compat.exists():
        # Compatibility-only recovery path: if an old deep_features file exists
        # but the primary features file is missing, copy it once to the primary
        # filename. This avoids rebuilding features and migrates the project to
        # the single primary filename.
        ensure_dir(out.parent)
        shutil.copy2(compat, out)
        return True, out, compat, "compat_exists_primary_copied_from_compat"

    return False, out, compat, "missing"


def prepare_symbol(
    symbol: str,
    cfg: dict[str, Any],
    timeframe: str | None = None,
    max_rows: int | None = None,
    *,
    force: bool = False,
    write_compat_copy: bool = True,
) -> Path:
    timeframe = str(timeframe or (cfg.get("trading", {}) or {}).get("timeframe", "M5")).upper()
    should_skip, out, compat, _ = _should_skip_existing_outputs(symbol, cfg, timeframe, force=force, write_compat_copy=write_compat_copy)
    if should_skip:
        return out

    raw_path = raw_path_for(symbol, cfg, timeframe)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw MT5 data for {symbol}: {raw_path}. Run python -m src.download_mt5_data first.")

    header = pd.read_csv(raw_path, nrows=0)
    parse_dates = ["time"] if "time" in header.columns else None
    raw = pd.read_csv(raw_path, parse_dates=parse_dates)
    if max_rows:
        raw = raw.tail(max_rows).reset_index(drop=True)
    raw["symbol"] = symbol.upper()
    raw = apply_spread_fallback(raw, symbol, cfg)
    feat = build_feature_frame(raw, cfg, symbol=symbol)
    feat = apply_spread_fallback(feat, symbol, cfg)
    feat = fix_spread_atr_units(feat, symbol, cfg)

    ensure_dir(out.parent)
    feat.to_csv(out, index=False)
    # The deep_features filename is compatibility-only. Do not build or CSV-write
    # the dataframe a second time. By default, create it as a byte-for-byte copy
    # of the primary file so old code paths keep working. Disable with
    # --no-compat-deep-features or data.write_deep_feature_compat_copy: false.
    if write_compat_copy and compat != out:
        ensure_dir(compat.parent)
        shutil.copy2(out, compat)
    return out


def _configured_workers(cfg: dict[str, Any], cli_workers: int | None, symbol_count: int) -> int:
    if symbol_count <= 1:
        return 1
    if cli_workers is not None:
        value = int(cli_workers)
        if value <= 0:
            return min(symbol_count, os.cpu_count() or 1, 4)
        return min(symbol_count, value)

    dcfg = cfg.get("data", {}) or {}
    tcfg = cfg.get("training", {}) or {}
    for key in ("prepare_workers", "mt5_prepare_workers", "workers"):
        value = dcfg.get(key, tcfg.get(key))
        if value is not None:
            value_int = int(value)
            if value_int <= 0:
                return min(symbol_count, os.cpu_count() or 1, 4)
            return min(symbol_count, value_int)

    return min(symbol_count, os.cpu_count() or 1, 4)




def _configured_write_compat_copy(cfg: dict[str, Any], cli_no_compat_deep_features: bool) -> bool:
    """Return whether to create SYMBOL_TIMEFRAME_deep_features.csv as a copy.

    Default is True to avoid breaking older scripts that still search for the
    deep_features filename. This does not regenerate features and does not
    serialize the dataframe twice; it only performs shutil.copy2(primary, compat).
    """
    if cli_no_compat_deep_features:
        return False
    dcfg = cfg.get("data", {}) or {}
    for key in ("write_deep_feature_compat_copy", "write_compat_deep_features", "create_deep_feature_copy"):
        if key in dcfg and dcfg[key] is not None:
            value = dcfg[key]
            if isinstance(value, str):
                return value.strip().lower() not in {"0", "false", "no", "off"}
            return bool(value)
    return True

def _prepare_symbol_job(
    symbol: str,
    cfg: dict[str, Any],
    *,
    timeframe: str | None,
    max_rows: int | None,
    force: bool,
    write_compat_copy: bool,
) -> dict[str, Any]:
    # Build features in parallel without sharing a mutable config object between workers.
    symbol_cfg = copy.deepcopy(cfg)
    should_skip, out, compat, skip_reason = _should_skip_existing_outputs(symbol, symbol_cfg, timeframe, force=force, write_compat_copy=write_compat_copy)
    if should_skip:
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": skip_reason,
            "path": str(out),
            "compat_path": str(compat),
            "compat_copy_enabled": bool(write_compat_copy),
            "compat_exists": bool(compat.exists()),
            "rows": _csv_data_row_count(out),
        }

    out = prepare_symbol(symbol, symbol_cfg, timeframe=timeframe, max_rows=max_rows, force=force, write_compat_copy=write_compat_copy)
    return {
        "symbol": symbol,
        "status": "saved",
        "reason": "prepared",
        "path": str(out),
        "compat_path": str(compatibility_path_for(symbol, symbol_cfg, timeframe)),
        "compat_copy_enabled": bool(write_compat_copy),
        "compat_exists": bool(compatibility_path_for(symbol, symbol_cfg, timeframe).exists()),
        "rows": _csv_data_row_count(out),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare processed MT5 feature CSVs for the direction policy model")
    p.add_argument("--config", default="config/direction_settings_generic_multisymbol_31_symbols.yaml")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of symbols to prepare in parallel. Default: data.prepare_workers or "
            "training.prepare_workers if set, otherwise min(symbols, CPU cores, 4). "
            "Use --workers 1 for serial execution; --workers 0 for auto."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed files. Without this, symbols with existing output files are skipped.",
    )
    p.add_argument(
        "--no-compat-deep-features",
        action="store_true",
        help=(
            "Do not create SYMBOL_TIMEFRAME_deep_features.csv. By default this "
            "file is created only as a shutil.copy2 copy of the primary "
            "SYMBOL_TIMEFRAME_features.csv for backward compatibility."
        ),
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    symbols = validate_forex_symbols(args.symbols or ((cfg.get("trading") or {}).get("symbols") or ["US500"]))
    workers = _configured_workers(cfg, args.workers, len(symbols))
    write_compat_copy = _configured_write_compat_copy(cfg, bool(args.no_compat_deep_features))

    print(
        f"Preparing {len(symbols)} symbol(s) with workers={workers}, "
        f"force={bool(args.force)}, "
        f"compat_deep_features_copy={write_compat_copy}, "
        f"timeframe={args.timeframe or (cfg.get('trading', {}) or {}).get('timeframe', 'M5')}"
    )

    results: list[dict[str, Any]] = []
    failures: list[tuple[str, BaseException]] = []

    if workers == 1:
        for symbol in symbols:
            try:
                result = _prepare_symbol_job(
                    symbol,
                    cfg,
                    timeframe=args.timeframe,
                    max_rows=args.max_rows,
                    force=bool(args.force),
                    write_compat_copy=write_compat_copy,
                )
                results.append(result)
                rows_text = f"{result['rows']:,}" if isinstance(result.get("rows"), int) else "unknown"
                if result["status"] == "skipped":
                    print(f"Skipped {symbol}: existing file ({result['reason']}); rows={rows_text}; path={result['path']}")
                else:
                    compat_msg = " + compat copy" if result.get("compat_copy_enabled") and result.get("compat_exists") else ""
                    print(f"Saved {symbol}: {rows_text} processed rows to {result['path']}{compat_msg}")
            except Exception as exc:
                failures.append((symbol, exc))
                print(f"FAILED {symbol}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prepare_mt5") as executor:
            future_to_symbol = {
                executor.submit(
                    _prepare_symbol_job,
                    symbol,
                    cfg,
                    timeframe=args.timeframe,
                    max_rows=args.max_rows,
                    force=bool(args.force),
                    write_compat_copy=write_compat_copy,
                ): symbol
                for symbol in symbols
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results.append(result)
                    rows_text = f"{result['rows']:,}" if isinstance(result.get("rows"), int) else "unknown"
                    if result["status"] == "skipped":
                        print(f"Skipped {symbol}: existing file ({result['reason']}); rows={rows_text}; path={result['path']}")
                    else:
                        compat_msg = " + compat copy" if result.get("compat_copy_enabled") and result.get("compat_exists") else ""
                        print(f"Saved {symbol}: {rows_text} processed rows to {result['path']}{compat_msg}")
                except Exception as exc:
                    failures.append((symbol, exc))
                    print(f"FAILED {symbol}: {exc}")

    saved = sum(1 for r in results if r.get("status") == "saved")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    print(f"Completed MT5 preparation: saved={saved}, skipped={skipped}, failed={len(failures)}")

    if failures:
        details = "; ".join(f"{symbol}: {exc}" for symbol, exc in failures[:10])
        more = f"; ... plus {len(failures) - 10} more" if len(failures) > 10 else ""
        raise RuntimeError(f"MT5 data preparation failed for {len(failures)} symbol(s): {details}{more}")


if __name__ == "__main__":
    main()
