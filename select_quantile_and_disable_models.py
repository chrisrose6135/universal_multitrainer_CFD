#!/usr/bin/env python3
"""Choose a live-trading rolling quantile from a sweep replay and disable weak models.

This utility is intended to be run after ``replay_live_trading_quantile_sweep.py``.
It reads the quantile sweep summary/results, prints a compact menu of viable
quantiles, lets you select one interactively (or with ``--quantile``), then
creates a filtered live ensemble manifest that excludes models with poor or
negative performance at the selected quantile.

Default behaviour is safe: it writes new filtered outputs and does not modify the
existing ``For Live Trading`` folder unless ``--apply`` is used.

Example
-------

python select_quantile_and_disable_models.py \
  --live-root "For Live Trading" \
  --quantile 0.985 \
  --min-net-pips 0 \
  --min-average-net-pips 0 \
  --min-trades 50

Apply to the live folder after reviewing the reports:

python select_quantile_and_disable_models.py \
  --live-root "For Live Trading" \
  --quantile 0.985 \
  --min-net-pips 0 \
  --min-average-net-pips 0 \
  --min-trades 50 \
  --apply
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


# ----------------------------- small utilities -----------------------------


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug_quantile(q: float) -> str:
    return f"q{q:.4f}".replace(".", "p")


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _maybe_float_arg(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        # JSON is valid YAML 1.2 and keeps the file machine-readable.
        _write_json(path, data)


def _format_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return ""


def _format_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return ""


def _print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], max_rows: int | None = None) -> None:
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        print("(no rows)")
        return
    headers = [label for _, label in columns]
    body: list[list[str]] = []
    for row in rows:
        vals = []
        for key, _ in columns:
            v = row.get(key, "")
            vals.append(str(v))
        body.append(vals)
    widths = [len(h) for h in headers]
    for vals in body:
        for i, v in enumerate(vals):
            widths[i] = max(widths[i], len(v))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for vals in body:
        print(fmt.format(*vals))


def _infer_days(args: argparse.Namespace, trades_rows: list[dict[str, str]] | None = None) -> tuple[int, int]:
    """Return (calendar_days, weekday_days) for daily averages."""
    start_s = args.replay_start
    end_s = args.replay_end
    if not start_s or not end_s:
        # Try to infer from trades timestamps if present.
        dates: set[_dt.date] = set()
        if trades_rows:
            for row in trades_rows:
                d = _parse_date(row.get("time") or row.get("date") or "")
                if d:
                    dates.add(d)
        if dates:
            start = min(dates)
            end = max(dates) + _dt.timedelta(days=1)
        else:
            # Fallback: one-year replay window, because most project replays are annual.
            start = _dt.date(2025, 1, 1)
            end = _dt.date(2026, 1, 1)
    else:
        start = _dt.date.fromisoformat(start_s)
        end = _dt.date.fromisoformat(end_s)
    if end <= start:
        raise ValueError("--replay-end must be after --replay-start")
    cal_days = (end - start).days
    weekdays = sum(1 for i in range(cal_days) if (start + _dt.timedelta(days=i)).weekday() < 5)
    return max(1, cal_days), max(1, weekdays)


def _parse_date(value: str) -> _dt.date | None:
    if not value:
        return None
    txt = value.strip()
    if not txt:
        return None
    # Handles YYYY-MM-DD, ISO timestamps, and timezone suffixes.
    if " " in txt:
        txt = txt.split(" ", 1)[0]
    if "T" in txt:
        txt = txt.split("T", 1)[0]
    try:
        return _dt.date.fromisoformat(txt[:10])
    except Exception:
        return None


# --------------------------- quantile summarising ---------------------------


@dataclass
class QuantileSummary:
    quantile: float
    models: int
    symbols: int
    trades: int
    net_pips: float
    win_rate: float
    avg_pips: float
    max_drawdown: float
    mean_drawdown: float
    profitable_models: int
    negative_models: int
    deployment_score: float
    trades_per_calendar_day: float
    trades_per_weekday: float
    net_per_calendar_day: float
    net_per_weekday: float

    def row(self) -> dict[str, Any]:
        return {
            "quantile": self.quantile,
            "models": self.models,
            "symbols": self.symbols,
            "trades": self.trades,
            "net_pips": self.net_pips,
            "win_rate": self.win_rate,
            "average_net_pips": self.avg_pips,
            "max_model_drawdown_pips": self.max_drawdown,
            "mean_model_drawdown_pips": self.mean_drawdown,
            "profitable_models": self.profitable_models,
            "negative_models": self.negative_models,
            "total_deployment_score": self.deployment_score,
            "trades_per_calendar_day": self.trades_per_calendar_day,
            "trades_per_weekday": self.trades_per_weekday,
            "net_pips_per_calendar_day": self.net_per_calendar_day,
            "net_pips_per_weekday": self.net_per_weekday,
        }


def _load_quantile_summaries(rows: list[dict[str, str]], cal_days: int, weekdays: int) -> list[QuantileSummary]:
    out: list[QuantileSummary] = []
    for r in rows:
        q = _to_float(r.get("quantile"), default=float("nan"))
        if math.isnan(q):
            continue
        trades = _to_int(r.get("trades"), 0)
        net = _to_float(r.get("net_pips"), 0.0)
        out.append(
            QuantileSummary(
                quantile=q,
                models=_to_int(r.get("models"), 0),
                symbols=_to_int(r.get("symbols"), 0),
                trades=trades,
                net_pips=net,
                win_rate=_to_float(r.get("win_rate"), 0.0),
                avg_pips=_to_float(r.get("average_net_pips"), 0.0),
                max_drawdown=_to_float(r.get("max_model_drawdown_pips"), 0.0),
                mean_drawdown=_to_float(r.get("mean_model_drawdown_pips"), 0.0),
                profitable_models=_to_int(r.get("profitable_models"), 0),
                negative_models=_to_int(r.get("negative_models"), 0),
                deployment_score=_to_float(r.get("total_deployment_score"), 0.0),
                trades_per_calendar_day=trades / cal_days,
                trades_per_weekday=trades / weekdays,
                net_per_calendar_day=net / cal_days,
                net_per_weekday=net / weekdays,
            )
        )
    out.sort(key=lambda x: x.quantile)
    return out


def _display_quantiles(summaries: list[QuantileSummary]) -> None:
    rows = []
    for s in summaries:
        rows.append(
            {
                "q": f"{s.quantile:.4f}",
                "net": _format_float(s.net_pips, 1),
                "dd": _format_float(s.max_drawdown, 1),
                "mean_dd": _format_float(s.mean_drawdown, 1),
                "trades": f"{s.trades:,}",
                "trades_day": _format_float(s.trades_per_calendar_day, 1),
                "trades_weekday": _format_float(s.trades_per_weekday, 1),
                "avg": _format_float(s.avg_pips, 3),
                "wr": _format_pct(s.win_rate, 1),
                "profitable": f"{s.profitable_models}/{s.models}",
                "score": _format_float(s.deployment_score, 1),
            }
        )
    _print_table(
        rows,
        [
            ("q", "Quantile"),
            ("net", "Net pips"),
            ("dd", "Max DD"),
            ("mean_dd", "Mean DD"),
            ("trades", "Trades"),
            ("trades_day", "Trades/day"),
            ("trades_weekday", "Trades/weekday"),
            ("avg", "Avg"),
            ("wr", "Win"),
            ("profitable", "Profitable"),
            ("score", "Score"),
        ],
    )


def _choose_quantile(args: argparse.Namespace, summaries: list[QuantileSummary]) -> float:
    if args.quantile is not None:
        q = float(args.quantile)
        candidates = [s.quantile for s in summaries]
        if not any(abs(q - c) < 1e-9 for c in candidates):
            raise SystemExit(f"Selected --quantile {q} is not present in the quantile summary: {candidates}")
        return q

    print("\nViable quantile values from the sweep:")
    _display_quantiles(summaries)

    best_score = max(summaries, key=lambda s: s.deployment_score)
    best_net = max(summaries, key=lambda s: s.net_pips)
    print(
        f"\nSuggested defaults: best score={best_score.quantile:.4f}, "
        f"best net={best_net.quantile:.4f}."
    )

    if not sys.stdin.isatty():
        print(f"Non-interactive input detected; selecting best score quantile {best_score.quantile:.4f}.")
        return best_score.quantile

    valid = {f"{s.quantile:.4f}": s.quantile for s in summaries}
    while True:
        ans = input("Choose quantile to apply, or press Enter for best score: ").strip()
        if not ans:
            return best_score.quantile
        try:
            q = float(ans)
        except ValueError:
            print("Please enter a numeric quantile, e.g. 0.985")
            continue
        for s in summaries:
            if abs(q - s.quantile) < 1e-9:
                return s.quantile
        print(f"Quantile {q} was not found. Valid choices: {', '.join(valid)}")


# ----------------------------- model filtering -----------------------------


def _row_combo_id(row: dict[str, Any]) -> str:
    return str(row.get("combo_id") or row.get("id") or "").strip()


def _manifest_model_id(model: dict[str, Any]) -> str:
    mid = str(model.get("id") or "").strip()
    if mid:
        return mid
    return "_".join(
        [
            str(model.get("symbol") or ""),
            str(model.get("model") or ""),
            str(model.get("side") or ""),
            f"epoch_{_to_int(model.get('epoch'), -1):03d}",
        ]
    )


def _index_model_results(rows: list[dict[str, str]], q: float) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        rq = _to_float(row.get("quantile"), default=float("nan"))
        if math.isnan(rq) or abs(rq - q) > 1e-9:
            continue
        cid = _row_combo_id(row)
        if cid:
            out[cid] = row
    return out


def _disable_reason(row: dict[str, str] | None, args: argparse.Namespace) -> str | None:
    if row is None:
        return "missing_from_quantile_sweep"

    net = _to_float(row.get("net_pips"), 0.0)
    avg = _to_float(row.get("average_net_pips"), 0.0)
    trades = _to_int(row.get("trades"), 0)
    win = _to_float(row.get("win_rate"), 0.0)
    dd = _to_float(row.get("max_drawdown_pips"), 0.0)
    pf = _to_float(row.get("profit_factor"), 0.0)
    score = _to_float(row.get("deployment_score"), 0.0)

    reasons: list[str] = []
    if net < args.min_net_pips:
        reasons.append(f"net_pips<{args.min_net_pips:g}")
    if avg < args.min_average_net_pips:
        reasons.append(f"average_net_pips<{args.min_average_net_pips:g}")
    if trades < args.min_trades:
        reasons.append(f"trades<{args.min_trades:g}")
    if args.min_win_rate is not None and win < args.min_win_rate:
        reasons.append(f"win_rate<{args.min_win_rate:g}")
    if args.max_drawdown_pips is not None and dd > args.max_drawdown_pips:
        reasons.append(f"drawdown>{args.max_drawdown_pips:g}")
    if args.min_profit_factor is not None and pf < args.min_profit_factor:
        reasons.append(f"profit_factor<{args.min_profit_factor:g}")
    if args.min_deployment_score is not None and score < args.min_deployment_score:
        reasons.append(f"deployment_score<{args.min_deployment_score:g}")
    if args.max_drawdown_to_net_ratio is not None:
        if net <= 0:
            reasons.append("drawdown_to_net_undefined_net<=0")
        else:
            ratio = dd / max(abs(net), 1e-9)
            if ratio > args.max_drawdown_to_net_ratio:
                reasons.append(f"drawdown_to_net>{args.max_drawdown_to_net_ratio:g}")

    return ";".join(reasons) if reasons else None


def _merge_metrics_into_model(model: dict[str, Any], row: dict[str, str], q: float) -> dict[str, Any]:
    out = dict(model)
    out["selected_quantile"] = float(q)
    out["enabled"] = True
    out["quantile_sweep_metrics"] = {
        "quantile": float(q),
        "trades": _to_int(row.get("trades"), 0),
        "wins": _to_int(row.get("wins"), 0),
        "losses": _to_int(row.get("losses"), 0),
        "win_rate": _to_float(row.get("win_rate"), 0.0),
        "average_net_pips": _to_float(row.get("average_net_pips"), 0.0),
        "net_pips": _to_float(row.get("net_pips"), 0.0),
        "max_drawdown_pips": _to_float(row.get("max_drawdown_pips"), 0.0),
        "profit_factor": _to_float(row.get("profit_factor"), 0.0),
        "deployment_score": _to_float(row.get("deployment_score"), 0.0),
    }
    return out


def _filter_manifest(
    manifest: dict[str, Any],
    model_results_at_q: dict[str, dict[str, str]],
    q: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []

    for model in manifest.get("models", []) or []:
        mid = _manifest_model_id(model)
        row = model_results_at_q.get(mid)
        reason = _disable_reason(row, args)
        if reason is None and row is not None:
            kept.append(_merge_metrics_into_model(model, row, q))
        else:
            m = dict(model)
            m["selected_quantile"] = float(q)
            m["enabled"] = False
            m["disabled_reason"] = reason or "disabled"
            if row is not None:
                m["quantile_sweep_metrics"] = _merge_metrics_into_model(model, row, q)["quantile_sweep_metrics"]
            disabled.append(m)

    new_manifest = dict(manifest)
    new_manifest["created_by"] = "select_quantile_and_disable_models.py"
    new_manifest["source_created_by"] = manifest.get("created_by")
    new_manifest["selected_quantile"] = float(q)
    new_manifest["quantile_filter"] = {
        "min_net_pips": args.min_net_pips,
        "min_average_net_pips": args.min_average_net_pips,
        "min_trades": args.min_trades,
        "min_win_rate": args.min_win_rate,
        "max_drawdown_pips": args.max_drawdown_pips,
        "max_drawdown_to_net_ratio": args.max_drawdown_to_net_ratio,
        "min_profit_factor": args.min_profit_factor,
        "min_deployment_score": args.min_deployment_score,
        "disabled_count": len(disabled),
        "kept_count": len(kept),
    }

    if args.disable_mode == "enabled_flag":
        new_manifest["models"] = kept + disabled
    else:
        new_manifest["models"] = kept
        new_manifest["disabled_models"] = disabled

    return new_manifest, kept, disabled


def _model_report_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in models:
        metrics = m.get("quantile_sweep_metrics") or {}
        rows.append(
            {
                "id": _manifest_model_id(m),
                "enabled": m.get("enabled", True),
                "disabled_reason": m.get("disabled_reason", ""),
                "symbol": m.get("symbol", ""),
                "model": m.get("model", ""),
                "side": m.get("side", ""),
                "epoch": m.get("epoch", ""),
                "selected_quantile": m.get("selected_quantile", ""),
                "trades": metrics.get("trades", ""),
                "net_pips": metrics.get("net_pips", ""),
                "average_net_pips": metrics.get("average_net_pips", ""),
                "win_rate": metrics.get("win_rate", ""),
                "max_drawdown_pips": metrics.get("max_drawdown_pips", ""),
                "profit_factor": metrics.get("profit_factor", ""),
                "deployment_score": metrics.get("deployment_score", ""),
                "config_path": m.get("config_path", ""),
                "model_path": m.get("model_path", ""),
                "scaler_path": m.get("scaler_path", ""),
                "features_path": m.get("features_path", ""),
            }
        )
    return rows


# --------------------------- optional daily reports --------------------------


def _daily_reports(trades_rows: list[dict[str, str]], output_dir: Path) -> None:
    if not trades_rows:
        return
    daily: dict[tuple[float, _dt.date], dict[str, Any]] = {}
    symbol_daily: dict[tuple[float, _dt.date, str], dict[str, Any]] = {}

    for row in trades_rows:
        q = _to_float(row.get("quantile"), default=float("nan"))
        d = _parse_date(row.get("time") or row.get("date") or "")
        if math.isnan(q) or d is None:
            continue
        pips = _to_float(row.get("pips"), 0.0)
        symbol = str(row.get("symbol") or "")

        for key, store in [((q, d), daily), ((q, d, symbol), symbol_daily)]:
            rec = store.setdefault(
                key,
                {
                    "quantile": q,
                    "date": d.isoformat(),
                    "symbol": symbol if len(key) == 3 else "ALL",
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "net_pips": 0.0,
                },
            )
            rec["trades"] += 1
            rec["wins"] += 1 if pips > 0 else 0
            rec["losses"] += 1 if pips <= 0 else 0
            rec["net_pips"] += pips

    for rec in list(daily.values()) + list(symbol_daily.values()):
        trades = max(1, int(rec["trades"]))
        rec["win_rate"] = rec["wins"] / trades
        rec["average_net_pips"] = rec["net_pips"] / trades

    daily_rows = sorted(daily.values(), key=lambda r: (float(r["quantile"]), str(r["date"])))
    sym_rows = sorted(symbol_daily.values(), key=lambda r: (float(r["quantile"]), str(r["date"]), str(r["symbol"])))
    _write_csv(output_dir / "daily_results_by_quantile.csv", daily_rows)
    _write_csv(output_dir / "daily_symbol_results_by_quantile.csv", sym_rows)

    # Compact quantile-by-daily distribution.
    by_q: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for r in daily_rows:
        by_q[float(r["quantile"])].append(r)

    summary_rows: list[dict[str, Any]] = []
    for q, rows in sorted(by_q.items()):
        if not rows:
            continue
        trades_vals = [int(r["trades"]) for r in rows]
        net_vals = [float(r["net_pips"]) for r in rows]
        wins = [1 if v > 0 else 0 for v in net_vals]
        summary_rows.append(
            {
                "quantile": q,
                "days_with_trades": len(rows),
                "total_trades": sum(trades_vals),
                "avg_trades_per_active_day": sum(trades_vals) / max(1, len(rows)),
                "max_trades_day": max(trades_vals),
                "min_trades_day": min(trades_vals),
                "total_net_pips": sum(net_vals),
                "avg_net_pips_per_active_day": sum(net_vals) / max(1, len(rows)),
                "best_day_pips": max(net_vals),
                "worst_day_pips": min(net_vals),
                "profitable_days": sum(wins),
                "losing_days": len(rows) - sum(wins),
                "profitable_day_rate": sum(wins) / max(1, len(rows)),
            }
        )
    _write_csv(output_dir / "daily_distribution_summary.csv", summary_rows)


# ------------------------- optional config patching -------------------------


def _resolve_project_path(project_root: Path, value: str) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    return project_root / p


def _set_replay_quantile(cfg: dict[str, Any], side: str, q: float) -> dict[str, Any]:
    rcfg = cfg.setdefault("replay", {})
    if not isinstance(rcfg, dict):
        cfg["replay"] = rcfg = {}
    rcfg["threshold_mode"] = rcfg.get("threshold_mode", "rolling_score_quantile")
    side_l = str(side).lower()
    rcfg[f"{side_l}_quantile"] = float(q)
    scfg = rcfg.setdefault(side_l, {})
    if not isinstance(scfg, dict):
        rcfg[side_l] = scfg = {}
    scfg["quantile"] = float(q)
    return cfg


def _patch_model_configs(project_root: Path, kept_models: list[dict[str, Any]], q: float, apply: bool, backup_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in kept_models:
        cfg_rel = str(m.get("config_path") or "").strip()
        if not cfg_rel:
            rows.append({"id": _manifest_model_id(m), "config_path": "", "status": "no_config_path"})
            continue
        cfg_path = _resolve_project_path(project_root, cfg_rel)
        if not cfg_path.exists():
            rows.append({"id": _manifest_model_id(m), "config_path": str(cfg_path), "status": "missing"})
            continue
        if yaml is None:
            rows.append({"id": _manifest_model_id(m), "config_path": str(cfg_path), "status": "skipped_no_pyyaml"})
            continue
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                rows.append({"id": _manifest_model_id(m), "config_path": str(cfg_path), "status": "skipped_not_mapping"})
                continue
            data = _set_replay_quantile(data, str(m.get("side") or ""), q)
            if apply:
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cfg_path, backup_dir / f"{cfg_path.name}.{_now_stamp()}.bak")
                with cfg_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
                status = "updated"
            else:
                status = "would_update"
            rows.append({"id": _manifest_model_id(m), "config_path": str(cfg_path), "status": status, "quantile": q})
        except Exception as e:
            rows.append({"id": _manifest_model_id(m), "config_path": str(cfg_path), "status": f"error:{e}"})
    return rows


# ----------------------------------- main -----------------------------------


def _default_paths(args: argparse.Namespace) -> None:
    live_root = Path(args.live_root)
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else live_root / "quantile_sweep_replay"

    def choose(user_value: str | None, sweep_name: str, local_name: str | None = None) -> Path:
        if user_value:
            return Path(user_value)
        p = sweep_dir / sweep_name
        if p.exists():
            return p
        if local_name:
            lp = Path(local_name)
            if lp.exists():
                return lp
        return p

    args.sweep_dir = str(sweep_dir)
    args.manifest = str(Path(args.manifest) if args.manifest else live_root / "live_ensemble_manifest.json")
    args.model_results_csv = str(choose(args.model_results_csv, "quantile_sweep_model_results.csv", "quantile_sweep_model_results.csv"))
    args.quantile_summary_csv = str(choose(args.quantile_summary_csv, "quantile_sweep_quantile_summary.csv", "quantile_sweep_quantile_summary.csv"))
    args.trades_csv = str(choose(args.trades_csv, "quantile_sweep_trades.csv", "quantile_sweep_trades.csv")) if args.trades_csv or (sweep_dir / "quantile_sweep_trades.csv").exists() or Path("quantile_sweep_trades.csv").exists() else None
    if args.output_dir is None:
        args.output_dir = str(sweep_dir / "selected_quantile_filter")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pick a quantile from quantile sweep replay results and disable weak/negative staged models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--live-root", default="For Live Trading", help="Live trading folder containing live_ensemble_manifest.json")
    p.add_argument("--project-root", default=".", help="Project root used to resolve manifest config paths")
    p.add_argument("--sweep-dir", default=None, help="Folder containing quantile sweep outputs; default <live-root>/quantile_sweep_replay")
    p.add_argument("--manifest", default=None, help="Path to live_ensemble_manifest.json; default <live-root>/live_ensemble_manifest.json")
    p.add_argument("--model-results-csv", default=None, help="Path to quantile_sweep_model_results.csv")
    p.add_argument("--quantile-summary-csv", default=None, help="Path to quantile_sweep_quantile_summary.csv")
    p.add_argument("--trades-csv", default=None, help="Optional quantile_sweep_trades.csv for exact daily by-date output")
    p.add_argument("--output-dir", default=None, help="Where to write filtered manifests/reports")
    p.add_argument("--replay-start", default=None, help="Replay start date YYYY-MM-DD for daily averages")
    p.add_argument("--replay-end", default=None, help="Replay end date YYYY-MM-DD for daily averages, exclusive")
    p.add_argument("--quantile", type=float, default=None, help="Quantile to select. If omitted, show menu and ask.")

    # Filter defaults are intentionally permissive but remove negative/zero-return models.
    p.add_argument("--min-net-pips", type=float, default=0.0, help="Disable models below this selected-quantile net return")
    p.add_argument("--min-average-net-pips", type=float, default=0.0, help="Disable models below this average net pips/trade")
    p.add_argument("--min-trades", type=float, default=1.0, help="Disable models below this selected-quantile trade count")
    p.add_argument("--min-win-rate", type=float, default=None, help="Optional selected-quantile win-rate filter, e.g. 0.43")
    p.add_argument("--max-drawdown-pips", type=float, default=None, help="Optional max model drawdown filter")
    p.add_argument("--max-drawdown-to-net-ratio", type=float, default=None, help="Optional drawdown/net-pips filter")
    p.add_argument("--min-profit-factor", type=float, default=None, help="Optional profit-factor filter")
    p.add_argument("--min-deployment-score", type=float, default=None, help="Optional deployment-score filter")

    p.add_argument("--disable-mode", choices=["remove", "enabled_flag"], default="remove", help="Remove disabled models from manifest or keep them with enabled=false")
    p.add_argument("--apply", action="store_true", help="Replace live manifest in-place after backing it up")
    p.add_argument("--update-configs", action="store_true", help="When --apply is used, also update kept model YAML configs to selected quantile")
    p.add_argument("--backup-dir", default=None, help="Backup folder; default <output-dir>/backups")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _default_paths(args)

    live_root = Path(args.live_root)
    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(args.backup_dir) if args.backup_dir else output_dir / "backups"

    manifest_path = Path(args.manifest)
    model_results_path = Path(args.model_results_csv)
    quantile_summary_path = Path(args.quantile_summary_csv)
    trades_path = Path(args.trades_csv) if args.trades_csv else None

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not model_results_path.exists():
        raise SystemExit(f"Model results CSV not found: {model_results_path}")
    if not quantile_summary_path.exists():
        raise SystemExit(f"Quantile summary CSV not found: {quantile_summary_path}")

    trades_rows = _read_csv(trades_path) if trades_path and trades_path.exists() else None
    cal_days, weekdays = _infer_days(args, trades_rows)
    quantile_rows = _read_csv(quantile_summary_path)
    summaries = _load_quantile_summaries(quantile_rows, cal_days, weekdays)
    if not summaries:
        raise SystemExit("No quantile rows found.")

    print("\nQuantile sweep overview")
    print(f"  manifest:      {manifest_path}")
    print(f"  model results: {model_results_path}")
    print(f"  summary:       {quantile_summary_path}")
    if trades_rows is not None:
        print(f"  trades:        {trades_path}")
    print(f"  daily averages: {cal_days} calendar days / {weekdays} weekdays\n")
    _display_quantiles(summaries)

    q = _choose_quantile(args, summaries)
    print(f"\nSelected quantile: {q:.4f}")

    manifest = _read_json(manifest_path)
    model_results = _read_csv(model_results_path)
    by_id = _index_model_results(model_results, q)
    new_manifest, kept, disabled = _filter_manifest(manifest, by_id, q, args)

    slug = _slug_quantile(q)
    filtered_json = output_dir / f"live_ensemble_manifest_{slug}_filtered.json"
    filtered_yaml = output_dir / f"live_ensemble_manifest_{slug}_filtered.yaml"
    kept_csv = output_dir / f"kept_models_{slug}.csv"
    disabled_csv = output_dir / f"disabled_models_{slug}.csv"
    viable_csv = output_dir / "viable_quantiles.csv"

    _write_csv(viable_csv, [s.row() for s in summaries])
    _write_json(filtered_json, new_manifest)
    _write_yaml(filtered_yaml, new_manifest)
    _write_csv(kept_csv, _model_report_rows(kept))
    _write_csv(disabled_csv, _model_report_rows(disabled))

    if trades_rows is not None:
        _daily_reports(trades_rows, output_dir)

    cfg_rows: list[dict[str, Any]] = []
    if args.update_configs:
        cfg_rows = _patch_model_configs(project_root, kept, q, apply=args.apply, backup_dir=backup_dir)
        _write_csv(output_dir / f"config_quantile_update_{slug}.csv", cfg_rows)

    if args.apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = _now_stamp()
        backup_manifest = backup_dir / f"{manifest_path.name}.{stamp}.bak"
        shutil.copy2(manifest_path, backup_manifest)
        shutil.copy2(filtered_json, manifest_path)
        # Maintain YAML sibling if present or if the filtered YAML was produced.
        manifest_yaml = manifest_path.with_suffix(".yaml")
        if manifest_yaml.exists():
            shutil.copy2(manifest_yaml, backup_dir / f"{manifest_yaml.name}.{stamp}.bak")
            shutil.copy2(filtered_yaml, manifest_yaml)
        print(f"\nAPPLIED: replaced {manifest_path}")
        print(f"Backup: {backup_manifest}")
    else:
        print("\nDry run only: original live manifest was not modified. Use --apply after reviewing outputs.")

    print("\nFilter result")
    print(f"  kept models:     {len(kept)}")
    print(f"  disabled models: {len(disabled)}")
    print(f"  filtered JSON:   {filtered_json}")
    print(f"  filtered YAML:   {filtered_yaml}")
    print(f"  kept CSV:        {kept_csv}")
    print(f"  disabled CSV:    {disabled_csv}")
    print(f"  quantiles CSV:   {viable_csv}")
    if trades_rows is not None:
        print(f"  daily CSV:       {output_dir / 'daily_results_by_quantile.csv'}")
        print(f"  daily summary:   {output_dir / 'daily_distribution_summary.csv'}")
    if args.update_configs:
        print(f"  config updates:  {output_dir / f'config_quantile_update_{slug}.csv'}")
        if yaml is None:
            print("  Note: PyYAML was not available, so model config YAMLs were not modified.")

    if disabled:
        print("\nTop disabled models by original selected-quantile net pips / reason:")
        rows = _model_report_rows(disabled)
        rows.sort(key=lambda r: _to_float(r.get("net_pips"), -1e9), reverse=True)
        table = []
        for r in rows[:12]:
            table.append(
                {
                    "id": r["id"],
                    "net": _format_float(r.get("net_pips"), 1),
                    "avg": _format_float(r.get("average_net_pips"), 3),
                    "dd": _format_float(r.get("max_drawdown_pips"), 1),
                    "reason": r.get("disabled_reason", ""),
                }
            )
        _print_table(table, [("id", "Model"), ("net", "Net"), ("avg", "Avg"), ("dd", "DD"), ("reason", "Reason")])

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
