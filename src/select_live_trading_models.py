#!/usr/bin/env python3
"""Select deployment-style best epochs and stage direction-policy models for live trading.

This script scans training report JSON files, chooses the best practical epoch for
live/demo deployment, copies or moves the matching epoch checkpoint artifacts into
an isolated "For Live Trading" folder, and writes per-symbol ensemble configs plus
a global manifest.

Typical usage from the project root:

    python select_live_trading_models.py \
      --logs-root logs \
      --project-root . \
      --output-root "For Live Trading" \
      --min-trades 50 \
      --max-drawdown-pips 150

The selector deliberately does not blindly trust the trainer's stored best_epoch.
It re-scores every epoch with replay results using a deployment-style score that
rewards net pips, average pips/trade and win rate, while penalising drawdown and
rejecting unstable high-drawdown epochs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


REPORT_GLOBS = [
    "**/*_direction_training_report.json",
    "**/direction_training_report.json",
    "**/*direction_training_replay_each_epoch_summary*.json",
]

ARTIFACT_KINDS = {
    "model": ("model_path", "direction_policy.pt"),
    "scaler": ("scaler_path", "direction_scaler.pkl"),
    "features": ("features_path", "direction_features.json"),
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except Exception:
            return default


def _normalise_side(side: Any) -> str:
    side_s = str(side or "").strip().lower()
    if side_s in {"long", "buy_only"}:
        return "buy"
    if side_s in {"short", "sell_only"}:
        return "sell"
    if side_s not in {"buy", "sell"}:
        return "unknown"
    return side_s


def _timeframe(report: dict[str, Any]) -> str:
    return str(report.get("timeframe") or ((report.get("config_snapshot") or {}).get("resolved_sections") or {}).get("project", {}).get("timeframe") or "M5").upper()


def _model_token(report: dict[str, Any], report_path: Path) -> str:
    symbol = str(report.get("symbol") or "").upper()
    artifacts = report.get("artifacts") or {}
    for key in ("model_path", "report_path"):
        p = str(artifacts.get(key) or "")
        if p:
            parts = Path(p).parts
            if symbol and symbol in parts:
                idx = parts.index(symbol)
                if idx > 0:
                    return re.sub(r"[^A-Za-z0-9_\-]+", "_", parts[idx - 1])
    # Side-named reports commonly live in logs/side_named_reports/<model>/...
    parts = report_path.parts
    if "side_named_reports" in parts:
        idx = parts.index("side_named_reports")
        if len(parts) > idx + 1:
            return re.sub(r"[^A-Za-z0-9_\-]+", "_", parts[idx + 1])
    model_type = str(report.get("model_type") or "model")
    for suffix in (
        "_hierarchical_gate_direction_side_setup_ranking",
        "_gate_direction_v1",
        "_direction_v1",
        "_side_setup_v1",
    ):
        model_type = model_type.replace(suffix, "")
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", model_type).strip("_") or "model"


def _find_report_files(logs_root: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in REPORT_GLOBS:
        for p in logs_root.glob(pattern):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def _report_records_from_file(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return one or more report payloads from a report or summary JSON file."""
    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("reports"), list):
        out: list[tuple[Path, dict[str, Any]]] = []
        for idx, item in enumerate(payload.get("reports") or []):
            if isinstance(item, dict) and "symbol" in item:
                # A synthetic path keeps the original filename in the CSV while
                # distinguishing multiple reports embedded in one summary.
                out.append((Path(f"{path}::report_{idx}"), item))
        return out
    if isinstance(payload, dict) and "symbol" in payload:
        return [(path, payload)]
    return []


def _iter_epoch_rows(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in report.get("history") or []:
        if not isinstance(item, dict):
            continue
        replay = item.get("replay") or {}
        if not isinstance(replay, dict):
            continue
        if "trades" not in replay and "net_pips" not in replay:
            continue
        epoch = _safe_int(item.get("epoch"), -1)
        if epoch < 0:
            continue
        yield {"epoch": epoch, "history_item": item, "replay": replay}


def _deployment_score(metrics: dict[str, float], args: argparse.Namespace) -> tuple[float, str, bool]:
    trades = metrics["trades"]
    net = metrics["net_pips"]
    avg = metrics["average_net_pips"]
    win_rate = metrics["win_rate"]
    dd = metrics["max_drawdown_pips"]

    reasons: list[str] = []
    usable = True
    if trades < args.min_trades:
        usable = False
        reasons.append(f"trades<{args.min_trades:g}")
    if net < args.min_net_pips:
        usable = False
        reasons.append(f"net_pips<{args.min_net_pips:g}")
    if avg < args.min_average_net_pips:
        usable = False
        reasons.append(f"avg_pips<{args.min_average_net_pips:g}")
    if args.min_win_rate is not None and win_rate < args.min_win_rate:
        usable = False
        reasons.append(f"win_rate<{args.min_win_rate:g}")
    if args.max_drawdown_pips is not None and dd > args.max_drawdown_pips:
        usable = False
        reasons.append(f"drawdown>{args.max_drawdown_pips:g}")
    if net > 0 and args.max_drawdown_to_net_ratio is not None:
        ratio = dd / max(abs(net), 1e-9)
        if ratio > args.max_drawdown_to_net_ratio:
            usable = False
            reasons.append(f"drawdown/net>{args.max_drawdown_to_net_ratio:g}")

    # Deployment-style ranking: total profit matters, but a model with high
    # average edge, decent win rate and controlled drawdown should beat a noisy
    # high-trade/high-drawdown epoch.
    score = (
        args.score_net_pips_weight * net
        + args.score_avg_pips_weight * avg
        + args.score_win_rate_weight * win_rate
        + args.score_trade_count_weight * min(trades, args.trade_count_cap)
        - args.score_drawdown_weight * dd
    )

    # Soft instability penalties. These do not necessarily exclude an epoch, but
    # they stop very high-drawdown early epochs from winning just because net pips
    # happened to be high.
    if dd > 0 and net > 0:
        dd_net_ratio = dd / max(net, 1e-9)
        if dd_net_ratio > 1.0:
            score -= args.drawdown_to_net_penalty_weight * (dd_net_ratio - 1.0)
    if avg <= 0:
        score -= args.negative_average_penalty
    if net <= 0:
        score -= args.negative_net_penalty
    if trades < args.min_trades:
        score -= args.low_trade_penalty + (args.min_trades - trades)
    if args.max_drawdown_pips is not None and dd > args.max_drawdown_pips:
        score -= args.high_drawdown_penalty + (dd - args.max_drawdown_pips)

    return float(score), ";".join(reasons) if reasons else "ok", bool(usable)


def _extract_metrics(epoch_row: dict[str, Any]) -> dict[str, float]:
    r = epoch_row["replay"]
    return {
        "trades": _safe_float(r.get("trades")),
        "net_pips": _safe_float(r.get("net_pips")),
        "win_rate": _safe_float(r.get("win_rate")),
        "average_net_pips": _safe_float(r.get("average_net_pips")),
        "max_drawdown_pips": _safe_float(r.get("max_drawdown_pips")),
        "replay_score": _safe_float(r.get("replay_score")),
        "buy_trades": _safe_float(r.get("buy_trades")),
        "buy_net_pips": _safe_float(r.get("buy_net_pips")),
        "sell_trades": _safe_float(r.get("sell_trades")),
        "sell_net_pips": _safe_float(r.get("sell_net_pips")),
    }


def _choose_epoch(report: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    ranked: list[dict[str, Any]] = []
    for epoch_row in _iter_epoch_rows(report):
        metrics = _extract_metrics(epoch_row)
        score, reason, usable = _deployment_score(metrics, args)
        item = {
            "epoch": int(epoch_row["epoch"]),
            "deployment_score": score,
            "deployment_usable": usable,
            "deployment_reason": reason,
            **metrics,
            "history_item": epoch_row["history_item"],
        }
        ranked.append(item)
    if not ranked:
        return None, []
    usable = [x for x in ranked if x["deployment_usable"]]
    pool = usable if usable else ranked
    pool = sorted(pool, key=lambda x: (x["deployment_score"], x["net_pips"], x["average_net_pips"]), reverse=True)
    return pool[0], sorted(ranked, key=lambda x: x["deployment_score"], reverse=True)


def _path_candidates(path_value: Any, *, project_root: Path, output_root: Path) -> list[Path]:
    if not path_value:
        return []
    raw = Path(str(path_value))
    candidates: list[Path] = []
    candidates.append(raw)
    if not raw.is_absolute():
        candidates.append(project_root / raw)
        candidates.append(output_root / raw)
    parts = raw.parts
    for marker in ("models", "logs", "data", "config"):
        if marker in parts:
            idx = parts.index(marker)
            rel = Path(*parts[idx:])
            candidates.append(project_root / rel)
            candidates.append(output_root / rel)
    # de-duplicate while preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            out.append(c)
            seen.add(key)
    return out


def _first_existing_path(path_value: Any, *, project_root: Path, output_root: Path) -> Path | None:
    for c in _path_candidates(path_value, project_root=project_root, output_root=output_root):
        if c.exists():
            return c
    return None


def _artifact_paths_for_epoch(report: dict[str, Any], chosen: dict[str, Any], *, project_root: Path, output_root: Path) -> dict[str, Path | None]:
    item = chosen.get("history_item") or {}
    ckp = item.get("checkpoint_artifacts") or {}
    artifacts = report.get("artifacts") or {}
    out: dict[str, Path | None] = {}
    for kind, (key, _) in ARTIFACT_KINDS.items():
        value = ckp.get(key)
        if not value:
            # Fallback: if the selected deployment epoch happens to equal the
            # trainer's best_epoch, the final artifact is usually the chosen one.
            if _safe_int(report.get("best_epoch"), -999) == int(chosen["epoch"]):
                value = artifacts.get(key)
        out[kind] = _first_existing_path(value, project_root=project_root, output_root=output_root) if value else None
    return out


def _copy_or_move(src: Path, dst: Path, action: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if action == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def _side_trade_mode(side: str) -> str:
    return "buy_only" if side == "buy" else "sell_only" if side == "sell" else "disabled"


def _live_config_from_report(
    report: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
    model_token: str,
    side: str,
    chosen: dict[str, Any],
    staged_paths: dict[str, Path],
    combo_id: str,
    live_root: Path,
) -> dict[str, Any]:
    """Return the lightweight per-model/per-side overlay config.

    The ensemble runner loads the copied universal generic config first, then
    overlays this file. Therefore this YAML intentionally avoids carrying shared
    live risk, execution, spread-control or analytic-gate settings. Those should
    be edited once in the generic config copied into the live trading folder.
    """
    resolved = (((report.get("config_snapshot") or {}).get("resolved_sections")) or {})
    resolved = resolved if isinstance(resolved, dict) else {}

    cfg: dict[str, Any] = {}

    cfg["project"] = {
        "name": f"live_{symbol}_{model_token}_{side}_epoch_{int(chosen['epoch']):03d}",
        "timeframe": timeframe,
    }
    cfg["trading"] = {
        "symbols": [symbol],
        "timeframe": timeframe,
    }

    # Model architecture/hyperparameters are model-specific and must remain with
    # the staged side config so the universal generic file can be shared by all
    # model families.
    if isinstance(resolved.get("model"), dict):
        cfg["model"] = deepcopy(resolved["model"])
    elif isinstance(report.get("model_config"), dict):
        cfg["model"] = deepcopy(report["model_config"])

    cfg["paths"] = {
        "model_dir": str(staged_paths["model"].parent).replace("\\", "/"),
        "log_dir": str((live_root / "logs" / model_token / symbol / side)).replace("\\", "/"),
    }

    # Side-specific replay restriction. Thresholds, spread limits and other
    # universal replay/live gates should come from the copied generic config.
    cfg["replay"] = {
        "allow_buy": side == "buy",
        "allow_sell": side == "sell",
    }

    cfg["training"] = {
        "train_side": side,
        "side_setup_train_side": side,
        "buy_setup_loss_weight": 1.0 if side == "buy" else 0.0,
        "sell_setup_loss_weight": 1.0 if side == "sell" else 0.0,
    }
    cfg["symbol_trade_modes"] = {"symbols": {symbol: _side_trade_mode(side)}}

    # Artifact paths are necessarily unique to this model/epoch. Shared live
    # logging, risk, execution and analytic-gate settings come from the copied
    # universal generic config used by the ensemble runner.
    cfg["live_direction_policy"] = {
        "model_path": str(staged_paths["model"]).replace("\\", "/"),
        "scaler_path": str(staged_paths["scaler"]).replace("\\", "/"),
        "features_path": str(staged_paths["features"]).replace("\\", "/"),
        "cooldown_state_json": str((live_root / "state" / f"{combo_id}_cooldown_state.json")).replace("\\", "/"),
    }

    cfg["live_model_selection"] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "model": model_token,
        "side": side,
        "epoch": int(chosen["epoch"]),
        "deployment_score": float(chosen["deployment_score"]),
        "deployment_usable": bool(chosen["deployment_usable"]),
        "deployment_reason": chosen["deployment_reason"],
        "net_pips": float(chosen["net_pips"]),
        "trades": int(chosen["trades"]),
        "win_rate": float(chosen["win_rate"]),
        "average_net_pips": float(chosen["average_net_pips"]),
        "max_drawdown_pips": float(chosen["max_drawdown_pips"]),
    }
    return cfg


def _resolve_generic_config(path_value: str | None, *, project_root: Path) -> Path | None:
    if not path_value:
        return None
    raw = Path(path_value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(project_root / raw)
        candidates.append(Path.cwd() / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _stage_generic_config(args: argparse.Namespace, *, project_root: Path, live_root: Path) -> Path | None:
    """Copy the universal live config into the live trading folder.

    The ensemble runner uses this copied config as the source of shared live
    parameters: risk, execution, spread/analytics gates, MT5 settings, feature
    preparation settings, and any other universal live configuration.
    """
    if not getattr(args, "generic_config", None):
        return None
    src = _resolve_generic_config(args.generic_config, project_root=project_root)
    if src is None:
        msg = f"Generic config not found: {args.generic_config}"
        if getattr(args, "require_generic_config", False):
            raise SystemExit(msg)
        print(f"[WARN] {msg}; live runner will need --universal-config or a config copied manually.")
        return None

    dst_name = getattr(args, "generic_config_output_name", None) or src.name
    dst = live_root / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage best deployment-style direction-policy epochs for live trading.")
    p.add_argument("--logs-root", default="logs", help="Training logs root to scan.")
    p.add_argument("--project-root", default=".", help="Project root used to resolve model/checkpoint paths.")
    p.add_argument("--output-root", default="For Live Trading", help="Folder to create/update with staged live models.")
    p.add_argument("--generic-config", default="config/direction_settings_generic_multisymbol_31_symbols.yaml", help="Universal live config to copy into the live trading folder and use as the base config at runtime.")
    p.add_argument("--generic-config-output-name", default=None, help="Filename for the copied universal config inside output-root. Defaults to the source filename.")
    p.add_argument("--require-generic-config", action="store_true", help="Fail if --generic-config cannot be found/copied.")
    p.add_argument("--timeframe", default=None, help="Optional timeframe override, e.g. M5.")
    p.add_argument("--symbols", nargs="*", default=None, help="Optional symbol whitelist.")
    p.add_argument("--models", nargs="*", default=None, help="Optional model-token whitelist.")
    p.add_argument("--sides", nargs="*", default=["buy", "sell"], help="Sides to stage.")
    p.add_argument("--action", choices=["copy", "move"], default="copy", help="Copy by default; use move only if you want to remove epoch files from their original folders.")
    p.add_argument("--require-artifacts", action="store_true", help="Fail if selected epoch checkpoint/scaler/features are missing.")
    p.add_argument("--include-unusable", action="store_true", help="Stage the best relaxed epoch even when it fails deployment filters.")

    # Hard deployment filters.
    p.add_argument("--min-trades", type=float, default=50)
    p.add_argument("--min-net-pips", type=float, default=0.0)
    p.add_argument("--min-average-net-pips", type=float, default=0.0)
    p.add_argument("--min-win-rate", type=float, default=None)
    p.add_argument("--max-drawdown-pips", type=float, default=150.0)
    p.add_argument("--max-drawdown-to-net-ratio", type=float, default=2.5)

    # Score weights.
    p.add_argument("--score-net-pips-weight", type=float, default=1.0)
    p.add_argument("--score-avg-pips-weight", type=float, default=50.0)
    p.add_argument("--score-win-rate-weight", type=float, default=50.0)
    p.add_argument("--score-drawdown-weight", type=float, default=0.35)
    p.add_argument("--score-trade-count-weight", type=float, default=0.02)
    p.add_argument("--trade-count-cap", type=float, default=300.0)
    p.add_argument("--drawdown-to-net-penalty-weight", type=float, default=40.0)
    p.add_argument("--negative-average-penalty", type=float, default=250.0)
    p.add_argument("--negative-net-penalty", type=float, default=500.0)
    p.add_argument("--low-trade-penalty", type=float, default=1000.0)
    p.add_argument("--high-drawdown-penalty", type=float, default=1000.0)
    args = p.parse_args()

    logs_root = Path(args.logs_root)
    project_root = Path(args.project_root)
    live_root = Path(args.output_root)
    universal_config_path = _stage_generic_config(args, project_root=project_root, live_root=live_root)
    symbols_filter = {s.upper() for s in args.symbols} if args.symbols else None
    models_filter = {m.lower() for m in args.models} if args.models else None
    sides_filter = {_normalise_side(s) for s in args.sides}

    report_files = _find_report_files(logs_root)
    if not report_files:
        raise SystemExit(f"No direction training report/summary JSON files found under {logs_root}")

    report_records: list[tuple[Path, dict[str, Any]]] = []
    for report_file in report_files:
        try:
            report_records.extend(_report_records_from_file(report_file))
        except Exception as exc:
            print(f"[WARN] Could not read {report_file}: {exc}")
            continue
    if not report_records:
        raise SystemExit(f"No embedded direction training reports found under {logs_root}")

    selections: list[dict[str, Any]] = []
    all_epoch_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    seen_reports: set[tuple[str, str, str, str]] = set()

    for report_path, report in report_records:
        if not isinstance(report, dict) or "symbol" not in report:
            continue
        symbol = str(report.get("symbol") or "").upper()
        if symbols_filter and symbol not in symbols_filter:
            continue
        side = _normalise_side(report.get("train_side") or ((report.get("training_decision_parameters") or {}).get("train_side")) or ((report.get("setup_loss") or {}).get("train_side")) or (((report.get("config_snapshot") or {}).get("resolved_sections") or {}).get("training", {}) or {}).get("train_side"))
        if side not in sides_filter:
            continue
        model_token = _model_token(report, report_path)
        if models_filter and model_token.lower() not in models_filter:
            continue
        timeframe = str(args.timeframe or _timeframe(report)).upper()
        report_key = (
            symbol,
            model_token,
            side,
            str((report.get("config_snapshot") or {}).get("config_sha256") or (report.get("artifacts") or {}).get("model_path") or report_path),
        )
        if report_key in seen_reports:
            continue
        seen_reports.add(report_key)

        chosen, ranked = _choose_epoch(report, args)
        for r in ranked:
            all_epoch_rows.append({
                "symbol": symbol,
                "model": model_token,
                "side": side,
                "timeframe": timeframe,
                "report_path": str(report_path),
                "epoch": r["epoch"],
                "deployment_score": r["deployment_score"],
                "deployment_usable": r["deployment_usable"],
                "deployment_reason": r["deployment_reason"],
                "trades": r["trades"],
                "net_pips": r["net_pips"],
                "win_rate": r["win_rate"],
                "average_net_pips": r["average_net_pips"],
                "max_drawdown_pips": r["max_drawdown_pips"],
                "replay_score": r["replay_score"],
            })
        if chosen is None:
            continue
        if not chosen["deployment_usable"] and not args.include_unusable:
            print(f"[SKIP] {symbol} {model_token} {side}: no epoch passed deployment filters; best relaxed epoch {chosen['epoch']} reason={chosen['deployment_reason']}")
            continue

        src_paths = _artifact_paths_for_epoch(report, chosen, project_root=project_root, output_root=live_root)
        missing = [kind for kind, pth in src_paths.items() if pth is None]
        if missing and args.require_artifacts:
            raise SystemExit(f"Missing selected epoch artifacts for {symbol} {model_token} {side} epoch {chosen['epoch']}: {missing}")

        combo_id = f"{symbol}_{model_token}_{side}_epoch_{int(chosen['epoch']):03d}"
        stage_dir = live_root / symbol / model_token / side
        staged_paths: dict[str, Path] = {
            "model": stage_dir / f"{symbol}_{timeframe}_{model_token}_{side}_epoch_{int(chosen['epoch']):03d}_direction_policy.pt",
            "scaler": stage_dir / f"{symbol}_{timeframe}_{model_token}_{side}_epoch_{int(chosen['epoch']):03d}_direction_scaler.pkl",
            "features": stage_dir / f"{symbol}_{timeframe}_{model_token}_{side}_epoch_{int(chosen['epoch']):03d}_direction_features.json",
        }
        copied: dict[str, str | None] = {}
        for kind, src in src_paths.items():
            if src is None:
                copied[kind] = None
                continue
            _copy_or_move(src, staged_paths[kind], args.action)
            copied[kind] = str(staged_paths[kind])

        if missing:
            print(f"[WARN] {symbol} {model_token} {side} epoch {chosen['epoch']}: missing artifacts {missing}; config will still be written for inspection.")

        if not missing:
            live_cfg = _live_config_from_report(
                report,
                symbol=symbol,
                timeframe=timeframe,
                model_token=model_token,
                side=side,
                chosen=chosen,
                staged_paths=staged_paths,
                combo_id=combo_id,
                live_root=live_root,
            )
            config_path = stage_dir / f"{symbol}_{timeframe}_{model_token}_{side}_live.yaml"
            _write_yaml(config_path, live_cfg)
        else:
            config_path = stage_dir / f"{symbol}_{timeframe}_{model_token}_{side}_live.MISSING_ARTIFACTS.yaml"
            _write_yaml(config_path, {"missing_artifacts": missing, "source_report": str(report_path)})

        row = {
            "symbol": symbol,
            "model": model_token,
            "side": side,
            "timeframe": timeframe,
            "epoch": int(chosen["epoch"]),
            "deployment_score": float(chosen["deployment_score"]),
            "deployment_usable": bool(chosen["deployment_usable"]),
            "deployment_reason": chosen["deployment_reason"],
            "trades": int(chosen["trades"]),
            "net_pips": float(chosen["net_pips"]),
            "win_rate": float(chosen["win_rate"]),
            "average_net_pips": float(chosen["average_net_pips"]),
            "max_drawdown_pips": float(chosen["max_drawdown_pips"]),
            "report_path": str(report_path),
            "live_config_path": str(config_path),
            "stage_dir": str(stage_dir),
            "missing_artifacts": ";".join(missing),
            **{f"staged_{k}_path": v for k, v in copied.items()},
        }
        selections.append(row)
        if not missing:
            manifest_entries.append({
                "id": combo_id,
                "symbol": symbol,
                "model": model_token,
                "side": side,
                "timeframe": timeframe,
                "epoch": int(chosen["epoch"]),
                "config_path": str(config_path),
                "model_path": str(staged_paths["model"]),
                "scaler_path": str(staged_paths["scaler"]),
                "features_path": str(staged_paths["features"]),
                "deployment_score": float(chosen["deployment_score"]),
                "net_pips": float(chosen["net_pips"]),
                "trades": int(chosen["trades"]),
                "win_rate": float(chosen["win_rate"]),
                "average_net_pips": float(chosen["average_net_pips"]),
                "max_drawdown_pips": float(chosen["max_drawdown_pips"]),
            })
        print(f"[SELECT] {symbol:6s} {model_token:22s} {side:4s} epoch={int(chosen['epoch']):03d} net={chosen['net_pips']:.1f} trades={chosen['trades']:.0f} avg={chosen['average_net_pips']:.3f} dd={chosen['max_drawdown_pips']:.1f} score={chosen['deployment_score']:.2f}")

    selections = sorted(selections, key=lambda x: (x["symbol"], x["model"], x["side"]))
    manifest_entries = sorted(manifest_entries, key=lambda x: (x["symbol"], x["model"], x["side"]))

    _csv_write(live_root / "live_model_selection_table.csv", selections)
    _csv_write(live_root / "all_epoch_deployment_scores.csv", sorted(all_epoch_rows, key=lambda x: (x["symbol"], x["model"], x["side"], -x["deployment_score"])))
    manifest = {
        "created_by": "select_live_trading_models.py",
        "live_root": str(live_root),
        "universal_config_path": str(universal_config_path) if universal_config_path is not None else None,
        "selection_logic": {
            "min_trades": args.min_trades,
            "min_net_pips": args.min_net_pips,
            "min_average_net_pips": args.min_average_net_pips,
            "min_win_rate": args.min_win_rate,
            "max_drawdown_pips": args.max_drawdown_pips,
            "max_drawdown_to_net_ratio": args.max_drawdown_to_net_ratio,
            "score": "net*w_net + avg*w_avg + win_rate*w_wr + min(trades,cap)*w_trades - drawdown*w_dd - instability_penalties",
        },
        "models": manifest_entries,
    }
    _write_json(live_root / "live_ensemble_manifest.json", manifest)
    _write_yaml(live_root / "live_ensemble_manifest.yaml", manifest)

    # Per-symbol ensemble configs for easy running/subsetting.
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for e in manifest_entries:
        by_symbol.setdefault(e["symbol"], []).append(e)
    for symbol, entries in sorted(by_symbol.items()):
        payload = {
            "symbol": symbol,
            "timeframe": entries[0].get("timeframe", "M5") if entries else "M5",
            "live_root": str(live_root),
            "universal_config_path": str(universal_config_path) if universal_config_path is not None else None,
            "models": entries,
        }
        _write_yaml(live_root / "configs" / f"{symbol}_live_ensemble.yaml", payload)

    if universal_config_path is not None:
        print(f"Copied universal config: {universal_config_path}")
    print(f"\nWrote selection table: {live_root / 'live_model_selection_table.csv'}")
    print(f"Wrote all-epoch scores: {live_root / 'all_epoch_deployment_scores.csv'}")
    print(f"Wrote manifest: {live_root / 'live_ensemble_manifest.json'}")
    print(f"Wrote per-symbol configs under: {live_root / 'configs'}")
    print(f"Staged {len(manifest_entries)} live-ready model(s).")
    missing_count = sum(1 for x in selections if x.get("missing_artifacts"))
    if missing_count:
        print(f"WARNING: {missing_count} selected model(s) had missing artifacts. Run this script on the machine where models/ exists, or use --project-root to point at it.")


if __name__ == "__main__":
    main()
