#!/usr/bin/env python3
"""Live/demo wrapper for universal direction-policy models.

This uses the existing ensemble live runner with a manifest produced by
replay_universal_direction_models.py. The manifest contains one entry per
universal-model x symbol, all pointing to the same universal model artifacts but
with the target symbol set for live data and order routing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Run universal direction-policy model ensemble in paper/demo/live mode.', add_help=True)
    ap.add_argument('--live-root', default='For Live Trading Universal')
    ap.add_argument('--manifest', default='logs/universal_replay/universal_live_ensemble_manifest.json')
    ap.add_argument('--universal-config', default='config/direction_settings_universal_models.yaml')
    # Parse only wrapper-specific values; forward all other args to the ensemble runner.
    args, passthrough = ap.parse_known_args()

    argv = [
        'live_direction_policy_ensemble.py',
        '--live-root', str(args.live_root),
        '--manifest', str(args.manifest),
        '--universal-config', str(args.universal_config),
    ] + passthrough
    sys.argv = argv
    from .live_direction_policy_ensemble import main as ensemble_main

    ensemble_main()


if __name__ == '__main__':
    main()
