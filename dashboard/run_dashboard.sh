#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set these once in your shell profile or systemd service in production.
: "${DASHBOARD_USER:=admin}"
: "${DASHBOARD_PASSWORD:=change-this-password}"
export DASHBOARD_USER DASHBOARD_PASSWORD

python3 -m venv .venv-dashboard
source .venv-dashboard/bin/activate
pip install -r requirements-dashboard.txt
python live_dashboard.py --config dashboard_config.yaml
