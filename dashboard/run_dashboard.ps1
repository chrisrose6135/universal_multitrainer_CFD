$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:DASHBOARD_USER) { $env:DASHBOARD_USER = "admin" }
if (-not $env:DASHBOARD_PASSWORD) { $env:DASHBOARD_PASSWORD = "Groover6135.." }

if (-not (Test-Path ".venv-dashboard")) {
    py -m venv .venv-dashboard
}
.\.venv-dashboard\Scripts\Activate.ps1
pip install -r requirements-dashboard.txt
py live_dashboard.py --config dashboard_config.yaml
