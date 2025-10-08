$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    Write-Error "[demo_chat] Missing .venv. Run scripts/bootstrap_env.ps1 first."
    exit 1
}

$activateScript = Join-Path ".venv" "Scripts/Activate.ps1"
if (-not (Test-Path $activateScript)) {
    throw "Не найден Activate.ps1 по пути $activateScript"
}

. $activateScript

& python scripts/run_chat.py @args
