param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

if (Test-Path ".venv") {
    Write-Host "[bootstrap] Reusing existing .venv directory"
} else {
    & $Python -m venv .venv
}

$activateScript = Join-Path ".venv" "Scripts/Activate.ps1"
if (-not (Test-Path $activateScript)) {
    throw "Не найден Activate.ps1 по пути $activateScript"
}

. $activateScript

& $Python -m pip install --upgrade pip
& $Python -m pip install -e .
