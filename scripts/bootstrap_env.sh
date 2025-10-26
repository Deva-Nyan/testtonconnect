#!/usr/bin/env bash
set -euo pipefail

if [ -d ".venv" ]; then
  echo "[bootstrap] Reusing existing .venv directory"
else
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
