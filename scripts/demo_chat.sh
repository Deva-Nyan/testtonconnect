#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  echo "[demo_chat] Missing .venv. Run scripts/bootstrap_env.sh first." >&2
  exit 1
fi

source .venv/bin/activate
python scripts/run_chat.py "$@"
