#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  echo "[run_tests] Missing .venv. Run scripts/bootstrap_env.sh first." >&2
  exit 1
fi

source .venv/bin/activate
python -m compileall anime_chatbot scripts
