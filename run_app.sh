#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment not found. Create .venv and install requirements first." >&2
  exit 1
fi

.venv/bin/python -m scripts.check_mcp

exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
