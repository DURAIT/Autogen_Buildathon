#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment not found. Create .venv and install requirements first." >&2
  exit 1
fi

exec .venv/bin/python mcp_server.py
