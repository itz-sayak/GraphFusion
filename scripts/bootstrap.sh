#!/usr/bin/env bash
# One-shot local setup (macOS/Linux):  ./scripts/bootstrap.sh [--with-nyc-data]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
[ -f .env ] || cp .env.example .env
.venv/bin/python scripts/make_sample_data.py
if [ "${1:-}" = "--with-nyc-data" ]; then .venv/bin/python scripts/download_demo_data.py; fi
(cd frontend && npm install)
echo "Ready. Backend: .venv/bin/uvicorn backend.main:app --port 8000   UI: cd frontend && npm run dev"
