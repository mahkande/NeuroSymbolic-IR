#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
REQ_FILE="requirements.txt"
if [[ "$MODE" == "lite" ]]; then
  REQ_FILE="requirements-lite.txt"
fi

echo "[setup] Starting NeuroGraph OS setup..."

if [[ ! -d ".venv" ]]; then
  echo "[setup] Creating virtual environment (.venv)"
  python -m venv .venv
fi

echo "[setup] Activating virtual environment"
source .venv/bin/activate

echo "[setup] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] Installing dependencies from ${REQ_FILE}"
python -m pip install -r "${REQ_FILE}"

echo "[setup] Downloading required NLTK data"
python - <<'PY'
import nltk
nltk.download("punkt_tab", quiet=True)
PY

echo ""
echo "[setup] Done."
echo "[setup] Run dashboard with:"
echo "        streamlit run ui/dashboard.py"
