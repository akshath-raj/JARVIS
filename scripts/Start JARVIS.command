#!/bin/bash
# Double-click this file in Finder to start JARVIS. No terminal typing needed.
# It activates the project's virtualenv and launches the voice console.
#
# First time only: Finder may block it. Right-click ▸ Open ▸ Open, or run once:
#   chmod +x "scripts/Start JARVIS.command"

set -euo pipefail

# Resolve the repo root from this script's own location, so it works no matter
# where the file is double-clicked from (Finder launches with cwd = /).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

clear
cat <<'BANNER'
   ┌─────────────────────────────────────────┐
   │   J A R V I S   —   starting up…         │
   └─────────────────────────────────────────┘
BANNER

# --- virtualenv -----------------------------------------------------------
if [[ ! -x ".venv/bin/python" ]]; then
  echo "✗  .venv not found at $REPO_ROOT/.venv"
  echo "   Set it up first:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  echo
  read -r -p "Press Return to close…" _
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- optional: Ollama (used by hybrid local actions / RAG embeddings) -----
# Cloud mode still works without it, but device tool-calling defaults to local.
if ! curl -sf --max-time 1 http://localhost:11434/api/tags >/dev/null 2>&1; then
  if command -v ollama >/dev/null 2>&1; then
    echo "•  Starting Ollama in the background…"
    (ollama serve >/dev/null 2>&1 &) || true
    sleep 1
  else
    echo "•  Ollama not detected — cloud brain still works; local actions/RAG will be limited."
  fi
fi

echo "•  Launching voice console.  Say “Hey Jarvis”.  Press Ctrl-C to stop."
echo

# --- run ------------------------------------------------------------------
set +e
python -m jarvis.agent console
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "JARVIS exited cleanly."
else
  echo "JARVIS exited (code $STATUS)."
fi
read -r -p "Press Return to close this window…" _
