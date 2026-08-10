#!/usr/bin/env bash
# Set up the ISOLATED browser-agent environment.
#
# browser-use pulls in versions of openai/pydantic/anyio/starlette that conflict
# with the main JARVIS stack (LiveKit + LangGraph), so it lives in its own venv
# and JARVIS invokes it out-of-process. Run this once:
#
#     bash scripts/setup_browser_agent.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${JARVIS_BROWSER_VENV:-.venv-browser}"
PY="${PYTHON:-python3}"

echo "Creating isolated venv at $VENV ..."
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
echo "Installing browser-use (isolated) ..."
"$VENV/bin/pip" install -q -r requirements-browser.txt

# browser-use 0.13 talks to Chrome directly over CDP (cdp-use) and drives your
# real system Chrome — no Playwright / bundled Chromium needed.

echo
echo "✅ Browser agent ready at $VENV"
echo "   The main .venv is untouched. JARVIS calls this venv as a subprocess."
echo "   It drives your system Google Chrome; make sure Chrome is closed when a"
echo "   browser task runs (it uses your real profile & saved logins)."
