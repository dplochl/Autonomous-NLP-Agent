#!/usr/bin/env bash
# Agent_4 — one-shot bootstrap + launcher.
#
# After cloning the repo, run:
#   ./run.sh                  # 60-min agent run (sweep + final submission)
#   ./run.sh --time-budget-minutes 10   # shorter smoke run
#   ./run.sh dashboard        # serve the live dashboard at http://localhost:5050
#
# The script is idempotent: re-running skips work that's already done
# (venv, dependencies, model pull) and goes straight to the agent.

set -euo pipefail

# --------- helpers ---------------------------------------------------------
say()  { printf "\033[1;36m→\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m✓\033[0m %s\n" "$*"; }
fail() { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# --------- 0. sanity check that we are in the repo root --------------------
cd "$(dirname "$0")"
[ -f requirements.txt ] || fail "Run this script from the repo root (where requirements.txt lives)."

# --------- 1. Python 3.11+ -------------------------------------------------
command -v python3 >/dev/null || fail "python3 not found. Install Python 3.11+ first (e.g. 'brew install python@3.11')."
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ok "Python ${PY_VERSION}"

# --------- 2. virtualenv ---------------------------------------------------
if [ ! -d .venv ]; then
    say "Creating .venv ..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "venv activated  (.venv)"

# --------- 3. dependencies -------------------------------------------------
if ! python -c "import pandas, torch, transformers, flask, sklearn" 2>/dev/null; then
    say "Installing requirements (~2 GB, a few minutes on first run) ..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
fi
ok "dependencies ready"

# --------- 4. Ollama is running --------------------------------------------
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    fail "Ollama is not running on http://localhost:11434.
   Open another terminal and run:  ollama serve
   Then re-run this script."
fi
ok "Ollama is up"

# --------- 5. code-gen model is pulled -------------------------------------
MODEL="qwen2.5-coder:14b"
if ! curl -s http://localhost:11434/api/tags | grep -q "\"$MODEL\""; then
    say "Pulling $MODEL (~9 GB, one-time download) ..."
    command -v ollama >/dev/null || fail "ollama CLI not found. Install from https://ollama.com/download"
    ollama pull "$MODEL"
fi
ok "model $MODEL is available"

# --------- 6. data is present ----------------------------------------------
[ -f data/train.csv ] && [ -f data/test.csv ] || fail "data/train.csv and data/test.csv must exist.
   Download with:  kaggle competitions download -c nlp-getting-started -p data && unzip data/nlp-getting-started.zip -d data"
ok "data/train.csv and data/test.csv present"

# --------- 7. launch -------------------------------------------------------
MODE="${1:-agent}"
case "$MODE" in
    dashboard)
        shift || true
        say "Starting dashboard at http://localhost:5050   (Ctrl+C to stop)"
        exec python src/Agent_4/dashboard.py "$@"
        ;;
    agent|"")
        say "Starting Agent_4 (default budget 60 min, override with --time-budget-minutes N)"
        echo
        exec python src/Agent_4/agent.py "$@"
        ;;
    *)
        # Anything else: pass straight through to the agent.
        say "Starting Agent_4 with args: $*"
        echo
        exec python src/Agent_4/agent.py "$@"
        ;;
esac
