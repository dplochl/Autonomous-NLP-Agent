# APA Disaster Tweets — Agent_4

Autonomous research agent for the Kaggle competition `nlp-getting-started`:
https://www.kaggle.com/competitions/nlp-getting-started

The active and only implementation in this branch is `src/Agent_4/`. The agent uses five distinct roles of a local Ollama-hosted LLM:

- **Sweep planner** — picks the next model family to try, family by family
- **Spec proposer** — writes a hypothesis + the exact keys to change for the next experiment
- **Code generator** — produces a full Python training script from the validated spec
- **Repair** — patches the script with surgical JSON edit-plans if it breaks
- **Analyst** — writes a structured conclusion after every successful trial

Around those five LLM roles sit **nine deterministic guard rails**: cross-launch memory, hypothesis-as-source-of-truth via `changed_keys`, a 2-key minimum diversity floor, `[orchestrator-added]` honesty annotations, cross-launch signature veto, per-call temperature split, plateau detection, spec validator, and a hardcoded final-submission tail. Full technical detail in [`src/Agent_4/README.md`](src/Agent_4/README.md).

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── run.sh                        # bootstrap script: venv + Ollama check + launch
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   └── Agent_4/                  ← all agent source (see src/Agent_4/README.md)
├── runs/
│   ├── agent_3/                  ← historical sessions from the earlier Agent_3 baseline (analysis only)
│   └── agent_4/
│       ├── current/              ← latest committed snapshot of a live run
│       ├── before_fix/           ← archived sessions from earlier code versions
│       ├── full_v1_with_opt/
│       └── v2_fixed/ … v16_pre_2key_floor/
├── logs/
│   ├── agent3_log.json           ← write-only in-launch log from Agent_3 days
│   ├── agent4_log.json           ← write-only in-launch log from Agent_4 launches (gitignored)
│   └── agent4_short_term_memory.json  ← 20-trial rolling cross-launch memory
└── submissions/                  ← final Kaggle CSVs (filled by the agent)
```

Earlier Agent_3 source code is intentionally absent on this branch. Only its runs history is retained so the experiment log is still available for analysis.

## Quick start — one command

After cloning, from the repo root:

```bash
./run.sh
```

That script creates the virtualenv on first run, installs dependencies, checks Ollama is up, makes sure `qwen2.5-coder:14b` is pulled, verifies the dataset is in place, and launches the agent for the default 60-minute budget.

Useful variants:

```bash
./run.sh --time-budget-minutes 10   # quick smoke run
./run.sh --family bow               # one trial of a specific family
./run.sh dashboard                  # serve the live dashboard at http://localhost:5050
```

## Prerequisites

The script handles the dependency install for you. You only need:

1. **Python 3.11+** (`brew install python@3.11` on macOS)
2. **Ollama** running locally on `http://localhost:11434`
   (`brew install ollama` then `ollama serve` in another terminal, or install from <https://ollama.com/download>)
3. **Kaggle Disaster Tweets data** at `data/train.csv` and `data/test.csv` (included on this branch).

Manual flow without the bootstrap script:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve &                                  # another terminal
ollama pull qwen2.5-coder:14b                   # one-time, ~9 GB
python3 src/Agent_4/agent.py                    # default 60-minute budget
```

## Architecture in one screen

1. **Sweep planner LLM** picks the next family to try (or stops the sweep).
2. Orchestrator filters out families that fail eligibility (recurring code-gen failures, recurring degenerate F1, plateaued, can't fit in remaining time).
3. **Spec proposer LLM** writes a JSON object: `hypothesis` + `changed_keys` + spec values. Temperature 0.5.
4. Orchestrator constrains the final spec to **only the keys the LLM declared in `changed_keys`** (silent changes reverted) and enforces a **2-key minimum** floor.
5. Cross-launch veto fires if the final spec signature matches any prior-launch trial.
6. **Code-generator LLM** writes a full training script for the chosen family and spec. Temperature 0.2.
7. Script is dry-run in a CPU sandbox (60 s timeout), then full-run (1000 s timeout, 2 000-row sample for the sweep phase).
8. On failure the **repair LLM** is asked for a small JSON edit-plan; up to 4 repair attempts.
9. **Analyst LLM** writes a structured conclusion. The trial outcome (`success`, `degenerate_success`, `code_gen_failed`, `training_crash`, `timeout`, `no_metrics`) is recorded.
10. Per-trial artifacts are written to `src/Agent_4/runs/<family>_<ts>/run_NNN/`.
11. The trial is saved to the cross-launch 20-trial rolling memory so the next launch can read it.

After the sweep window ends:

1. Orchestrator picks the best-overall trial's frozen `best_train.py`.
2. A **hardcoded** submission tail is appended (orchestrator owns the inference step — no LLM, no repairs).
3. Script reruns on a 5 000-row training sample and predicts the full test set.
4. `submissions/best_overall_submission.csv` is written.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AGENT4_TOTAL_TIME_BUDGET_SECONDS` | `3600` | Overall wall-clock budget |
| `AGENT4_SWEEP_DURATION_SECONDS` | `2700` (45 min) | Hard sweep cutoff |
| `AGENT4_FINAL_TRAIN_ROWS` | `5000` | Rows used by the final retrain step |
| `AGENT4_SWEEP_SAMPLE_ROWS` | `2000` | Rows in the fixed sweep sample |
| `AGENT4_VALIDATION_FRACTION` | `0.2` | Local val split |
| `AGENT4_MAX_ATTEMPTS_PER_FAMILY` | `5` | Hard safety cap |
| `AGENT4_SWEEP_PLANNER_MODEL` | `qwen2.5-coder:14b` | LLM for next-family decisions |
| `DISASTER_AGENT_DATA_DIR` | `data` | Where `train.csv` and `test.csv` live |
| `DISASTER_AGENT_MAX_REPAIRS` | `4` | Repair budget per trial (zero at final-submission time) |
| `AGENT4_AUTO_SUBMIT_KAGGLE` | unset | If `1`, upload `submissions/best_overall_submission.csv` to Kaggle |

## Optional Kaggle auto-submit

```bash
export AGENT4_AUTO_SUBMIT_KAGGLE=1
python3 src/Agent_4/agent.py
```

Requires `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`).

## Notes on the run logs included on this branch

- `runs/agent_3/` — 148 historical sessions from the earlier baseline implementation. Corresponding source code is not in this branch.
- `runs/agent_4/before_fix/`, `runs/agent_4/full_v1_with_opt/` and the `v2_fixed/` … `v16_pre_2key_floor/` folders are snapshots of Agent_4 at earlier code versions. Useful for tracking how the agent's behaviour changed across iterations.
- `runs/agent_4/current/` — latest live run committed at branch-push time.
- `logs/agent3_log.json` — write-only in-launch log from Agent_3 days.
- `logs/agent4_log.json` — write-only in-launch log from Agent_4 launches (gitignored — same data also in each session's `summary.json`).
- `logs/agent4_short_term_memory.json` — the 20-trial rolling cross-launch memory that the agent reads at every startup.
