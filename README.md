# APA Disaster Tweets Agent — Agent_4

Autonomous research agent for the Kaggle competition `nlp-getting-started`:
https://www.kaggle.com/competitions/nlp-getting-started

The active and only implementation in this branch is `src/Agent_4/`. It uses two
roles of a local Ollama-hosted LLM to:

- pick the next model family to try (sweep planner)
- generate runnable training code for that family + spec
- dry-run and execute the generated script in a CPU sandbox
- repair failing code with surgical JSON edit-plans
- log every planner decision and trial outcome
- retrain the best overall script on a larger sample and write a Kaggle-ready submission

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   └── Agent_4/                ← all agent source
├── runs/
│   ├── agent_3/                ← experiment logs from the earlier Agent_3 baseline (analysis only)
│   └── agent_4/
│       ├── current/            ← latest live run snapshot
│       ├── before_fix/         ← archived sessions, one folder per code version
│       ├── full_v1_with_opt/
│       ├── v2_fixed/
│       ├── v3_partial/
│       ├── v4_stuck_on_lstm/
│       ├── v5_qwen_planner/
│       ├── v6_broken_submission/
│       ├── v7_pandas_bug/
│       ├── v8_nested_writebranch/
│       ├── v9_partial/
│       ├── v10_old_code/
│       ├── v11_test_truncated/
│       ├── v12_old_repair/
│       ├── v13_missing_classifier/
│       ├── v14_old_2k/
│       └── v15_bow_validation/
├── logs/
│   ├── agent3_log.json         ← write-only in-launch log from Agent_3 days
│   └── agent4_log.json         ← write-only in-launch log from Agent_4 launches
└── submissions/                ← final Kaggle CSVs (filled by the agent)
```

The earlier Agent_3 source code is intentionally absent on this branch. Only its
runs/ history is retained so the experiment log is still available for analysis.

## Quick start — one command

After cloning, from the repo root:

```bash
./run.sh
```

That script creates the virtualenv on first run, installs the dependencies,
checks Ollama is up, makes sure `qwen2.5-coder:14b` is pulled, verifies the
dataset is in place, and then launches the agent for the default 60-minute
budget. Re-running it skips work already done and goes straight to the agent.

Useful variants:

```bash
./run.sh --time-budget-minutes 10   # quick smoke run
./run.sh --family bow               # one trial of a specific family
./run.sh dashboard                  # serve the live dashboard at http://localhost:5050
```

## Prerequisites

The script handles steps 1, 3 and the dependency install for you. You only need
to install these once on the machine:

1. **Python 3.11+** (`brew install python@3.11` on macOS)
2. **Ollama** running locally on `http://localhost:11434`
   (`brew install ollama` then `ollama serve` in another terminal, or install
   from <https://ollama.com/download>)
3. **Kaggle Disaster Tweets data** at `data/train.csv` and `data/test.csv`
   (already included in this branch — overwrite with your own copy if needed).

If you prefer the manual flow without the bootstrap script:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve &                                  # in another terminal, or already running
ollama pull qwen2.5-coder:14b                   # one-time, ~9 GB
python3 src/Agent_4/agent.py                    # default 60-minute budget
```

## Running the agent

The bootstrap script above (`./run.sh`) is the easiest way. If you want to call
the Python entrypoint directly, the equivalent commands are:

```bash
source .venv/bin/activate
python3 src/Agent_4/agent.py                # full LLM-driven 60-minute run
```

Force one trial of a specific family (bypasses the sweep planner):

```bash
python3 src/Agent_4/agent.py --family bertweet
```

Override the sweep planner LLM:

```bash
python3 src/Agent_4/agent.py --sweep-planner-model gemma4:e4b
```

Shorter run for a smoke test:

```bash
python3 src/Agent_4/agent.py --time-budget-minutes 10
```

Disable persistence of the in-launch log:

```bash
python3 src/Agent_4/agent.py --fresh
```

## Architecture in one screen

1. The **sweep planner LLM** picks the next family to try (or stops the sweep)
2. The orchestrator filters out families that fail eligibility (recurring
   code-gen failures, recurring degenerate F1, can't fit in remaining time)
3. The **code-generation LLM** writes a full training script for the chosen
   family and spec
4. The script is dry-run in a CPU sandbox (60 s timeout), then run to
   completion (1000 s timeout)
5. On failure the LLM is asked for a small JSON edit-plan (one of `replace`,
   `insert_before`, `insert_after`); up to 4 repair attempts per trial
6. The trial outcome is recorded (`success`, `degenerate_success`,
   `code_gen_failed`, `training_crash`, `timeout`, `no_metrics`)
7. Per-trial artifacts (spec, code, metrics, log, prompts, repair attempts) are
   written to `src/Agent_4/runs/<family>_<ts>/run_NNN/`

After the sweep window ends:

1. The orchestrator loads the best-overall trial's frozen `best_train.py`
2. A **hardcoded** submission tail is appended (orchestrator owns the inference
   step — no LLM, no repairs)
3. The script is rerun on a 5 000-row training sample and predicts the full
   test set
4. `submissions/best_overall_submission.csv` is written

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

## Optional Kaggle auto-submit

```bash
export AGENT4_AUTO_SUBMIT_KAGGLE=1
python3 src/Agent_4/agent.py
```

Requires `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`).

## Notes on the run logs included on this branch

- `runs/agent_3/` contains 148 historical sessions from the earlier baseline
  implementation. The corresponding source code is not in this branch.
- `runs/agent_4/before_fix/`, `runs/agent_4/full_v1_with_opt/` and the
  `v2_fixed/` … `v15_bow_validation/` folders are snapshots of Agent_4 runs at
  earlier code versions. Useful for tracking how the agent's behaviour changed
  across iterations.
- `runs/agent_4/current/` is the latest live run captured at branch-push time.
  May continue to grow if the live agent is still going when this snapshot was
  taken.
- `logs/agent3_log.json` and `logs/agent4_log.json` are the write-only
  in-launch logs.
