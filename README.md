# APA Disaster Tweets Agent

Autonomous research agent for the Kaggle competition `nlp-getting-started`:
https://www.kaggle.com/competitions/nlp-getting-started

The active implementation in this repository is `src/Agent_4/`. It uses two roles of a local Ollama-hosted LLM to:
- pick the next model family to try (sweep planner)
- generate runnable training code for that family + spec
- dry-run and execute the generated script in a CPU sandbox
- repair failing code with surgical JSON edit-plans
- log metrics, planner decisions, and artifacts for every trial
- retrain the best overall script on a larger sample and write a Kaggle-ready submission

## Current entry point

Run the agent from the repository root:

```bash
python3 src/Agent_4/agent.py
```

The default budget is 60 minutes (45 min sweep + 15 min final retrain + submission).

## Repository layout

```text
apa-disaster-tweets-agent/
├── data/
│   ├── train.csv
│   └── test.csv
├── submissions/
│   └── best_overall_submission.csv
├── src/
│   ├── Agent_3/            # earlier baseline implementation (retained for reference)
│   │   └── runs/           # logs from prior Agent_3 launches (for analysis)
│   └── Agent_4/            # active implementation
│       ├── agent.py
│       ├── sweep_planner.py
│       ├── llm.py
│       ├── sandbox.py
│       ├── search.py
│       ├── memory.py
│       ├── repair.py
│       ├── submit_tails.py
│       ├── families/
│       ├── templates/
│       └── runs/           # logs from Agent_4 launches
├── requirements.txt
└── README.md
```

Important output locations:
- per-trial artifacts: `src/Agent_4/runs/<family>_<ts>/run_NNN/`
- planner audit trail: `src/Agent_4/runs/sweep_decisions.jsonl`
- final submission CSV: `submissions/best_overall_submission.csv`
- in-launch write-only log: `agent4_log.json`

## Supported model families

`Agent_4` can sweep across these families:

- `bow` — TF-IDF + Logistic Regression
- `bow_advanced` — word + char n-grams + Logistic Regression
- `cnn` — 1D convolutional text classifier
- `lstm` — bidirectional LSTM
- `embedding_dl` — learned or GloVe embeddings + GRU/LSTM
- `roberta` — `roberta-base` fine-tuning
- `bertweet` — `vinai/bertweet-base` fine-tuning

## Prerequisites

1. Python 3.11+ in a virtual environment
2. Ollama running locally on `http://localhost:11434`
3. At least the code-gen model pulled:

```bash
ollama serve
ollama pull qwen2.5-coder:14b
```

4. Kaggle Disaster Tweets data available at `data/train.csv` and `data/test.csv`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset setup

The agent expects:
- `data/train.csv`
- `data/test.csv`

If they are not already present, download them with Kaggle:

```bash
mkdir -p data
kaggle competitions download -c nlp-getting-started -p data
unzip data/nlp-getting-started.zip -d data
```

Point the agent at a different dataset directory if needed:

```bash
export DISASTER_AGENT_DATA_DIR="/absolute/path/to/data"
```

## Running the agent

Full LLM-driven 1-hour sweep + final submission:

```bash
python3 src/Agent_4/agent.py
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

## What the agent does, per trial

1. The **sweep planner LLM** picks the next family to try (or stops the sweep)
2. The orchestrator filters out families that fail eligibility (recurring code-gen failures, recurring degenerate F1, can't fit in remaining time)
3. The **code-generation LLM** writes a full training script for the chosen family and spec
4. The script is dry-run in a CPU sandbox (60 s timeout), then run to completion (1000 s timeout)
5. On failure, the LLM is asked for a small JSON edit-plan (one of `replace`, `insert_before`, `insert_after`); up to 4 repair attempts per trial
6. The trial outcome is recorded (`success`, `degenerate_success`, `code_gen_failed`, `training_crash`, `timeout`, `no_metrics`)
7. Per-trial artifacts (spec, code, metrics, log, prompts, repair attempts) are written to `runs/<family>_<ts>/run_NNN/`

After the sweep window ends:

1. The orchestrator loads the best-overall trial's frozen `best_train.py`
2. A **hardcoded** submission tail is appended (orchestrator owns the inference step — no LLM, no repairs)
3. The script is rerun on a 5 000-row training sample and predicts the full test set
4. `submissions/best_overall_submission.csv` is written

## Configuration

Common runtime controls:

| Variable | Default | Purpose |
|---|---|---|
| `AGENT4_TOTAL_TIME_BUDGET_SECONDS` | `3600` | Overall wall-clock budget |
| `AGENT4_SWEEP_DURATION_SECONDS` | `2700` (45 min) | Hard sweep cutoff |
| `AGENT4_FINAL_TRAIN_ROWS` | `5000` | Rows used by the final retrain step |
| `AGENT4_SWEEP_SAMPLE_ROWS` | `2000` | Rows in the fixed sweep sample |
| `AGENT4_VALIDATION_FRACTION` | `0.2` | Local val split |
| `AGENT4_MAX_ATTEMPTS_PER_FAMILY` | `5` | Hard safety cap (rarely binds) |
| `AGENT4_SWEEP_PLANNER_MODEL` | `qwen2.5-coder:14b` | LLM for next-family decisions |
| `DISASTER_AGENT_DATA_DIR` | `data` | Where `train.csv` and `test.csv` live |
| `DISASTER_AGENT_MAX_REPAIRS` | `4` | Repair budget per trial (zero at final-submission time) |

## Optional Kaggle auto-submit

The agent can optionally submit the final CSV through the Kaggle CLI.

Requirements:
- Kaggle CLI installed in the environment
- Kaggle credentials configured via `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`

Enable it:

```bash
export AGENT4_AUTO_SUBMIT_KAGGLE=1
python3 src/Agent_4/agent.py
```

Optional submission controls:
- `AGENT4_KAGGLE_COMPETITION`
- `AGENT4_KAGGLE_MESSAGE`
- `AGENT4_KAGGLE_POLL_SECONDS`
- `AGENT4_KAGGLE_TIMEOUT_SECONDS`
- `KAGGLE_CLI_PATH`

## Notes

- The local LLM does planning, code generation, and repair. Final-submission inference is handled by a hardcoded tail in the orchestrator and uses **no** LLM.
- Hugging Face model families may download pretrained checkpoints on first use unless already cached locally.
- Generated scripts under `src/Agent_4/runs/` are part of the agent workflow; the hand-written source of truth is under `src/Agent_4/` itself.
- `src/Agent_3/runs/` is retained on this branch so the experiment history from the earlier baseline implementation remains available for analysis.
