# APA Disaster Tweets Agent

Autonomous research agent for the Kaggle competition `nlp-getting-started`:
https://www.kaggle.com/competitions/nlp-getting-started

The active implementation in this repository is `src/Agent_3/`. It uses a local Ollama-hosted LLM to:
- propose experiment specs
- generate runnable training code
- dry-run and execute experiments
- repair failing code
- analyze results
- iterate across multiple model families
- attempt one final Kaggle-style submission rerun from the best run

## Current Entry Point

Run the agent from:

```bash
python3 src/Agent_3/agent.py
```

There are no active `agent_vs.py`, `Agent_V2`, dashboard, or legacy entrypoints in the current repository flow.

## Repository Layout

```text
apa-disaster-tweets-agent/
├── data/
│   ├── train.csv
│   └── test.csv
├── submissions/
├── src/
│   └── Agent_3/
│       ├── agent.py
│       ├── llm.py
│       ├── sandbox.py
│       ├── search.py
│       ├── memory.py
│       ├── repair.py
│       ├── families/
│       ├── templates/
│       └── runs/
├── requirements.txt
└── README.md
```

Important output locations:
- run artifacts: `src/Agent_3/runs/`
- public final submission copy after a successful final rerun: `submissions/best_overall_submission.csv`
- invocation log: `agent3_log.json`

## Supported Model Families

`Agent_3` can sweep these families:
- `bow`
- `bow_advanced`
- `cnn`
- `embedding_dl`
- `lstm`
- `roberta`
- `bertweet`

## Prerequisites

1. Python 3.11+ with a virtual environment
2. Ollama running locally on `http://localhost:11434`
3. At least one pulled local model for code generation
4. Kaggle Disaster Tweets data available at `data/train.csv` and `data/test.csv`

Recommended Ollama setup:

```bash
ollama serve
ollama pull qwen2.5-coder:14b
```

The default agent model is `qwen2.5-coder:14b`. You can override it with `--model`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Setup

The agent expects:
- `data/train.csv`
- `data/test.csv`

If they are not already present, download them with Kaggle:

```bash
mkdir -p data
kaggle competitions download -c nlp-getting-started -p data
unzip data/nlp-getting-started.zip -d data
```

You can also point the agent to a different dataset directory:

```bash
export DISASTER_AGENT_DATA_DIR="/absolute/path/to/data"
```

## Running the Agent

Run all families with the default model:

```bash
python3 src/Agent_3/agent.py --time-budget-minutes 60
```

Run a single family:

```bash
python3 src/Agent_3/agent.py --family bertweet --max-runs 2 --time-budget-minutes 20
```

Use a different local Ollama model:

```bash
python3 src/Agent_3/agent.py --model gemma4:e4b
```

Run without writing to `agent3_log.json`:

```bash
python3 src/Agent_3/agent.py --fresh
```

Disable the winner-optimization phase:

```bash
python3 src/Agent_3/agent.py --no-winner-optimization
```

## What the Agent Does

For each family, the current flow is:
1. generate an experiment spec
2. validate and clamp the spec
3. render a family-specific prompt
4. ask the local LLM for a full Python training script
5. dry-run the script
6. execute the full run
7. attempt surgical repairs if the script fails
8. log metrics, stdout/stderr, analysis, and artifacts
9. rank families by best sweep F1
10. optimize the top architecture candidates
11. rerun the best overall model to produce a final submission file

## Configuration

Common runtime controls:

- `DISASTER_AGENT_DATA_DIR`: override the dataset directory
- `AGENT3_MAX_RUNS`: default max runs per family when `--max-runs` is not passed
- `AGENT3_TOTAL_TIME_BUDGET_SECONDS`: total wall-clock budget
- `AGENT3_SWEEP_BUDGET_FRACTION`: fraction reserved for sweep before winner optimization
- `AGENT3_SWEEP_SAMPLE_ROWS`: labeled rows used during sweep
- `AGENT3_FINAL_TRAIN_ROWS`: labeled rows used for the final best-model rerun
- `DISASTER_AGENT_LLM_TIMEOUT`: Ollama request timeout in seconds
- `DISASTER_AGENT_MAX_REPAIRS`: max repair attempts per generated script

## Optional Kaggle Auto-Submit

The agent can optionally submit the final CSV through the Kaggle CLI.

Requirements:
- Kaggle CLI installed in the environment
- Kaggle credentials configured via `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`

Enable it with:

```bash
export AGENT3_AUTO_SUBMIT_KAGGLE=1
python3 src/Agent_3/agent.py --time-budget-minutes 60
```

Optional submission controls:
- `AGENT3_KAGGLE_COMPETITION`
- `AGENT3_KAGGLE_MESSAGE`
- `AGENT3_KAGGLE_POLL_SECONDS`
- `AGENT3_KAGGLE_TIMEOUT_SECONDS`
- `KAGGLE_CLI_PATH`

## Notes

- The local LLM is used for planning, code generation, repair, and analysis.
- Hugging Face model families may download pretrained checkpoints on first use unless already cached locally.
- `src/Agent_3/runs/` contains generated code and experiment artifacts from previous runs.
- The generated scripts are part of the agent workflow; the hand-written source of truth is under `src/Agent_3/`.
