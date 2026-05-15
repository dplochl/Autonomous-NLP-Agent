# APA Disaster Tweets — Autonomous LLM Research Agent

This is the course project for **Advanced Predictive Analytics 2025/2026**. The task is the Kaggle competition [`nlp-getting-started`](https://www.kaggle.com/competitions/nlp-getting-started) — binary classification of tweets as referring to a real disaster (label 1) or not (label 0).

Rather than solving the task by hand, we built an **autonomous research agent** that runs experiments on its own. The agent uses a local LLM (Ollama-hosted) to plan experiments, write their training code, repair the code when it breaks, and interpret the results — all inside a 1-hour CPU-only budget. Every trial it runs is recorded as `hypothesis → spec → F1 → conclusion`, and the next trial reads that history before deciding what to try next.

The full design rationale, experiment log, results, and reflections are in the project report. This README focuses on **what the agent is** and **how to run it**.

---

## What the agent actually does

A single launch of the agent works like a scientific research session, run by an LLM:

1. **Memory load.** The agent reads `logs/agent4_short_term_memory.json` — a rolling window of the last 20 trials from previous launches. Each prior trial includes the spec that was run, the F1 it achieved, the hypothesis the LLM wrote at the time, and the analyst's conclusion afterwards.

2. **Pick a family.** A small LLM call (the *sweep planner*) chooses which model family to try next. The seven families are BoW, BoW_advanced, LSTM, CNN, EmbeddingDL, RoBERTa, and BERTweet. The planner reasons over the prior-trial history rather than following a fixed schedule.

3. **Propose an experiment.** A second LLM call (the *spec proposer*) writes a JSON object: a one-sentence hypothesis explaining what it's testing, the list of hyperparameter keys it intends to change vs. the prior best, and the values for those keys. The hypothesis must reference concrete prior evidence (e.g. *"lower lr to 1e-5 to test if BERTweet was undertrained at the prior best F1=0.7742"*).

4. **Constrain the spec.** The orchestrator constrains the actual experiment to **only** the keys the LLM explicitly declared. Any silent change the LLM tried to make is reverted. If the LLM declared fewer than 2 key changes, the orchestrator adds more so there's always meaningful movement. Any orchestrator-added keys get an honest annotation appended to the hypothesis.

5. **Generate the training code.** A third LLM call writes a complete Python training script from the validated spec.

6. **Run it in a sandbox.** A CPU subprocess executes the script. A 60-second dry run on a tiny data slice first, then the real 1000-second run on the sweep sample. If the script crashes, a fourth LLM call (the *repair LLM*) returns a small JSON edit-plan — up to four repair attempts before the trial is abandoned.

7. **Analyse the result.** A fifth LLM call (the *analyst*) reads the metrics + spec + hypothesis and writes a structured CONCLUSION / WHAT WORKED / WHAT FAILED / NEXT MOVE.

8. **Persist + repeat.** The trial — including its hypothesis, spec, F1, and conclusion — gets saved to the rolling 20-trial memory and to per-trial artifact files under `runs/agent_4/current/`. The next iteration of the loop reads everything that just happened.

After the 45-minute sweep window ends, the orchestrator picks the highest-F1 trial across all families, reloads its training script, appends a hardcoded inference tail (no LLM at this step), retrains on a 5 000-row sample, and writes `submissions/best_overall_submission.csv` for Kaggle.

---

## Why this design

Every layer in the agent fixes a specific failure mode we observed:

- LLMs propose values that crash → **validator** clamps numeric values to per-family safe ranges
- LLMs write hypotheses that don't match their specs → **`changed_keys` constraint** reverts silent changes
- LLMs anchor too hard on prior best, only tweaking one knob → **temperature 0.5** for spec calls + **2-key minimum floor**
- LLMs forget prior runs and re-propose known-failing specs → **cross-launch signature veto**
- Generated training code has bugs → **structured JSON-patch repair loop**, not full regeneration
- A single launch is short → **20-trial cross-launch memory** so each launch builds on the previous

Read [`src/Agent_4/README.md`](src/Agent_4/README.md) for the full technical reference (every mechanism with file:line citations), or the project report for the design rationale and the experiment log analysis.

---

## Quick start

After cloning, from the repo root:

```bash
./run.sh
```

That script handles the venv, dependencies, Ollama check (must be running on `localhost:11434`), and the qwen2.5-coder:14b model pull (~9 GB, one-time), then launches the agent for the default 60-minute budget.

Useful variants:

```bash
./run.sh --time-budget-minutes 10      # quick smoke run
./run.sh --family bertweet             # one trial of a specific family
./run.sh dashboard                     # serve the live dashboard at http://localhost:5050
```

Manual flow without the bootstrap script:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve &                           # in another terminal
ollama pull qwen2.5-coder:14b            # one-time, ~9 GB
python3 src/Agent_4/agent.py             # default 60-minute budget
```

While the agent runs, watch the terminal for tagged lines like `[Memory]`, `[Hypothesis]`, `[Conclusion]`, and `[Diversity]` — each tells you which stage of the loop the agent is in.

---

## Prerequisites

You need three things installed once:

1. **Python 3.11+**
2. **Ollama** running locally on `http://localhost:11434` (install from <https://ollama.com/download> or `brew install ollama`, then `ollama serve` in another terminal)
3. **Kaggle data** at `data/train.csv` and `data/test.csv` — included on this branch; replace with your own copy if needed

`./run.sh` will check each of these on first launch and tell you what's missing.

---

## Repository layout

```text
.
├── README.md                     ← this file (high-level walkthrough)
├── requirements.txt
├── run.sh                        ← bootstrap script: venv + Ollama check + launch
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   └── Agent_4/                  ← all agent source (technical detail: src/Agent_4/README.md)
├── runs/
│   └── agent_4/
│       ├── current/              ← latest committed snapshot of a live run
│       ├── before_fix/           ← archived sessions from earlier code versions
│       └── v2_fixed/ … v16_pre_2key_floor/
├── logs/
│   ├── agent4_log.json           ← in-launch write-only audit log (gitignored)
│   └── agent4_short_term_memory.json  ← 20-trial rolling cross-launch memory
└── submissions/                  ← Kaggle CSVs (filled by the agent)
```

A fresh clone of this branch already includes 20 prior trials in the cross-launch memory, so the first launch is not starting blind — the planner and spec proposer both see the prior runs' specs, F1s, hypotheses, and analyst conclusions.

---

## Where to go next

- **For the design rationale, experiment log analysis, course-content reflections, and limitations** → the project report
- **For the technical reference** (every mechanism, every file:line, the family hooks, the prompts, the safety nets) → [`src/Agent_4/README.md`](src/Agent_4/README.md)
- **For visualisation of past runs** → `./run.sh dashboard` then open <http://localhost:5050>
