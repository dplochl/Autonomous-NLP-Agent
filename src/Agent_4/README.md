# Agent_4

`Agent_4` is an autonomous experiment runner for the Kaggle Disaster Tweets binary text classification task. It uses local LLMs (via Ollama) to plan family selection, generate training scripts, and repair them when they break — all inside a fixed 1-hour CPU budget.

## Design at a glance

1. **Sweep planner LLM** decides which model family to try next, based on observed trial outcomes. Each decision = one trial of one family.
2. **Code-generation LLM** writes a full training script for the chosen family + spec, then the script runs in a sandboxed subprocess on CPU.
3. **Repair loop** asks the LLM for a structured JSON edit-plan whenever the script breaks (up to 4 patches per trial).
4. After the sweep window (45 min by default), the **final submission** step retrains the best-overall script on a 5 000-row sample and writes `id,target` predictions for the full test set. There is no separate "opt" phase — every trial in the budget is a sweep trial.

Every planner decision is logged to `runs/sweep_decisions.jsonl` — the audit trail for the agent's exploration behaviour.

## Families

- `bow` — TF-IDF + Logistic Regression
- `bow_advanced` — word + char n-grams + Logistic Regression
- `cnn` — 1D convolutional text classifier
- `lstm` — bidirectional LSTM
- `embedding_dl` — learned or GloVe embeddings + GRU/LSTM
- `roberta` — `roberta-base` fine-tuning
- `bertweet` — `vinai/bertweet-base` fine-tuning

## How a run goes

1. **Sweep phase (≤ 45 min by default).** The planner is asked at every step for the next decision. Each `try_family` triggers exactly one trial. Revisits seed `propose_next_spec` with the family's full prior history so the spec proposer can avoid repeating itself.
2. **Final submission (≤ 15 min by default).** Best-overall trial's frozen `best_train.py` is reloaded, a **hardcoded** submission tail is appended (orchestrator owns the inference step — no LLM involvement at this stage, no repair attempts), and the script is rerun with `AGENT_WRITE_SUBMISSION=1` on a 5 000-row training sample. Test predictions are written to `submissions/best_overall_submission.csv`. Optional Kaggle auto-submit if `AGENT4_AUTO_SUBMIT_KAGGLE=1`.

## Sweep planner actions

The planner returns one of three actions per decision:

- `try_family` — run one trial of one family
- `skip_family_permanently` — declare a family dead so it stops appearing in the eligible list
- `stop` — end the sweep early

The planner reasons purely from observed evidence (per-family state table). It has no prior beliefs about which family will perform best — it has to discover that through the trials it authorises.

## Safety nets (orchestrator-side)

The orchestrator enforces these defensively regardless of the planner's choice:

- **Hard per-family cap:** `MAX_ATTEMPTS_PER_FAMILY = 5`. Almost never binds.
- **Auto-skip after 2 code_gen_failed:** if the code-generation LLM cannot produce working code for a family twice in a row, the family drops out of eligibility.
- **Auto-skip after 2 degenerate_success:** if a trained model collapses to predicting one class (F1 < 0.4) twice in a row, the family drops out.
- **Time eligibility filter:** families whose estimated cost + buffer doesn't fit in remaining time are filtered out.
- **Fallback round-robin:** if the planner LLM can't be reached, the orchestrator falls back to deterministic family iteration.

## Trial outcome classes

Each trial is classified into one of:

- `success` — finished with `F1 ≥ 0.4`
- `degenerate_success` — finished, but `F1 < 0.4` (one-class predictor); counts towards the auto-skip rule
- `code_gen_failed` — the LLM never produced a runnable script, even after repairs
- `training_crash` — the script ran but raised an exception
- `timeout` — the script exceeded the sandbox timeout
- `no_metrics` — the script finished without printing the expected `METRICS` line

## Files

```
src/Agent_4/
├── agent.py                # main orchestrator + planner-driven sweep loop + final submission
├── sweep_planner.py        # per-family state, prompt, decision parsing, eligibility filter
├── prompts.py              # system prompts for code-gen, repair, spec, and sweep planner
├── submit_tails.py         # hardcoded final-submission tails (sparse / transformer / deep)
├── llm.py                  # Ollama client
├── repair.py               # surgical JSON-patch repair
├── sandbox.py              # CPU-only subprocess runner with dry-run + full-run timeouts
├── search.py               # intra-family spec proposer with stagnation detection
├── generate_spec.py        # initial spec generation
├── validate_spec.py        # type coercion + range clamping for specs
├── memory.py               # in-launch run history (write-only by default)
├── render_templates.py     # Jinja-ish {{key}} replacer
├── artifacts.py            # session/run directory helpers
├── kaggle_submit.py        # optional Kaggle CLI auto-submit
├── json_utils.py           # extract_json_object + pretty_json
├── families/               # one hook per architecture family
└── templates/              # one Jinja-ish prompt template per family
```

## Run

```bash
# Full LLM-driven 1-hour sweep + final submission
python3 src/Agent_4/agent.py

# Force one trial of a specific family (bypasses the sweep planner)
python3 src/Agent_4/agent.py --family bertweet

# Override the sweep planner model (defaults to qwen2.5-coder:14b)
python3 src/Agent_4/agent.py --sweep-planner-model gemma4:e4b

# Shorter run for a smoke test
python3 src/Agent_4/agent.py --time-budget-minutes 10
```

## Environment knobs

| Variable | Default | Purpose |
|---|---|---|
| `AGENT4_TOTAL_TIME_BUDGET_SECONDS` | `3600` | Overall wall-clock budget. |
| `AGENT4_SWEEP_DURATION_SECONDS` | `2700` (45 min) | Hard sweep cutoff before the final-submission step. |
| `AGENT4_FINAL_TRAIN_ROWS` | `5000` | Rows used by the final retrain step. |
| `AGENT4_SWEEP_SAMPLE_ROWS` | `2000` | Rows in the fixed sweep sample. |
| `AGENT4_VALIDATION_FRACTION` | `0.2` | Local val split. |
| `AGENT4_MAX_ATTEMPTS_PER_FAMILY` | `5` | Hard safety cap (rarely binds). |
| `AGENT4_RUN_START_BUFFER_SECONDS` | `120` | Cushion against budget overrun. |
| `AGENT4_SWEEP_PLANNER_MODEL` | `qwen2.5-coder:14b` | LLM for next-family decisions. |
| `DISASTER_AGENT_DATA_DIR` | `data` | Where `train.csv` and `test.csv` live. |
| `DISASTER_AGENT_MAX_REPAIRS` | `4` | Repair budget per trial during the sweep. The final submission step uses **zero** LLM repairs. |
| `AGENT4_AUTO_SUBMIT_KAGGLE` | unset | If `1`, upload `submissions/best_overall_submission.csv` to Kaggle. |

## Prerequisites

1. Python 3.11+ in a virtual environment
2. Ollama running locally on `http://localhost:11434`
3. The code-gen model pulled: `ollama pull qwen2.5-coder:14b`
4. `data/train.csv` and `data/test.csv` available (download with `kaggle competitions download -c nlp-getting-started -p data`)
5. Dependencies from the repo-level `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Outputs

- Per-trial artifacts: `src/Agent_4/runs/<family>_<ts>/run_NNN/` (spec.json, train.py, metrics.json, run.log, prompt.txt, repair_attempt_*.json)
- Per-family aggregate: `src/Agent_4/runs/<family>_<ts>/summary.json` + `best_train.py` + `best_metrics.json`
- Planner audit trail: `src/Agent_4/runs/sweep_decisions.jsonl` (plus `sweep_decision_<ts>_prompt.txt` and `_raw.txt` per decision)
- Cross-family summary: `src/Agent_4/runs/overall_best.json`
- Final-submission artifacts: `src/Agent_4/runs/final_submission_train.py` + `final_submission.log`
- Kaggle-ready CSV: `submissions/best_overall_submission.csv`
- Write-only in-launch log: `agent4_log.json` (at the repository root)
