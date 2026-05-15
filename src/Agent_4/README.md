# Agent_4

`Agent_4` is an autonomous research agent for the Kaggle "NLP with Disaster Tweets" competition. It uses local LLMs (via Ollama) to plan experiments, generate training scripts, repair them when they break, and analyse the results — all inside a 1-hour CPU-only budget.

The agent runs as a single Python process that loops through trials: each trial is one LLM-proposed spec + LLM-generated training script + sandboxed run + LLM-analysed conclusion. The conclusion feeds back into the next trial's prompt as evidence the LLM must reason over.

---

## Design at a glance

1. **Sweep planner LLM** decides which model family to try next, based on observed trial outcomes.
2. **Spec proposer LLM** writes a hypothesis + a list of which keys to change + the spec values. Output is constrained to "hypothesis-as-source-of-truth" — silent spec changes are reverted.
3. **Code-generator LLM** writes a full Python training script from the validated spec.
4. **Repair LLM** patches the script with structured JSON edits if the dry run or full run fails (up to 4 attempts).
5. **Analyst LLM** writes a structured CONCLUSION / WHAT WORKED / WHAT FAILED / NEXT MOVE after each successful trial.

After the sweep window (45 min by default), the orchestrator picks the best-overall script, appends a **hardcoded** submission tail (no LLM at this step), retrains on a 5 000-row sample, and writes the Kaggle-ready CSV.

---

## The nine mechanism layers

The agent is not just an LLM in a loop — it's an LLM wrapped in nine deterministic guard rails. Each layer addresses a real LLM failure mode observed during development.

| # | Layer | What it does | Where |
|---|---|---|---|
| 1 | **Cross-launch short-term memory** | Persists the last 20 trials across launches with their hypothesis, F1, and analyst conclusion. The next launch's spec proposer and planner both see this history. | `short_term_memory.py` |
| 2 | **`changed_keys` hypothesis-as-source-of-truth** | The LLM must declare which tunable keys it changes; silent changes are reverted to the anchor. Eliminates LLM "I claim 1 change but actually changed 5" hallucination. | `generate_spec.py:70-89`, `search.py:531-547` |
| 3 | **2-key minimum floor** | Final spec must differ from the anchor on at least 2 tunable keys. If LLM declared fewer, the orchestrator backfills. | `search.py:_ensure_phase_mutation:345` |
| 4 | **`[orchestrator-added: …]` honest annotation** | When the orchestrator must backfill, the keys it added get appended to the hypothesis so the research record stays truthful. | `generate_spec.py:137-139`, `search.py:617-619` |
| 5 | **Cross-launch veto** | If the final spec signature exactly matches a prior-launch trial of the same family, the orchestrator mutates further so the agent doesn't burn a trial on a known result. | `agent.py:584-678` |
| 6 | **Per-call temperature split** | Spec proposers use `temperature=0.5` (creative but bounded by validator); code generation, repair, planner, and analyst use `temperature=0.2` (deterministic). | `llm.py:39`, `generate_spec.py:55`, `search.py:515` |
| 7 | **Plateau detection** | A family with 5 consecutive successes within F1 ±0.005 of its best is flagged "plateaued" and dropped from eligibility. | `sweep_planner.py:104-117` |
| 8 | **Spec validator** | Clamps every numeric value to per-family safe ranges, strips fixed/non-tunable keys, coerces types. No spec leaves the validator with out-of-range values. | `validate_spec.py:30-62` |
| 9 | **Hardcoded submission tail** | At final-submission time, the orchestrator appends a deterministic Python tail (not LLM-written) that loads the model, predicts the test set, and writes the CSV. Wrapped in `try/except` with idempotent markers. | `submit_tails.py:347-356` |

---

## The five LLM roles

All five run through `OllamaClient` in `llm.py`, against the same model (`qwen2.5-coder:14b` by default). Each call passes the appropriate `temperature` and a different system prompt.

| Role | Where | System prompt | Temperature |
|---|---|---|---|
| Sweep planner | `sweep_planner.py:452` | `SWEEP_PLANNER_SYSTEM` | 0.2 |
| Spec proposer (initial) | `generate_spec.py:55` | `SPEC_SYSTEM` | **0.5** |
| Spec proposer (revisit) | `search.py:515` | `SEARCH_SYSTEM` | **0.5** |
| Code generator | `agent.py:709` | `FULL_SYSTEM` | 0.2 |
| Surgical repair | `repair.py:149` | `PATCH_REPAIR_SYSTEM` | 0.2 |
| Analyst | `llm.py:78` | `"You are a concise ML research analyst."` | 0.2 |

The 0.5 spec temperature is the only call that needs creativity (proposing genuinely new hyperparameter combinations); everything else needs determinism (correct Python syntax, valid JSON edit-plans, structured analyst reports).

---

## Spec proposer schema (the `changed_keys` contract)

Every spec-proposer LLM response is a single JSON object with three top-level fields:

```json
{
  "hypothesis": "Lowering lr=1e-5 (below the prior 1.5e-5 best) to test if BERTweet was undertrained at F1=0.7742.",
  "changed_keys": ["learning_rate"],
  "learning_rate": 1e-5,
  "max_len": 128,
  ... rest of the spec keys at top level ...
}
```

The orchestrator then:

1. **Validates the spec** — every numeric value clamped to family-safe ranges; fixed keys stripped.
2. **Constrains to `changed_keys`** — starts from the anchor (default spec for initial trial, prior best for revisits), copies LLM values **only for keys in `changed_keys`**, and reverts everything else to the anchor.
3. **Enforces the 2-key minimum** — if the validated, constrained spec differs from the anchor on fewer than 2 tunable keys, the orchestrator's mutator (`_ensure_phase_mutation`) adds keys until the floor is met. Added keys get an `[orchestrator-added: k1=v1, k2=v2]` suffix on the hypothesis.
4. **Cross-launch vetoes** — if the final spec signature matches any prior-launch trial of this family, the orchestrator mutates further and prefixes the hypothesis with `[orchestrator-generated]`.

End result: the spec.json that runs **only differs from the anchor on keys named in `changed_keys` plus any `[orchestrator-added: ...]` keys.** The hypothesis is a truthful description of the experiment.

---

## Families

| Family | Module identifier | Tunable keys |
|---|---|---|
| BoW | sklearn TF-IDF + LogReg | `max_features, ngram_max, min_df, logreg_c, threshold_*` |
| BoW_advanced | word + char TF-IDF + LogReg | `word_max_features, char_max_features, word_ngram_max, char_ngram_min, char_ngram_max, min_df, logreg_c, threshold_*` |
| CNN | PyTorch 1D Conv | `max_vocab, max_len, embedding_dim, channels, dropout, batch_size, epochs, learning_rate` |
| LSTM | PyTorch bidirectional LSTM | `max_vocab, max_len, embedding_dim, hidden_dim, num_layers, dropout, batch_size, epochs, learning_rate` |
| EmbeddingDL | learned/GloVe embedding + GRU | `embedding_source, max_vocab, max_len, embedding_dim, hidden_dim, dropout, batch_size, epochs, learning_rate` |
| RoBERTa | `roberta-base` fine-tune | `max_len, train_batch_size, eval_batch_size, learning_rate, weight_decay, num_epochs` |
| BERTweet | `vinai/bertweet-base` fine-tune | same as RoBERTa |

Per-family files in `families/`. RoBERTa and BERTweet share `experiment_hf_classifier.py` as a base.

---

## How one trial flows

1. **Planner picks family.** Returns a `SweepDecision`: `try_family` / `skip_family_permanently` / `stop`. Audit trail in `runs/sweep_decisions.jsonl`.
2. **Spec proposer LLM call** (`temperature=0.5`). Either `generate_initial_spec` (first attempt for this family in this launch) or `propose_next_spec` (revisit, anchored to prior best). Output: `hypothesis`, `changed_keys`, spec values.
3. **Constraint engine.** Validator clamps; `changed_keys` filter reverts silent changes; 2-key floor enforced; `[orchestrator-added: ...]` annotation appended if backfill happened.
4. **Cross-launch veto.** Signature check against the 20-trial memory; if collision, mutate further and synthesise a `[orchestrator-generated]` hypothesis.
5. **Code-gen LLM call** (`temperature=0.2`). Writes the full Python training script.
6. **Sandbox dry run** (60 s timeout, tiny data head). On failure, **repair LLM** is asked for a JSON edit-plan (up to 4 attempts).
7. **Sandbox full run** (1000 s timeout, 2 000-row sample for the sweep phase). Metrics parsed from the final `METRICS:` line in stdout.
8. **Analyst LLM call** (`temperature=0.2`). Writes structured CONCLUSION / WHAT WORKED / WHAT FAILED / NEXT MOVE.
9. **Persist.** Trial record saved to `runs/<session>/run_NNN/` (spec.json, train.py, metrics.json, run.log, hypothesis.txt, spec_prompt.txt, spec_response.txt, etc.). `ShortTermMemory.add_trial` saves family + spec + F1 + hypothesis + analysis for the next launch to see.

---

## Live terminal output

While a trial runs, you see these tagged lines:

```
[Memory] Loaded 20 prior-launch trial(s) from logs/agent4_short_term_memory.json
[Memory] By family: BoW_advanced:8, CNN:2, ...  |  best so far BERTweet F1=0.7742
[Sweep Planner] action=try_family family=bertweet reason=Untried family, new data point needed.
[LLM] Request started | model=qwen2.5-coder:14b | temp=0.5 | timeout=1000s | prompt='Plan one reliable BERTweet experiment...'
[Hypothesis] Reducing lr 1.5e-5→1e-5 and max_len 128→144 to test if undertrained at F1=0.7742.
[Diversity] Cross-launch duplicate detected — mutating spec.   ← only when veto fires
[EXECUTE] Running experiment...
  [Sandbox] Dry run passed. Starting metrics-only run...
[Result] run 1/1 | success=True | metrics={'f1': 0.7812, ...}
[Conclusion] Hypothesis confirmed. F1 improved from 0.7742 to 0.7812.
[Sweep] BERTweet produced a successful run; moving to the next family.
[Best] family=BERTweet | run=1 | metrics={'f1': 0.7812, ...}
```

---

## Final submission

After the sweep window closes:

1. The orchestrator picks the highest-F1 successful trial across all families.
2. Its `best_train.py` is reloaded, the **hardcoded** submission tail is appended (try/except wrap around the LLM-written section + a category-specific finaliser for sparse / deep / transformer).
3. Script reruns with `AGENT_WRITE_SUBMISSION=1` on a 5 000-row training sample. **Zero LLM repair attempts at this stage** — the orchestrator owns inference.
4. `submissions/best_overall_submission.csv` is written with `id,target` predictions for the full 3 263-row test set.
5. Optional Kaggle CLI upload if `AGENT4_AUTO_SUBMIT_KAGGLE=1`.

---

## Trial outcome classes

`sweep_planner.classify_trial_outcome` maps every sandbox result to one of:

| Outcome | Meaning |
|---|---|
| `success` | Finished with F1 ≥ 0.4 |
| `degenerate_success` | Finished but F1 < 0.4 (one-class collapse); 2 in a row drops the family from eligibility |
| `code_gen_failed` | LLM couldn't produce a runnable script even after 4 repair patches |
| `training_crash` | Script ran but raised an exception |
| `timeout` | Exceeded the 1000 s sandbox timeout |
| `no_metrics` | Script finished without printing a `METRICS:` line |

---

## Safety nets (orchestrator-side eligibility filter)

The planner LLM can propose any family, but the orchestrator filters out:

- Hard per-family attempt cap: `MAX_ATTEMPTS_PER_FAMILY = 5`
- **2 consecutive `code_gen_failed`** with no success → family dropped
- **2 consecutive `degenerate_success`** with no success → family dropped
- Families whose estimated cost exceeds the remaining time
- Plateaued families (5 consecutive successes within ±0.005 F1 of best)
- Fallback round-robin if the planner LLM is unreachable

---

## File map

```
src/Agent_4/
├── agent.py                # Main orchestrator. Sweep loop, execute_family, final submission.
├── sweep_planner.py        # Per-family state, planner prompt, plateau/eligibility logic.
├── prompts.py              # All system prompts (SPEC_SYSTEM, SEARCH_SYSTEM, etc.)
├── generate_spec.py        # generate_initial_spec — first-trial spec proposer.
├── search.py               # propose_next_spec — revisit spec proposer + _ensure_phase_mutation.
├── short_term_memory.py    # Cross-launch 20-trial rolling memory.
├── llm.py                  # Ollama client with per-call temperature.
├── repair.py               # Surgical JSON-patch repair loop.
├── sandbox.py              # CPU subprocess runner with dry/full run timeouts.
├── submit_tails.py         # Hardcoded submission tail + idempotent try/except wrap.
├── validate_spec.py        # Type coercion + range clamping.
├── memory.py               # In-launch write-only audit log (agent4_log.json).
├── render_templates.py     # {{key}} replacer for family templates.
├── artifacts.py            # Per-trial directory helpers.
├── kaggle_submit.py        # Optional Kaggle CLI auto-submit.
├── json_utils.py           # JSON object extraction from LLM responses.
├── families/               # Per-architecture hooks (BoW, BoW_advanced, CNN, LSTM, EmbeddingDL, RoBERTa, BERTweet) + autofix_utils + hf_classifier shared base.
└── templates/              # One Jinja-ish prompt template per family.
```

---

## Run

```bash
# Full LLM-driven 1-hour sweep + final submission
python3 src/Agent_4/agent.py

# Force one trial of a specific family (bypasses the sweep planner)
python3 src/Agent_4/agent.py --family bertweet

# Override the sweep planner model
python3 src/Agent_4/agent.py --sweep-planner-model gemma4:e4b

# Shorter run for a smoke test
python3 src/Agent_4/agent.py --time-budget-minutes 10

# Disable the in-launch write-only log (still keeps the runs/ artifacts)
python3 src/Agent_4/agent.py --fresh
```

---

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
| `DISASTER_AGENT_MAX_REPAIRS` | `4` | Repair budget per trial during the sweep. Zero at final-submission time. |
| `DISASTER_AGENT_LLM_TIMEOUT` | `1000` | Per-call LLM HTTP timeout (seconds). |
| `AGENT4_AUTO_SUBMIT_KAGGLE` | unset | If `1`, upload `submissions/best_overall_submission.csv` to Kaggle. |

---

## Prerequisites

1. Python 3.11+ in a virtual environment
2. Ollama running locally on `http://localhost:11434`
3. The code-gen model pulled: `ollama pull qwen2.5-coder:14b`
4. `data/train.csv` and `data/test.csv` available (download with `kaggle competitions download -c nlp-getting-started -p data`)
5. Dependencies from the repo-level `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Outputs

| Artifact | Location |
|---|---|
| Per-trial: spec, code, metrics, log, prompts, repair attempts | `src/Agent_4/runs/<family>_<ts>/run_NNN/` |
| Per-family aggregate | `src/Agent_4/runs/<family>_<ts>/summary.json` + `best_train.py` + `best_metrics.json` |
| Planner audit trail | `src/Agent_4/runs/sweep_decisions.jsonl` |
| Cross-family summary | `src/Agent_4/runs/overall_best.json` |
| Final-submission artifacts | `src/Agent_4/runs/final_submission_train.py` + `final_submission.log` |
| Kaggle-ready CSV | `submissions/best_overall_submission.csv` |
| Cross-launch memory | `logs/agent4_short_term_memory.json` |
| Snapshot of run committed to repo | `runs/agent_4/current/` |

`src/Agent_4/runs/` is the live working dir (gitignored). Snapshots are committed to `runs/agent_4/current/`.

---

## Dashboard

`python3 src/Agent_4/dashboard.py` serves a Flask UI at `http://localhost:5050` with per-trial cards, hypothesis text, `[orchestrator-added: ...]` annotations, F1 timelines, and family-level summary tables.

---

## Audit trail

Everything the agent does is reconstructable from these committed files:

- `runs/agent_4/current/<session>/run_NNN/` — full per-trial record
- `runs/agent_4/current/sweep_decisions.jsonl` — every planner decision with reasoning
- `runs/agent_4/current/sweep_decision_<ts>_prompt.txt` — exact prompt the planner saw
- `runs/agent_4/current/sweep_decision_<ts>_raw.txt` — exact planner response
- `logs/agent4_short_term_memory.json` — the cross-launch 20-trial rolling window
- `agent4_log.json` (repo root) — in-launch write-only audit log (also gitignored; same data also lives in `runs/agent_4/current/*/summary.json`)
