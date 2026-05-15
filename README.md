# APA Disaster Tweets — Autonomous LLM Research Agent

## To run everything

```bash
./run.sh
```

That's it. The script handles the virtualenv, installs dependencies, checks Ollama is up, pulls the `qwen2.5-coder:14b` model on first run (~9 GB, one-time), verifies the dataset is in place, and launches the agent for the default 60-minute budget.

Useful variants:

```bash
./run.sh --time-budget-minutes 10      # quick smoke run
./run.sh --family bertweet             # one trial of a specific family
./run.sh dashboard                     # serve the live dashboard at http://localhost:5050
```

---

## What this is

The Kaggle competition [`nlp-getting-started`](https://www.kaggle.com/competitions/nlp-getting-started) asks for binary classification of tweets — disaster (label 1) vs not (label 0). Rather than solving it by hand, this project is an **autonomous research agent** that uses a local LLM (Ollama-hosted) to plan experiments, write training code, repair the code when it breaks, and interpret the results — all inside a 1-hour CPU-only budget. Every trial is recorded as `hypothesis → spec → F1 → conclusion`, and the next trial reads that history before deciding what to try next.

![Agent_4 Architecture](src/Agent_4/docs/architecture_v4.png)

Full design rationale, experiment log, results, and reflections are in the project report. This README is the practical reference — what the agent is, how it works, and how to run it.

---

## What one trial looks like

A single launch loops through trials. Each trial runs through **five LLM round-trips** plus **nine deterministic guard rails**:

1. **Memory load.** The agent reads `logs/agent4_short_term_memory.json` — a rolling window of the last 20 trials from previous launches, with their specs, F1s, hypotheses, and analyst conclusions.

2. **Pick a family.** The *sweep planner* LLM (temperature 0.2) picks which of the seven model families to try next, reasoning over the prior-trial history.

3. **Propose an experiment.** The *spec proposer* LLM (temperature 0.5) writes a JSON object with three fields: `hypothesis` (one sentence under 25 words referencing prior evidence), `changed_keys` (the list of tunable keys it intends to change), and the spec values.

4. **Constrain.** The orchestrator constrains the experiment to **only** the keys the LLM declared in `changed_keys`. Silent changes are reverted to the anchor (default spec on first trial, prior best on revisits). A 2-key minimum floor applies — if the LLM declared fewer, the orchestrator backfills additional keys and tags them with `[orchestrator-added: ...]` in the hypothesis so the record stays honest.

5. **Cross-launch veto.** If the final spec signature exactly matches a prior-launch trial of the same family, the orchestrator mutates further so we don't burn a trial on a known result.

6. **Generate the code.** The *code-generator* LLM (temperature 0.2) writes a complete Python training script from the validated spec.

7. **Run it.** The script executes in a CPU subprocess sandbox: 60-second dry run on a tiny data slice first, then the real 1000-second run on the sweep sample. If anything crashes, the *repair* LLM (temperature 0.2) returns a structured JSON edit-plan — up to 4 repair attempts.

8. **Analyse.** The *analyst* LLM (temperature 0.2) reads the metrics + spec + hypothesis and writes a structured CONCLUSION / WHAT WORKED / WHAT FAILED / NEXT MOVE.

9. **Persist + repeat.** The trial — hypothesis, spec, F1, conclusion — gets saved to the rolling 20-trial memory and to per-trial artifact files. The next iteration reads everything that just happened.

After the 45-minute sweep window ends, the orchestrator picks the highest-F1 trial across all families, reloads its training script, appends a **hardcoded** inference tail (no LLM at this step), retrains on a 5 000-row sample, and writes `submissions/best_overall_submission.csv` for Kaggle.

---

## The nine mechanism layers

| # | Layer | What it does | Where |
|---|---|---|---|
| 1 | **Cross-launch short-term memory** | Persists the last 20 trials across launches with their hypothesis, F1, and analyst conclusion. The next launch's spec proposer and planner both see this history. | `short_term_memory.py` |
| 2 | **`changed_keys` hypothesis-as-source-of-truth** | The LLM must declare which tunable keys it changes; silent changes are reverted to the anchor. Eliminates "I claim 1 change but actually changed 5" hallucination. | `generate_spec.py`, `search.py` |
| 3 | **2-key minimum floor** | Final spec must differ from the anchor on at least 2 tunable keys. If the LLM declared fewer, the orchestrator backfills. | `search.py:_ensure_phase_mutation` |
| 4 | **`[orchestrator-added: …]` honest annotation** | When the orchestrator backfills, the keys it added get appended to the hypothesis so the research record stays truthful. | `generate_spec.py`, `search.py` |
| 5 | **Cross-launch veto** | If the final spec signature matches a prior-launch trial of the same family, the orchestrator mutates further. | `agent.py:execute_family` |
| 6 | **Per-call temperature split** | Spec proposers use 0.5 (creative but bounded by the validator); everything else (code-gen, repair, planner, analyst) uses 0.2 (deterministic). | `llm.py` |
| 7 | **Plateau detection** | A family with 5 consecutive successes within F1 ±0.005 of its best is dropped from eligibility. | `sweep_planner.py` |
| 8 | **Spec validator** | Clamps numeric values to per-family safe ranges, strips fixed/non-tunable keys, coerces types. | `validate_spec.py` |
| 9 | **Hardcoded submission tail** | At final-submission time, the orchestrator appends a deterministic Python tail (not LLM-written) that loads the model, predicts the test set, and writes the CSV. Wrapped in `try/except` with idempotent markers. | `submit_tails.py` |

---

## The five LLM roles

All five run through `OllamaClient` in `llm.py`, against the same model (`qwen2.5-coder:14b` by default). Each call passes the appropriate `temperature` and a different system prompt.

| Role | System prompt | Temperature | What it returns |
|---|---|---|---|
| Sweep planner | `SWEEP_PLANNER_SYSTEM` | 0.2 | JSON: `try_family` / `skip_family_permanently` / `stop` |
| Spec proposer (initial) | `SPEC_SYSTEM` | **0.5** | JSON: hypothesis + changed_keys + spec values |
| Spec proposer (revisit) | `SEARCH_SYSTEM` | **0.5** | same schema, anchored to prior best |
| Code generator | `FULL_SYSTEM` | 0.2 | one ```python``` block — a full training script |
| Surgical repair | `PATCH_REPAIR_SYSTEM` | 0.2 | JSON edit-plan (`replace` / `insert_before` / `insert_after`) |
| Analyst | hardcoded | 0.2 | structured CONCLUSION / WHAT WORKED / WHAT FAILED / NEXT MOVE |

The 0.5 spec temperature is the only call that needs creativity (proposing genuinely new hyperparameter combinations); everything else needs determinism.

---

## The seven model families

| Family | Approach | Tunable keys |
|---|---|---|
| BoW | sklearn TF-IDF + LogReg | `max_features, ngram_max, min_df, logreg_c, threshold_*` |
| BoW_advanced | word + char TF-IDF + LogReg | `word_max_features, char_max_features, word_ngram_*, char_ngram_*, min_df, logreg_c, threshold_*` |
| LSTM | PyTorch bidirectional LSTM | `max_vocab, max_len, embedding_dim, hidden_dim, num_layers, dropout, batch_size, epochs, learning_rate` |
| CNN | PyTorch 1D Conv | `max_vocab, max_len, embedding_dim, channels, dropout, batch_size, epochs, learning_rate` |
| EmbeddingDL | learned/GloVe embedding + GRU | `embedding_source, max_vocab, max_len, embedding_dim, hidden_dim, dropout, batch_size, epochs, learning_rate` |
| RoBERTa | `roberta-base` fine-tune | `max_len, train_batch_size, eval_batch_size, learning_rate, weight_decay, num_epochs` |
| BERTweet | `vinai/bertweet-base` fine-tune | same as RoBERTa |

Per-family files in `src/Agent_4/families/`. RoBERTa and BERTweet share `experiment_hf_classifier.py` as a base.

---

## Live terminal output

While the agent runs, you see tagged lines telling you which stage of the loop it's in:

```
[Memory] Loaded 20 prior-launch trial(s) from logs/agent4_short_term_memory.json
[Memory] By family: BoW_advanced:8, CNN:2, ...  |  best so far BERTweet F1=0.7742
[Sweep Planner] action=try_family family=bertweet reason=Untried family.
[LLM] Request started | model=qwen2.5-coder:14b | temp=0.5 | timeout=1000s | prompt='Plan one reliable BERTweet experiment...'
[Hypothesis] Reducing lr 1.5e-5→1e-5 and max_len 128→144 to test if undertrained at F1=0.7742.
[Diversity] Cross-launch duplicate detected — mutating spec.       ← only when veto fires
[EXECUTE] Running experiment...
  [Sandbox] Dry run passed. Starting metrics-only run...
[Result] run 1/1 | success=True | metrics={'f1': 0.7812, ...}
[Conclusion] Hypothesis confirmed. F1 improved from 0.7742 to 0.7812.
[Best] family=BERTweet | run=1 | metrics={'f1': 0.7812, ...}
```

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

## Repository layout

```text
.
├── README.md                     ← this file
├── requirements.txt
├── run.sh                        ← bootstrap script: venv + Ollama check + launch
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   └── Agent_4/                  ← all agent source
│       ├── agent.py              # main orchestrator (sweep loop + final submission)
│       ├── sweep_planner.py      # planner LLM, plateau, eligibility filter
│       ├── prompts.py            # all system prompts
│       ├── generate_spec.py      # first-trial spec proposer
│       ├── search.py             # revisit spec proposer + _ensure_phase_mutation
│       ├── short_term_memory.py  # cross-launch 20-trial rolling memory
│       ├── llm.py                # Ollama client (per-call temperature)
│       ├── repair.py             # surgical JSON-patch repair
│       ├── sandbox.py            # CPU subprocess runner
│       ├── submit_tails.py       # hardcoded final-submission tail
│       ├── validate_spec.py      # type coercion + range clamping
│       ├── families/             # per-architecture hooks
│       └── templates/            # one Jinja-ish prompt template per family
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
| `DISASTER_AGENT_LLM_TIMEOUT` | `1000` | Per-call LLM HTTP timeout (seconds) |
| `AGENT4_AUTO_SUBMIT_KAGGLE` | unset | If `1`, upload `submissions/best_overall_submission.csv` to Kaggle |

---

## Outputs

| Artifact | Location |
|---|---|
| Per-trial: spec, code, metrics, log, prompts, repair attempts | `src/Agent_4/runs/<family>_<ts>/run_NNN/` (live working dir, gitignored) |
| Per-family aggregate | `<session>/summary.json` + `best_train.py` + `best_metrics.json` |
| Planner audit trail | `src/Agent_4/runs/sweep_decisions.jsonl` |
| Cross-family summary | `src/Agent_4/runs/overall_best.json` |
| Final-submission artifacts | `src/Agent_4/runs/final_submission_train.py` + `final_submission.log` |
| Kaggle-ready CSV | `submissions/best_overall_submission.csv` |
| Cross-launch memory | `logs/agent4_short_term_memory.json` |
| Snapshot committed to repo | `runs/agent_4/current/` |

`src/Agent_4/runs/` is the live working dir (gitignored). Snapshots are committed to `runs/agent_4/current/`.

---

## Prerequisites (handled by `./run.sh`)

If you skip `./run.sh` and want to set things up manually:

1. **Python 3.11+**
2. **Ollama** running locally on `http://localhost:11434` (install from <https://ollama.com/download> or `brew install ollama`, then `ollama serve` in another terminal)
3. The code-gen model pulled: `ollama pull qwen2.5-coder:14b`
4. `data/train.csv` and `data/test.csv` available (already on this branch)
5. Dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dashboard

```bash
./run.sh dashboard
```

Serves a Flask UI at <http://localhost:5050> with per-trial cards, hypothesis text, `[orchestrator-added: ...]` annotations, F1 timelines, and family-level summary tables.

---

## Where to go next

- **Design rationale, experiment log analysis, course-content reflections, limitations** → the project report
- **Per-version architecture diagrams** → `src/Agent_4/docs/architecture_v*.png`
- **Visual playback of past runs** → `./run.sh dashboard`
