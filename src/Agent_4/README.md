# Agent_4

`Agent_4` is the LLM-driven evolution of `Agent_3`. The orchestration code, family hooks, repair contract, sandbox, and prompt templates are inherited from `Agent_3`. The thing that changes is who decides what to try next during the sweep.

## What's different from Agent_3

1. **Sweep order is decided by an LLM planner, not a hardcoded list.**
   Each iteration the planner reads the per-family state (attempts, best F1, last outcome, repair attempts, wall time so far) and picks one of three actions:
   - `try_family` — run one trial of one family.
   - `skip_family_permanently` — declare a family dead so it stops appearing in the eligible list.
   - `stop` — end the sweep and roll over to the opt phase.
2. **One trial per planner decision.** No per-family attempt cap baked into the sweep loop. The planner can revisit a family that already succeeded (and `propose_next_spec` evolves the prior winning spec), or retry one that failed if the failure mode looks fixable. A hard safety cap of 5 attempts per family exists only to stop a degenerate loop.
3. **Sweep ends at a fixed wall-clock boundary (default 40 min)** instead of as a fraction of the total budget. Whatever sweep is doing at 40 minutes, the agent rolls over to the opt phase.
4. **Final submission trains on a 2{,}000-row sample**, not the full 7{,}613 rows. Retraining on the full data on CPU often pushed past the 1-hour budget; the 2k retrain reliably finishes in the remaining time after sweep + opt.
5. **A new `sweep_decisions.jsonl` artifact** in `runs/` logs every planner decision (timestamp, eligible families, action, family chosen, one-sentence reason). Per-decision prompt and raw response are written next to it.

Everything else — family hooks, templates, the surgical-edit repair contract, the opt-phase planner, the Kaggle submission pipeline — is unchanged from `Agent_3`.

## Families (unchanged from Agent_3)

- `bow`, `bow_advanced`, `cnn`, `embedding_dl`, `lstm`, `roberta`, `bertweet`

## How one launch goes

1. **Sweep phase (≤ 40 min).** While inside the sweep window, the LLM planner is asked at every step for the next decision. Each `try_family` decision triggers one call to `execute_family(..., max_runs=1)`. On revisits the orchestrator seeds the call with the prior best trial for that family, so `propose_next_spec` evolves the previous winning spec.
2. **Opt phase.** When sweep ends, the opt-phase planner (gemma4:e4b) picks the highest-F1 family across all sweep trials and chooses 2–3 hyperparameters to tune. Up to 20 tuning trials, time-shared until the budget runs out.
3. **Final submission.** Best-overall trial across sweep + opt is rerun on a 2{,}000-row sample with `AGENT_FINAL_SUBMISSION=1`, then test predictions are written. Optional Kaggle auto-submit if `AGENT4_AUTO_SUBMIT_KAGGLE=1`.

## Files

```
src/Agent_4/
├── agent.py                # main orchestrator with LLM-driven sweep loop
├── sweep_planner.py        # NEW: per-family state, prompt, decision parsing
├── prompts.py              # adds SWEEP_PLANNER_SYSTEM
├── llm.py                  # unchanged Ollama client
├── repair.py               # unchanged surgical JSON-patch repair
├── sandbox.py              # unchanged CPU-only subprocess runner
├── search.py               # unchanged intra-family spec proposer
├── families/               # unchanged family hooks
├── templates/              # unchanged Jinja-ish prompt templates
└── runs/                   # per-launch artifacts (sweep_decisions.jsonl lives here)
```

## Run

```bash
# Full LLM-driven 1-hour sweep + opt + final
python3 src/Agent_4/agent.py

# Force one trial of a specific family (bypasses the sweep planner)
python3 src/Agent_4/agent.py --family bertweet

# Override the sweep planner model (defaults to gemma4:e4b)
python3 src/Agent_4/agent.py --sweep-planner-model gemma4:e4b

# Shorter run for a smoke test
python3 src/Agent_4/agent.py --time-budget-minutes 10
```

## Environment knobs

| Variable | Default | Purpose |
|---|---|---|
| `AGENT4_TOTAL_TIME_BUDGET_SECONDS` | `3600` | Overall wall-clock budget. |
| `AGENT4_SWEEP_DURATION_SECONDS` | `2400` (40 min) | Hard sweep cutoff. |
| `AGENT4_FINAL_TRAIN_ROWS` | `2000` | Rows used by the final retrain step. |
| `AGENT4_SWEEP_SAMPLE_ROWS` | `2000` | Rows in the fixed sweep+opt sample. |
| `AGENT4_VALIDATION_FRACTION` | `0.2` | Local val split. |
| `AGENT4_MAX_ATTEMPTS_PER_FAMILY` | `5` | Hard safety cap (rarely binds). |
| `AGENT4_WINNER_OPTIMIZATION_MAX_RUNS` | `20` | Opt-phase per-family cap. |
| `AGENT4_RUN_START_BUFFER_SECONDS` | `120` | Cushion against budget overrun. |
| `AGENT4_SWEEP_PLANNER_MODEL` | `gemma4:e4b` | LLM for next-family decisions. |
| `AGENT4_OPT_PLANNER_MODEL` | `gemma4:e4b` | LLM for opt-phase family + parameter focus. |
| `DISASTER_AGENT_DATA_DIR` | `data` | Where train.csv and test.csv live. |
| `DISASTER_AGENT_MAX_REPAIRS` | `4` | Repair budget per trial. |

## Why this design

Agent_3's sweep iterates families in a hardcoded list. That makes the "agent" part of the agent narrow — it chooses hyperparameters within a family, but not which family to try. Agent_4 widens that so the LLM has to reason at every step about cost-benefit:

- Untried families: information value vs. wall-clock cost.
- Stagnant successes: rarely improve on revisit; consider untried or stop.
- `code_gen_failed` families: the code-gen LLM is stuck — retrying usually fails the same way. Prefer skip.
- `training_crash` / `timeout` / `no_metrics` failures: often fixable by `propose_next_spec`. Worth one more shot if time allows.

These bullets are written verbatim into the planner's system prompt so the LLM has the cost-benefit logic explicit. Every decision is logged to `runs/sweep_decisions.jsonl` — that file is the audit trail for the agent's exploration behaviour.

## Falls-back when the planner LLM is unavailable

If the sweep-planner model can't be loaded or returns nonsense, the orchestrator falls back to a deterministic round-robin over untried families, matching `Agent_3`'s old behaviour. So Agent_4 is a strict superset of Agent_3's reliability.

## Notes

- Ollama must be running on `localhost:11434` and the requested models must be pulled.
- Data is read from `data/train.csv` and `data/test.csv` unless `DISASTER_AGENT_DATA_DIR` is set.
- Kaggle auto-submit reuses Agent_3's `kaggle_submit.py`. Set `AGENT3_AUTO_SUBMIT_KAGGLE=1` (the variable name kept for compatibility with the existing CLI helper).
