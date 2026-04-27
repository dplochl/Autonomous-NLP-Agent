# Agent_3

`Agent_3` is a separate prompt-first experiment runner.

Goal:
- keep model logic mostly in prompts
- keep only minimal orchestration hard-coded
- let the LLM write the actual training script
- run one family prompt repeatedly with adaptive parameter search

Families:
- `bow`
- `bow_advanced`
- `cnn`
- `lstm`
- `transformer`
- `roberta`
- `bertweet`

How it works:
1. `generate_spec.py` asks the LLM for a JSON experiment spec.
2. `validate_spec.py` clamps that spec to safe ranges.
3. `render_templates.py` builds the family prompt from a prompt template.
4. `llm.py` asks Ollama for one full Python training script.
5. `sandbox.py` runs a dry run, then a full run.
6. If code fails, `repair.py` asks the LLM for a surgical patch plan instead of rewriting the whole file.
7. `search.py` asks the LLM for the next spec based on prior run outcomes.
8. `memory.py` stores the current invocation history in `agent3_log.json`; old logs are not loaded for decisions.
9. `artifacts.py` writes run artifacts under `src/Agent_3/runs/`.

Run shape:
- one file per family hook in `families/`
- one prompt template per family in `templates/`
- up to 4 adaptive sweep runs by default
- sweep full runs use a random 4,000-row labeled sample split 80/20 for train/validation
- once a family has one successful sweep run, Agent_3 runs one follow-up experiment and then moves to the next family
- same family prompt contract across runs
- parameters change based on prior F1, crashes, and timeouts
- search uses only trials from the current invocation, not older `agent3_log.json` entries
- the best two sweep architectures are optimized with the remaining time split between them
- optimization uses 10,000 labeled rows when available; if fewer exist, it uses the whole labeled dataset split 80/20 for train/validation
- sweep and optimization trials are metrics-only; the full `test.csv` submission is generated once from the best overall model
- final submission reruns the best overall model on the whole available labeled training set before predicting the full unlabeled `test.csv`

Example:

```bash
python3 src/Agent_3/agent.py --family transformer --max-runs 5
```

Notes:
- Ollama must be running on `localhost:11434`
- the requested model must already be pulled locally
- data is read from `data/train.csv` and `data/test.csv` unless `DISASTER_AGENT_DATA_DIR` is set
