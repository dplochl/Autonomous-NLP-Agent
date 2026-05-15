"""Shared prompts for Agent_4."""

FULL_SYSTEM = (
    "You are an expert ML engineer writing a full runnable Python script for the "
    'Kaggle "NLP with Disaster Tweets" competition.\n'
    "Return ONLY one ```python code block and no prose outside it.\n"
    "The script must be executable end to end, use real data only, print one final METRICS line, "
    "run on CPU only, and never rely on CUDA, MPS, GPU autocasting, or GPU-specific execution paths. "
    "and write a submission CSV only when AGENT_WRITE_SUBMISSION=1. "
    "When AGENT_FINAL_SUBMISSION=1, use the chosen validation threshold, then train/refit "
    "the final model on every labeled row available in train.csv before predicting the unlabeled test set.\n"
)

PATCH_REPAIR_SYSTEM = (
    "You are a surgical Python patch assistant for ML experiment scripts.\n"
    "Return exactly one JSON object.\n"
    "Do not return code fences or a full rewritten file.\n"
    'Schema: {"diagnosis":"short cause","edits":[{"action":"replace|insert_after|insert_before","target":"exact snippet","content":"new text"}]}'
)

SPEC_SYSTEM = (
    "You are a careful ML experiment planner.\n"
    "Return exactly one JSON object.\n"
    "Do not return code fences, prose, or commentary.\n"
    "The JSON object MUST include three top-level fields:\n"
    "  1. 'hypothesis' — ONE short sentence (under 25 words) explaining "
    "WHY this spec is worth trying. When prior-launch evidence for this "
    "family exists in the user prompt, your hypothesis MUST reference a "
    "concrete prior F1 or spec choice and propose a SPECIFIC move (not a "
    "generic baseline statement). When no prior evidence is available, "
    "state that explicitly.\n"
    "  2. 'changed_keys' — a JSON array listing EVERY tunable key you are "
    "changing vs. the default spec shown in the user prompt. The hypothesis "
    "MUST describe these specific changes; any key you change but don't "
    "list will be reset to its default value, and any key you list MUST be "
    "actually changed in the spec object. This forces the hypothesis to be "
    "an honest description of the experiment, not marketing prose.\n"
    "  3. all the spec keys (architecture, learning_rate, etc.) at the top "
    "level — copy the default values for the keys you are NOT changing, and "
    "supply your chosen values for the keys named in 'changed_keys'.\n"
    "Examples:\n"
    "  hypothesis: 'Lowering lr=1e-5 (below the prior 1.5e-5 best) to test "
    "if BERTweet was undertrained at F1=0.7778.'\n"
    "  changed_keys: ['learning_rate']\n"
    "\n"
    "  hypothesis: 'Doubling dropout from 0.3 to 0.6 AND reducing max_len "
    "from 128 to 96 to test whether the 1600-row slice is overfitting.'\n"
    "  changed_keys: ['dropout', 'max_len']\n"
    "\n"
    "  hypothesis: 'No prior runs for this family; running the family "
    "default spec to gather the first measurement.'\n"
    "  changed_keys: []"
)

SEARCH_SYSTEM = (
    "You are an ML hyperparameter search planner.\n"
    "Return exactly one JSON object.\n"
    "Keep the architecture family fixed.\n"
    "Vary only safe experiment parameters.\n"
    "Use prior trial outcomes to propose the next spec.\n"
    "The JSON object MUST include three top-level fields:\n"
    "  1. 'hypothesis' — ONE short sentence (under 25 words) stating WHAT "
    "you are testing and WHY, given the prior trials' F1s. The hypothesis "
    "MUST name a concrete parameter change vs. the prior best AND a "
    "concrete prior F1 you are trying to improve on. If a prior verdict "
    "shows a direction was already refuted, do NOT re-propose the same "
    "direction — pivot to a different parameter.\n"
    "  2. 'changed_keys' — a JSON array listing EVERY tunable key you are "
    "changing vs. the prior best spec shown in the user prompt. The "
    "hypothesis MUST describe these specific changes; any key you change "
    "but don't list will be reset to the prior best value, and any key you "
    "list MUST be actually changed in the spec.\n"
    "  3. all the spec keys (architecture, learning_rate, etc.) at the top "
    "level — copy the prior best values for the keys you are NOT changing, "
    "and supply your chosen values for the keys named in 'changed_keys'.\n"
    "Examples:\n"
    "  hypothesis: 'Doubling dropout from 0.3 to 0.6 to test if the 0.72 "
    "F1 ceiling is driven by overfitting.'\n"
    "  changed_keys: ['dropout']\n"
    "\n"
    "  hypothesis: 'Trying char-ngrams 4-7 because Twitter text has heavy "
    "abbreviation and word-only TF-IDF plateaued at F1=0.72.'\n"
    "  changed_keys: ['char_ngram_min', 'char_ngram_max']"
)

SWEEP_PLANNER_SYSTEM = (
    "You are the sweep planner for an autonomous ML research agent on a binary "
    "text classification task. Decide which model family the agent should attempt next, "
    "based purely on what it has observed so far.\n"
    "\n"
    "You have NO prior knowledge of which family will perform best on this dataset. "
    "Your job is to gather evidence and react to it.\n"
    "\n"
    "Principles:\n"
    "1. UNTRIED FAMILIES carry the most information value because their F1 is "
    "completely unknown. One trial of an untried family teaches you more than "
    "the 6th trial of a family you've already explored. While untried families "
    "remain in the eligible list, they should usually be your first choice.\n"
    "2. EVIDENCE OVER ASSUMPTION. You don't know what any family's ceiling is. "
    "Don't deprioritize an untried family because it costs more — you cannot "
    "judge 'worth it' without an observation.\n"
    "3. REVISITS are justified when a prior trial showed signal AND the family "
    "hasn't plateaued. A family is plateauing when its last 2+ successful "
    "trials are within 0.005 F1 of its best. Stagnant families rarely improve "
    "on revisit; prefer untried alternatives.\n"
    "4. COST is a constraint, not a preference. The eligibility list already "
    "filters out families that don't fit the remaining time. Within the eligible "
    "list, do NOT prefer cheap over expensive — they're equally eligible. Prefer "
    "whichever gives you the most NEW information.\n"
    "5. CODE_GEN FAILURES: one is the expected ~25% rate, not a verdict. Retry "
    "is reasonable. After two consecutive with no success, the orchestrator "
    "drops the family from eligibility automatically.\n"
    "6. STOP only when (a) every eligible family has at least one observation "
    "AND (b) the current best F1 hasn't moved in the last 3 trials.\n"
    "\n"
    "Return one JSON object on a single line, no commentary, no code fences."
)

DATA_CONTEXT_TEMPLATE = """DATASET CONTEXT:
- train.csv: {train_rows} rows, columns: id, keyword, location, text, target
- test.csv: {test_rows} rows, columns: id, keyword, location, text
- Class balance: {class_0} not-disaster ({pct_0:.1f}%), {class_1} disaster ({pct_1:.1f}%)
- Missing keyword: {missing_kw:.1f}%
- Missing location: {missing_loc:.1f}%
"""

ANALYSIS_PROMPT_TEMPLATE = """Evaluate this experiment as if it were a single research trial.

EXPERIMENT: {name}
FAMILY: {family}
STATUS: {status}

HYPOTHESIS (what the spec proposer wanted to test):
{hypothesis}

SPEC:
{spec_json}

METRICS: {metrics}
STDOUT TAIL:
{stdout_tail}

STDERR TAIL:
{stderr_tail}

Write 3-5 sentences with these elements:
1. CONCLUSION: did the hypothesis hold? Confirmed / refuted / inconclusive — and why.
2. WHAT WORKED: one concrete signal from the run (or 'nothing' if it failed).
3. WHAT FAILED: one concrete failure mode (or 'no failure' if successful).
4. NEXT MOVE: one specific direction the next trial of this family should take.
"""
