"""Shared prompts for Agent_3."""

FULL_SYSTEM = (
    "You are an expert ML engineer writing a full runnable Python script for the "
    'Kaggle "NLP with Disaster Tweets" competition.\n'
    "Return ONLY one ```python code block and no prose outside it.\n"
    "The script must be executable end to end, use real data only, print one final METRICS line, "
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
    "Do not return code fences, prose, or commentary."
)

SEARCH_SYSTEM = (
    "You are an ML hyperparameter search planner.\n"
    "Return exactly one JSON object.\n"
    "Keep the architecture family fixed.\n"
    "Vary only safe experiment parameters.\n"
    "Use prior trial outcomes to propose the next spec."
)

ANALYSIS_SYSTEM = "You are a concise ML experiment analyst."

DATA_CONTEXT_TEMPLATE = """DATASET CONTEXT:
- train.csv: {train_rows} rows, columns: id, keyword, location, text, target
- test.csv: {test_rows} rows, columns: id, keyword, location, text
- Class balance: {class_0} not-disaster ({pct_0:.1f}%), {class_1} disaster ({pct_1:.1f}%)
- Missing keyword: {missing_kw:.1f}%
- Missing location: {missing_loc:.1f}%
"""

ANALYSIS_PROMPT_TEMPLATE = """Analyze this experiment briefly.

EXPERIMENT: {name}
FAMILY: {family}
STATUS: {status}
SPEC:
{spec_json}

METRICS: {metrics}
STDOUT TAIL:
{stdout_tail}

STDERR TAIL:
{stderr_tail}

Write 3-5 sentences on what worked, what failed, and what the next run should change.
"""
