"""Prompt-first advanced sparse-text family hook for Agent_3."""

from __future__ import annotations

import re

from families.autofix_utils import fix_text_column_fillna


FAMILY = "BoW_advanced"


def default_max_runs() -> int:
    return 4


def freeze_after_first_success() -> bool:
    return True


def _replace_assignment(code: str, name: str, value: str) -> str:
    return re.sub(
        rf"({re.escape(name)}\s*=\s*)([^,\n)]+)",
        rf"\g<1>{value}",
        code,
    )


def tune_frozen_code(code: str, spec: dict[str, object], run_name: str) -> str:
    fixed = code
    fixed = _replace_assignment(fixed, "word_max_features", str(int(spec["word_max_features"])))
    fixed = _replace_assignment(fixed, "char_max_features", str(int(spec["char_max_features"])))
    fixed = _replace_assignment(fixed, "min_df", str(int(spec["min_df"])))
    fixed = _replace_assignment(fixed, "C", repr(float(spec["logreg_c"])))
    fixed = _replace_assignment(fixed, "logreg_c", repr(float(spec["logreg_c"])))
    fixed = re.sub(r"ngram_range\s*=\s*\(\s*1\s*,\s*\d+\s*\)", f"ngram_range=(1, {int(spec['word_ngram_max'])})", fixed, count=1)
    fixed = re.sub(
        r"analyzer\s*=\s*['\"]char['\"][\s\S]*?ngram_range\s*=\s*\(\s*\d+\s*,\s*\d+\s*\)",
        lambda m: re.sub(
            r"ngram_range\s*=\s*\(\s*\d+\s*,\s*\d+\s*\)",
            f"ngram_range=({int(spec['char_ngram_min'])}, {int(spec['char_ngram_max'])})",
            m.group(0),
        ),
        fixed,
        count=1,
    )
    fixed = re.sub(
        r"thresholds\s*=\s*np\.linspace\([^)]*\)",
        f"thresholds = np.linspace({float(spec['threshold_min'])}, {float(spec['threshold_max'])}, {int(spec['threshold_steps'])})",
        fixed,
        count=1,
    )
    fixed = re.sub(r"train_df\s*=\s*train_df\.head\(\d+\)", f"train_df = train_df.head({int(spec['dry_run_head'])})", fixed)
    fixed = re.sub(
        r"(['\"])submissions/[^'\"]+_submission\.csv\1",
        lambda m: f"{m.group(1)}{spec['submission_path']}{m.group(1)}",
        fixed,
    )
    return fixed


def get_default_spec(name: str, submission_path: str) -> dict[str, object]:
    return {
        "architecture": FAMILY,
        "word_max_features": 30000,
        "char_max_features": 20000,
        "word_ngram_max": 2,
        "char_ngram_min": 3,
        "char_ngram_max": 5,
        "min_df": 2,
        "logreg_c": 4.0,
        "val_size": 0.2,
        "threshold_min": 0.3,
        "threshold_max": 0.7,
        "threshold_steps": 41,
        "dry_run_head": 200,
        "experiment_name": name,
        "submission_path": submission_path,
    }


def get_spec_ranges() -> dict[str, tuple[float, float]]:
    return {
        "word_max_features": (10000, 100000),
        "char_max_features": (5000, 80000),
        "word_ngram_max": (1, 3),
        "char_ngram_min": (2, 4),
        "char_ngram_max": (4, 6),
        "min_df": (1, 10),
        "logreg_c": (0.1, 10.0),
        "val_size": (0.1, 0.3),
        "threshold_min": (0.1, 0.6),
        "threshold_max": (0.4, 0.9),
        "threshold_steps": (11, 81),
        "dry_run_head": (50, 500),
    }


def get_fixed_spec_keys() -> set[str]:
    return {"architecture", "experiment_name", "submission_path"}


def get_tunable_keys() -> list[str]:
    return [
        "word_max_features",
        "char_max_features",
        "word_ngram_max",
        "char_ngram_min",
        "char_ngram_max",
        "min_df",
        "logreg_c",
        "threshold_min",
        "threshold_max",
        "threshold_steps",
    ]


def get_template_name() -> str:
    return "train_bow_advanced.py.j2"


def get_arch_prompt() -> str:
    return (
        "Use a stronger sparse baseline with both word and character TF-IDF features merged into one "
        "logistic regression model. Keep it entirely sklearn/scipy."
    )


def get_spec_prompt() -> str:
    return "Return a reliable dual-vectorizer sparse-text spec with one validation split and conservative values."


def get_search_prompt() -> str:
    return (
        "Search around the best successful sparse combination by adjusting word/char feature capacity and regularization. "
        "If a run improves, make a nearby variation instead of a drastic jump."
    )


def get_repair_prompt() -> str:
    return "Patch only the broken part of the advanced sparse script. Keep the sparse feature merge approach."


def preflight_issues(code: str, spec: dict[str, object]) -> list[str]:
    issues = []
    required = ["TfidfVectorizer(", "hstack(", "LogisticRegression(", "predict_proba(", "METRICS:"]
    for item in required:
        if item not in code:
            issues.append(f"Missing required element: {item}")
    banned = [
        (r"\b(torch|tensorflow|keras|Trainer|AutoModel)\b", "Advanced BoW must stay in sklearn/scipy."),
        (r"\bStratifiedKFold\b", "Use one validation split instead of K-fold."),
    ]
    for pattern, message in banned:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    return issues


def apply_light_autofixes(code: str, spec: dict[str, object]) -> str:
    fixed = fix_text_column_fillna(code)
    if "from scipy.sparse import hstack" not in fixed and "hstack(" in fixed:
        fixed = fixed.replace("from sklearn.feature_extraction.text import TfidfVectorizer\n",
                              "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom scipy.sparse import hstack\n", 1)
    return fixed


def build_repair_hint(stderr_text: str) -> str:
    return (
        "\nBoW_advanced repair target:\n"
        "- keep word + character TF-IDF features\n"
        "- keep sparse hstack merge and LogisticRegression\n"
        "- keep one validation split and threshold tuning\n"
    )
