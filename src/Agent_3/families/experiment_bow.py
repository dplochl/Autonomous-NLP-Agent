"""Prompt-first BoW family hook for Agent_3."""

from __future__ import annotations

import re


FAMILY = "BoW"


def default_max_runs() -> int:
    return 5


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
    fixed = _replace_assignment(fixed, "max_features", str(int(spec["max_features"])))
    fixed = _replace_assignment(fixed, "min_df", str(int(spec["min_df"])))
    fixed = _replace_assignment(fixed, "C", repr(float(spec["logreg_c"])))
    fixed = _replace_assignment(fixed, "logreg_c", repr(float(spec["logreg_c"])))
    fixed = _replace_assignment(fixed, "VAL_SIZE", repr(float(spec["val_size"])))
    fixed = fixed.replace("ngram_range=(1, 2)", f"ngram_range=(1, {int(spec['ngram_max'])})")
    fixed = re.sub(r"ngram_range\s*=\s*\(\s*1\s*,\s*\d+\s*\)", f"ngram_range=(1, {int(spec['ngram_max'])})", fixed)
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
        "vectorizer_type": "tfidf_word",
        "max_features": 20000,
        "ngram_max": 2,
        "min_df": 2,
        "logreg_c": 3.0,
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
        "max_features": (5000, 80000),
        "ngram_max": (1, 3),
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
        "max_features",
        "ngram_max",
        "min_df",
        "logreg_c",
        "val_size",
        "threshold_min",
        "threshold_max",
        "threshold_steps",
    ]


def get_template_name() -> str:
    return "train_bow.py.j2"


def get_arch_prompt() -> str:
    return (
        "Use a simple sklearn bag-of-words baseline with TF-IDF features and logistic regression. "
        "Keep the script compact, deterministic, and focused on real validation F1."
    )


def get_spec_prompt() -> str:
    return (
        "Return a reliable sparse-text baseline spec. Prefer conservative TF-IDF settings and a "
        "single validation split. Do not propose deep learning or ensembles."
    )


def get_search_prompt() -> str:
    return (
        "Explore nearby sparse-text settings around the best successful run. If F1 improves, vary max_features, "
        "ngrams, or C locally. If performance is flat, try a slightly different feature density before changing thresholds."
    )


def get_repair_prompt() -> str:
    return "Patch only the broken part of the BoW script and keep the sklearn pipeline intact."


def preflight_issues(code: str, spec: dict[str, object]) -> list[str]:
    issues = []
    required = ["TfidfVectorizer(", "LogisticRegression(", "train_test_split(", "predict_proba(", "METRICS:"]
    for item in required:
        if item not in code:
            issues.append(f"Missing required element: {item}")
    banned = [
        (r"\b(torch|tensorflow|keras|Trainer|AutoModel)\b", "BoW must not use deep learning libraries."),
        (r"\bStratifiedKFold\b", "Use a single validation split instead of K-fold."),
    ]
    for pattern, message in banned:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    return issues


def apply_light_autofixes(code: str, spec: dict[str, object]) -> str:
    fixed = code.replace("train_test_split(X, y, test_size=VAL_SIZE, random_state=SEED, stratify=y)",
                         "train_test_split(X, y, test_size=VAL_SIZE, random_state=SEED, stratify=stratify_labels)")
    if "stratify_labels = y if" not in fixed and "y = train_df['target'].values" in fixed:
        fixed = fixed.replace(
            "y = train_df['target'].values\n",
            "y = train_df['target'].values\n"
            "stratify_labels = y if len(np.unique(y)) > 1 and np.min(np.bincount(y.astype(int))) >= 2 else None\n",
            1,
        )
    return fixed


def build_repair_hint(stderr_text: str) -> str:
    return (
        "\nBoW repair target:\n"
        "- keep sklearn TF-IDF + LogisticRegression\n"
        "- keep one train/validation split\n"
        "- keep threshold search and METRICS output\n"
    )
