"""Prompt-first RoBERTa language-model family hook for Agent_3."""

from __future__ import annotations

import re

from families import experiment_transformer as base


FAMILY = "RoBERTa"
MODEL_NAME = "roberta-base"


default_max_runs = base.default_max_runs
freeze_after_first_success = base.freeze_after_first_success
tune_frozen_code = base.tune_frozen_code
normalize_spec = base.normalize_spec
apply_light_autofixes = base.apply_light_autofixes


def get_default_spec(name: str, submission_path: str) -> dict[str, object]:
    spec = base.get_default_spec(name, submission_path)
    spec["architecture"] = FAMILY
    spec["model_name"] = MODEL_NAME
    spec["max_len"] = 128
    spec["train_batch_size"] = 16
    spec["eval_batch_size"] = 16
    spec["learning_rate"] = 1.5e-5
    spec["weight_decay"] = 0.01
    spec["num_epochs"] = 3
    return spec


def get_spec_ranges() -> dict[str, tuple[float, float]]:
    return base.get_spec_ranges()


def get_fixed_spec_keys() -> set[str]:
    return {"architecture", "model_name", "experiment_name", "submission_path"}


def get_tunable_keys() -> list[str]:
    return base.get_tunable_keys()


def get_template_name() -> str:
    return "train_transformer.py.j2"


def get_arch_prompt() -> str:
    return (
        "Use Hugging Face RoBERTa fine-tuning with AutoTokenizer, "
        "AutoModelForSequenceClassification, and Trainer. Treat this as a strong "
        "language-model comparison against DistilBERT."
    )


def get_spec_prompt() -> str:
    return (
        "Return a reliable roberta-base spec with one validation split and conservative training values. "
        "Use threshold tuning over a practical mid-range to maximize F1."
    )


def get_search_prompt() -> str:
    return (
        "Search locally around the best RoBERTa settings. Prefer nearby changes in sequence length, "
        "batch size, learning rate, weight decay, or epochs instead of drastic jumps."
    )


def get_repair_prompt() -> str:
    return (
        "Patch only the broken part of the RoBERTa script. "
        "Keep roberta-base, Trainer, and the single validation split."
    )


def preflight_issues(code: str, spec: dict[str, object]) -> list[str]:
    issues = []
    required_patterns = [
        (re.escape(MODEL_NAME), f"Missing required element: {MODEL_NAME}."),
        (r"AutoTokenizer", "Missing required element: AutoTokenizer."),
        (r"AutoModelForSequenceClassification", "Missing required element: AutoModelForSequenceClassification."),
        (r"Trainer\(", "Missing required element: Trainer."),
        (r"TrainingArguments\(", "Missing required element: TrainingArguments."),
        (r"train_test_split\(", "Missing required element: train_test_split."),
        (r"stratify_labels\s*=", "Missing required element: stratify_labels fallback."),
        (r"trainer\.predict\((?:val|valid)_dataset\)\.(?:predictions|logits)", "Missing required validation predict call."),
        (r"(?:softmax|np\.exp\()", "Missing required stable softmax/probability conversion from logits."),
        (r"METRICS:", "Missing required element: METRICS output."),
    ]
    for pattern, message in required_patterns:
        if not re.search(pattern, code):
            issues.append(message)
    if re.search(r"['\"]ids['\"]\s*:", code):
        issues.append("Dataset must return key 'input_ids', not 'ids'.")
    if re.search(r"['\"]mask['\"]\s*:", code):
        issues.append("Dataset must return key 'attention_mask', not 'mask'.")
    banned = [
        (r"\bStratifiedKFold\b", "Use a single validation split instead of K-fold."),
        (r"train_test_split\([^)]*stratify\s*=\s*y[^)]*\)", "Use stratify_labels fallback instead of raw stratify=y."),
        (r"\bDataLoader\b|\bTensorDataset\b", "Do not use DataLoader or TensorDataset in the RoBERTa template."),
        (r"\bkeras\b|\btensorflow\b", "RoBERTa must use Hugging Face + PyTorch."),
    ]
    for pattern, message in banned:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    return issues


def build_repair_hint(stderr_text: str) -> str:
    return (
        "\nRoBERTa repair target:\n"
        "- keep roberta-base with Trainer\n"
        "- keep one validation split with stratify_labels fallback\n"
        "- keep softmax-based validation probabilities and threshold tuning\n"
        "- keep exact METRICS output and submission path\n"
    )
