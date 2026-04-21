"""Prompt-first Transformer family hook for Agent_3."""

from __future__ import annotations

import re

from families.autofix_utils import fix_text_column_fillna


FAMILY = "Transformer"


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


def _dataset_class_name(code: str) -> str:
    match = re.search(r"(?m)^class\s+(\w+)\s*\(\s*Dataset\s*\)\s*:", code)
    return match.group(1) if match else "TextDataset"


def _ensure_submission_makedirs(code: str) -> str:
    """Make submission directory creation idempotent and indentation-safe."""
    fixed = re.sub(
        r"(?m)^[ \t]*os\.makedirs\(os\.path\.dirname\(submission_path\),[ \t]*exist_ok=True\)[ \t]*\n?",
        "",
        code,
    )

    def add_before_to_csv(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}os.makedirs(os.path.dirname(submission_path), exist_ok=True)\n"
            f"{match.group(0)}"
        )

    return re.sub(
        r"(?m)^(?P<indent>[ \t]*)submission_df\.to_csv\(submission_path,\s*index=False\)",
        add_before_to_csv,
        fixed,
        count=1,
    )


def _flatten_tokenizer_tensors(code: str) -> str:
    fixed = code
    for variable in ("inputs", "encoding"):
        fixed = fixed.replace(
            f"input_ids = {variable}['input_ids']\n",
            f"input_ids = {variable}['input_ids'].flatten()\n",
        )
        fixed = fixed.replace(
            f'attention_mask = {variable}["attention_mask"]\n',
            f'attention_mask = {variable}["attention_mask"].flatten()\n',
        )
        fixed = fixed.replace(
            f"attention_mask = {variable}['attention_mask']\n",
            f"attention_mask = {variable}['attention_mask'].flatten()\n",
        )
        fixed = fixed.replace(
            f'input_ids = {variable}["input_ids"]\n',
            f'input_ids = {variable}["input_ids"].flatten()\n',
        )
    fixed = re.sub(
        r"torch\.tensor\(\s*input_ids\s*,\s*dtype=torch\.long\s*\)",
        "input_ids",
        fixed,
    )
    fixed = re.sub(
        r"torch\.tensor\(\s*attention_mask\s*,\s*dtype=torch\.long\s*\)",
        "attention_mask",
        fixed,
    )
    fixed = fixed.replace(".flatten().flatten()", ".flatten()")
    return fixed


def tune_frozen_code(code: str, spec: dict[str, object], run_name: str) -> str:
    fixed = code
    fixed = re.sub(r"max_length\s*=\s*\d+", f"max_length={int(spec['max_len'])}", fixed)
    fixed = re.sub(r"test_size\s*=\s*[0-9.]+", f"test_size={float(spec['val_size'])}", fixed, count=1)
    fixed = _replace_assignment(fixed, "num_train_epochs", str(int(spec["num_epochs"])))
    fixed = _replace_assignment(fixed, "per_device_train_batch_size", str(int(spec["train_batch_size"])))
    fixed = _replace_assignment(fixed, "per_device_eval_batch_size", str(int(spec["eval_batch_size"])))
    fixed = _replace_assignment(fixed, "learning_rate", repr(float(spec["learning_rate"])))
    fixed = _replace_assignment(fixed, "weight_decay", repr(float(spec["weight_decay"])))
    fixed = re.sub(
        r"thresholds\s*=\s*np\.linspace\([^)]*\)",
        f"thresholds = np.linspace({float(spec['threshold_min'])}, {float(spec['threshold_max'])}, {int(spec['threshold_steps'])})",
        fixed,
        count=1,
    )
    fixed = re.sub(r"train_df\s*=\s*train_df\.head\(\d+\)", f"train_df = train_df.head({int(spec['dry_run_head'])})", fixed)
    fixed = re.sub(r"test_df\s*=\s*test_df\.head\(\d+\)", f"test_df = test_df.head({int(spec['dry_run_head'])})", fixed)
    fixed = re.sub(
        r"(['\"])submissions/[^'\"]+_submission\.csv\1",
        lambda m: f"{m.group(1)}{spec['submission_path']}{m.group(1)}",
        fixed,
    )
    fixed = re.sub(
        r"model_name\s*=\s*['\"][^'\"]+['\"]",
        f"model_name = '{spec['model_name']}'",
        fixed,
    )
    return fixed


def get_default_spec(name: str, submission_path: str) -> dict[str, object]:
    return {
        "architecture": FAMILY,
        "model_name": "distilbert-base-uncased",
        "max_len": 128,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "num_epochs": 3,
        "val_size": 0.2,
        "threshold_min": 0.3,
        "threshold_max": 0.7,
        "threshold_steps": 41,
        "dry_run_head": 16,
        "experiment_name": name,
        "submission_path": submission_path,
    }


def get_spec_ranges() -> dict[str, tuple[float, float]]:
    return {
        "max_len": (64, 256),
        "train_batch_size": (8, 32),
        "eval_batch_size": (8, 32),
        "learning_rate": (1e-6, 5e-4),
        "weight_decay": (0.0, 0.3),
        "num_epochs": (1, 3),
        "val_size": (0.1, 0.3),
        "threshold_min": (0.1, 0.6),
        "threshold_max": (0.4, 0.9),
        "threshold_steps": (11, 81),
        "dry_run_head": (8, 64),
    }


def get_fixed_spec_keys() -> set[str]:
    return {"architecture", "model_name", "experiment_name", "submission_path"}


def get_tunable_keys() -> list[str]:
    return ["max_len", "train_batch_size", "eval_batch_size", "learning_rate", "weight_decay", "num_epochs"]


def get_template_name() -> str:
    return "train_transformer.py.j2"


def get_arch_prompt() -> str:
    return (
        "Use Hugging Face DistilBERT fine-tuning with AutoTokenizer, "
        "AutoModelForSequenceClassification, and Trainer."
    )


def get_spec_prompt() -> str:
    return (
        "Return a reliable DistilBERT spec with one validation split, conservative training values, "
        "and no alternative transformer family. Prefer a script that is likely to run on the first try. "
        "Use threshold tuning over a practical mid-range to maximize F1. "
        "Keep training reproducible with explicit seeding."
    )


def get_search_prompt() -> str:
    return (
        "Search locally around the best transformer settings. Prefer nearby changes in sequence length, "
        "batch size, learning rate, weight decay, or epochs instead of drastic jumps. "
        "Keep threshold tuning in the standard 0.3 to 0.7 range. "
        "Preserve deterministic seeded training and do not wander away from the best successful run."
    )


def normalize_spec(spec: dict[str, object]) -> dict[str, object]:
    normalized = dict(spec)
    normalized["threshold_min"] = 0.3
    normalized["threshold_max"] = 0.7
    normalized["threshold_steps"] = 41
    return normalized


def get_repair_prompt() -> str:
    return (
        "Patch only the broken part of the transformer script. "
        "Keep DistilBERT, Trainer, and the single validation split. "
        "Accept either val_dataset or valid_dataset names. "
        "Keep explicit seeding if it is already present."
    )


def preflight_issues(code: str, spec: dict[str, object]) -> list[str]:
    issues = []
    required_patterns = [
        (r"distilbert-base-uncased", "Missing required element: distilbert-base-uncased."),
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
    banned = [
        (r"\bStratifiedKFold\b", "Use a single validation split instead of K-fold."),
        (r"train_test_split\([^)]*stratify\s*=\s*y[^)]*\)", "Use stratify_labels fallback instead of raw stratify=y."),
        (r"\bDataLoader\b|\bTensorDataset\b", "Do not use DataLoader or TensorDataset in the transformer template."),
        (r"\bkeras\b|\btensorflow\b", "Transformer must use Hugging Face + PyTorch."),
    ]
    for pattern, message in banned:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    return issues


def apply_light_autofixes(code: str, spec: dict[str, object]) -> str:
    fixed = fix_text_column_fillna(code)
    dataset_class = _dataset_class_name(fixed)
    if "import torch" not in fixed and "from transformers import" in fixed:
        fixed = fixed.replace("from transformers import", "import torch\nfrom transformers import", 1)
    fixed = fixed.replace(
        "stratify=stratify_labels\n)",
        "stratify=stratify_labels\n)",
    )
    fixed = fixed.replace(
        "stratify=train_df['target']",
        "stratify=stratify_labels",
    )
    fixed = fixed.replace(
        "stratify=y",
        "stratify=stratify_labels",
    )
    if "stratify_labels =" not in fixed and "train_df['target']" in fixed:
        fixed = fixed.replace(
            "train_df = train_df.head(8)\n",
            "train_df = train_df.sample(n=min(8, len(train_df)), random_state=42)\n"
            "stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None\n",
            1,
        )
        fixed = fixed.replace(
            "test_df = test_df.head(8)\n",
            "test_df = test_df.head(8)\n"
            "stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None\n",
            1,
        )
        fixed = fixed.replace(
            "test_df = test_df.head(8)\n\n# Stratify labels\n",
            "test_df = test_df.head(8)\n\n# Stratify labels\nstratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None\n",
            1,
        )
    fixed = re.sub(
        r"train_df\s*=\s*train_df\.head\(([^)]+)\)",
        r"train_df = train_df.sample(n=min(\1, len(train_df)), random_state=42)",
        fixed,
    )
    fixed = re.sub(
        r"stratify_labels\s*=\s*y\s*if\s*.*",
        "stratify_labels = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None",
        fixed,
    )
    fixed = re.sub(
        r"stratify_labels\s*=\s*train_df\['target'\]\s*if\s*.*",
        "stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None",
        fixed,
    )
    fixed = fixed.replace(
        "stratify_labels = train_df['target'] if len(train_df['target'].unique()) == 2 else None",
        "stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None",
    )
    fixed = fixed.replace("self.texts = texts", "self.texts = list(texts)")
    fixed = fixed.replace("self.labels = labels", "self.labels = list(labels) if labels is not None else None")
    fixed = fixed.replace("text = self.texts[idx]", "text = str(self.texts[idx])")
    fixed = fixed.replace("text = str(self.texts[idx])", "text = str(self.texts[idx])")
    fixed = fixed.replace("self.labels[idx]", "self.labels[idx]")
    fixed = re.sub(r"(['\"])ids\1\s*:", "'input_ids':", fixed)
    fixed = re.sub(r"(['\"])mask\1\s*:", "'attention_mask':", fixed)
    fixed = _flatten_tokenizer_tensors(fixed)
    fixed = fixed.replace(".tolist().tolist()", ".tolist()")
    fixed = fixed.replace("train_texts.tolist()", "train_texts")
    fixed = fixed.replace("val_texts.tolist()", "val_texts")
    fixed = fixed.replace("train_labels.tolist()", "train_labels")
    fixed = fixed.replace("val_labels.tolist()", "val_labels")
    fixed = fixed.replace("test_df['text'].tolist()", "list(test_df['text'])")
    fixed = fixed.replace('test_df["text"].tolist()', 'list(test_df["text"])')
    fixed = fixed.replace(
        "train_texts, val_texts, train_labels, val_labels = train_test_split(",
        "train_texts, val_texts, train_labels, val_labels = train_test_split(",
    )
    if "train_texts = list(train_texts)" not in fixed:
        fixed = re.sub(
            r"(train_texts,\s*val_texts,\s*train_labels,\s*val_labels\s*=\s*train_test_split\([\s\S]*?\)\n)",
            r"\1train_texts = list(train_texts)\nval_texts = list(val_texts)\ntrain_labels = list(train_labels)\nval_labels = list(val_labels)\n",
            fixed,
            count=1,
        )
    fixed = fixed.replace("trainer.predict(valid_dataset).logits", "trainer.predict(valid_dataset).predictions")
    fixed = fixed.replace("trainer.predict(val_dataset).logits", "trainer.predict(val_dataset).predictions")
    fixed = fixed.replace("trainer.predict(test_dataset).logits", "trainer.predict(test_dataset).predictions")
    fixed = fixed.replace("trainer.predict(test_df['text']).predictions", "trainer.predict(test_dataset).predictions")
    fixed = re.sub(r"best_threshold\s*=\s*None", "best_threshold = 0.5", fixed)
    test_dataset_expr = (
        f"test_dataset = {dataset_class}(list(test_df['text']), labels=None, "
        f"tokenizer=tokenizer, max_len={int(spec['max_len'])})"
    )
    fixed = re.sub(
        r"(?m)^test_dataset\s*=\s*\w+\(.*test_df\[['\"]text['\"]\].*\).*$",
        test_dataset_expr,
        fixed,
    )
    while f"{test_dataset_expr}\n{test_dataset_expr}" in fixed:
        fixed = fixed.replace(f"{test_dataset_expr}\n{test_dataset_expr}", test_dataset_expr)
    fixed = fixed.replace("val_logits = trainer.predict(val_dataset).predictions", "val_logits = trainer.predict(val_dataset).predictions")
    fixed = fixed.replace("test_logits = trainer.predict(test_dataset).predictions", "test_logits = trainer.predict(test_dataset).predictions")
    if "test_dataset = TextDataset(list(test_df['text']), labels=None" not in fixed:
        fixed = re.sub(
            r"(val_dataset\s*=\s*TextDataset\([^\n]+\)\n)",
            rf"\1{test_dataset_expr}\n",
            fixed,
            count=1,
        )
    fixed = re.sub(
        r"(\n[ \t]*)label = self\.labels\[idx\]\n([\s\S]*?)\n[ \t]*return \{\n[ \t]*['\"]text['\"]: text,\n[ \t]*['\"]input_ids['\"]: encoding\[['\"]input_ids['\"]\]\.flatten\(\),\n[ \t]*['\"]attention_mask['\"]: encoding\[['\"]attention_mask['\"]\]\.flatten\(\),\n[ \t]*['\"]labels['\"]: torch\.tensor\(label, dtype=torch\.long\)\n[ \t]*\}",
        (
            r"\1item = {\n"
            r"\1    'text': text,\n"
            r"\1    'input_ids': encoding['input_ids'].flatten(),\n"
            r"\1    'attention_mask': encoding['attention_mask'].flatten(),\n"
            r"\1}\n"
            r"\1if self.labels is not None:\n"
            r"\1    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)\n"
            r"\1return item"
        ),
        fixed,
        count=1,
    )
    fixed = fixed.replace("trainer.train()    # Predict validation and test logits", "trainer.train()")
    fixed = fixed.replace("trainer.train()# Predict validation and test logits", "trainer.train()")
    return _ensure_submission_makedirs(fixed)


def build_repair_hint(stderr_text: str) -> str:
    return (
        "\nTransformer repair target:\n"
        "- keep DistilBERT with Trainer\n"
        "- keep one validation split with stratify_labels fallback\n"
        "- keep softmax-based validation probabilities and threshold tuning\n"
        "- keep exact METRICS output and submission path\n"
    )
