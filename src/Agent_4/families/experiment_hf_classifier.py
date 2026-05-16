"""Shared Hugging Face sequence-classifier helpers for Agent_4 families."""

from __future__ import annotations

import re

from families.autofix_utils import fix_text_column_fillna, force_cpu_execution


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


def _force_hf_cpu(code: str) -> str:
    fixed = force_cpu_execution(code)
    fixed = re.sub(r"use_cpu\s*=\s*False", "use_cpu=True", fixed)
    fixed = re.sub(r"use_cpu\s*=\s*True", "use_cpu=True", fixed)
    fixed = re.sub(r"(?m)^[ \t]*use_cuda\s*=\s*(?:True|False)\s*,?\n", "", fixed)
    fixed = re.sub(r"(?m)^[ \t]*no_cuda\s*=\s*(?:True|False)\s*,?\n", "", fixed)
    fixed = re.sub(r"dataloader_pin_memory\s*=\s*True", "dataloader_pin_memory=False", fixed)
    fixed = re.sub(r"dataloader_pin_memory\s*=\s*False", "dataloader_pin_memory=False", fixed)
    fixed = re.sub(r"fp16\s*=\s*True", "fp16=False", fixed)
    fixed = re.sub(r"bf16\s*=\s*True", "bf16=False", fixed)
    if "TrainingArguments(" in fixed:
        if "use_cpu=" not in fixed:
            fixed = fixed.replace(
                "TrainingArguments(",
                "TrainingArguments(\n    use_cpu=True,\n    dataloader_pin_memory=False,",
                1,
            )
        if "bf16=" not in fixed:
            fixed = fixed.replace(
                "dataloader_pin_memory=False,",
                "dataloader_pin_memory=False,\n    bf16=False,",
                1,
            )
        fixed = _normalize_training_arguments_block(fixed)
    return fixed


def _normalize_training_arguments_block(code: str) -> str:
    match = re.search(r"TrainingArguments\((?P<body>[\s\S]*?)\n\)", code, re.MULTILINE)
    if not match:
        return code
    body = match.group("body")
    lines = body.splitlines()
    cleaned: list[str] = []
    seen_keys: set[str] = set()
    indent = "    "
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        key_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", stripped)
        if not key_match:
            cleaned.append(line)
            continue
        key = key_match.group(1)
        if key == "use_cuda":
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        indent_match = re.match(r"([ \t]*)", line)
        if indent_match:
            indent = indent_match.group(1) or indent
        cleaned.append(line)

    required_defaults = [
        ("use_cpu", "True"),
        ("dataloader_pin_memory", "False"),
        ("bf16", "False"),
        ("fp16", "False"),
    ]
    present = {re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", line.strip()).group(1) for line in cleaned if re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", line.strip())}
    inserts = [f"{indent}{key}={value}," for key, value in required_defaults if key not in present]
    if inserts:
        cleaned = inserts + cleaned

    replacement = "TrainingArguments(" + ("\n" + "\n".join(cleaned) if cleaned else "") + "\n)"
    return code[: match.start()] + replacement + code[match.end() :]


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


def get_tunable_keys() -> list[str]:
    return ["max_len", "train_batch_size", "eval_batch_size", "learning_rate", "weight_decay", "num_epochs"]


def get_template_name() -> str:
    return "train_hf_classifier.py.j2"


def normalize_spec(spec: dict[str, object]) -> dict[str, object]:
    normalized = dict(spec)
    normalized["threshold_min"] = 0.3
    normalized["threshold_max"] = 0.7
    normalized["threshold_steps"] = 41
    return normalized


def apply_light_autofixes(code: str, spec: dict[str, object]) -> str:
    fixed = _force_hf_cpu(fix_text_column_fillna(code))
    dataset_class = _dataset_class_name(fixed)
    if "import torch" not in fixed and "from transformers import" in fixed:
        fixed = fixed.replace("from transformers import", "import torch\nfrom transformers import", 1)
    fixed = fixed.replace("stratify=train_df['target']", "stratify=stratify_labels")
    fixed = fixed.replace("stratify=y", "stratify=stratify_labels")
    if "stratify_labels =" not in fixed and "train_df['target']" in fixed:
        fixed = fixed.replace(
            "train_df = train_df.head(8)\n",
            "train_df = train_df.sample(n=min(8, len(train_df)), random_state=42)\n"
            "stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None\n",
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
    fixed = fixed.replace("self.texts = texts", "self.texts = list(texts)")
    fixed = fixed.replace("self.labels = labels", "self.labels = list(labels) if labels is not None else None")
    fixed = fixed.replace("text = self.texts[idx]", "text = str(self.texts[idx])")
    fixed = re.sub(r"(['\"])ids\1\s*:", "'input_ids':", fixed)
    fixed = re.sub(r"(['\"])mask\1\s*:", "'attention_mask':", fixed)
    fixed = _flatten_tokenizer_tensors(fixed)
    # The code-gen LLM frequently emits the deprecated `pad_to_max_length=True`
    # kwarg (old transformers API). On modern transformers this is silently
    # demoted to "pad to longest in batch", which means tokenized sequences
    # come out at variable lengths and the default data collator crashes with
    # "RuntimeError: stack expects each tensor to be equal size". Rewrite to
    # the modern kwarg so the tokenizer actually pads to max_length.
    fixed = re.sub(r"pad_to_max_length\s*=\s*True", 'padding="max_length"', fixed)
    fixed = re.sub(r"pad_to_max_length\s*=\s*False", "padding=False", fixed)
    # The LLM sometimes wraps `stratify_labels` in a try/except where the
    # train_test_split line ends up OUTSIDE the try body (column 0), making
    # the except clause orphan and triggering "SyntaxError: expected 'except'
    # or 'finally' block". The repair LLM can't reliably unwind this. Rewrite
    # the malformed block to the canonical safe if/else form so the script
    # parses cleanly. Pattern observed: try:\n  stratify_labels = ...\n
    # X_... = train_test_split(...stratify=stratify_labels)\n except ValueError:\n
    # X_... = train_test_split(...)
    _stratify_try_except = re.compile(
        r"try:\s*\n"
        r"[ \t]+stratify_labels\s*=\s*train_df\[['\"]target['\"]\]\s*\n"
        r"(?P<split>X_train\s*,\s*X_val\s*,\s*y_train\s*,\s*y_val\s*=\s*train_test_split\([^)]*stratify\s*=\s*stratify_labels[^)]*\))\s*\n"
        r"except\s+(?:ValueError|Exception)\s*:\s*\n"
        r"[ \t]+X_train\s*,\s*X_val\s*,\s*y_train\s*,\s*y_val\s*=\s*train_test_split\([^)]*\)\s*\n",
        re.MULTILINE,
    )
    fixed = _stratify_try_except.sub(
        lambda m: (
            "stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 "
            "and train_df['target'].value_counts().min() >= 2 else None\n"
            f"{m.group('split')}\n"
        ),
        fixed,
    )
    fixed = fixed.replace(".tolist().tolist()", ".tolist()")
    fixed = fixed.replace("train_texts.tolist()", "train_texts")
    fixed = fixed.replace("val_texts.tolist()", "val_texts")
    fixed = fixed.replace("train_labels.tolist()", "train_labels")
    fixed = fixed.replace("val_labels.tolist()", "val_labels")
    fixed = fixed.replace("test_df['text'].tolist()", "list(test_df['text'])")
    fixed = fixed.replace('test_df["text"].tolist()', 'list(test_df["text"])')
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
    fixed = re.sub(
        rf"(val_dataset\s*=\s*{dataset_class}\([^\n]+\)\n)",
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
