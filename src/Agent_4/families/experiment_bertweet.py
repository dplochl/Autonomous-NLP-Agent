"""Prompt-first BERTweet language-model family hook for Agent_3."""

from __future__ import annotations

import re

from families import experiment_hf_classifier as base


FAMILY = "BERTweet"
MODEL_NAME = "vinai/bertweet-base"


default_max_runs = base.default_max_runs
freeze_after_first_success = base.freeze_after_first_success
tune_frozen_code = base.tune_frozen_code
normalize_spec = base.normalize_spec


def get_default_spec(name: str, submission_path: str) -> dict[str, object]:
    return {
        "architecture": FAMILY,
        "model_name": MODEL_NAME,
        "max_len": 128,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "learning_rate": 1.5e-5,
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
    return base.get_spec_ranges()


def get_fixed_spec_keys() -> set[str]:
    return {"architecture", "model_name", "experiment_name", "submission_path"}


def get_tunable_keys() -> list[str]:
    return base.get_tunable_keys()


def get_template_name() -> str:
    return base.get_template_name()


def get_arch_prompt() -> str:
    return (
        "Use Hugging Face BERTweet fine-tuning with AutoTokenizer(use_fast=False), "
        "AutoModelForSequenceClassification, and Trainer. Normalize tweet URLs to HTTPURL "
        "and mentions to @USER before tokenization."
    )


def get_spec_prompt() -> str:
    return (
        "Return a reliable vinai/bertweet-base spec with one validation split and conservative "
        "training values. Use the slow tokenizer with use_fast=False and threshold tuning over "
        "a practical mid-range to maximize F1."
    )


def get_search_prompt() -> str:
    return (
        "Search locally around the best BERTweet settings. Prefer nearby changes in sequence length, "
        "batch size, learning rate, weight decay, or epochs instead of drastic jumps."
    )


def get_repair_prompt() -> str:
    return (
        "Patch only the broken part of the BERTweet script. Keep vinai/bertweet-base, "
        "AutoTokenizer(use_fast=False), Trainer, and the single validation split."
    )


def _force_slow_tokenizer(code: str) -> str:
    fixed = re.sub(
        r"AutoTokenizer\.from_pretrained\((model_name|['\"]vinai/bertweet-base['\"])\)",
        r"AutoTokenizer.from_pretrained(\1, use_fast=False)",
        code,
    )
    return re.sub(
        r"AutoTokenizer\.from_pretrained\((model_name|['\"]vinai/bertweet-base['\"]),\s*use_fast\s*=\s*True\)",
        r"AutoTokenizer.from_pretrained(\1, use_fast=False)",
        fixed,
    )


def _ensure_re_import(code: str) -> str:
    if "import re" in code:
        return code
    if "import os" in code:
        return code.replace("import os\n", "import os\nimport re\n", 1)
    return "import re\n" + code


def _ensure_submission_path(code: str, spec: dict[str, object]) -> str:
    if "submission_path =" in code:
        return code
    assignment = f"submission_path = os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', {str(spec['submission_path'])!r})\n"
    if "SAMPLE_SEED =" in code:
        return re.sub(r"(?m)^(SAMPLE_SEED\s*=.*\n)", r"\1" + assignment, code, count=1)
    return assignment + code


def _ensure_dry_run_head(code: str, spec: dict[str, object]) -> str:
    if re.search(r"if\s+DRY_RUN\s*:\s*\n\s*train_df\s*=", code):
        return code
    dry_head = int(spec.get("dry_run_head", 16))
    block = (
        f"\nif DRY_RUN:\n"
        f"    train_df = train_df.head({dry_head})\n"
        f"    test_df = test_df.head({dry_head})\n"
    )
    marker = "# Split data"
    if marker in code:
        return code.replace(marker, block + "\n" + marker, 1)
    return code


def _skip_training_during_dry_run(code: str) -> str:
    fixed = re.sub(
        r"if\s+DRY_RUN\s*:\n(?:[ \t]+[^\n]*\n)*?[ \t]+trainer\.train\(\)\n",
        "if not DRY_RUN:\n    trainer.train()\n",
        code,
        count=1,
    )
    fixed = fixed.replace("# Train if not DRY_RUN\nif DRY_RUN:\n    trainer.train()", "# Train if not DRY_RUN\nif not DRY_RUN:\n    trainer.train()")
    if "trainer.train()" in fixed and "if not DRY_RUN:\n    trainer.train()" not in fixed:
        fixed = fixed.replace("trainer.train()", "if not DRY_RUN:\n    trainer.train()", 1)
    return fixed


def _safe_stratify_fallback(code: str) -> str:
    return re.sub(
        r"stratify_labels\s*=\s*train_df\[[\"']target[\"']\]\s*if\s*.*",
        "stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None",
        code,
    )


def _apply_tweet_normalization(code: str) -> str:
    if "def preprocess_text" not in code:
        return code
    if re.search(r"train_df\[[\"']text[\"']\]\s*=\s*train_df\[[\"']text[\"']\]\.apply\(preprocess_text\)", code):
        return code
    marker = "# Sample train data"
    block = (
        'train_df["text"] = train_df["text"].apply(preprocess_text)\n'
        'test_df["text"] = test_df["text"].apply(preprocess_text)\n\n'
    )
    if marker in code:
        return code.replace(marker, block + marker, 1)
    return code


def _replace_encode_plus(code: str) -> str:
    fixed = code.replace("self.tokenizer.encode_plus(", "self.tokenizer(")
    return re.sub(
        r"(self\.tokenizer\(\s*\n\s*text,\s*\n)\s*None,\s*\n",
        r"\1",
        fixed,
    )


def _use_dataset_predictions(code: str) -> str:
    fixed = code
    fixed = re.sub(
        r"val_predictions\s*=\s*trainer\.predict\(\s*val_texts\s*\)\.(?:predictions|logits)",
        "val_predictions = trainer.predict(val_dataset).predictions",
        fixed,
    )
    fixed = re.sub(
        r"val_predictions\s*=\s*trainer\.predict\(\s*valid_texts\s*\)\.(?:predictions|logits)",
        "val_predictions = trainer.predict(valid_dataset).predictions",
        fixed,
    )
    fixed = re.sub(
        r"test_predictions\s*=\s*trainer\.predict\(\s*test_df\[['\"]text['\"]\]\s*\)\.(?:predictions|logits)",
        "test_predictions = trainer.predict(test_dataset).predictions",
        fixed,
    )
    fixed = re.sub(
        r"test_predictions\s*=\s*trainer\.predict\(\s*list\(test_df\[['\"]text['\"]\]\)\s*\)\.(?:predictions|logits)",
        "test_predictions = trainer.predict(test_dataset).predictions",
        fixed,
    )
    return fixed


def _ensure_probability_conversion(code: str) -> str:
    fixed = code
    if "val_logits = trainer.predict(val_dataset).predictions" in fixed and "val_probs =" not in fixed:
        fixed = fixed.replace(
            "val_logits = trainer.predict(val_dataset).predictions",
            "val_logits = trainer.predict(val_dataset).predictions\n"
            "val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))\n"
            "val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)",
            1,
        )
        fixed = fixed.replace("val_logits[:, 1]", "val_probs[:, 1]")
    if "val_logits = trainer.predict(valid_dataset).predictions" in fixed and "val_probs =" not in fixed:
        fixed = fixed.replace(
            "val_logits = trainer.predict(valid_dataset).predictions",
            "val_logits = trainer.predict(valid_dataset).predictions\n"
            "val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))\n"
            "val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)",
            1,
        )
        fixed = fixed.replace("val_logits[:, 1]", "val_probs[:, 1]")
    if "val_probs = np.exp" not in fixed:
        fixed = fixed.replace(
            "val_predictions = trainer.predict(val_dataset).predictions",
            "val_logits = trainer.predict(val_dataset).predictions\n"
            "val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))\n"
            "val_predictions = val_probs / val_probs.sum(axis=1, keepdims=True)",
        )
        fixed = fixed.replace(
            "val_predictions = trainer.predict(valid_dataset).predictions",
            "val_logits = trainer.predict(valid_dataset).predictions\n"
            "val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))\n"
            "val_predictions = val_probs / val_probs.sum(axis=1, keepdims=True)",
        )
    if "test_probs = np.exp" not in fixed:
        fixed = fixed.replace(
            "test_predictions = trainer.predict(test_dataset).predictions",
            "test_logits = trainer.predict(test_dataset).predictions\n"
            "test_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))\n"
            "test_predictions = test_probs / test_probs.sum(axis=1, keepdims=True)",
        )
    fixed = re.sub(
        r"test_logits\s*=\s*trainer\.predict\(test_dataset\)\.predictions",
        "test_predictor = final_trainer if FINAL_SUBMISSION else trainer\n    test_logits = test_predictor.predict(test_dataset).predictions",
        fixed,
    )
    fixed = re.sub(
        r"test_probs\s*=\s*softmax\(test_logits,\s*axis=1\)\s*\[:,\s*1\s*\]",
        "test_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))\n    test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)\n    test_probs = test_probs[:, 1]",
        fixed,
    )
    return fixed


def _canonicalize_dataset_section(code: str, spec: dict[str, object]) -> str:
    max_len = int(spec["max_len"])
    train_label_var = "train_labels" if "train_labels" in code else "y_train"
    val_label_var = "val_labels" if "val_labels" in code else "y_val"
    replacement = f"""# Dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=tokenizer, max_len={max_len}):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {{
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Create datasets
train_dataset = TweetDataset(train_texts, {train_label_var}, tokenizer=tokenizer, max_len={max_len})
val_dataset = TweetDataset(val_texts, {val_label_var}, tokenizer=tokenizer, max_len={max_len})
test_dataset = TweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len={max_len})

"""
    return re.sub(
        r"# (?:Tokenize data|Create Dataset class|Dataset class)[\s\S]*?(?=# Training arguments)",
        replacement,
        code,
        count=1,
    )


def _canonicalize_final_submission_section(code: str, spec: dict[str, object]) -> str:
    max_len = int(spec["max_len"])
    model_name_expr = 'spec["model_name"]' if 'spec =' in code else "model_name"
    val_label_var = "val_labels" if "val_labels" in code else "y_val"
    replacement = f"""# Predict validation logits
val_logits = trainer.predict(val_dataset).predictions
val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

# Choose best threshold
best_threshold = 0.5
best_f1 = 0.0
for threshold in np.linspace({float(spec["threshold_min"])}, {float(spec["threshold_max"])}, {int(spec["threshold_steps"])}):
    val_preds = (val_probs[:, 1] > threshold).astype(int)
    f1 = f1_score({val_label_var}, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission training and prediction
if FINAL_SUBMISSION:
    final_model = AutoModelForSequenceClassification.from_pretrained({model_name_expr}, num_labels=2)
    final_train_dataset = TweetDataset(train_df['text'], train_df['target'], tokenizer=tokenizer, max_len={max_len})
    final_trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=final_train_dataset
    )
    if not DRY_RUN:
        final_trainer.train()
    test_predictor = final_trainer
else:
    test_predictor = trainer

# Write submission if required
if WRITE_SUBMISSION:
    test_logits = test_predictor.predict(TweetDataset(test_df['text'], labels=None, tokenizer=tokenizer, max_len={max_len})).predictions
    test_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
    test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
    test_preds = (test_probs[:, 1] > best_threshold).astype(int)
    submission_df = pd.DataFrame({{'id': test_df['id'], 'target': test_preds}})
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

# Metrics
val_preds = (val_probs[:, 1] > best_threshold).astype(int)
f1 = f1_score({val_label_var}, val_preds)
acc = accuracy_score({val_label_var}, val_preds)

"""
    return re.sub(
        r"# (?:Predict validation logits|Evaluate on validation set|Final submission|FINAL_SUBMISSION: train final model on full train data|Final submission training and prediction)[\s\S]*?(?=# Print metrics|# Metrics|# Calculate metrics)",
        replacement,
        code,
        count=1,
    )


def apply_light_autofixes(code: str, spec: dict[str, object]) -> str:
    fixed = base.apply_light_autofixes(code, spec)
    fixed = _force_slow_tokenizer(fixed)
    fixed = _ensure_re_import(fixed)
    fixed = _ensure_submission_path(fixed, spec)
    fixed = _ensure_dry_run_head(fixed, spec)
    fixed = _skip_training_during_dry_run(fixed)
    fixed = _safe_stratify_fallback(fixed)
    fixed = _apply_tweet_normalization(fixed)
    fixed = _replace_encode_plus(fixed)
    fixed = _use_dataset_predictions(fixed)
    fixed = _ensure_probability_conversion(fixed)
    fixed = _canonicalize_dataset_section(fixed, spec)
    return _canonicalize_final_submission_section(fixed, spec)



def preflight_issues(code: str, spec: dict[str, object]) -> list[str]:
    issues = []
    required_patterns = [
        (re.escape(MODEL_NAME), f"Missing required element: {MODEL_NAME}."),
        (r"AutoTokenizer", "Missing required element: AutoTokenizer."),
        (r"use_fast\s*=\s*False", "BERTweet must load AutoTokenizer with use_fast=False."),
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
        (r"\bDataLoader\b|\bTensorDataset\b", "Do not use DataLoader or TensorDataset in the BERTweet template."),
        (r"\bkeras\b|\btensorflow\b", "BERTweet must use Hugging Face + PyTorch."),
    ]
    for pattern, message in banned:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    return issues


def build_repair_hint(stderr_text: str) -> str:
    return (
        "\nBERTweet repair target:\n"
        "- keep vinai/bertweet-base with AutoTokenizer(use_fast=False) and Trainer\n"
        "- keep one validation split with stratify_labels fallback\n"
        "- keep softmax-based validation probabilities and threshold tuning\n"
        "- keep exact METRICS output and submission path\n"
    )
