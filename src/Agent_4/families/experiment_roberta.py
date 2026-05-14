"""Prompt-first RoBERTa language-model family hook for Agent_4."""

from __future__ import annotations

import re

from families import experiment_hf_classifier as base


FAMILY = "RoBERTa"
MODEL_NAME = "roberta-base"


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


def _canonicalize_dataset_section(code: str, spec: dict[str, object]) -> str:
    max_len = int(spec["max_len"])
    replacement = f"""# Dataset class
class DisasterTweetDataset(Dataset):
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
train_dataset = DisasterTweetDataset(train_texts, train_labels, tokenizer=tokenizer, max_len={max_len})
val_dataset = DisasterTweetDataset(val_texts, val_labels, tokenizer=tokenizer, max_len={max_len})
test_dataset = DisasterTweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len={max_len})

"""
    return re.sub(
        r"# (?:Tokenize data|Create Dataset class|Dataset class)[\s\S]*?(?=# Training arguments)",
        replacement,
        code,
        count=1,
    )


def _ensure_probability_and_metrics(code: str) -> str:
    fixed = code
    fixed = fixed.replace(
        "val_probs = torch.softmax(torch.tensor(val_logits), dim=1)[:, 1].numpy()",
        "val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))\n"
        "val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)\n"
        "val_probs = val_probs[:, 1]",
    )
    fixed = fixed.replace(
        "val_probabilities = torch.softmax(torch.tensor(val_predictions), dim=1)[:, 1].numpy()",
        "val_probabilities = np.exp(val_predictions - np.max(val_predictions, axis=1, keepdims=True))\n"
        "val_probabilities = val_probabilities / val_probabilities.sum(axis=1, keepdims=True)\n"
        "val_probabilities = val_probabilities[:, 1]",
    )
    fixed = fixed.replace("val_preds = (val_logits[:, 1] > best_threshold).astype(int)", "val_preds = (val_probs > best_threshold).astype(int)")
    fixed = fixed.replace("val_preds = (val_predictions[:, 1] > best_threshold).astype(int)", "val_preds = (val_probabilities > best_threshold).astype(int)")
    return fixed


def _canonicalize_final_submission_section(code: str, spec: dict[str, object]) -> str:
    max_len = int(spec["max_len"])
    model_name = str(spec["model_name"])
    submission_path = str(spec["submission_path"])
    replacement = f"""# Final submission training and prediction
if FINAL_SUBMISSION:
    final_model = AutoModelForSequenceClassification.from_pretrained('{model_name}', num_labels=2)
    final_train_dataset = DisasterTweetDataset(train_df['text'], train_df['target'], tokenizer=tokenizer, max_len={max_len})
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
    test_logits = test_predictor.predict(DisasterTweetDataset(test_df['text'], labels=None, tokenizer=tokenizer, max_len={max_len})).predictions
    test_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
    test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
    test_preds = (test_probs[:, 1] > best_threshold).astype(int)
    submission_df = pd.DataFrame({{'id': test_df['id'], 'target': test_preds}})
    os.makedirs(os.path.dirname({submission_path!r}), exist_ok=True)
    submission_df.to_csv({submission_path!r}, index=False)

"""
    return re.sub(
        r"# (?:Final submission: train on full data and predict test|Final submission training and prediction)[\s\S]*?(?=# Metrics)",
        replacement,
        code,
        count=1,
    )


def apply_light_autofixes(code: str, spec: dict[str, object]) -> str:
    fixed = base.apply_light_autofixes(code, spec)
    fixed = _canonicalize_dataset_section(fixed, spec)
    fixed = _ensure_probability_and_metrics(fixed)
    return _canonicalize_final_submission_section(fixed, spec)


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
