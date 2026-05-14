"""Prompt-first LSTM family hook for Agent_4."""

from __future__ import annotations

import re

from families.autofix_utils import fix_text_column_fillna, force_cpu_execution


FAMILY = "LSTM"


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
    fixed = _replace_assignment(fixed, "max_vocab", str(int(spec["max_vocab"])))
    fixed = _replace_assignment(fixed, "max_len", str(int(spec["max_len"])))
    fixed = _replace_assignment(fixed, "embedding_dim", str(int(spec["embedding_dim"])))
    fixed = _replace_assignment(fixed, "hidden_dim", str(int(spec["hidden_dim"])))
    fixed = _replace_assignment(fixed, "num_layers", str(int(spec["num_layers"])))
    fixed = _replace_assignment(fixed, "dropout", repr(float(spec["dropout"])))
    fixed = _replace_assignment(fixed, "batch_size", str(int(spec["batch_size"])))
    fixed = _replace_assignment(fixed, "epochs", str(int(spec["epochs"])))
    fixed = _replace_assignment(fixed, "learning_rate", repr(float(spec["learning_rate"])))
    fixed = _replace_assignment(fixed, "VAL_SIZE", repr(float(spec["val_size"])))
    fixed = re.sub(
        r"build_vocab\(train_texts,\s*max_vocab=\d+\)",
        f"build_vocab(train_texts, max_vocab={int(spec['max_vocab'])})",
        fixed,
    )
    fixed = re.sub(
        r"text_to_sequence\(text,\s*vocab,\s*\d+\)",
        f"text_to_sequence(text, vocab, {int(spec['max_len'])})",
        fixed,
    )
    fixed = re.sub(
        r"train_loader\s*=\s*DataLoader\(\s*train_dataset,\s*batch_size=\d+",
        f"train_loader = DataLoader(train_dataset, batch_size={int(spec['batch_size'])}",
        fixed,
    )
    fixed = re.sub(
        r"val_loader\s*=\s*DataLoader\(\s*val_dataset,\s*batch_size=\d+",
        f"val_loader = DataLoader(val_dataset, batch_size={int(spec['batch_size'])}",
        fixed,
    )
    fixed = re.sub(
        r"test_loader\s*=\s*DataLoader\(\s*test_dataset,\s*batch_size=\d+",
        f"test_loader = DataLoader(test_dataset, batch_size={int(spec['batch_size'])}",
        fixed,
    )
    fixed = re.sub(
        r"model\s*=\s*LSTMClassifier\(\s*len\(vocab\)\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[0-9.]+\s*\)",
        f"model = LSTMClassifier(len(vocab), {int(spec['embedding_dim'])}, {int(spec['hidden_dim'])}, {int(spec['num_layers'])}, {float(spec['dropout'])})",
        fixed,
    )
    fixed = re.sub(
        r"optimizer\s*=\s*optim\.Adam\(model\.parameters\(\),\s*lr=[0-9.eE+-]+\)",
        f"optimizer = optim.Adam(model.parameters(), lr={float(spec['learning_rate'])})",
        fixed,
    )
    fixed = re.sub(
        r"for epoch in range\(\d+\):",
        f"for epoch in range({int(spec['epochs'])}):",
        fixed,
    )
    fixed = re.sub(
        r"thresholds\s*=\s*np\.linspace\([^)]*\)",
        f"thresholds = np.linspace({float(spec['threshold_min'])}, {float(spec['threshold_max'])}, {int(spec['threshold_steps'])})",
        fixed,
        count=1,
    )
    fixed = re.sub(r"train_df\s*=\s*train_df\.head\(\d+\)", f"train_df = train_df.head({int(spec['dry_run_head'])})", fixed)
    fixed = re.sub(
        r"os\.makedirs\(os\.path\.dirname\((['\"]).*?submission\.csv\1\),\s*exist_ok=True\)",
        f"os.makedirs(os.path.dirname({spec['submission_path']!r}), exist_ok=True)",
        fixed,
    )
    fixed = re.sub(
        r"submission_df\.to_csv\((['\"]).*?submission\.csv\1,\s*index=False\)",
        f"submission_df.to_csv({spec['submission_path']!r}, index=False)",
        fixed,
    )
    fixed = re.sub(
        r"submission_path\s*=\s*(['\"]).*?submission\.csv\1",
        f"submission_path = {spec['submission_path']!r}",
        fixed,
    )
    return fixed


def get_default_spec(name: str, submission_path: str) -> dict[str, object]:
    return {
        "architecture": FAMILY,
        "max_vocab": 20000,
        "max_len": 64,
        "embedding_dim": 128,
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.3,
        "batch_size": 64,
        "epochs": 3,
        "learning_rate": 0.001,
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
        "max_vocab": (5000, 50000),
        "max_len": (24, 128),
        "embedding_dim": (64, 256),
        "hidden_dim": (64, 256),
        "num_layers": (1, 3),
        "dropout": (0.1, 0.6),
        "batch_size": (16, 128),
        "epochs": (2, 3),
        "learning_rate": (0.0001, 0.01),
        "val_size": (0.1, 0.3),
        "threshold_min": (0.1, 0.6),
        "threshold_max": (0.4, 0.9),
        "threshold_steps": (11, 81),
        "dry_run_head": (50, 500),
    }


def get_fixed_spec_keys() -> set[str]:
    return {"architecture", "experiment_name", "submission_path"}


def get_tunable_keys() -> list[str]:
    return ["max_vocab", "max_len", "embedding_dim", "hidden_dim", "num_layers", "dropout", "batch_size", "epochs", "learning_rate"]


def get_template_name() -> str:
    return "train_lstm.py.j2"


def get_arch_prompt() -> str:
    return "Use a PyTorch bidirectional LSTM text classifier with an embedding layer and a compact classifier head."


def get_spec_prompt() -> str:
    return (
        "Return a reliable BiLSTM spec with one validation split and conservative sequence settings. "
        "Prefer simple settings that are likely to run on the first try."
    )


def get_search_prompt() -> str:
    return (
        "Search locally around the best successful LSTM settings. Adjust hidden size, max length, dropout, "
        "learning rate, and epochs with nearby changes."
    )


def get_repair_prompt() -> str:
    return (
        "Patch only the broken part of the LSTM script and keep the PyTorch recurrent pipeline. "
        "Accept either torch.nn.* or nn.* module references."
    )


def preflight_issues(code: str, spec: dict[str, object]) -> list[str]:
    issues = []
    required_patterns = [
        (r"(?:torch\.nn|nn)\.Embedding", "Missing required element: Embedding layer."),
        (r"(?:torch\.nn|nn)\.LSTM", "Missing required element: LSTM layer."),
        (r"bidirectional\s*=\s*True", "Missing required element: bidirectional=True."),
        (r"(?:from\s+torch\.utils\.data\s+import\s+.*Dataset|torch\.utils\.data\.Dataset|class\s+\w+\(Dataset\))",
         "Missing required element: Dataset definition."),
        (r"(?:from\s+torch\.utils\.data\s+import\s+.*DataLoader|\bDataLoader\()", "Missing required element: DataLoader."),
        (r"train_test_split\(", "Missing required element: train_test_split."),
        (r"METRICS:", "Missing required element: METRICS output."),
    ]
    for pattern, message in required_patterns:
        if not re.search(pattern, code):
            issues.append(message)
    banned = [
        (r"\bTrainer\b|\bAutoModel\b|\bAutoTokenizer\b", "LSTM should not use Hugging Face Trainer."),
        (r"\btensorflow\b|\bkeras\b", "LSTM must use PyTorch only."),
    ]
    for pattern, message in banned:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    return issues


def apply_light_autofixes(code: str, spec: dict[str, object]) -> str:
    fixed = force_cpu_execution(fix_text_column_fillna(code))
    fixed = fixed.replace(
        "sequence = [vocab.get(word, 0) for word in text.split()]",
        "sequence = [vocab.get(word, 0) for word in text.split()][:max_len]",
    )
    fixed = fixed.replace(
        "stratify_labels = train_df['target'] if len(train_df['target'].unique()) == 2 else None",
        "stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None",
    )
    fixed = re.sub(
        r"self\.labels\[idx\]",
        "self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]",
        fixed,
    )
    fixed = fixed.replace(
        "self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]",
        "self.labels[idx]",
    )
    if "train_labels = np.asarray(train_labels)" not in fixed:
        fixed = re.sub(
            r"(train_texts,\s*val_texts,\s*train_labels,\s*val_labels\s*=\s*train_test_split\([\s\S]*?\)\n)",
            r"\1train_labels = np.asarray(train_labels)\nval_labels = np.asarray(val_labels)\n",
            fixed,
            count=1,
        )
    fixed = fixed.replace(
        "val_probs.extend(outputs)",
        "val_probs.extend(np.atleast_1d(outputs))",
    )
    fixed = fixed.replace(
        "test_probs.extend(outputs)",
        "test_probs.extend(np.atleast_1d(outputs))",
    )
    fixed = fixed.replace(
        "val_preds.extend(outputs)",
        "val_preds.extend(np.atleast_1d(outputs).tolist())",
    )
    fixed = fixed.replace(
        "test_preds.extend(outputs)",
        "test_preds.extend(np.atleast_1d(outputs).tolist())",
    )
    fixed = fixed.replace(
        "val_preds.extend(outputs.numpy())",
        "val_preds.extend(np.atleast_1d(outputs.numpy()))",
    )
    fixed = fixed.replace(
        "test_preds.extend(outputs.numpy())",
        "test_preds.extend(np.atleast_1d(outputs.numpy()))",
    )
    fixed = fixed.replace(
        "return best_threshold",
        "return best_threshold, np.asarray(val_preds)",
    )
    fixed = fixed.replace(
        "best_threshold = train_model(model, train_loader, val_loader, epochs=3)",
        "best_threshold, val_preds = train_model(model, train_loader, val_loader, epochs=3)",
    )
    fixed = re.sub(
        r"(?ms)\n(?:[ \t]*submission_df\s*=\s*pd\.DataFrame\(\{.*?submission_df\.to_csv\(submission_path,\s*index=False\)\n)+",
        "\n",
        fixed,
    )
    return _canonicalize_eval_and_submission(fixed, spec)


def _canonicalize_eval_and_submission(code: str, spec: dict[str, object]) -> str:
    batch_size = int(spec["batch_size"])
    epochs = int(spec["epochs"])
    max_len = int(spec["max_len"])
    threshold_min = float(spec["threshold_min"])
    threshold_max = float(spec["threshold_max"])
    threshold_steps = int(spec["threshold_steps"])
    val_block = f"""# Validation
def collect_probs_and_labels(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            all_probs.extend(np.atleast_1d(outputs.detach().cpu().numpy()).tolist())
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()).tolist())
    return np.asarray(all_probs), np.asarray(all_labels)

val_probs, val_labels = collect_probs_and_labels(model, val_loader)

# Choose best threshold
best_threshold = 0.5
best_f1 = 0.0
for threshold in np.linspace({threshold_min}, {threshold_max}, {threshold_steps}):
    val_preds = (val_probs > threshold).astype(int)
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission
test_preds = np.array([], dtype=int)
if WRITE_SUBMISSION:
    class _UnlabeledTextDataset(Dataset):
        def __init__(self, texts):
            self.texts = list(texts)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            token_ids = text_to_sequence(text, vocab, {max_len})
            return torch.tensor(token_ids, dtype=torch.long)

    test_loader = DataLoader(_UnlabeledTextDataset(test_df['text']), batch_size={batch_size}, shuffle=False, pin_memory=False)
    if FINAL_SUBMISSION:
        full_train_dataset = DisasterDataset(train_sequences + val_seq, train_label + val_label)
        full_train_loader = DataLoader(full_train_dataset, batch_size={batch_size}, shuffle=True, pin_memory=False)
        train_model(model, full_train_loader, optimizer, criterion, {epochs})

    model.eval()
    test_probs = []
    with torch.no_grad():
        for sequences in test_loader:
            if isinstance(sequences, (list, tuple)):
                sequences = sequences[0]
            sequences = sequences.to(device)
            outputs = model(sequences)
            test_probs.extend(np.atleast_1d(outputs.detach().cpu().numpy()).tolist())
    test_preds = (np.asarray(test_probs) > best_threshold).astype(int)

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = pd.DataFrame({{'id': test_df['id'], 'target': test_preds}})
    submission_df.to_csv(submission_path, index=False)

# Metrics
val_preds = (val_probs > best_threshold).astype(int)
f1 = f1_score(val_labels, val_preds)
acc = accuracy_score(val_labels, val_preds)
"""
    return re.sub(
        r"# Validation[\s\S]*?(?=# Metrics)",
        val_block,
        code,
        count=1,
    )


def build_repair_hint(stderr_text: str) -> str:
    return (
        "\nLSTM repair target:\n"
        "- keep PyTorch embedding + bidirectional LSTM architecture\n"
        "- keep one validation split\n"
        "- keep threshold tuning and METRICS output\n"
    )
