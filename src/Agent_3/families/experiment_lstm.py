"""Prompt-first LSTM family hook for Agent_3."""

from __future__ import annotations

import re

from families.autofix_utils import fix_text_column_fillna


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
    fixed = fix_text_column_fillna(code)
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
        "val_preds.extend(outputs.numpy())",
        "val_preds.extend(np.atleast_1d(outputs.numpy()))",
    )
    fixed = fixed.replace(
        "test_preds.extend(outputs.numpy())",
        "test_preds.extend(np.atleast_1d(outputs.numpy()))",
    )
    fixed = re.sub(
        r"(\n[ \t]*)for (\w+) in ((?:val|test)_loader):\n(?![ \t]*if isinstance)",
        r"\1for \2 in \3:\n\1    if isinstance(\2, (list, tuple)):\n\1        \2 = \2[0]\n",
        fixed,
    )
    if "final_model.eval()" in fixed and "final_model = model" not in fixed:
        fixed = fixed.replace("\n# Final submission\n", "\n# Final submission\nfinal_model = model\n", 1)
    fixed = fixed.replace(
        "submission_df.to_csv(submission_path, index=False)",
        "os.makedirs(os.path.dirname(submission_path), exist_ok=True)\nsubmission_df.to_csv(submission_path, index=False)",
    )
    return fixed


def build_repair_hint(stderr_text: str) -> str:
    return (
        "\nLSTM repair target:\n"
        "- keep PyTorch embedding + bidirectional LSTM architecture\n"
        "- keep one validation split\n"
        "- keep threshold tuning and METRICS output\n"
    )
