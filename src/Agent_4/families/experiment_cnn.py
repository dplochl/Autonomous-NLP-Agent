"""Prompt-first CNN family hook for Agent_4."""

from __future__ import annotations

import re

from families.autofix_utils import fix_text_column_fillna, force_cpu_execution


FAMILY = "CNN"


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
    fixed = _replace_assignment(fixed, "channels", str(int(spec["channels"])))
    fixed = _replace_assignment(fixed, "dropout", repr(float(spec["dropout"])))
    fixed = _replace_assignment(fixed, "batch_size", str(int(spec["batch_size"])))
    fixed = _replace_assignment(fixed, "epochs", str(int(spec["epochs"])))
    fixed = _replace_assignment(fixed, "learning_rate", repr(float(spec["learning_rate"])))
    fixed = _replace_assignment(fixed, "VAL_SIZE", repr(float(spec["val_size"])))
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
        "max_vocab": 20000,
        "max_len": 48,
        "embedding_dim": 128,
        "channels": 128,
        "kernel_sizes": [3, 4, 5],
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
        "channels": (64, 256),
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
    return ["max_vocab", "max_len", "embedding_dim", "channels", "dropout", "batch_size", "epochs", "learning_rate"]


def get_template_name() -> str:
    return "train_cnn.py.j2"


def get_arch_prompt() -> str:
    return (
        "Use a PyTorch text CNN with an embedding layer, Conv1d blocks, pooling, and a sigmoid-style binary output. "
        "The embedding must always receive a tensor of token ids, never a Python list or a (inputs, labels) tuple."
    )


def get_spec_prompt() -> str:
    return (
        "Return a reliable text CNN spec with a single validation split and conservative sequence settings. "
        "Prefer simple settings that are likely to run on the first try. "
        "Validation loaders with labels must unpack batches as (inputs, labels), while test loaders must pass only inputs to the model."
    )


def get_search_prompt() -> str:
    return (
        "Search the local CNN parameter space around the best successful run. Adjust sequence length, channels, "
        "dropout, learning rate, or epochs instead of proposing a completely new design."
    )


def get_repair_prompt() -> str:
    return (
        "Patch only the broken part of the CNN script and keep the PyTorch CNN pipeline. "
        "Accept either torch.nn.* or nn.* module references. "
        "When fixing evaluation code, make sure labeled loaders unpack batches correctly and the embedding sees a tensor."
    )


def preflight_issues(code: str, spec: dict[str, object]) -> list[str]:
    issues = []
    required_patterns = [
        (r"(?:torch\.nn|nn)\.Embedding", "Missing required element: Embedding layer."),
        (r"(?:torch\.nn|nn)\.Conv1d", "Missing required element: Conv1d layer."),
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
        (r"\bTrainer\b|\bAutoModel\b|\bAutoTokenizer\b", "CNN should not use Hugging Face Trainer."),
        (r"\btensorflow\b|\bkeras\b", "CNN must use PyTorch only."),
    ]
    for pattern, message in banned:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    if "F." in code and "import torch.nn.functional as F" not in code:
        issues.append("Missing required import: torch.nn.functional as F.")
    if re.search(r"for\s+\w+\s+in\s+val_loader\s*:", code) and not re.search(r"for\s+\w+\s*,\s*\w+\s+in\s+val_loader\s*:", code):
        if "if isinstance(" not in code:
            issues.append("Validation loader likely returns (inputs, labels); unpack it or strip labels before model(...).")
    return issues


def apply_light_autofixes(code: str, spec: dict[str, object]) -> str:
    fixed = force_cpu_execution(fix_text_column_fillna(code))
    if "F." in fixed and "import torch.nn.functional as F" not in fixed:
        fixed = fixed.replace("import torch.nn as nn\n", "import torch.nn as nn\nimport torch.nn.functional as F\n", 1)
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
        "val_probs.extend(probs)",
        "val_probs.extend(np.atleast_1d(probs))",
    )
    fixed = fixed.replace(
        "test_probs.extend(probs)",
        "test_probs.extend(np.atleast_1d(probs))",
    )
    fixed = fixed.replace(
        "all_probs.extend(probs)",
        "all_probs.extend(np.atleast_1d(probs))",
    )
    fixed = fixed.replace(
        "val_preds.extend(preds)",
        "val_preds.extend(np.atleast_1d(preds))",
    )
    fixed = fixed.replace(
        "test_preds.extend(preds)",
        "test_preds.extend(np.atleast_1d(preds))",
    )
    fixed = fixed.replace(
        "all_preds.extend(preds)",
        "all_preds.extend(np.atleast_1d(preds).tolist())",
    )
    fixed = fixed.replace(
        "all_labels.extend(labels.cpu().numpy())",
        "all_labels.extend(np.atleast_1d(labels.cpu().numpy()).tolist())",
    )
    fixed = fixed.replace("x = x[0].to(device)", "x = x.to(device)")
    fixed = fixed.replace(
        "self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]",
        "self.labels[idx]",
    )
    fixed = fixed.replace(
        "val_labels = val_labels.to_numpy()\nval_labels = val_labels.numpy()\n",
        "val_labels = np.asarray(val_labels)\n",
    )
    fixed = re.sub(
        r"(?ms)\n(?:[ \t]*submission_df\s*=\s*pd\.DataFrame\(\{.*?submission_df\.to_csv\(submission_path,\s*index=False\)\n)+",
        "\n",
        fixed,
    )
    return _canonicalize_eval_and_submission(fixed, spec)


def _canonicalize_eval_and_submission(code: str, spec: dict[str, object]) -> str:
    epochs = int(spec["epochs"])
    threshold_min = float(spec["threshold_min"])
    threshold_max = float(spec["threshold_max"])
    threshold_steps = int(spec["threshold_steps"])
    max_vocab = int(spec["max_vocab"])
    embedding_dim = int(spec["embedding_dim"])
    channels = int(spec["channels"])
    dropout = float(spec["dropout"])
    batch_size = int(spec["batch_size"])
    learning_rate = float(spec["learning_rate"])
    val_block = f"""# Validation
def collect_probs_and_labels(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_probs.extend(np.atleast_1d(probs).tolist())
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
            if 'tokenize' in globals():
                token_ids = tokenize(text, vocab, {int(spec["max_len"])})
            else:
                token_ids = text_to_sequence(text, vocab, {int(spec["max_len"])})
            return torch.tensor(token_ids, dtype=torch.long)

    test_loader = DataLoader(_UnlabeledTextDataset(test_df['text']), batch_size={batch_size}, shuffle=False, pin_memory=False)
    if FINAL_SUBMISSION:
        final_model = TextCNN(vocab_size=len(vocab) + 1, embedding_dim={embedding_dim}, channels={channels}, kernel_sizes=[3, 4, 5], dropout={dropout}).to(device)
        optimizer_final = torch.optim.Adam(final_model.parameters(), lr={learning_rate})
        criterion_final = nn.BCEWithLogitsLoss()
        final_train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, pin_memory=False)
        for epoch in range({epochs}):
            final_model.train()
            for inputs, labels in final_train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer_final.zero_grad()
                outputs = final_model(inputs).squeeze()
                loss = criterion_final(outputs, labels)
                loss.backward()
                optimizer_final.step()
        test_model = final_model
    else:
        test_model = model

    test_model.eval()
    test_probs = []
    with torch.no_grad():
        for inputs in test_loader:
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(device)
            outputs = test_model(inputs).squeeze()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            test_probs.extend(np.atleast_1d(probs).tolist())
    test_preds = (np.asarray(test_probs) > best_threshold).astype(int)

    submission_df = pd.DataFrame({{
        'id': test_df['id'],
        'target': test_preds
    }})
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

"""
    fixed = re.sub(
        r"# Validation[\s\S]*?(?=# Metrics)",
        val_block,
        code,
        count=1,
    )
    fixed = re.sub(
        r"TextCNN\(\s*SPEC\['max_vocab'\]\s*,\s*SPEC\['embedding_dim'\]\s*,\s*SPEC\['channels'\]\s*,\s*SPEC\['kernel_sizes'\]\s*,\s*SPEC\['dropout'\]\s*\)",
        f"TextCNN(vocab_size=len(vocab) + 1, embedding_dim={embedding_dim}, channels={channels}, kernel_sizes=[3, 4, 5], dropout={dropout})",
        fixed,
    )
    fixed = re.sub(
        r"TextCNN\(\s*spec\[[\"']max_vocab[\"']\]\s*,\s*spec\[[\"']embedding_dim[\"']\]\s*,\s*spec\[[\"']channels[\"']\]\s*,\s*spec\[[\"']kernel_sizes[\"']\]\s*,\s*spec\[[\"']dropout[\"']\]\s*\)",
        f"TextCNN(vocab_size=len(vocab) + 1, embedding_dim={embedding_dim}, channels={channels}, kernel_sizes=[3, 4, 5], dropout={dropout})",
        fixed,
    )
    return fixed


def build_repair_hint(stderr_text: str) -> str:
    return (
        "\nCNN repair target:\n"
        "- keep PyTorch embedding + Conv1d architecture\n"
        "- keep one validation split\n"
        "- keep threshold tuning and METRICS output\n"
        "- labeled validation batches may be tuples; pass only the input tensor into model(...)\n"
        "- if F.relu or F.max_pool1d is used, import torch.nn.functional as F\n"
    )
