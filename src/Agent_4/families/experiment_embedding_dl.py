"""Prompt-first learned/pretrained embedding deep-text family hook for Agent_3."""

from __future__ import annotations

import re
import textwrap

from families.autofix_utils import fix_text_column_fillna, force_cpu_execution


FAMILY = "EmbeddingDL"


def default_max_runs() -> int:
    return 4


def freeze_after_first_success() -> bool:
    return True


def _replace_assignment(code: str, name: str, value: str) -> str:
    return re.sub(
        rf"(?m)^({re.escape(name)}\s*=\s*)([^\n#]+)",
        rf"\g<1>{value}",
        code,
    )


def normalize_spec(spec: dict[str, object]) -> dict[str, object]:
    normalized = dict(spec)
    source = str(normalized.get("embedding_source", "learned")).strip().lower()
    normalized["embedding_source"] = "glove" if source == "glove" else "learned"
    normalized["glove_path"] = str(normalized.get("glove_path", "data/glove.6B.100d.txt"))
    normalized["embedding_dim"] = int(normalized.get("embedding_dim", 100))
    if normalized["embedding_source"] == "glove":
        allowed_dims = (50, 100, 200, 300)
        requested_dim = normalized["embedding_dim"]
        normalized["embedding_dim"] = min(allowed_dims, key=lambda dim: abs(dim - requested_dim))
    return normalized


def tune_frozen_code(code: str, spec: dict[str, object], run_name: str) -> str:
    fixed = code
    fixed = _replace_assignment(fixed, "max_vocab", str(int(spec["max_vocab"])))
    fixed = _replace_assignment(fixed, "max_len", str(int(spec["max_len"])))
    fixed = _replace_assignment(fixed, "embedding_dim", str(int(spec["embedding_dim"])))
    fixed = _replace_assignment(fixed, "hidden_dim", str(int(spec["hidden_dim"])))
    fixed = _replace_assignment(fixed, "dropout", repr(float(spec["dropout"])))
    fixed = _replace_assignment(fixed, "batch_size", str(int(spec["batch_size"])))
    fixed = _replace_assignment(fixed, "epochs", str(int(spec["epochs"])))
    fixed = _replace_assignment(fixed, "learning_rate", repr(float(spec["learning_rate"])))
    fixed = _replace_assignment(fixed, "VAL_SIZE", repr(float(spec["val_size"])))
    fixed = re.sub(
        r"embedding_source\s*=\s*['\"][^'\"]+['\"]",
        f"embedding_source = {str(spec['embedding_source'])!r}",
        fixed,
    )
    fixed = re.sub(
        r"glove_path\s*=\s*['\"][^'\"]+['\"]",
        f"glove_path = {str(spec['glove_path'])!r}",
        fixed,
    )
    fixed = re.sub(
        r"thresholds\s*=\s*np\.linspace\([^)]*\)",
        f"thresholds = np.linspace({float(spec['threshold_min'])}, {float(spec['threshold_max'])}, {int(spec['threshold_steps'])})",
        fixed,
        count=1,
    )
    fixed = re.sub(
        r"train_df\s*=\s*train_df\.head\(\d+\)",
        f"train_df = train_df.head({int(spec['dry_run_head'])})",
        fixed,
    )
    fixed = re.sub(
        r"submission_path\s*=\s*(['\"]).*?submission\.csv\1",
        f"submission_path = {spec['submission_path']!r}",
        fixed,
    )
    return fixed


def fallback_code(spec: dict[str, object]) -> str:
    max_vocab = int(spec.get("max_vocab", 25000))
    max_len = int(spec.get("max_len", 72))
    embedding_dim = int(spec.get("embedding_dim", 100))
    hidden_dim = int(spec.get("hidden_dim", 128))
    dropout = float(spec.get("dropout", 0.35))
    batch_size = int(spec.get("batch_size", 64))
    epochs = int(spec.get("epochs", 4))
    learning_rate = float(spec.get("learning_rate", 0.001))
    val_size = float(spec.get("val_size", 0.2))
    threshold_min = float(spec.get("threshold_min", 0.3))
    threshold_max = float(spec.get("threshold_max", 0.7))
    threshold_steps = int(spec.get("threshold_steps", 41))
    dry_run_head = int(spec.get("dry_run_head", 200))
    embedding_source = str(spec.get("embedding_source", "learned"))
    glove_path = str(spec.get("glove_path", "data/glove.6B.100d.txt"))
    submission_path = str(spec.get("submission_path", "submissions/submission.csv"))

    return textwrap.dedent(
        f"""\
        # AGENT3_FALLBACK
        import os
        import random
        import re
        from collections import Counter

        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, Dataset

        DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
        DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
        WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
        FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
        TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
        SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))
        VAL_SIZE = {val_size}

        max_vocab = {max_vocab}
        max_len = {max_len}
        embedding_dim = {embedding_dim}
        hidden_dim = {hidden_dim}
        dropout = {dropout}
        batch_size = {batch_size}
        epochs = {epochs}
        learning_rate = {learning_rate}
        embedding_source = {embedding_source!r}
        glove_path = os.environ.get("GLOVE_PATH", {glove_path!r})
        submission_path = {submission_path!r}

        def seed_everything(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        seed_everything(42)
        device = torch.device("cpu")
        token_pattern = re.compile(r"[a-z0-9#@']+")

        def tokenize(text):
            return token_pattern.findall(str(text).lower())

        def build_vocab(texts, max_vocab):
            counter = Counter()
            for text in texts:
                counter.update(tokenize(text))
            vocab = {{"<pad>": 0, "<unk>": 1}}
            for token, _count in counter.most_common(max_vocab - 2):
                vocab[token] = len(vocab)
            return vocab

        def text_to_sequence(text, vocab, max_len):
            ids = [vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
            if len(ids) < max_len:
                ids.extend([0] * (max_len - len(ids)))
            return ids

        def load_glove_matrix(vocab, path, embedding_dim):
            if not path or not os.path.exists(path):
                print("GLOVE_STATUS: missing; using learned embeddings")
                return None
            matrix = np.random.normal(0, 0.05, size=(len(vocab), embedding_dim)).astype("float32")
            matrix[0] = 0.0
            hits = 0
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.rstrip().split(" ")
                    if len(parts) != embedding_dim + 1:
                        continue
                    token = parts[0]
                    idx = vocab.get(token)
                    if idx is None:
                        continue
                    matrix[idx] = np.asarray(parts[1:], dtype="float32")
                    hits += 1
            print(f"GLOVE_STATUS: loaded hits={{hits}} vocab={{len(vocab)}} path={{path}}")
            return torch.tensor(matrix, dtype=torch.float32)

        class TextDataset(Dataset):
            def __init__(self, texts, labels, vocab):
                self.sequences = [text_to_sequence(text, vocab, max_len) for text in texts]
                self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, idx):
                x = torch.tensor(self.sequences[idx], dtype=torch.long)
                if self.labels is None:
                    return x
                return x, torch.tensor(self.labels[idx], dtype=torch.float32)

        class EmbeddingTextClassifier(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, embedding_weights=None):
                super().__init__()
                if embedding_weights is not None:
                    self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False, padding_idx=0)
                    embedding_dim = embedding_weights.shape[1]
                else:
                    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.encoder = nn.GRU(
                    embedding_dim,
                    hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim * 2, 1)

            def forward(self, x):
                emb = self.embedding(x)
                output, _hidden = self.encoder(emb)
                mask = (x != 0).unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                return self.fc(self.dropout(pooled)).squeeze(1)

        def prepare_frames():
            train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
            test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
            for df in (train_df, test_df):
                for col in ("keyword", "location", "text"):
                    if col in df.columns:
                        df[col] = df[col].fillna("").astype(str)
            train_df["text"] = np.where(
                train_df["keyword"].astype(str).str.len() > 0,
                train_df["keyword"].astype(str) + " [SEP] " + train_df["text"].astype(str),
                train_df["text"].astype(str),
            )
            test_df["text"] = np.where(
                test_df["keyword"].astype(str).str.len() > 0,
                test_df["keyword"].astype(str) + " [SEP] " + test_df["text"].astype(str),
                test_df["text"].astype(str),
            )
            if DRY_RUN:
                train_df = train_df.head({dry_run_head})
            elif TRAIN_FRACTION < 1.0:
                train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)
            return train_df, test_df

        def fit_model(model, train_loader, epochs):
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            criterion = nn.BCEWithLogitsLoss()
            if DRY_RUN:
                return model
            model.train()
            for epoch in range(epochs):
                losses = []
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                print(f"EPOCH_END: {{epoch + 1}} loss={{np.mean(losses):.4f}}")
            return model

        def predict_probs(model, loader):
            model.eval()
            probs = []
            with torch.no_grad():
                for batch in loader:
                    xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                    logits = model(xb.to(device))
                    probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            return np.asarray(probs)

        train_df, test_df = prepare_frames()
        y = train_df["target"]
        stratify_labels = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_df["text"],
            y,
            test_size=VAL_SIZE,
            random_state=42,
            stratify=stratify_labels,
        )
        train_texts = list(train_texts)
        val_texts = list(val_texts)
        train_labels = np.asarray(train_labels)
        val_labels = np.asarray(val_labels)

        vocab = build_vocab(train_texts, max_vocab=max_vocab)
        embedding_weights = load_glove_matrix(vocab, glove_path, embedding_dim) if embedding_source == "glove" else None
        train_dataset = TextDataset(train_texts, train_labels, vocab)
        val_dataset = TextDataset(val_texts, val_labels, vocab)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = EmbeddingTextClassifier(len(vocab), embedding_dim, hidden_dim, dropout, embedding_weights).to(device)
        model = fit_model(model, train_loader, epochs)
        val_probs = predict_probs(model, val_loader)

        thresholds = np.linspace({threshold_min}, {threshold_max}, {threshold_steps})
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in thresholds:
            preds = (val_probs > threshold).astype(int)
            score = f1_score(val_labels, preds)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        val_preds = (val_probs > best_threshold).astype(int)
        f1 = f1_score(val_labels, val_preds)
        acc = accuracy_score(val_labels, val_preds)

        final_model = model
        if FINAL_SUBMISSION:
            full_texts = list(train_df["text"])
            full_labels = np.asarray(train_df["target"])
            vocab = build_vocab(full_texts, max_vocab=max_vocab)
            embedding_weights = load_glove_matrix(vocab, glove_path, embedding_dim) if embedding_source == "glove" else None
            full_dataset = TextDataset(full_texts, full_labels, vocab)
            full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
            final_model = EmbeddingTextClassifier(len(vocab), embedding_dim, hidden_dim, dropout, embedding_weights).to(device)
            final_model = fit_model(final_model, full_loader, epochs)

        if WRITE_SUBMISSION:
            test_dataset = TextDataset(list(test_df["text"]), None, vocab)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_probs = predict_probs(final_model, test_loader)
            test_preds = (test_probs > best_threshold).astype(int)
            submission_dir = os.path.dirname(submission_path)
            if submission_dir:
                os.makedirs(submission_dir, exist_ok=True)
            pd.DataFrame({{"id": test_df["id"], "target": test_preds}}).to_csv(submission_path, index=False)

        print('METRICS: {{"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}}')
        """
    )


def get_default_spec(name: str, submission_path: str) -> dict[str, object]:
    return {
        "architecture": FAMILY,
        "embedding_source": "learned",
        "glove_path": "data/glove.6B.100d.txt",
        "max_vocab": 25000,
        "max_len": 72,
        "embedding_dim": 100,
        "hidden_dim": 128,
        "dropout": 0.35,
        "batch_size": 64,
        "epochs": 4,
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
        "max_len": (24, 160),
        "embedding_dim": (50, 300),
        "hidden_dim": (64, 256),
        "dropout": (0.1, 0.6),
        "batch_size": (16, 128),
        "epochs": (2, 8),
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
    return [
        "embedding_source",
        "max_vocab",
        "max_len",
        "embedding_dim",
        "hidden_dim",
        "dropout",
        "batch_size",
        "epochs",
        "learning_rate",
    ]


def get_template_name() -> str:
    return "train_embedding_dl.py.j2"


def get_arch_prompt() -> str:
    return (
        "Compare deep text classifiers with learned embeddings versus optional local GloVe initialization. "
        "Use a compact PyTorch embedding encoder with a bidirectional GRU and treat embedding_source as "
        "the main learned-vs-pretrained switch."
    )


def get_spec_prompt() -> str:
    return (
        "Return a reliable deep-learning text spec for learned embeddings versus local GloVe. "
        "Use embedding_source='learned' unless a local GloVe path is likely available; never download embeddings."
    )


def get_search_prompt() -> str:
    return (
        "Search locally around the best embedding deep-learning settings. Try embedding_source learned versus glove "
        "when useful, and otherwise adjust max length, hidden size, dropout, learning rate, batch size, and epochs."
    )


def get_repair_prompt() -> str:
    return (
        "Patch only the broken part of the embedding deep-learning script. "
        "Keep PyTorch, one validation split, threshold tuning, and learned/GloVe embedding support."
    )


def preflight_issues(code: str, spec: dict[str, object]) -> list[str]:
    issues = []
    required_patterns = [
        (r"(?:torch\.nn|nn)\.Embedding", "Missing required element: Embedding layer."),
        (r"(?:GRU|LSTM|Conv1d)", "Missing required deep text encoder: GRU, LSTM, or Conv1d."),
        (
            r"(?:from\s+torch\.utils\.data\s+import\s+.*Dataset|torch\.utils\.data\.Dataset|class\s+\w+\(Dataset\))",
            "Missing required element: Dataset definition.",
        ),
        (r"(?:from\s+torch\.utils\.data\s+import\s+.*DataLoader|\bDataLoader\()", "Missing required element: DataLoader."),
        (r"train_test_split\(", "Missing required element: train_test_split."),
        (r"embedding_source", "Missing required element: embedding_source switch."),
        (r"glove", "Missing required element: GloVe support or fallback."),
        (r"METRICS:", "Missing required element: METRICS output."),
    ]
    for pattern, message in required_patterns:
        if not re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    banned = [
        (r"\bTrainer\b|\bAutoModel\b|\bAutoTokenizer\b|\btransformers\b", "EmbeddingDL should not use Hugging Face transformers."),
        (r"\btensorflow\b|\bkeras\b", "EmbeddingDL must use PyTorch only."),
        (r"\brequests\.|\burllib|\bwget\b", "Do not download GloVe or any other external asset."),
    ]
    for pattern, message in banned:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    return issues


def apply_light_autofixes(code: str, spec: dict[str, object]) -> str:
    if "AGENT3_FALLBACK" in code:
        return code
    lowered = code.lower()
    has_embedding_switch = "embedding_source" in lowered and "embedding_source ==" in lowered
    has_glove_path = "glove_path" in lowered
    has_glove_loader = "load_glove" in lowered or "from_pretrained" in lowered
    if not (has_embedding_switch and has_glove_path and has_glove_loader):
        return fallback_code(spec)
    fixed = force_cpu_execution(fix_text_column_fillna(code))
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
        "submission_df.to_csv(submission_path, index=False)",
        "os.makedirs(os.path.dirname(submission_path), exist_ok=True)\nsubmission_df.to_csv(submission_path, index=False)",
    )
    return fixed


def build_repair_hint(stderr_text: str) -> str:
    return (
        "\nEmbeddingDL repair target:\n"
        "- keep a PyTorch embedding-based deep text classifier\n"
        "- keep embedding_source learned versus glove, with local-file-only GloVe fallback\n"
        "- keep one validation split\n"
        "- keep threshold tuning and METRICS output\n"
    )
