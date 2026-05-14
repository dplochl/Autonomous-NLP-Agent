# AGENT4_FALLBACK
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
VAL_SIZE = 0.2

max_vocab = 25000
max_len = 72
embedding_dim = 100
hidden_dim = 128
dropout = 0.35
batch_size = 64
epochs = 4
learning_rate = 0.001
embedding_source = 'learned'
glove_path = os.environ.get("GLOVE_PATH", '')
submission_path = '/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/embedding_dl_20260514_145010/run_001/submission.csv'

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
    vocab = {"<pad>": 0, "<unk>": 1}
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
    print(f"GLOVE_STATUS: loaded hits={hits} vocab={len(vocab)} path={path}")
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
        train_df = train_df.head(200)
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
        print(f"EPOCH_END: {epoch + 1} loss={np.mean(losses):.4f}")
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

thresholds = np.linspace(0.3, 0.7, 41)
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
    pd.DataFrame({"id": test_df["id"], "target": test_preds}).to_csv(submission_path, index=False)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')
