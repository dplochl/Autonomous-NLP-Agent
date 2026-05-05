import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

# Configuration
spec = {
    "architecture": "CNN",
    "max_vocab": 20000,
    "max_len": 48,
    "embedding_dim": 128,
    "channels": 64,
    "kernel_sizes": [3, 4, 5],
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 0.001,
    "val_size": 0.2,
    "threshold_min": 0.3,
    "threshold_max": 0.7,
    "threshold_steps": 41,
    "dry_run_head": 200,
    "experiment_name": "cnn_20260505_160840_run_01",
    "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260505_160840/run_001/submission.csv"
}

# Environment variables
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Data directory
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Fill missing values
# Build text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# DRY_RUN
if DRY_RUN:
    train_df = train_df.head(spec["dry_run_head"])

# Sample training data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Tokenizer and vocabulary
def build_vocab(texts, max_vocab):
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common(max_vocab - 1))}
    return vocab

vocab = build_vocab(train_df['text'], spec["max_vocab"])

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 0) for word in text.split()][:max_len][:max_len]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))

train_df['sequence'] = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec["max_len"]))
test_df['sequence'] = test_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec["max_len"]))

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx] if self.labels is not None else None
        return sequence, label

train_sequences = train_df['sequence'].tolist()
train_labels = train_df['target'].tolist()

stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_seq, val_seq, train_label, val_label = train_test_split(train_sequences, train_labels, test_size=spec["val_size"], random_state=42, stratify=stratify_labels)

train_dataset = TextDataset(train_seq, train_label)
val_dataset = TextDataset(val_seq, val_label)
test_dataset = TextDataset(test_df['sequence'].tolist())

train_loader = DataLoader(train_dataset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)

# Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        x = [torch.relu(conv(x)) for conv in self.convs]
        x = [torch.max_pool1d(cx, cx.size(2)).squeeze(2) for cx in x]  # Global max pooling
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x).sigmoid()

model = TextCNN(spec["max_vocab"], spec["embedding_dim"], spec["channels"], spec["kernel_sizes"], spec["dropout"])
device = torch.device("cpu")
model.to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
criterion = nn.BCELoss()

def train_epoch(model, loader, optimizer):
    model.train()
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

if not DRY_RUN:
    for epoch in range(spec["epochs"]):
        train_epoch(model, train_loader, optimizer)

# Validation
def validate(model, loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for sequences, _ in loader:
            sequences = sequences.to(device)
            outputs = model(sequences).squeeze().cpu().numpy()
            all_preds.extend(outputs)
    return np.array(all_preds)

val_probs = validate(model, val_loader)

# Choose best threshold
best_threshold = None
best_f1 = 0

for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(val_label, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f'Best threshold: {best_threshold}, Best F1: {best_f1}')

# Final submission
if FINAL_SUBMISSION:
    final_model = TextCNN(spec["max_vocab"], spec["embedding_dim"], spec["channels"], spec["kernel_sizes"], spec["dropout"])
    final_model.to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=spec["learning_rate"])
    
    full_train_dataset = TextDataset(train_sequences + val_seq, train_label + val_label)
    full_train_loader = DataLoader(full_train_dataset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)
    
    for epoch in range(spec["epochs"]):
        train_epoch(final_model, full_train_loader, optimizer)
    
    test_probs = validate(final_model, test_loader)
    test_preds = (test_probs >= best_threshold).astype(int)

if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
    submission_df.to_csv(spec["submission_path"], index=False)

# Metrics
val_preds = (val_probs >= best_threshold).astype(int)
acc = accuracy_score(val_label, val_preds)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')