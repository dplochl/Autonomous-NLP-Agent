import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

# Environment variables
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Constants
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/cnn_20260514_140549/run_001/submission.csv"
device = torch.device("cpu")

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Preprocessing
train_df['text'] = train_df['keyword'].fillna('') + ' [SEP] ' + train_df['text']
test_df['text'] = test_df['keyword'].fillna('') + ' [SEP] ' + test_df['text']

if DRY_RUN:
    train_df = train_df.head(200)

# Sample training data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Tokenizer and vocabulary
def build_vocab(texts, max_vocab):
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(max_vocab - 1))}
    vocab['<PAD>'] = max_vocab - 1
    return vocab

spec = {
    "architecture": "CNN",
    "batch_size": 32,
    "channels": 64,
    "dropout": 0.3,
    "dry_run_head": 200,
    "embedding_dim": 128,
    "epochs": 3,
    "experiment_name": "cnn_20260514_140549_run_01",
    "kernel_sizes": [3, 4, 5],
    "learning_rate": 0.001,
    "max_len": 48,
    "max_vocab": 20000,
    "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/cnn_20260514_140549/run_001/submission.csv",
    "threshold_max": 0.7,
    "threshold_min": 0.3,
    "threshold_steps": 41,
    "val_size": 0.2
}
vocab = build_vocab(train_df['text'], spec["max_vocab"])

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<PAD>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_df['sequence'] = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec["max_len"]))
test_df['sequence'] = test_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec["max_len"]))

# Dataset and DataLoader
class TweetDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_sequences = train_df['sequence'].tolist()
train_labels = train_df['target'].tolist()

stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    train_sequences, train_labels, test_size=0.2, random_state=42, stratify=stratify_labels
)

train_dataset = TweetDataset(train_sequences, train_labels)
val_dataset = TweetDataset(val_sequences, val_labels)
test_dataset = TweetDataset(test_df['sequence'].tolist())

train_loader = DataLoader(train_dataset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)

# Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, channels, kernel_size=k) for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        conv_outputs = [self.pool(torch.relu(conv(x))).squeeze(-1) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, channels * len(kernel_sizes))
        x = self.dropout(x)
        logit = self.fc(x)
        return logit

model = TextCNN(
    vocab_size=len(vocab),
    embedding_dim=spec["embedding_dim"],
    channels=spec["channels"],
    kernel_sizes=spec["kernel_sizes"],
    dropout=spec["dropout"]
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=spec["learning_rate"])

# Training
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(sequences).squeeze()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * sequences.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences).squeeze()
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()).tolist())
    return f1_score(all_labels, all_preds), accuracy_score(all_labels, all_preds)

if not DRY_RUN:
    for epoch in range(spec["epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_f1, val_acc = evaluate(model, val_loader)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}')

# Choose best threshold
best_threshold = 0.5
best_val_f1 = 0

for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    val_preds = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences).squeeze()
            preds = torch.sigmoid(logits) > threshold
            val_preds.extend(preds.cpu().numpy())
    val_f1 = f1_score(val_labels, val_preds)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_threshold = threshold

print(f'Best Threshold: {best_threshold:.4f}, Best Val F1: {best_val_f1:.4f}')

# Final submission
if FINAL_SUBMISSION:
    final_model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=spec["embedding_dim"],
        channels=spec["channels"],
        kernel_sizes=spec["kernel_sizes"],
        dropout=spec["dropout"]
    ).to(device)
    
    final_optimizer = optim.Adam(final_model.parameters(), lr=spec["learning_rate"])
    
    # Refit on all training data
    for epoch in range(spec["epochs"]):
        train_epoch(final_model, train_loader, criterion, final_optimizer)

# Predict test set
test_preds = []
with torch.no_grad():
    for sequences in test_loader:
        sequences = sequences.to(device)
        logits = final_model(sequences).squeeze()
        preds = torch.sigmoid(logits) > best_threshold
        test_preds.extend(preds.cpu().numpy())

# Write submission
if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

# Metrics
val_f1, val_acc = evaluate(final_model, val_loader)
print('METRICS: {"f1": ' + str(round(val_f1, 4)) + ', "accuracy": ' + str(round(val_acc, 4)) + '}')