import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configuration
spec = {
    "architecture": "LSTM",
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
    "experiment_name": "lstm_20260514_130652_run_01",
    "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/lstm_20260514_130652/run_001/submission.csv"
}

# Environment variables
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Data directory
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")

# Device configuration
device = torch.device("cpu")

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Fill missing values
# Build text field
train_df['text'] = train_df['keyword'] + " [SEP] " + train_df['text']
test_df['text'] = test_df['keyword'] + " [SEP] " + test_df['text']

# DRY_RUN handling
if DRY_RUN:
    train_df = train_df.head(spec["dry_run_head"])

# Random sampling if needed
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Stratify labels
stratify_labels = train_df['target'] if len(train_df['target'].unique()) > 1 else None

# Train-test split
train_texts, val_texts, y_train, y_val = train_test_split(
    train_df['text'], train_df['target'], test_size=spec["val_size"], random_state=42, stratify=stratify_labels
)

# Tokenizer and vocabulary
def build_vocab(texts):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text in texts:
        words = text.split()
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(train_texts)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    sequence = sequence[:max_len]
    sequence += [vocab['<PAD>']] * (max_len - len(sequence))
    return sequence

train_sequences = np.array([text_to_sequence(text, vocab, spec["max_len"]) for text in train_texts])
val_sequences = np.array([text_to_sequence(text, vocab, spec["max_len"]) for text in val_texts])

# Dataset and DataLoader
class TextDataset(Dataset):
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

train_dataset = TextDataset(train_sequences, y_train)
val_dataset = TextDataset(val_sequences, y_val)

train_loader = DataLoader(train_dataset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)

# Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return torch.sigmoid(out)

model = LSTMClassifier(len(vocab), spec["embedding_dim"], spec["hidden_dim"], spec["num_layers"], spec["dropout"]).to(device)

# Training
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=spec["learning_rate"])

if not DRY_RUN:
    for epoch in range(spec["epochs"]):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Validation
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
for threshold in np.linspace(0.3, 0.7, 41):
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
            token_ids = text_to_sequence(text, vocab, 64)
            return torch.tensor(token_ids, dtype=torch.long)

    test_loader = DataLoader(_UnlabeledTextDataset(test_df['text']), batch_size=64, shuffle=False, pin_memory=False)
    if FINAL_SUBMISSION:
        full_train_dataset = DisasterDataset(train_sequences + val_seq, train_label + val_label)
        full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True, pin_memory=False)
        train_model(model, full_train_loader, optimizer, criterion, 3)

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
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
    submission_df.to_csv(submission_path, index=False)

# Metrics
val_preds = (val_probs > best_threshold).astype(int)
f1 = f1_score(val_labels, val_preds)
acc = accuracy_score(val_labels, val_preds)
# Metrics
y_pred_val = (np.array(val_preds) > best_threshold).astype(int)
f1 = f1_score(y_val, y_pred_val)
acc = accuracy_score(y_val, y_pred_val)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')