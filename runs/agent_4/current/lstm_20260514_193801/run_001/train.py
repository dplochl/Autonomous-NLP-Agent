import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

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
    "experiment_name": "lstm_20260514_193801_run_01",
    "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/lstm_20260514_193801/run_001/submission.csv"
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
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Fill missing values
train_df["keyword"] = train_df["keyword"].fillna("")
train_df["location"] = train_df["location"].fillna("")
train_df["text"] = train_df["text"].fillna("")

test_df["keyword"] = test_df["keyword"].fillna("")
test_df["location"] = test_df["location"].fillna("")
test_df["text"] = test_df["text"].fillna("")

# Build text field
train_df["text"] = train_df["keyword"] + " [SEP] " + train_df["text"]
test_df["text"] = test_df["keyword"] + " [SEP] " + test_df["text"]

# DRY_RUN or sample data
if DRY_RUN:
    train_df = train_df.head(spec["dry_run_head"])
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    train_df["text"], train_df["target"],
    test_size=spec["val_size"], random_state=42,
    stratify=train_df["target"] if len(train_df["target"].unique()) > 1 else None
)

# Tokenizer and vocabulary
def build_vocab(texts, max_vocab):
    word_freq = {}
    for text in texts:
        words = text.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: i+2 for i, (word, _) in enumerate(sorted_words[:max_vocab-2])}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

vocab = build_vocab(train_df["text"], spec["max_vocab"])

def text_to_seq(text, vocab, max_len):
    words = text.split()
    seq = [vocab.get(word, vocab["<UNK>"]) for word in words]
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq + [vocab["<PAD>"]] * (max_len - len(seq))

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_seq = torch.tensor(text_to_seq(self.texts[idx], vocab, spec["max_len"]), dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return text_seq, label
        else:
            return text_seq

train_dataset = TextDataset(X_train.tolist(), y_train.tolist())
val_dataset = TextDataset(X_val.tolist(), y_val.tolist())

train_loader = DataLoader(train_dataset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)

# Model
class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return out.squeeze()

device = torch.device("cpu")
model = LSTMClassifier(len(vocab), spec["embedding_dim"], spec["hidden_dim"], spec["num_layers"], spec["dropout"]).to(device)

# Training
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if not DRY_RUN:
    train_model(model, train_loader, criterion, optimizer, spec["epochs"])

# Validation
def validate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

val_probs, val_labels = validate_model(model, val_loader)

# Tune threshold
best_threshold = 0.5
best_f1 = 0

for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

acc = accuracy_score(val_labels, (val_probs >= best_threshold).astype(int))

print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')

# Final submission
if WRITE_SUBMISSION:
    if FINAL_SUBMISSION:
        # Retrain on full train_df
        full_train_dataset = TextDataset(train_df["text"].tolist(), train_df["target"].tolist())
        full_train_loader = DataLoader(full_train_dataset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)
        train_model(model, full_train_loader, criterion, optimizer, spec["epochs"])

    # Predict test set
    test_dataset = TextDataset(test_df["text"].tolist())
    test_loader = DataLoader(test_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)

    def predict_test(model, test_loader):
        model.eval()
        all_preds = []
        with torch.no_grad():
            for texts in test_loader:
                texts = texts.to(device)
                outputs = model(texts)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
        return np.array(all_preds)

    test_probs = predict_test(model, test_loader)
    test_preds = (test_probs >= best_threshold).astype(int)

    # Write submission
    submission_dir = os.path.dirname(spec["submission_path"])
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "target": test_preds
    })
    submission_df.to_csv(spec["submission_path"], index=False)