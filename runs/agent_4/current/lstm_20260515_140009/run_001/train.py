import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Configuration
spec = {
    "architecture": "LSTM",
    "max_vocab": 14375,
    "max_len": 64,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.2,
    "batch_size": 64,
    "epochs": 3,
    "learning_rate": 0.001,
    "val_size": 0.2,
    "threshold_min": 0.3,
    "threshold_max": 0.7,
    "threshold_steps": 41,
    "dry_run_head": 200,
    "experiment_name": "lstm_20260515_140009_run_01",
    "submission_path": "/Users/niccogermani/Library/Containers/com.apple.iMovieApp/Data/Documents/Catolica/apa-disaster-tweets-clean/src/Agent_4/runs/lstm_20260515_140009/run_001/submission.csv"
}

# Environment variables
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))
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
train_df["text_combined"] = train_df["keyword"] + " [SEP] " + train_df["text"]
test_df["text_combined"] = test_df["keyword"] + " [SEP] " + test_df["text"]

# DRY_RUN
if DRY_RUN:
    train_df = train_df.head(spec["dry_run_head"])

# Sample training data if needed
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Tokenizer and vocabulary
def build_vocab(texts, max_vocab):
    word_counts = {}
    for text in texts:
        words = text.split()
        for word in words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: i + 2 for i, (word, _) in enumerate(sorted_words[:max_vocab - 2])}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

vocab = build_vocab(train_df["text_combined"], spec["max_vocab"])

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab["<UNK>"]) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab["<PAD>"]] * (max_len - len(sequence))

train_df["sequence"] = train_df["text_combined"].apply(lambda x: text_to_sequence(x, vocab, spec["max_len"]))
test_df["sequence"] = test_df["text_combined"].apply(lambda x: text_to_sequence(x, vocab, spec["max_len"]))

# Dataset and DataLoader
class DisasterDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_sequences = train_df["sequence"].tolist()
train_labels = train_df["target"].tolist()

stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    train_sequences, train_labels, test_size=spec["val_size"], random_state=42, stratify=stratify_labels
)

train_dataset = DisasterDataset(train_sequences, train_labels)
val_dataset = DisasterDataset(val_sequences, val_labels)
test_dataset = DisasterDataset(test_df["sequence"].tolist())

train_loader = DataLoader(train_dataset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)

# Model
class BiLSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
                        x = torch.tensor(x, dtype=torch.long).to(device)
embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

device = torch.device("cpu")
model = BiLSTMClassifier(spec["max_vocab"], spec["embedding_dim"], spec["hidden_dim"], spec["num_layers"], spec["dropout"]).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
criterion = torch.nn.BCEWithLogitsLoss()

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
model.eval()
val_probs = []
with torch.no_grad():
    for sequences in val_loader:
        
        outputs = model(sequences).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        val_probs.extend(probs)

best_threshold = 0.5
best_f1 = 0

for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    preds = (np.array(val_probs) >= threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

acc = accuracy_score(val_labels, (np.array(val_probs) >= best_threshold).astype(int))

# Final submission
if FINAL_SUBMISSION:
    full_train_dataset = DisasterDataset(train_df["sequence"].tolist(), train_df["target"].tolist())
    full_train_loader = DataLoader(full_train_dataset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)
    
    model = BiLSTMClassifier(spec["max_vocab"], spec["embedding_dim"], spec["hidden_dim"], spec["num_layers"], spec["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(spec["epochs"]):
        model.train()
        for sequences, labels in full_train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    test_probs = []
    with torch.no_grad():
        for sequences in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_probs.extend(probs)

# Write submission
if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
    submission_df = pd.DataFrame({"id": test_df["id"], "target": (np.array(test_probs) >= best_threshold).astype(int)})
    submission_df.to_csv(spec["submission_path"], index=False)

print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')