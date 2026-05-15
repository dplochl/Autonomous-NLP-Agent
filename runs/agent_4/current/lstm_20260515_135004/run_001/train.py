import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load environment variables
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Define constants
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
device = torch.device("cpu")

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Preprocess data
train_df["keyword"] = train_df["keyword"].fillna("")
train_df["location"] = train_df["location"].fillna("")
train_df["text"] = train_df["text"].fillna("")

test_df["keyword"] = test_df["keyword"].fillna("")
test_df["location"] = test_df["location"].fillna("")
test_df["text"] = test_df["text"].fillna("")

train_df["combined_text"] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df["combined_text"] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# DRY_RUN
if DRY_RUN:
    train_df = train_df.head(200)

# Sample train data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Tokenizer and vocabulary
def build_vocab(texts, max_vocab):
    word_freq = {}
    for text in texts:
        words = text.split()
        for word in words:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    vocab = {word: i + 2 for i, (word, _) in enumerate(sorted_words[:max_vocab - 2])}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

spec = {
vocab = build_vocab(train_df["combined_text"], spec["max_vocab"])
}

def text_to_sequence(text, vocab, max_len):
    sequence = []
    words = text.split()
    for word in words:
        if word in vocab:
            sequence.append(vocab[word])
        else:
            sequence.append(vocab["<UNK>"])
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab["<PAD>"]] * (max_len - len(sequence))

train_texts = train_df["combined_text"].tolist()
train_labels = train_df["target"].tolist()

test_texts = test_df["combined_text"].tolist()

train_sequences = [text_to_sequence(text, vocab, spec["max_len"]) for text in train_texts]
test_sequences = [text_to_sequence(text, vocab, spec["max_len"]) for text in test_texts]

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

train_dataset = TextDataset(train_sequences, train_labels)
test_dataset = TextDataset(test_sequences)

val_size = spec["val_size"]
stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_indices, val_indices, _, _ = train_test_split(
    list(range(len(train_dataset))), train_labels, test_size=val_size, random_state=42, stratify=stratify_labels
)

train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)
val_loader = DataLoader(val_subset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False)

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
        return out

model = LSTMClassifier(len(vocab), spec["embedding_dim"], spec["hidden_dim"], spec["num_layers"], spec["dropout"]).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
criterion = torch.nn.BCEWithLogitsLoss()

def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if not DRY_RUN:
    train_model(model, train_loader, optimizer, criterion, spec["epochs"])

# Validation
def validate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences).squeeze()
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

val_probs, val_labels = validate_model(model, val_loader)

# Tune threshold
best_threshold = None
best_f1 = 0.0

for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

acc = accuracy_score(val_labels, (val_probs >= best_threshold).astype(int))

# Final submission
if FINAL_SUBMISSION:
    full_train_dataset = TextDataset(train_sequences + test_sequences, train_labels)
    full_train_loader = DataLoader(full_train_dataset, batch_size=spec["batch_size"], shuffle=True, pin_memory=False)

    final_model = LSTMClassifier(len(vocab), spec["embedding_dim"], spec["hidden_dim"], spec["num_layers"], spec["dropout"]).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=spec["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    train_model(final_model, full_train_loader, optimizer, criterion, spec["epochs"])

    test_probs = []
    final_model.eval()
    with torch.no_grad():
        for sequences in DataLoader(test_dataset, batch_size=spec["batch_size"], shuffle=False, pin_memory=False):
            sequences = sequences.to(device)
            outputs = final_model(sequences).squeeze()
            preds = torch.sigmoid(outputs)
            test_probs.extend(preds.cpu().numpy())

    test_preds = (np.array(test_probs) >= best_threshold).astype(int)

if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
    submission_df = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission_df.to_csv(spec["submission_path"], index=False)

print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')