import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    "experiment_name": "cnn_20260420_221524_run_01",
    "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260420_221524/run_001/submission.csv"
}

# Environment variables
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Preprocessing
train_df['keyword'].fillna('', inplace=True)
train_df['location'].fillna('', inplace=True)
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

test_df['keyword'].fillna('', inplace=True)
test_df['location'].fillna('', inplace=True)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

if DRY_RUN:
    train_df = train_df.head(spec["dry_run_head"])

# Tokenizer and Vocabulary
def build_vocab(texts, max_vocab):
    word_freq = {}
    for text in texts:
        words = text.split()
        for word in words:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: idx + 2 for idx, (word, freq) in enumerate(sorted_words[:max_vocab - 2])}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(train_df['text'], spec["max_vocab"])

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_df['sequence'] = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec["max_len"]))
test_df['sequence'] = test_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec["max_len"]))

# Dataset and DataLoader
class DisasterDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_sequences = train_df['sequence'].tolist()
train_labels = train_df['target'].tolist()

stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    train_sequences, train_labels, test_size=spec["val_size"], random_state=42, stratify=stratify_labels
)

train_dataset = DisasterDataset(train_sequences, train_labels)
val_dataset = DisasterDataset(val_sequences, val_labels)
test_dataset = DisasterDataset(test_df['sequence'].tolist())

train_loader = DataLoader(train_dataset, batch_size=spec["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=spec["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=spec["batch_size"])

# Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = [torch.relu(conv(x)).squeeze(2) for conv in self.convs]
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

model = TextCNN(spec["max_vocab"], spec["embedding_dim"], spec["channels"], spec["kernel_sizes"], spec["dropout"])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=spec["learning_rate"])

# Training
if not DRY_RUN:
    for epoch in range(spec["epochs"]):
        model.train()
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Validation and Test
def predict(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for sequences in loader:

            if isinstance(sequences, (list, tuple)):

                sequences = sequences[0]
            outputs = model(sequences).squeeze()
            probs = torch.sigmoid(outputs).numpy()
            all_probs.extend(probs)
    return np.array(all_probs)

val_probs = predict(model, val_loader)
test_probs = predict(model, test_loader)

# Choose best threshold
best_threshold = 0.5
best_f1 = 0

for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    val_preds = (val_probs > threshold).astype(int)
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Evaluate on validation set
val_preds = (val_probs > best_threshold).astype(int)
acc = accuracy_score(val_labels, val_preds)

# Create submission
os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
submission_df = pd.DataFrame({'id': test_df['id'], 'target': (test_probs > best_threshold).astype(int)})
submission_df.to_csv(spec["submission_path"], index=False)

print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')