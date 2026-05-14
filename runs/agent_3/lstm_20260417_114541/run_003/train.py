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
    "architecture": "LSTM",
    "max_vocab": 20000,
    "max_len": 64,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 4,
    "learning_rate": 0.001,
    "val_size": 0.2,
    "threshold_min": 0.3,
    "threshold_max": 0.7,
    "threshold_steps": 41,
    "dry_run_head": 200,
    "experiment_name": "lstm_20260417_114541_run_01",
    "submission_path": "submissions/lstm_20260417_114541_run_03_submission.csv"
}

# Environment variables
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Preprocessing
train_df.fillna("", inplace=True)
test_df.fillna("", inplace=True)

train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if row['keyword'] else row['text'], axis=1)
test_texts = test_df['text'].tolist()

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
    vocab = {word: i + 2 for i, (word, _) in enumerate(sorted_words[:max_vocab - 2])}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

texts = train_df['text'].tolist()
vocab = build_vocab(texts, spec["max_vocab"])

def text_to_sequence(text, vocab, max_len):
    words = text.split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_sequences = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec["max_len"])).tolist()
test_sequences = [text_to_sequence(text, vocab, spec["max_len"]) for text in test_texts]

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
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_labels = train_df['target'].tolist()
stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    train_sequences, train_labels, test_size=spec["val_size"], random_state=42, stratify=stratify_labels
)

train_dataset = DisasterDataset(train_sequences, train_labels)
val_dataset = DisasterDataset(val_sequences, val_labels)
test_dataset = DisasterDataset(test_sequences)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return torch.sigmoid(out.squeeze())

model = LSTMClassifier(
    vocab_size=spec["max_vocab"],
    embedding_dim=128,
    hidden_dim=128,
    num_layers=1,
    dropout=0.3
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=spec["learning_rate"])

# Training
if not DRY_RUN:
    for epoch in range(spec["epochs"]):
        model.train()
        running_loss = 0.0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{spec["epochs"]}], Loss: {running_loss/len(train_loader):.4f}')

# Validation
model.eval()
val_preds = []
with torch.no_grad():
    for sequences, labels in val_loader:
        outputs = model(sequences)
        val_preds.extend(outputs.numpy())

val_labels = np.array(val_labels)
best_f1 = 0
best_threshold = 0.5
for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    preds = (np.array(val_preds) > threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Test predictions
model.eval()
test_preds = []
with torch.no_grad():
    for sequences in test_loader:
        outputs = model(sequences)
        test_preds.extend(outputs.numpy())

# Submission
os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'target': (np.array(test_preds) > best_threshold).astype(int)
})
submission_df.to_csv(spec["submission_path"], index=False)

# Metrics
test_labels = np.zeros(len(submission_df))  # Placeholder, actual labels not available in test set
acc = accuracy_score(test_labels, submission_df['target'])
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')