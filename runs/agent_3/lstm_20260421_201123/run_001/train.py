import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load environment variables
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Load data
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
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
    train_df = train_df.head(200)

# Sample training data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Train-test split
stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None
train_texts, val_texts, y_train, y_val = train_test_split(train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels)

# Tokenizer and vocabulary
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

vocab = build_vocab(train_texts, max_vocab=20000)

# Text to sequence conversion
def text_to_sequence(text, vocab, max_len):
    words = text.split()
    seq = [vocab.get(word, vocab['<UNK>']) for word in words]
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq + [vocab['<PAD>']] * (max_len - len(seq))

train_sequences = np.array([text_to_sequence(text, vocab, 64) for text in train_texts])
val_sequences = np.array([text_to_sequence(text, vocab, 64) for text in val_texts])

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
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_dataset = TextDataset(train_sequences, y_train)
val_dataset = TextDataset(val_sequences, y_val)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return torch.sigmoid(out.squeeze())

# Training
model = LSTMClassifier(len(vocab), 128, 128, 1, 0.3)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if DRY_RUN:
    train_model(model, train_loader, criterion, optimizer, epochs=1)
else:
    train_model(model, train_loader, criterion, optimizer, epochs=3)

# Validation
model.eval()
val_preds = []
with torch.no_grad():
    for sequences in val_loader:
        outputs = model(sequences)
        val_preds.extend(outputs.numpy())

val_preds = np.array(val_preds)
best_threshold = 0.5
best_f1 = 0

for threshold in np.linspace(0.3, 0.7, 41):
    y_pred = (val_preds > threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Best F1 score: {best_f1}")

# Final submission
if FINAL_SUBMISSION:
    final_model = LSTMClassifier(len(vocab), 128, 128, 1, 0.3)
    full_train_sequences = np.array([text_to_sequence(text, vocab, 64) for text in train_df['text']])
    full_train_dataset = TextDataset(full_train_sequences, train_df['target'])
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    train_model(final_model, full_train_loader, criterion, optimizer, epochs=3)

# Test predictions
test_sequences = np.array([text_to_sequence(text, vocab, 64) for text in test_df['text']])
test_dataset = TextDataset(test_sequences)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

final_model.eval()
test_preds = []
with torch.no_grad():
    for sequences in test_loader:
        outputs = final_model(sequences)
        test_preds.extend(outputs.numpy())

test_preds = np.array(test_preds)
y_pred_test = (test_preds > best_threshold).astype(int)

# Write submission
if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/lstm_20260421_201123/run_001/submission.csv"), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_pred_test})
    submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/lstm_20260421_201123/run_001/submission.csv", index=False)

# Metrics
acc = accuracy_score(y_val, (val_preds > best_threshold).astype(int))
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')