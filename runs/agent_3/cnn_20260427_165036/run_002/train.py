import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Build text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# DRY_RUN
if DRY_RUN:
    train_df = train_df.head(int(os.environ.get("DRY_RUN_HEAD", "200")))

# Sample training data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Stratify labels
stratify_labels = train_df['target'] if len(train_df['target'].unique()) > 1 else None

# Train-test split
train_texts, val_texts, y_train, y_val = train_test_split(
    train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels
)

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
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words[:max_vocab - 1])}
    return vocab

vocab = build_vocab(train_texts, max_vocab=20000)

# Text to sequence
def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 0) for word in text.split()][:max_len][:max_len][:max_len]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))

train_sequences = np.array([text_to_sequence(text, vocab, max_len=56) for text in train_texts])
val_sequences = np.array([text_to_sequence(text, vocab, max_len=56) for text in val_texts])

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
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_dataset = TextDataset(train_sequences, y_train)
val_dataset = TextDataset(val_sequences, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, kernel_size) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(c, c.size(2)).squeeze(2) for c in conved]  # (batch_size, channels)
        cat = torch.cat(pooled, dim=1)
        dropped = self.dropout(cat)
        output = self.fc(dropped)
        return output.squeeze()

# Training
model = TextCNN(vocab_size=len(vocab) + 1, embedding_dim=192, channels=192, kernel_sizes=[3, 4, 5], dropout=0.2)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if not DRY_RUN:
    for epoch in range(3):
        model.train()
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Validation
model.eval()
val_preds = []
with torch.no_grad():
    for sequences, _ in val_loader:
        outputs = model(sequences)
        val_preds.extend(torch.sigmoid(outputs).numpy())

# Choose best threshold
best_threshold = None
best_f1 = 0.0
for threshold in np.linspace(0.3, 0.7, 41):
    y_pred = (np.array(val_preds) > threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission
if FINAL_SUBMISSION and WRITE_SUBMISSION:
    # Retrain on full train data
    full_train_sequences = np.array([text_to_sequence(text, vocab, max_len=56) for text in train_df['text']])
    full_train_dataset = TextDataset(full_train_sequences, train_df['target'])
    full_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True)

    model = TextCNN(vocab_size=len(vocab) + 1, embedding_dim=192, channels=192, kernel_sizes=[3, 4, 5], dropout=0.2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        model.train()
        for sequences, labels in full_train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Predict test set
    test_sequences = np.array([text_to_sequence(text, vocab, max_len=56) for text in test_df['text']])
    test_dataset = TextDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    test_preds = []
    with torch.no_grad():
        for sequences in test_loader:

            if isinstance(sequences, (list, tuple)):

                sequences = sequences[0]

            if isinstance(sequences, (list, tuple)):

                sequences = sequences[0]

            if isinstance(sequences, (list, tuple)):

                sequences = sequences[0]
            outputs = model(sequences)
            test_preds.extend(torch.sigmoid(outputs).numpy())

    # Write submission
    if WRITE_SUBMISSION:
        os.makedirs(os.path.dirname(os.environ.get("submission_path", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260427_165036/run_001/submission.csv")), exist_ok=True)
        submission_df = pd.DataFrame({'id': test_df['id'], 'target': (np.array(test_preds) > best_threshold).astype(int)})
        submission_df.to_csv(os.environ.get("submission_path", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260427_165036/run_001/submission.csv"), index=False)

# Metrics
y_pred = (np.array(val_preds) > best_threshold).astype(int)
f1 = f1_score(y_val, y_pred)
acc = accuracy_score(y_val, y_pred)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')