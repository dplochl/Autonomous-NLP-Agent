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
    "channels": 128,
    "kernel_sizes": [3, 4, 5],
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 4,
    "learning_rate": 0.001,
    "val_size": 0.2,
    "threshold_min": 0.3,
    "threshold_max": 0.7,
    "threshold_steps": 41,
    "dry_run_head": 200,
    "experiment_name": "cnn_20260420_220250_run_01",
    "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260420_220250/run_001/submission.csv"
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
    train_df = train_df.head(spec['dry_run_head'])

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

vocab = build_vocab(train_df['text'], spec['max_vocab'])

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_texts = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec['max_len'])).tolist()
train_labels = train_df['target'].tolist()

test_texts = test_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec['max_len'])).tolist()

# Train-Validation Split
stratify_labels = train_labels if len(set(train_labels)) > 1 else None
X_train, X_val, y_train, y_val = train_test_split(train_texts, train_labels, test_size=spec['val_size'], random_state=42, stratify=stratify_labels)

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return text, label
        else:
            return text

train_dataset = TextDataset(X_train, y_train)
val_dataset = TextDataset(X_val, y_val)
test_dataset = TextDataset(test_texts)

train_loader = DataLoader(train_dataset, batch_size=spec['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=spec['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=spec['batch_size'])

# Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, kernel_size) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        dropped = self.dropout(cat)
        output = self.fc(dropped)
        return output

model = TextCNN(spec['max_vocab'], spec['embedding_dim'], spec['channels'], spec['kernel_sizes'], spec['dropout'])
optimizer = optim.Adam(model.parameters(), lr=spec['learning_rate'])
criterion = nn.BCEWithLogitsLoss()

# Training
if not DRY_RUN:
    for epoch in range(spec['epochs']):
        model.train()
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Validation
model.eval()
val_probs = []
with torch.no_grad():
    for texts, _ in val_loader:
        outputs = model(texts).squeeze()
        probs = torch.sigmoid(outputs).numpy()
        val_probs.extend(np.atleast_1d(probs))

# Test
test_probs = []
with torch.no_grad():
    for texts in test_loader:

        if isinstance(texts, (list, tuple)):

            texts = texts[0]
        outputs = model(texts).squeeze()
        probs = torch.sigmoid(outputs).numpy()
        test_probs.extend(np.atleast_1d(probs))

# Choose best threshold
best_f1 = 0
best_threshold = 0.5
for threshold in np.linspace(spec['threshold_min'], spec['threshold_max'], spec['threshold_steps']):
    val_preds = (np.array(val_probs) > threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Submission
os.makedirs(os.path.dirname(spec['submission_path']), exist_ok=True)
test_preds = (np.array(test_probs) > best_threshold).astype(int)
submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
submission_df.to_csv(spec['submission_path'], index=False)

# Metrics
val_preds = (np.array(val_probs) > best_threshold).astype(int)
acc = accuracy_score(y_val, val_preds)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')