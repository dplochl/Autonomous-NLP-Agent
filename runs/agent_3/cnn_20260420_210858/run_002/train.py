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
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Fill missing values
train_df = train_df.assign(keyword=train_df['keyword'].fillna(''))
train_df = train_df.assign(location=train_df['location'].fillna(''))
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

test_df = test_df.assign(keyword=test_df['keyword'].fillna(''))
test_df = test_df.assign(location=test_df['location'].fillna(''))
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# DRY_RUN
if DRY_RUN:
    train_df = train_df.head(200)
    test_df = test_df.head(200)

# Define spec
spec = {
    "architecture": "CNN",
    "max_vocab": 20000,
    "max_len": 48,
    "embedding_dim": 128,
    "channels": 64,
    "kernel_sizes": [3, 4, 5],
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 0.0005,
    "val_size": 0.2,
    "threshold_min": 0.3,
    "threshold_max": 0.7,
    "threshold_steps": 41,
    "dry_run_head": 200,
    "experiment_name": "cnn_20260420_210858_run_02",
    "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260420_210858/run_002/submission.csv"
}

# Tokenizer and vocabulary
def build_vocab(texts, max_vocab):
    word_freq = {}
    for text in texts:
        words = text.split()
        for word in words:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab-2]
    vocab = {word: idx+2 for idx, (word, _) in enumerate(sorted_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(train_df['text'], spec["max_vocab"])

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

# Dataset and DataLoader
class TweetDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_seq = torch.tensor(text_to_sequence(self.texts[idx], vocab, spec["max_len"]))
        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx])
            return text_seq, label
        else:
            return text_seq

train_texts = train_df['text'].tolist()
train_labels = train_df['target'].tolist()

stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=spec["val_size"], random_state=42, stratify=stratify_labels
)
train_labels = np.asarray(train_labels)
val_labels = np.asarray(val_labels)

train_dataset = TweetDataset(train_texts, train_labels)
val_dataset = TweetDataset(val_texts, val_labels)
test_dataset = TweetDataset(test_df['text'].tolist())

train_loader = DataLoader(train_dataset, batch_size=spec["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=spec["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=spec["batch_size"])

# CNN Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        x = [F.relu(conv(x)) for conv in self.convs]  # [(batch_size, channels, max_len-k+1)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, channels)]
        x = torch.cat(x, dim=1)  # (batch_size, channels * len(kernel_sizes))
        x = self.dropout(x)
        logit = self.fc(x)  # (batch_size, 1)
        return logit

model = TextCNN(spec["max_vocab"], spec["embedding_dim"], spec["channels"], spec["kernel_sizes"], spec["dropout"])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=spec["learning_rate"])

# Training
if not DRY_RUN:
    for epoch in range(spec["epochs"]):
        model.train()
        for batch_texts, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_texts).squeeze()
            loss = criterion(outputs, batch_labels.float())
            loss.backward()
            optimizer.step()

# Validation
model.eval()
val_probs = []
with torch.no_grad():
    for batch_texts in val_loader:
        outputs = model(batch_texts).squeeze()
        probs = torch.sigmoid(outputs)
        val_probs.extend(probs.numpy())

# Test
test_probs = []
with torch.no_grad():
    for batch_texts in test_loader:
        outputs = model(batch_texts).squeeze()
        probs = torch.sigmoid(outputs)
        test_probs.extend(probs.numpy())

# Choose best threshold
best_f1 = 0
best_threshold = 0.5
for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    val_preds = (np.array(val_probs) > threshold).astype(int)
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Submission
os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
test_preds = (np.array(test_probs) > best_threshold).astype(int)
submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
submission_df.to_csv(spec["submission_path"], index=False)

# Metrics
val_preds = (np.array(val_probs) > best_threshold).astype(int)
acc = accuracy_score(val_labels, val_preds)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')