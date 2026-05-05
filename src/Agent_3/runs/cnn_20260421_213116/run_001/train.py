import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Preprocess data
train_df['text'] = train_df['keyword'].fillna('') + ' [SEP] ' + train_df['text']
test_df['text'] = test_df['keyword'].fillna('') + ' [SEP] ' + test_df['text']

if DRY_RUN:
    train_df = train_df.head(int(os.environ.get("AGENT_DRY_RUN_HEAD", "200")))

# Sample data if needed
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
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab-2]
    vocab = {word: i+2 for i, (word, _) in enumerate(sorted_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(train_df['text'], max_vocab=20000)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_df['sequence'] = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, 48))
test_df['sequence'] = test_df['text'].apply(lambda x: text_to_sequence(x, vocab, 48))

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

# Split data
stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None
train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    train_df['sequence'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels
)

train_dataset = DisasterDataset(train_sequences.tolist(), train_labels.tolist())
val_dataset = DisasterDataset(val_sequences.tolist(), val_labels.tolist())
test_dataset = DisasterDataset(test_df['sequence'].tolist())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # Global max pooling
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x).sigmoid()

model = TextCNN(vocab_size=len(vocab), embedding_dim=128, channels=128, kernel_sizes=[3, 4, 5], dropout=0.3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
def train_model(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = nn.BCELoss()(outputs, labels)
            loss.backward()
            optimizer.step()

# Validation
def validate_model(model, val_loader):
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for sequences, labels in val_loader:
            outputs = model(sequences).squeeze().numpy()
            all_preds.extend(outputs)
            all_labels.extend(labels.numpy())
        return np.array(all_preds), np.array(all_labels)

# Find best threshold
best_threshold = 0.5
best_f1 = 0

for threshold in np.linspace(0.3, 0.7, 41):
    train_model(model, train_loader, optimizer, epochs=3)
    val_preds, val_labels = validate_model(model, val_loader)
    val_preds_binary = (val_preds > threshold).astype(int)
    f1 = f1_score(val_labels, val_preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission
if FINAL_SUBMISSION:
    train_model(model, train_loader, optimizer, epochs=3)  # Retrain on full training data
    test_preds = []
    model.eval()
    with torch.no_grad():
        for sequences in test_loader:

            if isinstance(sequences, (list, tuple)):

                sequences = sequences[0]
            outputs = model(sequences).squeeze().numpy()
            test_preds.extend(outputs)

# Write submission
if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260421_213116/run_001/submission.csv"), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': (np.array(test_preds) > best_threshold).astype(int)})
    submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260421_213116/run_001/submission.csv", index=False)

# Metrics
val_preds_binary = (val_preds > best_threshold).astype(int)
f1 = f1_score(val_labels, val_preds_binary)
acc = accuracy_score(val_labels, val_preds_binary)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')