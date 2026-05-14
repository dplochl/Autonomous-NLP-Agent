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
train_df.fillna({'keyword': '', 'location': '', 'text': ''}, inplace=True)
test_df.fillna({'keyword': '', 'location': '', 'text': ''}, inplace=True)

# Build text field
train_df['text'] = train_df['keyword'] + " [SEP] " + train_df['text']
test_df['text'] = test_df['keyword'] + " [SEP] " + test_df['text']

# DRY_RUN
if DRY_RUN:
    train_df = train_df.head(200)

# Sample if needed
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Stratify labels
stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels
)
train_labels = np.asarray(train_labels)
val_labels = np.asarray(val_labels)

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

train_sequences = np.array([text_to_sequence(text, vocab, 48) for text in train_texts])
val_sequences = np.array([text_to_sequence(text, vocab, 48) for text in val_texts])
test_sequences = np.array([text_to_sequence(text, vocab, 48) for text in test_df['text']])

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
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_dataset = TextDataset(train_sequences, train_labels)
val_dataset = TextDataset(val_sequences, val_labels)
test_dataset = TextDataset(test_sequences)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        x = [nn.functional.relu(conv(x)).squeeze(2) for conv in self.convs]
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x).sigmoid()

model = TextCNN(vocab_size=len(vocab), embedding_dim=128, channels=128, kernel_sizes=[3, 4, 5], dropout=0.3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = nn.BCELoss()(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for sequences, _ in val_loader:
                outputs = model(sequences).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                

        val_preds = np.array(val_preds)
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.linspace(0.3, 0.7, 41):
            preds = (val_preds > threshold).astype(int)
            f1 = f1_score(val_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    return best_threshold

# Train model
best_threshold = train_model(model, train_loader, val_loader, epochs=3)

# Final submission
if FINAL_SUBMISSION:
    # Retrain on full training data
    full_train_dataset = TextDataset(train_sequences, train_labels)
    full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True)
    train_model(model, full_train_loader, val_loader, epochs=3)

# Predict test set
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

        if isinstance(sequences, (list, tuple)):

            sequences = sequences[0]

        if isinstance(sequences, (list, tuple)):

            sequences = sequences[0]
        outputs = model(sequences).squeeze().numpy()
        test_preds.append(outputs.cpu().numpy())

test_preds = np.array(test_preds)
test_labels = (test_preds > best_threshold).astype(int)

# Write submission if needed
if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260421_195457/run_001/submission.csv"), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_labels})
    submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260421_195457/run_001/submission.csv", index=False)

# Metrics
val_preds = (np.array(val_preds) > best_threshold).astype(int)
f1 = f1_score(val_labels, val_preds)
acc = accuracy_score(val_labels, val_preds)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')