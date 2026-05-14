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

# Preprocess data
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

if DRY_RUN:
    train_df = train_df.head(200)

# Sample data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Tokenizer and vocabulary
def build_vocab(texts, max_vocab):
    word_freq = {}
    for text in texts:
        words = text.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    vocab = {word: i + 2 for i, (word, _) in enumerate(sorted_words[:max_vocab - 2])}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(train_df['text'], max_vocab=25000)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_df['sequence'] = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, 72))
test_df['sequence'] = test_df['text'].apply(lambda x: text_to_sequence(x, vocab, 72))

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
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_sequences = train_df['sequence'].tolist()
train_labels = train_df['target'].tolist()

stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    train_sequences, train_labels, test_size=0.2, random_state=42, stratify=stratify_labels
)

train_dataset = DisasterDataset(train_sequences, train_labels)
val_dataset = DisasterDataset(val_sequences, val_labels)
test_dataset = DisasterDataset(test_df['sequence'].tolist(), test_df['target'].tolist())

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_df['target'] = None  # No labels for test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model definition
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return torch.sigmoid(output.squeeze())

# Training
vocab_size = len(vocab)
embedding_dim = 160
hidden_dim = 144
num_layers = 2
dropout = 0.4
learning_rate = 0.0005
epochs = 3

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if not DRY_RUN:
    for epoch in range(epochs):
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
        val_preds.extend([output.item() for output in outputs])

val_labels = np.array(val_labels)
best_threshold = 0.5
best_f1 = 0

for threshold in np.linspace(0.3, 0.7, 41):
    preds = (np.array(val_preds) > threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission
if FINAL_SUBMISSION:
    model.train()
    full_train_dataset = DisasterDataset(train_sequences + val_sequences, train_labels + val_labels)
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for sequences, labels in full_train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

model.eval()
test_preds = []
with torch.no_grad():
    for sequences, _ in test_loader:
        outputs = model(sequences)
        test_preds.extend(output.squeeze().item() for output in outputs)

# Write submission
if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': (np.array(test_preds) > best_threshold).astype(int)})
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission_df.to_csv(submission_path, index=False)

# Metrics
preds = (np.array(val_preds) > best_threshold).astype(int)
f1 = f1_score(val_labels, preds)
acc = accuracy_score(val_labels, preds)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')