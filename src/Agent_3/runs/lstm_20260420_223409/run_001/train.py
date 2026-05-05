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
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Build text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# DRY RUN
if DRY_RUN:
    train_df = train_df.head(200)
    test_df = test_df.head(200)

# Define stratify labels
stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels
)
train_labels = np.asarray(train_labels)
val_labels = np.asarray(val_labels)

# Build vocabulary and tokenizer
vocab = {'<PAD>': 0, '<UNK>': 1}
for text in train_texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)
max_vocab = min(len(vocab), 20000)

def tokenize(text):
    return [vocab.get(word, vocab['<UNK>']) for word in text.split()]

# Convert texts to padded integer sequences
train_sequences = [tokenize(text)[:64] + [0] * (64 - len(tokenize(text))) for text in train_texts]
val_sequences = [tokenize(text)[:64] + [0] * (64 - len(tokenize(text))) for text in val_texts]
test_sequences = [tokenize(text)[:64] + [0] * (64 - len(tokenize(text))) for text in test_df['text']]

# Define Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float)
        else:
            return torch.tensor(self.sequences[idx], dtype=torch.long)

train_dataset = TextDataset(train_sequences, train_labels)
val_dataset = TextDataset(val_sequences, val_labels)
test_dataset = TextDataset(test_sequences)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define LSTM classifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1])
        return torch.sigmoid(output).squeeze()

# Initialize model, loss function, and optimizer
model = LSTMClassifier(max_vocab, 128, 128, 1, 0.3)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
if not DRY_RUN:
    for epoch in range(4):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Validation probabilities
model.eval()
val_probs = []
with torch.no_grad():
    for inputs, _ in val_loader:
        outputs = model(inputs)
        val_probs.extend(outputs.numpy())

# Test probabilities
test_probs = []
with torch.no_grad():
    for inputs in test_loader:
        outputs = model(inputs)
        test_probs.extend(outputs.numpy())

# Choose best cutoff
best_f1 = 0
best_threshold = 0.5
for threshold in np.linspace(0.3, 0.7, 41):
    val_preds = (np.array(val_probs) > threshold).astype(int)
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Compute final metrics on validation set
val_preds = (np.array(val_probs) > best_threshold).astype(int)
f1 = f1_score(val_labels, val_preds)
acc = accuracy_score(val_labels, val_preds)

# Save submission
os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/lstm_20260420_223409/run_001/submission.csv"), exist_ok=True)
submission_df = pd.DataFrame({'id': test_df['id'], 'target': (np.array(test_probs) > best_threshold).astype(int)})
submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/lstm_20260420_223409/run_001/submission.csv", index=False)

# Print final metrics
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')