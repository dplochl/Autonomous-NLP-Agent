import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Constants
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

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
    train_df = train_df.head(200)

# Sample training data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Train-test split
stratify_labels = train_df['target'] if len(train_df['target'].unique()) > 1 else None
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
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab-2]
    word_to_idx = {word: idx+2 for idx, (word, _) in enumerate(sorted_words)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    return word_to_idx

vocab = build_vocab(train_texts, max_vocab=20000)

# Text to sequence conversion
def text_to_sequence(text, vocab, max_len):
    words = text.split()
    seq = [vocab.get(word, vocab['<UNK>']) for word in words]
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq + [0] * (max_len - len(seq))

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
        if self.labels is not None:
            return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return torch.tensor(self.sequences[idx], dtype=torch.long)

train_dataset = TextDataset(train_sequences, y_train)
val_dataset = TextDataset(val_sequences, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=False)

# Model
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

model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.3)
device = torch.device("cpu")
model.to(device)

# Training
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if not DRY_RUN:
    train_model(model, train_loader, criterion, optimizer, epochs=3)

# Validation
def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc

f1, acc = evaluate_model(model, val_loader)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')

# Final submission
if WRITE_SUBMISSION:
    if FINAL_SUBMISSION:
        # Retrain on full train data
        full_train_dataset = TextDataset(train_sequences, y_train)
        full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True, pin_memory=False)
        train_model(model, full_train_loader, criterion, optimizer, epochs=3)

    # Predict test set
    test_texts = test_df['text'].apply(lambda text: f"{text} [SEP] {test_df.loc[test_df['text'] == text, 'keyword'].values[0]}"
    if pd.notna(test_df.loc[test_df['text'] == text, 'keyword'].values[0]) else text)
 if pd.notna(test_df.loc[test_df['text'] == text, 'keyword'].values[0]) else text)
    test_sequences = np.array([text_to_sequence(text, vocab, 64) for text in test_texts])
    test_dataset = TextDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)

    # Choose best threshold
    best_threshold = None
    best_f1 = 0
    for threshold in np.linspace(0.3, 0.7, 41):
        preds = (np.array(all_preds) > threshold).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Final prediction with best threshold
    final_preds = (np.array(all_preds) > best_threshold).astype(int)

    # Write submission
    submission_dir = os.path.dirname(os.environ.get("AGENT_SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/lstm_20260514_115327/run_001/submission.csv"))
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': final_preds})
    submission_df.to_csv(os.environ.get("AGENT_SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/lstm_20260514_115327/run_001/submission.csv"), index=False)