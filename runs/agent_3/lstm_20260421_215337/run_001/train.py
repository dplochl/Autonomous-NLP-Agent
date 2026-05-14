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
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
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
stratify_labels = train_df['target'] if len(train_df['target'].unique()) > 1 else None
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
    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words[:max_vocab-1])}
    vocab['<PAD>'] = 0
    return vocab

vocab = build_vocab(train_texts, max_vocab=20000)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 0) for word in text.split()][:max_len]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_seq = torch.tensor(text_to_sequence(self.texts[idx], vocab, max_len=64), dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return text_seq, label
        else:
            return text_seq

train_dataset = TextDataset(train_texts.tolist(), y_train.tolist())
val_dataset = TextDataset(val_texts.tolist(), y_val.tolist())
test_dataset = TextDataset(test_df['text'].tolist())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.3)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

if not DRY_RUN:
    train_model(model, train_loader, criterion, optimizer, epochs=3)

# Validation
def validate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc

f1, acc = validate_model(model, val_loader)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')

# Final submission
if WRITE_SUBMISSION:
    if FINAL_SUBMISSION:
        # Retrain on full train data
        full_train_dataset = TextDataset(train_df['text'].tolist(), train_df['target'].tolist())
        full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True)
        train_model(model, full_train_loader, criterion, optimizer, epochs=3)

    # Predict test set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:

            if isinstance(inputs, (list, tuple)):

                inputs = inputs[0]
            outputs = model(inputs)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)

    # Choose best threshold
    if FINAL_SUBMISSION:
        thresholds = np.linspace(0.3, 0.7, 41)
        best_f1 = 0
        best_threshold = 0.5
        for threshold in thresholds:
            preds = (np.array(all_preds) > threshold).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Retrain on full train data with best threshold
        model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs in test_loader:

                if isinstance(inputs, (list, tuple)):

                    inputs = inputs[0]
                outputs = model(inputs)
                preds = (outputs > best_threshold).float().cpu().numpy()
                all_preds.extend(preds)

    # Write submission
    os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/lstm_20260421_215337/run_001/submission.csv"), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': all_preds})
    submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/lstm_20260421_215337/run_001/submission.csv", index=False)