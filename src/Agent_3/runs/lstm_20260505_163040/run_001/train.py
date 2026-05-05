import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Environment variables
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Constants
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/lstm_20260505_163040/run_001/submission.csv"
device = torch.device("cpu")

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Preprocessing
train_df['text'] = train_df['keyword'].fillna('') + ' [SEP] ' + train_df['text']
test_df['text'] = test_df['keyword'].fillna('') + ' [SEP] ' + test_df['text']

if DRY_RUN:
    train_df = train_df.head(200)

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
    word_to_idx = {word: idx+2 for idx, (word, _) in enumerate(sorted_words)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    return word_to_idx

vocab = build_vocab(train_df['text'], max_vocab=20000)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_df['sequence'] = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, 64))
test_df['sequence'] = test_df['text'].apply(lambda x: text_to_sequence(x, vocab, 64))

# Dataset and DataLoader
class DisasterDataset(Dataset):
    def __init__(self, data, targets=None):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y
        else:
            return x

train_sequences = train_df['sequence'].tolist()
train_labels = train_df['target'].tolist()

stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_seq, val_seq, train_label, val_label = train_test_split(
    train_sequences, train_labels, test_size=0.2, random_state=42, stratify=stratify_labels
)

train_dataset = DisasterDataset(train_seq, train_label)
val_dataset = DisasterDataset(val_seq, val_label)
test_dataset = DisasterDataset(test_df['sequence'].tolist())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=False)

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
        out = self.fc(last_output)
        return torch.sigmoid(out)

model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0).to(device)

# Training
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, epochs):
    best_f1 = 0
    best_threshold = 0.5
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        if not DRY_RUN:
            model.eval()
            val_preds = []
            with torch.no_grad():
                for x in val_loader:

                    if isinstance(x, (list, tuple)):

                        x = x[0]

                    if isinstance(x, (list, tuple)):

                        x = x[0]
                    x = x.to(device)
                    outputs = model(x).squeeze().cpu().numpy()
                    outputs = outputs.squeeze()
                    val_preds.extend(outputs)

            best_f1, best_threshold = find_best_threshold(val_label, val_preds, threshold_min=0.3, threshold_max=0.7, threshold_steps=41)

    return best_threshold

def find_best_threshold(y_true, y_pred, threshold_min, threshold_max, threshold_steps):
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
        y_pred_class = (y_pred > threshold).astype(int)
        f1 = f1_score(y_true, y_pred_class)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_f1, best_threshold

best_threshold = train_model(model, train_loader, val_loader, epochs=3)

# Final submission
if FINAL_SUBMISSION:
    model.train()
    final_train_dataset = DisasterDataset(train_seq + val_seq, train_label + val_label)
    final_train_loader = DataLoader(final_train_dataset, batch_size=64, shuffle=True, pin_memory=False)
    for epoch in range(3):
        for x, y in final_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

model.eval()
test_preds = []
with torch.no_grad():
    for x in test_loader:

        if isinstance(x, (list, tuple)):

            x = x[0]

        if isinstance(x, (list, tuple)):

            x = x[0]
        x = x.to(device)
        outputs = model(x).squeeze().cpu().numpy()
        test_preds.extend(outputs)

if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': (np.array(test_preds) > best_threshold).astype(int)})
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission_df.to_csv(submission_path, index=False)

# Metrics
val_preds_class = (np.array(val_preds) > best_threshold).astype(int)
f1 = f1_score(val_label, val_preds_class)
acc = accuracy_score(val_label, val_preds_class)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')