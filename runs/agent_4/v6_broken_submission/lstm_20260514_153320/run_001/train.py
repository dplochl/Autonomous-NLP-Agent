import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

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

# Preprocessing
train_df['text'] = train_df['keyword'].fillna('') + ' [SEP] ' + train_df['text']
test_df['text'] = test_df['keyword'].fillna('') + ' [SEP] ' + test_df['text']

if DRY_RUN:
    train_df = train_df.head(200)

# Sample if needed
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Tokenizer and Vocabulary
def build_vocab(texts, max_vocab):
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common(max_vocab - 1))}
    return vocab

vocab = build_vocab(train_df['text'], max_vocab=25000)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 0) for word in text.split()][:max_len]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))

train_df['sequence'] = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, max_len=72))
test_df['sequence'] = test_df['text'].apply(lambda x: text_to_sequence(x, vocab, max_len=72))

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
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

# Split data
X_train, X_val, y_train, y_val = train_test_split(train_df['sequence'], train_df['target'], test_size=0.2, random_state=42)

train_dataset = DisasterDataset(X_train.tolist(), y_train.tolist())
val_dataset = DisasterDataset(X_val.tolist(), y_val.tolist())
test_dataset = DisasterDataset(test_df['sequence'].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=False)

# Model
class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

device = torch.device("cpu")
model = LSTMClassifier(vocab_size=25000, embedding_dim=160, hidden_dim=192, num_layers=2, dropout=0.4).to(device)

# Training
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if DRY_RUN:
    epochs = 1
else:
    epochs = 3

train_model(model, train_loader, criterion, optimizer, epochs)

# Validation
def collect_probs_and_labels(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            all_probs.extend(np.atleast_1d(outputs.detach().cpu().numpy()).tolist())
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()).tolist())
    return np.asarray(all_probs), np.asarray(all_labels)

val_probs, val_labels = collect_probs_and_labels(model, val_loader)

# Choose best threshold
best_threshold = 0.5
best_f1 = 0.0
for threshold in np.linspace(0.3, 0.7, 41):
    val_preds = (val_probs > threshold).astype(int)
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission
test_preds = np.array([], dtype=int)
if WRITE_SUBMISSION:
    class _UnlabeledTextDataset(Dataset):
        def __init__(self, texts):
            self.texts = list(texts)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            token_ids = text_to_sequence(text, vocab, 72)
            return torch.tensor(token_ids, dtype=torch.long)

    test_loader = DataLoader(_UnlabeledTextDataset(test_df['text']), batch_size=32, shuffle=False, pin_memory=False)
    if FINAL_SUBMISSION:
        full_train_dataset = DisasterDataset(train_sequences + val_seq, train_label + val_label)
        full_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True, pin_memory=False)
        train_model(model, full_train_loader, optimizer, criterion, 3)

    model.eval()
    test_probs = []
    with torch.no_grad():
        for sequences in test_loader:
            if isinstance(sequences, (list, tuple)):
                sequences = sequences[0]
            sequences = sequences.to(device)
            outputs = model(sequences)
            test_probs.extend(np.atleast_1d(outputs.detach().cpu().numpy()).tolist())
    test_preds = (np.asarray(test_probs) > best_threshold).astype(int)

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
    submission_df.to_csv(submission_path, index=False)

# Metrics
val_preds = (val_probs > best_threshold).astype(int)
f1 = f1_score(val_labels, val_preds)
acc = accuracy_score(val_labels, val_preds)
# Metrics
y_pred_val = (np.array(val_preds) > best_threshold).astype(int)
acc = accuracy_score(y_val, y_pred_val)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')