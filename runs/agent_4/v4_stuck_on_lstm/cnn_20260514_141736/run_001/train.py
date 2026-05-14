import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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

# Fill missing values
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Build text field
train_df['text'] = train_df['keyword'] + ' [SEP] ' + train_df['text']
test_df['text'] = test_df['keyword'] + ' [SEP] ' + test_df['text']

# DRY_RUN handling
if DRY_RUN:
    train_df = train_df.head(200)

# Sample training data if needed
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Train-test split
stratify_labels = train_df['target'] if len(Counter(train_df['target'])) == 2 else None
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels
)
train_labels = np.asarray(train_labels)
val_labels = np.asarray(val_labels)

# Tokenizer and vocabulary
def build_vocab(texts, max_vocab):
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common(max_vocab - 1))}
    return vocab

vocab = build_vocab(train_texts, max_vocab=20000)

# Text to sequence conversion
def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 0) for word in text.split()][:max_len][:max_len]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))

train_sequences = np.array([text_to_sequence(text, vocab, max_len=48) for text in train_texts])
val_sequences = np.array([text_to_sequence(text, vocab, max_len=48) for text in val_texts])
test_sequences = np.array([text_to_sequence(text, vocab, max_len=48) for text in test_df['text']])

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
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_dataset = TextDataset(train_sequences, train_labels)
val_dataset = TextDataset(val_sequences, val_labels)
test_dataset = TextDataset(test_sequences)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=False)

# CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, channels, kernel_size=k) for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        x = [self.pool(conv(x)).squeeze(-1) for conv in self.convs]  # [(batch_size, channels), ...]
        x = torch.cat(x, dim=1)  # (batch_size, channels * len(kernel_sizes))
        x = self.dropout(x)
        return self.fc(x).sigmoid()

# Training loop
device = torch.device("cpu")
model = TextCNN(vocab_size=len(vocab) + 1, embedding_dim=128, channels=128, kernel_sizes=[3, 4, 5], dropout=0.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

def train_model(model, dataloader, optimizer, criterion):
    model.train()
    for batch in dataloader:
        sequences, labels = batch
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences).squeeze().cpu().numpy()
            all_preds.extend(outputs)
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()).tolist())
    return np.array(all_preds), np.array(all_labels)

if not DRY_RUN:
    for epoch in range(3):
        train_model(model, train_loader, optimizer, criterion)

# Validation
def collect_probs_and_labels(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_probs.extend(np.atleast_1d(probs).tolist())
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
            if 'tokenize' in globals():
                token_ids = tokenize(text, vocab, 48)
            else:
                token_ids = text_to_sequence(text, vocab, 48)
            return torch.tensor(token_ids, dtype=torch.long)

    test_loader = DataLoader(_UnlabeledTextDataset(test_df['text']), batch_size=64, shuffle=False, pin_memory=False)
    if FINAL_SUBMISSION:
        final_model = TextCNN(vocab_size=len(vocab) + 1, embedding_dim=128, channels=128, kernel_sizes=[3, 4, 5], dropout=0.3).to(device)
        optimizer_final = torch.optim.Adam(final_model.parameters(), lr=0.001)
        criterion_final = nn.BCEWithLogitsLoss()
        final_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=False)
        for epoch in range(3):
            final_model.train()
            for inputs, labels in final_train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer_final.zero_grad()
                outputs = final_model(inputs).squeeze()
                loss = criterion_final(outputs, labels)
                loss.backward()
                optimizer_final.step()
        test_model = final_model
    else:
        test_model = model

    test_model.eval()
    test_probs = []
    with torch.no_grad():
        for inputs in test_loader:
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(device)
            outputs = test_model(inputs).squeeze()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            test_probs.extend(np.atleast_1d(probs).tolist())
    test_preds = (np.asarray(test_probs) > best_threshold).astype(int)

    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_preds
    })
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

# Metrics
acc = accuracy_score(val_labels, (val_preds >= best_threshold).astype(int))
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')