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
submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/cnn_20260514_114908/run_001/submission.csv"
device = torch.device("cpu")

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Preprocess data
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
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(sorted_words[:max_vocab - 2])}
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
    def __init__(self, df):
        self.sequences = df['sequence'].tolist()
        self.labels = df['target'].tolist() if 'target' in df.columns else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32) if self.labels is not None else None
        return sequence, label

train_dataset = DisasterDataset(train_df)
test_dataset = DisasterDataset(test_df)

stratify_labels = train_df['target'] if len(set(train_df['target'])) > 1 else None
train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42, stratify=stratify_labels)
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=False)

# Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = [torch.relu(conv(x)) for conv in self.convs]
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x).sigmoid()
        return x

model = TextCNN(vocab_size=len(vocab), embedding_dim=128, channels=128, kernel_sizes=[3, 4, 5], dropout=0.3).to(device)

# Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if not DRY_RUN:
    train_model(model, train_loader, optimizer, criterion, epochs=3)

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
preds_binary = (val_preds >= best_threshold).astype(int)
f1 = f1_score(val_labels, preds_binary)
acc = accuracy_score(val_labels, preds_binary)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')