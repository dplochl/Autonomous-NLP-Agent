import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load environment variables
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Define constants
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
device = torch.device("cpu")

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Fill missing values
train_df["keyword"] = train_df["keyword"].fillna("")
train_df["location"] = train_df["location"].fillna("")
train_df["text"] = train_df["text"].fillna("")

test_df["keyword"] = test_df["keyword"].fillna("")
test_df["location"] = test_df["location"].fillna("")
test_df["text"] = test_df["text"].fillna("")

# Build text field
train_df["text"] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df["text"] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# DRY_RUN
if DRY_RUN:
    train_df = train_df.head(200)

# Sample train data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Train-test split
y = train_df["target"].values
stratify_labels = y if len(np.unique(y)) > 1 else None
train_texts, val_texts, train_labels, val_labels = train_test_split(train_df["text"], y, test_size=0.2, random_state=42, stratify=stratify_labels)
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
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words[:max_vocab - 1])}
    vocab["<PAD>"] = 0
    return vocab

vocab = build_vocab(train_texts, max_vocab=14375)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 0) for word in text.split()][:max_len][:max_len]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, vocab=vocab, max_len=35):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        sequence = torch.tensor(text_to_sequence(text, self.vocab, self.max_len), dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=False)

# Model
class TextCNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(embedding_dim, channels, kernel_size) for kernel_size in kernel_sizes])
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]
        pooled_outputs = [self.pool(conv_output).squeeze(-1) for conv_output in conv_outputs]
        concatenated = torch.cat(pooled_outputs, dim=1)
        x = self.dropout(concatenated)
        logit = self.fc(x)
        return logit

model = TextCNN(vocab_size=len(vocab), embedding_dim=104, channels=104, kernel_sizes=[3, 4, 5], dropout=0.3).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_texts, batch_labels in train_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_texts).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

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
                token_ids = tokenize(text, vocab, 35)
            else:
                token_ids = text_to_sequence(text, vocab, 35)
            return torch.tensor(token_ids, dtype=torch.long)

    test_loader = DataLoader(_UnlabeledTextDataset(test_df['text']), batch_size=64, shuffle=False, pin_memory=False)
    if FINAL_SUBMISSION:
        final_model = TextCNN(vocab_size=len(vocab) + 1, embedding_dim=104, channels=104, kernel_sizes=[3, 4, 5], dropout=0.3).to(device)
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
acc = accuracy_score(val_labels, (val_probs >= best_threshold).astype(int))
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')