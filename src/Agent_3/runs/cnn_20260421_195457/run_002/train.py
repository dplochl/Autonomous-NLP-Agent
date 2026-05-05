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
stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None
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
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab-2]
    word_to_idx = {word: idx+2 for idx, (word, _) in enumerate(sorted_words)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    return word_to_idx

vocab = build_vocab(train_texts, max_vocab=25000)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_sequences = np.array([text_to_sequence(text, vocab, 56) for text in train_texts])
val_sequences = np.array([text_to_sequence(text, vocab, 56) for text in val_texts])
test_sequences = np.array([text_to_sequence(text, vocab, 56) for text in test_df['text']])

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
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_dataset = TextDataset(train_sequences, train_labels)
val_dataset = TextDataset(val_sequences, val_labels)
test_dataset = TextDataset(test_sequences)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        conv_outputs = [self.pool(torch.relu(conv(x))).squeeze(-1) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x).sigmoid()

model = TextCNN(vocab_size=len(vocab), embedding_dim=192, channels=144, kernel_sizes=[3, 4, 5], dropout=0.25)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.BCELoss()

# Training
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if not DRY_RUN:
    train_model(model, train_loader, optimizer, criterion, epochs=3)

# Validation
def validate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            outputs = model(sequences).squeeze().numpy()
            preds = (outputs > 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy())
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc

f1, acc = validate_model(model, val_loader)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')

# Final submission
if FINAL_SUBMISSION:
    # Retrain on full train data
    full_train_dataset = TextDataset(train_sequences, train_labels)
    full_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True)
    train_model(model, full_train_loader, optimizer, criterion, epochs=3)

# Choose best threshold
best_threshold = None
best_f1 = 0
for threshold in np.linspace(0.3, 0.7, 41):
    model.eval()
    all_preds = []
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

                        sequences = sequences[0]
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
            preds = (outputs > threshold).astype(int)
            all_preds.extend(preds)
    if WRITE_SUBMISSION:
        submission_df = pd.DataFrame({'id': test_df['id'], 'target': all_preds})
        os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260421_195457/run_002/submission.csv"), exist_ok=True)
        submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260421_195457/run_002/submission.csv", index=False)

    # Assuming we have validation data to choose the best threshold
    val_f1, _ = validate_model(model, val_loader)
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_threshold = threshold

print(f'Best threshold: {best_threshold}')