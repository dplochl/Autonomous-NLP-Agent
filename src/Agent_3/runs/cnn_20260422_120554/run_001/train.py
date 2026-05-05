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
    train_df = train_df.head(int(os.environ.get("DRY_RUN_HEAD", "200")))

# Sample training data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Stratify labels
stratify_labels = train_df['target'] if len(train_df['target'].unique()) > 1 else None

# Train-test split
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
    vocab = {word: idx+2 for idx, (word, _) in enumerate(sorted_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(train_texts, max_vocab=20000)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_sequences = np.array([text_to_sequence(text, vocab, 48) for text in train_texts])
val_sequences = np.array([text_to_sequence(text, vocab, 48) for text in val_texts])

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
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_dataset = TextDataset(train_sequences, train_labels)
val_dataset = TextDataset(val_sequences, val_labels)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# CNN Model
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
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, channels * len(kernel_sizes))
        x = self.dropout(x)
        return self.fc(x).sigmoid()

model = TextCNN(vocab_size=len(vocab), embedding_dim=128, channels=128, kernel_sizes=[3, 4, 5], dropout=0.3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
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
model.eval()
val_preds = []
with torch.no_grad():
    for sequences in val_loader:

        if isinstance(sequences, (list, tuple)):

            sequences = sequences[0]

        if isinstance(sequences, (list, tuple)):
    sequences = sequences[0]
                assert isinstance(sequences, torch.Tensor), 'Input to the model must be a tensor'
outputs = model(sequences).squeeze().numpy()
        val_preds.extend(outputs)

val_labels_np = np.array(val_labels)
best_threshold = 0.5
best_f1 = 0

for threshold in np.linspace(0.3, 0.7, 41):
    preds = (np.array(val_preds) > threshold).astype(int)
    f1 = f1_score(val_labels_np, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission training
if FINAL_SUBMISSION:
    final_model = TextCNN(vocab_size=len(vocab), embedding_dim=128, channels=128, kernel_sizes=[3, 4, 5], dropout=0.3)
    final_optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    final_criterion = nn.BCELoss()
    
    full_train_sequences = np.array([text_to_sequence(text, vocab, 48) for text in train_df['text']])
    full_train_labels = np.array(train_df['target'])
    full_train_dataset = TextDataset(full_train_sequences, full_train_labels)
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    
    train_model(final_model, full_train_loader, final_optimizer, final_criterion, epochs=3)

# Test prediction
test_sequences = np.array([text_to_sequence(text, vocab, 48) for text in test_df['text']])
test_dataset = TextDataset(test_sequences)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

final_model = model
final_model.eval()
test_preds = []
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
        outputs = final_model(sequences).squeeze().numpy()
        test_preds.extend(outputs)

# Write submission
if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(os.environ.get("SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260422_120554/run_001/submission.csv")), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': (np.array(test_preds) > best_threshold).astype(int)})
    submission_df.to_csv(os.environ.get("SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260422_120554/run_001/submission.csv"), index=False)

# Metrics
preds = (np.array(val_preds) > best_threshold).astype(int)
acc = accuracy_score(val_labels_np, preds)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')