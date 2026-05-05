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
train_df['text'] = train_df.apply(lambda row: f"{row.keyword} [SEP] {row.text}" if pd.notna(row.keyword) else row.text, axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row.keyword} [SEP] {row.text}" if pd.notna(row.keyword) else row.text, axis=1)

# DRY_RUN
if DRY_RUN:
    train_df = train_df.head(200)

# Define stratify_labels
stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels
)
train_labels = np.asarray(train_labels)
val_labels = np.asarray(val_labels)

# Build tokenizer/vocabulary
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

# Convert texts to padded integer sequences
def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_sequences = np.array([text_to_sequence(text, vocab, 48) for text in train_texts])
val_sequences = np.array([text_to_sequence(text, vocab, 48) for text in val_texts])
test_sequences = np.array([text_to_sequence(text, vocab, 48) for text in test_df['text']])

# Define Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_dataset = TextDataset(train_sequences, train_labels)
val_dataset = TextDataset(val_sequences, val_labels)
test_dataset = TextDataset(test_sequences)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        x = [nn.functional.relu(conv(x)) for conv in self.convs]
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # Global max pooling
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x).sigmoid()

model = TextCNN(vocab_size=len(vocab), embedding_dim=128, channels=64, kernel_sizes=[3, 4, 5], dropout=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
if not DRY_RUN:
    for epoch in range(3):
        model.train()
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = nn.BCELoss()(outputs, labels)
            loss.backward()
            optimizer.step()

# Validation probabilities
model.eval()
val_probs = []
with torch.no_grad():
    for sequences, _ in val_loader:
        outputs = model(sequences).squeeze().numpy()
        val_probs.extend(outputs)

# Test probabilities
test_probs = []
with torch.no_grad():
    for sequences in test_loader:

        if isinstance(sequences, (list, tuple)):

            sequences = sequences[0]
        outputs = model(sequences).squeeze().numpy()
        test_probs.extend(outputs)

# Choose best cutoff
best_f1 = 0
best_threshold = 0.5
for threshold in np.linspace(0.3, 0.7, 41):
    val_preds = (np.array(val_probs) > threshold).astype(int)
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Generate submission
test_preds = (np.array(test_probs) > best_threshold).astype(int)
submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260420_214747/run_001/submission.csv"), exist_ok=True)
submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260420_214747/run_001/submission.csv", index=False)

# Metrics
val_preds = (np.array(val_probs) > best_threshold).astype(int)
acc = accuracy_score(val_labels, val_preds)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')