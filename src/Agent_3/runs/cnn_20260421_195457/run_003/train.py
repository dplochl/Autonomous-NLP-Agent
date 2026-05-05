import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

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

# Preprocessing
train_df['text'] = train_df['keyword'].fillna('') + ' [SEP] ' + train_df['text']
test_df['text'] = test_df['keyword'].fillna('') + ' [SEP] ' + test_df['text']

if DRY_RUN:
    train_df = train_df.head(200)

# Sample if needed
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
    vocab = {word: idx+2 for idx, (word, _) in enumerate(sorted_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

spec = {
    'batch_size': 64,
    'channels': 128,
    'dropout': 0.3,
    'dry_run_head': 200,
    'embedding_dim': 128,
    'epochs': 2,
    'experiment_name': 'cnn_20260421_195457_run_03',
    'kernel_sizes': [3, 4, 5],
    'learning_rate': 0.001,
    'max_len': 48,
    'max_vocab': 20000,
    'submission_path': '/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260421_195457/run_003/submission.csv',
    'threshold_max': 0.7,
    'threshold_min': 0.3,
    'threshold_steps': 41,
    'val_size': 0.2
}
vocab = build_vocab(train_df['text'], spec['max_vocab'])

def text_to_sequence(text, vocab, max_len):
    words = text.split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

train_df['sequence'] = train_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec['max_len']))
test_df['sequence'] = test_df['text'].apply(lambda x: text_to_sequence(x, vocab, spec['max_len']))

# Dataset and DataLoader
class TweetDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float32)
            return sequence, label
        else:
            return sequence

train_sequences = train_df['sequence'].tolist()
train_labels = train_df['target'].tolist()

stratify_labels = train_labels if len(set(train_labels)) > 1 else None
train_seq, val_seq, train_label, val_label = train_test_split(
    train_sequences, train_labels, test_size=spec['val_size'], random_state=42, stratify=stratify_labels
)

train_dataset = TweetDataset(train_seq, train_label)
val_dataset = TweetDataset(val_seq, val_label)
test_dataset = TweetDataset(test_df['sequence'].tolist())

train_loader = DataLoader(train_dataset, batch_size=spec['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=spec['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=spec['batch_size'])

# Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]
        pool_outputs = [torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in conv_outputs]
        concat_output = torch.cat(pool_outputs, dim=1)
        output = self.dropout(concat_output)
        logit = self.fc(output)
        return logit

model = TextCNN(
    vocab_size=len(vocab),
    embedding_dim=spec['embedding_dim'],
    channels=spec['channels'],
    kernel_sizes=spec['kernel_sizes'],
    dropout=spec['dropout']
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=spec['learning_rate'])

# Training
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if not DRY_RUN:
    train_model(model, train_loader, criterion, optimizer, spec['epochs'])

# Validation
def validate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            outputs = model(sequences).squeeze()
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)

val_preds, val_labels = validate_model(model, val_loader)
best_threshold = None
best_f1 = 0

for threshold in np.linspace(spec['threshold_min'], spec['threshold_max'], spec['threshold_steps']):
    preds_binary = (val_preds > threshold).astype(int)
    f1 = f1_score(val_labels, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission
if FINAL_SUBMISSION:
    train_dataset_full = TweetDataset(train_sequences + val_seq, train_label + val_label)
    train_loader_full = DataLoader(train_dataset_full, batch_size=spec['batch_size'], shuffle=True)
    train_model(model, train_loader_full, criterion, optimizer, spec['epochs'])

test_preds = []
model.eval()
with torch.no_grad():
    for sequences in test_loader:

        if isinstance(sequences, (list, tuple)):

            sequences = sequences[0]

        if isinstance(sequences, (list, tuple)):

            sequences = sequences[0]

        if isinstance(sequences, (list, tuple)):
    sequences = sequences[0]
  if isinstance(sequences, torch.Tensor):
    sequences = sequences.squeeze()

        if isinstance(sequences, (list, tuple)):
                sequences = sequences[0]
                if isinstance(sequences, torch.Tensor):
                    sequences = sequences.squeeze()
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.squeeze()

        if isinstance(sequences, (list, tuple)):

            sequences = sequences[0]
        outputs = model(sequences).squeeze()
        preds = torch.sigmoid(outputs)
        test_preds.extend(preds.numpy())

if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(spec['submission_path']), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': (np.array(test_preds) > best_threshold).astype(int)})
    submission_df.to_csv(spec['submission_path'], index=False)

# Metrics
preds_binary = (val_preds > best_threshold).astype(int)
acc = accuracy_score(val_labels, preds_binary)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')