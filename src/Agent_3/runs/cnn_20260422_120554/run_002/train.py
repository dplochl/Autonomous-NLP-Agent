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

# Fill missing values
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Build text field
train_df['text'] = train_df['keyword'] + ' [SEP] ' + train_df['text']
test_df['text'] = test_df['keyword'] + ' [SEP] ' + test_df['text']

# DRY_RUN handling
if DRY_RUN:
    train_df = train_df.head(int(os.environ.get("AGENT_DRY_RUN_HEAD", "200")))

# Sample training data if needed
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
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: i + 2 for i, (word, _) in enumerate(sorted_words[:max_vocab - 2])}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(train_texts, max_vocab=20000)

def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    return sequence + [vocab['<PAD>']] * (max_len - len(sequence))

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_seq = torch.tensor(text_to_sequence(self.texts.iloc[idx], self.vocab, self.max_len), dtype=torch.long)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
        return text_seq, label

train_dataset = TextDataset(train_texts, train_labels, vocab, max_len=56)
val_dataset = TextDataset(val_texts, val_labels, vocab, max_len=56)
test_dataset = TextDataset(test_df['text'], pd.Series([0] * len(test_df)), vocab, max_len=56)

train_loader = DataLoader(train_dataset, batch_size=56, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=56, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=56, shuffle=False)

# CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        x = [torch.relu(conv(x)) for conv in self.convs]
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # Global max pooling
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x).sigmoid()

model = TextCNN(vocab_size=len(vocab), embedding_dim=192, channels=160, kernel_sizes=[3, 4, 5], dropout=0.4)
optimizer = optim.Adam(model.parameters(), lr=0.0008)
criterion = nn.BCELoss()

# Training loop
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    for batch_texts, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_texts).squeeze()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

def validate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_texts, batch_labels in dataloader:
            outputs = model(batch_texts).squeeze().cpu().numpy()
            preds = (outputs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
    return f1_score(all_labels, all_preds), accuracy_score(all_labels, all_preds)

if not DRY_RUN:
    for epoch in range(3):
        train_model(model, train_loader, optimizer, criterion)
        val_f1, val_acc = validate_model(model, val_loader)
        print(f'Epoch {epoch+1}, Val F1: {val_f1:.4f}, Val Accuracy: {val_acc:.4f}')

# Choose best threshold
best_threshold = 0.5
best_f1 = 0

if not DRY_RUN:
    for threshold in np.linspace(0.3, 0.7, 41):
        val_preds = []
        with torch.no_grad():
            for batch_texts, _ in val_loader:
                outputs = model(batch_texts).squeeze().cpu().numpy()
                preds = (outputs > threshold).astype(int)
                val_preds.extend(np.atleast_1d(preds))
        f1 = f1_score(val_labels, val_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

# Final submission model training
if FINAL_SUBMISSION and not DRY_RUN:
    final_model = TextCNN(vocab_size=len(vocab), embedding_dim=192, channels=160, kernel_sizes=[3, 4, 5], dropout=0.4)
    optimizer_final = optim.Adam(final_model.parameters(), lr=0.0008)
    criterion_final = nn.BCELoss()
    
    final_train_dataset = TextDataset(train_df['text'], train_df['target'], vocab, max_len=56)
    final_train_loader = DataLoader(final_train_dataset, batch_size=56, shuffle=True)
    
    for epoch in range(3):
        train_model(final_model, final_train_loader, optimizer_final, criterion_final)

# Predict test set
test_preds = []
with torch.no_grad():
    if FINAL_SUBMISSION and not DRY_RUN:
        model = final_model
    for batch_texts, _ in test_loader:
        outputs = model(batch_texts).squeeze().cpu().numpy()
        preds = (outputs > best_threshold).astype(int)
        test_preds.extend(np.atleast_1d(preds))

# Write submission
if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(os.environ.get("AGENT_SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260422_120554/run_002/submission.csv")), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
    submission_df.to_csv(os.environ.get("AGENT_SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/cnn_20260422_120554/run_002/submission.csv"), index=False)

# Print final metrics
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(val_acc, 4)) + '}')