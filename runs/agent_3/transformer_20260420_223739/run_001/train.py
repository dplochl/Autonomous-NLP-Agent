import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

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
    train_df = train_df.sample(n=min(16, len(train_df)), random_state=42)

# Stratify labels
stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels
)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=128)
        self.labels = list(labels) if labels is not None else None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)
test_dataset = TextDataset(list(test_df['text']))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-05,
    weight_decay=0.01,
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    fp16=False,
    disable_tqdm=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train if not DRY_RUN
if not DRY_RUN:
    trainer.train()

# Predict on validation and test sets
val_predictions = trainer.predict(val_dataset).predictions
test_predictions = trainer.predict(test_dataset).predictions


# Convert logits to probabilities
val_probs = np.exp(val_predictions) / np.sum(np.exp(val_predictions), axis=1, keepdims=True)
test_probs = np.exp(test_predictions) / np.sum(np.exp(test_predictions), axis=1, keepdims=True)

# Choose best cutoff
best_f1 = 0
best_threshold = 0.5
thresholds = np.linspace(0.1, 0.9, 81)
for threshold in thresholds:
    val_preds = (val_probs[:, 1] >= threshold).astype(int)
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Evaluate on validation set
val_preds = (val_probs[:, 1] >= best_threshold).astype(int)
f1 = f1_score(val_labels, val_preds)
acc = accuracy_score(val_labels, val_preds)

# Prepare submission
test_preds = (test_probs[:, 1] >= best_threshold).astype(int)
submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})

# Create submissions directory if it doesn't exist
os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260420_223739/run_001/submission.csv"), exist_ok=True)

# Save submission
submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260420_223739/run_001/submission.csv", index=False)

# Print metrics
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')