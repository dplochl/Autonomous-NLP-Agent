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
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).fillna("")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).fillna("")

# Preprocess text
train_df["text"] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df["text"] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# DRY_RUN handling
if DRY_RUN:
    train_df = train_df.sample(n=min(16, len(train_df)), random_state=42)
else:
    if TRAIN_FRACTION < 1.0:
        train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
y = train_df["target"]
stratify_labels = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
train_texts, val_texts, train_labels, val_labels = train_test_split(train_df["text"], y, test_size=0.2, random_state=42, stratify=stratify_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)

# Load tokenizer and model
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_encodings = train_texts.apply(lambda x: tokenize_function({"text": [x]})).tolist()
val_encodings = val_texts.apply(lambda x: tokenize_function({"text": [x]})).tolist()

# Create Dataset class
class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = list(labels) if labels is not None else None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)
test_dataset = TextDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=128)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
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
val_logits = trainer.predict(val_dataset).predictions
test_logits = trainer.predict(test_df["text"].apply(lambda x: tokenize_function({"text": [x]})).tolist()).predictions

# Convert logits to probabilities
val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True)) / np.sum(np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True)), axis=1, keepdims=True)
test_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True)) / np.sum(np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True)), axis=1, keepdims=True)

# Choose best threshold and probability orientation
thresholds = np.linspace(0.01, 0.99, 99)
best_f1 = -1.0
best_threshold = 0.5
best_column = 1
best_val_preds = None
for column in (1, 0):
    for threshold in thresholds:
        candidate_preds = (val_probs[:, column] > threshold).astype(int)
        candidate_f1 = f1_score(val_df['target'], candidate_preds)
        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best_threshold = threshold
            best_column = column
            best_val_preds = candidate_preds
val_preds = best_val_preds if best_val_preds is not None else (val_probs[:, 1] > 0.5).astype(int)
f1 = best_f1
acc = accuracy_score(val_df['target'], val_preds)
test_preds = (test_probs[:, best_column] > best_threshold).astype(int)

submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})


# Create submissions directory and save CSV
os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/deberta_20260421_134928/run_001/submission.csv"), exist_ok=True)
submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/deberta_20260421_134928/run_001/submission.csv", index=False)

# Print metrics
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')