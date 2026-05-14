import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

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
train_df["text"] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df["text"] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Spec
spec = {
  "architecture": "RoBERTa",
  "model_name": "roberta-base",
  "max_len": 128,
  "train_batch_size": 16,
  "eval_batch_size": 16,
  "learning_rate": 1.5e-05,
  "weight_decay": 0.01,
  "num_epochs": 3,
  "val_size": 0.2,
  "threshold_min": 0.3,
  "threshold_max": 0.7,
  "threshold_steps": 41,
  "dry_run_head": 16,
  "experiment_name": "roberta_20260514_152035_run_01",
  "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/roberta_20260514_152035/run_001/submission.csv"
}

# Sample train data if needed
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
X = train_df["text"]
y = train_df["target"]

stratify_labels = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, test_size=spec["val_size"], random_state=42, stratify=stratify_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(spec["model_name"])
model = AutoModelForSequenceClassification.from_pretrained(spec["model_name"], num_labels=2)

# Dataset class
class DisasterTweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=tokenizer, max_len=116):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Create datasets
train_dataset = DisasterTweetDataset(train_texts, train_labels, tokenizer=tokenizer, max_len=116)
val_dataset = DisasterTweetDataset(val_texts, val_labels, tokenizer=tokenizer, max_len=116)
test_dataset = DisasterTweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=116)

# Training arguments
training_args = TrainingArguments(



    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=15,
    per_device_eval_batch_size=16,
    learning_rate=1.5e-05,
    weight_decay=0.01,
    eval_steps=100,
    logging_dir="./logs",
    use_cpu=True,
    dataloader_pin_memory=False,
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    fp16=False,
    bf16=False,
    disable_tqdm=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# DRY_RUN check
if not DRY_RUN:
    trainer.train()

# Predict validation logits
val_predictions = trainer.predict(val_dataset).predictions
val_probs = np.exp(val_predictions) / np.sum(np.exp(val_predictions), axis=1, keepdims=True)
val_preds = (val_probs[:, 1] >= 0.5).astype(int)

# Choose best threshold if FINAL_SUBMISSION
if FINAL_SUBMISSION:
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
        preds = (val_probs[:, 1] >= threshold).astype(int)
        f1 = f1_score(val_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Retrain on full train data with best threshold
    model = AutoModelForSequenceClassification.from_pretrained(spec["model_name"], num_labels=2)
    trainer.model = model
    trainer.train()

# Predict test logits if WRITE_SUBMISSION
if WRITE_SUBMISSION:
    test_dataset = DisasterTweetDataset(list(test_df["text"]), labels=None)
    test_predictions = trainer.predict(test_dataset).predictions
    test_probs = np.exp(test_predictions) / np.sum(np.exp(test_predictions), axis=1, keepdims=True)
    test_preds = (test_probs[:, 1] >= best_threshold if FINAL_SUBMISSION else 0.5).astype(int)

    # Write submission
    os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
    submission_df = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission_df.to_csv(spec["submission_path"], index=False)

# Metrics
f1 = f1_score(val_labels, val_preds)
acc = accuracy_score(val_labels, val_preds)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')