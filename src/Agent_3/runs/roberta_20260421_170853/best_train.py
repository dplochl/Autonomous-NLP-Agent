import os
import numpy as np
import pandas as pd
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
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).fillna("")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).fillna("")

# Preprocess data
train_df["text"] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df["text"] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Sample data
if DRY_RUN:
    train_df = train_df.sample(n=min(16, len(train_df)), random_state=42)
elif TRAIN_FRACTION < 1.0:
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
    "experiment_name": "roberta_20260421_170853_run_01",
    "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/roberta_20260421_170853/run_001/submission.csv"
}

tokenizer = AutoTokenizer.from_pretrained(spec["model_name"])
model = AutoModelForSequenceClassification.from_pretrained(spec["model_name"], num_labels=2)

# Create Dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=tokenizer, max_len=spec["max_len"]):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if self.labels is not None:
            label = self.labels[idx]
            return {
                'input_ids': input_ids.squeeze(),
            'attention_mask': attention_mask.squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }

# Create datasets
train_dataset = TweetDataset(train_texts, train_labels)
val_dataset = TweetDataset(val_texts, val_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=spec["num_epochs"],
    per_device_train_batch_size=spec["train_batch_size"],
    per_device_eval_batch_size=spec["eval_batch_size"],
    learning_rate=spec["learning_rate"],
    weight_decay=spec["weight_decay"],
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

if not DRY_RUN:
    trainer.train()

# Predict validation logits
val_logits = trainer.predict(val_dataset).predictions

# Choose best threshold
best_threshold = 0.5
best_f1 = 0
for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    val_preds = (val_logits[:, 1] > threshold).astype(int)
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Evaluate on validation set
val_preds = (val_logits[:, 1] > best_threshold).astype(int)
f1 = f1_score(val_labels, val_preds)
acc = accuracy_score(val_labels, val_preds)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')

# Final submission
if WRITE_SUBMISSION:
    if FINAL_SUBMISSION:
        # Retrain on full train data with best threshold
        final_train_dataset = TweetDataset(train_df["text"], train_df["target"])
        trainer.train_dataset = final_train_dataset
        trainer.train()

    test_dataset = TweetDataset(test_df["text"], labels=None)
    test_logits = trainer.predict(test_dataset).predictions
    test_preds = (test_logits[:, 1] > best_threshold).astype(int)

    submission_df = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
    submission_df.to_csv(spec["submission_path"], index=False)