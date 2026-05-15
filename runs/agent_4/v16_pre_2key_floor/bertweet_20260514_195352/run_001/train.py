import os
import re
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
submission_path = os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', '/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/bertweet_20260514_195352/run_001/submission.csv')
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Fill missing values
for col in ["keyword", "location", "text"]:
    train_df[col] = train_df[col].fillna("")
    test_df[col] = test_df[col].fillna("")

# Build text field
train_df["text"] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df["text"] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Normalize URLs and mentions
train_df["text"] = train_df["text"].str.replace(r"http\S+", "HTTPURL", regex=True)
train_df["text"] = train_df["text"].str.replace(r"@\S+", "@USER", regex=True)
test_df["text"] = test_df["text"].str.replace(r"http\S+", "HTTPURL", regex=True)
test_df["text"] = test_df["text"].str.replace(r"@\S+", "@USER", regex=True)

# Load model and tokenizer
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=tokenizer, max_len=128):
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
train_dataset = TweetDataset(train_texts, train_labels, tokenizer=tokenizer, max_len=128)
val_dataset = TweetDataset(val_texts, val_labels, tokenizer=tokenizer, max_len=128)
test_dataset = TweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=128)

# Training arguments
training_args = TrainingArguments(

    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1.5e-05,
    weight_decay=0.01,
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

if not DRY_RUN:
    trainer.train()

# Predict validation logits
val_logits = trainer.predict(val_dataset).predictions
val_probs = np.exp(val_logits) / np.sum(np.exp(val_logits), axis=-1, keepdims=True)

# Tune threshold
threshold_min = 0.3
threshold_max = 0.7
threshold_steps = 41
best_threshold = 0.5
best_f1 = 0

for candidate in np.linspace(threshold_min, threshold_max, threshold_steps):
    val_preds = (val_probs[:, 1] >= candidate).astype(int)
    val_f1 = f1_score(val_labels, val_preds)
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_threshold = candidate

# Compute accuracy
acc = accuracy_score(val_labels, (val_probs[:, 1] >= best_threshold).astype(int))

print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')

# Final submission
if FINAL_SUBMISSION:
    if not DRY_RUN:
        full_train_dataset = TweetDataset(train_df['text'].tolist(), train_labels, tokenizer, max_len=128)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=full_train_dataset,
            eval_dataset=val_dataset
        )
        trainer.train()

    test_predictor = final_trainer if FINAL_SUBMISSION else trainer
    test_logits = test_predictor.predict(test_dataset).predictions
    test_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=-1, keepdims=True)
    test_preds = (test_probs[:, 1] >= best_threshold).astype(int)

    if WRITE_SUBMISSION:
        submission_dir = os.path.dirname(spec["submission_path"])
        if not os.path.exists(submission_dir):
            os.makedirs(submission_dir)
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'target': test_preds
        })
        submission_df.to_csv(spec["submission_path"], index=False)