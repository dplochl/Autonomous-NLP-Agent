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
submission_path = os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', '/Users/niccogermani/Library/Containers/com.apple.iMovieApp/Data/Documents/Catolica/apa-disaster-tweets-clean/src/Agent_4/runs/bertweet_20260515_131228/run_001/submission.csv')

# Load data
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
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
train_df["text"] = train_df["keyword"] + " [SEP] " + train_df["text"]
test_df["text"] = test_df["keyword"] + " [SEP] " + test_df["text"]

# Normalize URLs and mentions
def normalize_text(text):
    return re.sub(r"http\S+", "HTTPURL", text)
    return re.sub(r"@\S+", "@USER", text)

train_df["text"] = train_df["text"].apply(normalize_text)
test_df["text"] = test_df["text"].apply(normalize_text)

# Load model and tokenizer
spec = {
  "architecture": "BERTweet",
  "model_name": "vinai/bertweet-base",
  "max_len": 144,
  "train_batch_size": 13,
  "eval_batch_size": 16,
  "learning_rate": 1e-05,
  "weight_decay": 0.01,
  "num_epochs": 3,
  "val_size": 0.2,
  "threshold_min": 0.3,
  "threshold_max": 0.7,
  "threshold_steps": 41,
  "dry_run_head": 16,
  "experiment_name": "bertweet_20260515_131228_run_01",
  "submission_path": "/Users/niccogermani/Library/Containers/com.apple.iMovieApp/Data/Documents/Catolica/apa-disaster-tweets-clean/src/Agent_4/runs/bertweet_20260515_131228/run_001/submission.csv"
}

stratify_labels = train_df['target']
train_texts, val_texts, y_train, val_labels = train_test_split(train_df['text'], train_df['target'], test_size=spec['val_size'], random_state=SAMPLE_SEED, stratify=stratify_labels)
tokenizer = AutoTokenizer.from_pretrained(spec["model_name"], use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(spec["model_name"])

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=tokenizer, max_len=144):
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
train_dataset = TweetDataset(train_texts, y_train, tokenizer=tokenizer, max_len=144)
val_dataset = TweetDataset(val_texts, val_labels, tokenizer=tokenizer, max_len=144)
test_dataset = TweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=144)

# Training arguments
training_args = TrainingArguments(




    output_dir="./results",
    num_train_epochs=spec["num_epochs"],
    per_device_train_batch_size=spec["train_batch_size"],
    per_device_eval_batch_size=spec["eval_batch_size"],
    learning_rate=spec["learning_rate"],
    weight_decay=spec["weight_decay"],
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
y_val = val_dataset.labels
val_labels = np.array(y_val)

# Tune threshold
best_threshold = 0.5
best_f1 = 0.0

for candidate in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    val_pred = (val_probs[:, 1] >= candidate).astype(int)
    val_f1 = f1_score(val_labels, val_pred)
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_threshold = candidate

# Compute accuracy
acc = accuracy_score(val_labels, (val_probs[:, 1] >= best_threshold).astype(int))

print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')

# Final submission
if WRITE_SUBMISSION:
    if FINAL_SUBMISSION:
        full_train_dataset = DisasterTweetDataset(train_df["text"].tolist(), train_df["target"].tolist())
        final_trainer = Trainer(
            model=AutoModelForSequenceClassification.from_pretrained(spec["model_name"]),
            args=training_args,
            train_dataset=full_train_dataset
        )
        final_trainer.train()

    test_logits = trainer.predict(DisasterTweetDataset(list(test_df["text"]))).predictions
    test_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=-1, keepdims=True)
    test_pred = (test_probs[:, 1] >= best_threshold).astype(int)

    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "target": test_pred
    })

    os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
    submission_df.to_csv(spec["submission_path"], index=False)