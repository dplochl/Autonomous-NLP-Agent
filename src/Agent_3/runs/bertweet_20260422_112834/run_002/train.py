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
submission_path = os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', '/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bertweet_20260422_112834/run_001/submission.csv')

# Load data
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Preprocess text
def preprocess_text(text):
    return re.sub(r"http\S+", "HTTPURL", re.sub(r"@\S+", "@USER", text))

train_df['text'] = train_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)
test_df['text'] = test_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)

# Split data
if DRY_RUN:
    train_df = train_df.sample(n=min(16, len(train_df)), random_state=42)
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None
train_texts, val_texts, y_train, y_val = train_test_split(train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels)

# Tokenizer and model
model_name = 'vinai/bertweet-base'
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
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        if self.labels is not None:
            return {
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(mask, dtype=torch.long),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(mask, dtype=torch.long)
            }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-05,
    weight_decay=0.001,
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    fp16=False,
    disable_tqdm=True
)

# Create datasets
train_dataset = TweetDataset(train_texts, y_train)
val_dataset = TweetDataset(val_texts, y_val)

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
val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

# Choose best threshold
best_threshold = 0.5
best_f1 = 0
for threshold in np.linspace(0.3, 0.7, 41):
    val_preds = (val_probs[:, 1] > threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission training
if FINAL_SUBMISSION:
    final_train_dataset = TweetDataset(train_df['text'], train_df['target'])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_train_dataset
    )
    trainer.train()

# Predict test logits and write submission
if WRITE_SUBMISSION:
    test_texts = list(test_df['text'])
    test_dataset = TweetDataset(test_texts, labels=None)
    test_logits = trainer.predict(test_dataset).predictions
    test_preds = (test_logits[:, 1] > best_threshold).astype(int)

    # Create submission directory if it doesn't exist
    os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bertweet_20260422_112834/run_001/submission.csv"), exist_ok=True)

    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_preds
    })
    submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bertweet_20260422_112834/run_001/submission.csv", index=False)

# Metrics
val_preds = (val_probs[:, 1] > best_threshold).astype(int)
f1 = f1_score(y_val, val_preds)
acc = accuracy_score(y_val, val_preds)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')