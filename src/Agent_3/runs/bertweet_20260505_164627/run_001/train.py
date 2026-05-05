import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Constants
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))
submission_path = os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', '/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bertweet_20260505_164627/run_001/submission.csv')

# Load data
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
train_df['text'] = train_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)
test_df['text'] = test_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)

# Sample train data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)


if DRY_RUN:
    train_df = train_df.sample(n=min(16, len(train_df)), random_state=42)
    test_df = test_df.head(16)

# Split data
y = train_df['target']
stratify_labels = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
train_texts, val_texts, train_labels, val_labels = train_test_split(train_df['text'], y, test_size=0.2, random_state=42, stratify=stratify_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)

# Tokenizer and model
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Dataset class
class DisasterTweetDataset(Dataset):
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
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        if self.labels is not None:
            label = self.labels[idx]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

# Create datasets
train_dataset = DisasterTweetDataset(train_texts, train_labels)
val_dataset = DisasterTweetDataset(val_texts, val_labels)
test_dataset = DisasterTweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=128)
test_dataset = DisasterTweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=128)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1.5e-05,
    weight_decay=0.01,
    logging_dir='./logs',
    use_cpu=True,
    no_cuda=True,
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
val_logits = trainer.predict(val_dataset).predictions
val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
val_predictions = val_probs / val_probs.sum(axis=1, keepdims=True)
val_probabilities = np.exp(val_predictions) / np.sum(np.exp(val_predictions), axis=1, keepdims=True)
val_labels_pred = (val_probabilities[:, 1] >= 0.5).astype(int)

# Choose best threshold
threshold_min = 0.3
threshold_max = 0.7
threshold_steps = 41
best_threshold = 0.5
best_f1 = 0

for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
    val_labels_pred_thresholded = (val_probabilities[:, 1] >= threshold).astype(int)
    f1 = f1_score(val_labels, val_labels_pred_thresholded)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# FINAL_SUBMISSION check
if FINAL_SUBMISSION:
    # Retrain on full train data with the best threshold
    final_train_dataset = DisasterTweetDataset(train_df['text'], train_df['target'])
    trainer.train(resume_from_checkpoint=False, train_dataset=final_train_dataset)
    trainer.train(resume_from_checkpoint=False, train_dataset=final_train_dataset)

# Predict test logits if WRITE_SUBMISSION is true
if WRITE_SUBMISSION:
    test_dataset = DisasterTweetDataset(test_df['text'], labels=None)
    test_predictor = trainer if not FINAL_SUBMISSION else final_trainer
    test_logits = test_predictor.predict(test_dataset).logits
test_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
test_predictions = test_probs / test_probs.sum(axis=1, keepdims=True)
test_probabilities = np.exp(test_predictions) / np.sum(np.exp(test_predictions), axis=1, keepdims=True)
test_labels_pred = (test_probabilities[:, 1] >= best_threshold).astype(int)

    # Write submission CSV
    os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bertweet_20260505_164627/run_001/submission.csv"), exist_ok=True)
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_labels_pred})
    submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bertweet_20260505_164627/run_001/submission.csv", index=False)

# Metrics
acc = accuracy_score(val_labels, val_labels_pred)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')