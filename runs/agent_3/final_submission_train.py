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

# Preprocess data
train_df["text"] = train_df["keyword"] + " [SEP] " + train_df["text"]
test_df["text"] = test_df["keyword"] + " [SEP] " + test_df["text"]

# Sample training data if needed
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
if 'target' not in train_df.columns:
    raise ValueError('Missing required column: target')
if 'stratify_labels' in train_df.columns:
    stratify_labels = train_df['stratify_labels']
else:
    stratify_labels = None
X_train, X_val, y_train, y_val = train_test_split(train_df["text"], train_df["target"], test_size=0.2, random_state=42, stratify=stratify_labels)

# Define Dataset class
class DisasterTweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=192):
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
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

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

# Load tokenizer and model
model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create datasets
train_dataset = DisasterTweetDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_len=192)
val_dataset = DisasterTweetDataset(X_val.tolist(), y_val.tolist(), tokenizer, max_len=192)
test_dataset = DisasterTweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=192)
test_dataset = DisasterTweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=192)

if WRITE_SUBMISSION:
    test_dataset = DisasterTweetDataset(list(test_df["text"]), labels=None, tokenizer=tokenizer, max_len=192)

# Define training arguments
training_args = TrainingArguments(
    use_cpu=True,
    dataloader_pin_memory=False,
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-05,
    weight_decay=0.001,
    logging_dir='./logs',
    use_cuda=True,
    dataloader_pin_memory=False,
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    fp16=False,
    bf16=False,
    disable_tqdm=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# DRY_RUN: skip training
if not DRY_RUN:
    trainer.train()

# Predict validation logits
val_logits = trainer.predict(val_dataset).predictions
val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
val_probs = val_probs[:, 1]

# Choose best threshold
thresholds = np.linspace(0.3, 0.7, 41)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    val_preds = (val_logits[:, 1] > threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission: train on full data and predict test set
if FINAL_SUBMISSION:
    final_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    full_train_dataset = DisasterTweetDataset(train_df["text"].tolist(), train_df["target"].tolist(), tokenizer, max_len=192)
    final_trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=full_train_dataset
    )
    final_trainer.train()
    if WRITE_SUBMISSION:
        test_logits = final_trainer.predict(test_dataset).predictions
        test_preds = (test_logits[:, 1] > best_threshold).astype(int)
else:
    val_preds = (val_probs > best_threshold).astype(int)
    acc = accuracy_score(y_val, val_preds)

# Write submission if needed
if WRITE_SUBMISSION:
    os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
    test_df["target"] = test_preds
    test_df[["id", "target"]].to_csv(spec["submission_path"], index=False)

# Print metrics
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')