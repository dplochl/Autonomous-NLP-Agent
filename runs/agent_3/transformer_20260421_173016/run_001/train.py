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
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).fillna("")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).fillna("")

# Build text field
train_df["text"] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df["text"] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Sample train data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
X_train, X_val, y_train, y_val = train_test_split(train_df["text"], train_df["target"], test_size=0.2, random_state=42, stratify=train_df["target"])
stratify_labels = y_train if len(np.unique(y_train)) > 1 else None

# Tokenizer
model_name = "distilbert-base-uncased"
max_len = 128
tokenizer = AutoTokenizer.from_pretrained(model_name)

class DisasterTweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=tokenizer, max_len=max_len):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        if self.labels is not None:
            label = self.labels[idx]
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label, dtype=torch.long)}
        else:
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

train_dataset = DisasterTweetDataset(X_train.tolist(), y_train.tolist())
val_dataset = DisasterTweetDataset(X_val.tolist(), y_val.tolist())

# Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    fp16=False,
    disable_tqdm=True,
    learning_rate=2e-05,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
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
threshold_min = 0.3
threshold_max = 0.7
threshold_steps = 41
best_f1 = 0
best_threshold = 0.5

for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
    val_preds = (val_logits[:, 1] > threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Best F1: {best_f1}")

# Final submission
if FINAL_SUBMISSION or WRITE_SUBMISSION:
    if FINAL_SUBMISSION:
        # Retrain on full train data
        full_train_dataset = DisasterTweetDataset(train_df["text"].tolist(), train_df["target"].tolist())
        trainer.train_dataset = full_train_dataset
        trainer.train()

    # Predict test logits
    test_logits = trainer.predict(DisasterTweetDataset(list(test_df["text"]))).predictions

    # Convert logits to probabilities and apply threshold
    test_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)
    test_preds = (test_probs[:, 1] > best_threshold).astype(int)

    # Write submission
    if WRITE_SUBMISSION:
    submission_path = '/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260421_173016/run_001/submission.csv'
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df = pd.DataFrame({"id": test_df["id"], "target": test_preds})
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission_df.to_csv(submission_path, index=False)

# Metrics
val_preds = (val_logits[:, 1] > best_threshold).astype(int)
f1 = f1_score(y_val, val_preds)
acc = accuracy_score(y_val, val_preds)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')