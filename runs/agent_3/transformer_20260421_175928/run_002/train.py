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
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Spec
spec = {
    "architecture": "Transformer",
    "model_name": "distilbert-base-uncased",
    "max_len": 128,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "learning_rate": 5e-05,
    "weight_decay": 0.001,
    "num_epochs": 3,
    "val_size": 0.2,
    "threshold_min": 0.3,
    "threshold_max": 0.7,
    "threshold_steps": 41,
    "dry_run_head": 16,
    "experiment_name": "transformer_20260421_175928_run_02",
    "submission_path": "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260421_175928/run_002/submission.csv"
}

# Preprocess data
train_df['text'] = train_df['keyword'] + " [SEP] " + train_df['text']
test_df['text'] = test_df['keyword'] + " [SEP] " + test_df['text']

if DRY_RUN:
    train_df = train_df.sample(n=min(spec["dry_run_head"], len(train_df)), random_state=42)

# Sample data if needed
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
X = train_df['text']
y = train_df['target']

stratify_labels = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, test_size=spec["val_size"], random_state=42, stratify=stratify_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(spec["model_name"])
model = AutoModelForSequenceClassification.from_pretrained(spec["model_name"], num_labels=2)

class DisasterTweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=tokenizer, max_len=spec["max_len"]):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if self.labels is not None:
            label = self.labels[idx]
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }

train_dataset = DisasterTweetDataset(train_texts, train_labels)
val_dataset = DisasterTweetDataset(val_texts, val_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=spec["num_epochs"],
    per_device_train_batch_size=spec["train_batch_size"],
    per_device_eval_batch_size=spec["eval_batch_size"],
    warmup_steps=500,
    weight_decay=spec["weight_decay"],
    logging_dir='./logs',
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    fp16=False,
    disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

if not DRY_RUN:
    trainer.train()

# Predict validation logits
val_predictions = trainer.predict(val_dataset).predictions
val_probabilities = np.exp(val_predictions) / np.sum(np.exp(val_predictions), axis=-1, keepdims=True)
val_preds = (val_probabilities[:, 1] > 0.5).astype(int)

# Choose best threshold
best_threshold = None
best_f1 = 0

for threshold in np.linspace(spec["threshold_min"], spec["threshold_max"], spec["threshold_steps"]):
    preds = (val_probabilities[:, 1] > threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission prediction
if WRITE_SUBMISSION:
    if FINAL_SUBMISSION:
        final_train_dataset = DisasterTweetDataset(train_df['text'], train_df['target'])
        trainer.train_dataset = final_train_dataset
        trainer.train()

    test_texts = list(test_df['text'])
    test_dataset = DisasterTweetDataset(test_texts, labels=None)
    test_predictions = trainer.predict(test_dataset).predictions
    test_probabilities = np.exp(test_predictions) / np.sum(np.exp(test_predictions), axis=-1, keepdims=True)
    test_preds = (test_probabilities[:, 1] > best_threshold).astype(int)

    # Write submission
    os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_preds
    })
    submission_df.to_csv(spec["submission_path"], index=False)

# Metrics
acc = accuracy_score(val_labels, val_preds)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')