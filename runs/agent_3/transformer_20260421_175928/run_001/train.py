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

# Preprocess data
def preprocess_text(row):
    if row['keyword']:
        return f"{row['keyword']} [SEP] {row['text']}"
    else:
        return row['text']

train_df['text'] = train_df.apply(preprocess_text, axis=1)
test_df['text'] = test_df.apply(preprocess_text, axis=1)

# Split data
if DRY_RUN:
    train_df = train_df.sample(n=min(16, len(train_df)), random_state=42)
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None
train_texts, val_texts, y_train, y_val = train_test_split(
    train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels
)

# Tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Dataset class
class DisasterTweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
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
train_dataset = DisasterTweetDataset(train_texts, y_train.tolist(), tokenizer, max_len=128)
val_dataset = DisasterTweetDataset(val_texts, y_val.tolist(), tokenizer, max_len=128)

if WRITE_SUBMISSION:
    test_dataset = DisasterTweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=128)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
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

if not DRY_RUN:
    trainer.train()

# Predict validation logits
val_logits = trainer.predict(val_dataset).predictions

# Choose best threshold
thresholds = np.linspace(0.3, 0.7, 41)
best_f1 = 0
best_threshold = 0.5
for threshold in thresholds:
    val_preds = (val_logits[:, 1] > threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Best F1: {best_f1}")

# Final submission
if FINAL_SUBMISSION and WRITE_SUBMISSION:
    final_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    final_trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    final_trainer.train()

    test_logits = final_trainer.predict(test_dataset).predictions
    test_preds = (test_logits[:, 1] > best_threshold).astype(int)

    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_preds
    })

        submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260421_175928/run_001/submission.csv"
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