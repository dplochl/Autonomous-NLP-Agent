import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load environment variables
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Fill missing values
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Build text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# DRY_RUN or sample train data
if DRY_RUN:
    train_df = train_df.sample(n=min(16, len(train_df)), random_state=42)
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
stratify_labels = train_df['target']
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels)

# Define Dataset class
class DisasterTweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=104):
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
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}
        else:
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create datasets
train_dataset = DisasterTweetDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_len=104)
val_dataset = DisasterTweetDataset(X_val.tolist(), y_val.tolist(), tokenizer, max_len=104)
test_dataset = DisasterTweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=104)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    fp16=False,
    disable_tqdm=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train if not DRY_RUN
if not DRY_RUN:
    trainer.train()

# Predict validation and test logits
val_logits = trainer.predict(val_dataset).predictions
test_logits = trainer.predict(test_dataset).predictions

# Convert logits to probabilities
val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True)) / np.sum(np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True)), axis=1, keepdims=True)
test_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True)) / np.sum(np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True)), axis=1, keepdims=True)

# Choose best threshold
thresholds = np.linspace(0.3, 0.7, 41)
best_f1 = -1
best_threshold = 0

for threshold in thresholds:
    val_preds = (val_probs[:, 1] > threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Predict test labels with best threshold
test_preds = (test_probs[:, 1] > best_threshold).astype(int)

# Create submission directory and write CSV
submission_dir = os.path.dirname(os.environ.get("DISASTER_AGENT_SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260421_143559/run_003/submission.csv"))
os.makedirs(submission_dir, exist_ok=True)
submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
submission_df.to_csv(os.environ.get("DISASTER_AGENT_SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260421_143559/run_003/submission.csv"), index=False)

# Calculate final metrics
acc = accuracy_score(y_val, (val_probs[:, 1] > best_threshold).astype(int))
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')