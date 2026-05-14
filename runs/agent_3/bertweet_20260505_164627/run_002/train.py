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
submission_path = os.environ.get('DISASTER_AGENT_SUBMISSION_PATH', '/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bertweet_20260505_164627/run_002/submission.csv')

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
    text = re.sub(r"http\S+", "HTTPURL", text)
    text = re.sub(r"@\S+", "@USER", text)
    return text

train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Split data
if DRY_RUN:
    train_df = train_df.sample(n=min(16, len(train_df)), random_state=42)
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

y = train_df['target']
stratify_labels = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
train_texts, val_texts, train_labels, val_labels = train_test_split(train_df['text'].tolist(), y.tolist(), test_size=0.2, random_state=42, stratify=stratify_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)

train_texts = [preprocess_text(str(text)) for text in train_texts]
val_texts = [preprocess_text(str(text)) for text in val_texts]
train_labels = list(train_labels)
val_labels = list(val_labels)

# Tokenizer and model
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
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
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
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

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-05,
    weight_decay=0.001,
    logging_dir='./logs',
    use_cpu=True,
    dataloader_pin_memory=False,
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    fp16=False,
    bf16=False,
    disable_tqdm=True
)

# Create datasets
train_dataset = TweetDataset(train_texts, train_labels, tokenizer, max_len=128)
val_dataset = TweetDataset(val_texts, val_labels, tokenizer, max_len=128)
test_dataset = TweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=128)
test_dataset = TweetDataset(list(test_df['text']), labels=None, tokenizer=tokenizer, max_len=128)

if WRITE_SUBMISSION:
    test_dataset = TweetDataset(test_df['text'], labels=None, tokenizer=tokenizer, max_len=128)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train if not DRY_RUN
if not DRY_RUN:
    trainer.train()

# Predict validation logits
val_logits = trainer.predict(val_dataset).predictions
val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

# Choose best threshold
thresholds = np.linspace(0.3, 0.7, 41)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    val_preds = (val_probs[:, 1] > threshold).astype(int)
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission training if FINAL_SUBMISSION is true
if FINAL_SUBMISSION:
    final_train_dataset = TweetDataset(train_df['text'], train_df['target'], tokenizer, max_len=128)
    trainer.train(resume_from_checkpoint=False)

# Predict test logits and write submission if WRITE_SUBMISSION is true
if WRITE_SUBMISSION:
    test_predictor = final_trainer if FINAL_SUBMISSION else trainer
    test_logits = test_predictor.predict(test_dataset).predictions
    test_preds = (test_logits[:, 1] > best_threshold).astype(int)
    
    # Create submission directory if it doesn't exist
    os.makedirs(os.path.dirname(spec["submission_path"]), exist_ok=True)
    
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_preds
    })
    submission_df.to_csv(spec["submission_path"], index=False)

# Calculate final metrics on validation set
val_preds = (val_probs[:, 1] > best_threshold).astype(int)
f1 = f1_score(val_labels, val_preds)
acc = accuracy_score(val_labels, val_preds)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')