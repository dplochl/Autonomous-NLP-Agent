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
train_df["text"] = train_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)
test_df["text"] = test_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)

# Sample train data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
stratify_labels = train_df["target"] if len(train_df["target"].unique()) > 1 else None
train_texts, val_texts, y_train, y_val = train_test_split(
    train_df["text"], train_df["target"], test_size=0.2, random_state=42, stratify=stratify_labels
)

# Load tokenizer and model
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
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
train_dataset = TweetDataset(train_texts, y_train.tolist(), tokenizer, max_len=128)
val_dataset = TweetDataset(val_texts, y_val.tolist(), tokenizer, max_len=128)

if WRITE_SUBMISSION:
    test_dataset = TweetDataset(list(test_df["text"]), labels=None, tokenizer=tokenizer, max_len=128)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-05,
    weight_decay=0.001,
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

# DRY_RUN check
if not DRY_RUN:
    trainer.train()

# Predict validation logits
val_predictions = trainer.predict(val_dataset).predictions
val_probabilities = torch.softmax(torch.tensor(val_predictions), dim=1)[:, 1].numpy()

# Choose best threshold
thresholds = np.linspace(0.3, 0.7, 41)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    val_preds = (val_probabilities > threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission training
if FINAL_SUBMISSION:
    final_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    final_train_dataset = TweetDataset(train_df["text"].tolist(), train_df["target"].tolist(), tokenizer, max_len=128)
    final_trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=final_train_dataset
    )
    final_trainer.train()

# Predict test logits and write submission
if WRITE_SUBMISSION:
    if FINAL_SUBMISSION:
        test_predictions = final_trainer.predict(test_dataset).predictions
    else:
        test_predictions = trainer.predict(test_dataset).predictions

    test_preds = (test_predictions[:, 1] > best_threshold).astype(int)
    submission_df = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    
    # Create submissions directory if it doesn't exist
    os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260421_193538/run_001/submission.csv"), exist_ok=True)
    
    submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260421_193538/run_001/submission.csv", index=False)

# Print metrics
val_preds = (val_predictions[:, 1] > best_threshold).astype(int)
f1 = f1_score(y_val, val_preds)
acc = accuracy_score(y_val, val_preds)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')