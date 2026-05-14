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

# Preprocess text
def preprocess_text(text):
    return re.sub(r"http\S+", "HTTPURL", text)
    return re.sub(r"@\S+", "@USER", text)

train_df["text"] = train_df.apply(lambda row: preprocess_text(row["keyword"] + " [SEP] " + row["text"]) if pd.notna(row["keyword"]) else preprocess_text(row["text"]), axis=1)
test_df["text"] = test_df.apply(lambda row: preprocess_text(row["keyword"] + " [SEP] " + row["text"]) if pd.notna(row["keyword"]) else preprocess_text(row["text"]), axis=1)

# Sample train data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None
train_texts, val_texts, y_train, y_val = train_test_split(train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels)

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
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=False
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

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1.5e-05,
    weight_decay=0.01,
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    fp16=False,
    disable_tqdm=True
)

# Create datasets
train_dataset = TweetDataset(train_texts, y_train, tokenizer, max_len=128)
val_dataset = TweetDataset(val_texts, y_val, tokenizer, max_len=128)

# Trainer
trainer = Trainer(
callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train if not DRY_RUN
if not DRY_RUN:
    trainer.train()

# Predict validation logits
val_outputs = trainer.predict(val_dataset)
trainer.evaluate(eval_dataset=val_dataset)
val_outputs = trainer.predict(val_dataset)
val_outputs = trainer.predict(val_dataset)
val_predictions = torch.nn.functional.softmax(torch.tensor(val_outputs.predictions), dim=-1)[:, 1].numpy()

# Choose best threshold
best_threshold = 0.5
best_f1 = 0
for threshold in np.linspace(0.3, 0.7, 41):
    val_preds = (val_predictions[:, 1] > threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission training if FINAL_SUBMISSION is true
if FINAL_SUBMISSION:
    final_train_dataset = TweetDataset(train_df['text'], train_df['target'], tokenizer, max_len=128)
    trainer.train(resume_from_checkpoint=False)

# Predict test logits and write submission if WRITE_SUBMISSION is true
if WRITE_SUBMISSION:
    test_predictions = trainer.predict(test_dataset).predictions
    test_predictions = torch.nn.functional.softmax(torch.tensor(test_predictions), dim=-1)[:, 1].numpy()
    test_preds = (test_predictions[:, 1] > best_threshold).astype(int)
    
    # Create submission directory if it doesn't exist
    
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_preds
    })
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

# Metrics
val_preds = (val_predictions[:, 1] > best_threshold).astype(int)
f1 = f1_score(y_val, val_preds)
acc = accuracy_score(y_val, val_preds)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')