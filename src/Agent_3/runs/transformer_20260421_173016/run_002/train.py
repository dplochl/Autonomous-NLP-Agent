import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

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
train_df["text"] = train_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)
test_df["text"] = test_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)

# Sample train data
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
X = train_df["text"].tolist()
y = train_df["target"].tolist()

stratify_labels = np.array(y) if len(set(y)) > 1 and min(np.bincount(y)) >= 2 else None
train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)
train_texts = list(train_texts)
val_texts = list(val_texts)
train_labels = list(train_labels)
val_labels = list(val_labels)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=128)

# Create Dataset class
class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = list(labels) if labels is not None else None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = DisasterDataset(train_encodings, train_labels)
val_dataset = DisasterDataset(val_encodings, val_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=3e-05,
    weight_decay=0.005,
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
    f1 = f1_score(val_labels, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission
if FINAL_SUBMISSION:
    # Retrain on full train data
    full_train_encodings = tokenizer(train_df["text"].tolist(), truncation=True, padding='max_length', max_length=128)
    full_train_dataset = DisasterDataset(full_train_encodings, train_df["target"].tolist())
    
    trainer.train(resume_from_checkpoint=False)

# Predict test logits
if WRITE_SUBMISSION:
    test_encodings = tokenizer(list(test_df["text"]), truncation=True, padding='max_length', max_length=128)
    test_dataset = DisasterDataset(test_encodings, labels=None)
    test_logits = trainer.predict(test_dataset).predictions
    
    # Convert logits to probabilities
    test_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)
    
    # Apply best threshold
    test_preds = (test_probs[:, 1] > best_threshold).astype(int)
    
    # Write submission
    submission_df = pd.DataFrame({"id": test_df["id"], "target": test_preds})
        submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/transformer_20260421_173016/run_002/submission.csv"
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission_df.to_csv(submission_path, index=False)

# Print metrics
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(accuracy_score(val_labels, (val_logits[:, 1] > best_threshold).astype(int)), 4)) + '}')