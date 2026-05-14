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
train_df.fillna("", inplace=True)
test_df.fillna("", inplace=True)

# Build text field
train_df['text'] = train_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)
test_df['text'] = test_df.apply(lambda x: f"{x['keyword']} [SEP] {x['text']}" if pd.notna(x['keyword']) else x['text'], axis=1)

# DRY_RUN or sample train data
if DRY_RUN:
    train_df = train_df.sample(n=min(16, len(train_df)), random_state=42)
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Define stratify labels
stratify_labels = train_df['target'] if train_df['target'].nunique() > 1 and train_df['target'].value_counts().min() >= 2 else None

# Train-test split
train_texts, val_texts, y_train, y_val = train_test_split(
    train_df['text'], train_df['target'], test_size=0.2, random_state=42, stratify=stratify_labels
)

# Load tokenizer and model
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding='max_length', max_length=128)
        self.labels = list(labels) if labels is not None else None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = TextDataset(train_texts.tolist(), y_train.tolist())
val_dataset = TextDataset(val_texts.tolist(), y_val.tolist())
test_dataset = TextDataset(list(test_df['text']))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1.5e-05,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
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

# Train if not DRY_RUN
if not DRY_RUN:
    trainer.train()

# Predict on validation and test datasets
val_preds = trainer.predict(val_dataset).predictions
test_preds = trainer.predict(test_df['text'].tolist()).predictions

# Convert logits to probabilities
val_probs = np.exp(val_preds) / np.sum(np.exp(val_preds), axis=1, keepdims=True)
test_probs = np.exp(test_preds) / np.sum(np.exp(test_preds), axis=1, keepdims=True)

# Choose best cutoff
best_f1 = 0
best_threshold = 0.5
for threshold in np.linspace(0.3, 0.7, 41):
    val_pred_labels = (val_probs[:, 1] > threshold).astype(int)
    f1 = f1_score(y_val, val_pred_labels)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Evaluate on validation set
val_pred_labels = (val_probs[:, 1] > best_threshold).astype(int)
acc = accuracy_score(y_val, val_pred_labels)

# Prepare submission
test_pred_labels = (test_probs[:, 1] > best_threshold).astype(int)
submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_pred_labels})

# Create submissions directory if it doesn't exist
os.makedirs(os.path.dirname("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/roberta_20260421_125459/run_001/submission.csv"), exist_ok=True)

# Save submission
submission_df.to_csv("/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/roberta_20260421_125459/run_001/submission.csv", index=False)

# Print metrics
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')