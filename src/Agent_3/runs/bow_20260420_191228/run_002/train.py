import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# Load environment variables
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Fill missing values
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Build text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Create X and y
X_train = train_df['text'].astype(str).to_numpy()
y_train = train_df['target'].values

if DRY_RUN:
    X_train = X_train[:int(os.environ.get("DRY_RUN_HEAD", 100))]
    y_train = y_train[:int(os.environ.get("DRY_RUN_HEAD", 100))]

# Stratify labels

# Train-test split
stratify_labels = y_train if len(np.unique(np.asarray(y_train, dtype=int))) > 1 and np.min(np.bincount(np.asarray(y_train, dtype=int))) >= 2 else None
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizer and model
vectorizer = TfidfVectorizer(max_features=5625, ngram_range=(1, 1), min_df=3)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

model = LogisticRegression(C=2.0, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predict probabilities
val_probs = model.predict_proba(X_val_tfidf)[:, 1]
test_probs = model.predict_proba(vectorizer.transform(test_df['text']))[:, 1]

# Find best threshold
thresholds = np.linspace(0.4, 0.6, 21)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred_val = (val_probs > threshold).astype(int)
    f1 = f1_score(y_val, y_pred_val)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Predict test set with best threshold
y_pred_test = (test_probs > best_threshold).astype(int)

# Save submission
submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_pred_test})
submission_path = os.environ.get("SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bow_20260420_191228/run_001/submission.csv")
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission_df.to_csv(submission_path, index=False)

# Metrics
y_pred_val = (val_probs > best_threshold).astype(int)
f1 = f1_score(y_val, y_pred_val)
acc = accuracy_score(y_val, y_pred_val)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')