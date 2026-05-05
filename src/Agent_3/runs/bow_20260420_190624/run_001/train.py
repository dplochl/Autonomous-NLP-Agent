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
X_train = train_df['text'].values
y_train = train_df['target'].values

if DRY_RUN:
    stratify_labels = stratify_labels[:100] if stratify_labels is not None else None
    X_train = X_train[:100]
    y_train = y_train[:100]

# Stratify labels if class counts allow

# Train-test split
stratify_labels = y_train if len(np.unique(np.asarray(y_train, dtype=int))) > 1 and np.min(np.bincount(np.asarray(y_train, dtype=int))) >= 2 else None
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizer and model
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1), min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test_df['text'])

model = LogisticRegression(C=1.0, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predict probabilities
y_val_prob = model.predict_proba(X_val_tfidf)[:, 1]
y_test_prob = model.predict_proba(X_test_tfidf)[:, 1]

# Choose best cutoff
best_f1 = 0
best_threshold = 0.5
thresholds = np.linspace(0.4, 0.6, 21)

for threshold in thresholds:
    y_val_pred = (y_val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Predict test set with best cutoff
y_test_pred = (y_test_prob >= best_threshold).astype(int)

# Save submission
submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_test_pred})
submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bow_20260420_190624/run_001/submission.csv"
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission_df.to_csv(submission_path, index=False)

# Calculate final metrics
y_val_pred = (y_val_prob >= best_threshold).astype(int)
f1 = f1_score(y_val, y_val_pred)
acc = accuracy_score(y_val, y_val_pred)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')