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
X = train_df['text'].astype(str).to_numpy()
y = train_df['target'].values

# DRY_RUN
if DRY_RUN:
    X = X[:int(os.environ.get("DRY_RUN_HEAD", 100))]
    y = y[:int(os.environ.get("DRY_RUN_HEAD", 100))]

# Stratify labels

# Train-test split
stratify_labels = y if len(np.unique(np.asarray(y, dtype=int))) > 1 and np.min(np.bincount(np.asarray(y, dtype=int))) >= 2 else None
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizer
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Logistic Regression
logreg = LogisticRegression(C=1.0, random_state=42)
logreg.fit(X_train_tfidf, y_train)

# Predict probabilities
y_val_prob = logreg.predict_proba(X_val_tfidf)[:, 1]
y_test_prob = logreg.predict_proba(vectorizer.transform(test_df['text']))[:, 1]

# Find best threshold
thresholds = np.linspace(0.4, 0.6, 21)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_val_pred = (y_val_prob > threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Apply best threshold to test predictions
y_test_pred = (y_test_prob > best_threshold).astype(int)

# Save submission
submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_test_pred})
submission_df.to_csv(os.environ.get("SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bow_20260420_205626/run_001/submission.csv"), index=False)

# Metrics
y_val_pred = (y_val_prob > best_threshold).astype(int)
f1 = f1_score(y_val, y_val_pred)
acc = accuracy_score(y_val, y_val_pred)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')