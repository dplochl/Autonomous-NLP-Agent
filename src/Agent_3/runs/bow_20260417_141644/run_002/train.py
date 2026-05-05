import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

# Create text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Prepare data
X_train = train_df['text'].tolist()
y_train = train_df['target'].tolist()

if DRY_RUN:
    X_train = X_train[:100]
    y_train = y_train[:100]

import numpy as np
stratify_labels = np.array(y_train, dtype=int) if len(set(y_train)) > 1 else None

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizer
vectorizer = TfidfVectorizer(max_features=14000, ngram_range=(1, 2), min_df=3)
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
    y_val_pred = (y_val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Predict test set
y_test_pred = (y_test_prob >= best_threshold).astype(int)

# Save submission
submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_test_pred})
submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bow_20260417_141644/run_001/submission.csv"
submission_df.to_csv(submission_path, index=False)

# Calculate final metrics
y_val_pred = (y_val_prob >= best_threshold).astype(int)
f1 = f1_score(y_val, y_val_pred)
acc = accuracy_score(y_val, y_val_pred)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')