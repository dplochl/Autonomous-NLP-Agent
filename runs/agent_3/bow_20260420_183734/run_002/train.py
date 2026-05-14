import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Load environment variables
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"

# Load data
train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Fill missing values
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Create text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Prepare data
X_train = train_df['text'].to_numpy()
y_train = train_df['target'].to_numpy()

if DRY_RUN:
    X_train = X_train[:100]
    y_train = y_train[:100]

# Stratify labels if class counts allow
stratify_labels = y_train if len(np.unique(y_train)) > 1 else None

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizer and model
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 1), min_df=3)
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
    y_pred_val = (val_probs >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred_val)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Evaluate on validation set
y_pred_val_best = (val_probs >= best_threshold).astype(int)
f1 = f1_score(y_val, y_pred_val_best)
acc = accuracy_score(y_val, y_pred_val_best)

# Prepare submission
test_df['target'] = (test_probs >= best_threshold).astype(int)
submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bow_20260420_183734/run_002/submission.csv"
test_df[['id', 'target']].to_csv(submission_path, index=False)

# Print metrics
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')