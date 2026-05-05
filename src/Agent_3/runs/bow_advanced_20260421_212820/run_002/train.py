import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import hstack

# Load environment variables
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
WRITE_SUBMISSION = os.environ.get("AGENT_WRITE_SUBMISSION") == "1"
FINAL_SUBMISSION = os.environ.get("AGENT_FINAL_SUBMISSION") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Fill missing values
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Build text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Prepare data
X = train_df['text'].to_numpy()
y = train_df['target'].to_numpy()

if DRY_RUN:
    X = X[:100]
    y = y[:100]

elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)
    X = train_df['text'].values
    y = train_df['target'].values

# Stratify labels if class counts allow
stratify_labels = y if len(set(y)) > 1 else None

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizers
word_vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 3), min_df=2)
char_vectorizer = TfidfVectorizer(analyzer='char', max_features=15000, ngram_range=(3, 5))

X_train_word = word_vectorizer.fit_transform(X_train)
X_val_word = word_vectorizer.transform(X_val)

X_train_char = char_vectorizer.fit_transform(X_train)
X_val_char = char_vectorizer.transform(X_val)

# Merge features
X_train_combined = hstack([X_train_word, X_train_char])
X_val_combined = hstack([X_val_word, X_val_char])

# Fit model
model = LogisticRegression(C=2.0, random_state=42)
model.fit(X_train_combined, y_train)

# Predict validation probabilities
y_val_pred_proba = model.predict_proba(X_val_combined)[:, 1]

# Choose best cutoff
threshold_min = 0.4
threshold_max = 0.6
threshold_steps = 21
best_threshold = None
best_f1 = 0

import numpy as np
for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
    y_val_pred = (y_val_pred_proba > threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission prediction
if FINAL_SUBMISSION:
    X_combined = hstack([word_vectorizer.transform(train_df['text']), char_vectorizer.transform(train_df['text'])])
    model.fit(X_combined, train_df['target'])
    X_test_word = word_vectorizer.transform(test_df['text'])
    X_test_char = char_vectorizer.transform(test_df['text'])
    X_test_combined = hstack([X_test_word, X_test_char])
    y_test_pred_proba = model.predict_proba(X_test_combined)[:, 1]
    y_test_pred = (y_test_pred_proba > best_threshold).astype(int)
else:
    y_test_pred_proba = model.predict_proba(X_val_combined)[:, 1]
    y_test_pred = (y_test_pred_proba > best_threshold).astype(int)

# Metrics
acc = accuracy_score(y_val, y_test_pred)
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')

# Write submission
if WRITE_SUBMISSION:
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_test_pred})
    submission_df.to_csv(os.environ.get("AGENT_SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bow_advanced_20260421_212820/run_001/submission.csv"), index=False)