import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
for col in ['keyword', 'location', 'text']:
    train_df[col] = train_df[col].fillna("")
    test_df[col] = test_df[col].fillna("")

# Build text field
train_df['text'] = train_df['keyword'] + " [SEP] " + train_df['text']
test_df['text'] = test_df['keyword'] + " [SEP] " + test_df['text']

# Create X and y
X = train_df['text'].astype(str).to_numpy()
y = train_df['target'].values

# DRY_RUN or sample the data
if DRY_RUN:
    train_df = train_df.head(100)
    X = train_df['text'].astype(str).to_numpy()
    y = train_df['target'].values
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)
    X = train_df['text'].astype(str).to_numpy()
    y = train_df['target'].values

# Stratify labels if possible
stratify_labels = y if len(np.unique(np.asarray(y, dtype=int))) > 1 and np.min(np.bincount(np.asarray(y, dtype=int))) >= 2 else None

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)

# Build vectorizers
word_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), min_df=3)
char_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(4, 6), analyzer='char', min_df=3)

# Fit vectorizers
X_train_word = word_vectorizer.fit_transform(X_train)
X_val_word = word_vectorizer.transform(X_val)
X_train_char = char_vectorizer.fit_transform(X_train)
X_val_char = char_vectorizer.transform(X_val)

# Merge features
X_train_combined = hstack([X_train_word, X_train_char])
X_val_combined = hstack([X_val_word, X_val_char])

# Fit logistic regression model
logreg = LogisticRegression(C=5.0, random_state=42)
logreg.fit(X_train_combined, y_train)

# Predict validation probabilities
val_probs = logreg.predict_proba(X_val_combined)[:, 1]

# Tune decision threshold
best_threshold = None
best_f1 = 0

for threshold in np.linspace(0.4, 0.6, 21):
    val_preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Calculate accuracy at best threshold
acc = accuracy_score(y_val, (val_probs >= best_threshold).astype(int))

# Final submission logic
if FINAL_SUBMISSION:
    # Refit vectorizers and classifier on full train_df
    X_full_word = word_vectorizer.fit_transform(train_df['text'])
    X_full_char = char_vectorizer.fit_transform(train_df['text'])
    X_full_combined = hstack([X_full_word, X_full_char])
    logreg.fit(X_full_combined, y_train)

    # Predict test probabilities
    X_test_word = word_vectorizer.transform(test_df['text'])
    X_test_char = char_vectorizer.transform(test_df['text'])
    X_test_combined = hstack([X_test_word, X_test_char])
    test_probs = logreg.predict_proba(X_test_combined)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    # Write submission
    if WRITE_SUBMISSION:
        submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_preds})
        submission_df.to_csv(spec["submission_path"], index=False)

# Print metrics
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')