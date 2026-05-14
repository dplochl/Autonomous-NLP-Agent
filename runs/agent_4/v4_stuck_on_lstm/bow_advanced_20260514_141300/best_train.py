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

# Create text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Prepare X and y
X = train_df['text'].astype(str).to_numpy()
y = train_df['target'].values

# DRY_RUN or sample data
if DRY_RUN:
    train_df = train_df.head(100)
    X = train_df['text'].astype(str).to_numpy()
    y = train_df['target'].values
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)
    X = train_df['text'].astype(str).to_numpy()
    y = train_df['target'].values

# Stratify labels
import numpy as np
stratify_labels = y if len(np.unique(np.asarray(y, dtype=int))) > 1 and np.min(np.bincount(np.asarray(y, dtype=int))) >= 2 else None

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizers
word_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), min_df=3)
char_vectorizer = TfidfVectorizer(analyzer='char', max_features=15000, ngram_range=(4, 6))

# Fit vectorizers
X_train_word = word_vectorizer.fit_transform(X_train)
X_val_word = word_vectorizer.transform(X_val)
X_train_char = char_vectorizer.fit_transform(X_train)
X_val_char = char_vectorizer.transform(X_val)

# Merge features
X_train_combined = hstack([X_train_word, X_train_char])
X_val_combined = hstack([X_val_word, X_val_char])

# Fit model
model = LogisticRegression(C=5.0, max_iter=1000)
model.fit(X_train_combined, y_train)

# Predict validation probabilities
y_val_prob = model.predict_proba(X_val_combined)[:, 1]

# Choose best threshold
threshold_min = 0.4
threshold_max = 0.6
threshold_steps = 21
best_threshold = None
best_f1 = 0

for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
    y_pred = (y_val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission prediction
if FINAL_SUBMISSION:
    # Refit on full train data
    X_train_full_word = word_vectorizer.fit_transform(train_df['text'])
    X_train_full_char = char_vectorizer.fit_transform(train_df['text'])
    X_train_full_combined = hstack([X_train_full_word, X_train_full_char])
    model.fit(X_train_full_combined, train_df['target'])

    # Predict test probabilities
    X_test_word = word_vectorizer.transform(test_df['text'])
    X_test_char = char_vectorizer.transform(test_df['text'])
    X_test_combined = hstack([X_test_word, X_test_char])
    y_test_prob = model.predict_proba(X_test_combined)[:, 1]
    y_pred_final = (y_test_prob >= best_threshold).astype(int)

    # Write submission
    if WRITE_SUBMISSION:
        submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_pred_final})
        submission_df.to_csv(os.environ.get("AGENT_SUBMISSION_PATH", "submission.csv"), index=False)

# Metrics
y_val_pred = (y_val_prob >= best_threshold).astype(int)
f1 = f1_score(y_val, y_val_pred)
acc = accuracy_score(y_val, y_val_pred)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')