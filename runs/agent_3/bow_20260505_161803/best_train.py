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
# Create text field
train_df['text'] = train_df['keyword'] + " [SEP] " + train_df['text']
test_df['text'] = test_df['keyword'] + " [SEP] " + test_df['text']

# DRY_RUN handling
if DRY_RUN:
    train_df = train_df.head(500)

# Sample if needed
if TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Prepare data
X = train_df['text'].astype(str).to_numpy()
y = train_df['target'].values

# Stratify labels if class counts allow

# Train-test split
stratify_labels = y if len(np.unique(np.asarray(y, dtype=int))) > 1 and np.min(np.bincount(np.asarray(y, dtype=int))) >= 2 else None
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizer and model
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

model = LogisticRegression(C=1.0, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predict validation probabilities
y_val_prob = model.predict_proba(X_val_tfidf)[:, 1]

# Choose best threshold
thresholds = np.linspace(0.4, 0.6, 21)
best_threshold = None
best_f1 = 0

for threshold in thresholds:
    y_pred = (y_val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission handling
if FINAL_SUBMISSION:
    # Refit on all training data
    X_all_tfidf = vectorizer.fit_transform(train_df['text'])
    model.fit(X_all_tfidf, train_df['target'])

    # Predict test probabilities
    X_test_tfidf = vectorizer.transform(test_df['text'])
    y_test_prob = model.predict_proba(X_test_tfidf)[:, 1]
    y_pred_final = (y_test_prob >= best_threshold).astype(int)

    if WRITE_SUBMISSION:
        submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_pred_final})
        submission_df.to_csv(os.environ.get("AGENT_SUBMISSION_PATH", "submission.csv"), index=False)

# Calculate final metrics
y_val_pred = (y_val_prob >= best_threshold).astype(int)
f1 = f1_score(y_val, y_val_pred)
acc = accuracy_score(y_val, y_val_pred)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')