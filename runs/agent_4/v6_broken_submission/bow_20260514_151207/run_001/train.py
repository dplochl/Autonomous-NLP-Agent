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
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Create text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Prepare data
X = train_df['text'].astype(str).to_numpy()
y = train_df['target'].values

if DRY_RUN:
    X = X[:100]
    y = y[:100]

elif TRAIN_FRACTION < 1.0:
    train_df_sampled = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)
    X = train_df_sampled['text'].values
    y = train_df_sampled['target'].values

# Stratify labels if class counts allow

# Train-test split
stratify_labels = y if len(np.unique(np.asarray(y, dtype=int))) > 1 and np.min(np.bincount(np.asarray(y, dtype=int))) >= 2 else None
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizer and model
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1), min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

model = LogisticRegression(C=1.0, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predict validation probabilities
y_val_prob = model.predict_proba(X_val_tfidf)[:, 1]

# Choose best cutoff
thresholds = np.linspace(0.4, 0.6, 21)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred = (y_val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final submission prediction
if FINAL_SUBMISSION:
    X_final_tfidf = vectorizer.fit_transform(train_df['text'])
    model.fit(X_final_tfidf, train_df['target'])

X_test_tfidf = vectorizer.transform(test_df['text'])
y_test_prob = model.predict_proba(X_test_tfidf)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)

# Write submission
if WRITE_SUBMISSION:
    submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/bow_20260514_151207/run_001/submission.csv"
    submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_test_pred})
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

# Print metrics
y_val_pred_final = (y_val_prob >= best_threshold).astype(int)
f1 = f1_score(y_val, y_val_pred_final)
acc = accuracy_score(y_val, y_val_pred_final)
print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')