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
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for _df in (train_df, test_df):
    for _col in ('keyword', 'location', 'text'):
        if _col in _df.columns:
            _df[_col] = _df[_col].fillna('').astype(str)

# Fill missing values
train_df["keyword"] = train_df["keyword"].fillna("")
train_df["location"] = train_df["location"].fillna("")
train_df["text"] = train_df["text"].fillna("")

test_df["keyword"] = test_df["keyword"].fillna("")
test_df["location"] = test_df["location"].fillna("")
test_df["text"] = test_df["text"].fillna("")

# Build text field
train_df["text_combined"] = train_df["keyword"] + " [SEP] " + train_df["text"]
test_df["text_combined"] = test_df["keyword"] + " [SEP] " + test_df["text"]

# DRY_RUN or sample data
if DRY_RUN:
    train_df = train_df.head(100)
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Split data
X = train_df["text_combined"].astype(str).to_numpy()
y = train_df["target"].values

val_size = 0.2
stratify_labels = y if len(np.unique(y)) > 1 else None
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42, stratify=stratify_labels)

# Vectorizer and model
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=3)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

model = LogisticRegression(C=0.9025, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predict validation probabilities
val_probs = model.predict_proba(X_val_tfidf)[:, 1]

# Tune threshold
threshold_min = 0.4
threshold_max = 0.6
threshold_steps = 21
best_threshold = None
best_f1 = 0

for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
    y_pred = (val_probs >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Calculate accuracy with best threshold
acc = accuracy_score(y_val, (val_probs >= best_threshold).astype(int))

# Final submission
if FINAL_SUBMISSION:
    X_full_tfidf = vectorizer.fit_transform(train_df["text_combined"].values)
    model.fit(X_full_tfidf, train_df["target"].values)
    test_probs = model.predict_proba(vectorizer.transform(test_df["text_combined"].values))[:, 1]
else:
    test_probs = model.predict_proba(vectorizer.transform(test_df["text_combined"].values))[:, 1]

# Write submission
if WRITE_SUBMISSION:
    submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/bow_20260514_173301/run_001/submission.csv"
    submission_df = pd.DataFrame({"id": test_df["id"], "target": (test_probs >= best_threshold).astype(int)})
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

# Print metrics
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')