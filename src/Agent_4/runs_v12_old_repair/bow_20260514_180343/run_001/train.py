import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

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
for col in ["keyword", "location", "text"]:
    train_df[col] = train_df[col].fillna("")
    test_df[col] = test_df[col].fillna("")

# Build text field
train_df["text"] = train_df["keyword"] + " [SEP] " + train_df["text"]
test_df["text"] = test_df["keyword"] + " [SEP] " + test_df["text"]

# Create X and y
X_train = train_df["text"].astype(str).to_numpy()
y_train = train_df["target"].values

if DRY_RUN:
    X_train = X_train[:200]
    y_train = y_train[:200]

elif TRAIN_FRACTION < 1.0:
    train_df_sampled = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)
    X_train = train_df_sampled["text"].values
    y_train = train_df_sampled["target"].values

# Stratify labels if class counts allow
stratify_labels = y_train if len(np.unique(y_train)) > 1 else None

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizer and classifier
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

classifier = LogisticRegression(C=1.0, random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Predict validation probabilities
val_probs = classifier.predict_proba(X_val_tfidf)[:, 1]

# Tune decision threshold
threshold_min = 0.3
threshold_max = 0.7
threshold_steps = 41
best_threshold = None
best_f1 = 0

for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
    y_pred = (val_probs >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

acc = accuracy_score(y_val, (val_probs >= best_threshold).astype(int))

# Refit on full train data for final submission
if FINAL_SUBMISSION:
    X_full_tfidf = vectorizer.fit_transform(train_df["text"].values)
    classifier.fit(X_full_tfidf, train_df["target"].values)

    # Predict test probabilities
    X_test_tfidf = vectorizer.transform(test_df["text"].values)
    test_probs = classifier.predict_proba(X_test_tfidf)[:, 1]
    y_pred_final = (test_probs >= best_threshold).astype(int)

    if WRITE_SUBMISSION:
        submission_df = pd.DataFrame({"id": test_df["id"], "target": y_pred_final})
        submission_df.to_csv(os.environ.get("AGENT_SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/bow_20260514_180343/run_001/submission.csv"), index=False)

print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')