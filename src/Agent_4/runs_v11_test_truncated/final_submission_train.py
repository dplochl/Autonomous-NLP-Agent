import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
for col in ["keyword", "location", "text"]:
    train_df[col] = train_df[col].fillna("")
    test_df[col] = test_df[col].fillna("")

# Build text field
train_df["text"] = train_df["keyword"] + " [SEP] " + train_df["text"]
test_df["text"] = test_df["keyword"] + " [SEP] " + test_df["text"]

# Prepare data
X_train = train_df["text"].astype(str).to_numpy()
y_train = train_df["target"].values

if DRY_RUN:
    X_train = X_train[:100]
    y_train = y_train[:100]

elif TRAIN_FRACTION < 1.0:
    train_df_sampled = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)
    X_train = train_df_sampled["text"].values
    y_train = train_df_sampled["target"].values

# Stratify labels if class counts allow
stratify_labels = y_train if len(np.unique(y_train)) > 1 else None

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_labels)

# Vectorizer and classifier
vectorizer = TfidfVectorizer(max_features=9687, ngram_range=(1, 2), min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

classifier = LogisticRegression(C=1.0, random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Predict validation probabilities
val_probs = classifier.predict_proba(X_val_tfidf)[:, 1]

# Tune decision threshold
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

# Calculate accuracy at best threshold
acc = accuracy_score(y_val, (val_probs >= best_threshold).astype(int))

# Final submission
if FINAL_SUBMISSION:
    # Refit on full train data
    X_full_tfidf = vectorizer.fit_transform(train_df["text"].values)
    classifier.fit(X_full_tfidf, train_df["target"].values)

    # Predict test probabilities
    X_test_tfidf = vectorizer.transform(test_df["text"].values)
    test_probs = classifier.predict_proba(X_test_tfidf)[:, 1]
    y_pred_final = (test_probs >= best_threshold).astype(int)

    if WRITE_SUBMISSION:
        submission_df = pd.DataFrame({"id": test_df["id"], "target": y_pred_final})
        submission_df.to_csv(os.environ.get("AGENT_SUBMISSION_PATH", "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/apa-disaster-tweets-agent-Nicc-copy/src/Agent_4/runs/bow_20260514_175058/run_001/submission.csv"), index=False)

# Print metrics
print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')

# === AGENT_4 HARDCODED FINAL SUBMISSION TAIL (sparse) ===
import os as _os_sub
import pandas as _pd_sub

# Build the same text field the training code used (text or keyword + [SEP] + text).
_text_col = None
for _candidate in ("text_for_model", "x", "text_input"):
    if _candidate in test_df.columns:
        _text_col = _candidate
        break
if _text_col is None:
    # Fall back: rebuild it from raw test_df columns.
    if "keyword" in test_df.columns:
        test_df["_agent_submit_text"] = test_df.apply(
            lambda r: f"{r['keyword']} [SEP] {r['text']}" if str(r.get('keyword', '')) else str(r.get('text', '')),
            axis=1,
        )
        _text_col = "_agent_submit_text"
    else:
        _text_col = "text"

_test_X = vectorizer.transform(test_df['text'].astype(str))
_test_probs = classifier.predict_proba(_test_X)[:, 1]
_test_preds = (_test_probs >= float(best_threshold)).astype(int)
_sub_df = _pd_sub.DataFrame({"id": test_df["id"].astype(int), "target": _test_preds})

_sub_path = _os_sub.environ.get(
    "DISASTER_AGENT_SUBMISSION_PATH",
    _os_sub.environ.get("AGENT_SUBMISSION_PATH", "submission.csv"),
)
_os_sub.makedirs(_os_sub.path.dirname(_sub_path) or ".", exist_ok=True)
_sub_df.to_csv(_sub_path, index=False)
_pos = int((_sub_df["target"] == 1).sum())
_neg = int((_sub_df["target"] == 0).sum())
print(f"[AGENT_SUBMIT] wrote {len(_sub_df)} rows to {_sub_path} (threshold={float(best_threshold):.4f}, pos={_pos}, neg={_neg})")
