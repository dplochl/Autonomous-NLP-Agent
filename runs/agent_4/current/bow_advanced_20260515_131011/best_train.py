import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
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
train_df["text_combined"] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df["text_combined"] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Create X and y
X_train = train_df["text_combined"].astype(str).to_numpy()
y_train = train_df["target"].values

if DRY_RUN:
    X_train = X_train[:200]
    y_train = y_train[:200]

elif TRAIN_FRACTION < 1.0:
    train_df_sampled = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)
    X_train = train_df_sampled["text_combined"].values
    y_train = train_df_sampled["target"].values

# Stratify labels if class counts allow
stratify_labels = y_train if len(np.unique(y_train)) > 1 else None

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_labels)

# Define vectorizers
word_vectorizer = TfidfVectorizer(max_features=18750, ngram_range=(1, 1), min_df=2)
char_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(3, 5))

# Fit and transform train data
X_train_word = word_vectorizer.fit_transform(X_train)
X_train_char = char_vectorizer.fit_transform(X_train)

# Merge features
X_train_combined = hstack([X_train_word, X_train_char])

# Fit logistic regression model
logreg = LogisticRegression(C=3.0, random_state=42)
logreg.fit(X_train_combined, y_train)

# Transform validation data
X_val_word = word_vectorizer.transform(X_val)
X_val_char = char_vectorizer.transform(X_val)
X_val_combined = hstack([X_val_word, X_val_char])

# Predict validation probabilities
val_probs = logreg.predict_proba(X_val_combined)[:, 1]

# Tune decision threshold
thresholds = np.linspace(0.3, 0.7, 41)
best_threshold = None
best_f1 = 0

for threshold in thresholds:
    y_pred = (val_probs >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Calculate accuracy at best threshold
acc = accuracy_score(y_val, (val_probs >= best_threshold).astype(int))

print('METRICS: {"f1": ' + str(round(best_f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + ', "best_threshold": ' + str(round(best_threshold, 4)) + '}')

# Final submission
if FINAL_SUBMISSION:
    # Refit vectorizers and classifier on full train data
    X_train_full_word = word_vectorizer.fit_transform(train_df["text_combined"].values)
    X_train_full_char = char_vectorizer.fit_transform(train_df["text_combined"].values)
    X_train_full_combined = hstack([X_train_full_word, X_train_full_char])
    
    logreg.fit(X_train_full_combined, train_df["target"].values)

# Transform test data
X_test_word = word_vectorizer.transform(test_df["text_combined"].values)
X_test_char = char_vectorizer.transform(test_df["text_combined"].values)
X_test_combined = hstack([X_test_word, X_test_char])

# Predict test probabilities and apply best threshold
test_probs = logreg.predict_proba(X_test_combined)[:, 1]
y_pred_test = (test_probs >= best_threshold).astype(int)

if WRITE_SUBMISSION:
    submission_df = pd.DataFrame({"id": test_df["id"], "target": y_pred_test})
    submission_df.to_csv(os.environ.get("AGENT_SUBMISSION_PATH", "/Users/niccogermani/Library/Containers/com.apple.iMovieApp/Data/Documents/Catolica/apa-disaster-tweets-clean/src/Agent_4/runs/bow_advanced_20260515_131011/run_001/submission.csv"), index=False)