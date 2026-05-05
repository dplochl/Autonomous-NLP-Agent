import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import hstack

# Load environment variables
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")
DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
TRAIN_FRACTION = float(os.environ.get("AGENT_TRAIN_FRACTION", "1.0"))
SAMPLE_SEED = int(os.environ.get("AGENT_SAMPLE_SEED", "42"))

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Fill missing values
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Build text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Create X and y
X = train_df['text'].values
y = train_df['target'].values

# DRY_RUN or SAMPLE_FRACTION
if DRY_RUN:
    train_df = train_df.head(150)
elif TRAIN_FRACTION < 1.0:
    train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=SAMPLE_SEED).reset_index(drop=True)

# Stratify labels
stratify_labels = y if len(set(y)) > 1 else None

# Train-test split
import numpy as np
X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y), test_size=0.25, random_state=42, stratify=np.array(stratify_labels))

# Vectorizers
word_vectorizer = TfidfVectorizer(max_features=28000, ngram_range=(1, 3), min_df=2)
char_vectorizer = TfidfVectorizer(max_features=19000, ngram_range=(4, 5), analyzer='char')

# Fit and transform
X_train_word = word_vectorizer.fit_transform(X_train)
X_val_word = word_vectorizer.transform(X_val)
X_test_word = word_vectorizer.transform(test_df['text'])

X_train_char = char_vectorizer.fit_transform(X_train)
X_val_char = char_vectorizer.transform(X_val)
X_test_char = char_vectorizer.transform(test_df['text'])

# Merge features
X_train_combined = hstack([X_train_word, X_train_char])
X_val_combined = hstack([X_val_word, X_val_char])
X_test_combined = hstack([X_test_word, X_test_char])

# Logistic Regression model
logreg = LogisticRegression(C=3.5, random_state=42)
logreg.fit(X_train_combined, y_train)

# Predict probabilities
y_val_prob = logreg.predict_proba(X_val_combined)[:, 1]
y_test_prob = logreg.predict_proba(X_test_combined)[:, 1]

# Choose best cutoff
best_f1 = 0
best_threshold = 0.5
for threshold in np.linspace(0.4, 0.6, 21):
    y_val_pred = (y_val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Predict test set using best threshold
y_test_pred = (y_test_prob >= best_threshold).astype(int)

# Save submission
submission_df = pd.DataFrame({'id': test_df['id'], 'target': y_test_pred})
submission_df.to_csv('/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bow_advanced_20260421_145934/run_002/submission.csv', index=False)

# Calculate metrics on validation set
y_val_pred = (y_val_prob >= best_threshold).astype(int)
f1 = f1_score(y_val, y_val_pred)
acc = accuracy_score(y_val, y_val_pred)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')