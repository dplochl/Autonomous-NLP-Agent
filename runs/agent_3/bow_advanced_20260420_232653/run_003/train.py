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

# Load data
train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Fill missing values
train_df[['keyword', 'location', 'text']] = train_df[['keyword', 'location', 'text']].fillna('')
test_df[['keyword', 'location', 'text']] = test_df[['keyword', 'location', 'text']].fillna('')

# Build text field
train_df['text'] = train_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)
test_df['text'] = test_df.apply(lambda row: f"{row['keyword']} [SEP] {row['text']}" if pd.notna(row['keyword']) else row['text'], axis=1)

# Create X and y
X_train = train_df['text'].values
y_train = train_df['target'].values

if DRY_RUN:
    X_train = X_train[:200]
    y_train = y_train[:200]

# Stratify labels if class counts allow
stratify_labels = y_train if len(set(y_train)) > 1 else None

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_labels)


# Build TF-IDF vectorizers
word_vectorizer = TfidfVectorizer(max_features=18000, ngram_range=(1, 3), min_df=2)
char_vectorizer = TfidfVectorizer(max_features=13000, ngram_range=(2, 4), analyzer='char')

# Fit and transform
X_train_word = word_vectorizer.fit_transform(X_train)
X_val_word = word_vectorizer.transform(X_val)

X_train_char = char_vectorizer.fit_transform(X_train)
X_val_char = char_vectorizer.transform(X_val)

# Merge features
X_train_combined = hstack([X_train_word, X_train_char])
X_val_combined = hstack([X_val_word, X_val_char])

# Fit Logistic Regression model
logreg = LogisticRegression(C=1.5, random_state=42)
logreg.fit(X_train_combined, y_train)

# Predict probabilities
y_val_prob = logreg.predict_proba(X_val_combined)[:, 1]

# Find best threshold
best_f1 = 0
best_threshold = 0.35
thresholds = np.linspace(0.35, 0.65, 31)

for threshold in thresholds:
    y_pred = (y_val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Predict test probabilities
X_test_word = word_vectorizer.transform(test_df['text'])
X_test_char = char_vectorizer.transform(test_df['text'])
X_test_combined = hstack([X_test_word, X_test_char])
y_test_prob = logreg.predict_proba(X_test_combined)[:, 1]

# Apply best threshold
test_predictions = (y_test_prob >= best_threshold).astype(int)

# Save submission
submission_path = "/Users/niccogermani/Desktop/gitrepo/apa-disaster-tweets-agent/src/Agent_3/runs/bow_advanced_20260420_232653/run_003/submission.csv"
submission_df = pd.DataFrame({'id': test_df['id'], 'target': test_predictions})
submission_df.to_csv(submission_path, index=False)

# Calculate metrics on validation set
y_val_pred = (y_val_prob >= best_threshold).astype(int)
f1 = f1_score(y_val, y_val_pred)
acc = accuracy_score(y_val, y_val_pred)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')