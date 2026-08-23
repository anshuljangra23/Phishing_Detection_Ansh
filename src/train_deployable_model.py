import os
import joblib
import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from url_features import extract_url_features


# =====================================================
# 1. LOAD DATA
# =====================================================

DATA_FILE = "data/processed/clean_urls.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded")
print("Shape:", df.shape)


# =====================================================
# 2. URLS AND LABELS
# =====================================================

urls = (
    df["domain"]
    .fillna("")
    .astype(str)
    .values
)

y = (
    df["label"]
    .astype(int)
    .values
)


# =====================================================
# 3. EXTRACT 30 NUMERIC FEATURES
# =====================================================

print("\nExtracting 30 URL features...")

feature_list = []

for url in urls:
    feature_list.append(
        extract_url_features(url)[0]
    )

X_numeric = np.array(feature_list)

print("Numeric feature shape:", X_numeric.shape)


# =====================================================
# 4. TRAIN / TEST SPLIT
# =====================================================

indices = np.arange(len(urls))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

urls_train = urls[train_idx]
urls_test = urls[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

X_numeric_train = X_numeric[train_idx]
X_numeric_test = X_numeric[test_idx]

print("\nTraining samples:", len(train_idx))
print("Testing samples:", len(test_idx))


# =====================================================
# 5. CHARACTER TF-IDF
# =====================================================

print("\nCreating character-level TF-IDF...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=10000,
    sublinear_tf=True
)

# IMPORTANT:
# Fit ONLY on training URLs
X_text_train = vectorizer.fit_transform(
    urls_train
)

X_text_test = vectorizer.transform(
    urls_test
)

print(
    "Training TF-IDF shape:",
    X_text_train.shape
)

print(
    "Testing TF-IDF shape:",
    X_text_test.shape
)


# =====================================================
# 6. COMBINE FEATURES
# =====================================================

X_train = hstack([
    X_numeric_train,
    X_text_train
])

X_test = hstack([
    X_numeric_test,
    X_text_test
])

print("\nCombined training shape:", X_train.shape)
print("Combined testing shape:", X_test.shape)


# =====================================================
# 7. RANDOM FOREST
# =====================================================

print("\nTraining final deployable Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# =====================================================
# 8. PREDICTION
# =====================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# =====================================================
# 9. EVALUATION
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# =====================================================
# 10. RESULTS
# =====================================================

print("\n" + "=" * 50)
print("FINAL DEPLOYABLE MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =====================================================
# 11. SAVE MODEL AND VECTORIZER
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/final_url_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/final_url_tfidf_vectorizer.pkl"
)

print("\nModels saved:")

print("models/final_url_model.pkl")
print("models/final_url_tfidf_vectorizer.pkl")