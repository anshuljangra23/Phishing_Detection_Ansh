import joblib
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# =====================================================
# 1. LOAD DATASETS
# =====================================================

original = pd.read_csv(
    "data/processed/clean_urls.csv"
)

engineered = pd.read_csv(
    "data/processed/url_features.csv"
)

print("Original shape:", original.shape)
print("Engineered shape:", engineered.shape)


# =====================================================
# 2. ORIGINAL FEATURES
# =====================================================

original_features = [
    column
    for column in original.columns
    if column not in ["domain", "label"]
]

X_original = original[original_features]


# =====================================================
# 3. ENGINEERED FEATURES
# =====================================================

engineered_features = [
    column
    for column in engineered.columns
    if column != "label"
]

X_engineered = engineered[engineered_features]


# =====================================================
# 4. COMBINE FEATURES
# =====================================================

X = pd.concat(
    [
        X_original.reset_index(drop=True),
        X_engineered.reset_index(drop=True)
    ],
    axis=1
)

y = original["label"].astype(int)


print("\nCombined feature shape:", X.shape)
print("Number of features:", X.shape[1])


# =====================================================
# 5. TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================================================
# 6. RANDOM FOREST
# =====================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# =====================================================
# 7. TRAIN
# =====================================================

print("\nTraining combined-feature Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")

# =====================================================
# SAVE TRAINED MODEL
# =====================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/phishing_combined_model.pkl"
)

print("\nModel saved successfully:")
print("models/phishing_combined_model.pkl")

# =====================================================
# 8. PREDICTION
# =====================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# =====================================================
# 9. METRICS
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

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
print("COMBINED MODEL PERFORMANCE")
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
