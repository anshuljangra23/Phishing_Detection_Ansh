import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# =====================================================
# 1. LOAD DATASET
# =====================================================

DATA_FILE = "data/processed/clean_urls.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# =====================================================
# 2. SEPARATE FEATURES AND LABEL
# =====================================================

X = df.drop(columns=["domain", "label"])
y = df["label"].astype(int)

print("\nFeature shape:", X.shape)
print("Label shape:", y.shape)

print("\nClass distribution:")
print(y.value_counts())


# =====================================================
# 3. TRAIN / TEST SPLIT
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
# 4. RANDOM FOREST MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# =====================================================
# 5. TRAIN
# =====================================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# =====================================================
# 6. PREDICTION
# =====================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# =====================================================
# 7. EVALUATION
# =====================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))