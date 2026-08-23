import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/processed/clean_urls.csv")

X = df.drop(columns=["domain", "label"])
y = df["label"].astype(int)


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================================
# FEATURE GROUPS
# =====================================================

top_features = [
    "card_rem",
    "ranking",
    "jaccard_ARrem",
    "ratio_Rrem",
    "ratio_Arem",
    "jaccard_ARrd",
    "mld_res"
]


# =====================================================
# MODEL EVALUATION FUNCTION
# =====================================================

def evaluate_model(name, X_train_data, X_test_data):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train_data, y_train)

    predictions = model.predict(X_test_data)
    probabilities = model.predict_proba(X_test_data)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Features : {X_train_data.shape[1]}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


# =====================================================
# MODEL A - ALL FEATURES
# =====================================================

evaluate_model(
    "MODEL A - ALL FEATURES",
    X_train,
    X_test
)


# =====================================================
# MODEL B - TOP 7 FEATURES
# =====================================================

X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

evaluate_model(
    "MODEL B - TOP 7 FEATURES",
    X_train_top,
    X_test_top
)