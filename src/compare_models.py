import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from url_features import extract_url_features


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/processed/url_features.csv"
)

url_df = pd.read_csv(
    "data/processed/clean_urls.csv"
)

urls = url_df["domain"].fillna("").astype(str)

y = df["label"].astype(int)


# =====================================================
# NUMERIC FEATURES
# =====================================================

feature_columns = [
    column
    for column in df.columns
    if column != "label"
]

X_numeric = df[feature_columns]

print("\nDataset shape:", df.shape)
print("Training numeric features:", X_numeric.shape[1])


# =====================================================
# TF-IDF
# =====================================================

print("\nCreating character-level TF-IDF...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=10000,
    sublinear_tf=True
)

X_text = vectorizer.fit_transform(urls)

print("TF-IDF shape:", X_text.shape)


# =====================================================
# SAME TRAIN / TEST SPLIT
# =====================================================

indices = list(range(len(df)))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# =====================================================
# MODEL A: NUMERIC ONLY
# =====================================================

print("\n" + "=" * 60)
print("MODEL A - NUMERIC FEATURES ONLY")
print("=" * 60)

model_numeric = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model_numeric.fit(
    X_numeric.iloc[train_idx],
    y_train
)

pred_numeric = model_numeric.predict(
    X_numeric.iloc[test_idx]
)

prob_numeric = model_numeric.predict_proba(
    X_numeric.iloc[test_idx]
)[:, 1]

print(
    "Accuracy:",
    f"{accuracy_score(y_test, pred_numeric):.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc_score(y_test, prob_numeric):.4f}"
)


# =====================================================
# MODEL B: TF-IDF ONLY
# =====================================================

print("\n" + "=" * 60)
print("MODEL B - TF-IDF ONLY")
print("=" * 60)

model_tfidf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model_tfidf.fit(
    X_text[train_idx],
    y_train
)

pred_tfidf = model_tfidf.predict(
    X_text[test_idx]
)

prob_tfidf = model_tfidf.predict_proba(
    X_text[test_idx]
)[:, 1]

print(
    "Accuracy:",
    f"{accuracy_score(y_test, pred_tfidf):.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc_score(y_test, prob_tfidf):.4f}"
)


# =====================================================
# MODEL C: COMBINED
# =====================================================

print("\n" + "=" * 60)
print("MODEL C - COMBINED")
print("=" * 60)

X_combined = hstack([
    X_numeric.values,
    X_text
]).tocsr()

model_combined = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model_combined.fit(
    X_combined[train_idx],
    y_train
)

pred_combined = model_combined.predict(
    X_combined[test_idx]
)

prob_combined = model_combined.predict_proba(
    X_combined[test_idx]
)[:, 1]

print(
    "Accuracy:",
    f"{accuracy_score(y_test, pred_combined):.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc_score(y_test, prob_combined):.4f}"
)


# =====================================================
# REZONI TEST
# =====================================================

URL = "https://www.rezoni.com/"

print("\n" + "=" * 60)
print("REZONI COMPARISON")
print("=" * 60)

print("\nURL:")
print(URL)


# =====================================================
# IMPORTANT:
# Model A was trained on the DATASET'S numeric features.
# Therefore we must select those exact same features.
# =====================================================

rezoni_all_features = extract_url_features(URL)

print(
    "\nExtracted URL feature shape:",
    rezoni_all_features.shape
)


# -----------------------------------------------------
# Model A expects the same number/order of features
# as X_numeric.
# -----------------------------------------------------

numeric_feature_count = X_numeric.shape[1]

if rezoni_all_features.shape[1] < numeric_feature_count:

    raise ValueError(
        f"REZONI has only "
        f"{rezoni_all_features.shape[1]} features, "
        f"but Model A expects "
        f"{numeric_feature_count}."
    )


rezoni_features = rezoni_all_features[
    :, :numeric_feature_count
]


# =====================================================
# TF-IDF FEATURES
# =====================================================

rezoni_tfidf = vectorizer.transform([URL])

print(
    "REZONI TF-IDF shape:",
    rezoni_tfidf.shape
)


# =====================================================
# COMBINED FEATURES
# =====================================================

rezoni_combined = hstack([
    rezoni_features,
    rezoni_tfidf
]).tocsr()

print(
    "REZONI combined shape:",
    rezoni_combined.shape
)


# =====================================================
# MODEL A PREDICTION
# =====================================================

p1 = model_numeric.predict_proba(
    rezoni_features
)[0][1]


# =====================================================
# MODEL B PREDICTION
# =====================================================

p2 = model_tfidf.predict_proba(
    rezoni_tfidf
)[0][1]


# =====================================================
# MODEL C PREDICTION
# =====================================================

p3 = model_combined.predict_proba(
    rezoni_combined
)[0][1]


# =====================================================
# RESULTS
# =====================================================

print(
    "\nNumeric-only probability:",
    f"{p1 * 100:.2f}%"
)

print(
    "TF-IDF-only probability:",
    f"{p2 * 100:.2f}%"
)

print(
    "Combined probability:",
    f"{p3 * 100:.2f}%"
)


# =====================================================
# INTERPRETATION
# =====================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

if p1 < 0.50:
    print("Numeric model: LEGITIMATE")
else:
    print("Numeric model: PHISHING")

if p2 < 0.50:
    print("TF-IDF model: LEGITIMATE")
else:
    print("TF-IDF model: PHISHING")

if p3 < 0.50:
    print("Combined model: LEGITIMATE")
else:
    print("Combined model: PHISHING")


print("\nDone.")