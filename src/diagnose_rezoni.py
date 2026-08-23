import joblib
from scipy.sparse import hstack

from url_features import extract_url_features


MODEL_PATH = "models/final_url_model.pkl"
VECTORIZER_PATH = "models/final_url_tfidf_vectorizer.pkl"

URL = "https://www.rezoni.com/"


print("=" * 60)
print("REZONI MODEL DIAGNOSIS")
print("=" * 60)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -----------------------------------------------------
# 30 engineered features
# -----------------------------------------------------

numeric = extract_url_features(URL)

# -----------------------------------------------------
# TF-IDF
# -----------------------------------------------------

tfidf = vectorizer.transform([URL])

# -----------------------------------------------------
# FULL MODEL
# -----------------------------------------------------

combined = hstack([
    numeric,
    tfidf
])

full_probability = model.predict_proba(combined)[0][1]
full_prediction = model.predict(combined)[0]

print("\nFULL MODEL")
print("Features:", combined.shape[1])
print(
    "Prediction:",
    "PHISHING" if full_prediction == 1 else "LEGITIMATE"
)
print(f"Probability: {full_probability * 100:.2f}%")

# -----------------------------------------------------
# NUMERIC ONLY
# -----------------------------------------------------

numeric_prediction = model.predict(numeric)[0]

print("\nNUMERIC FEATURES ONLY")
print("Features:", numeric.shape[1])

print(
    "Prediction:",
    "PHISHING" if numeric_prediction == 1 else "LEGITIMATE"
)

print(
    "Note: the Random Forest was trained with "
    "combined features, so this numeric-only result "
    "is diagnostic rather than a valid final prediction."
)

print("\n" + "=" * 60)