import pandas as pd
import joblib

from scipy.sparse import hstack
from url_features import extract_url_features


# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "models/final_url_model.pkl"
VECTORIZER_PATH = "models/final_url_tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/processed/clean_urls.csv"
)

legitimate = df[
    df["label"] == 0
].copy()


# Remove duplicates
legitimate = legitimate.drop_duplicates(
    subset=["domain"]
)


# Test first 100 legitimate URLs
test_urls = legitimate["domain"].head(100)


# =====================================================
# TEST
# =====================================================

results = []

for url in test_urls:

    url = str(url)

    numeric = extract_url_features(url)

    tfidf = vectorizer.transform([url])

    combined = hstack([
        numeric,
        tfidf
    ])

    probability = model.predict_proba(
        combined
    )[0][1]

    prediction = int(
        probability >= 0.50
    )

    results.append({
        "url": url,
        "probability": probability,
        "prediction": prediction
    })


result_df = pd.DataFrame(results)


# =====================================================
# RESULTS
# =====================================================

false_positives = result_df[
    result_df["prediction"] == 1
]

correct = result_df[
    result_df["prediction"] == 0
]


print("=" * 70)
print("LEGITIMATE URL FALSE-POSITIVE TEST")
print("=" * 70)

print(
    f"\nLegitimate URLs tested: {len(result_df)}"
)

print(
    f"Correctly classified: {len(correct)}"
)

print(
    f"False positives: {len(false_positives)}"
)

print(
    f"False-positive rate: "
    f"{len(false_positives) / len(result_df) * 100:.2f}%"
)


# =====================================================
# WORST FALSE POSITIVES
# =====================================================

print("\n" + "=" * 70)
print("WORST LEGITIMATE FALSE POSITIVES")
print("=" * 70)

if len(false_positives) > 0:

    print(
        false_positives
        .sort_values(
            "probability",
            ascending=False
        )
        .head(30)
        .to_string(index=False)
    )

else:

    print("\nNo false positives found.")


print("\nDone.")