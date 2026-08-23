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
].drop_duplicates(
    subset=["domain"]
).copy()


urls = legitimate["domain"].fillna("").astype(str)


print("=" * 70)
print("FULL LEGITIMATE DATASET EVALUATION")
print("=" * 70)

print(
    f"\nLegitimate URLs: {len(urls)}"
)


# =====================================================
# PREDICT IN BATCHES
# =====================================================

probabilities = []

batch_size = 1000

for start in range(
    0,
    len(urls),
    batch_size
):

    batch_urls = urls.iloc[
        start:start + batch_size
    ]

    numeric_features = [
        extract_url_features(url)[0]
        for url in batch_urls
    ]

    X_numeric = pd.DataFrame(
        numeric_features
    ).values

    X_text = vectorizer.transform(
        batch_urls
    )

    X_combined = hstack([
        X_numeric,
        X_text
    ])

    batch_probabilities = model.predict_proba(
        X_combined
    )[:, 1]

    probabilities.extend(
        batch_probabilities
    )

    print(
        f"Processed "
        f"{min(start + batch_size, len(urls))}"
        f"/{len(urls)}"
    )


# =====================================================
# RESULTS
# =====================================================

legitimate["phishing_probability"] = probabilities


print("\n" + "=" * 70)
print("FALSE POSITIVE ANALYSIS")
print("=" * 70)


for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:

    count = (
        legitimate["phishing_probability"]
        >= threshold
    ).sum()

    rate = (
        count / len(legitimate) * 100
    )

    print(
        f"\nThreshold {threshold:.2f}:"
    )

    print(
        f"False positives: {count}"
    )

    print(
        f"False-positive rate: {rate:.2f}%"
    )


# =====================================================
# TOP FALSE POSITIVES
# =====================================================

print("\n" + "=" * 70)
print("TOP 30 LEGITIMATE URLs WITH HIGHEST RISK")
print("=" * 70)

top = legitimate.sort_values(
    "phishing_probability",
    ascending=False
).head(30)

print(
    top[
        [
            "domain",
            "phishing_probability"
        ]
    ].to_string(index=False)
)


# =====================================================
# REZONI POSITION
# =====================================================

rezoni_probability = 0.835

rank = (
    legitimate[
        "phishing_probability"
    ] >= rezoni_probability
).sum()

percentile = (
    1 -
    rank / len(legitimate)
) * 100


print("\n" + "=" * 70)
print("REZONI ANALYSIS")
print("=" * 70)

print(
    f"\nRezoni probability: "
    f"{rezoni_probability * 100:.2f}%"
)

print(
    f"Legitimate URLs with "
    f"equal/higher probability: {rank}"
)

print(
    f"Rezoni is above approximately "
    f"{percentile:.2f}% of legitimate URLs."
)

print("\nDone.")