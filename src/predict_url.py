import sys
import os
import joblib
from scipy.sparse import hstack

# =====================================================
# IMPORTS
# =====================================================

try:
    # When imported by Flask / as a package
    from src.domain_rules import is_trusted_domain
    from src.url_features import extract_url_features

except ImportError:
    # When running directly:
    # python src\predict_url.py "URL"
    from domain_rules import is_trusted_domain
    from url_features import extract_url_features


# =====================================================
# MODEL FILES
# =====================================================

MODEL_PATH = "models/final_url_model.pkl"
VECTORIZER_PATH = "models/final_url_tfidf_vectorizer.pkl"


# =====================================================
# LOAD MODEL
# =====================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# =====================================================
# PREDICT URL
# =====================================================


def predict_url(url):

    model, vectorizer = load_model()

    url = str(url).strip()

    # Extract the exact same 30 features
    # used during model training.
    numeric_features = extract_url_features(url)

    # Character-level TF-IDF
    text_features = vectorizer.transform([url])

    # Combine features
    combined_features = hstack([
        numeric_features,
        text_features
    ])

    # ML prediction
    prediction = model.predict(
        combined_features
    )[0]

    probability = model.predict_proba(
        combined_features
    )[0][1]

    # =================================================
    # TRUSTED DOMAIN OVERRIDE
    # =================================================

    if is_trusted_domain(url):

        prediction = 0
        probability = 0.01

    return prediction, probability


# =====================================================
# RISK LEVEL
# =====================================================

def get_risk_level(probability):

    if probability >= 0.80:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print('python src\\predict_url.py "URL"')

        print("\nExample:")
        print(
            'python src\\predict_url.py '
            '"https://example.com/login"'
        )

        sys.exit(1)

    url = sys.argv[1].strip()

    # =================================================
    # ML PREDICTION
    # =================================================

    prediction, probability = predict_url(url)

    risk = get_risk_level(probability)

    # =================================================
    # DISPLAY RESULT
    # =================================================

    print("\n" + "=" * 55)
    print("AI PHISHING URL DETECTOR")
    print("=" * 55)

    print("\nURL:")
    print(url)

    print("\nPrediction:")

    if prediction == 1:
        print("PHISHING")
    else:
        print("LEGITIMATE")

    print(
        f"\nPhishing Probability: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Risk Level: {risk}"
    )

    print("\n" + "=" * 55)