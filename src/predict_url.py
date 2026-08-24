import sys
import os
import joblib
from scipy.sparse import hstack

try:
    from src.domain_rules import is_trusted_domain
    from src.url_features import extract_url_features
except ImportError:
    from domain_rules import is_trusted_domain
    from url_features import extract_url_features


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATH = "models/final_url_model.pkl"
VECTORIZER_PATH = "models/final_url_tfidf_vectorizer.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

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


# ============================================================
# RAW ML PREDICTION
# ============================================================

def predict_url(url):

    model, vectorizer = load_model()

    url = str(url).strip()

    numeric_features = extract_url_features(url)

    text_features = vectorizer.transform([url])

    combined_features = hstack([
        numeric_features,
        text_features
    ])

    prediction = model.predict(
        combined_features
    )[0]

    probability = model.predict_proba(
        combined_features
    )[0][1]

    return prediction, probability


# ============================================================
# FINAL AI ANALYSIS
# ============================================================

def analyze_url(url):

    _, ml_probability = predict_url(url)

    try:
        from src.ai_analyzer import analyze_url as local_ai_analysis
    except ImportError:
        from ai_analyzer import analyze_url as local_ai_analysis

    return local_ai_analysis(
        url,
        ml_probability=ml_probability
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):

    if probability >= 0.80:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            'python src\\predict_url.py "URL"'
        )

        print("\nExample:")
        print(
            'python src\\predict_url.py '
            '"https://example.com/login"'
        )

        sys.exit(1)

    url = sys.argv[1].strip()

    result = analyze_url(url)

    print("\n" + "=" * 60)
    print("AI PHISHING URL DETECTOR")
    print("=" * 60)

    print("\nURL:")
    print(url)

    print("\nPrediction:")
    print(result["verdict"])

    print(
        f"\nPhishing Probability: "
        f"{result['probability'] * 100:.2f}%"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )

    print(
        f"AI Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"\nExplanation:"
    )

    print(
        result["explanation"]
    )

    if result["indicators"]:

        print("\nIndicators:")

        for indicator in result["indicators"]:

            print(
                f"  - {indicator}"
            )

    print("\n" + "=" * 60)