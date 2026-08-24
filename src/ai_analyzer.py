import re
from urllib.parse import urlparse

try:
    from src.domain_rules import is_trusted_domain
    from src.url_features import extract_url_features
except ImportError:
    from domain_rules import is_trusted_domain
    from url_features import extract_url_features


# ============================================================
# LOCAL AI-STYLE URL ANALYZER
# ============================================================
#
# This module does NOT use an external AI API.
#
# It combines:
#   1. ML phishing probability
#   2. URL structural indicators
#   3. Suspicious keywords
#   4. Brand impersonation indicators
#   5. Trusted-domain rules
#
# The ML model remains the primary signal.
# ============================================================


SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "account",
    "password",
    "secure",
    "security",
    "confirm",
    "confirmation",
    "update",
    "bank",
    "winner",
    "gift",
    "free",
    "prize",
    "credential",
    "payment",
    "wallet",
    "recover",
    "unlock",
    "authenticate",
    "authentication",
]


BRAND_KEYWORDS = [
    "paypal",
    "amazon",
    "microsoft",
    "google",
    "apple",
    "facebook",
    "instagram",
    "netflix",
    "linkedin",
    "twitter",
    "youtube",
    "whatsapp",
    "bank",
]


# ============================================================
# URL ANALYSIS
# ============================================================

def analyze_url_structure(url):

    url = str(url).strip()

    parsed = urlparse(url)

    hostname = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()

    full_text = url.lower()

    indicators = []
    score = 0

    # --------------------------------------------------------
    # Suspicious keywords
    # --------------------------------------------------------

    found_suspicious = []

    for keyword in SUSPICIOUS_KEYWORDS:

        if keyword in full_text:
            found_suspicious.append(keyword)

    if found_suspicious:

        indicators.append(
            "Suspicious keywords: "
            + ", ".join(sorted(set(found_suspicious)))
        )

        score += min(len(set(found_suspicious)) * 10, 35)

    # --------------------------------------------------------
    # Brand keyword
    # --------------------------------------------------------

    found_brands = []

    for brand in BRAND_KEYWORDS:

        if brand in hostname:

            found_brands.append(brand)

    if found_brands:

        indicators.append(
            "Brand-related hostname: "
            + ", ".join(sorted(set(found_brands)))
        )

        score += 10

    # --------------------------------------------------------
    # HTTP instead of HTTPS
    # --------------------------------------------------------

    if parsed.scheme.lower() == "http":

        indicators.append(
            "URL uses HTTP instead of HTTPS"
        )

        score += 10

    # --------------------------------------------------------
    # IP address
    # --------------------------------------------------------

    if re.match(
        r"^\d{1,3}(\.\d{1,3}){3}$",
        hostname
    ):

        indicators.append(
            "Hostname is an IP address"
        )

        score += 25

    # --------------------------------------------------------
    # Hyphens
    # --------------------------------------------------------

    hyphen_count = hostname.count("-")

    if hyphen_count >= 2:

        indicators.append(
            f"Hostname contains {hyphen_count} hyphens"
        )

        score += min(hyphen_count * 5, 20)

    # --------------------------------------------------------
    # Excessive subdomains
    # --------------------------------------------------------

    parts = hostname.split(".")

    if len(parts) >= 4:

        indicators.append(
            "Hostname contains many subdomains"
        )

        score += 15

    # --------------------------------------------------------
    # @ symbol
    # --------------------------------------------------------

    if "@" in url:

        indicators.append(
            "URL contains @ symbol"
        )

        score += 25

    # --------------------------------------------------------
    # Excessive URL length
    # --------------------------------------------------------

    if len(url) >= 100:

        indicators.append(
            "URL is unusually long"
        )

        score += 10

    # --------------------------------------------------------
    # Encoded characters
    # --------------------------------------------------------

    if "%" in url:

        indicators.append(
            "URL contains encoded characters"
        )

        score += 5

    # --------------------------------------------------------
    # Multiple query parameters
    # --------------------------------------------------------

    if query:

        parameter_count = query.count("&") + 1

        if parameter_count >= 4:

            indicators.append(
                "URL contains many query parameters"
            )

            score += 10

    # --------------------------------------------------------
    # Deep path
    # --------------------------------------------------------

    path_parts = [
        x for x in path.split("/")
        if x
    ]

    if len(path_parts) >= 5:

        indicators.append(
            "URL contains a deep path"
        )

        score += 10

    return {
        "score": min(score, 100),
        "indicators": indicators,
        "suspicious_keywords": sorted(
            set(found_suspicious)
        ),
        "brand_keywords": sorted(
            set(found_brands)
        ),
    }


# ============================================================
# FINAL DECISION
# ============================================================

def analyze_url(url, ml_probability=None):

    url = str(url).strip()

    # --------------------------------------------------------
    # Trusted domain
    # --------------------------------------------------------

    trusted = is_trusted_domain(url)

    if trusted:

        return {
            "prediction": 0,
            "probability": 0.01,
            "risk_level": "LOW",
            "verdict": "LEGITIMATE",
            "confidence": 0.99,
            "explanation": (
                "The domain is present in the trusted-domain "
                "database."
            ),
            "indicators": [],
            "trusted": True,
        }

    # --------------------------------------------------------
    # Get ML probability
    # --------------------------------------------------------

    if ml_probability is None:

        try:

            from src.predict_url import predict_url

        except ImportError:

            from predict_url import predict_url

        _, ml_probability = predict_url(url)

    ml_probability = float(ml_probability)

    # --------------------------------------------------------
    # Analyze URL structure
    # --------------------------------------------------------

    structure = analyze_url_structure(url)

    structure_score = structure["score"]

    indicators = structure["indicators"]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # ML probability is the primary signal.
    #
    # Structural analysis can increase confidence,
    # but a clean-looking URL cannot erase a strong
    # ML phishing probability.
    # --------------------------------------------------------

    if ml_probability >= 0.90:

        verdict = "PHISHING"
        risk_level = "HIGH"

        confidence = min(
            0.95,
            0.70 + (ml_probability * 0.25)
        )

        if indicators:

            explanation = (
                "The machine-learning model detected a "
                "very high phishing probability and the URL "
                "also contains suspicious indicators."
            )

        else:

            explanation = (
                "The machine-learning model detected a "
                "very high phishing probability. The URL "
                "does not contain many obvious structural "
                "indicators, so further verification is "
                "recommended."
            )

    elif ml_probability >= 0.80:

        verdict = "SUSPICIOUS"
        risk_level = "HIGH"

        confidence = min(
            0.90,
            0.60 + (ml_probability * 0.25)
        )

        explanation = (
            "The machine-learning model gives this URL "
            "a high phishing probability. The URL should "
            "be treated as suspicious even though its "
            "visible structure may appear normal."
        )

    elif ml_probability >= 0.60:

        verdict = "SUSPICIOUS"
        risk_level = "MEDIUM"

        confidence = 0.60 + (
            (ml_probability - 0.60) * 0.50
        )

        explanation = (
            "The machine-learning model detected a "
            "moderately high phishing probability. "
            "Additional verification is recommended."
        )

    elif ml_probability >= 0.40:

        verdict = "NEEDS REVIEW"
        risk_level = "MEDIUM"

        confidence = 0.55

        explanation = (
            "The model result is uncertain. The URL "
            "should be reviewed before being considered "
            "safe."
        )

    else:

        verdict = "LIKELY LEGITIMATE"
        risk_level = "LOW"

        confidence = max(
            0.55,
            1.0 - ml_probability
        )

        if indicators:

            explanation = (
                "The machine-learning probability is low, "
                "but some URL indicators deserve attention."
            )

        else:

            explanation = (
                "The URL has a low machine-learning "
                "phishing probability and no major "
                "structural warning signs were detected."
            )

    # --------------------------------------------------------
    # Add structural information
    # --------------------------------------------------------

    if structure_score >= 40:

        explanation += (
            " Several suspicious URL characteristics "
            "were also detected."
        )

    elif structure_score >= 20:

        explanation += (
            " Some suspicious URL characteristics "
            "were detected."
        )

    return {
        "prediction": 1 if verdict in (
            "PHISHING",
            "SUSPICIOUS"
        ) else 0,

        "probability": ml_probability,

        "risk_level": risk_level,

        "verdict": verdict,

        "confidence": min(
            max(confidence, 0.0),
            1.0
        ),

        "explanation": explanation,

        "indicators": indicators,

        "trusted": False,

        "structure_score": structure_score,

        "suspicious_keywords": structure[
            "suspicious_keywords"
        ],

        "brand_keywords": structure[
            "brand_keywords"
        ],
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    test_urls = [

        "https://www.google.com",

        "https://www.rezoni.com/",

        "http://google-login-security.com",

        "http://secure-login-account-verify.com",
    ]

    print("\n" + "=" * 60)
    print("LOCAL AI URL ANALYZER")
    print("=" * 60)

    for url in test_urls:

        try:

            result = analyze_url(url)

            print("\nURL:")
            print(url)

            print(
                f"\nML probability: "
                f"{result['probability'] * 100:.2f}%"
            )

            print(
                f"AI-style verdict: "
                f"{result['verdict']}"
            )

            print(
                f"Risk level: "
                f"{result['risk_level']}"
            )

            print(
                f"Confidence: "
                f"{result['confidence'] * 100:.2f}%"
            )

            print(
                f"Explanation: "
                f"{result['explanation']}"
            )

            if result["indicators"]:

                print("\nIndicators:")

                for indicator in result["indicators"]:

                    print(
                        f"  - {indicator}"
                    )

        except Exception as e:

            print("\nERROR:")
            print(e)

    print("\n" + "=" * 60)
    print("Done.")