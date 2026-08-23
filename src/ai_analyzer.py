import re
from urllib.parse import urlparse


# ============================================================
# LOCAL AI-STYLE URL ANALYZER
# ============================================================
#
# This module does NOT call an external AI API.
# It analyzes multiple independent URL signals and produces
# an interpretable assessment that can be combined with the
# machine-learning model.
#
# ============================================================


SUSPICIOUS_KEYWORDS = [
    "login",
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
    "signin",
    "credential",
    "payment",
    "wallet",
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
    "ebay",
]


def analyze_url(url, ml_probability, features):
    """
    Analyze a URL using interpretable local evidence.

    Parameters
    ----------
    url : str
        URL being analyzed.

    ml_probability : float
        Probability produced by the Random Forest.

    features : numpy array
        30 engineered URL features.

    Returns
    -------
    dict
        AI-style analysis result.
    """

    url = str(url).strip()
    lower_url = url.lower()

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Remove numpy array wrapper.
    values = features.flatten().tolist()

    # Current 30-feature layout.
    url_length = values[0]
    hostname_length = values[1]
    path_length = values[2]
    query_length = values[3]
    dot_count = values[4]
    hyphen_count = values[5]
    underscore_count = values[6]
    slash_count = values[7]
    question_count = values[8]
    equals_count = values[9]
    at_count = values[10]
    percent_count = values[11]
    ampersand_count = values[12]
    digit_count = values[13]
    special_char_count = values[14]
    digit_ratio = values[15]
    subdomain_count = values[16]
    contains_ip = values[17]
    suspicious_keyword_count = values[18]
    suspicious_keyword_present = values[19]
    brand_keyword_count = values[20]
    brand_keyword_present = values[21]
    entropy = values[22]
    path_depth = values[23]
    query_parameter_count = values[24]
    double_slash_count = values[25]

    suspicious_reasons = []
    legitimate_reasons = []

    score = 0

    # ========================================================
    # HTTPS
    # ========================================================

    if parsed.scheme.lower() == "https":
        legitimate_reasons.append(
            "Uses HTTPS."
        )
    elif parsed.scheme.lower() == "http":
        suspicious_reasons.append(
            "Uses HTTP instead of HTTPS."
        )
        score += 1

    # ========================================================
    # IP ADDRESS
    # ========================================================

    if contains_ip:
        suspicious_reasons.append(
            "Uses an IP address instead of a normal domain."
        )
        score += 4

    # ========================================================
    # SUSPICIOUS KEYWORDS
    # ========================================================

    found_suspicious = [
        keyword
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in lower_url
    ]

    if found_suspicious:
        suspicious_reasons.append(
            "Contains suspicious terms: "
            + ", ".join(found_suspicious)
            + "."
        )

        score += min(len(found_suspicious) * 2, 8)

    # ========================================================
    # BRAND IMPERSONATION
    # ========================================================

    found_brands = [
        brand
        for brand in BRAND_KEYWORDS
        if brand in hostname.lower()
    ]

    if found_brands:

        # A brand appearing directly in its expected domain
        # is not automatically suspicious.
        base_parts = hostname.lower().split(".")

        brand_is_domain = any(
            brand == part
            for brand in found_brands
            for part in base_parts
        )

        # Suspicious when a brand is combined with deceptive
        # words or separated by hyphens.
        deceptive_words = [
            "login",
            "verify",
            "security",
            "secure",
            "account",
            "confirm",
            "password",
            "signin",
        ]

        deceptive_brand_pattern = (
            "-" in hostname
            and any(
                word in hostname.lower()
                for word in deceptive_words
            )
        )

        if deceptive_brand_pattern:

            suspicious_reasons.append(
                "Possible brand impersonation detected."
            )

            score += 6

        elif brand_is_domain:

            legitimate_reasons.append(
                "Brand name appears as part of the domain."
            )

    # ========================================================
    # HYPHENS
    # ========================================================

    if hyphen_count >= 2:

        suspicious_reasons.append(
            "Domain contains multiple hyphens."
        )

        score += 2

    # ========================================================
    # SUBDOMAINS
    # ========================================================

    if subdomain_count >= 3:

        suspicious_reasons.append(
            "Uses an unusually deep subdomain structure."
        )

        score += 2

    # ========================================================
    # URL LENGTH
    # ========================================================

    if url_length >= 100:

        suspicious_reasons.append(
            "URL is unusually long."
        )

        score += 2

    # ========================================================
    # QUERY PARAMETERS
    # ========================================================

    if query_parameter_count >= 4:

        suspicious_reasons.append(
            "Contains many query parameters."
        )

        score += 2

    # ========================================================
    # SPECIAL CHARACTERS
    # ========================================================

    if at_count > 0:

        suspicious_reasons.append(
            "Contains '@' character, which can obscure the "
            "actual destination."
        )

        score += 4

    if percent_count >= 3:

        suspicious_reasons.append(
            "Contains several encoded characters."
        )

        score += 2

    # ========================================================
    # PATH COMPLEXITY
    # ========================================================

    if path_depth >= 5:

        suspicious_reasons.append(
            "Uses a deeply nested URL path."
        )

        score += 2

    # ========================================================
    # ENTROPY
    # ========================================================

    if entropy >= 4.5:

        suspicious_reasons.append(
            "URL has relatively high character randomness."
        )

        score += 2

    # ========================================================
    # CLEAN ROOT DOMAIN
    # ========================================================

    if (
        parsed.scheme.lower() == "https"
        and
        path in ("", "/")
        and
        query == ""
        and
        hyphen_count == 0
        and
        contains_ip == 0
        and
        suspicious_keyword_count == 0
        and
        at_count == 0
    ):

        legitimate_reasons.append(
            "Clean HTTPS root-domain structure with no "
            "obvious phishing indicators."
        )

    # ========================================================
    # LOCAL EVIDENCE SCORE
    # ========================================================

    # Convert the rule score to a bounded risk score.
    local_risk = min(score / 20.0, 1.0)

    # ========================================================
    # COMBINE ML + LOCAL EVIDENCE
    # ========================================================

    # We intentionally do NOT simply replace the ML model.
    #
    # ML remains the primary statistical signal.
    # Local analysis provides additional interpretable evidence.

    if ml_probability >= 0.90 and local_risk >= 0.25:

        final_probability = (
            ml_probability * 0.75
            + local_risk * 0.25
        )

    elif ml_probability >= 0.70:

        final_probability = (
            ml_probability * 0.55
            + local_risk * 0.45
        )

    else:

        final_probability = (
            ml_probability * 0.70
            + local_risk * 0.30
        )

    final_probability = max(
        0.0,
        min(final_probability, 1.0)
    )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    if final_probability >= 0.80:

        verdict = "PHISHING"
        confidence = final_probability

    elif final_probability >= 0.50:

        verdict = "SUSPICIOUS"
        confidence = final_probability

    else:

        verdict = "LIKELY LEGITIMATE"
        confidence = 1.0 - final_probability

    # ========================================================
    # EXPLANATION
    # ========================================================

    if verdict == "PHISHING":

        if suspicious_reasons:

            explanation = (
                "Multiple phishing-related indicators were "
                "detected."
            )

        else:

            explanation = (
                "The machine-learning model considers this URL "
                "strongly similar to phishing URLs."
            )

    elif verdict == "SUSPICIOUS":

        explanation = (
            "The URL contains some potentially suspicious "
            "characteristics but does not provide enough "
            "evidence for a high-confidence phishing verdict."
        )

    else:

        if legitimate_reasons:

            explanation = (
                "The URL has a relatively clean structure and "
                "few obvious phishing indicators."
            )

        else:

            explanation = (
                "The available URL evidence does not strongly "
                "indicate phishing."
            )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "final_probability": final_probability,
        "local_risk": local_risk,
        "ml_probability": ml_probability,
        "score": score,
        "suspicious_reasons": suspicious_reasons,
        "legitimate_reasons": legitimate_reasons,
        "explanation": explanation,
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LOCAL AI URL ANALYZER")
    print("=" * 60)

    test_urls = [
        (
            "https://www.google.com",
            0.01
        ),
        (
            "https://www.rezoni.com/",
            0.8733
        ),
        (
            "http://google-login-security.com",
            0.8933
        ),
        (
            "http://secure-login-account-verify.com",
            0.9933
        ),
    ]

    # Minimal test features.
    # The real predictor passes actual 30-feature vectors.
    import numpy as np

    for url, probability in test_urls:

        dummy_features = np.zeros(
            (1, 30),
            dtype=float
        )

        result = analyze_url(
            url,
            probability,
            dummy_features
        )

        print("\nURL:")
        print(url)

        print(
            "ML probability:",
            f"{probability * 100:.2f}%"
        )

        print(
            "AI-style verdict:",
            result["verdict"]
        )

        print(
            "Confidence:",
            f"{result['confidence'] * 100:.2f}%"
        )

        print(
            "Explanation:",
            result["explanation"]
        )

    print("\nDone.")