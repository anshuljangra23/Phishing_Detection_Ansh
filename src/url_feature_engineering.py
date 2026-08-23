import re
import math
import pandas as pd
from urllib.parse import urlparse


INPUT_FILE = "data/processed/clean_urls.csv"
OUTPUT_FILE = "data/processed/url_features.csv"


SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "password",
    "passwd",
    "secure",
    "security",
    "update",
    "confirm",
    "confirmation",
    "bank",
    "payment",
    "wallet",
    "credential",
    "recover",
    "reset",
    "suspend",
    "unlock",
    "billing",
    "webscr",
]


BRAND_WORDS = [
    "paypal",
    "amazon",
    "microsoft",
    "apple",
    "google",
    "facebook",
    "instagram",
    "netflix",
    "linkedin",
    "bank",
]


def calculate_entropy(text):
    """Calculate Shannon entropy of a string."""

    if not text:
        return 0.0

    probabilities = []

    for char in set(text):
        probabilities.append(text.count(char) / len(text))

    entropy = 0.0

    for probability in probabilities:
        entropy -= probability * math.log2(probability)

    return entropy


def extract_features(url):
    """Extract lexical and structural features from a URL-like string."""

    url = str(url).strip()
    url_lower = url.lower()

    # Add a temporary scheme so urlparse can process the string.
    parsed = urlparse(
        url if "://" in url else "http://" + url
    )

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Count suspicious words
    suspicious_count = sum(
        word in url_lower
        for word in SUSPICIOUS_WORDS
    )

    # Count brand-related words
    brand_count = sum(
        word in url_lower
        for word in BRAND_WORDS
    )

    # Detect IPv4 address
    ip_pattern = (
        r"^(?:\d{1,3}\.){3}\d{1,3}$"
    )

    contains_ip = int(
        re.match(ip_pattern, hostname) is not None
    )

    # Count subdomains
    hostname_parts = [
        part for part in hostname.split(".")
        if part
    ]

    subdomain_count = max(
        len(hostname_parts) - 2,
        0
    )

    # Count special characters
    special_chars = sum(
        not char.isalnum()
        for char in url
    )

    # Digit ratio
    digit_count = sum(
        char.isdigit()
        for char in url
    )

    digit_ratio = (
        digit_count / len(url)
        if url
        else 0
    )

    features = {
        "url_length": len(url),

        "hostname_length": len(hostname),

        "path_length": len(path),

        "query_length": len(query),

        "dot_count": url.count("."),

        "hyphen_count": url.count("-"),

        "underscore_count": url.count("_"),

        "slash_count": url.count("/"),

        "question_count": url.count("?"),

        "equals_count": url.count("="),

        "at_count": url.count("@"),

        "percent_count": url.count("%"),

        "ampersand_count": url.count("&"),

        "digit_count": digit_count,

        "special_char_count": special_chars,

        "digit_ratio": digit_ratio,

        "subdomain_count": subdomain_count,

        "contains_ip": contains_ip,

        "suspicious_keyword_count": suspicious_count,

        "suspicious_keyword_present": int(
            suspicious_count > 0
        ),

        "brand_keyword_count": brand_count,

        "brand_keyword_present": int(
            brand_count > 0
        ),

        "entropy": calculate_entropy(url),

        "path_depth": path.count("/"),

        "query_parameter_count": (
            len(
                [x for x in query.split("&") if x]
            )
            if query
            else 0
        ),

        "double_slash_count": url.count("//"),
    }

    return features


# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)


# =====================================================
# EXTRACT FEATURES
# =====================================================

print("\nExtracting URL features...")

feature_rows = []

for url in df["domain"]:
    feature_rows.append(
        extract_features(url)
    )


features_df = pd.DataFrame(feature_rows)


# =====================================================
# ADD LABEL
# =====================================================

features_df["label"] = df["label"].astype(int)


# =====================================================
# SAVE
# =====================================================

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\nFeature extraction completed.")

print(
    "New dataset shape:",
    features_df.shape
)

print("\nFeatures created:")

for column in features_df.columns:
    print(" -", column)

print("\nFirst 5 rows:")

print(
    features_df.head().to_string(
        index=False
    )
)

print("\nSaved to:")

print(OUTPUT_FILE)