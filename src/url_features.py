import re
import math
import numpy as np


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


def calculate_entropy(text):

    if not text:
        return 0.0

    probabilities = []

    for char in set(text):
        probabilities.append(
            text.count(char) / len(text)
        )

    return -sum(
        p * math.log2(p)
        for p in probabilities
        if p > 0
    )


def extract_url_features(url):

    url = str(url).strip()
    url_lower = url.lower()

    clean_url = re.sub(
        r"^https?://",
        "",
        url_lower
    )

    hostname = clean_url.split("/")[0]

    remaining = clean_url[len(hostname):]

    if "?" in remaining:
        path = remaining.split("?")[0]
        query = remaining.split("?", 1)[1]
    else:
        path = remaining
        query = ""

    # -----------------------------
    # Basic URL features
    # -----------------------------

    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    dot_count = url.count(".")
    hyphen_count = url.count("-")
    underscore_count = url.count("_")
    slash_count = url.count("/")
    question_count = url.count("?")
    equals_count = url.count("=")
    at_count = url.count("@")
    percent_count = url.count("%")
    ampersand_count = url.count("&")

    digit_count = sum(
        char.isdigit()
        for char in url
    )

    special_char_count = sum(
        not char.isalnum()
        for char in url
    )

    digit_ratio = (
        digit_count / url_length
        if url_length > 0
        else 0
    )

    # -----------------------------
    # Hostname
    # -----------------------------

    hostname_parts = [
        part
        for part in hostname.split(".")
        if part
    ]

    subdomain_count = max(
        len(hostname_parts) - 2,
        0
    )

    contains_ip = int(
        re.match(
            r"^(?:\d{1,3}\.){3}\d{1,3}$",
            hostname
        ) is not None
    )

    # -----------------------------
    # Suspicious keywords
    # -----------------------------

    suspicious_keyword_count = sum(
        keyword in url_lower
        for keyword in SUSPICIOUS_KEYWORDS
    )

    suspicious_keyword_present = int(
        suspicious_keyword_count > 0
    )

    # -----------------------------
    # Brand analysis
    # -----------------------------

    registered_domain = ""

    if len(hostname_parts) >= 2:

        registered_domain = (
            hostname_parts[-2]
            + "."
            + hostname_parts[-1]
        )

    registered_name = (
        hostname_parts[-2]
        if len(hostname_parts) >= 2
        else ""
    )

    # Brand is actually the registered domain
    brand_in_registered_domain = int(
        registered_name in BRAND_KEYWORDS
    )

    # Brand appears in subdomain
    brand_in_subdomain = int(
        any(
            brand in hostname_parts[:-2]
            for brand in BRAND_KEYWORDS
        )
    )

    # Brand appears inside path
    brand_in_path = int(
        any(
            brand in path
            for brand in BRAND_KEYWORDS
        )
    )

    # Brand + suspicious word combination
    brand_with_suspicious_keyword = int(
        any(
            brand in url_lower
            for brand in BRAND_KEYWORDS
        )
        and suspicious_keyword_present == 1
    )

    brand_keyword_count = sum(
        keyword in url_lower
        for keyword in BRAND_KEYWORDS
    )

    brand_keyword_present = int(
        brand_keyword_count > 0
    )

    # -----------------------------
    # Other structural features
    # -----------------------------

    entropy = calculate_entropy(url_lower)

    path_depth = len(
        [
            part
            for part in path.split("/")
            if part
        ]
    )

    query_parameter_count = (
        len(
            [
                parameter
                for parameter in query.split("&")
                if parameter
            ]
        )
        if query
        else 0
    )

    double_slash_count = url.count("//")

    # -----------------------------
    # Final feature vector
    # -----------------------------

    features = [

        url_length,
        hostname_length,
        path_length,
        query_length,

        dot_count,
        hyphen_count,
        underscore_count,
        slash_count,
        question_count,
        equals_count,
        at_count,
        percent_count,
        ampersand_count,

        digit_count,
        special_char_count,
        digit_ratio,

        subdomain_count,
        contains_ip,

        suspicious_keyword_count,
        suspicious_keyword_present,

        brand_keyword_count,
        brand_keyword_present,

        entropy,
        path_depth,
        query_parameter_count,
        double_slash_count,

        brand_in_registered_domain,
        brand_in_subdomain,
        brand_in_path,
        brand_with_suspicious_keyword,
    ]

    return np.array(
        features,
        dtype=float
    ).reshape(1, -1)