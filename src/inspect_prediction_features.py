import sys
import pandas as pd

from predict_url import extract_url_features


FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "slash_count",
    "question_count",
    "equals_count",
    "at_count",
    "percent_count",
    "ampersand_count",
    "digit_count",
    "special_char_count",
    "digit_ratio",
    "subdomain_count",
    "contains_ip",
    "suspicious_keyword_count",
    "suspicious_keyword_present",
    "brand_keyword_count",
    "brand_keyword_present",
    "entropy",
    "path_depth",
    "query_parameter_count",
    "double_slash_count",
]


urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.amazon.com",
    "http://secure-login-account-verify.com",
    "https://www.rezoni.com/",
]


for url in urls:

    features = extract_url_features(url)[0]

    print("\n" + "=" * 60)
    print(url)
    print("=" * 60)

    for name, value in zip(FEATURE_NAMES, features):
        print(f"{name:30} {value}")