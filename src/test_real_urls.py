from predict_url import predict_url


test_urls = [

    # Known legitimate examples
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.amazon.com",
    "https://www.wikipedia.org",
    "https://www.apple.com",
    "https://www.github.com",

    # Suspicious examples
    "http://secure-login-account-verify.com",
    "http://bank-login-confirm-password.com",
    "http://free-gift-winner-prize.com",
    "http://verify-account-security-alert.com",
]


print("\n" + "=" * 70)
print("REAL-WORLD URL VALIDATION")
print("=" * 70)


for url in test_urls:

    prediction, probability = predict_url(url)

    if prediction == 1:
        result = "PHISHING"
    else:
        result = "LEGITIMATE"

    print("\nURL:", url)
    print("Prediction:", result)
    print(f"Phishing probability: {probability * 100:.2f}%")


print("\n" + "=" * 70)