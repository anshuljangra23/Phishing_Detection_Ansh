from predict_url import predict_url


test_urls = [

    # Legitimate
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.amazon.com",
    "https://www.apple.com",
    "https://www.wikipedia.org",
    "https://github.com",

    # Suspicious
    "http://secure-login-account-verify.com",
    "http://bank-login-confirm-password.com",
    "http://free-gift-winner-prize.com",
    "http://verify-account-security-alert.com",

    # Brand impersonation
    "http://google-login-security.com",
    "http://microsoft-account-verify.com",
    "http://amazon-payment-confirm.com",
    "http://paypal-security-login.com",
]