from flask import Flask, render_template, request
import sys
import os

# Allow importing from src/
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict_url import predict_url
from src.domain_rules import is_trusted_domain


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        url = request.form.get("url", "").strip()

        if not url:
            result = {
                "error": "Please enter a URL."
            }

        else:
            try:
                # Trusted-domain override
                trusted = is_trusted_domain(url)

                prediction, probability = predict_url(url)

                if trusted:
                    prediction = 0
                    probability = min(probability, 0.01)

                if probability >= 0.80:
                    risk = "HIGH"
                elif probability >= 0.50:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"

                result = {
                    "url": url,
                    "prediction": (
                        "PHISHING"
                        if prediction == 1
                        else "LEGITIMATE"
                    ),
                    "probability": round(probability * 100, 2),
                    "risk": risk
                }

            except Exception as e:
                result = {
                    "error": str(e)
                }

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )