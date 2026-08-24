from flask import Flask, render_template, request
import os
import sys

# Allow importing from src/
sys.path.insert(
    0,
    os.path.abspath(os.path.dirname(__file__))
)

from src.ai_analyzer import analyze_url


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

                # ==========================================
                # LOCAL AI + ML ANALYSIS
                # ==========================================

                analysis = analyze_url(url)

                # ==========================================
                # PREPARE RESULT FOR WEB PAGE
                # ==========================================

                result = {
                    "url": url,

                    "prediction": analysis["verdict"],

                    "probability": round(
                        analysis["probability"] * 100,
                        2
                    ),

                    "risk": analysis["risk_level"],

                    "confidence": round(
                        analysis["confidence"] * 100,
                        2
                    ),

                    "explanation": analysis["explanation"],

                    "indicators": analysis.get(
                        "indicators",
                        []
                    ),

                    "structure_score": analysis.get(
                        "structure_score",
                        0
                    ),

                    "trusted": analysis.get(
                        "trusted",
                        False
                    )
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