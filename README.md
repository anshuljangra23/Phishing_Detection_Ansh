🛡️ Phishing Detection System

A machine learning based cybersecurity application that analyzes URLs and predicts whether they are Phishing or Legitimate.

The project combines URL feature engineering, machine learning models, domain-based rules, and a web interface to provide a practical phishing URL detection system.

✨ Features
🔍 Phishing URL Detection
🤖 Machine learning based URL classification
🧠 URL feature extraction and feature engineering
🌐 Web-based interface for URL analysis
⚡ Real-time URL prediction
🛡️ Trusted-domain verification
📊 Model evaluation and comparison
🔬 Multiple trained ML models
📁 Large trained models managed using Git LFS
🧪 Scripts for model training, testing, and analysis
🖥️ Application

The application allows a user to enter a URL and receive a prediction from the trained phishing detection system.

Example workflow
User enters URL
       ↓
URL preprocessing
       ↓
Feature extraction
       ↓
Feature engineering
       ↓
Domain/rule analysis
       ↓
Machine Learning Model
       ↓
Prediction
       ↓
Phishing / Legitimate

Add a screenshot of your application here once you have one.

screenshots/
└── home.png

Then you can display it using:

![Phishing Detection System](screenshots/home.png)
🧠 How It Works

The system analyzes different characteristics of a URL before making a prediction.

Some of the analyzed characteristics include:

URL length
Domain structure
Number of subdomains
Special characters
Suspicious URL patterns
HTTPS usage
Path characteristics
Query parameters
Domain-related features
URL text patterns
Trusted-domain information

These features are processed by the machine learning pipeline to classify the URL.

🤖 Machine Learning Models

The repository contains multiple trained models developed during experimentation and model evaluation.

Model	Description
deployable_url_model.pkl	Model intended for application deployment
final_url_model.pkl	Final URL classification model
phishing_combined_model.pkl	Combined phishing detection model
redesigned_url_model.pkl	Redesigned URL classification model
final_url_tfidf_vectorizer.pkl	TF-IDF vectorizer
url_tfidf_vectorizer.pkl	URL text vectorizer

The large model files are stored using Git Large File Storage (Git LFS).

📁 Project Structure
Phishing_Detection_Ansh/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── data/
│   └── trusted_domains.txt
│
├── models/
│   ├── deployable_url_model.pkl
│   ├── final_url_model.pkl
│   ├── final_url_tfidf_vectorizer.pkl
│   ├── phishing_combined_model.pkl
│   ├── redesigned_url_model.pkl
│   └── url_tfidf_vectorizer.pkl
│
├── src/
│   ├── ai_analyzer.py
│   ├── clean_url_dataset.py
│   ├── compare_models.py
│   ├── domain_rules.py
│   ├── feature_ablation.py
│   ├── feature_importance.py
│   ├── find_similar_urls.py
│   ├── predict_url.py
│   ├── train_combined_model.py
│   ├── train_deployable_model.py
│   ├── train_engineered_model.py
│   ├── train_url_model.py
│   ├── url_feature_engineering.py
│   ├── url_features.py
│   └── ...
│
└── templates/
    └── index.html
🛠️ Technologies Used
Python
Scikit-learn
Pandas
NumPy
Flask
HTML/CSS
Machine Learning
Git
GitHub
Git LFS
⚙️ Installation
1. Clone the repository
git clone https://github.com/anshuljangra23/Phishing_Detection_Ansh.git
2. Open the project
cd Phishing_Detection_Ansh
3. Install Git LFS

The trained models are stored using Git LFS.

Install Git LFS from:

https://git-lfs.com/

Then run:

git lfs install

If you clone the repository after Git LFS is installed, the model files should be downloaded automatically.

4. Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate
5. Install dependencies
pip install -r requirements.txt
▶️ Running the Application

Start the application:

python app.py

The Flask development server should provide a local address, typically:

http://127.0.0.1:5000

Open the address in your browser.

🔎 URL Prediction

Enter a URL into the application.

For example:

https://www.google.com

The system processes the URL and returns a classification.

Possible results:

LEGITIMATE

or

PHISHING

The prediction should be treated as an automated assessment rather than an absolute security guarantee.

🧪 Model Development

The repository also contains scripts used during model development and experimentation.

Train URL model
python src/train_url_model.py
Train combined model
python src/train_combined_model.py
Train deployable model
python src/train_deployable_model.py

Additional scripts are available for:

Model comparison
Feature importance analysis
Feature ablation
Dataset cleaning
Prediction testing
URL analysis
Model evaluation
Testing against legitimate URLs
📊 Model Evaluation

The project includes several scripts for evaluating and comparing the trained models.

Examples include:

compare_models.py
feature_importance.py
feature_ablation.py
test_legitimate_dataset.py
test_prediction_set.py
test_real_urls.py

If you have measured accuracy, precision, recall, F1-score, ROC-AUC, or other metrics, they can be added here.

Example
Metric	Score
Accuracy	Add measured value
Precision	Add measured value
Recall	Add measured value
F1 Score	Add measured value

Only add performance numbers that were actually measured from your evaluation results.

📦 Dataset

The project uses URL data for training and evaluating phishing detection models.

Large datasets are not included directly in this repository to keep the repository manageable.

The repository currently includes:

data/trusted_domains.txt

If an external dataset is used, users should follow the dataset's original license and redistribution requirements.

🔐 Security Disclaimer

This project is intended for educational, research, and defensive cybersecurity purposes.

Phishing detection is not perfect. Attackers continuously change URL structures, domains, redirects, and social-engineering techniques.

A URL classified as legitimate should not automatically be considered safe, and a phishing prediction should be investigated using appropriate security procedures.

Do not enter passwords, authentication tokens, API keys, or other sensitive information into untrusted websites while testing the system.

🚀 Future Improvements
 Improve model performance
 Add confidence scores
 Add explainable AI for predictions
 Integrate real-time threat intelligence
 Add domain reputation APIs
 Add browser extension support
 Improve web interface
 Add REST API
 Add automated model retraining
 Deploy the application online
 Add continuous evaluation against newly observed URLs
👨‍💻 Author
Anshul Jangra

GitHub:
https://github.com/anshuljangra23

⭐ Contributing

Contributions, suggestions, and improvements are welcome.

If you find a bug or have an idea for improving the project, feel free to open an Issue or submit a Pull Request.

📄 License

This project currently does not specify an open-source license.

If you decide to make the project open source, consider adding an appropriate license such as the MIT License, depending on your intended usage and the licenses of any third-party datasets or dependencies.

⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.