from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first run)
nltk.download('stopwords')

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("email_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Stopwords (same as training)
stop_words = set(stopwords.words('english'))

# EXACT SAME cleaning as training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']

    # Clean input
    cleaned_email = clean_text(email)

    # Vectorize
    email_vector = vectorizer.transform([cleaned_email])

    # Predict probability
    prob_phishing = model.predict_proba(email_vector)[0][1] * 100

    # Decision threshold
    if prob_phishing >= 60:
        result = f"⚠ PHISHING EMAIL DETECTED ({prob_phishing:.2f}% confidence)"
    else:
        result = f"✅ SAFE EMAIL ({100 - prob_phishing:.2f}% confidence)"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
