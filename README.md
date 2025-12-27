# Email Fraud Detection Project

## Introduction
This project is used to detect whether an email message is Safe or Phishing.
It uses Machine Learning and Natural Language Processing (NLP) techniques.

## What this project does
- Takes email text as input
- Analyzes the content of the email
- Predicts whether the email is:
  - Safe Email
  - Phishing Email

## Technologies Used
- Python
- Machine Learning
- NLP (Natural Language Processing)
- Scikit-learn
- Flask (for web application)

## Dataset
The project is trained using a phishing email dataset.
The dataset contains:
- Email Text
- Email Type (Safe or Phishing)

## How the system works
1. Email text is cleaned using NLP.
2. Important words are converted into numbers using TF-IDF.
3. A Machine Learning model (Naive Bayes) is trained.
4. The model predicts whether the email is Safe or Phishing.
5. Result is shown using a web application.

## Project Files
- app.py → Runs the web application
- train_model.py → Trains the machine learning model
- email_model.pkl → Saved trained model
- vectorizer.pkl → Saved text vectorizer
- dataset/ → Contains the dataset
- templates/ → Contains HTML file

## How to run the project
1. Install required libraries:
pip install pandas numpy scikit-learn nltk flask

2. Train the model:
python train_model.py

3. Run the application:
python app.py

4. Open browser and go to:
http://127.0.0.1:5000/

## Output
The system displays whether the entered email is Safe or Phishing.

## Author
Pranjal Singh

