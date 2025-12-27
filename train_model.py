import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (only first time)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/Phishing_Email.csv")

# Drop unnecessary column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Convert labels to numeric
# Safe Email -> 0, Phishing Email -> 1
df['Email Type'] = df['Email Type'].map({
    'Safe Email': 0,
    'Phishing Email': 1
})

# Text cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Apply cleaning
df['clean_text'] = df['Email Text'].apply(clean_text)

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['Email Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("email_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
