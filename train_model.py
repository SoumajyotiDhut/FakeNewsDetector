# train_model.py
import pandas as pd
import string
import re
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
fake_df = pd.read_csv("data/Fake.csv", encoding='ISO-8859-1', low_memory=False)[['title', 'text']]
true_df = pd.read_csv("data/True.csv", encoding='ISO-8859-1', low_memory=False)[['title', 'text']]

# Add labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Balance dataset
min_size = min(len(fake_df), len(true_df))
fake_df = fake_df.sample(n=min_size, random_state=42)
true_df = true_df.sample(n=min_size, random_state=42)

# Combine and drop nulls
df = pd.concat([fake_df, true_df], ignore_index=True).dropna()

# Remove very short text (junk)
df = df[df['text'].str.strip().str.len() > 30]

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)              # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)           # remove punctuation and digits
    text = re.sub(r"\s+", " ", text).strip()          # normalize whitespace
    return text

df['text'] = df['text'].apply(clean_text)

# Features and labels
X = df['text']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Choose model: Random Forest or Logistic Regression
use_random_forest = True  # 🔁 Set to False to switch to Logistic Regression

if use_random_forest:
    print("🔁 Using Random Forest Classifier")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    print("🔁 Using Logistic Regression")
    model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%\n")
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))

print("\n✅ Model and vectorizer saved to 'model/' folder")

# Make sure to download the Kaggle dataset and place it in `data/`
# Then run this file to train and generate:
#  - model/fake_news_model.pkl
#  - model/tfidf_vectorizer.pkl
