import pandas as pd
import string
import re
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
try:
    fake_df = pd.read_csv("data/Fake.csv", encoding='ISO-8859-1', low_memory=False)[['title', 'text']]
    true_df = pd.read_csv("data/True.csv", encoding='ISO-8859-1', low_memory=False)[['title', 'text']]
except FileNotFoundError as e:
    print(f"Data file missing: {e}")
    exit(1)

# Add labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Basic EDA
print("Dataset Info:")
print("Fake News Count:", len(fake_df))
print("Real News Count:", len(true_df))
print("\nMissing values (fake):\n", fake_df.isnull().sum())
print("\nMissing values (real):\n", true_df.isnull().sum())

# Balance dataset
min_size = min(len(fake_df), len(true_df))
fake_df = fake_df.sample(n=min_size, random_state=42)
true_df = true_df.sample(n=min_size, random_state=42)

# Combine, drop nulls and duplicates
df = pd.concat([fake_df, true_df], ignore_index=True)
df.dropna(subset=['title', 'text'], inplace=True)
df.drop_duplicates(subset=['text'], inplace=True)

# Add text length column
df['text_length'] = df['text'].apply(lambda x: len(x.split()))

# Remove outliers: very short/long articles
df = df[(df['text_length'] > 30) & (df['text_length'] < 3000)]

# Combine title and text
df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)             
    text = re.sub(r"[^a-zA-Z\s]", "", text)         
    text = re.sub(r"\s+", " ", text).strip()          
    return text

# Clean the combined column
df['combined'] = df['combined'].apply(clean_text)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and labels
X = df['combined']
y = df['label']  

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

use_random_forest = True

if use_random_forest:
    print("Using Random Forest Classifier")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    print("Using Logistic Regression")
    model = LogisticRegression(max_iter=1000)
    
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved in 'model/' folder")


# Make sure to download the Kaggle dataset and place it in `data/`
# Then run this file to train and generate:
#  - model/fake_news_model.pkl
#  - model/tfidf_vectorizer.pkl
