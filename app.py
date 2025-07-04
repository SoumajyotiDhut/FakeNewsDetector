from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key' 

# Load model and vectorizer
try:
    model = pickle.load(open("model/fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
except Exception as e:
    raise Exception(f"Error loading model/vectorizer: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form.get('news', '').strip()

        if not news:
            flash("Please enter some news text to analyze.", "warning")
            return redirect(url_for('home'))

        vect = vectorizer.transform([news])

        # Predict label and probability
        prediction = model.predict(vect)[0]
        confidence = round(np.max(model.predict_proba(vect)) * 100, 2)

        # Extract top keywords
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = vect.toarray()[0]
        scored_keywords = sorted(
            [(word, score) for word, score in zip(feature_names, tfidf_scores) if score > 0],
            key=lambda x: x[1],
            reverse=True
        )
        top_keywords = [word for word, _ in scored_keywords[:10]]

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            keywords=top_keywords,
            news=news
        )

if __name__ == '__main__':
    app.run(debug=True)

