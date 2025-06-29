from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = [news]
        vect = vectorizer.transform(data)

        # Predict label and get confidence
        prediction = model.predict(vect)[0]
        probas = model.predict_proba(vect)
        confidence = max(probas[0]) * 100

        # Top influential keywords based on TF-IDF
        feature_names = vectorizer.get_feature_names_out()
        tfidf_vector = vect[0]
        tfidf_scores = zip(feature_names, tfidf_vector.toarray()[0])
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, score in sorted_scores if score > 0][:10]

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=round(confidence, 2),
            keywords=top_keywords
        )

if __name__ == '__main__':
    app.run(debug=True)
