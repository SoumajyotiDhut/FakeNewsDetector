# 📰 Fake News Detector

## 🧠 Objective

This project aims to develop a machine learning-based web application that detects whether a news article is **Fake** or **Real**. It also displays a **confidence score** for the prediction and highlights the **top keywords** that influenced the decision.

---

## 🔍 Project Description

This application leverages **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles. It features a clean, responsive **Flask web interface** built with **Bootstrap 5**.

### 📌 Key Features

- 🌐 **Flask-based web interface**: A user-friendly and interactive online platform.
- 🔍 **Real-time prediction with confidence score**: Get instant results with an indication of how certain the model is.
- 🧠 **Random Forest classifier**: Utilizes a robust machine learning algorithm, achieving approximately **99% accuracy**.
- 🧾 **Highlights important TF-IDF keywords**: Understand which words are most influential in the fake/real news classification.
- ⚙️ **Clean, modular folder structure**: Easy to navigate and maintain the codebase.

---

## 📂 Dataset

The project uses the **Fake and Real News Dataset** from Kaggle.

👉 [Download Dataset from Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

After downloading, create a `data/` folder in your project's root directory and place both `Fake.csv` and `True.csv` inside it:
# 📰 Fake News Detector

## 🧠 Objective

This project aims to develop a machine learning-based web application that detects whether a news article is **Fake** or **Real**. It also displays a **confidence score** for the prediction and highlights the **top keywords** that influenced the decision.

---

## 🔍 Project Description

This application leverages **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles. It features a clean, responsive **Flask web interface** built with **Bootstrap 5**.

### 📌 Key Features

- 🌐 **Flask-based web interface**: A user-friendly and interactive online platform.
- 🔍 **Real-time prediction with confidence score**: Get instant results with an indication of how certain the model is.
- 🧠 **Random Forest classifier**: Utilizes a robust machine learning algorithm, achieving approximately **99% accuracy**.
- 🧾 **Highlights important TF-IDF keywords**: Understand which words are most influential in the fake/real news classification.
- ⚙️ **Clean, modular folder structure**: Easy to navigate and maintain the codebase.

---

## 📂 Dataset

The project uses the **Fake and Real News Dataset** from Kaggle.

👉 [Download Dataset from Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

After downloading, create a `data/` folder in your project's root directory and place both `Fake.csv` and `True.csv` inside it:
data/
├── Fake.csv
└── True.csv


> ❗ **Note:** The dataset is not included in this repository to keep it lightweight.

---

## 📷 Web App Screenshots

### 🏠 Home Page

![Home](screenshots/home.png)

### 📊 Prediction Result Page

![Result](screenshots/result.png)

---

## 🧰 Tech Stack

| Tool           | Purpose               |
|----------------|------------------------|
| Python 3.10    | Programming Language   |
| Flask          | Web Framework          |
| Scikit-learn   | Machine Learning       |
| Pandas / NumPy | Data Processing        |
| Bootstrap 5    | Frontend Styling       |
| Joblib         | Model Saving/Loading   |

---

project: Fake News Detector
description: |
  A machine learning-powered Flask web app that detects if a news article is Fake or Real,
  displays a confidence score, and highlights important keywords using TF-IDF.

steps:

  - step: Clone the Repository
    commands:
      - git clone https://github.com/SoumajyotiDhut/FakeNewsDetector.git
      - cd FakeNewsDetector

  - step: (Optional) Create Virtual Environment
    windows:
      - python -m venv venv
      - venv\Scripts\activate
    linux_or_mac:
      - python3 -m venv venv
      - source venv/bin/activate

  - step: Install Required Packages
    commands:
      - pip install -r requirements.txt

  - step: Download Dataset from Kaggle
    description: |
      Visit the Kaggle dataset page:
      https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

      After downloading, place the files like this:
    structure: |
      data/
      ├── Fake.csv
      └── True.csv

  - step: Train the Model
    commands:
      - python train_model.py
    output:
      - model/fake_news_model.pkl
      - model/tfidf_vectorizer.pkl

  - step: Run the Web App
    commands:
      - python app.py
    browser: http://127.0.0.1:5000

  - step: Stop the Server
    commands:
      - Press Ctrl + C in the terminal

notes: |
  Make sure you have Python 3.9 or above installed.
  This project uses Flask, scikit-learn, Pandas, NumPy, and Bootstrap 5.

author: Soumajyoti Dhut
license: MIT
repository: https://github.com/SoumajyotiDhut/FakeNewsDetector

