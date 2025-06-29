# 📰 Fake News Detector 

## 🧠 Objective

To develop a machine learning-based web application that detects whether a news article is **Fake** or **Real**, displays a **confidence score**, and highlights the **top keywords** influencing the prediction.

---

## 🔍 Project Description

This project leverages **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles. It features a clean, responsive **Flask web interface** powered by **Bootstrap 5**.

### 📌 Key Features

- 🌐 Flask-based web interface
- 🔍 Real-time prediction with confidence score
- 🧠 Random Forest classifier (~99% accuracy)
- 🧾 Highlights important TF-IDF keywords
- ⚙️ Clean, modular folder structure

---

## 📂 Dataset

We use the **Fake and Real News Dataset** from Kaggle:

👉 [Download Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

After downloading manually, create a `data/` folder in your project root and place:
data/
├── Fake.csv
└── True.csv


> ❗ **Note:** Dataset is NOT included in this repository to keep it lightweight.

---

## 📷 Web App Screenshots

### 🏠 Home Page
![Home](screenshots/home.png)

### 📊 Prediction Result Page
![Result](screenshots/result.png)

---

## 🧰 Tech Stack

| Tool            | Purpose                    |
|-----------------|----------------------------|
| Python 3.10      | Programming Language       |
| Flask           | Web Framework              |
| Scikit-learn    | Machine Learning            |
| Pandas / NumPy  | Data Processing            |
| Bootstrap 5     | Frontend Styling           |
| Joblib          | Model Saving/Loading       |

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/SoumajyotiDhut/FakeNewsDetector.git
cd FakeNewsDetector

