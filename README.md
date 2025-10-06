# 🍷 Flask Wine Classifier API

A simple **Flask REST API** that trains, tests, and predicts using the **Wine dataset** or your own uploaded CSV file.  
The model uses **Logistic Regression** from `scikit-learn` and can save, test, and predict results easily.

---

## 🚀 Features
- Train a Logistic Regression model using your own dataset (CSV)
- Test the trained model using another dataset
- Use the built-in **Wine dataset** from scikit-learn
- Download the wine dataset as a CSV
- Make predictions from JSON data

---

## 🧠 Requirements
Install dependencies:
```bash
pip install flask pandas scikit-learn
```

project/
```
│── app.py                # Flask API
│── model.pkl             # Saved trained model (auto-generated)
│── wine_dataset.csv      # Saved Wine dataset (auto-generated)
│── README.md             # Project description
```
```
# ▶️ Run the API

Start the Flask server:
python app.py

Server runs on:
http://127.0.0.1:5000
```
# API Endpoints
```
1️⃣ Train on Uploaded CSV

POST /train
Uploads a CSV file and trains the model.

Request:
POST /train
Form Data:
  file: your_dataset.csv
```
# Response:

{"message": "Model trained successfully"}

# 2️⃣ Test on Uploaded CSV
```
POST /test
Uploads a CSV file and tests the trained model.

Request:

POST /test
Form Data:
  file: test_dataset.csv


Response:

{"accuracy": 0.92}
```
