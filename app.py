from flask import Flask, request, jsonify, send_file
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

app = Flask(__name__)
MODEL_PATH = "model.pkl"
WINE_CSV_PATH = "wine_dataset.csv"


# -------- Train on uploaded CSV --------
@app.route("/train", methods=["POST"])
def train():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    df = pd.read_csv(file)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return jsonify({"message": "Model trained successfully"})

# -------- Test on uploaded CSV --------
@app.route("/test", methods=["POST"])
def test():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "No trained model found"}), 400

    file = request.files["file"]
    df = pd.read_csv(file)
    X_test = df.iloc[:, :-1]
    y_test = df.iloc[:, -1]

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return jsonify({"accuracy": acc})


# -------- Test on Wine dataset --------
@app.route("/test_wine", methods=["POST"])
def test_wine():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "No trained model found"}), 400

    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    return jsonify({"wine_accuracy": acc})


# -------- Save Wine dataset to CSV --------
@app.route("/save_wine_csv", methods=["GET"])
def save_wine_csv():
    wine = load_wine(as_frame=True)
    df = wine.data.copy()
    df["target"] = wine.target  # add target column

    df.to_csv(WINE_CSV_PATH, index=False)

    return send_file(WINE_CSV_PATH, as_attachment=True)


# -------- Predict --------
@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "No trained model found"}), 400

    data = request.get_json()
    if not data or "data" not in data:
        return jsonify({"error": "No data provided"}), 400

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([data["data"]])[0]
    return jsonify({"prediction": str(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
    
