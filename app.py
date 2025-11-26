from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
temp_model = joblib.load("models/temp_model.pkl")
co2_model = joblib.load("models/co2_model.pkl")
sea_model = joblib.load("models/sea_model.pkl")

@app.route("/")
def home():
    return "Climate Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "year" not in data:
        return jsonify({"error": "year is required"}), 400
    
    year = float(data["year"])
    X = pd.DataFrame([[year]], columns=["Year"])

    temp_pred = temp_model.predict(X)[0]
    co2_pred = co2_model.predict(X)[0]
    sea_pred = sea_model.predict(X)[0]

    return jsonify({
        "year": year,
        "predictions": {
            "temperature_anomaly": float(temp_pred),
            "co2_ppm": float(co2_pred),
            "sea_level_change_mm": float(sea_pred)
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
