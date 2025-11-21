
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import time
import json
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ---------------------------------------
# JSON FILE PATHS (Render allows ONLY /tmp)
# ---------------------------------------
LATEST_FILE = "/tmp/iot_latest.json"
HISTORY_FILE = "/tmp/iot_history.json"


# ---------------------------------------
# Load JSON safely
# ---------------------------------------
def load_json(path, default):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(default, f, indent=4)
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return default


# ---------------------------------------
# Save JSON safely
# ---------------------------------------
def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------
# Load ML Models
# ---------------------------------------
print("🔄 Loading Random Forest model...")
rf_model = joblib.load("rf_model.pkl")

print("🔄 Loading Label Encoder...")
label_encoder = joblib.load("label_encoder.pkl")

print("🔄 Loading LSTM model (.h5)...")
lstm_model = load_model("lstm_model.h5")

print("🔄 Loading LSTM scaler...")
scaler = joblib.load("scaler.pkl")


# ---------------------------------------
# Load permanent IoT data from JSON
# ---------------------------------------
iot_latest = load_json(LATEST_FILE, {
    "timestamp": time.time(),
    "tds": 500,
    "turbidity": 3.0,
    "city": "Unknown",
    "latitude": None,
    "longitude": None
})

iot_history = load_json(HISTORY_FILE, [])


# ---------------------------------------
# Root Route
# ---------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Water Quality API (RF + LSTM) is live 🚀",
        "routes": {
            "/predict": "POST → Predict using Random Forest",
            "/predict_future_quality": "POST → Predict future with LSTM",
            "/iot_latest": "GET → Latest sensor data",
            "/iot_history": "GET → Full IoT history",
            "/add_history": "POST → Add new entry",
            "/search_history": "GET → Search history by city"
        }
    })


# ---------------------------------------
# Predict Quality (RF)
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict_quality():
    try:
        data = request.get_json()
        tds = float(data["tds"])
        turbidity = float(data["turbidity"])

        prediction = rf_model.predict([[tds, turbidity]])[0]
        label = label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(rf_model.predict_proba([[tds, turbidity]]) * 100)

        return jsonify({
            "prediction": label,
            "confidence": round(float(confidence), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# LSTM Future Prediction
# ---------------------------------------
@app.route("/predict_future_quality", methods=["POST"])
def predict_future_quality():
    try:
        data = request.get_json()
        steps = int(data.get("steps", 7))

        last_known = np.array([[500, 3]], dtype=float)
        scaled = scaler.transform(last_known).reshape(1, 1, 2)

        future_predictions = []

        for _ in range(steps):
            pred = lstm_model.predict(scaled)[0]
            inv = scaler.inverse_transform([pred])[0]

            tds_pred = float(inv[0])
            turb_pred = float(inv[1])

            quality = "Safe"
            if tds_pred > 900 or turb_pred > 5:
                quality = "Unsafe"
            elif tds_pred > 600 or turb_pred > 3:
                quality = "Moderate"

            future_predictions.append({
                "TDS": tds_pred,
                "Turbidity": turb_pred,
                "Quality": quality
            })

            scaled = pred.reshape(1, 1, 2)

        return jsonify(future_predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# Get Latest IoT Values
# ---------------------------------------
@app.route("/iot_latest", methods=["GET"])
def get_latest():
    return jsonify(iot_latest)


# ---------------------------------------
# Get IoT History
# ---------------------------------------
@app.route("/iot_history", methods=["GET"])
def get_history():
    return jsonify(iot_history)


# ---------------------------------------
# Add New IoT Entry (Permanent Storage)
# ---------------------------------------
@app.route("/add_history", methods=["POST"])
def add_history():
    try:
        global iot_latest, iot_history

        data = request.get_json()

        entry = {
            "timestamp": time.time(),
            "tds": float(data["tds"]),
            "turbidity": float(data["turbidity"]),
            "city": data.get("city", "Unknown"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
        }

        # Update latest reading
        iot_latest = entry
        save_json(LATEST_FILE, iot_latest)

        # Add to full history
        iot_history.append(entry)
        save_json(HISTORY_FILE, iot_history)

        return jsonify({"status": "saved", "entry": entry})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# Search History by City
# ---------------------------------------
@app.route("/search_history", methods=["GET"])
def search_history():
    try:
        city = request.args.get("city", "").lower()

        results = [
            h for h in iot_history
            if h.get("city", "").lower() == city
        ]

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# Run locally
# ---------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
