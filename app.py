from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import math
import os
import time
import json
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# ============================================================
# HELPERS
# ============================================================
def clean(value):
    try:
        if value is None:
            return 0
        if isinstance(value, float) and math.isnan(value):
            return 0
        return float(value)
    except:
        return 0


# ============================================================
# LOAD MODELS
# ============================================================
print("Loading RandomForest model...")
with open("rf_model.pkl", "rb") as f:
    classifier_model = pickle.load(f)

print("Loading LabelEncoder...")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Loading LSTM model + scaler...")
lstm_model = load_model("lstm_model.h5")
lstm_scaler = joblib.load("scaler.pkl")

# Load full dataset for LSTM windowing
df = pd.read_csv("water_quality_big_dataset.csv", parse_dates=["Date"]).sort_values("Date")
df = df.reset_index(drop=True)

print("Backend Ready!")


# ============================================================
# 1️⃣ RF CLASSIFICATION
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        tds = clean(data.get("TDS"))
        turb = clean(data.get("Turbidity"))

        X = pd.DataFrame([[tds, turb]], columns=["TDS", "Turbidity"])
        pred_class = classifier_model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        probabilities = classifier_model.predict_proba(X)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

        return jsonify({
            "prediction": pred_label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 2️⃣ LSTM FUTURE FORECAST (TDS + Turbidity)
# ============================================================
@app.route("/predict_future", methods=["POST"])
def predict_future():
    try:
        data = request.get_json()
        steps = int(data.get("steps", 7))

        window = 14
        recent = df[["TDS", "Turbidity"]].tail(window).values
        scaled = lstm_scaler.transform(recent)
        input_seq = np.array([scaled])

        predictions = []

        for _ in range(steps):
            scaled_pred = lstm_model.predict(input_seq, verbose=0)[0]
            real = lstm_scaler.inverse_transform([scaled_pred])[0]

            predictions.append({
                "TDS": float(real[0]),
                "Turbidity": float(real[1])
            })

            input_seq = np.array([np.vstack([input_seq[0][1:], scaled_pred])])

        return jsonify({"future_predictions": predictions}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 3️⃣ LSTM + WHO QUALITY FORECAST
# ============================================================
@app.route("/predict_future_quality", methods=["POST"])
def predict_future_quality():
    try:
        data = request.get_json()
        steps = int(data.get("steps", 7))

        window = 14
        recent = df[["TDS", "Turbidity"]].tail(window).values
        scaled = lstm_scaler.transform(recent)
        input_seq = np.array([scaled])

        results = []

        for _ in range(steps):
            scaled_pred = lstm_model.predict(input_seq, verbose=0)[0]
            real = lstm_scaler.inverse_transform([scaled_pred])[0]

            tds, turb = float(real[0]), float(real[1])

            if tds < 500:
                q = "Safe"
            elif tds < 1000:
                q = "Moderate"
            else:
                q = "Unsafe"

            results.append({
                "TDS": tds,
                "Turbidity": turb,
                "Quality": q
            })

            input_seq = np.array([np.vstack([input_seq[0][1:], scaled_pred])])

        return jsonify({"forecast": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 4️⃣ IOT ESP32 → BACKEND UPDATE
# ============================================================
@app.route("/iot_update", methods=["POST"])
def iot_update():
    try:
        data = request.get_json()

        row = {
            "TDS": clean(data.get("tds")),
            "Turbidity": clean(data.get("turbidity")),
            "Safe": bool(data.get("safe", False)),
            "Timestamp": int(data.get("ts", time.time()))
        }

        json.dump(row, open("latest_iot.json", "w"))

        hist = []
        if os.path.exists("iot_history.json"):
            hist = json.load(open("iot_history.json"))

        hist.append(row)
        json.dump(hist, open("iot_history.json", "w"))

        return jsonify({"msg": "IoT data saved"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 5️⃣ GET LATEST SENSOR VALUE
# ============================================================
@app.route("/iot_latest", methods=["GET"])
def iot_latest():
    try:
        if not os.path.exists("latest_iot.json"):
            return jsonify({"tds": 0, "turbidity": 0, "safe": True}), 200

        raw = json.load(open("latest_iot.json"))

        return jsonify({
            "tds": float(raw["TDS"]),
            "turbidity": float(raw["Turbidity"]),
            "safe": bool(raw["Safe"]),
            "ts": int(raw["Timestamp"])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 6️⃣ FULL HISTORY
# ============================================================
@app.route("/iot_history", methods=["GET"])
def iot_history():
    try:
        if not os.path.exists("iot_history.json"):
            return jsonify([])

        hist = json.load(open("iot_history.json"))
        hist.reverse()
        return jsonify(hist)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 7️⃣ HEALTHCHECK FOR RENDER
# ============================================================
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "OK"}), 200


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
