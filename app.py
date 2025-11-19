from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)   # allow frontend requests


# ============================================================
# LOAD MODELS
# ============================================================
print("🔄 Loading Random Forest model...")
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

print("🔄 Loading Label Encoder...")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("🔄 Loading LSTM model (.keras)...")
lstm_model = load_model("lstm_model.keras")

print("🔄 Loading LSTM scaler...")
lstm_scaler = joblib.load("scaler.pkl")

# Load dataset (needed for last 14 days)
df = pd.read_csv("water_quality_big_dataset.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# HELPERS
# ============================================================
def get_float(data, *keys, default=0.0):
    for key in keys:
        if key in data:
            try:
                return float(data[key])
            except:
                pass
    return float(default)


# ============================================================
# 1) RANDOM FOREST PREDICTION
# ============================================================
@app.route("/predict", methods=["POST"])
def predict_quality():
    try:
        data = request.get_json()

        tds = get_float(data, "TDS", "tds", default=0)
        turb = get_float(data, "Turbidity", "turbidity", default=0)

        X = pd.DataFrame([[tds, turb]], columns=["TDS", "Turbidity"])

        pred_class = rf_model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        confidence = float(max(rf_model.predict_proba(X)[0]) * 100)

        return jsonify({
            "prediction": pred_label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 2) LSTM FUTURE VALUES (TDS + TURBIDITY)
# ============================================================
@app.route("/predict_future", methods=["POST"])
def predict_future():
    try:
        data = request.get_json()
        steps = int(data.get("steps", 7))

        window = 14
        recent = df[["TDS", "Turbidity"]].tail(window).values
        scaled_recent = lstm_scaler.transform(recent)
        seq = np.array([scaled_recent])

        predictions = []

        for _ in range(steps):
            pred_scaled = lstm_model.predict(seq, verbose=0)[0]
            real = lstm_scaler.inverse_transform([pred_scaled])[0]

            predictions.append({
                "TDS": float(real[0]),
                "Turbidity": float(real[1])
            })

            seq = np.array([np.vstack([seq[0][1:], pred_scaled])])

        return jsonify({"future_predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 3) LSTM QUALITY FORECAST (Safe / Moderate / Unsafe)
# ============================================================
@app.route("/predict_future_quality", methods=["POST"])
def predict_future_quality():
    try:
        data = request.get_json()
        steps = int(data.get("steps", 7))

        window = 14
        recent = df[["TDS", "Turbidity"]].tail(window).values
        scaled_recent = lstm_scaler.transform(recent)
        seq = np.array([scaled_recent])

        results = []

        for _ in range(steps):
            pred_scaled = lstm_model.predict(seq, verbose=0)[0]
            real = lstm_scaler.inverse_transform([pred_scaled])[0]

            tds = float(real[0])
            turb = float(real[1])

            # Simple rule-based quality
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

            seq = np.array([np.vstack([seq[0][1:], pred_scaled])])

        return jsonify({"forecast": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# HEALTHCHECK
# ============================================================
@app.route("/healthcheck")
def healthcheck():
    return jsonify({"status": "OK"})


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
