from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import math
import os
import time
import json
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf

app = Flask(__name__)

# ============================================================
# ✅ CLEAN FUNCTION
# ============================================================
def clean(value):
    try:
        if value is None:
            return 0
        if isinstance(value, float) and math.isnan(value):
            return 0
        return value
    except:
        return 0


# ============================================================
# ✅ LOAD ML MODELS
# ============================================================
with open("water_quality_model.pkl", "rb") as f:
    classifier_model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

lstm_model = load_model("lstm_model.h5")
lstm_scaler = joblib.load("scaler.pkl")

# ✅ Load dataset only for last 3 rows (LSTM seed)
df = pd.read_csv("water_quality_dataset.csv").replace({np.nan: 0})


# ============================================================
# ✅ AI PREDICT (CLASSIFICATION)
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
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ OPTIMIZED LSTM FUTURE FORECAST
# ============================================================
@app.route("/predict_future", methods=["POST"])
def predict_future():
    try:
        data = request.get_json()
        steps = int(data.get("steps", 5))

        tf.config.run_functions_eagerly(False)

        window = 3
        recent = df[["TDS", "Turbidity"]].tail(window).values
        scaled = lstm_scaler.transform(recent)

        input_seq = np.array([scaled])
        predictions = []

        for _ in range(steps):
            scaled_pred = lstm_model(input_seq, training=False).numpy()[0]
            real = lstm_scaler.inverse_transform([scaled_pred])[0]

            predictions.append({
                "TDS": float(real[0]),
                "Turbidity": float(real[1])
            })

            # Move sliding window
            input_seq = np.array([np.vstack([input_seq[0][1:], scaled_pred])])

        return jsonify({"future_predictions": predictions}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ IOT UPDATE — ESP32 sends data
# ============================================================
@app.route("/iot_update", methods=["POST"])
def iot_update():
    try:
        data = request.get_json()

        row = {
            "TDS": clean(data.get("tds")),
            "Turbidity": clean(data.get("turbidity")),
            "Safe": data.get("safe", False),
            "Timestamp": int(data.get("ts", time.time()))
        }

        with open("latest_iot.json", "w") as f:
            json.dump(row, f)

        if not os.path.exists("iot_history.json"):
            with open("iot_history.json", "w") as f:
                json.dump([], f)

        with open("iot_history.json", "r") as f:
            hist = json.load(f)

        hist.append(row)

        with open("iot_history.json", "w") as f:
            json.dump(hist, f)

        return jsonify({"msg": "IoT data received OK"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ IOT LATEST
# ============================================================
@app.route("/iot_latest", methods=["GET"])
def iot_latest():
    try:
        if not os.path.exists("latest_iot.json"):
            return jsonify({"error": "No IoT data yet"}), 404

        with open("latest_iot.json", "r") as f:
            return jsonify(json.load(f)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ IOT HISTORY
# ============================================================
@app.route("/iot_history", methods=["GET"])
def iot_history():
    try:
        if not os.path.exists("iot_history.json"):
            return jsonify([]), 200

        with open("iot_history.json", "r") as f:
            return jsonify(json.load(f)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ START SERVER
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
