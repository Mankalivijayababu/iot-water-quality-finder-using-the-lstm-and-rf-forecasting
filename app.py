from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import time
import json
import os
import platform
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# =====================================================
#  JSON FILE PATHS (Local = project folder, Render = /tmp)
# =====================================================
if platform.system() == "Windows":
    BASE_DIR = "."
else:
    BASE_DIR = "/tmp"

LATEST_FILE = f"{BASE_DIR}/iot_latest.json"
HISTORY_FILE = f"{BASE_DIR}/iot_history.json"


# =====================================================
#  JSON HELPERS
# =====================================================
def load_json(path, default):
    """Safely load/create JSON"""
    try:
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(default, f, indent=4)
            return default

        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[JSON LOAD ERROR] {path}: {e}")
        return default


def save_json(path, data):
    """Safely save JSON"""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[JSON SAVE ERROR] {path}: {e}")


# =====================================================
#  LAZY LOADED ML MODELS (RF + LSTM)
# =====================================================
rf_model = None
label_encoder = None
lstm_model = None
scaler = None


def load_models():
    """Load ML models only when needed"""
    global rf_model, label_encoder, lstm_model, scaler

    if all([rf_model, label_encoder, lstm_model, scaler]):
        return

    print("⚙️ Loading ML models...")

    rf_model = joblib.load("rf_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    lstm_model = load_model("lstm_model.h5")
    scaler = joblib.load("scaler.pkl")

    # ⚡ Warm-up RF model (fixes slow first response)
    _ = rf_model.predict([[500, 3]])
    _ = rf_model.predict_proba([[500, 3]])

    print("✅ ML Models Loaded & Pre-Warmed")


# =====================================================
#  LOAD SAVED IoT DATA
# =====================================================
iot_latest = load_json(
    LATEST_FILE,
    {
        "timestamp": time.time(),
        "tds": 500,
        "turbidity": 3.0,
        "city": "Unknown",
        "latitude": None,
        "longitude": None,
    },
)

iot_history = load_json(HISTORY_FILE, [])


# =====================================================
#  WARMUP ENDPOINT (Flutter will call once)
# =====================================================
@app.route("/warmup", methods=["GET"])
def warmup():
    try:
        load_models()
        return jsonify({"status": "warm", "message": "Models loaded"})
    except Exception as e:
        print(f"[WARMUP ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# =====================================================
#  HOME ROUTE
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Water Quality API (RF + LSTM) is live 🚀",
        "routes": {
            "/warmup": "GET → Preload ML models",
            "/predict": "POST → Predict quality using RF",
            "/predict_future_quality": "POST → LSTM 7-day prediction",
            "/iot_latest": "GET → Latest IoT values",
            "/iot_history": "GET → IoT history",
            "/add_history": "POST → Add new sensor entry",
            "/search_history": "GET → Search by city",
        }
    })


# =====================================================
#  RANDOM FOREST QUALITY PREDICTION
# =====================================================
@app.route("/predict", methods=["POST"])
def predict_quality():
    try:
        load_models()

        data = request.get_json()
        tds = float(data["tds"])
        turbidity = float(data["turbidity"])

        pred_idx = rf_model.predict([[tds, turbidity]])[0]
        label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = np.max(rf_model.predict_proba([[tds, turbidity]]) * 100)

        return jsonify({
            "prediction": label,
            "confidence": round(float(confidence), 2)
        })

    except Exception as e:
        print(f"[PREDICT ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# =====================================================
#  LSTM FUTURE FORECAST
# =====================================================
@app.route("/predict_future_quality", methods=["POST"])
def predict_future_quality():
    try:
        load_models()

        data = request.get_json()
        steps = int(data.get("steps", 7))

        last_tds = float(iot_latest.get("tds", 500))
        last_turb = float(iot_latest.get("turbidity", 3.0))

        last_known = np.array([[last_tds, last_turb]], dtype=float)
        scaled = scaler.transform(last_known).reshape(1, 1, 2)

        predictions = []

        for _ in range(steps):
            pred = lstm_model.predict(scaled)[0]
            inv = scaler.inverse_transform([pred])[0]

            tds_pred = float(inv[0])
            turb_pred = float(inv[1])

            # Quality rules
            if tds_pred > 900 or turb_pred > 5:
                quality = "Unsafe"
            elif tds_pred > 600 or turb_pred > 3:
                quality = "Moderate"
            else:
                quality = "Safe"

            predictions.append({
                "TDS": tds_pred,
                "Turbidity": turb_pred,
                "Quality": quality
            })

            scaled = pred.reshape(1, 1, 2)

        return jsonify(predictions)

    except Exception as e:
        print(f"[LSTM ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# =====================================================
#  GET LATEST IoT DATA
# =====================================================
@app.route("/iot_latest", methods=["GET"])
def get_latest():
    return jsonify(iot_latest)


# =====================================================
#  GET IoT HISTORY
# =====================================================
@app.route("/iot_history", methods=["GET"])
def get_history():
    return jsonify(iot_history)


# =====================================================
#  ADD IoT ENTRY
# =====================================================
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

        iot_latest = entry
        save_json(LATEST_FILE, iot_latest)

        iot_history.append(entry)
        save_json(HISTORY_FILE, iot_history)

        return jsonify({"status": "saved", "entry": entry})

    except Exception as e:
        print(f"[ADD_HISTORY ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# =====================================================
#  SEARCH HISTORY BY CITY
# =====================================================
@app.route("/search_history", methods=["GET"])
def search_history():
    try:
        city = request.args.get("city", "").lower().strip()
        results = [h for h in iot_history if h.get("city", "").lower() == city]
        return jsonify(results)

    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# =====================================================
#  RUN LOCAL
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
