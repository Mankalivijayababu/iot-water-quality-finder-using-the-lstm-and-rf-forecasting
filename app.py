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
#  JSON PATHS (Local = project, Render = /tmp)
# =====================================================
BASE_DIR = "." if platform.system() == "Windows" else "/tmp"
LATEST_FILE = os.path.join(BASE_DIR, "iot_latest.json")
HISTORY_FILE = os.path.join(BASE_DIR, "iot_history.json")


# =====================================================
#  JSON HELPERS
# =====================================================
def load_json(path, default):
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
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[JSON SAVE ERROR] {path}: {e}")


# =====================================================
#  LAZY LOADED MODELS
# =====================================================
rf_model = None
label_encoder = None
lstm_model = None
scaler = None


def load_models():
    """Load ML models once."""
    global rf_model, label_encoder, lstm_model, scaler

    if all([rf_model, label_encoder, lstm_model, scaler]):
        return

    print("⚙️ Loading ML models...")

    rf_model = joblib.load("rf_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    lstm_model = load_model("lstm_model.keras")  # ONLY KERAS

    # Pre-warm RF
    _ = rf_model.predict([[500, 3]])
    _ = rf_model.predict_proba([[500, 3]])
    print("✅ ML Models Loaded & Warmed")


# =====================================================
#  LOAD IoT DATA
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
#  WARMUP ENDPOINT
# =====================================================
@app.route("/warmup", methods=["GET"])
def warmup():
    try:
        load_models()
        return jsonify({"status": "warm", "message": "Models loaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
#  HOME ROUTE
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "running",
            "message": "Water Quality API (RF + LSTM) 🚀",
            "routes": {
                "/warmup": "GET → Preload models",
                "/predict": "POST → RF Quality Prediction",
                "/predict_future_quality": "POST → 7-day LSTM forecast",
                "/iot_latest": "GET → Last sensor value",
                "/iot_history": "GET → All history",
                "/add_history": "POST → Add sensor record",
                "/search_history": "GET → Filter by city",
            },
        }
    )


# =====================================================
#  RF QUALITY PREDICTION
# =====================================================
@app.route("/predict", methods=["POST"])
def predict_quality():
    try:
        load_models()
        data = request.get_json()
        tds = float(data["tds"])
        turbidity = float(data["turbidity"])

        idx = rf_model.predict([[tds, turbidity]])[0]
        label = label_encoder.inverse_transform([idx])[0]
        confidence = rf_model.predict_proba([[tds, turbidity]]).max() * 100

        return jsonify(
            {"prediction": label, "confidence": round(float(confidence), 2)}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
#  LSTM FORECAST (FIXED PADDING)
# =====================================================
@app.route("/predict_future", methods=["POST"])
@app.route("/predict_future_quality", methods=["POST"])
def predict_future_quality():
    try:
        load_models()
        data = request.get_json()
        steps = int(data.get("steps", 7))

        history = []

        # Use up to last 14 entries
        for h in reversed(iot_history[-14:]):
            history.append([float(h["tds"]), float(h["turbidity"])])

        # If history empty, repeat latest
        if len(history) == 0:
            history = [[iot_latest["tds"], iot_latest["turbidity"]] for _ in range(14)]

        # Pad if < 14
        while len(history) < 14:
            history.insert(0, history[0])

        history = np.array(history, dtype=float)
        scaled = scaler.transform(history).reshape(1, 14, 2)

        predictions = []

        for _ in range(steps):
            pred_scaled = lstm_model.predict(scaled)[0]
            pred_real = scaler.inverse_transform([pred_scaled])[0]

            tds_pred, turb_pred = map(float, pred_real)

            if tds_pred > 900 or turb_pred > 5:
                q = "Unsafe"
            elif tds_pred > 600 or turb_pred > 3:
                q = "Moderate"
            else:
                q = "Safe"

            predictions.append(
                {"TDS": tds_pred, "Turbidity": turb_pred, "Quality": q}
            )

            scaled = np.roll(scaled, -1, axis=1)
            scaled[0, -1, :] = scaler.transform([[tds_pred, turb_pred]])[0]

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
#  IoT DATA ROUTES
# =====================================================
@app.route("/iot_latest", methods=["GET"])
def get_latest():
    return jsonify(iot_latest)


@app.route("/iot_history", methods=["GET"])
def get_history():
    return jsonify(iot_history)


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
        return jsonify({"error": str(e)}), 500


@app.route("/search_history", methods=["GET"])
def search_history():
    try:
        city = request.args.get("city", "").lower().strip()
        results = [h for h in iot_history if h.get("city", "").lower() == city]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
#  RUN LOCAL
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
