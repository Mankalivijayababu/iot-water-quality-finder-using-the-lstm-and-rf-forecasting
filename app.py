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
# JSON FILE PATHS (Render: only /tmp is writable)
# ---------------------------------------
LATEST_FILE = "/tmp/iot_latest.json"
HISTORY_FILE = "/tmp/iot_history.json"


# ---------------------------------------
# JSON HELPERS
# ---------------------------------------
def load_json(path, default):
  """
  Safely load JSON from a file, or create it with default if missing.
  """
  if not os.path.exists(path):
    with open(path, "w") as f:
      json.dump(default, f, indent=4)
    return default

  try:
    with open(path, "r") as f:
      return json.load(f)
  except Exception:
    return default


def save_json(path, data):
  """
  Safely save JSON to file.
  """
  with open(path, "w") as f:
    json.dump(data, f, indent=4)


# ---------------------------------------
# ML MODELS (LAZY LOADED)
# ---------------------------------------
rf_model = None
label_encoder = None
lstm_model = None
scaler = None


def load_models():
  """
  Lazy-load all ML models the first time they are needed.
  Keeps /iot_latest and /iot_history endpoints fast.
  """
  global rf_model, label_encoder, lstm_model, scaler

  if rf_model is not None and label_encoder is not None and lstm_model is not None and scaler is not None:
    # Already loaded
    return

  print("⚙️ Loading ML models into memory...")

  rf_model = joblib.load("rf_model.pkl")
  label_encoder = joblib.load("label_encoder.pkl")
  lstm_model = load_model("lstm_model.h5")
  scaler = joblib.load("scaler.pkl")

  print("✅ ML models loaded.")


# ---------------------------------------
# Load IoT data from JSON (persistent while container is alive)
# ---------------------------------------
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


# ---------------------------------------
# Root Route
# ---------------------------------------
@app.route("/", methods=["GET"])
def home():
  return jsonify(
    {
      "status": "running",
      "message": "Water Quality API (RF + LSTM) is live 🚀",
      "routes": {
        "/predict": "POST → Predict using Random Forest",
        "/predict_future_quality": "POST → Predict future with LSTM",
        "/iot_latest": "GET → Latest sensor data",
        "/iot_history": "GET → Full IoT history",
        "/add_history": "POST → Add new entry",
        "/search_history": "GET → Search history by city",
      },
    }
  )


# ---------------------------------------
# Predict Quality (Random Forest)
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict_quality():
  try:
    load_models()  # lazy load

    data = request.get_json()
    tds = float(data["tds"])
    turbidity = float(data["turbidity"])

    pred_idx = rf_model.predict([[tds, turbidity]])[0]
    label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = np.max(rf_model.predict_proba([[tds, turbidity]]) * 100)

    return jsonify(
      {
        "prediction": label,
        "confidence": round(float(confidence), 2),
      }
    )

  except Exception as e:
    return jsonify({"error": str(e)}), 500


# ---------------------------------------
# LSTM Future Prediction
# ---------------------------------------
@app.route("/predict_future_quality", methods=["POST"])
def predict_future_quality():
  try:
    load_models()  # lazy load

    data = request.get_json()
    steps = int(data.get("steps", 7))

    # For now; you can later use last real readings
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

      future_predictions.append(
        {
          "TDS": tds_pred,
          "Turbidity": turb_pred,
          "Quality": quality,
        }
      )

      # feed back into sequence
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
# Add New IoT Entry (ESP32 + App)
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

    # Append to full history
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
      h
      for h in iot_history
      if h.get("city", "").lower() == city
    ]

    return jsonify(results)

  except Exception as e:
    return jsonify({"error": str(e)}), 500


# ---------------------------------------
# Local run
# ---------------------------------------
if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
