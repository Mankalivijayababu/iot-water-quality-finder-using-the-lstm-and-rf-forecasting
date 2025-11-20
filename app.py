from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import time
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------
# Root Route (Homepage)
# -------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Water Quality API (RF + LSTM) is live 🚀",
        "routes": {
            "/predict": "POST → Predict water quality using Random Forest",
            "/predict_future_quality": "POST → Predict future water quality using LSTM",
            "/iot_latest": "GET → Latest IoT reading",
            "/iot_history": "GET → IoT history",
            "/add_history": "POST → Add history entry",
            "/search_history": "GET → Search history by city"
        }
    })


# -------------------------------------------------------
# Load ML Models
# -------------------------------------------------------
print("🔄 Loading Random Forest model...")
rf_model = joblib.load("rf_model.pkl")

print("🔄 Loading Label Encoder...")
label_encoder = joblib.load("label_encoder.pkl")

print("🔄 Loading LSTM model (.h5)...")
lstm_model = load_model("lstm_model.h5")

print("🔄 Loading LSTM scaler...")
scaler = joblib.load("scaler.pkl")

# -------------------------------------------------------
# Dummy IoT Data (In-memory)
# -------------------------------------------------------
iot_data = {
    "latest": {
        "timestamp": time.time(),
        "tds": 500,
        "turbidity": 3.0
    },
    "history": []
}


# -------------------------------------------------------
# 1️⃣ Random Forest → Predict Quality
# -------------------------------------------------------
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


# -------------------------------------------------------
# 2️⃣ LSTM → Future Prediction
# -------------------------------------------------------
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


# -------------------------------------------------------
# 3️⃣ Get Latest IoT Reading
# -------------------------------------------------------
@app.route("/iot_latest", methods=["GET"])
def get_latest():
    return jsonify(iot_data["latest"])


# -------------------------------------------------------
# 4️⃣ Get Complete History
# -------------------------------------------------------
@app.route("/iot_history", methods=["GET"])
def get_history():
    return jsonify(iot_data["history"])


# -------------------------------------------------------
# 5️⃣ Save History Entry
# -------------------------------------------------------
@app.route("/add_history", methods=["POST"])
def add_history():
    try:
        data = request.get_json()

        entry = {
            "timestamp": time.time(),
            "tds": float(data["tds"]),
            "turbidity": float(data["turbidity"]),
            "city": data.get("city", "Unknown"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
        }

        iot_data["history"].append(entry)

        return jsonify({"status": "saved", "entry": entry})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------
# 6️⃣ Search History by City
# -------------------------------------------------------
@app.route("/search_history", methods=["GET"])
def search_history():
    try:
        city = request.args.get("city", "").lower()

        results = [
            h for h in iot_data["history"]
            if h.get("city", "").lower() == city
        ]

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------
# Start Server
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
