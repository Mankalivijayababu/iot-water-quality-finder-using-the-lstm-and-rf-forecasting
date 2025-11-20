# ------------------------------------------------------------
#  train_model.py  (FINAL FIXED VERSION – 100% RENDER SAFE)
#  ✔ Random Forest Classification
#  ✔ LSTM Time-Series Forecasting
#  ✔ Saves: rf_model.pkl, label_encoder.pkl,
#            lstm_model.h5, scaler.pkl
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv("water_quality_big_dataset.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("✔ Loaded dataset:", len(df), "rows")

# ------------------------------------------------------------
# RANDOM FOREST CLASSIFICATION
# ------------------------------------------------------------
print("\n==============================")
print("TRAINING RANDOM FOREST MODEL")
print("==============================")

label_encoder = LabelEncoder()
df["Quality_encoded"] = label_encoder.fit_transform(df["Quality"])

X = df[["TDS", "Turbidity"]]
y = df["Quality_encoded"]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
except:
    print("⚠ Stratify failed — using all data for training")
    X_train, X_test, y_train, y_test = X, X, y, y

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=3,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

y_pred = rf_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n✔ RANDOM FOREST ACCURACY:", round(acc * 100, 2), "%")
print(classification_report(y_test, y_pred))

# Save RF model + Label Encoder
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✔ Saved: rf_model.pkl, label_encoder.pkl")

# ------------------------------------------------------------
# LSTM TIME-SERIES MODEL
# ------------------------------------------------------------
print("\n==============================")
print("TRAINING LSTM FORECAST MODEL")
print("==============================")

features = ["TDS", "Turbidity"]
data = df[features].values.astype(float)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

window = 14
X_lstm, y_lstm = [], []

for i in range(len(data_scaled) - window):
    X_lstm.append(data_scaled[i:i + window])
    y_lstm.append(data_scaled[i + window])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

split = int(len(X_lstm) * 0.8)
X_train_lstm, X_val_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_val_lstm = y_lstm[:split], y_lstm[split:]

# ------------------ FIXED LSTM MODEL ------------------------
# IMPORTANT: No deprecated metrics like "mse" or "keras.metrics.mse"
# This ensures Render can load lstm_model.h5 safely.
# ------------------------------------------------------------
model = Sequential([
    Input(shape=(window, len(features))),
    LSTM(64, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(len(features))  # output: [future_TDS, future_Turbidity]
])

model.compile(
    optimizer=Adam(0.001),
    loss="mse",
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
)

model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_val_lstm, y_val_lstm),
    epochs=40,
    batch_size=8,
    verbose=1
)

# Save LSTM model (Render safe)
model.save("lstm_model.h5", include_optimizer=False)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("✔ Saved: lstm_model.h5, scaler.pkl")

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")
