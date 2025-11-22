import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load your dataset (keep same column names)
df = pd.read_csv("water_quality_big_dataset.csv")

# Input Features and Target label
X = df[["TDS", "Turbidity"]]
y = df["Quality"]

# Encode labels (Safe, Moderate, Unsafe)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train RF Model
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    bootstrap=True
)
rf.fit(X, y_encoded)

# Save model + encoder
joblib.dump(rf, "rf_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("🎉 RF model and Label Encoder saved successfully!")
