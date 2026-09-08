"""Step 12 - Flask API for the Step 6 XGBoost treatment-outcome model.

Run:
    gsk_env/Scripts/python.exe app.py

Test from another terminal:
    gsk_env/Scripts/python.exe step12_api_client.py
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = os.path.join("outputs", "xgboost_best_model.pkl")
MODEL_LABEL = "Step 6 XGBoost"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Run Step 6 first to create {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]
FEATURE_COLS = preprocessor.feature_names_in_.tolist()

if not hasattr(preprocessor, "transform") or not hasattr(classifier, "predict_proba"):
    raise TypeError("The saved model is not the fitted Step 6 preprocessing/classifier pipeline.")

app = Flask(__name__)


def validate_request(data):
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    missing = [column for column in FEATURE_COLS if column not in data]
    if missing:
        return None, f"Missing fields: {missing}"

    unknown = sorted(set(data).difference(FEATURE_COLS))
    if unknown:
        return None, f"Unknown fields: {unknown}. Send only model input fields."

    categorical_columns = set(preprocessor.transformers_[1][2])
    normalized_data = {
        column: (
            str(data[column]).lower()
            if column in categorical_columns and isinstance(data[column], bool)
            else data[column]
        )
        for column in FEATURE_COLS
    }
    return pd.DataFrame([normalized_data], columns=FEATURE_COLS), None


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    features, error = validate_request(data)
    if error:
        return jsonify({"error": error, "required_fields": FEATURE_COLS}), 400

    try:
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0, 1])
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid feature value: {exc}"}), 400

    distance_from_threshold = abs(probability - 0.5)
    confidence = (
        "High" if distance_from_threshold > 0.30 else
        "Medium" if distance_from_threshold > 0.15 else
        "Low"
    )
    return jsonify(
        {
            "prediction": "Effective" if prediction else "Ineffective",
            "probability_effective": round(probability, 4),
            "confidence": confidence,
            "threshold": 0.5,
            "model": MODEL_LABEL,
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "model": MODEL_LABEL,
            "feature_count": len(FEATURE_COLS),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
