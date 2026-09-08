"""Step 12 - Send a real project row to the Flask prediction API."""

import joblib
import numpy as np
import pandas as pd
import requests


MODEL_PATH = "outputs/xgboost_best_model.pkl"
model = joblib.load(MODEL_PATH)
feature_columns = model.named_steps["preprocessor"].feature_names_in_.tolist()
row = pd.read_csv("data_feature_engineered.csv", nrows=1, low_memory=False).iloc[0]
patient_data = {
	column: (
		None
		if pd.isna(row[column])
		else str(row[column]).lower()
		if isinstance(row[column], (bool, np.bool_))
		else row[column].item()
		if isinstance(row[column], np.generic)
		else row[column]
	)
	for column in feature_columns
}

response = requests.post("http://127.0.0.1:5000/predict", json=patient_data, timeout=30)
print(f"HTTP {response.status_code}")
print(response.json())
