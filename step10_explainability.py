"""Step 10 - Explain the saved XGBoost model with SHAP and LIME."""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split


DATA_PATH = "data_feature_engineered.csv"
MODEL_PATH = os.path.join("outputs", "xgboost_best_model.pkl")
SHAP_SUMMARY_PATH = os.path.join("outputs", "shap_summary.png")
SHAP_FORCE_PATH = os.path.join("outputs", "shap_force_patient0.png")
LIME_PATH = os.path.join("outputs", "lime_explanation_patient0.png")
SHAP_SAMPLE_SIZE = 500
LIME_BACKGROUND_SIZE = 500
RANDOM_STATE = 42
STEP6_REMAINING_SIZE = 0.4
STEP6_TEST_SIZE_WITHIN_REMAINING = 0.5
TARGET = "treatment_outcome"
TARGET_MAP = {
    "0": 0,
    "1": 1,
    "no": 0,
    "ineffective": 0,
    "yes": 1,
    "effective": 1,
}


def prepare_features():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    target_values = df[TARGET].astype(str).str.strip().str.lower().map(TARGET_MAP)
    valid_target = target_values.notna()
    df = df.loc[valid_target].copy()
    y = target_values.loc[valid_target].astype("int8")

    drop_columns = [TARGET, "patient_id", "patient_id.1", "adverse_event", "readmission_30d"]
    X = df.drop(columns=[column for column in drop_columns if column in df.columns])
    if "admission_date" in X.columns:
        admission_dates = pd.to_datetime(X.pop("admission_date"), errors="coerce")
        X["admission_year"] = admission_dates.dt.year.fillna(0).astype("int16")
        X["admission_month"] = admission_dates.dt.month.fillna(0).astype("int8")

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=STEP6_REMAINING_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    _, X_test, _, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=STEP6_TEST_SIZE_WITHIN_REMAINING,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )
    return X_train, X_test, y_test


X_train, X_test, y_test = prepare_features()
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Run Step 6 first to create the tuned model: {MODEL_PATH}")
model_pipeline = joblib.load(MODEL_PATH)
preprocessor = model_pipeline.named_steps["preprocessor"]
classifier = model_pipeline.named_steps["classifier"]
if list(X_test.columns) != list(preprocessor.feature_names_in_):
    raise ValueError("Step 10 features do not match the fitted Step 6 preprocessor schema.")
print(f"Loaded fitted Step 6 XGBoost model: {MODEL_PATH}")
print(f"Using matching Step 6 test split: {len(X_test)} rows")

X_shap = X_test.sample(n=min(SHAP_SAMPLE_SIZE, len(X_test)), random_state=RANDOM_STATE)
X_shap_transformed = preprocessor.transform(X_shap)
if hasattr(X_shap_transformed, "toarray"):
    X_shap_transformed = X_shap_transformed.toarray()
X_shap_transformed = np.asarray(X_shap_transformed, dtype="float32")
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_shap_transformed)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap_values = np.asarray(shap_values)

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_shap_transformed,
    feature_names=feature_names,
    max_display=20,
    show=False,
)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig(SHAP_SUMMARY_PATH, dpi=150, bbox_inches="tight")
plt.close()

patient_index = 0
expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = np.asarray(expected_value).reshape(-1)[-1]
shap.force_plot(
    expected_value,
    shap_values[patient_index],
    X_shap_transformed[patient_index],
    feature_names=feature_names,
    matplotlib=True,
    show=False,
)
plt.tight_layout()
plt.savefig(SHAP_FORCE_PATH, dpi=150, bbox_inches="tight")
plt.close()

X_lime_background = X_train.sample(n=min(LIME_BACKGROUND_SIZE, len(X_train)), random_state=RANDOM_STATE)
X_lime_transformed = preprocessor.transform(X_lime_background)
if hasattr(X_lime_transformed, "toarray"):
    X_lime_transformed = X_lime_transformed.toarray()
X_lime_transformed = np.asarray(X_lime_transformed, dtype="float32")

lime_explainer = LimeTabularExplainer(
    X_lime_transformed,
    feature_names=feature_names.tolist(),
    class_names=["Ineffective", "Effective"],
    mode="classification",
    random_state=RANDOM_STATE,
)
lime_explanation = lime_explainer.explain_instance(
    X_shap_transformed[patient_index],
    classifier.predict_proba,
    num_features=10,
)
print("LIME explanation for Patient 0:")
for feature, weight in lime_explanation.as_list():
    direction = "Effective" if weight > 0 else "Ineffective"
    print(f"  {feature}: {weight:+.4f} -> {direction}")

lime_figure = lime_explanation.as_pyplot_figure()
lime_figure.tight_layout()
lime_figure.savefig(LIME_PATH, dpi=150, bbox_inches="tight")
plt.close(lime_figure)

patient_probability = classifier.predict_proba(X_shap_transformed[[patient_index]])[0, 1]
print(f"Explained test rows: {len(X_shap)}")
print(f"Patient 0 predicted probability: {patient_probability:.4f}")
print(f"Patient 0 actual outcome: {int(y_test.loc[X_shap.index[patient_index]])}")
print(f"SHAP summary saved to {SHAP_SUMMARY_PATH}")
print(f"SHAP force plot saved to {SHAP_FORCE_PATH}")
print(f"LIME explanation saved to {LIME_PATH}")
