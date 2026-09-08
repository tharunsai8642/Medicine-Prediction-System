"""Step 5 - Train and evaluate a random-forest treatment-outcome model."""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data_feature_engineered.csv"
MODEL_PATH = os.path.join("outputs", "random_forest_model.pkl")
PLOT_PATH = os.path.join("outputs", "rf_feature_importance.png")
TARGET = "treatment_outcome"
TARGET_MAP = {
    "0": 0,
    "1": 1,
    "no": 0,
    "ineffective": 0,
    "yes": 1,
    "effective": 1,
}


# ---- Load data and standardize the target ----
df = pd.read_csv(DATA_PATH, low_memory=False)
target_values = df[TARGET].astype(str).str.strip().str.lower().map(TARGET_MAP)
valid_target = target_values.notna()
df = df.loc[valid_target].copy()
y = target_values.loc[valid_target].astype("int8")

# IDs and other outcomes are not predictors of treatment outcome.
drop_columns = [TARGET, "patient_id", "patient_id.1", "adverse_event", "readmission_30d"]
X = df.drop(columns=[column for column in drop_columns if column in df.columns])

if "admission_date" in X.columns:
    admission_dates = pd.to_datetime(X.pop("admission_date"), errors="coerce")
    X["admission_year"] = admission_dates.dt.year.fillna(0).astype("int16")
    X["admission_month"] = admission_dates.dt.month.fillna(0).astype("int8")

numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=15,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print(f"Features: {X.shape[1]} source columns")
print(f"Target: {y.value_counts().sort_index().to_dict()}")

# ---- Split into training, validation, and testing sets ----
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_remaining, y_remaining, test_size=0.5, random_state=42, stratify=y_remaining
)
print(f"Train: {X_train.shape[0]} rows (60%), Validation: {X_validation.shape[0]} rows (20%), Test: {X_test.shape[0]} rows (20%)")

# ---- Train Random Forest ----
rf_model.fit(X_train, y_train)
print("Random Forest trained with 200 trees!")

# ---- Predict and evaluate ----
for split_name, split_features, split_target in [
    ("Train", X_train, y_train),
    ("Validation", X_validation, y_validation),
    ("Test", X_test, y_test),
]:
    split_predictions = rf_model.predict(split_features)
    split_probabilities = rf_model.predict_proba(split_features)[:, 1]
    print(
        f"{split_name} - Accuracy: {accuracy_score(split_target, split_predictions):.4f}, "
        f"Precision: {precision_score(split_target, split_predictions, zero_division=0):.4f}, "
        f"Recall: {recall_score(split_target, split_predictions, zero_division=0):.4f}, "
        f"F1: {f1_score(split_target, split_predictions, zero_division=0):.4f}, "
        f"ROC-AUC: {roc_auc_score(split_target, split_probabilities):.4f}"
    )

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
train_f1 = f1_score(y_train, rf_model.predict(X_train), zero_division=0)
test_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
print(f"Overfitting check - Train/Test F1 gap: {train_f1 - test_f1:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Ineffective", "Effective"]))

# ---- Feature importance plot ----
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
importance_rf = pd.Series(
    rf_model.named_steps["classifier"].feature_importances_, index=feature_names
).sort_values(ascending=True)
plt.figure(figsize=(10, 8))
importance_rf.tail(15).plot(kind="barh", color="teal")
plt.title("Random Forest - Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
plt.savefig(PLOT_PATH, dpi=150)
plt.close()
print(f"Feature importance plot saved to {PLOT_PATH}")

# ---- Save model and preprocessing together ----
joblib.dump(rf_model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")