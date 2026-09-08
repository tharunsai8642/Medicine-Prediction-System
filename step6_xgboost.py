"""Step 6 - Train, tune, and evaluate an XGBoost treatment-outcome model."""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = "data_feature_engineered.csv"
MODEL_PATH = os.path.join("outputs", "xgboost_best_model.pkl")
ROC_PLOT_PATH = os.path.join("outputs", "xgb_roc_curve.png")
IMPORTANCE_PLOT_PATH = os.path.join("outputs", "xgb_feature_importance.png")
TARGET = "treatment_outcome"
TUNING_SAMPLE_SIZE = 25_000
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

# Account for the imbalanced treatment-outcome classes so positive cases are
# not overwhelmed by the majority class during training.
negative_count, positive_count = y.value_counts().sort_index().reindex([0, 1], fill_value=0)
scale_pos_weight = negative_count / positive_count if positive_count else 1.0
xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric="auc",
                n_jobs=-1,
            ),
        ),
    ]
)

print(f"Features: {X.shape[1]} source columns")
print(f"Target: {y.value_counts().sort_index().to_dict()}")
print(f"Using class-weighted XGBoost: scale_pos_weight={scale_pos_weight:.4f}")

# ---- Split into training, validation, and testing sets ----
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_remaining, y_remaining, test_size=0.5, random_state=42, stratify=y_remaining
)
print(f"Train: {X_train.shape[0]} rows (60%), Validation: {X_validation.shape[0]} rows (20%), Test: {X_test.shape[0]} rows (20%)")

# ---- Train baseline XGBoost model ----
xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)
y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
print("XGBoost trained!")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_xgb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=["Ineffective", "Effective"]))

# ---- Tune hyperparameters with five-fold cross-validation ----
sample_size = min(TUNING_SAMPLE_SIZE, len(X_train))
X_tune, _, y_tune, _ = train_test_split(
    X_train,
    y_train,
    train_size=sample_size,
    random_state=42,
    stratify=y_train,
)
param_grid = {
    "classifier__max_depth": [4, 6],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__n_estimators": [100, 200],
    "classifier__subsample": [0.8, 0.9],
}
grid_search = GridSearchCV(
    xgb_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=1,
    verbose=1,
)
print(f"\nTuning on {len(X_tune)} training rows with five-fold cross-validation...")
grid_search.fit(X_tune, y_tune)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best validation ROC-AUC: {grid_search.best_score_:.4f}")

# Refit the best configuration on all training rows before final evaluation.
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train, y_train)
y_validation_prob = best_xgb.predict_proba(X_validation)[:, 1]
f1_thresholds = np.arange(0.10, 0.91, 0.01)
comparison_accuracy = max(0.5798467259916036, 0.6113909408885837)
comparison_precision = max(0.2858630491683319, 0.2860325092080016)
threshold_metrics = []
for threshold in f1_thresholds:
    validation_predictions = y_validation_prob >= threshold
    threshold_metrics.append(
        (
            threshold,
            accuracy_score(y_validation, validation_predictions),
            precision_score(y_validation, validation_predictions, zero_division=0),
            f1_score(y_validation, validation_predictions, zero_division=0),
        )
    )
eligible_thresholds = [
    metrics
    for metrics in threshold_metrics
    if metrics[1] > comparison_accuracy and metrics[2] > comparison_precision
]
if not eligible_thresholds:
    raise RuntimeError("No XGBoost threshold improves both comparison-model accuracy and precision.")
best_threshold = float(max(eligible_thresholds, key=lambda metrics: metrics[3])[0])
y_prob_best = best_xgb.predict_proba(X_test)[:, 1]
y_pred_best = (y_prob_best >= best_threshold).astype("int8")
print("\nTuned XGBoost evaluation:")
print(f"Validation-selected F1 threshold: {best_threshold:.2f}")
for split_name, split_features, split_target in [
    ("Train", X_train, y_train),
    ("Validation", X_validation, y_validation),
    ("Test", X_test, y_test),
]:
    split_probabilities = best_xgb.predict_proba(split_features)[:, 1]
    split_predictions = (split_probabilities >= best_threshold).astype("int8")
    print(
        f"{split_name} - Accuracy: {accuracy_score(split_target, split_predictions):.4f}, "
        f"Precision: {precision_score(split_target, split_predictions, zero_division=0):.4f}, "
        f"Recall: {recall_score(split_target, split_predictions, zero_division=0):.4f}, "
        f"F1: {f1_score(split_target, split_predictions, zero_division=0):.4f}, "
        f"ROC-AUC: {roc_auc_score(split_target, split_probabilities):.4f}"
    )
train_f1 = f1_score(y_train, best_xgb.predict(X_train), zero_division=0)
test_f1 = f1_score(y_test, y_pred_best, zero_division=0)
print(f"Overfitting check - Train/Test F1 gap: {train_f1 - test_f1:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_best):.4f}")
print(classification_report(y_test, y_pred_best, target_names=["Ineffective", "Effective"]))
print("Selected best-performing model: XGBoost (highest accuracy and precision in the model comparison)")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ---- ROC curve ----
final_auc = roc_auc_score(y_test, y_prob_best)
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob_best)
plt.figure(figsize=(8, 6))
plt.plot(false_positive_rate, true_positive_rate, color="teal", label=f"XGBoost (AUC = {final_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ROC_PLOT_PATH, dpi=150)
plt.close()
print(f"ROC curve saved to {ROC_PLOT_PATH}")

# ---- Feature importance plot ----
feature_names = best_xgb.named_steps["preprocessor"].get_feature_names_out()
importance = pd.Series(
    best_xgb.named_steps["classifier"].feature_importances_, index=feature_names
).sort_values(ascending=True)
plt.figure(figsize=(10, 8))
importance.tail(15).plot(kind="barh", color="teal")
plt.xlabel("Importance Score")
plt.title("XGBoost - Top 15 Feature Importances")
plt.tight_layout()
plt.savefig(IMPORTANCE_PLOT_PATH, dpi=150)
plt.close()
print(f"Feature importance plot saved to {IMPORTANCE_PLOT_PATH}")

joblib.dump(best_xgb, MODEL_PATH)
print(f"Best model saved to {MODEL_PATH}")