"""Step 7 - Train and evaluate a neural-network treatment-outcome model."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from keras import callbacks, layers


DATA_PATH = "data_feature_engineered.csv"
MODEL_PATH = os.path.join("outputs", "neural_network_model.keras")
HISTORY_PLOT_PATH = os.path.join("outputs", "nn_training_history.png")
METRICS_PATH = os.path.join("outputs", "neural_network_metrics.csv")
TARGET = "treatment_outcome"
TARGET_MAP = {
    "0": 0,
    "1": 1,
    "no": 0,
    "ineffective": 0,
    "yes": 1,
    "effective": 1,
}


np.random.seed(42)
tf.random.set_seed(42)

# ---- Load and prepare the same feature set used by the tree models ----
print("Loading feature-engineered data...", flush=True)
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

numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_columns),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                max_categories=100,
                sparse_output=False,
                dtype=np.float32,
            ),
            categorical_columns,
        ),
    ]
)

X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_remaining, y_remaining, test_size=0.5, random_state=42, stratify=y_remaining
)

print("Encoding numeric and categorical features...", flush=True)
X_train_encoded = preprocessor.fit_transform(X_train).astype("float32")
X_validation_encoded = preprocessor.transform(X_validation).astype("float32")
X_test_encoded = preprocessor.transform(X_test).astype("float32")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded).astype("float32")
X_validation_scaled = scaler.transform(X_validation_encoded).astype("float32")
X_test_scaled = scaler.transform(X_test_encoded).astype("float32")

negative_count, positive_count = y_train.value_counts().sort_index().reindex([0, 1], fill_value=0)
class_weight_dict = {
    0: len(y_train) / (2 * negative_count),
    1: len(y_train) / (2 * positive_count),
}

print(f"Encoded features: {X_train_scaled.shape[1]}")
print(f"Train: {len(y_train)} rows, Validation: {len(y_validation)} rows, Test: {len(y_test)} rows")
print(f"Class weights: {class_weight_dict}")

# ---- Build and compile the neural network ----
model = keras.Sequential(
    [
        keras.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)
model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=10,
    restore_best_weights=True,
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
)

# ---- Train with an explicit validation set and class balancing ----
history = model.fit(
    X_train_scaled,
    y_train.to_numpy(),
    validation_data=(X_validation_scaled, y_validation.to_numpy()),
    epochs=100,
    batch_size=256,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ---- Evaluate on the untouched test set ----
y_prob_nn = model.predict(X_test_scaled, batch_size=2048, verbose=0).flatten()
y_pred_nn = (y_prob_nn >= 0.5).astype("int8")
test_metrics = {
    "model": "Step 7 Neural Network",
    "threshold": 0.5,
    "accuracy": accuracy_score(y_test, y_pred_nn),
    "precision": precision_score(y_test, y_pred_nn, zero_division=0),
    "recall": recall_score(y_test, y_pred_nn, zero_division=0),
    "f1_score": f1_score(y_test, y_pred_nn, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob_nn),
}
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"AUC-ROC: {test_metrics['roc_auc']:.4f}")
print(classification_report(y_test, y_pred_nn, target_names=["Ineffective", "Effective"]))

# ---- Plot training history ----
figure, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("Loss Over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history["auc"], label="Train AUC")
axes[1].plot(history.history["val_auc"], label="Validation AUC")
axes[1].set_title("AUC Over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].legend()
figure.tight_layout()
os.makedirs("outputs", exist_ok=True)
figure.savefig(HISTORY_PLOT_PATH, dpi=150)
plt.close(figure)

model.save(MODEL_PATH)
pd.DataFrame([test_metrics]).to_csv(METRICS_PATH, index=False)
print(f"Training history saved to {HISTORY_PLOT_PATH}")
print(f"Neural network model saved to {MODEL_PATH}")
print(f"Metrics saved to {METRICS_PATH}")