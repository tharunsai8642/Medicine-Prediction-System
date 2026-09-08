"""Step 8 - Train and evaluate an LSTM on repeated patient visits."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import callbacks, layers


DATA_PATH = "data_feature_engineered.csv"
MODEL_PATH = os.path.join("outputs", "lstm_model.keras")
HISTORY_PLOT_PATH = os.path.join("outputs", "lstm_training_history.png")
METRICS_PATH = os.path.join("outputs", "lstm_metrics.csv")
TARGET = "treatment_outcome"
MAX_TIMESTEPS = 4
PADDING_VALUE = -999.0
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "age",
    "weight_kg",
    "height_cm",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "temperature_f",
    "hemoglobin",
    "wbc_count",
    "creatinine",
    "egfr",
    "hba1c",
    "total_cholesterol",
    "dosage",
    "duration_days",
    "concurrent_drugs",
    "admission_year",
    "admission_month",
]
TARGET_MAP = {
    "0": 0,
    "1": 1,
    "no": 0,
    "ineffective": 0,
    "yes": 1,
    "effective": 1,
}


np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def build_sequences(data, transformed_features, patient_ids):
    data = data.reset_index(drop=True)
    transformed_features = np.asarray(transformed_features, dtype="float32")
    patient_to_position = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    sequences = np.full(
        (len(patient_ids), MAX_TIMESTEPS, transformed_features.shape[1]),
        PADDING_VALUE,
        dtype="float32",
    )
    sequence_labels = np.zeros(len(patient_ids), dtype="int8")
    grouped_indices = data.groupby("patient_id", sort=False).indices

    for patient_id, patient_position in patient_to_position.items():
        row_indices = np.asarray(grouped_indices[patient_id], dtype="intp")
        visit_features = transformed_features[row_indices]
        visit_features = visit_features[-MAX_TIMESTEPS:]
        sequences[patient_position, -len(visit_features) :] = visit_features
        sequence_labels[patient_position] = data.iloc[row_indices[-1]]["_target"]

    return sequences, sequence_labels


# Keep only patients with actual repeated visits so the LSTM learns from real histories.
df = pd.read_csv(DATA_PATH, low_memory=False)
df["_target"] = df[TARGET].astype(str).str.strip().str.lower().map(TARGET_MAP)
df = df.loc[df["_target"].notna()].copy()
df["_row_order"] = np.arange(len(df))
df["_parsed_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
df = df.sort_values(["patient_id", "_parsed_date", "_row_order"], na_position="last")
visit_counts = df.groupby("patient_id")["patient_id"].transform("size")
df = df.loc[visit_counts >= 2].copy()

available_features = [column for column in FEATURE_COLUMNS if column in df.columns]
if not available_features:
    raise ValueError("No configured clinical feature columns are available in the dataset.")

patient_frame = df.groupby("patient_id", sort=False).tail(1)
patient_ids = patient_frame["patient_id"].to_numpy()
patient_labels = patient_frame["_target"].astype("int8").to_numpy()
train_ids, remaining_ids, train_labels, remaining_labels = train_test_split(
    patient_ids,
    patient_labels,
    test_size=0.4,
    random_state=RANDOM_STATE,
    stratify=patient_labels,
)
validation_ids, test_ids, validation_labels, test_labels = train_test_split(
    remaining_ids,
    remaining_labels,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=remaining_labels,
)

scaler = StandardScaler()
train_rows = df["patient_id"].isin(train_ids)
scaler.fit(df.loc[train_rows, available_features].apply(pd.to_numeric, errors="coerce").fillna(0))
encoded_features = scaler.transform(
    df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
).astype("float32")
label_series = df["_target"].astype("int8")

train_data = df[df["patient_id"].isin(train_ids)]
validation_data = df[df["patient_id"].isin(validation_ids)]
test_data = df[df["patient_id"].isin(test_ids)]
X_train, y_train = build_sequences(
    train_data,
    encoded_features[df["patient_id"].isin(train_ids)],
    train_ids,
)
X_validation, y_validation = build_sequences(
    validation_data,
    encoded_features[df["patient_id"].isin(validation_ids)],
    validation_ids,
)
X_test, y_test = build_sequences(
    test_data,
    encoded_features[df["patient_id"].isin(test_ids)],
    test_ids,
)

negative_count, positive_count = np.bincount(y_train, minlength=2)
class_weight_dict = {
    0: len(y_train) / (2 * negative_count),
    1: len(y_train) / (2 * positive_count),
}
print(f"Patients with repeated visits: {len(patient_ids)}")
print(f"Features per visit: {len(available_features)}")
print(f"Train: {len(y_train)} patients, Validation: {len(y_validation)} patients, Test: {len(y_test)} patients")
print(f"Class weights: {class_weight_dict}")

lstm_model = keras.Sequential(
    [
        keras.Input(shape=(MAX_TIMESTEPS, len(available_features))),
        layers.Masking(mask_value=PADDING_VALUE),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)
lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)
lstm_model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=8,
    restore_best_weights=True,
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
)
history = lstm_model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=50,
    batch_size=256,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

y_prob_lstm = lstm_model.predict(X_test, batch_size=2048, verbose=0).flatten()
y_pred_lstm = (y_prob_lstm >= 0.5).astype("int8")
test_metrics = {
    "model": "Step 8 LSTM",
    "threshold": 0.5,
    "accuracy": accuracy_score(y_test, y_pred_lstm),
    "precision": precision_score(y_test, y_pred_lstm, zero_division=0),
    "recall": recall_score(y_test, y_pred_lstm, zero_division=0),
    "f1_score": f1_score(y_test, y_pred_lstm, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob_lstm),
}
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"AUC-ROC: {test_metrics['roc_auc']:.4f}")
print(classification_report(y_test, y_pred_lstm, target_names=["Ineffective", "Effective"]))

figure, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("LSTM Loss Over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history["auc"], label="Train AUC")
axes[1].plot(history.history["val_auc"], label="Validation AUC")
axes[1].set_title("LSTM AUC Over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].legend()
figure.tight_layout()
os.makedirs("outputs", exist_ok=True)
figure.savefig(HISTORY_PLOT_PATH, dpi=150)
plt.close(figure)

lstm_model.save(MODEL_PATH)
pd.DataFrame([test_metrics]).to_csv(METRICS_PATH, index=False)
print(f"Training history saved to {HISTORY_PLOT_PATH}")
print(f"LSTM model saved to {MODEL_PATH}")
print(f"Metrics saved to {METRICS_PATH}")
