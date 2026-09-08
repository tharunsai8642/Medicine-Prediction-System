"""Step 9 - Compare the saved treatment-outcome model results."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TUNED_METRICS_PATH = os.path.join("outputs", "model_metrics_60_20_20_threshold_tuned.csv")
SHARED_INPUT_PATH = os.path.join("outputs", "shared_test_set_model_comparison.csv")
NEURAL_NETWORK_METRICS_PATH = os.path.join("outputs", "neural_network_metrics.csv")
LSTM_METRICS_PATH = os.path.join("outputs", "lstm_metrics.csv")
SHARED_COMPARISON_PATH = os.path.join("outputs", "shared_test_set_model_comparison.csv")
SHARED_PLOT_PATH = os.path.join("outputs", "shared_test_set_model_comparison.png")
LSTM_COMPARISON_PATH = os.path.join("outputs", "lstm_separate_evaluation.csv")
LSTM_PLOT_PATH = os.path.join("outputs", "lstm_separate_evaluation.png")
METRICS = ["accuracy", "precision", "recall", "F1", "AUC-ROC"]
LABELS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
SHARED_MODEL_ORDER = ["Decision Tree", "Random Forest", "XGBoost", "Neural Network"]


def load_metrics(path):
    metrics = pd.read_csv(path)
    metrics = metrics.rename(columns={"model": "Model", "roc_auc": "AUC-ROC", "f1_score": "F1"})
    required_columns = {"Model", "accuracy", "precision", "recall", "F1", "AUC-ROC"}
    missing_columns = required_columns.difference(metrics.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {sorted(missing_columns)}")
    return metrics[["Model", "accuracy", "precision", "recall", "F1", "AUC-ROC"]]


metrics_input_path = TUNED_METRICS_PATH if os.path.exists(TUNED_METRICS_PATH) else SHARED_INPUT_PATH
if not os.path.exists(metrics_input_path):
    raise FileNotFoundError(f"No shared model metrics found: {TUNED_METRICS_PATH} or {SHARED_INPUT_PATH}")

shared_df = load_metrics(metrics_input_path)
if os.path.exists(NEURAL_NETWORK_METRICS_PATH):
    shared_df = pd.concat([shared_df, load_metrics(NEURAL_NETWORK_METRICS_PATH)], ignore_index=True)
else:
    print(f"Neural Network metrics not found; run Step 7 to include it: {NEURAL_NETWORK_METRICS_PATH}")

shared_df["Model"] = shared_df["Model"].replace(
    {
        "Step 4 Decision Tree": "Decision Tree",
        "Step 5 Random Forest": "Random Forest",
        "Step 6 XGBoost": "XGBoost",
        "Step 7 Neural Network": "Neural Network",
    }
)
shared_df = shared_df.drop_duplicates(subset="Model", keep="last")
shared_df = shared_df[shared_df["Model"].isin(SHARED_MODEL_ORDER)].copy()
shared_df["selection_score"] = (
    0.40 * shared_df["AUC-ROC"]
    + 0.30 * shared_df["precision"]
    + 0.20 * shared_df["F1"]
    + 0.10 * shared_df["accuracy"]
)
shared_df["Model"] = pd.Categorical(shared_df["Model"], categories=SHARED_MODEL_ORDER, ordered=True)
shared_df = shared_df.sort_values("Model").reset_index(drop=True)
os.makedirs("outputs", exist_ok=True)
shared_df.to_csv(SHARED_COMPARISON_PATH, index=False)
print("Shared 60/20/20 test-set models:")
print(shared_df.to_string(index=False))
selected_model = shared_df.loc[shared_df["selection_score"].idxmax(), "Model"]
print(f"Selected deployment model: {selected_model} (highest composite selection score)")
print(f"Shared comparison table saved to {SHARED_COMPARISON_PATH}")

positions = np.arange(len(shared_df))
bar_width = 0.15
figure, axis = plt.subplots(figsize=(13, 7))
for offset, (metric, label) in enumerate(zip(METRICS, LABELS)):
    bars = axis.bar(
        positions + (offset - 2) * bar_width,
        shared_df[metric],
        bar_width,
        label=label,
    )
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)

axis.set_title("Treatment-Outcome Model Comparison")
axis.set_ylabel("Score")
axis.set_ylim(0, 1)
axis.set_xticks(positions)
axis.set_xticklabels(shared_df["Model"], rotation=15, ha="right")
axis.legend()
axis.grid(axis="y", alpha=0.25)
figure.tight_layout()
figure.savefig(SHARED_PLOT_PATH, dpi=150)
plt.close(figure)
print(f"Shared comparison chart saved to {SHARED_PLOT_PATH}")

if os.path.exists(LSTM_METRICS_PATH):
    lstm_df = load_metrics(LSTM_METRICS_PATH)
    lstm_df.to_csv(LSTM_COMPARISON_PATH, index=False)
    print("Separate repeated-visit LSTM evaluation:")
    print(lstm_df.to_string(index=False))
    print(f"LSTM evaluation saved to {LSTM_COMPARISON_PATH}")

    figure, axis = plt.subplots(figsize=(9, 6))
    bars = axis.bar(LABELS, lstm_df.iloc[0][METRICS].astype(float), color="teal")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_title("LSTM Evaluation on Repeated-Visit Patients")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(LSTM_PLOT_PATH, dpi=150)
    plt.close(figure)
    print(f"LSTM evaluation chart saved to {LSTM_PLOT_PATH}")
else:
    print(f"LSTM metrics not found: {LSTM_METRICS_PATH}")
