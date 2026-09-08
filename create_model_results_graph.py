"""Create the model comparison graph used in the business report."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS_PATH = "outputs/model_metrics_60_20_20_threshold_tuned.csv"
GRAPH_PATH = "outputs/model_results.png"
METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
LABELS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

metrics = pd.read_csv(METRICS_PATH)
models = metrics["model"].tolist()
positions = np.arange(len(models))
bar_width = 0.15

figure, axis = plt.subplots(figsize=(12, 7))
for offset, metric in enumerate(METRICS):
    bars = axis.bar(
        positions + (offset - 2) * bar_width,
        metrics[metric],
        bar_width,
        label=LABELS[offset],
    )
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

axis.set_title("Model Results")
axis.set_ylabel("Score")
axis.set_xticks(positions)
axis.set_xticklabels(models)
axis.set_ylim(0, 1)
axis.legend()
axis.grid(axis="y", alpha=0.25)
figure.tight_layout()
figure.savefig(GRAPH_PATH, dpi=150)
plt.close(figure)