"""Step 1 - Exploratory Data Analysis for the clinical dataset."""

import os

import matplotlib

matplotlib.use("Agg")  # Save plots without requiring a graphical session.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---- Configuration ----
DATA_PATH = "clinical_data_raw.csv"
OUTPUT_DIR = "outputs"
TARGET = "treatment_outcome"
MISSING_VALUES = ["", "NA", "N/A", "na", "n/a", "None", "none", "null", "NULL", "-"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---- Load and clean column names ----
df = pd.read_csv(DATA_PATH, na_values=MISSING_VALUES, low_memory=False)
clean_names = df.columns.astype(str).str.strip().str.lower()
duplicated_names = clean_names.duplicated(keep=False)
name_counts = {}
unique_names = []
for name in clean_names:
    count = name_counts.get(name, 0)
    unique_names.append(name if count == 0 else f"{name}.{count}")
    name_counts[name] = count + 1
df.columns = unique_names

# Pandas adds .1, .2, etc. to duplicate headers. This makes the result explicit.
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
if duplicated_names.any():
    print("\nDuplicate header groups detected:", clean_names[duplicated_names].tolist())

# ---- Basic statistics ----
print("\nColumn Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDescriptive Statistics:")
print("Numeric columns:")
print(df.describe(include="number").T)
print("\nCategorical columns (unique values):")
categorical_cols = df.select_dtypes(exclude="number").columns
print(df[categorical_cols].nunique(dropna=False).sort_values(ascending=False))

# ---- Data quality checks ----
missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_percent": missing_pct})
print("\nMissing Values:")
print(missing_report[missing_report["missing_count"] > 0])

print("\nDuplicate Rows:", int(df.duplicated().sum()))

if "patient_id" in df.columns:
    print("Duplicate patient_id values:", int(df["patient_id"].duplicated(keep=False).sum()))

# Parse the date when available and report invalid non-missing values.
if "admission_date" in df.columns:
    parsed_dates = pd.to_datetime(df["admission_date"], errors="coerce")
    invalid_dates = df["admission_date"].notna() & parsed_dates.isna()
    print("Invalid admission_date values:", int(invalid_dates.sum()))

# ---- Required EDA findings ----
quality_flags = []

if "age" in df.columns:
    age = pd.to_numeric(df["age"], errors="coerce")
    age_skew = age.skew()
    age_extreme = (age < 0) | (age > 150)
    print("\nAge findings:")
    print(f"Skewness: {age_skew:.3f} ({'right-skewed' if age_skew > 0.5 else 'left-skewed' if age_skew < -0.5 else 'approximately symmetric'})")
    print(f"Values outside 0-150 years: {int(age_extreme.sum())}")
    print(f"Values equal to 0: {int((age == 0).sum())}")
    quality_flags.append({"check": "age_range", "column": "age", "flagged_rows": int(age_extreme.sum()), "details": "Expected range is 0-150 years"})

# These are typical adult reference intervals, used as screening flags rather
# than definitive diagnoses; ranges can vary by laboratory and patient group.
clinical_ranges = {
    "creatinine": (0.6, 1.2),
    "hemoglobin": (12.0, 17.5),
    "wbc_count": (4.0, 11.0),
    "heart_rate": (60.0, 100.0),
    "temperature_f": (97.0, 99.0),
    "hba1c": (4.0, 5.6),
}
print("\nClinical-range findings:")
for column, (lower, upper) in clinical_ranges.items():
    if column not in df.columns:
        continue
    values = pd.to_numeric(df[column], errors="coerce")
    flagged = values.notna() & ((values < lower) | (values > upper))
    count = int(flagged.sum())
    print(f"{column}: {count} outside typical reference range [{lower}, {upper}]")
    quality_flags.append({"check": "clinical_range", "column": column, "flagged_rows": count, "details": f"Typical reference range [{lower}, {upper}]"})

# ---- Target variable balance ----
if TARGET in df.columns:
    print("\nTarget Distribution:")
    target_counts = df[TARGET].value_counts(dropna=False)
    print(target_counts)
    print("\nTarget Proportions:")
    print(df[TARGET].value_counts(normalize=True, dropna=False).round(4))
else:
    print(f"\nWarning: target column '{TARGET}' was not found.")

# ---- Correlation and class-balance findings ----
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [column for column in numeric_cols if column not in {"patient_id", "patient_id.1"}]

if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr(numeric_only=True)
    high_correlation_pairs = []
    for index, first_column in enumerate(corr.columns):
        for second_column in corr.columns[index + 1:]:
            value = corr.loc[first_column, second_column]
            if pd.notna(value) and abs(value) > 0.9:
                high_correlation_pairs.append({"feature_1": first_column, "feature_2": second_column, "correlation": round(float(value), 4)})
    print("\nHighly correlated feature pairs (|correlation| > 0.9):")
    print(high_correlation_pairs if high_correlation_pairs else "None found")

if TARGET in df.columns:
    target_proportions = df[TARGET].value_counts(normalize=True, dropna=False)
    majority_share = float(target_proportions.max()) if not target_proportions.empty else 0
    print("\nTarget-balance finding:")
    if majority_share >= 0.70:
        print(f"Imbalanced: majority class is {majority_share:.1%}. Consider class weights or SMOTE on the training set only.")
    else:
        print(f"No severe imbalance detected: majority class is {majority_share:.1%}.")

# ---- Distribution plots ----
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [column for column in numeric_cols if column not in {"patient_id", "patient_id.1"}]

if numeric_cols:
    plot_cols = numeric_cols[:9]
    rows = (len(plot_cols) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, column in zip(axes, plot_cols):
        sns.histplot(df[column].dropna(), kde=True, ax=axis, color="teal")
        axis.set_title(f"Distribution of {column}")

    for axis in axes[len(plot_cols):]:
        axis.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step1_clinical_distributions.png"), dpi=150)
    plt.close(fig)
    print("Saved distribution plots")
else:
    print("No numeric columns available for distribution plots.")

# ---- Correlation heatmap ----
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, axis = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, annot=len(numeric_cols) <= 15, cmap="RdYlGn", center=0, fmt=".2f", ax=axis)
    axis.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step1_correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved correlation heatmap")
else:
    print("Not enough numeric columns for a correlation heatmap.")

# ---- Categorical association heatmap ----
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
categorical_cols = [column for column in categorical_cols if df[column].nunique(dropna=False) <= 20]


def cramers_v(first_column, second_column):
    """Calculate Cramer's V for two categorical columns."""
    table = pd.crosstab(first_column, second_column, dropna=False).to_numpy(dtype=float)
    if table.size == 0 or min(table.shape) <= 1:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    chi_squared = ((table - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    sample_size = table.sum()
    phi_squared = chi_squared / sample_size
    rows, columns = table.shape
    correction = max(0, min(columns - 1, rows - 1))
    if sample_size <= 1 or correction == 0:
        return 0.0
    return float(np.sqrt(phi_squared / correction))


if len(categorical_cols) >= 2:
    categorical_association = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
    for first_column in categorical_cols:
        for second_column in categorical_cols:
            categorical_association.loc[first_column, second_column] = (
                1.0 if first_column == second_column else cramers_v(df[first_column], df[second_column])
            )

    fig, axis = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        categorical_association,
        annot=len(categorical_cols) <= 12,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        fmt=".2f",
        ax=axis,
    )
    axis.set_title("Categorical Feature Association Matrix (Cramer's V)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step1_categorical_association_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved categorical association heatmap")
else:
    print("Not enough low-cardinality categorical columns for an association heatmap.")

print(f"\nEDA complete. Outputs saved in: {OUTPUT_DIR}")
