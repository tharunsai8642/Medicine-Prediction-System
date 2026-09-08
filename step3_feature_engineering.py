"""Step 3 - Create clinically meaningful features from cleaned clinical data.

Run this step on an imputed, deduplicated dataset before StandardScaler is
applied to clinical measurements. The thresholds below use original units.
"""

import numpy as np
import pandas as pd


DATA_PATH = "data_cleaned.csv"
OUTPUT_PATH = "data_feature_engineered.csv"
TARGET_COLUMNS = {"treatment_outcome", "adverse_event", "readmission_30d"}

REQUIRED_COLUMNS = [
    "egfr",
    "alt_enzyme",
    "ast_enzyme",
    "concurrent_drugs",
    "bmi",
    "age",
    "dosage",
]


def kidney_stage(egfr):
    """Classify kidney function using eGFR in mL/min/1.73 m2."""
    if egfr >= 90:
        return "Normal"
    if egfr >= 60:
        return "Mild"
    if egfr >= 30:
        return "Moderate"
    return "Severe"


def bmi_category(bmi):
    """Classify BMI using standard adult BMI cutoffs."""
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


# ---- Load data ----
df = pd.read_csv(DATA_PATH, low_memory=False)
missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Target-derived fields are outcomes, not predictors. Keep them unchanged in
# the output for later splitting, but never use them to create features.
target_present = TARGET_COLUMNS.intersection(df.columns)
print(f"Excluded target/outcome columns from feature engineering: {sorted(target_present)}")

# Clinical thresholds below are invalid for standardized columns. This catches
# accidental execution after the scaling stage of step 2.
scaled_columns = [column for column in ["age", "bmi", "egfr", "dosage"] if column in df.columns]
approximately_standardized = [
    column
    for column in scaled_columns
    if abs(df[column].mean()) < 0.1 and 0.8 < df[column].std() < 1.2
]
if approximately_standardized:
    raise ValueError(
        "Clinical feature engineering requires original units. "
        f"These columns appear standardized: {approximately_standardized}. "
        "Run this step before StandardScaler in step2_cleaning.py."
    )

# ---- Feature 1: Kidney Function Category ----
df["kidney_stage"] = df["egfr"].apply(kidney_stage)

# ---- Feature 2: Liver Risk Flag ----
df["liver_risk"] = ((df["alt_enzyme"] > 40) | (df["ast_enzyme"] > 40)).astype("int8")

# ---- Feature 3: Polypharmacy Flag ----
df["polypharmacy"] = (df["concurrent_drugs"] >= 5).astype("int8")

# ---- Feature 4: BMI Category ----
df["bmi_category"] = df["bmi"].apply(bmi_category)

# ---- Feature 5: Age Group ----
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 40, 60, 80, 120],
    labels=["Pediatric", "Young Adult", "Middle Aged", "Senior", "Elderly"],
    include_lowest=True,
)

# ---- Feature 6: Drug-Age Interaction ----
df["elderly_high_dose"] = (
    (df["age"] > 65) & (df["dosage"] > df["dosage"].median())
).astype("int8")

# ---- Feature 7: Lab Value Ratio ----
df["de_ritis_ratio"] = df["ast_enzyme"] / df["alt_enzyme"].clip(lower=0).add(0.01)
df["de_ritis_ratio"] = df["de_ritis_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)

# ---- Feature 8: Drug x Kidney Function Interaction ----
# This works with the original drug_name column or one-hot drug columns.
drug_columns = [column for column in ["drug_name", "drug_name.1"] if column in df.columns]
if drug_columns:
    drug_column = drug_columns[0]
    df["drug_kidney_interaction"] = (
        df[drug_column].astype(str) + "_" + df["kidney_stage"].astype(str)
    )
else:
    encoded_drugs = [column for column in df.columns if column.startswith("drug_name_")]
    for drug_column in encoded_drugs:
        df[f"{drug_column}_kidney_interaction"] = df[drug_column] * df["kidney_stage"].map(
            {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        )

# ---- Feature 9: Treatment Duration ----
if "duration_days" in df.columns:
    df["long_treatment"] = (df["duration_days"] > 30).astype("int8")

# ---- Feature 10: Admission Time Features ----
if "admission_date" in df.columns:
    admission_dates = pd.to_datetime(df["admission_date"], errors="coerce")
    df["admission_year"] = admission_dates.dt.year.fillna(0).astype("int16")
    df["admission_month"] = admission_dates.dt.month.fillna(0).astype("int8")

new_features = [
    "kidney_stage",
    "liver_risk",
    "polypharmacy",
    "bmi_category",
    "age_group",
    "elderly_high_dose",
    "de_ritis_ratio",
]
new_features.extend(
    column
    for column in ["drug_kidney_interaction", "long_treatment", "admission_year", "admission_month"]
    if column in df.columns
)
print(f"Features after engineering: {df.shape[1]} columns")
print("New features:", new_features)
print("\nNew feature summary:")
print(df[new_features].describe(include="all").T)

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved feature-engineered data: {OUTPUT_PATH}")
