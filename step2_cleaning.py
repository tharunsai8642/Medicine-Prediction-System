"""Step 2 - Clean and prepare the clinical dataset for modelling."""

import pandas as pd


DATA_PATH = "clinical_data_raw.csv"
OUTPUT_PATH = "data_cleaned.csv"
REPORT_PATH = "cleaning_report.csv"
MISSING_VALUES = ["", "NA", "N/A", "na", "n/a", "None", "none", "null", "NULL", "-"]
MISSING_THRESHOLD = 0.50


def make_unique_columns(columns):
    """Normalize headers and add suffixes when the raw file repeats a name."""
    seen = {}
    unique = []
    for column in columns:
        name = str(column).strip().lower()
        count = seen.get(name, 0)
        unique.append(name if count == 0 else f"{name}.{count}")
        seen[name] = count + 1
    return unique


def cap_outliers(dataframe, column):
    """Cap numeric outliers at the 1.5 IQR lower and upper bounds."""
    series = dataframe[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((series < lower) | (series > upper)).sum()
    dataframe[column] = series.clip(lower=lower, upper=upper)
    return int(outliers), float(lower), float(upper)


# ---- Load data and standardize headers ----
df = pd.read_csv(DATA_PATH, na_values=MISSING_VALUES, low_memory=False)
df.columns = make_unique_columns(df.columns)
initial_shape = df.shape
report = []

# ---- Drop columns with more than 50% missing values ----
missing_fraction = df.isna().mean()
dropped_columns = missing_fraction[missing_fraction > MISSING_THRESHOLD].index.tolist()
df = df.drop(columns=dropped_columns)
for column in dropped_columns:
    report.append({"step": "drop_column", "column": column, "details": "more than 50% missing"})
print(f"Dropped {len(dropped_columns)} columns with more than 50% missing values.")

# ---- Convert known numeric fields before imputation ----
known_numeric = [
    "patient_id", "patient_id.1", "age", "age.1", "weight_kg", "weight_lbs",
    "height_cm", "bmi", "systolic_bp", "diastolic_bp", "heart_rate",
    "temperature_f", "hemoglobin", "wbc_count", "alt_enzyme", "ast_enzyme",
    "creatinine", "egfr", "hba1c", "total_cholesterol", "dosage",
    "duration_days", "concurrent_drugs", "readmission_30d", "extra_col_1",
    "extra_col_2", "unnamed_0",
]
for column in known_numeric:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# ---- Fill missing values ----
for column in df.columns:
    missing_count = int(df[column].isna().sum())
    if missing_count == 0:
        continue
    if pd.api.types.is_numeric_dtype(df[column]):
        fill_value = df[column].median()
        if pd.isna(fill_value):
            fill_value = 0
        df[column] = df[column].fillna(fill_value)
        details = f"median={fill_value:.4g}"
    else:
        mode = df[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df[column] = df[column].fillna(fill_value)
        details = f"mode={fill_value}"
    report.append({"step": "impute", "column": column, "details": details, "rows_affected": missing_count})
    print(f"Filled {column}: {details}")

# ---- Remove duplicate records ----
duplicate_rows = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
report.append({"step": "drop_duplicates", "column": "*", "details": f"removed {duplicate_rows} rows"})
print(f"Removed duplicate rows: {duplicate_rows}")

# ---- Cap implausible extreme numeric values using IQR ----
clinical_columns = [
    "age", "bmi", "dosage", "hemoglobin", "creatinine", "alt_enzyme", "ast_enzyme", "egfr", "hba1c"
]
for column in clinical_columns:
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        count, lower, upper = cap_outliers(df, column)
        report.append({"step": "cap_outliers", "column": column, "details": f"bounds=[{lower:.4g}, {upper:.4g}]", "rows_affected": count})
        print(f"Capped {column}: {count} values to [{lower:.2f}, {upper:.2f}]")

# ---- Save results ----
pd.DataFrame(report).to_csv(REPORT_PATH, index=False)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Initial shape: {initial_shape}")
print(f"Final shape: {df.shape}")
print(f"Saved cleaned data: {OUTPUT_PATH}")
print(f"Saved cleaning report: {REPORT_PATH}")
