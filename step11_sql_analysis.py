"""Step 11 - Run clinical outcome analyses with SQLite."""

import os
import sqlite3

import pandas as pd


DATA_PATH = "data_cleaned.csv"
OUTPUT_DIRECTORY = os.path.join("outputs", "step11_sql")
TARGET_MAP = {
    "0": 0,
    "1": 1,
    "no": 0,
    "ineffective": 0,
    "yes": 1,
    "effective": 1,
}
ADVERSE_EVENT_MAP = {
    "0": 0,
    "1": 1,
    "no": 0,
    "yes": 1,
}

QUERIES = {
    "q1_overall_treatment_outcomes": (
        "Q1: Overall Treatment Outcomes",
        """
        SELECT
            treatment_outcome,
            COUNT(*) AS patient_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM patients), 2) AS pct
        FROM patients
        GROUP BY treatment_outcome
        ORDER BY treatment_outcome;
        """,
    ),
    "q2_efficacy_by_age_group": (
        "Q2: Efficacy by Age Group",
        """
        SELECT
            CASE
                WHEN age < 30 THEN 'Under 30'
                WHEN age BETWEEN 30 AND 50 THEN '30-50'
                WHEN age BETWEEN 51 AND 65 THEN '51-65'
                ELSE 'Over 65'
            END AS age_group,
            COUNT(*) AS total_patients,
            SUM(treatment_outcome) AS effective_count,
            ROUND(AVG(treatment_outcome) * 100, 2) AS efficacy_pct
        FROM patients
        GROUP BY age_group
        ORDER BY efficacy_pct DESC;
        """,
    ),
    "q3_top_drugs_by_efficacy": (
        "Q3: Drug Efficacy Ranking",
        """
        SELECT
            drug_name,
            COUNT(*) AS total_prescribed,
            SUM(treatment_outcome) AS effective,
            ROUND(AVG(treatment_outcome) * 100, 2) AS efficacy_rate
        FROM patients
        WHERE drug_name IS NOT NULL AND TRIM(drug_name) <> ''
        GROUP BY drug_name
        HAVING COUNT(*) >= 100
        ORDER BY efficacy_rate DESC
        LIMIT 10;
        """,
    ),
    "q4_adverse_events_by_drug_and_age": (
        "Q4: ADR Rate Difference Between Elderly and Non-Elderly Patients by Drug",
        """
        SELECT
            LOWER(TRIM(drug_name)) AS drug_name,
            SUM(CASE WHEN age > 65 THEN 1 ELSE 0 END) AS elderly_patients,
            SUM(CASE WHEN age > 65 THEN adverse_event ELSE 0 END) AS elderly_adr_count,
            ROUND(
                SUM(CASE WHEN age > 65 THEN adverse_event ELSE 0 END) * 100.0
                / SUM(CASE WHEN age > 65 THEN 1 ELSE 0 END),
                2
            ) AS elderly_adr_rate,
            SUM(CASE WHEN age <= 65 THEN 1 ELSE 0 END) AS non_elderly_patients,
            SUM(CASE WHEN age <= 65 THEN adverse_event ELSE 0 END) AS non_elderly_adr_count,
            ROUND(
                SUM(CASE WHEN age <= 65 THEN adverse_event ELSE 0 END) * 100.0
                / SUM(CASE WHEN age <= 65 THEN 1 ELSE 0 END),
                2
            ) AS non_elderly_adr_rate,
            ROUND(
                (
                    SUM(CASE WHEN age > 65 THEN adverse_event ELSE 0 END) * 100.0
                    / SUM(CASE WHEN age > 65 THEN 1 ELSE 0 END)
                )
                - (
                    SUM(CASE WHEN age <= 65 THEN adverse_event ELSE 0 END) * 100.0
                    / SUM(CASE WHEN age <= 65 THEN 1 ELSE 0 END)
                ),
                2
            ) AS rate_difference_pp
        FROM patients
        WHERE drug_name IS NOT NULL
          AND TRIM(drug_name) <> ''
                    AND LOWER(TRIM(drug_name)) NOT IN ('na', 'n/a', 'unknown', '-', '--')
          AND age IS NOT NULL
        GROUP BY LOWER(TRIM(drug_name))
        HAVING SUM(CASE WHEN age > 65 THEN 1 ELSE 0 END) >= 100
           AND SUM(CASE WHEN age <= 65 THEN 1 ELSE 0 END) >= 100
        ORDER BY rate_difference_pp DESC
        LIMIT 15;
        """,
    ),
    "q5_high_risk_patients": (
        "Q5: High-Risk Patients (Multiple Risk Factors)",
        """
        SELECT
            patient_id,
            age,
            bmi,
            creatinine,
            concurrent_drugs,
            treatment_outcome,
            adverse_event
        FROM patients
        WHERE age > 65
          AND creatinine > 1.5
          AND concurrent_drugs >= 5
        ORDER BY creatinine DESC
        LIMIT 20;
        """,
    ),
    "q6_outcome_vs_dosage": (
        "Q6: Outcome vs Dosage Level",
        """
        SELECT
            CASE
                WHEN dosage < 100 THEN 'Low (<100)'
                WHEN dosage BETWEEN 100 AND 500 THEN 'Medium (100-500)'
                ELSE 'High (>500)'
            END AS dosage_level,
            COUNT(*) AS patients,
            ROUND(AVG(treatment_outcome) * 100, 2) AS efficacy_pct,
            ROUND(AVG(adverse_event) * 100, 2) AS adr_pct
        FROM patients
        WHERE dosage IS NOT NULL
        GROUP BY dosage_level
        ORDER BY CASE dosage_level
            WHEN 'Low (<100)' THEN 1
            WHEN 'Medium (100-500)' THEN 2
            ELSE 3
        END;
        """,
    ),
}


def run_query(connection, key, title, sql):
    print(f"\n{title}")
    print("=" * 60)
    result = pd.read_sql_query(sql, connection)
    print(result.to_string(index=False))
    result.to_csv(os.path.join(OUTPUT_DIRECTORY, f"{key}.csv"), index=False)
    return result


os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
df = pd.read_csv(DATA_PATH, low_memory=False)
normalized_outcome = df["treatment_outcome"].astype(str).str.strip().str.lower().map(TARGET_MAP)
normalized_adverse_event = df["adverse_event"].astype(str).str.strip().str.lower().map(ADVERSE_EVENT_MAP)
valid_rows = normalized_outcome.notna() & normalized_adverse_event.notna()
df = df.loc[valid_rows].copy()
df["treatment_outcome"] = normalized_outcome.loc[df.index].astype("int8")
df["adverse_event"] = normalized_adverse_event.loc[df.index].astype("int8")
with sqlite3.connect(":memory:") as connection:
    df.to_sql("patients", connection, index=False, if_exists="replace")
    for query_key, (title, sql) in QUERIES.items():
        run_query(connection, query_key, title, sql)

print(f"\nSaved SQL results to {OUTPUT_DIRECTORY}")
