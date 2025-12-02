"""
Data Quality & Bias Analysis for Student Performance Dataset
Option 2 - Data Mining Quality & Bias Analysis

- Works with either student-mat.csv or student-por.csv
- Checks:
  * Completeness (missing values)
  * Consistency (categorical codes, duplicates)
  * Validity (ranges / outliers)
  * Bias (sampling, demographics)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. CONFIGURATION ==========

# Change this to "student-por.csv" if you want the Portuguese dataset instead
DATA_FILE = "student-mat.csv"

# Set this to True if you want to show plots
SHOW_PLOTS = False  # you can change to True


# ========== 2. LOAD DATA ==========

print("=== Loading dataset ===")
df = pd.read_csv(DATA_FILE, sep=";")  # UCI uses semicolon in some versions; change sep=";" if needed
print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {DATA_FILE}\n")

print("=== First 5 rows ===")
print(df.head(), "\n")

print("=== Info ===")
print(df.info(), "\n")


# ========== 3. COMPLETENESS CHECK (MISSING VALUES) ==========

print("=== Missing Values (Completeness) ===")
missing_counts = df.isna().sum()
missing_percent = (missing_counts / len(df)) * 100

missing_df = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percent": missing_percent.round(2)
}).sort_values(by="missing_count", ascending=False)

print(missing_df, "\n")

# You can also save this to a CSV if you want to include it as evidence
missing_df.to_csv("missing_values_summary.csv")


# ========== 4. CONSISTENCY CHECKS (CATEGORICAL VALUES & DUPLICATES) ==========

print("=== Categorical Columns & Unique Values (Consistency) ===")
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"Categorical columns: {cat_cols}\n")

for col in cat_cols:
    print(f"--- {col} ---")
    print(df[col].value_counts(dropna=False))
    print()

# Check duplicates (exact row duplicates)
print("=== Duplicate Rows Check ===")
dup_count = df.duplicated().sum()
print(f"Number of completely duplicated rows: {dup_count}\n")

# If you want to check potential duplicates based on a subset of columns:
# e.g., same student by school, sex, age, address, etc.
potential_id_cols = ["school", "sex", "age", "address", "famsize", "Pstatus"]
existing_id_cols = [c for c in potential_id_cols if c in df.columns]

if existing_id_cols:
    dup_subset_count = df.duplicated(subset=existing_id_cols).sum()
    print(f"Possible duplicates based on {existing_id_cols}: {dup_subset_count}\n")


# ========== 5. VALIDITY CHECKS (RANGES / OUTLIERS) ==========

print("=== Validity Checks for Numeric Columns ===")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {num_cols}\n")

print("=== Descriptive Stats ===")
print(df[num_cols].describe().T, "\n")

# Example validity checks for grades (expected 0–20)
grade_cols = [c for c in ["G1", "G2", "G3"] if c in df.columns]

for g in grade_cols:
    invalid_low = df[df[g] < 0]
    invalid_high = df[df[g] > 20]
    print(f"--- {g} validity ---")
    print(f"Values < 0: {len(invalid_low)}")
    print(f"Values > 20: {len(invalid_high)}\n")

# Example validity check for absences (e.g., extremely large values)
if "absences" in df.columns:
    print("=== Absences distribution ===")
    print(df["absences"].describe(), "\n")

    if SHOW_PLOTS:
        plt.figure()
        df["absences"].hist(bins=30)
        plt.title("Absences Distribution")
        plt.xlabel("Absences")
        plt.ylabel("Count")
        plt.show()

# Turn this ON if you want plots saved to files:
SAVE_PLOTS = True

if "absences" in df.columns and SAVE_PLOTS:
    plt.figure()
    df["absences"].hist(bins=20)
    plt.title("Absences Distribution")
    plt.xlabel("Number of absences")
    plt.ylabel("Number of students")
    plt.tight_layout()
    plt.savefig("plot_absences_histogram.png")
    plt.close()



# ========== 6. BIAS / SAMPLING ANALYSIS (DATA MINING PERSPECTIVE) ==========

print("=== Sampling / Demographic Bias Checks ===")

# School distribution
if "school" in df.columns:
    print("--- School distribution ---")
    print(df["school"].value_counts(normalize=True) * 100, "\n")

# Sex distribution
if "sex" in df.columns:
    print("--- Sex distribution ---")
    print(df["sex"].value_counts(normalize=True) * 100, "\n")

# Age distribution
if "age" in df.columns:
    print("--- Age distribution ---")
    print(df["age"].value_counts().sort_index(), "\n")

    if SHOW_PLOTS:
        plt.figure()
        df["age"].hist(bins=range(df["age"].min(), df["age"].max() + 1))
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.show()

# Parent education (socioeconomic bias proxy)
parent_edu_cols = [c for c in ["Medu", "Fedu"] if c in df.columns]
for col in parent_edu_cols:
    print(f"--- {col} (Parent Education) ---")
    print(df[col].value_counts(normalize=True).sort_index() * 100, "\n")

# If you want to define a pass/fail label to inspect class imbalance:
if set(grade_cols):
    target = grade_cols[-1]  # use final grade (G3) if available
    print(f"=== Pass/Fail label based on {target} >= 10 ===")
    df["pass_fail"] = (df[target] >= 10).astype(int)
    print(df["pass_fail"].value_counts(normalize=True) * 100, "\n")

# === Summary tables for key categorical variables ===

summary_tables = {}

if "school" in df.columns:
    school_dist = (df["school"].value_counts()
                   .to_frame(name="count"))
    school_dist["percent"] = (school_dist["count"] / len(df) * 100).round(2)
    summary_tables["school_distribution"] = school_dist
    school_dist.to_csv("table_school_distribution.csv")
    print("=== School distribution ===")
    print(school_dist, "\n")

if "sex" in df.columns:
    sex_dist = (df["sex"].value_counts()
                .to_frame(name="count"))
    sex_dist["percent"] = (sex_dist["count"] / len(df) * 100).round(2)
    summary_tables["sex_distribution"] = sex_dist
    sex_dist.to_csv("table_sex_distribution.csv")
    print("=== Sex distribution ===")
    print(sex_dist, "\n")

# Pass/fail distribution (if you created pass_fail)
if "pass_fail" in df.columns:
    pf_dist = (df["pass_fail"].value_counts()
               .rename(index={0: "fail", 1: "pass"})
               .to_frame(name="count"))
    pf_dist["percent"] = (pf_dist["count"] / len(df) * 100).round(2)
    summary_tables["pass_fail_distribution"] = pf_dist
    pf_dist.to_csv("table_pass_fail_distribution.csv")
    print("=== Pass/Fail distribution ===")
    print(pf_dist, "\n")


# ========== 7. FEATURE QUALITY / CORRELATION (DATA MINING) ==========

if grade_cols:
    print("=== Correlation with final grade (Feature Quality) ===")
    corr_with_G3 = df[num_cols].corr()[grade_cols[-1]].sort_values(ascending=False)
    print(corr_with_G3, "\n")

    # Save correlation to CSV if needed
    corr_with_G3.to_csv("correlation_with_final_grade.csv")


print("=== Analysis complete. Check generated CSV files for summaries. ===")
