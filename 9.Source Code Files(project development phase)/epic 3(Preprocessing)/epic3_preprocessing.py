# Epic 3: Data Pre-processing
# This script cleans the raw data, merges the two files together,
# creates the target column (approved or risky), and prepares the
# data so it is ready for model training in Epic 4.

import pandas as pd

# Read the raw files (same ones used in Epic 1 and Epic 2)
application_df = pd.read_csv("application_record.csv")
credit_df = pd.read_csv("credit_record.csv")

print("Before cleaning:")
print("Application rows:", len(application_df))
print("Credit rows:", len(credit_df))


# 1. Drop Duplicate Features
# Some applicants appear more than once with the exact same details.
# We remove exact duplicate rows, keeping only the first copy.
application_df = application_df.drop_duplicates()
application_df = application_df.drop_duplicates(subset="ID")

print("\nApplication rows after dropping duplicates:", len(application_df))


# 2. Handling Missing Values
# OCCUPATION_TYPE has a lot of missing values. Instead of dropping
# those rows, we label them as "Unknown" so we don't lose applicants.
application_df["OCCUPATION_TYPE"] = application_df["OCCUPATION_TYPE"].fillna("Unknown")


# 3. Data Cleaning and Merging
# First we build the target label from the credit history file.
# STATUS codes 2, 3, 4, 5 mean the applicant was 60+ days overdue,
# which we treat as "risky". Everything else (0, 1, C, X) is "safe".
def is_risky(status_list):
    risky_codes = ["2", "3", "4", "5"]
    for status in status_list:
        if status in risky_codes:
            return 1
    return 0

credit_status_by_id = credit_df.groupby("ID")["STATUS"].apply(list)
target_by_id = credit_status_by_id.apply(is_risky)
target_df = target_by_id.reset_index()
target_df.columns = ["ID", "TARGET"]

# Now merge the application data with the target labels.
# Only applicants that exist in both files will remain after this step.
merged_df = application_df.merge(target_df, on="ID", how="inner")

print("\nRows after merging application data with the target labels:", len(merged_df))
print("Target value counts (0 = safe, 1 = risky):")
print(merged_df["TARGET"].value_counts())


# 4. Feature Engineering
# Convert day-based columns into more useful year-based columns.
merged_df["AGE_YEARS"] = -merged_df["DAYS_BIRTH"] // 365
merged_df["EMPLOYED_YEARS"] = -merged_df["DAYS_EMPLOYED"] / 365

# DAYS_EMPLOYED has a placeholder value for people who are not
# currently employed (pensioners, etc). We set those to 0 years.
merged_df.loc[merged_df["EMPLOYED_YEARS"] < 0, "EMPLOYED_YEARS"] = 0

# We no longer need the original day-based columns or the ID column.
merged_df = merged_df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "ID"])


# 5. Handling Categorical Values
# Simple Yes/No flags become 1/0.
merged_df["FLAG_OWN_CAR"] = merged_df["FLAG_OWN_CAR"].map({"Y": 1, "N": 0})
merged_df["FLAG_OWN_REALTY"] = merged_df["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0})
merged_df["CODE_GENDER"] = merged_df["CODE_GENDER"].map({"M": 1, "F": 0})

# Categorical columns with several categories become one-hot encoded columns.
categorical_columns = ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
                        "OCCUPATION_TYPE"]

merged_df = pd.get_dummies(merged_df, columns=categorical_columns, drop_first=True)

print("\nFinal processed data shape:", merged_df.shape)
print("\nFirst few rows of processed data:")
print(merged_df.head())

# Save the cleaned and processed data so Epic 4 can use it directly.
merged_df.to_csv("processed_data.csv", index=False)
print("\nDone. Saved cleaned data as processed_data.csv")
