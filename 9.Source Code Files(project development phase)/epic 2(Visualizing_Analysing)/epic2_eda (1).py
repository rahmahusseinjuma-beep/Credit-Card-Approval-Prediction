# Epic 2: Visualizing and Analysing the Data
# This script explores the credit card approval dataset before we clean
# and model it. It covers: importing libraries, reading the data,
# univariate analysis, multivariate analysis, and descriptive analysis.

# 1. Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Read the Dataset
application_df = pd.read_csv("application_record.csv")
credit_df = pd.read_csv("credit_record.csv")

print("Application data shape:", application_df.shape)
print("Credit data shape:", credit_df.shape)

print(application_df.head())
print(application_df.info())

# The dataset stores age and employment length as negative day counts,
# so we convert them into normal positive years to make analysis easier.
application_df["AGE_YEARS"] = -application_df["DAYS_BIRTH"] // 365
application_df["EMPLOYED_YEARS"] = -application_df["DAYS_EMPLOYED"] / 365

# Some rows have a placeholder value for DAYS_EMPLOYED meant for people
# who are not currently employed (pensioners, etc). We treat these as missing.
application_df.loc[application_df["EMPLOYED_YEARS"] < 0, "EMPLOYED_YEARS"] = None


# 3. Univariate Analysis
# (looking at one column at a time)

# Income distribution
plt.figure(figsize=(8, 5))
plt.hist(application_df["AMT_INCOME_TOTAL"], bins=50)
plt.title("Income Distribution")
plt.xlabel("Total Income")
plt.ylabel("Number of Applicants")
plt.xlim(0, 600000)
plt.savefig("income_distribution.png")
plt.close()

# Age distribution
plt.figure(figsize=(8, 5))
plt.hist(application_df["AGE_YEARS"], bins=40)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Applicants")
plt.savefig("age_distribution.png")
plt.close()

# Gender count
plt.figure(figsize=(6, 5))
sns.countplot(data=application_df, x="CODE_GENDER")
plt.title("Applicants by Gender")
plt.savefig("gender_count.png")
plt.close()

# Income type count
plt.figure(figsize=(8, 5))
sns.countplot(data=application_df, y="NAME_INCOME_TYPE")
plt.title("Applicants by Income Type")
plt.savefig("income_type_count.png")
plt.close()

# Education type count
plt.figure(figsize=(8, 5))
sns.countplot(data=application_df, y="NAME_EDUCATION_TYPE")
plt.title("Applicants by Education Type")
plt.savefig("education_type_count.png")
plt.close()


# 4. Multivariate Analysis
# (looking at relationships between two or more columns)

# Correlation between numeric columns
numeric_columns = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "AGE_YEARS",
                    "EMPLOYED_YEARS", "CNT_FAM_MEMBERS"]

plt.figure(figsize=(7, 6))
sns.heatmap(application_df[numeric_columns].corr(), annot=True)
plt.title("Correlation Between Numeric Columns")
plt.savefig("correlation_heatmap.png")
plt.close()

# Income by education level
plt.figure(figsize=(8, 5))
sns.boxplot(data=application_df, x="AMT_INCOME_TOTAL", y="NAME_EDUCATION_TYPE",
            showfliers=False)
plt.title("Income by Education Level")
plt.xlim(0, 400000)
plt.savefig("income_by_education.png")
plt.close()

# Income by gender
plt.figure(figsize=(6, 5))
sns.boxplot(data=application_df, x="CODE_GENDER", y="AMT_INCOME_TOTAL",
            showfliers=False)
plt.title("Income by Gender")
plt.ylim(0, 400000)
plt.savefig("income_by_gender.png")
plt.close()


# 5. Descriptive Analysis
print("\nSummary statistics:")
print(application_df[numeric_columns].describe())

print("\nMissing values in each column:")
print(application_df.isnull().sum())

print("\nIncome type counts:")
print(application_df["NAME_INCOME_TYPE"].value_counts())

print("\nEducation type counts:")
print(application_df["NAME_EDUCATION_TYPE"].value_counts())

print("\nFamily status counts:")
print(application_df["NAME_FAMILY_STATUS"].value_counts())

print("\nHousing type counts:")
print(application_df["NAME_HOUSING_TYPE"].value_counts())

print("\nPayment status counts (from credit_record.csv):")
print(credit_df["STATUS"].value_counts())

print("\nDone. All charts have been saved as PNG files in this folder.")
