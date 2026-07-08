# Epic 4: Model Building
# This script trains four classification models on the processed data
# from Epic 3, compares how well they perform, and saves the best one
# so it can be used in the Flask app in Epic 5.

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Read the cleaned data from Epic 3
data = pd.read_csv("processed_data.csv")

X = data.drop(columns=["TARGET"])
y = data["TARGET"]

print("Total rows:", len(data))
print("Target counts:\n", y.value_counts())

# Split into training and testing sets.
# We use stratify=y because the target is very imbalanced (few risky
# applicants), so this keeps the same ratio in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Logistic Regression needs scaled features to work well.
# Tree-based models (Decision Tree, Random Forest, XGBoost) do not need this,
# but scaling does not hurt them, so we use the same scaled data for all models.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Because risky applicants are rare, we tell the models to pay
# more attention to the minority class using class_weight="balanced".
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
                              random_state=42, eval_metric="logloss"),
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
    })

    trained_models[name] = model
    print(f"\n{name} done.")

# Show all results side by side
results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df)

# Pick the best model based on F1 score, since the data is imbalanced
# and F1 balances precision and recall better than plain accuracy.
best_model_name = results_df.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")

# Save the best model, the scaler, and the column names.
# All three are needed later to make predictions in the Flask app.
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

results_df.to_csv("model_comparison_results.csv", index=False)

print("\nDone. Saved best_model.pkl, scaler.pkl, model_columns.pkl, and model_comparison_results.csv")
