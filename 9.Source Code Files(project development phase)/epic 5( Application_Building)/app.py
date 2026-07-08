# Epic 5: Application Building
# This is the Flask backend. It loads the model we trained in Epic 4
# and uses it to predict credit card approval for new applicants.

from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved model, scaler, and column list from Epic 4
with open("model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    # Start with every column the model expects, set to 0.
    # We will fill in the real values below.
    input_data = {column: 0 for column in model_columns}

    # Simple numeric and yes/no fields
    input_data["CODE_GENDER"] = 1 if form["gender"] == "M" else 0
    input_data["FLAG_OWN_CAR"] = 1 if form["own_car"] == "Y" else 0
    input_data["FLAG_OWN_REALTY"] = 1 if form["own_realty"] == "Y" else 0
    input_data["CNT_CHILDREN"] = int(form["children"])
    input_data["AMT_INCOME_TOTAL"] = float(form["income"])
    input_data["FLAG_MOBIL"] = 1
    input_data["FLAG_WORK_PHONE"] = 1 if form["work_phone"] == "Y" else 0
    input_data["FLAG_PHONE"] = 1 if form["phone"] == "Y" else 0
    input_data["FLAG_EMAIL"] = 1 if form["email"] == "Y" else 0
    input_data["CNT_FAM_MEMBERS"] = int(form["family_members"])

    # The model was trained on AGE_YEARS and EMPLOYED_YEARS
    # instead of the raw DAYS_BIRTH / DAYS_EMPLOYED columns.
    input_data["AGE_YEARS"] = int(form["age"])
    input_data["EMPLOYED_YEARS"] = float(form["employed_years"])

    # One-hot encoded columns: set the matching column to 1 if it exists.
    # (drop_first=True was used in Epic 3, so one category per group has
    # no column at all -- that is expected and already handled by leaving
    # every column at 0 by default.)
    income_type_column = "NAME_INCOME_TYPE_" + form["income_type"]
    if income_type_column in input_data:
        input_data[income_type_column] = 1

    education_column = "NAME_EDUCATION_TYPE_" + form["education_type"]
    if education_column in input_data:
        input_data[education_column] = 1

    family_status_column = "NAME_FAMILY_STATUS_" + form["family_status"]
    if family_status_column in input_data:
        input_data[family_status_column] = 1

    housing_column = "NAME_HOUSING_TYPE_" + form["housing_type"]
    if housing_column in input_data:
        input_data[housing_column] = 1

    occupation_column = "OCCUPATION_TYPE_" + form["occupation_type"]
    if occupation_column in input_data:
        input_data[occupation_column] = 1

    # Build a single-row DataFrame in the exact same column order used
    # during training, then scale it the same way training data was scaled.
    input_df = pd.DataFrame([input_data])[model_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        result = "Rejected"
        risk_category = "High Risk"
    else:
        result = "Approved"
        risk_category = "Low Risk"

    return render_template("result.html", result=result, risk_category=risk_category)


if __name__ == "__main__":
    app.run(debug=False)
