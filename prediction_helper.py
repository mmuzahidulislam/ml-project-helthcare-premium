import joblib
import pandas as pd
model_young = joblib.load('artifacts/model_young.joblib')
model_rest = joblib.load('artifacts/model_rest.joblib')


def total_risk_score(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found
    return total_risk_score


def preprocess_input(dict):
    input_df = pd.DataFrame({
        'age': [dict['Age']],
        'gender': [dict['Gender']],
        'region': [dict['Region']],
        'marital_status': [dict['Marital Status']],
        'number_of_dependants': [dict['Number of Dependants']],
        'bmi_category': [dict['BMI Category']],
        'smoking_status': [dict['Smoking Status']],
        'employment_status': [dict['Employment Status']],
        'income_level': [dict['Income Level']],
        'income_lakhs': [dict['Income in Lakhs']],
        'insurance_plan': [dict['Insurance Plan']],
        'genetical_risk': [dict['Genetical Risk']],
        'total_risk_score': [total_risk_score(dict['Medical History'])]
    })
    return input_df

def predict(input_dict):
    input_df = preprocess_input(input_dict)
    if input_dict['Age']<=25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction)