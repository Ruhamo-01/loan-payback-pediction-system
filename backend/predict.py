import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# -------------------------
# Load pipeline
BASE_DIR = os.path.dirname(__file__)  # backend folder
MODEL_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH = os.path.join(MODEL_DIR, "full_pipeline_model.pkl")

if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(f"Pipeline not found at {PIPELINE_PATH}")

pipeline = joblib.load(PIPELINE_PATH)
print("Full pipeline loaded successfully!")

# -------------------------
# Sample input (RAW data only)
sample_data = [
    {
        "full_name": "John Smith",
        "annual_income": 60000,
        "debt_to_income_ratio": 0.25,
        "credit_score": 720,
        "loan_amount": 15000,
        "interest_rate": 0.07,
        "gender": "Male",
        "marital_status": "Single",
        "education_level": "High School",
        "employment_status": "Unemployed",
        "loan_purpose": "Home",
        "grade_subgrade": "A1"
    },
    {
        "full_name": "Jane Doe",
        "annual_income": 75000,
        "debt_to_income_ratio": 0.30,
        "credit_score": 750,
        "loan_amount": 25000,
        "interest_rate": 0.05,
        "gender": "Female",
        "marital_status": "Single",
        "education_level": "Master's",
        "employment_status": "Employed",
        "loan_purpose": "Medical",
        "grade_subgrade": "A2"
    },
      {
        "full_name": "Torry lane",
        "annual_income": 100000,
        "debt_to_income_ratio": 0.3,
        "credit_score": 500,
        "loan_amount": 50000,
        "interest_rate": 0.05,
        "gender": "Female",
        "marital_status": "Married",
        "education_level": "Bachelor's",
        "employment_status": "Employed",
        "loan_purpose": "Education",
        "grade_subgrade": "C1"
    }, 
    
    {
        "full_name": "Alice Johnson",
        "annual_income": 85000,
        "debt_to_income_ratio": 0.22,
        "credit_score": 710,
        "loan_amount": 20000,
        "interest_rate": 0.06,
        "gender": "Female",
        "marital_status": "Single",
        "education_level": "Master's",
        "employment_status": "Employed",
        "loan_purpose": "Medical",
        "grade_subgrade": "B2"
    },
    {
        "full_name": "Michael Brown",
        "annual_income": 60000,
        "debt_to_income_ratio": 0.28,
        "credit_score": 680,
        "loan_amount": 15000,
        "interest_rate": 0.07,
        "gender": "Male",
        "marital_status": "Married",
        "education_level": "Bachelor's",
        "employment_status": "Employed",
        "loan_purpose": "Car",
        "grade_subgrade": "B1"
    },
    {
        "full_name": "Sophia Williams",
        "annual_income": 95000,
        "debt_to_income_ratio": 0.18,
        "credit_score": 730,
        "loan_amount": 30000,
        "interest_rate": 0.05,
        "gender": "Female",
        "marital_status": "Single",
        "education_level": "PhD",
        "employment_status": "Employed",
        "loan_purpose": "Home",
        "grade_subgrade": "A2"
    },
    {
        "full_name": "David Lee",
        "annual_income": 45000,
        "debt_to_income_ratio": 0.35,
        "credit_score": 640,
        "loan_amount": 10000,
        "interest_rate": 0.08,
        "gender": "Male",
        "marital_status": "Divorced",
        "education_level": "High School",
        "employment_status": "Self-employed",
        "loan_purpose": "Debt consolidation",
        "grade_subgrade": "C3"
    },
    {
        "full_name": "Emma Davis",
        "annual_income": 120000,
        "debt_to_income_ratio": 0.12,
        "credit_score": 750,
        "loan_amount": 40000,
        "interest_rate": 0.04,
        "gender": "Female",
        "marital_status": "Married",
        "education_level": "Master's",
        "employment_status": "Employed",
        "loan_purpose": "Medical",
        "grade_subgrade": "A1"
    }      
]

df = pd.DataFrame(sample_data)

# -------------------------
# Predictions
predictions = pipeline.predict(df)
probabilities = pipeline.predict_proba(df)

# -------------------------
# Output
output_df = df.copy()
output_df["prediction_label"] = ["Will Pay Back" if p==1 else "Will Not Pay Back" for p in predictions]

# Assign probabilities
output_df["probability_not_payback"] = probabilities[:, 0]
output_df["probability_payback"] = probabilities[:, 1]
output_df["confidence"] = np.max(probabilities, axis=1)

# Add timestamp
output_df["prediction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save CSV
output_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

# Print in a human-readable format
for i, row in output_df.iterrows():
    print(f"\nPrediction for {row['full_name']}: {row['prediction_label']}")
    print(f"Probability of Paying Back: {row['probability_payback']*100:.2f}%")
    print(f"Probability of Not Paying Back: {row['probability_not_payback']*100:.2f}%")
