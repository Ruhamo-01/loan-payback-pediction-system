from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
from datetime import datetime

# Initialize app and point template_folder to 'frontend'
app = Flask(__name__, template_folder='../frontend')
app.secret_key = 'supersecretkey'
CORS(app)

# Load model pipeline
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PIPELINE_PATH = os.path.join(MODEL_DIR, "full_pipeline_model.pkl")
pipeline = joblib.load(PIPELINE_PATH)
print("Full pipeline loaded successfully!")

# Log file setup
LOG_FILE = "prediction_logs.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "timestamp", "full_name", "annual_income", "debt_to_income_ratio",
        "credit_score", "loan_amount", "interest_rate", "gender",
        "marital_status", "education_level", "employment_status",
        "loan_purpose", "grade_subgrade", "prediction"
    ]).to_csv(LOG_FILE, index=False)

def log_prediction(input_data, prediction):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **input_data,
        "prediction": prediction
    }
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# Routes
@app.route('/')
def home():
    return render_template('index.html')  # Flask will now look in ../frontend

# Single prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        pred = pipeline.predict(df)[0]

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df)[0]
            prob_payback = round(probs[1]*100, 2)
            prob_not_payback = round(probs[0]*100, 2)
        else:
            prob_payback = None
            prob_not_payback = None

        label = "Will Pay Back" if pred == 1 else "Will Not Pay Back"

        log_prediction(data, label)

        return jsonify({
            "prediction": label,
            "prob_payback": prob_payback,
            "prob_not_payback": prob_not_payback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------------
# Batch Prediction
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)
        df_original = df.copy()

        # Predict
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)

        df_original["Prediction"] = ["Will Pay Back" if p==1 else "Will Not Pay Back" for p in predictions]
        df_original["Probability_Paid_Back"] = probabilities[:, 1]
        df_original["Probability_Not_Paid_Back"] = probabilities[:, 0]

        # Log predictions
        for _, row in df_original.iterrows():
            log_prediction(
                row.drop(["Prediction", "Probability_Paid_Back", "Probability_Not_Paid_Back"]).to_dict(),
                row["Prediction"]
            )

        # Summary
        total = len(predictions)
        will_pay_back = int(sum(predictions == 1))
        will_not_pay_back = int(sum(predictions == 0))
        summary = {
            "total": total,
            "will_pay_back": will_pay_back,
            "will_not_pay_back": will_not_pay_back,
            "percent_pay_back": round(will_pay_back / total * 100, 2),
            "percent_not_pay_back": round(will_not_pay_back / total * 100, 2)
        }

        # Human-readable predictions with type conversion
        human_readable = []
        for _, row in df_original.iterrows():
            row_dict = {col: row[col] for col in df.columns}
            row_dict.update({
                "Prediction": row["Prediction"],
                "Probability_Paid_Back": round(float(row["Probability_Paid_Back"])*100, 2),
                "Probability_Not_Paid_Back": round(float(row["Probability_Not_Paid_Back"])*100, 2)
            })
            human_readable.append(row_dict)

        return jsonify({
            "predictions": human_readable,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True, port=5000)
