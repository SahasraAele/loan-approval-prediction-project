"""Predict approval class and probability for one applicant."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = {
    "ApplicantIncome": 72000,
    "LoanAmount": 160000,
    "Credit_History": 1,
    "Employment_Type": "Salaried",
    "Education": "Graduate",
    "Property_Area": "Semiurban",
    "Dependents": "0",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--input", help="JSON file containing one applicant's feature values")
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise SystemExit("Model not found. Train it first with: python src/train.py")
    applicant = EXAMPLE
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            applicant = json.load(f)
    frame = pd.DataFrame([applicant])
    prediction = int(model.predict(frame)[0])
    labels = {0: "Not approved", 1: "Approved"}
    print("Predicted class:", labels.get(prediction, str(prediction)))
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0][1])
        print(f"Model probability for class 1: {probability:.1%}")
    print("Educational demo only; not a real lending decision.")

if __name__ == "__main__":
    main()
