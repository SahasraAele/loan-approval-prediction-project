"""Generate a synthetic loan-approval dataset for testing the project."""
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    n = 1200
    income = rng.normal(65000, 24000, n).clip(12000, 180000).round(0)
    loan_amount = rng.normal(180000, 70000, n).clip(20000, 500000).round(0)
    credit_history = rng.choice([0, 1], n, p=[0.22, 0.78])
    employment = rng.choice(["Salaried", "Self-employed"], n, p=[0.68, 0.32])
    education = rng.choice(["Graduate", "Not Graduate"], n, p=[0.72, 0.28])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n, p=[0.35, 0.40, 0.25])
    dependents = rng.choice(["0", "1", "2", "3+"], n, p=[0.42, 0.22, 0.20, 0.16])
    debt_to_income = loan_amount / np.maximum(income, 1)

    # Synthetic outcome relationship solely for demonstrating a runnable pipeline.
    score = (
        1.8 * credit_history
        + 0.000018 * income
        - 0.000004 * loan_amount
        - 0.65 * debt_to_income
        + 0.25 * (employment == "Salaried")
        + 0.22 * (education == "Graduate")
        + 0.30 * (property_area == "Semiurban")
        - 0.18 * (property_area == "Rural")
        + rng.normal(0, 0.75, n)
    )
    status = np.where(score > np.median(score), "Y", "N")
    df = pd.DataFrame({
        "ApplicantIncome": income,
        "LoanAmount": loan_amount,
        "Credit_History": credit_history,
        "Employment_Type": employment,
        "Education": education,
        "Property_Area": property_area,
        "Dependents": dependents,
        "Loan_Status": status,
    })
    out = Path(__file__).resolve().parent / "loan_data.csv"
    df.to_csv(out, index=False)
    print(f"Created {out} with {len(df)} synthetic rows.")

if __name__ == "__main__":
    main()
