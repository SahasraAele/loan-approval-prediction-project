# Loan Approval Prediction — Machine Learning Analytics Pipeline

A machine-learning project that compares classification models to predict whether a loan application is likely to be approved, based on applicant and loan attributes.

> **Important:** The included dataset generator creates synthetic demo data for testing the pipeline. It is not real banking data and model outputs must not be used to make real lending decisions. Replace it with a properly licensed dataset before reporting real-world performance.

## Features
- Data validation and preprocessing with scikit-learn pipelines
- Model comparison: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC (when applicable), and a confusion matrix
- Hyperparameter tuning using `GridSearchCV`
- Optional Bayesian-style search with Hyperopt
- Optional TensorFlow neural-network baseline
- Saves the best scikit-learn model and evaluation results
- Predicts approval probability for a new applicant

## Repository structure
```text
loan-approval-prediction/
├── data/
│   └── generate_sample_data.py
├── models/                 # generated model files (not committed)
├── notebooks/              # add exploration notebooks here
├── reports/                # generated metrics and confusion matrix
├── src/
│   ├── train.py
│   ├── tune_models.py
│   ├── predict.py
│   └── tensorflow_baseline.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Setup
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Generate demo data
From the repository root:
```bash
python data/generate_sample_data.py
```
This writes `data/loan_data.csv`. The generated data is synthetic and is only intended to verify that the code runs.

## 3. Train and compare models
```bash
python src/train.py --data data/loan_data.csv --target Loan_Status
```
The script writes model comparison metrics to `reports/model_metrics.csv`, a confusion matrix to `reports/confusion_matrix.png`, and the selected model pipeline to `models/best_model.joblib`.

## 4. Tune hyperparameters
GridSearchCV:
```bash
python src/tune_models.py --data data/loan_data.csv --target Loan_Status --method grid
```

Hyperopt (optional; install requirements first):
```bash
python src/tune_models.py --data data/loan_data.csv --target Loan_Status --method hyperopt
```

## 5. Predict a new application
After training:
```bash
python src/predict.py --model models/best_model.joblib
```
Edit the example applicant values in `src/predict.py` or use `--input applicant.json` with a JSON object containing the same feature names as the training data.

## 6. Optional TensorFlow baseline
```bash
python src/tensorflow_baseline.py --data data/loan_data.csv --target Loan_Status
```
This provides a neural-network baseline for comparison. It is not guaranteed to outperform the classical models.

## Dataset format
The target column should be named `Loan_Status` and contain two classes such as `Y`/`N` or `1`/`0`. Features may include numeric and categorical applicant attributes. Remove IDs and any columns that leak the outcome.

## Evaluation and responsible use
- Check class balance and use precision/recall alongside accuracy.
- Tune thresholds based on the cost of false approvals and false rejections.
- Validate on a separate test set and, where possible, test on data from a later time period.
- Audit for unfair or disparate outcomes across relevant groups and comply with applicable lending laws.
- This is an educational portfolio project, not a credit decision system.

## Suggested GitHub description
`End-to-end loan approval prediction pipeline using scikit-learn, model evaluation, GridSearchCV, Hyperopt, and an optional TensorFlow baseline.`

## Resume bullet (use only after running and verifying the project)
Built a Python machine-learning pipeline to compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting classifiers for loan approval prediction; evaluated performance with classification metrics and optimized hyperparameters using GridSearchCV and Hyperopt.
