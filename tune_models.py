"""Hyperparameter tuning with GridSearchCV or optional Hyperopt."""
import argparse
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from train import ROOT, load_data, build_preprocessor

def grid_search(X_train, y_train):
    pipe = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])
    params = {
        "model__n_estimators": [100, 250],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    }
    search = GridSearchCV(pipe, params, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    print("Best cross-validation F1:", round(search.best_score_, 4))
    return search.best_estimator_

def hyperopt_search(X_train, y_train):
    try:
        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
        from sklearn.model_selection import cross_val_score
    except ImportError as exc:
        raise SystemExit("Hyperopt is not installed. Run: pip install hyperopt") from exc

    space = {
        "n_estimators": hp.choice("n_estimators", [100, 200, 300]),
        "max_depth": hp.choice("max_depth", [None, 5, 8, 12]),
        "min_samples_split": hp.choice("min_samples_split", [2, 4, 8]),
    }
    def objective(params):
        pipe = Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", RandomForestClassifier(
                random_state=42, class_weight="balanced", **params
            )),
        ])
        score = cross_val_score(pipe, X_train, y_train, cv=3, scoring="f1", n_jobs=-1).mean()
        return {"loss": -score, "status": STATUS_OK}
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20,
                trials=trials, rstate=__import__("numpy").random.default_rng(42))
    # hp.choice returns indices; use the same value lists to recover chosen values.
    decoded = {
        "n_estimators": [100, 200, 300][best["n_estimators"]],
        "max_depth": [None, 5, 8, 12][best["max_depth"]],
        "min_samples_split": [2, 4, 8][best["min_samples_split"]],
    }
    model = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced", **decoded)),
    ])
    model.fit(X_train, y_train)
    print("Best parameters:", decoded)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/loan_data.csv")
    parser.add_argument("--target", default="Loan_Status")
    parser.add_argument("--method", choices=["grid", "hyperopt"], default="grid")
    args = parser.parse_args()
    path = Path(args.data)
    if not path.is_absolute():
        path = ROOT / path
    X, y = load_data(path, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = grid_search(X_train, y_train) if args.method == "grid" else hyperopt_search(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Held-out F1: {f1_score(y_test, pred, zero_division=0):.4f}")
    out = ROOT / "models" / f"tuned_{args.method}_model.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(model, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
