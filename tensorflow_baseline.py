"""Optional TensorFlow neural-network baseline for binary classification."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from train import ROOT, load_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/loan_data.csv")
    parser.add_argument("--target", default="Loan_Status")
    args = parser.parse_args()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is optional. Install it with: pip install tensorflow") from exc

    path = Path(args.data)
    if not path.is_absolute():
        path = ROOT / path
    X, y = load_data(path, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)
    if hasattr(X_train_t, "toarray"):
        X_train_t, X_test_t = X_train_t.toarray(), X_test_t.toarray()
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_t.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    model.fit(X_train_t, np.asarray(y_train), validation_split=0.2, epochs=25,
              batch_size=32, verbose=1, callbacks=[
                  tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
              ])
    probabilities = model.predict(X_test_t, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    print(classification_report(y_test, predictions, zero_division=0))
    print("Test ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    out = ROOT / "models" / "tensorflow_baseline.keras"
    out.parent.mkdir(exist_ok=True)
    model.save(out)
    print("Saved neural network:", out)
    print("Save the fitted preprocessing pipeline separately before deploying this model.")

if __name__ == "__main__":
    main()
