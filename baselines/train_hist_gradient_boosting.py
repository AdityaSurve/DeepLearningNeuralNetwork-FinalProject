import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.metrics import evaluate_model


def load_preprocessed(encoding="ord"):
    data = np.load(f"data/processed/data_{encoding}.npz")
    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


def main():
    print("Loading ordinal data for HistGradientBoosting...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed("ord")

    model_name = "HistGradientBoosting"
    output_dir = "outputs/hist_gradient_boosting"
    os.makedirs(output_dir, exist_ok=True)

    # sklearn HGB: early stopping holds out part of training data (no external val API).
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    print("Training HistGradientBoostingClassifier...")
    t0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_tv, y_tv)
    print(f"Training took {time.time() - t0:.2f}s; n_iter_={clf.n_iter_}")

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test).astype(int).ravel()

    metrics = evaluate_model(y_test, y_prob, y_pred, output_dir, model_name)
    print(
        f"Test ROC_AUC: {metrics['roc_auc']:.4f}, accuracy: {metrics['accuracy']:.4f}, "
        f"balanced_acc: {metrics['balanced_accuracy']:.4f}"
    )

    joblib.dump(clf, os.path.join(output_dir, "best_model.joblib"))
    print(f"Saved {output_dir}/best_model.joblib")


if __name__ == "__main__":
    main()
