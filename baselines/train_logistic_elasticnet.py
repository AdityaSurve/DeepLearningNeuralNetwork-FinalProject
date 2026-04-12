"""
Elastic-net logistic regression on one-hot features (distinct from train_logistic_regression.py,
which uses L2-only grid). Uses saga + balanced class weights.
"""
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.metrics import evaluate_model


def load_preprocessed(encoding="ohe"):
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
    print("Loading OHE data for elastic-net logistic regression...")
    X_train, y_train, _, _, X_test, y_test = load_preprocessed("ohe")

    model_name = "LogisticElasticNet"
    output_dir = "outputs/logistic_elasticnet"
    os.makedirs(output_dir, exist_ok=True)

    base = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=3000,
        random_state=42,
        class_weight="balanced",
    )
    param_grid = {
        "C": [0.05, 0.1, 0.5, 1.0, 5.0],
        "l1_ratio": [0.15, 0.5, 0.85],
    }
    print("GridSearchCV (3-fold on train, roc_auc)...")
    t0 = time.time()
    grid = GridSearchCV(
        base,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_} | fit+search {time.time() - t0:.2f}s")

    best = grid.best_estimator_
    y_prob = best.predict_proba(X_test)[:, 1]
    y_pred = best.predict(X_test).astype(int).ravel()

    metrics = evaluate_model(y_test, y_prob, y_pred, output_dir, model_name)
    print(
        f"Test ROC_AUC: {metrics['roc_auc']:.4f}, accuracy: {metrics['accuracy']:.4f}, "
        f"balanced_acc: {metrics['balanced_accuracy']:.4f}"
    )

    joblib.dump(best, os.path.join(output_dir, "best_model.joblib"))
    print(f"Saved {output_dir}/best_model.joblib")


if __name__ == "__main__":
    main()
