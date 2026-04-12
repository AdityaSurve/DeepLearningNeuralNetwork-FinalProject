import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from catboost import CatBoostClassifier

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
    print("Loading ordinal data for CatBoost...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed("ord")

    model_name = "CatBoost"
    output_dir = "outputs/catboost"
    os.makedirs(output_dir, exist_ok=True)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1.0)
    print(f"scale_pos_weight={scale_pos_weight:.2f}")

    print("Training CatBoost...")
    t0 = time.time()
    clf = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.8,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        task_type="CPU",
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    print(f"Training took {time.time() - t0:.2f}s; best_iteration={clf.get_best_iteration()}")

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test).astype(int).ravel()

    metrics = evaluate_model(y_test, y_prob, y_pred, output_dir, model_name)
    print(
        f"Test ROC_AUC: {metrics['roc_auc']:.4f}, accuracy: {metrics['accuracy']:.4f}, "
        f"balanced_acc: {metrics['balanced_accuracy']:.4f}"
    )

    clf.save_model(os.path.join(output_dir, "catboost_model.cbm"))
    print(f"Saved {output_dir}/catboost_model.cbm")


if __name__ == "__main__":
    main()
