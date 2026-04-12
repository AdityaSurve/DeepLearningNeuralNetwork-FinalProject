import os
import json
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _custom_architecture_output_names() -> list[str]:
    """Folder names produced by run_custom_architecture.py (OBJECTIVE_MODE × BIAS_MITIGATION)."""
    names: list[str] = []
    for obj in ("balanced", "accuracy", "hybrid"):
        names.append(f"custom_architecture_{obj}")
        for mit in ("both", "class", "reweigh"):
            names.append(f"custom_architecture_{obj}_mit_{mit}")
    return names


def _all_candidate_models() -> list[str]:
    """Ordered list; compare pulls metrics only for paths that exist."""
    classical = [
        "logistic_regression",
        "logistic_elasticnet",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "hist_gradient_boosting",
    ]
    other_nn = [
        "mlp",
        "tabular_transformer",
    ]
    ensembles = [
        "ensemble",
        "ensemble_full",
    ]
    return classical + other_nn + _custom_architecture_output_names() + ensembles


def _is_ensemble(model: str) -> bool:
    return model.startswith("ensemble")


def _is_deep_tabular(model: str) -> bool:
    return model.startswith("custom_architecture") or model in ("mlp", "tabular_transformer")


def _is_classical_baseline(model: str) -> bool:
    return not _is_ensemble(model) and not _is_deep_tabular(model)


def _collect_records(model_ids: Iterable[str]) -> list[dict]:
    records = []
    for m in model_ids:
        metrics_path = f"outputs/{m}/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
                data["Model"] = m
                records.append(data)
    return records


def main():
    models = _all_candidate_models()
    records = _collect_records(models)

    if not records:
        print("No metrics found.")
        return

    df = pd.DataFrame(records)
    df = df.sort_values(by="roc_auc", ascending=False)

    os.makedirs("outputs/comparisons", exist_ok=True)
    df.to_csv("outputs/comparisons/master_results.csv", index=False)
    df[["Model", "roc_auc", "f1_score", "recall", "accuracy"]].to_csv(
        "outputs/comparisons/leaderboard.csv", index=False
    )

    h = max(6, 0.35 * len(df))
    plt.figure(figsize=(10, h))
    sns.barplot(data=df, x="roc_auc", y="Model")
    plt.title("ROC AUC (all runs with metrics.json)")
    plt.tight_layout()
    plt.savefig("outputs/comparisons/model_comparison_auc.png")
    plt.close()

    plt.figure(figsize=(10, h))
    sns.barplot(data=df, x="recall", y="Model")
    plt.title("Recall (positive class)")
    plt.tight_layout()
    plt.savefig("outputs/comparisons/model_comparison_recall.png")
    plt.close()

    # Best classical (no MLP / transformer / custom / ensemble) vs best deep tabular run
    classical_df = df[df["Model"].apply(_is_classical_baseline)]
    deep_df = df[df["Model"].apply(_is_deep_tabular)]
    ensemble_df = df[df["Model"].apply(_is_ensemble)]

    if not classical_df.empty and not deep_df.empty:
        best_base = classical_df.iloc[0]
        best_deep = deep_df.sort_values(by="roc_auc", ascending=False).iloc[0]
        rows = [best_base, best_deep]
        labels = ["Best classical baseline", "Best deep tabular (MLP / transformer / custom)"]
        if not ensemble_df.empty:
            best_ens = ensemble_df.sort_values(by="roc_auc", ascending=False).iloc[0]
            rows.append(best_ens)
            labels.append("Best ensemble")
        comp_df = pd.DataFrame(rows)
        comp_df["Label"] = labels

        plt.figure(figsize=(9, 5))
        sns.barplot(data=comp_df, x="Label", y="roc_auc", palette=["gray", "steelblue", "darkgreen"][: len(comp_df)])
        plt.title("ROC AUC: best classical vs best deep vs best ensemble (if present)")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=15, ha="right")
        for i, val in enumerate(comp_df["roc_auc"]):
            plt.text(i, val + 0.012, f"{val:.4f}", ha="center")
        plt.tight_layout()
        plt.savefig("outputs/comparisons/custom_vs_baseline.png")
        plt.close()

    # Compact plot: top 15 by AUC (poster-friendly)
    top = df.head(min(15, len(df)))
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="roc_auc", y="Model")
    plt.title("Top models by ROC AUC (up to 15)")
    plt.tight_layout()
    plt.savefig("outputs/comparisons/model_comparison_auc_top15.png")
    plt.close()

    print(f"Model comparison complete ({len(df)} models with metrics).")


if __name__ == "__main__":
    main()
