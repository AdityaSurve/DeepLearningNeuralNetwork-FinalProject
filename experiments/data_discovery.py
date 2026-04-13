import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def _profile_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Copy with string sentinels like ``?`` treated as NaN for profiling."""
    out = df.copy()
    for c in out.columns:
        if c == target_col:
            continue
        if out[c].dtype == object or str(out[c].dtype) == "category":
            out[c] = out[c].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    return out


def main():
    print("Starting Data Discovery (Heart Disease via OpenML)...")
    raw = fetch_openml("heart-disease", version=1, as_frame=True, parser="auto")
    df = raw.frame.copy()
    target_col = "target"
    df[target_col] = df[target_col].astype(float).astype(int)

    df_profile = _profile_frame(df, target_col)
    sns.set_theme(style="whitegrid")

    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("reports/summary", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/figures/feature_distributions", exist_ok=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    dup_n = int(df.duplicated().sum())
    uniq_n = len(df) - dup_n
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Unique rows", "Duplicate rows"], [uniq_n, dup_n], color=["steelblue", "coral"])
    ax.set_ylabel("Count")
    ax.set_title("Heart Disease: duplicate rows (raw OpenML frame)")
    ymax = max(uniq_n, dup_n, 1)
    for i, v in enumerate([uniq_n, dup_n]):
        ax.text(i, v + 0.02 * ymax, f"{v:,}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("reports/figures/duplicate_row_counts.png", bbox_inches="tight")
    plt.close()
    print(f"Duplicates: {dup_n} ({100 * dup_n / max(len(df), 1):.2f}%) — saved duplicate_row_counts.png")

    col_summary = pd.DataFrame(
        {
            "Dtype": df.dtypes,
            "Missing": df_profile.isnull().sum(),
            "Missing_%": (df_profile.isnull().sum() / len(df_profile)) * 100,
            "Unique": df.nunique(),
        }
    )
    col_summary.to_csv("reports/tables/column_summary.csv")
    print("Saved column_summary.csv")

    missing_summary = col_summary[col_summary["Missing"] > 0]
    missing_summary.to_csv("reports/tables/missing_values_summary.csv")
    print("Saved missing_values_summary.csv")

    if not missing_summary.empty:
        m = missing_summary.sort_values("Missing_%", ascending=True)
        plt.figure(figsize=(8, max(3.5, 0.35 * len(m))))
        plot_df = m.reset_index().rename(columns={"index": "column"})
        sns.barplot(data=plot_df, x="Missing_%", y="column", color="steelblue")
        plt.ylabel("Column")
        plt.xlabel(r"Missing (\%)")
        plt.title("Columns with missing values (Heart Disease / OpenML)")
        plt.tight_layout()
        plt.savefig(
            "reports/figures/missing_values_by_column.png",
            bbox_inches="tight",
        )
        plt.close()
        print("Saved missing_values_by_column.png")

    sex_cols = [c for c in df.columns if c.lower() == "sex"]
    race_cols = [c for c in df.columns if c.lower() == "race"]
    print(f"Inferred sex columns: {sex_cols}")
    print(f"Inferred race columns: {race_cols}")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    plt.figure()
    sns.countplot(data=df, x=target_col)
    plt.title("Class distribution: heart disease present (1) vs absent (0)")
    plt.savefig("reports/figures/class_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved class_distribution.png")

    if sex_cols:
        plt.figure()
        sns.countplot(data=df, x=sex_cols[0], hue=target_col)
        plt.title(f"Target by {sex_cols[0]}")
        plt.savefig(
            "reports/figures/protected_attribute_distributions.png",
            bbox_inches="tight",
        )
        plt.close()
        print("Saved protected_attribute_distributions.png")

    if len(num_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation heatmap (numeric features)")
        plt.savefig("reports/figures/correlation_heatmap.png", bbox_inches="tight")
        plt.close()
        print("Saved correlation_heatmap.png")

    with open("reports/summary/dataset_overview.md", "w", encoding="utf-8") as f:
        f.write("# Dataset overview\n\n")
        f.write(f"- **Source**: UCI / OpenML Heart Disease (version 1)\n")
        f.write(f"- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"- **Target**: `{target_col}` (1 => heart disease present)\n")
        f.write(f"- **Sex column**: {sex_cols}\n")
        f.write(f"- **Race column**: {race_cols}\n\n")
        f.write("## Observations\n")
        f.write("- See `reports/tables/column_summary.csv` for dtypes and missingness.\n")

    print("Done profiling data.")


if __name__ == "__main__":
    main()
