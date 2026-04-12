import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def main():
    print("Starting Data Discovery (Adult Census Income via OpenML)...")
    raw = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = raw.frame.copy()
    target_col = "class"
    lab = df[target_col].astype(str).str.strip()
    df[target_col] = (lab == ">50K").astype(int)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("reports/summary", exist_ok=True)
    os.makedirs("reports/figures/feature_distributions", exist_ok=True)

    col_summary = pd.DataFrame(
        {
            "Dtype": df.dtypes,
            "Missing": df.isnull().sum(),
            "Missing_%": (df.isnull().sum() / len(df)) * 100,
            "Unique": df.nunique(),
        }
    )
    col_summary.to_csv("reports/tables/column_summary.csv")
    print("Saved column_summary.csv")

    missing_summary = col_summary[col_summary["Missing"] > 0]
    missing_summary.to_csv("reports/tables/missing_values_summary.csv")
    print("Saved missing_values_summary.csv")

    sex_cols = [c for c in df.columns if c.lower() == "sex"]
    race_cols = [c for c in df.columns if c.lower() == "race"]
    print(f"Inferred sex columns: {sex_cols}")
    print(f"Inferred race columns: {race_cols}")

    sns.set_theme(style="whitegrid")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    plt.figure()
    sns.countplot(data=df, x=target_col)
    plt.title("Class distribution: income >50K (1) vs not (0)")
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
        f.write(f"- **Source**: UCI / OpenML Adult Census Income (version 2)\n")
        f.write(f"- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"- **Target**: `{target_col}` (1 => income >50K)\n")
        f.write(f"- **Sex column**: {sex_cols}\n")
        f.write(f"- **Race column**: {race_cols}\n\n")
        f.write("## Observations\n")
        f.write("- See `reports/tables/column_summary.csv` for dtypes and missingness.\n")

    print("Done profiling data.")


if __name__ == "__main__":
    main()
