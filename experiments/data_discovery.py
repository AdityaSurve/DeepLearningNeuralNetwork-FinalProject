import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("Starting Data Discovery...")
    dataset_path = "../dataset/heart_2020_cleaned.csv"
    
    # Using absolute path just to be safe
    abs_path = os.path.abspath(dataset_path)
    if not os.path.exists(abs_path):
        print(f"Dataset not found at {abs_path}")
        # fallback
        dataset_path = "dataset/heart_2020_cleaned.csv"
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Write column summary
    col_summary = pd.DataFrame({
        'Dtype': df.dtypes,
        'Missing': df.isnull().sum(),
        'Missing_%': (df.isnull().sum() / len(df)) * 100,
        'Unique': df.nunique()
    })
    col_summary.to_csv("reports/tables/column_summary.csv")
    print("Saved column_summary.csv")
    
    # Write missing values summary
    missing_summary = col_summary[col_summary['Missing'] > 0]
    missing_summary.to_csv("reports/tables/missing_values_summary.csv")
    print("Saved missing_values_summary.csv")
    
    # Target inference (look for 'HeartDisease' or '_MICHD')
    targets = [c for c in df.columns if 'heart' in c.lower() or 'michd' in c.lower()]
    target_col = targets[0] if targets else df.columns[-1]
    print(f"Inferred target column: {target_col}")
    
    # Sex and age columns inference
    sex_cols = [c for c in df.columns if 'sex' in c.lower() or 'gender' in c.lower()]
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    print(f"Inferred Sex columns: {sex_cols}")
    print(f"Inferred Age columns: {age_cols}")
    
    sns.set_theme(style="whitegrid")
    
    # Feature distributions
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Distribution of target
    plt.figure()
    sns.countplot(data=df, x=target_col)
    plt.title(f"Class Distribution: {target_col}")
    plt.savefig("reports/figures/class_distribution.png", bbox_inches='tight')
    plt.close()
    print("Saved class_distribution.png")

    # Protected attributes distribution
    if sex_cols:
        plt.figure()
        sns.countplot(data=df, x=sex_cols[0], hue=target_col)
        plt.title(f"Target by {sex_cols[0]}")
        plt.savefig("reports/figures/protected_attribute_distributions.png", bbox_inches='tight')
        plt.close()
        print("Saved protected_attribute_distributions.png")

    if age_cols:
        plt.figure(figsize=(10,6))
        sns.countplot(data=df, x=age_cols[0], hue=target_col)
        plt.title(f"Target by {age_cols[0]}")
        plt.xticks(rotation=45)
        plt.savefig("reports/figures/feature_distributions/age_distribution.png", bbox_inches='tight')
        plt.close()
        
    # Correlation heatmap for numerical
    if len(num_cols) > 1:
        plt.figure(figsize=(12,10))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig("reports/figures/correlation_heatmap.png", bbox_inches='tight')
        plt.close()
        print("Saved correlation_heatmap.png")
        
    # Save a markdown overview
    with open("reports/summary/dataset_overview.md", "w") as f:
        f.write("# Dataset Overview\n\n")
        f.write(f"- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"- **Target Column (inferred)**: {target_col}\n")
        f.write(f"- **Sex/Gender Columns (inferred)**: {sex_cols}\n")
        f.write(f"- **Age Columns (inferred)**: {age_cols}\n\n")
        f.write("## Observations\n")
        f.write("- Dataset loaded successfully. See `reports/tables/column_summary.csv` for details.\n")
        
    print("Done profiling data.")

if __name__ == "__main__":
    main()
