import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'mlp', 'custom_architecture']
    
    records = []
    
    for m in models:
        metrics_path = f'outputs/{m}/metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                data['Model'] = m
                records.append(data)
                
    if not records:
        print("No metrics found.")
        return
        
    df = pd.DataFrame(records)
    
    # Sort backwards so biggest is first
    df = df.sort_values(by='roc_auc', ascending=False)
    
    os.makedirs('outputs/comparisons', exist_ok=True)
    df.to_csv('outputs/comparisons/master_results.csv', index=False)
    df[['Model', 'roc_auc', 'f1_score', 'recall', 'accuracy']].to_csv('outputs/comparisons/leaderboard.csv', index=False)
    
    # Plot overall AUC Comparison
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='roc_auc', y='Model')
    plt.title('ROC AUC Comparison')
    plt.tight_layout()
    plt.savefig('outputs/comparisons/model_comparison_auc.png')
    plt.close()
    
    # Plot overall Recall Comparison
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='recall', y='Model')
    plt.title('Recall Comparison')
    plt.tight_layout()
    plt.savefig('outputs/comparisons/model_comparison_recall.png')
    plt.close()
    
    # Custom vs Best Baseline
    baselines = df[df['Model'] != 'custom_architecture']
    if not baselines.empty and 'custom_architecture' in df['Model'].values:
        best_baseline = baselines.iloc[0]
        custom_arch = df[df['Model'] == 'custom_architecture'].iloc[0]
        
        comp_df = pd.DataFrame([best_baseline, custom_arch])
        
        plt.figure(figsize=(8,6))
        sns.barplot(data=comp_df, x='Model', y='roc_auc', palette=['gray', 'blue'])
        plt.title('Best Baseline vs Custom Architecture (ROC AUC)')
        plt.ylim(0, 1.0)
        for i, val in enumerate(comp_df['roc_auc']):
            plt.text(i, val + 0.01, f"{val:.4f}", ha='center')
        plt.savefig('outputs/comparisons/custom_vs_baseline.png')
        plt.close()

    print("Model comparison complete.")

if __name__ == "__main__":
    main()
