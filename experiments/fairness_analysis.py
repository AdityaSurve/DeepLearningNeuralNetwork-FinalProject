import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_fairness(y_true, y_pred, y_prob, group, group_name):
    # Overall
    results = {}
    groups = sorted(group.unique())
    
    for g in groups:
        mask = (group == g)
        if mask.sum() == 0:
            continue
        try:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
            rec = recall_score(y_true[mask], y_pred[mask], zero_division=0)
            prec = precision_score(y_true[mask], y_pred[mask], zero_division=0)
            auc = roc_auc_score(y_true[mask], y_prob[mask])
            
            results[str(g)] = {
                'Accuracy': acc,
                'F1': f1,
                'Recall': rec,
                'Precision': prec,
                'ROC_AUC': float(auc),
                'Count': int(mask.sum())
            }
        except:
            pass
            
    return results

def main():
    models = [
        'logistic_regression',
        'random_forest',
        'xgboost',
        'lightgbm',
        'mlp',
        'custom_architecture_hybrid',
        'custom_architecture_hybrid_mit_both',
        'custom_architecture_hybrid_mit_class',
        'custom_architecture_hybrid_mit_reweigh',
        'custom_architecture_balanced',
        'custom_architecture_accuracy',
    ]
    
    # Load test labels and raw features
    df_raw = pd.read_csv('data/processed/X_test_raw.csv')
    
    protected_attrs = ["sex", "race"]
    available_attrs = [a for a in protected_attrs if a in df_raw.columns]
    
    if not available_attrs:
        print("No protected attributes found for fairness analysis.")
        return
        
    os.makedirs('outputs/fairness', exist_ok=True)
    all_results = []
    
    for model_name in models:
        pred_path = f'outputs/{model_name}/predictions.csv'
        if not os.path.exists(pred_path):
            continue
            
        preds = pd.read_csv(pred_path)
        y_true = preds['y_true'].values
        y_prob = preds['y_prob'].values
        y_pred = preds['y_pred'].values
        
        model_fairness = {}
        for attr in available_attrs:
            group_results = evaluate_fairness(y_true, y_pred, y_prob, df_raw[attr], attr)
            model_fairness[attr] = group_results
            
            # Format for dataframe
            for grp, m in group_results.items():
                m['Model'] = model_name
                m['Attribute'] = attr
                m['Group'] = grp
                all_results.append(m)
                
        with open(f'outputs/{model_name}/fairness_metrics.json', 'w') as f:
            json.dump(model_fairness, f, indent=4)
            
    if not all_results:
        return
        
    fair_df = pd.DataFrame(all_results)
    fair_df.to_csv('outputs/fairness/fairness_comparison_table.csv', index=False)
    
    # Plot Recall variations
    for attr in available_attrs:
        attr_df = fair_df[fair_df['Attribute'] == attr]
        if attr_df.empty: continue
            
        plt.figure(figsize=(12, 6))
        sns.barplot(data=attr_df, x='Group', y='Recall', hue='Model')
        plt.title(f'Recall Fairness Comparison by {attr}')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'outputs/fairness/{attr}_recall_comparison.png')
        plt.close()

if __name__ == "__main__":
    main()
