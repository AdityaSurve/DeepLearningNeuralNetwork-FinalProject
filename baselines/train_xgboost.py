import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import xgboost as xgb
from src.metrics import evaluate_model
import joblib

def load_preprocessed(encoding='ord'):
    data = np.load(f'data/processed/data_{encoding}.npz')
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def main():
    print("Loading ordinal data for XGBoost...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed('ord')
    
    model_name = "XGBoost"
    output_dir = "outputs/xgboost"
    os.makedirs(output_dir, exist_ok=True)
    
    # We will use scale_pos_weight to handle class imbalance
    # negative / positive instances
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    print("Training XGBoost...")
    start_time = time.time()
    
    # We can pass early stopping directly to fit in modern XGBoost, so we create eval_set
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=20,
        tree_method='hist', # Enable fast histogram tree method
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train, eval_set=eval_set, verbose=50)
    
    print(f"Training took {time.time() - start_time:.2f} seconds")
    print(f"Best iteration: {clf.best_iteration}")
    
    # Evaluate on Test
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    metrics = evaluate_model(y_test, y_prob, y_pred, output_dir, model_name)
    print(f"Test ROC_AUC: {metrics['roc_auc']:.4f}, Test F1: {metrics['f1_score']:.4f}")
    
    # Plot feature importance
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    feature_meta = joblib.load('data/metadata/feature_names.joblib')
    features = feature_meta['ord_features']
    
    importance = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10,8))
    sns.barplot(data=importance.head(20), x='Importance', y='Feature')
    plt.title("Top 20 Features - XGBoost")
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), bbox_inches='tight')
    plt.close()
    
    # Save model
    joblib.dump(clf, f"{output_dir}/best_model.joblib")
    # Save learning curve
    results = clf.evals_result()
    plt.figure()
    plt.plot(results['validation_0']['auc'], label='train')
    plt.plot(results['validation_1']['auc'], label='val')
    plt.legend()
    plt.title('XGBoost AUC Learning Curve')
    plt.savefig(os.path.join(output_dir, "learning_curve.png"))
    plt.close()
    
    print(f"Successfully finished {model_name} baseline.")

if __name__ == "__main__":
    main()
