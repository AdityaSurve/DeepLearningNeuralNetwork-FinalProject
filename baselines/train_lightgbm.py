import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import lightgbm as lgb
from src.metrics import evaluate_model
import joblib

def load_preprocessed(encoding='ord'):
    data = np.load(f'data/processed/data_{encoding}.npz')
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def main():
    print("Loading ordinal data for LightGBM...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed('ord')
    
    model_name = "LightGBM"
    output_dir = "outputs/lightgbm"
    os.makedirs(output_dir, exist_ok=True)
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    print("Training LightGBM...")
    start_time = time.time()
    
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    
    print(f"Training took {time.time() - start_time:.2f} seconds")
    print(f"Best iteration: {clf.best_iteration_}")
    
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
    plt.title("Top 20 Features - LightGBM")
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), bbox_inches='tight')
    plt.close()
    
    # Save model
    joblib.dump(clf, f"{output_dir}/best_model.joblib")
    print(f"Successfully finished {model_name} baseline.")

if __name__ == "__main__":
    main()
