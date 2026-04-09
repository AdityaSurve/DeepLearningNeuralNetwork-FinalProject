import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.metrics import evaluate_model
import joblib

def load_preprocessed(encoding='ohe'):
    data = np.load(f'data/processed/data_{encoding}.npz')
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def main():
    print("Loading OHE data for Random Forest...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed('ohe')
    
    # Subsample training data for RF tuning to save time (it's huge)
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_train), size=100000, replace=False)
    X_train_sub = X_train[sample_idx]
    y_train_sub = y_train[sample_idx]
    
    model_name = "RandomForest"
    output_dir = "outputs/random_forest"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Training Random Forest...")
    start_time = time.time()
    
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    
    param_dist = {
        'n_estimators': [100, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Random search on subsample
    rs = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
    rs.fit(X_train_sub, y_train_sub)
    
    print(f"Best params from tuning: {rs.best_params_}")
    
    # Retrain best on full train
    best_rf = rs.best_estimator_
    best_rf.fit(X_train, y_train)
    
    print(f"Training took {time.time() - start_time:.2f} seconds")
    
    # Evaluate on Test
    y_prob = best_rf.predict_proba(X_test)[:, 1]
    y_pred = best_rf.predict(X_test)
    
    metrics = evaluate_model(y_test, y_prob, y_pred, output_dir, model_name)
    print(f"Test ROC_AUC: {metrics['roc_auc']:.4f}, Test F1: {metrics['f1_score']:.4f}")
    
    # Calculate feature importances
    import pandas as pd
    feature_meta = joblib.load('data/metadata/feature_names.joblib')
    features = feature_meta['ohe_features']
    
    importance = pd.DataFrame({'Feature': features, 'Importance': best_rf.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10,8))
    sns.barplot(data=importance.head(20), x='Importance', y='Feature')
    plt.title("Top 20 Features - Random Forest")
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), bbox_inches='tight')
    plt.close()
    
    # Save model
    joblib.dump(best_rf, f"{output_dir}/best_model.joblib")
    print(f"Successfully finished {model_name} baseline.")

if __name__ == "__main__":
    main()
