import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.metrics import evaluate_model
import joblib

def load_preprocessed(encoding='ohe'):
    data = np.load(f'data/processed/data_{encoding}.npz')
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def main():
    print("Loading OHE data for Logistic Regression...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed('ohe')
    
    model_name = "LogisticRegression"
    output_dir = "outputs/logistic_regression"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train Logistic Regression with class weights to handle imbalance
    # We will use GridSearchCV on the validation set or simply just basic grid
    print("Training Logistic Regression...")
    start_time = time.time()
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    # Basic tuning: C parameter
    param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
    
    # To keep it memory efficient, taking a subset of train/val isn't needed for LR, but we use PredefinedSplit if we really want to use val set
    # Using simple 5-fold cross validation on train for tuning
    grid = GridSearchCV(lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_lr = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Training took {time.time() - start_time:.2f} seconds")
    
    # Evaluate on Test
    y_prob = best_lr.predict_proba(X_test)[:, 1]
    y_pred = best_lr.predict(X_test)
    
    metrics = evaluate_model(y_test, y_prob, y_pred, output_dir, model_name)
    print(f"Test ROC_AUC: {metrics['roc_auc']:.4f}, Test F1: {metrics['f1_score']:.4f}")
    
    # Save model
    joblib.dump(best_lr, f"{output_dir}/best_model.joblib")
    print(f"Successfully finished {model_name} baseline.")

if __name__ == "__main__":
    main()
