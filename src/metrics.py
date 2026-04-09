import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score, brier_score_loss, roc_auc_score, f1_score, recall_score, accuracy_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_prob, y_pred, output_dir, model_name):
    """
    Evaluates predictions and saves metrics to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Classification report
    cr = classification_report(y_true, y_pred)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(cr)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    
    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.3f})')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()
    
    # 5. Core Metrics dict
    metrics = {
        'roc_auc': roc_auc,
        'average_precision': ap,
        'brier_score': brier_score_loss(y_true, y_prob),
        'f1_score': f1_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save predictions
    pd.DataFrame({'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred}).to_csv(
        os.path.join(output_dir, "predictions.csv"), index=False
    )
    
    return metrics
