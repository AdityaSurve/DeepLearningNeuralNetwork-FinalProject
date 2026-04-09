import os
import time
import json
import copy
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)
from src.metrics import evaluate_model
import matplotlib.pyplot as plt

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        # We save directly
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=5, checkpoint_path='checkpoint.pth', device='cpu'):
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            if isinstance(X_batch, (list, tuple)):
                X_batch = [x.to(device) for x in X_batch]
            else:
                X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            if isinstance(X_batch, list):
                y_pred = model(X_batch[0], X_batch[1])
            else:
                y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        y_true_val = []
        y_prob_val = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if isinstance(X_batch, (list, tuple)):
                    X_batch = [x.to(device) for x in X_batch]
                else:
                    X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                if isinstance(X_batch, list):
                    y_pred = model(X_batch[0], X_batch[1])
                else:
                    y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * y_batch.size(0)
                y_true_val.extend(y_batch.cpu().numpy())
                y_prob_val.extend(y_pred.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        
        y_true_val = np.array(y_true_val)
        y_prob_val = np.array(y_prob_val)
        val_auc = roc_auc_score(y_true_val, y_prob_val)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    # Load best
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history

def plot_learning_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    plt.figure()
    plt.plot(epochs, history['val_auc'], label='Validation')
    plt.title('AUC Curve')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'auroc_curve.png'))
    plt.close()

    if 'val_acc' in history and len(history['val_acc']) == len(history['train_loss']):
        plt.figure()
        plt.plot(epochs, history['val_acc'], label='Validation accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'val_accuracy_curve.png'))
        plt.close()

def find_best_threshold_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Threshold that maximizes accuracy on the given set (for imbalanced tabular targets)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    best_t = 0.5
    best_acc = -1.0
    for t in np.linspace(0.005, 0.995, 199):
        pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def find_best_threshold_by_metric(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str
) -> Tuple[float, float]:
    """
    Find threshold that maximizes a threshold-based metric on the given set.
    Supported: acc, balanced_acc, f1
    """
    metric = metric.strip().lower()
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.005, 0.995, 199):
        pred = (y_prob >= t).astype(int)
        if metric in ("acc", "accuracy"):
            score = accuracy_score(y_true, pred)
        elif metric in ("balanced_acc", "balanced_accuracy", "bacc"):
            score = balanced_accuracy_score(y_true, pred)
        elif metric in ("f1", "f1_score"):
            score = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t, best_score


def compute_val_summary(
    y_true: np.ndarray, y_prob: np.ndarray, thr_metric: str = "balanced_acc"
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    pred_05 = (y_prob >= 0.5).astype(int)
    acc_05 = accuracy_score(y_true, pred_05)
    bacc_05 = balanced_accuracy_score(y_true, pred_05)
    f1_05 = f1_score(y_true, pred_05, zero_division=0)

    best_thr, best_thr_score = find_best_threshold_by_metric(
        y_true, y_prob, thr_metric
    )
    pred_best = (y_prob >= best_thr).astype(int)
    acc_best = accuracy_score(y_true, pred_best)
    bacc_best = balanced_accuracy_score(y_true, pred_best)
    f1_best = f1_score(y_true, pred_best, zero_division=0)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "acc_05": float(acc_05),
        "bacc_05": float(bacc_05),
        "f1_05": float(f1_05),
        "best_thr_metric": thr_metric,
        "best_thr": float(best_thr),
        "best_thr_score": float(best_thr_score),
        "acc_best": float(acc_best),
        "bacc_best": float(bacc_best),
        "f1_best": float(f1_best),
    }


def evaluate_pytorch_model(
    model,
    test_loader,
    output_dir,
    model_name,
    device="cpu",
    threshold=0.5,
):
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if isinstance(X_batch, (list, tuple)):
                X_batch = [x.to(device) for x in X_batch]
            else:
                X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if isinstance(X_batch, list):
                y_pred = model(X_batch[0], X_batch[1])
            else:
                y_pred = model(X_batch)
            y_prob_batch = torch.sigmoid(y_pred).cpu().numpy()
            y_true.extend(y_batch.cpu().numpy())
            y_prob.extend(y_prob_batch)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob).flatten()
    y_pred_class = (y_prob >= threshold).astype(int)

    metrics = evaluate_model(y_true, y_prob, y_pred_class, output_dir, model_name)
    metrics["n_parameters"] = count_parameters(model)
    metrics["threshold"] = float(threshold)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Test ROC_AUC: {metrics['roc_auc']:.4f}")
    return metrics


def train_model_max_val_accuracy(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=80,
    patience=15,
    checkpoint_path="checkpoint.pth",
    device="cpu",
    checkpoint_on_val_acc_at_best_threshold: bool = True,
    val_metric: str = "auprc",
    threshold_metric: str = "balanced_acc",
    ema_decay: Optional[float] = None,
):
    """
    Train while tracking validation accuracy; checkpoint best run.

    If ``checkpoint_on_val_acc_at_best_threshold`` is True, the score matched to
    final evaluation is validation accuracy after a probability threshold sweep
    (same idea as ``find_best_threshold_accuracy``). Otherwise uses accuracy at 0.5.

    Optional EMA weights (``ema_decay``) stabilize generalization; validation and
    the saved checkpoint use the EMA model when enabled.
    """
    model.to(device)
    ema_model: Optional[nn.Module] = None
    if ema_decay is not None:
        ema_model = copy.deepcopy(model)
        ema_model.to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    val_metric = val_metric.strip().lower()
    threshold_metric = threshold_metric.strip().lower()

    best_val_score = -1.0
    counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_score": [],
        "val_metric": [],
        "val_auroc": [],
        "val_auprc": [],
        "val_acc_best": [],
        "val_bacc_best": [],
        "val_f1_best": [],
        "val_best_thr": [],
        "val_acc_05": [],
        "val_bacc_05": [],
        "val_f1_05": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            if isinstance(X_batch, (list, tuple)):
                X_batch = [x.to(device) for x in X_batch]
            else:
                X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            if isinstance(X_batch, list):
                y_pred = model(X_batch[0], X_batch[1])
            else:
                y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_loss += loss.item() * y_batch.size(0)

            if ema_model is not None:
                with torch.no_grad():
                    for p, p_ema in zip(model.parameters(), ema_model.parameters()):
                        p_ema.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)

        train_loss /= len(train_loader.dataset)

        eval_net = ema_model if ema_model is not None else model
        eval_net.eval()
        val_loss = 0.0
        y_true_val = []
        y_prob_val = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if isinstance(X_batch, (list, tuple)):
                    X_batch = [x.to(device) for x in X_batch]
                else:
                    X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                if isinstance(X_batch, list):
                    y_pred = eval_net(X_batch[0], X_batch[1])
                else:
                    y_pred = eval_net(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * y_batch.size(0)
                y_true_val.extend(y_batch.cpu().numpy())
                y_prob_val.extend(torch.sigmoid(y_pred).cpu().numpy())

        val_loss /= len(val_loader.dataset)
        y_true_val = np.array(y_true_val).ravel()
        y_prob_val = np.array(y_prob_val).ravel()
        summary = compute_val_summary(
            y_true_val, y_prob_val, thr_metric=threshold_metric
        )

        if val_metric in ("auprc", "average_precision", "ap"):
            val_score = summary["auprc"]
        elif val_metric in ("auroc", "roc_auc", "auc"):
            val_score = summary["auroc"]
        elif val_metric in ("balanced_acc", "balanced_accuracy", "bacc"):
            val_score = (
                summary["bacc_best"]
                if checkpoint_on_val_acc_at_best_threshold
                else summary["bacc_05"]
            )
        elif val_metric in ("f1", "f1_score"):
            val_score = (
                summary["f1_best"]
                if checkpoint_on_val_acc_at_best_threshold
                else summary["f1_05"]
            )
        elif val_metric in ("acc", "accuracy"):
            val_score = (
                summary["acc_best"]
                if checkpoint_on_val_acc_at_best_threshold
                else summary["acc_05"]
            )
        else:
            raise ValueError(f"Unsupported val_metric: {val_metric}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_score"].append(val_score)
        history["val_metric"].append(val_metric)
        history["val_auroc"].append(summary["auroc"])
        history["val_auprc"].append(summary["auprc"])
        history["val_acc_best"].append(summary["acc_best"])
        history["val_bacc_best"].append(summary["bacc_best"])
        history["val_f1_best"].append(summary["f1_best"])
        history["val_best_thr"].append(summary["best_thr"])
        history["val_acc_05"].append(summary["acc_05"])
        history["val_bacc_05"].append(summary["bacc_05"])
        history["val_f1_05"].append(summary["f1_05"])

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val {val_metric.upper()}: {val_score:.4f} | "
            f"AUROC: {summary['auroc']:.4f} | AUPRC: {summary['auprc']:.4f} | "
            f"BestThr({threshold_metric})={summary['best_thr']:.3f} "
            f"(Acc={summary['acc_best']:.4f}, BAcc={summary['bacc_best']:.4f}, F1={summary['f1_best']:.4f})",
            flush=True,
        )

        if val_score > best_val_score:
            best_val_score = val_score
            counter = 0
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save(eval_net.state_dict(), checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping (val {val_metric})")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, history
