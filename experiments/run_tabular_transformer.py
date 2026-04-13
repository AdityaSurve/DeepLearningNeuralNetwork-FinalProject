import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.tabular_transformer import TabTransformerLite
from src.train_common import (
    get_device,
    train_model_max_val_accuracy,
    evaluate_pytorch_model,
    plot_learning_curves,
    compute_val_summary,
)


def collect_probs(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            y_true.extend(y_batch.cpu().numpy().ravel())
            y_prob.extend(p.tolist())
    return np.array(y_true), np.array(y_prob)


def main():
    print("Loading OHE data for TabTransformerLite...")
    data = np.load("data/processed/data_ohe.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    device = get_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    batch_size = int(os.environ.get("BATCH_SIZE", "128"))
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_features = int(X_train.shape[1])
    d_model = int(os.environ.get("TT_D_MODEL", "64"))
    nhead = int(os.environ.get("TT_NHEAD", "4"))
    num_layers = int(os.environ.get("TT_LAYERS", "2"))
    dropout = float(os.environ.get("TT_DROPOUT", "0.1"))

    model = TabTransformerLite(
        num_features=n_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"TabTransformerLite: n_features={n_features}, d_model={d_model}, "
        f"nhead={nhead}, layers={num_layers}, params={n_params:,}",
        flush=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    max_lr = float(os.environ.get("TT_MAX_LR", "1e-3"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr / 25.0, weight_decay=1e-4
    )
    epochs = int(os.environ.get("TT_EPOCHS", "120"))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    output_dir = "outputs/tabular_transformer"
    os.makedirs("checkpoints/tabular_transformer", exist_ok=True)
    checkpoint_path = "checkpoints/tabular_transformer/best_model.pt"

    composite_alpha = float(os.environ.get("COMPOSITE_ALPHA", "0.5"))
    print(
        f"Training (val/threshold: composite, alpha={composite_alpha})...",
        flush=True,
    )
    best_model, history = train_model_max_val_accuracy(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=epochs,
        patience=12,
        checkpoint_path=checkpoint_path,
        device=device,
        checkpoint_on_val_acc_at_best_threshold=True,
        val_metric="composite",
        threshold_metric="composite",
        ema_decay=0.999,
        composite_alpha=composite_alpha,
    )

    y_val_true, y_val_prob = collect_probs(best_model, val_loader, device)
    summary = compute_val_summary(
        y_val_true,
        y_val_prob,
        thr_metric="composite",
        composite_alpha=composite_alpha,
    )
    thr = summary["best_thr"]
    print(
        f"Validation: best composite thr={thr:.4f} | "
        f"Acc={summary['acc_best']:.4f} BAcc={summary['bacc_best']:.4f}",
        flush=True,
    )

    metrics = evaluate_pytorch_model(
        best_model,
        test_loader,
        output_dir,
        "TabTransformerLite",
        device=device,
        threshold=thr,
    )
    plot_learning_curves(history, output_dir)
    print(
        f"Test ROC_AUC: {metrics['roc_auc']:.4f}, accuracy: {metrics['accuracy']:.4f}, "
        f"balanced_acc: {metrics['balanced_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
