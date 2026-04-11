import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models.custom_architecture import CustomTabularNet
from src.bias_mitigation import build_instance_weights
from src.train_common import (
    get_device,
    train_model_max_val_accuracy,
    evaluate_pytorch_model,
    plot_learning_curves,
    compute_val_summary,
)


def _is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def _dataloader_kwargs() -> dict:
    """Kaggle T4: set NUM_WORKERS=2 (default on Kaggle); local CPU/GPU often use 0."""
    default_workers = "2" if _is_kaggle() else "0"
    num_workers = int(os.environ.get("NUM_WORKERS", default_workers))
    use_cuda = torch.cuda.is_available()
    kw: dict = {"num_workers": num_workers, "pin_memory": use_cuda}
    if num_workers > 0:
        kw["persistent_workers"] = True
    return kw


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
    print("Loading OHE data for Custom Architecture...")
    data = np.load("data/processed/data_ohe.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    device = get_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # Unweighted BCE + threshold tuning favors overall accuracy (vs heavy pos_weight → ~72% acc).
    # On Kaggle T4 (~16 GB), BATCH_SIZE=4096–8192 is usually fine; override via env.
    default_bs = 4096 if torch.cuda.is_available() else 2048
    batch_size = int(os.environ.get("BATCH_SIZE", str(default_bs)))
    dl_kw = _dataloader_kwargs()
    print(
        f"DataLoader: batch_size={batch_size}, num_workers={dl_kw['num_workers']}, "
        f"pin_memory={dl_kw['pin_memory']}",
        flush=True,
    )
    pos_rate = float(np.mean(y_train))
    baseline_acc = max(pos_rate, 1.0 - pos_rate)
    print(
        f"Train pos_rate={pos_rate:.4f} (baseline acc ~{baseline_acc:.4f} by predicting all-majority)",
        flush=True,
    )

    # Kamiran–Calders-style reweighing (+ optional class balance) as weighted loss.
    mitigation = os.environ.get("BIAS_MITIGATION", "both").strip().lower()
    if mitigation not in ("none", "reweigh", "class", "both"):
        raise ValueError(
            "BIAS_MITIGATION must be one of: none, reweigh, class, both"
        )
    raw_train_path = "data/processed/X_train_raw.csv"
    prot_cols = [
        c.strip()
        for c in os.environ.get("PROTECTED_ATTRS", "Sex,AgeCategory").split(",")
        if c.strip()
    ]
    eff_mitigation = mitigation
    prot_df = None
    if eff_mitigation in ("reweigh", "both"):
        if not os.path.isfile(raw_train_path):
            if eff_mitigation == "reweigh":
                raise FileNotFoundError(
                    f"{raw_train_path} not found. Run src/preprocess.py to write "
                    "raw train rows for demographic reweighing."
                )
            print(
                "BIAS_MITIGATION=both but X_train_raw missing — using class-only "
                "weights. Re-run preprocess to enable reweighing.",
                flush=True,
            )
            eff_mitigation = "class"
        else:
            prot_full = pd.read_csv(raw_train_path)
            avail = [c for c in prot_cols if c in prot_full.columns]
            if not avail:
                if eff_mitigation == "reweigh":
                    raise ValueError(
                        f"No protected columns {prot_cols} in {raw_train_path}"
                    )
                print(
                    "BIAS_MITIGATION=both but no protected columns — class-only weights.",
                    flush=True,
                )
                eff_mitigation = "class"
            else:
                prot_df = prot_full[avail]

    train_weights = None
    mit_desc = "none"
    if eff_mitigation != "none":
        w_np, mit_desc = build_instance_weights(
            prot_df if eff_mitigation in ("reweigh", "both") else None,
            y_train,
            eff_mitigation,
            protected_columns=list(prot_df.columns) if prot_df is not None else None,
        )
        train_weights = torch.tensor(w_np, dtype=torch.float32).unsqueeze(1)
        print(
            f"Bias mitigation: requested={mitigation} effective={eff_mitigation} "
            f"({mit_desc}), mean_weight=1.0",
            flush=True,
        )

    if train_weights is not None:
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            train_weights,
        )
    else:
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

    # Objective profile:
    # - balanced: better minority recall/balanced accuracy (fairness-oriented)
    # - accuracy: better plain accuracy (majority-oriented)
    objective_mode = os.environ.get("OBJECTIVE_MODE", "hybrid").strip().lower()
    if objective_mode not in ("balanced", "accuracy", "hybrid"):
        raise ValueError(
            "OBJECTIVE_MODE must be 'balanced', 'accuracy', or 'hybrid'"
        )

    # hybrid: high overall accuracy and strong per-class behavior via composite
    # val/threshold objective (alpha * acc + (1-alpha) * balanced_accuracy).
    if objective_mode == "balanced":
        default_balanced_sampler = "1"
        default_use_pos_weight = "1"
        default_val_metric = "auprc"
        default_thr_metric = "balanced_acc"
    elif objective_mode == "accuracy":
        default_balanced_sampler = "0"
        default_use_pos_weight = "0"
        default_val_metric = "acc"
        default_thr_metric = "acc"
    else:
        default_balanced_sampler = "0"
        default_use_pos_weight = "0"
        default_val_metric = "composite"
        default_thr_metric = "composite"

    # Balanced sampling vs instance weights: both up-weight rare (Y) cases; using
    # both is usually redundant with BIAS_MITIGATION=class/both.
    use_balanced_sampler = os.environ.get("BALANCED_SAMPLER", default_balanced_sampler).strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if train_weights is not None and use_balanced_sampler:
        print(
            "Note: BALANCED_SAMPLER=1 with instance weights — consider BALANCED_SAMPLER=0 "
            "unless you intentionally want extra minority oversampling.",
            flush=True,
        )
    if use_balanced_sampler:
        y_train_i = y_train.astype(int)
        class_count = np.bincount(y_train_i)
        class_weight = 1.0 / np.maximum(class_count, 1)
        sample_weight = class_weight[y_train_i]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weight, dtype=torch.double),
            num_samples=len(sample_weight),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, **dl_kw
        )
        print("Training sampler: balanced (WeightedRandomSampler)", flush=True)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **dl_kw
        )
        print("Training sampler: shuffled", flush=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **dl_kw
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **dl_kw
    )

    input_dim = int(X_train.shape[1])
    model = CustomTabularNet(
        input_dim=input_dim,
        n_tokens=10,
        d_model=320,
        n_heads=8,
        n_layers=5,
        dim_ff=1280,
        dropout=0.1,
        num_skip_blocks=4,
    )

    # For imbalanced targets: pos_weight > 1 improves minority detection,
    # but can reduce plain accuracy when that's the primary objective.
    use_pos_weight = os.environ.get("USE_POS_WEIGHT", default_use_pos_weight).strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if use_pos_weight:
        num_pos = float(y_train.sum())
        num_neg = float(len(y_train) - y_train.sum())
        pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: BCEWithLogitsLoss(pos_weight={pos_weight.item():.2f})", flush=True)
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss(unweighted)", flush=True)
    max_lr = 2e-3
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr / 25.0, weight_decay=2e-4
    )
    epochs = 55
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # Write separate artifacts per mode (and mitigation profile).
    model_tag = f"custom_architecture_{objective_mode}"
    if mitigation != "none":
        model_tag += f"_mit_{eff_mitigation}"
    output_dir = f"outputs/{model_tag}"
    os.makedirs(f"checkpoints/{model_tag}", exist_ok=True)
    checkpoint_path = f"checkpoints/{model_tag}/best_model.pt"

    use_ema = os.environ.get("USE_EMA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    ema_decay = 0.999 if use_ema else None
    val_metric = os.environ.get("VAL_METRIC", default_val_metric).strip().lower()
    thr_metric = os.environ.get("THRESH_METRIC", default_thr_metric).strip().lower()
    composite_alpha = float(os.environ.get("COMPOSITE_ALPHA", "0.5"))
    print(
        "Training Custom Architecture "
        f"(mode={objective_mode}, checkpoint val_metric={val_metric}, thr_metric={thr_metric}, "
        f"composite_alpha={composite_alpha}, "
        f"OneCycleLR, EMA={'on' if use_ema else 'off'})..."
    )
    best_model, history = train_model_max_val_accuracy(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=epochs,
        patience=18,
        checkpoint_path=checkpoint_path,
        device=device,
        checkpoint_on_val_acc_at_best_threshold=True,
        val_metric=val_metric,
        threshold_metric=thr_metric,
        ema_decay=ema_decay,
        composite_alpha=composite_alpha,
    )

    y_val_true, y_val_prob = collect_probs(best_model, val_loader, device)
    summary = compute_val_summary(
        y_val_true,
        y_val_prob,
        thr_metric=thr_metric,
        composite_alpha=composite_alpha,
    )
    thr = summary["best_thr"]
    print(
        f"Validation summary: AUROC={summary['auroc']:.4f}, AUPRC={summary['auprc']:.4f}, "
        f"BestThr({thr_metric})={thr:.4f} -> Acc={summary['acc_best']:.4f}, "
        f"BAcc={summary['bacc_best']:.4f}, F1={summary['f1_best']:.4f}",
        flush=True,
    )

    metrics = evaluate_pytorch_model(
        best_model,
        test_loader,
        output_dir,
        model_tag,
        device=device,
        threshold=thr,
    )
    plot_learning_curves(history, output_dir)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Test ROC_AUC: {metrics['roc_auc']:.4f}")
    print(
        f"Per-class recall (0/1): {metrics['recall_class_0']:.4f} / "
        f"{metrics['recall_class_1']:.4f}"
    )


if __name__ == "__main__":
    main()
