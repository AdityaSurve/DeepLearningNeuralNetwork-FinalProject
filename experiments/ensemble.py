"""
Ensemble: combine multiple trained models via soft (mean prob + val threshold) or hard vote.

Default: every member below is **included if its artifact exists** (paths can be overridden
with env vars). Set ``ENSEMBLE_MEMBERS`` to a comma-separated subset to restrict.

Ordinal features (``data_ord.npz``): xgboost, lightgbm, catboost, hist_gradient_boosting.
One-hot (``data_ohe.npz``): logistic_elasticnet, tabular_transformer, custom (CustomTabularNet).

- ENSEMBLE_MODE=soft (default): weighted average of member probabilities, threshold on val.
- ENSEMBLE_MODE=hard: per-member threshold votes, then (weighted) majority.
"""
import contextlib
import io
import json
import logging
import os
import sys
import warnings
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.custom_architecture import CustomTabularNet
from models.tabular_transformer import TabTransformerLite
from src.metrics import evaluate_model
from src.train_common import find_best_threshold_by_metric, get_device

# PyTorch TransformerEncoder emits this on first forward (norm_first=True); filter early.
warnings.filterwarnings(
    "ignore",
    message=".*enable_nested_tensor.*",
    category=UserWarning,
)


def _ensemble_mode() -> str:
    m = os.environ.get("ENSEMBLE_MODE", "soft").strip().lower()
    if m in ("vote", "voting", "hard"):
        return "hard"
    if m in ("avg", "average", "mean", "soft"):
        return "soft"
    raise ValueError(
        "ENSEMBLE_MODE must be 'soft' (default) or 'hard' / 'voting'"
    )


def _quiet_lightgbm() -> None:
    for name in ("lightgbm", "LightGBM"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=".*num_leaves.*",
        category=UserWarning,
    )


def _load_ord():
    data = np.load("data/processed/data_ord.npz")
    return (
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


def _load_ohe():
    data = np.load("data/processed/data_ohe.npz")
    return data["X_val"], data["X_test"]


def _tree_probs(clf, X, *, silence_stderr: bool = False):
    """LightGBM often prints native [Warning] lines to stderr on predict — optional mute."""
    if silence_stderr:
        with contextlib.redirect_stderr(io.StringIO()):
            return clf.predict_proba(X)[:, 1].astype(np.float64)
    return clf.predict_proba(X)[:, 1].astype(np.float64)


def _torch_probs(model, X, device, batch_size: int):
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            out.append(p)
    return np.concatenate(out, axis=0).astype(np.float64)


def _parse_weights(s: str, n: int) -> np.ndarray:
    parts = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(parts) != n:
        raise ValueError(
            f"ENSEMBLE_WEIGHTS must have {n} comma-separated floats, got {len(parts)}"
        )
    w = np.array(parts, dtype=np.float64)
    ssum = w.sum()
    if ssum <= 0:
        raise ValueError("ENSEMBLE_WEIGHTS must sum to a positive value")
    return w / ssum


def _hard_vote_preds(
    probs_list: list[np.ndarray],
    weights: np.ndarray,
    vote_threshold: float,
    weighted_majority: bool,
) -> np.ndarray:
    n = len(probs_list)
    votes = np.stack(
        [(p >= vote_threshold).astype(np.float64) for p in probs_list], axis=0
    )
    if weighted_majority:
        s = np.dot(weights, votes)
        return (s >= 0.5).astype(int)
    maj = (n + 1) // 2
    return (votes.sum(axis=0) >= maj).astype(int)


def _normalize_member_ids(raw: Optional[str]) -> Optional[list[str]]:
    """Return ordered list of member ids, or None = use all (auto)."""
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    if s in ("all", "auto", "*", "full"):
        return None
    aliases = {
        "xgb": "xgboost",
        "lgb": "lightgbm",
        "hist": "hist_gradient_boosting",
        "histgb": "hist_gradient_boosting",
        "hgb": "hist_gradient_boosting",
        "elasticnet": "logistic_elasticnet",
        "lr_elasticnet": "logistic_elasticnet",
        "tt": "tabular_transformer",
        "tabtransformer": "tabular_transformer",
    }
    out = []
    for part in raw.split(","):
        k = part.strip().lower()
        if not k:
            continue
        out.append(aliases.get(k, k))
    return out


def _parse_members_env() -> Optional[list[str]]:
    return _normalize_member_ids(os.environ.get("ENSEMBLE_MEMBERS"))


def main():
    _quiet_lightgbm()

    device = get_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    default_bs = 256 if torch.cuda.is_available() else 128
    batch_size = int(os.environ.get("BATCH_SIZE", str(default_bs)))

    member_filter = _parse_members_env()
    allowed: Optional[set[str]] = set(member_filter) if member_filter else None

    thr_metric = os.environ.get("THRESH_METRIC", "balanced_acc").strip().lower()
    composite_alpha = float(os.environ.get("COMPOSITE_ALPHA", "0.5"))

    hidden_dim = int(os.environ.get("CUSTOM_HIDDEN", "256"))
    num_blocks = int(os.environ.get("CUSTOM_BLOCKS", "2"))
    dropout = float(os.environ.get("CUSTOM_DROPOUT", "0.2"))

    custom_tag = os.environ.get(
        "ENSEMBLE_CUSTOM_TAG",
        "custom_architecture_hybrid_mit_both",
    )
    custom_ckpt = os.environ.get(
        "ENSEMBLE_CUSTOM_CKPT",
        f"checkpoints/{custom_tag}/best_model.pt",
    )

    tt_d_model = int(os.environ.get("TT_D_MODEL", "64"))
    tt_nhead = int(os.environ.get("TT_NHEAD", "4"))
    tt_layers = int(os.environ.get("TT_LAYERS", "2"))
    tt_dropout = float(os.environ.get("TT_DROPOUT", "0.1"))
    tt_ckpt = os.environ.get(
        "ENSEMBLE_TABTRANSFORMER_CKPT",
        "checkpoints/tabular_transformer/best_model.pt",
    )

    output_dir = os.environ.get("ENSEMBLE_OUTPUT_DIR", "outputs/ensemble")
    mode = _ensemble_mode()
    vote_threshold = float(os.environ.get("ENSEMBLE_VOTE_THRESHOLD", "0.5"))
    weighted_vote = os.environ.get("ENSEMBLE_HARD_WEIGHTED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    default_name = (
        f"Ensemble hard ({mode})"
        if mode == "hard"
        else "Ensemble soft (multi)"
    )
    model_name = os.environ.get("ENSEMBLE_MODEL_NAME", default_name)

    X_val_ord, y_val, X_test_ord, y_test = _load_ord()
    X_val_ohe, X_test_ohe = _load_ohe()
    n_ohe = int(X_val_ohe.shape[1])

    y_val = np.asarray(y_val).astype(int).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()

    probs_val: list[np.ndarray] = []
    probs_test: list[np.ndarray] = []
    labels: list[str] = []

    def want(mid: str) -> bool:
        return allowed is None or mid in allowed

    # --- ordinal sklearn / xgb / lgb / hist ---
    paths_ord: list[tuple[str, str]] = [
        ("xgboost", os.environ.get("ENSEMBLE_XGB_PATH", "outputs/xgboost/best_model.joblib")),
        ("lightgbm", os.environ.get("ENSEMBLE_LGB_PATH", "outputs/lightgbm/best_model.joblib")),
        (
            "hist_gradient_boosting",
            os.environ.get(
                "ENSEMBLE_HIST_GB_PATH",
                "outputs/hist_gradient_boosting/best_model.joblib",
            ),
        ),
    ]
    for mid, path in paths_ord:
        if not want(mid):
            continue
        if not os.path.isfile(path):
            print(f"Skip {mid}: missing {path}", flush=True)
            continue
        clf = joblib.load(path)
        lgb_stderr = mid == "lightgbm"
        probs_val.append(_tree_probs(clf, X_val_ord, silence_stderr=lgb_stderr))
        probs_test.append(_tree_probs(clf, X_test_ord, silence_stderr=lgb_stderr))
        labels.append(mid)

    # --- catboost (cbm) ---
    if want("catboost"):
        cb_path = os.environ.get(
            "ENSEMBLE_CATBOOST_PATH", "outputs/catboost/catboost_model.cbm"
        )
        if os.path.isfile(cb_path):
            try:
                from catboost import CatBoostClassifier
            except ImportError:
                print(
                    "Skip catboost: catboost not installed (pip install catboost)",
                    flush=True,
                )
            else:
                cb = CatBoostClassifier()
                cb.load_model(cb_path)
                probs_val.append(cb.predict_proba(X_val_ord)[:, 1].astype(np.float64))
                probs_test.append(cb.predict_proba(X_test_ord)[:, 1].astype(np.float64))
                labels.append("catboost")
        else:
            print(f"Skip catboost: missing {cb_path}", flush=True)

    # --- OHE: elastic net logistic ---
    if want("logistic_elasticnet"):
        el_path = os.environ.get(
            "ENSEMBLE_ELASTICNET_PATH",
            "outputs/logistic_elasticnet/best_model.joblib",
        )
        if os.path.isfile(el_path):
            el = joblib.load(el_path)
            probs_val.append(_tree_probs(el, X_val_ohe))
            probs_test.append(_tree_probs(el, X_test_ohe))
            labels.append("logistic_elasticnet")
        else:
            print(f"Skip logistic_elasticnet: missing {el_path}", flush=True)

    # --- OHE: tabular transformer ---
    if want("tabular_transformer") and os.path.isfile(tt_ckpt):
        if tt_d_model % tt_nhead != 0:
            raise ValueError("TT_D_MODEL must be divisible by TT_NHEAD")
        tt = TabTransformerLite(
            num_features=n_ohe,
            d_model=tt_d_model,
            nhead=tt_nhead,
            num_layers=tt_layers,
            dropout=tt_dropout,
        )
        state = torch.load(tt_ckpt, map_location=device)
        tt.load_state_dict(state)
        tt.to(device)
        probs_val.append(_torch_probs(tt, X_val_ohe, device, batch_size))
        probs_test.append(_torch_probs(tt, X_test_ohe, device, batch_size))
        labels.append("tabular_transformer")
    elif want("tabular_transformer"):
        print(f"Skip tabular_transformer: missing {tt_ckpt}", flush=True)

    # --- OHE: custom MLP ---
    if want("custom"):
        if os.path.isfile(custom_ckpt):
            model = CustomTabularNet(
                input_dim=n_ohe,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=dropout,
            )
            state = torch.load(custom_ckpt, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            probs_val.append(_torch_probs(model, X_val_ohe, device, batch_size))
            probs_test.append(_torch_probs(model, X_test_ohe, device, batch_size))
            labels.append(f"custom:{custom_tag}")
        else:
            print(f"Skip custom: missing {custom_ckpt}", flush=True)

    if len(probs_val) < 2:
        raise RuntimeError(
            "Ensemble needs at least two components with existing artifacts. "
            f"Train models or set ENSEMBLE_MEMBERS. Active: {labels}"
        )

    n = len(probs_val)
    if os.environ.get("ENSEMBLE_WEIGHTS"):
        w = _parse_weights(os.environ["ENSEMBLE_WEIGHTS"], n)
    else:
        w = np.ones(n, dtype=np.float64) / n

    P_val = sum(w[i] * probs_val[i] for i in range(n))
    P_test = sum(w[i] * probs_test[i] for i in range(n))

    if mode == "soft":
        best_thr, _ = find_best_threshold_by_metric(
            y_val, P_val, thr_metric, composite_alpha=composite_alpha
        )
        y_pred = (P_test >= best_thr).astype(int)
    else:
        y_pred = _hard_vote_preds(
            probs_test, w, vote_threshold, weighted_vote
        )
        best_thr, _ = find_best_threshold_by_metric(
            y_val, P_val, thr_metric, composite_alpha=composite_alpha
        )

    meta = {
        "ensemble_mode": mode,
        "components": labels,
        "weights": w.tolist(),
        "threshold_metric": thr_metric,
        "composite_alpha": composite_alpha,
        "best_threshold_val_soft_prob": float(best_thr),
        "vote_threshold": float(vote_threshold),
        "hard_weighted_majority": bool(weighted_vote),
        "custom_tag": custom_tag,
        "member_filter": member_filter,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ensemble_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if mode == "soft":
        print(
            f"Ensemble (soft): members={labels}, n={n}, weights={w}, thr_metric={thr_metric}, "
            f"best_thr(val) on mean prob={best_thr:.4f}",
            flush=True,
        )
    else:
        vm = "weighted" if weighted_vote else "majority"
        print(
            f"Ensemble (hard {vm}): members={labels}, n={n}, weights={w}, vote_thr={vote_threshold:.4f} "
            f"(reference: soft-style best_thr on val mean-prob={best_thr:.4f})",
            flush=True,
        )

    metrics = evaluate_model(y_test, P_test, y_pred, output_dir, model_name)
    print(
        f"Test ROC_AUC: {metrics['roc_auc']:.4f}, accuracy: {metrics['accuracy']:.4f}, "
        f"balanced_acc: {metrics['balanced_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
