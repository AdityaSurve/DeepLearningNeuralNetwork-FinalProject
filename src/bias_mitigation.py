"""
Bias mitigation for training (Kamiran & Calders reweighing + optional class balance).

Reweighing matches the idea used in IBM AI Fairness 360's Reweighing: assign each
training instance a weight w ∝ P(A)*P(Y)/P(A,Y) for joint protected attributes A,
so the weighted training distribution has (approximately) independent A and Y.

Combined with class-balance weights, optimization pays more attention to rare labels
and to underrepresented (A,Y) cells — improving minority-class and subgroup recall
when disparities stem from skewed sampling.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_mean_one(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w)
    return w * (len(w) / s)


def kamiran_calders_reweighing_weights(
    protected: pd.DataFrame, y: np.ndarray
) -> np.ndarray:
    """
    Instance weights for independence of joint protected attributes and label Y.

    w_i = P(A=a_i) P(Y=y_i) / P(A=a_i, Y=y_i)  (then scaled so mean weight is 1).
    """
    n = len(y)
    if len(protected) != n:
        raise ValueError("protected rows must match y length")
    y = np.asarray(y).astype(int).ravel()
    df = protected.reset_index(drop=True).fillna("__NA__").astype(str)
    key_a = df.apply(lambda r: "\x1f".join(r.astype(str).values), axis=1)
    tab = pd.crosstab(key_a, y)
    count_a = tab.sum(axis=1)
    count_y = tab.sum(axis=0)

    w = np.ones(n, dtype=np.float64)
    for i in range(n):
        a = key_a.iloc[i]
        yi = int(y[i])
        if yi not in tab.columns:
            continue
        n_ay = float(tab.loc[a, yi])
        if n_ay <= 0:
            w[i] = 1.0
            continue
        ca = float(count_a[a])
        cy = float(count_y[yi])
        w[i] = (ca / n) * (cy / n) / (n_ay / n)
    return _normalize_mean_one(w)


def class_balance_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights per class; normalized to mean 1."""
    y = np.asarray(y).astype(int).ravel()
    counts = np.bincount(y)
    inv = 1.0 / np.maximum(counts[y].astype(np.float64), 1.0)
    return _normalize_mean_one(inv)


def build_instance_weights(
    protected: Optional[pd.DataFrame],
    y: np.ndarray,
    mode: str,
    protected_columns: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, str]:
    """
    Returns (weights per training row, short description for logging).
    """
    mode = str(mode).strip().lower()
    if mode not in ("none", "reweigh", "class", "both"):
        raise ValueError(
            "mode must be 'none', 'reweigh', 'class', or 'both'"
        )
    if mode == "none":
        return np.ones(len(y), dtype=np.float64), "none"

    y = np.asarray(y).astype(int).ravel()
    n = len(y)
    w = np.ones(n, dtype=np.float64)
    parts: List[str] = []

    if mode in ("reweigh", "both"):
        if protected is None or protected.empty:
            raise ValueError("reweighing requires protected attribute columns")
        cols = list(protected_columns) if protected_columns is not None else list(protected.columns)
        missing = [c for c in cols if c not in protected.columns]
        if missing:
            raise ValueError(f"Protected columns not in dataframe: {missing}")
        sub = protected[cols]
        w_rw = kamiran_calders_reweighing_weights(sub, y)
        w *= w_rw
        parts.append("reweigh")

    if mode in ("class", "both"):
        w_cb = class_balance_weights(y)
        w *= w_cb
        parts.append("class_balance")

    w = _normalize_mean_one(w)
    return w, "+".join(parts)
