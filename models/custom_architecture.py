"""
Custom tabular classifier for one-hot encoded features.

Assignment-friendly design: a compact residual MLP (not the baseline SimpleMLP in
mlp.py). Residual connections help optimization on mixed tabular inputs while
keeping the architecture easy to describe and tune.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-activation style MLP block with a single residual skip."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(F.gelu(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return self.drop(F.gelu(x + out))


class CustomTabularNet(nn.Module):
    """
    Stem projection + stacked residual blocks + linear logit head.
    End-to-end from scratch; outputs raw logits for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim too small")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)
