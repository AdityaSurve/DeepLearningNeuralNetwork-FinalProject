import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFeatureFusion(nn.Module):
    """Custom gated residual on a vector (used on the skip pathway)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x) * self.gate(x)


class ResNetBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.bn1(F.gelu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = self.dropout(x)
        return F.gelu(x + residual)


class TabularTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer block (often more stable for deeper stacks on tabular data)."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, need_weights=False
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class CustomTabularNet(nn.Module):
    """
    Custom tabular network: reshape flat OHE vector into learned tokens,
    stack transformer-style blocks for cross-feature interaction, fuse with
    a gated residual MLP skip path, then classify.

    All weights are trained end-to-end from scratch (no pretrained backbones).
    """

    def __init__(
        self,
        input_dim: int,
        n_tokens: int = 10,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.12,
        num_skip_blocks: int = 3,
    ):
        super().__init__()
        if input_dim % n_tokens != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by n_tokens ({n_tokens})"
            )
        token_in = input_dim // n_tokens
        self.n_tokens = n_tokens
        self.d_model = d_model

        self.token_embed = nn.Linear(token_in, d_model)
        # CLS at index 0 + one position per feature token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_tokens + 1, d_model) * 0.02
        )

        self.token_in_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [
                TabularTransformerEncoderLayer(
                    d_model, n_heads, dim_ff, dropout
                )
                for _ in range(n_layers)
            ]
        )

        # Skip: parallel deep path on full vector for stable gradient + accuracy
        self.skip_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip_trunk = nn.ModuleList(
            [ResNetBlock(d_model, dropout) for _ in range(num_skip_blocks)]
        )
        self.skip_fuse = GatedFeatureFusion(d_model)

        fused_dim = d_model * 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tok = x.view(b, self.n_tokens, -1)
        tok = self.token_embed(tok)
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        seq = seq + self.pos_embedding
        seq = self.token_in_norm(seq)
        for layer in self.layers:
            seq = layer(seq)

        cls_out = seq[:, 0, :]
        rest = seq[:, 1:, :]
        pooled_mean = rest.mean(dim=1)
        pooled_max, _ = rest.max(dim=1)
        token_summary = cls_out + (pooled_mean + pooled_max) * 0.25

        skip = self.skip_proj(x)
        for blk in self.skip_trunk:
            skip = blk(skip)
        skip = self.skip_fuse(skip)

        fused = torch.cat([token_summary, skip], dim=1)
        return self.head(fused)
