"""
Lightweight tabular encoder: shared linear tokenizer per scalar feature + CLS +
TransformerEncoder (FT-Transformer-style, simplified).
"""

import torch
import torch.nn as nn


class TabTransformerLite(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.num_features = num_features
        self.input_proj = nn.Linear(1, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_features)
        b, f = x.shape
        tok = self.input_proj(x.unsqueeze(-1))
        cls = self.cls.expand(b, -1, -1)
        h = torch.cat([cls, tok], dim=1)
        h = self.encoder(h)
        h = self.norm(h[:, 0])
        return self.head(h)
