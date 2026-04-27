"""Matrix factorization baseline: user/item embeddings + biases + squashed output."""

from __future__ import annotations

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    """Biased MF: rating ≈ sigmoid(u·i + b_u + b_i + b_0) scaled to [r_min, r_max].

    The sigmoid squash keeps predictions inside the valid rating range, which
    speeds convergence and empirically improves RMSE vs the unclipped form.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        rating_min: float = 0.5,
        rating_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.rating_min = rating_min
        self.rating_range = rating_max - rating_min

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        dot = (u * i).sum(dim=-1)
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        logit = dot + bu + bi + self.global_bias
        return self.rating_min + self.rating_range * torch.sigmoid(logit)
