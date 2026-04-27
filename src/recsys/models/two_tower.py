"""Two-tower retrieval model: user/item towers + L2-normalized dot product."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Tower(nn.Module):
    """Embedding -> MLP -> L2-normalized output."""

    def __init__(self, n_ids: int, embedding_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_ids, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.emb.weight, std=0.01)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = F.normalize(self.mlp(self.emb(ids)), p=2, dim=-1)
        return out


class TwoTower(nn.Module):
    """Dot-product retrieval model. Compatible with FAISS inner-product search."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.user_tower = Tower(n_users, embedding_dim, hidden_dim, out_dim)
        self.item_tower = Tower(n_items, embedding_dim, hidden_dim, out_dim)

    def user_embedding(self, user_idx: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.user_tower(user_idx)
        return emb

    def item_embedding(self, item_idx: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.item_tower(item_idx)
        return emb

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        score: torch.Tensor = (self.user_embedding(user_idx) * self.item_embedding(item_idx)).sum(dim=-1)
        return score
