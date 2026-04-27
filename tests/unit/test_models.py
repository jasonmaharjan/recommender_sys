"""Forward-shape and output-range tests for MF and Two-Tower models."""

from __future__ import annotations

import pytest
import torch

from recsys.models.mf import MatrixFactorization
from recsys.models.two_tower import TwoTower


@pytest.fixture
def small_dims() -> tuple[int, int]:
    return 50, 30  # n_users, n_items


class TestMatrixFactorization:
    def test_forward_shape(self, small_dims: tuple[int, int]) -> None:
        n_users, n_items = small_dims
        model = MatrixFactorization(n_users, n_items, embedding_dim=8)
        u = torch.randint(0, n_users, (16,))
        i = torch.randint(0, n_items, (16,))
        out = model(u, i)
        assert out.shape == (16,)

    def test_output_in_rating_range(self, small_dims: tuple[int, int]) -> None:
        n_users, n_items = small_dims
        model = MatrixFactorization(n_users, n_items, rating_min=0.5, rating_max=5.0)
        u = torch.randint(0, n_users, (32,))
        i = torch.randint(0, n_items, (32,))
        out = model(u, i)
        assert (out >= 0.5).all() and (out <= 5.0).all()

    def test_training_step_reduces_loss(self, small_dims: tuple[int, int]) -> None:
        n_users, n_items = small_dims
        torch.manual_seed(0)
        model = MatrixFactorization(n_users, n_items, embedding_dim=16)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        u = torch.randint(0, n_users, (256,))
        i = torch.randint(0, n_items, (256,))
        target = torch.full((256,), 4.0)

        loss_before = ((model(u, i) - target) ** 2).mean().item()
        for _ in range(20):
            loss = ((model(u, i) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        loss_after = ((model(u, i) - target) ** 2).mean().item()
        assert loss_after < loss_before * 0.5


class TestTwoTower:
    def test_forward_shape(self, small_dims: tuple[int, int]) -> None:
        n_users, n_items = small_dims
        model = TwoTower(n_users, n_items, embedding_dim=8, hidden_dim=16, out_dim=8)
        u = torch.randint(0, n_users, (16,))
        i = torch.randint(0, n_items, (16,))
        assert model(u, i).shape == (16,)

    def test_embeddings_are_l2_normalized(self, small_dims: tuple[int, int]) -> None:
        n_users, n_items = small_dims
        model = TwoTower(n_users, n_items, embedding_dim=8, hidden_dim=16, out_dim=8)
        ids = torch.arange(n_users)
        embs = model.user_embedding(ids)
        norms = embs.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_user_and_item_towers_are_independent(self, small_dims: tuple[int, int]) -> None:
        n_users, n_items = small_dims
        model = TwoTower(n_users, n_items, embedding_dim=8, hidden_dim=16, out_dim=8)
        # Same id passed to user vs item tower should generally yield different embeddings.
        ids = torch.arange(min(n_users, n_items))
        ue = model.user_embedding(ids)
        ie = model.item_embedding(ids)
        assert not torch.allclose(ue, ie)
