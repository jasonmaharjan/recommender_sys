"""API endpoint tests with a mocked state — no real model load required."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# Avoid macOS OpenMP collision before any faiss-importing module is touched.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from recsys.api import main as api_module
from recsys.api.main import State


class _FakeModel:
    """Minimal stand-in: returns a fixed unit-vector regardless of user_idx."""

    def user_embedding(self, _: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)


class _FakeIndex:
    """Index that returns the first k contiguous item indices."""

    def search(self, _: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        ids = np.arange(k, dtype=np.int64).reshape(1, k)
        sims = np.full((1, k), 0.42, dtype=np.float32)
        return sims, ids


@pytest.fixture
def client() -> TestClient:
    n_items = 20
    api_module.state = State(
        model=_FakeModel(),  # type: ignore[arg-type]
        index=_FakeIndex(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        user_id_to_idx={1: 0, 2: 1, 42: 2},
        item_idx_to_movie_id={i: 100 + i for i in range(n_items)},
        item_idx_to_title={i: f"Movie {i} (1995)" for i in range(n_items)},
        redis=None,
    )
    return TestClient(api_module.app)


class TestHealth:
    def test_returns_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["n_users"] == 3
        assert body["n_items"] == 20
        assert body["redis_connected"] is False


class TestRecommend:
    def test_known_user_returns_top_k(self, client: TestClient) -> None:
        r = client.get("/recommend/1?k=3")
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == 1
        assert body["k"] == 3
        assert len(body["recommendations"]) == 3
        first = body["recommendations"][0]
        assert {"movie_id", "title", "score"} <= first.keys()
        assert body["cache_hit"] is False
        assert body["latency_ms"] > 0

    def test_default_k_is_ten(self, client: TestClient) -> None:
        r = client.get("/recommend/1")
        assert r.status_code == 200
        assert r.json()["k"] == 10

    def test_unknown_user_returns_404(self, client: TestClient) -> None:
        r = client.get("/recommend/9999")
        assert r.status_code == 404
        assert "not in training set" in r.json()["detail"]

    def test_invalid_k_returns_400(self, client: TestClient) -> None:
        for bad in (0, -1, 101):
            r = client.get(f"/recommend/1?k={bad}")
            assert r.status_code == 400, f"k={bad} should be 400"

    def test_response_includes_titles(self, client: TestClient) -> None:
        r = client.get("/recommend/1?k=2")
        body: dict[str, Any] = r.json()
        titles = [rec["title"] for rec in body["recommendations"]]
        assert any(t.startswith("Movie ") for t in titles)
