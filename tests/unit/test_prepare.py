"""Unit tests for data-prep pure functions (no network, no real dataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.data.prepare import filter_sparse, reindex, temporal_split


def _synth(
    n_users: int = 20, n_items: int = 30, n_per_user: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Build a small ratings DataFrame where every user has `n_per_user` distinct items."""
    assert n_per_user <= n_items, "need enough items to sample without replacement"
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        items = rng.choice(n_items, size=n_per_user, replace=False)
        for i, it in enumerate(items):
            rows.append((u, int(it), float(rng.integers(1, 6)), u * 1000 + i))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


class TestFilterSparse:
    def test_drops_low_count_user(self) -> None:
        # Users 1..5 each rate the same 5 items; user 6 rates only one of them.
        # With min_count=2, user 6 should be dropped; all items survive (≥5 ratings each).
        rows = [(u, i, 5.0, 0) for u in range(1, 6) for i in range(10, 15)]
        rows.append((6, 10, 5.0, 0))
        df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
        out = filter_sparse(df, min_count=2)
        assert 6 not in out["user_id"].to_numpy()
        assert set(out["user_id"].unique()) == {1, 2, 3, 4, 5}

    def test_drops_low_count_item(self) -> None:
        # Item 10 has 5 ratings; item 99 has only 1. With min_count=2, item 99 is dropped.
        rows = [(u, 10, 5.0, 0) for u in range(1, 6)]
        rows.append((1, 99, 5.0, 0))
        df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
        out = filter_sparse(df, min_count=2)
        assert 99 not in out["movie_id"].to_numpy()

    def test_converges_when_removals_cascade(self) -> None:
        # Users 1 and 2 each rate items 10..14 (5 items, survives min_count=5).
        # User 3 rates item 10 plus four singleton items 90..93.
        # Pass 1: items 90..93 dropped (only 1 rating each).
        # Pass 2: user 3 now has only 1 rating (item 10), dropped.
        rows = [(u, i, 5.0, 0) for u in (1, 2) for i in range(10, 15)]
        rows.append((3, 10, 5.0, 0))
        rows += [(3, i, 5.0, 0) for i in range(90, 94)]
        df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
        out = filter_sparse(df, min_count=2)
        assert set(out["user_id"].unique()) == {1, 2}
        assert set(out["movie_id"].unique()) == {10, 11, 12, 13, 14}


class TestReindex:
    def test_ids_are_contiguous_from_zero(self) -> None:
        df = _synth(n_users=10, n_items=20, n_per_user=8)
        out, umap, imap = reindex(df)
        assert sorted(out["user_idx"].unique().tolist()) == list(range(len(umap)))
        assert sorted(out["item_idx"].unique().tolist()) == list(range(len(imap)))

    def test_mapping_is_injective(self) -> None:
        df = _synth()
        _, umap, imap = reindex(df)
        assert len(set(umap.values())) == len(umap)
        assert len(set(imap.values())) == len(imap)

    def test_preserves_row_count(self) -> None:
        df = _synth()
        out, _, _ = reindex(df)
        assert len(out) == len(df)


class TestTemporalSplit:
    def test_no_leakage_across_splits(self) -> None:
        df = _synth(n_users=50, n_items=40, n_per_user=10)
        train, val, test = temporal_split(df)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_fractions_roughly_match(self) -> None:
        df = _synth(n_users=100, n_items=60, n_per_user=10)
        train, val, test = temporal_split(df)
        total = len(df)
        assert abs(len(val) / total - 0.1) < 0.02
        assert abs(len(test) / total - 0.1) < 0.02
        assert len(train) + len(val) + len(test) == total

    def test_empty_input_does_not_crash(self) -> None:
        empty = pd.DataFrame(
            {"user_id": [], "movie_id": [], "rating": [], "timestamp": []}
        ).astype({"user_id": "int32", "movie_id": "int32", "rating": "float32", "timestamp": "int64"})
        train, val, test = temporal_split(empty)
        assert len(train) == len(val) == len(test) == 0
