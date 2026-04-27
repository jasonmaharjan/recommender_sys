"""Correctness tests for ranking metrics against hand-computed expected values."""

from __future__ import annotations

import math

import numpy as np

from recsys.eval.metrics import ndcg_at_k, recall_at_k


class TestRecallAtK:
    def test_perfect_ranking(self) -> None:
        # Truth = {0, 1}, top-K places both at the front → recall = 1.0.
        top_k = np.array([[0, 1, 2, 3, 4]])
        assert recall_at_k(top_k, [{0, 1}]) == 1.0

    def test_no_hits(self) -> None:
        top_k = np.array([[5, 6, 7, 8, 9]])
        assert recall_at_k(top_k, [{0, 1, 2}]) == 0.0

    def test_partial_hits(self) -> None:
        # Truth = {0, 1, 2}, top-K has only 0 and 2 → 2/3.
        top_k = np.array([[0, 99, 2, 98, 97]])
        assert recall_at_k(top_k, [{0, 1, 2}]) == 2 / 3

    def test_averages_over_users(self) -> None:
        # User A: 1/2 hits = 0.5; user B: 2/2 hits = 1.0; mean = 0.75.
        top_k = np.array([[0, 99], [10, 11]])
        assert recall_at_k(top_k, [{0, 5}, {10, 11}]) == 0.75

    def test_skips_users_with_empty_truth(self) -> None:
        top_k = np.array([[0, 1], [2, 3]])
        assert recall_at_k(top_k, [{0}, set()]) == 1.0


class TestNdcgAtK:
    def test_perfect_ranking_ndcg_is_one(self) -> None:
        top_k = np.array([[0, 1]])
        assert ndcg_at_k(top_k, [{0, 1}]) == 1.0

    def test_single_relevant_at_first_position(self) -> None:
        # Only one truth item, placed at rank 0: dcg = idcg = 1/log2(2) → ndcg = 1.
        top_k = np.array([[5, 99, 98]])
        assert ndcg_at_k(top_k, [{5}]) == 1.0

    def test_relevant_at_second_position(self) -> None:
        # Truth at rank 1: dcg = 1/log2(3); idcg = 1/log2(2) = 1.
        top_k = np.array([[99, 5]])
        expected = (1.0 / math.log2(3)) / 1.0
        assert math.isclose(ndcg_at_k(top_k, [{5}]), expected, rel_tol=1e-9)

    def test_no_relevant_items_is_zero(self) -> None:
        top_k = np.array([[0, 1, 2]])
        assert ndcg_at_k(top_k, [{99}]) == 0.0
