"""Ranking metrics: Recall@K and NDCG@K, plus a top-K scorer."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def recall_at_k(top_k_items: np.ndarray, ground_truth: list[set[int]]) -> float:
    """Mean fraction of ground-truth items that appear in the top-K list per user.

    Parameters
    ----------
    top_k_items : (n_users, k) array of item indices, ranked best-first.
    ground_truth : per-user set of held-out positive item indices.
    """
    if not ground_truth:
        return 0.0
    recalls = []
    for ranked, truth in zip(top_k_items, ground_truth, strict=True):
        if not truth:
            continue
        hits = sum(1 for it in ranked if int(it) in truth)
        recalls.append(hits / len(truth))
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(top_k_items: np.ndarray, ground_truth: list[set[int]]) -> float:
    """Normalized discounted cumulative gain @ K, mean over users with truth.

    Binary relevance: an item is relevant iff it's in the user's truth set.
    """
    if not ground_truth:
        return 0.0
    scores = []
    for ranked, truth in zip(top_k_items, ground_truth, strict=True):
        if not truth:
            continue
        dcg = sum(
            (1.0 / math.log2(rank + 2)) for rank, it in enumerate(ranked) if int(it) in truth
        )
        ideal_hits = min(len(truth), ranked.shape[0])
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def rank_top_k(
    model: nn.Module,
    user_ids: torch.Tensor,
    n_items: int,
    k: int,
    seen: dict[int, set[int]],
    device: torch.device,
    score_batch: int = 1024,
) -> np.ndarray:
    """Score every item for each user in ``user_ids``, mask seen items, return top-K.

    Returns
    -------
    (len(user_ids), k) numpy array of item indices.
    """
    model.eval()
    all_items = torch.arange(n_items, device=device)
    out = np.empty((len(user_ids), k), dtype=np.int64)

    for start in range(0, len(user_ids), score_batch):
        batch_users = user_ids[start : start + score_batch].to(device)
        u_rep = batch_users.unsqueeze(1).expand(-1, n_items).reshape(-1)
        i_rep = all_items.unsqueeze(0).expand(len(batch_users), -1).reshape(-1)
        scores = model(u_rep, i_rep).reshape(len(batch_users), n_items)

        for i, u in enumerate(batch_users.tolist()):
            seen_items = seen.get(u)
            if seen_items:
                scores[i, list(seen_items)] = -float("inf")

        top = torch.topk(scores, k=k, dim=1).indices
        out[start : start + len(batch_users)] = top.cpu().numpy()
    return out
