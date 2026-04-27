"""Train the two-tower retrieval model with in-batch softmax + FAISS eval."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

# macOS faiss/torch share OpenMP; loading order can SIGSEGV without this flag.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import click
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

faiss.omp_set_num_threads(1)  # avoid OpenMP collision with torch on macOS

from recsys.data.dataset import PositivesDataset, load_stats
from recsys.data.prepare import PROCESSED, ROOT
from recsys.eval.metrics import ndcg_at_k, recall_at_k
from recsys.models.two_tower import TwoTower

log = logging.getLogger("recsys.train_tt")
MODELS_DIR = ROOT / "models"


@dataclass
class Config:
    embedding_dim: int = 64
    hidden_dim: int = 128
    out_dim: int = 64
    batch_size: int = 4096
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-6
    temperature: float = 0.07
    pos_threshold: float = 4.0
    num_workers: int = 2
    seed: int = 42
    rank_eval_users: int = 5000
    rank_eval_k: int = 10


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def in_batch_softmax_loss(user_emb: torch.Tensor, item_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """Treat each in-batch positive as the only positive vs every other item as negative."""
    logits = user_emb @ item_emb.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def encode_all_items(model: TwoTower, n_items: int, device: torch.device, batch: int = 8192) -> np.ndarray:
    model.eval()
    out_dim = int(model.item_tower.mlp[-1].out_features)  # type: ignore[arg-type]
    out = np.empty((n_items, out_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_items, batch):
            ids = torch.arange(start, min(start + batch, n_items), device=device)
            out[start : start + ids.shape[0]] = model.item_embedding(ids).cpu().numpy()
    return out


def faiss_eval(
    model: TwoTower, cfg: Config, n_items: int, device: torch.device
) -> tuple[dict[str, float], faiss.IndexFlatIP]:
    """Build FAISS index over item embeddings, sample test users, score top-K retrieval."""
    log.info("encoding items + building FAISS index")
    item_embs = encode_all_items(model, n_items, device)
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)

    train_df = pd.read_parquet(PROCESSED / "train.parquet", columns=["user_idx", "item_idx"])
    test_df = pd.read_parquet(PROCESSED / "test.parquet", columns=["user_idx", "item_idx", "rating"])
    test_df = test_df[test_df["rating"] >= cfg.pos_threshold]

    seen: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy(), strict=True):
        seen[int(u)].add(int(i))
    truth: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(test_df["user_idx"].to_numpy(), test_df["item_idx"].to_numpy(), strict=True):
        truth[int(u)].add(int(i))

    eligible = np.array(sorted(truth.keys()))
    rng = np.random.default_rng(cfg.seed)
    sample_n = min(cfg.rank_eval_users, len(eligible))
    sampled = rng.choice(eligible, size=sample_n, replace=False)
    truth_lists = [truth[int(u)] for u in sampled]

    log.info("FAISS retrieval: %d users x %d items @ K=%d", sample_n, n_items, cfg.rank_eval_k)
    t0 = time.perf_counter()
    user_ids = torch.from_numpy(sampled).long().to(device)
    with torch.no_grad():
        user_embs = model.user_embedding(user_ids).cpu().numpy()

    # Over-fetch then mask train-seen items so we still return K unseen results.
    over_k = cfg.rank_eval_k + max(50, max((len(seen.get(int(u), [])) for u in sampled), default=0) // 50)
    _, top_ids = index.search(user_embs, over_k)
    top_k = np.empty((sample_n, cfg.rank_eval_k), dtype=np.int64)
    for row, u in enumerate(sampled.tolist()):
        seen_u = seen.get(int(u), set())
        filtered = [i for i in top_ids[row] if int(i) not in seen_u][: cfg.rank_eval_k]
        top_k[row] = filtered + [-1] * (cfg.rank_eval_k - len(filtered))

    rec = recall_at_k(top_k, truth_lists)
    ndcg = ndcg_at_k(top_k, truth_lists)
    log.info(
        "recall@%d=%.4f ndcg@%d=%.4f (n=%d, %.1fs)",
        cfg.rank_eval_k,
        rec,
        cfg.rank_eval_k,
        ndcg,
        sample_n,
        time.perf_counter() - t0,
    )
    return {f"recall@{cfg.rank_eval_k}": rec, f"ndcg@{cfg.rank_eval_k}": ndcg, "rank_eval_n": sample_n}, index


def train(cfg: Config) -> dict[str, object]:
    torch.manual_seed(cfg.seed)
    device = pick_device()
    log.info("device=%s config=%s", device, cfg)

    stats = load_stats()
    train_ds = PositivesDataset(PROCESSED / "train.parquet", threshold=cfg.pos_threshold)
    log.info("positives: train=%d (rating>=%.1f)", len(train_ds), cfg.pos_threshold)

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = TwoTower(
        n_users=stats.n_users,
        n_items=stats.n_items,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[float] = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "two_tower.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, steps = 0.0, 0
        for u, i in loader:
            u, i = u.to(device), i.to(device)
            user_emb = model.user_embedding(u)
            item_emb = model.item_embedding(i)
            loss = in_batch_softmax_loss(user_emb, item_emb, cfg.temperature)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            running += loss.item()
            steps += 1
        avg_loss = running / steps
        history.append(avg_loss)
        log.info("epoch=%d loss=%.4f elapsed=%.1fs", epoch, avg_loss, time.perf_counter() - t0)

    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "stats": asdict(stats)}, ckpt_path)
    rank_metrics, index = faiss_eval(model, cfg, stats.n_items, device)
    faiss.write_index(index, str(MODELS_DIR / "items.faiss"))

    metrics = {
        "model": "two_tower",
        "loss_per_epoch": history,
        "config": asdict(cfg),
        **rank_metrics,
    }
    (MODELS_DIR / "two_tower_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


@click.command()
@click.option("--epochs", default=10, show_default=True)
@click.option("--batch-size", default=4096, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--temperature", default=0.07, show_default=True)
def main(epochs: int, batch_size: int, lr: float, temperature: float) -> None:
    """Train the two-tower model (registered as ``recsys-train-tt``)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(Config(epochs=epochs, batch_size=batch_size, lr=lr, temperature=temperature))


if __name__ == "__main__":
    main()
