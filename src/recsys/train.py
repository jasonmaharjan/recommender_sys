"""Train the matrix factorization baseline on MovieLens 10M."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys.data.dataset import load_splits, load_stats
from recsys.data.prepare import PROCESSED, ROOT
from recsys.eval.metrics import ndcg_at_k, rank_top_k, recall_at_k
from recsys.models.mf import MatrixFactorization

log = logging.getLogger("recsys.train")

MODELS_DIR = ROOT / "models"


@dataclass
class Config:
    embedding_dim: int = 64
    batch_size: int = 4096
    epochs: int = 10
    lr: float = 5e-3
    weight_decay: float = 1e-5
    num_workers: int = 2
    seed: int = 42
    rank_eval_users: int = 5000
    rank_eval_k: int = 10
    pos_threshold: float = 4.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rmse(model: MatrixFactorization, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    sq_err, n = 0.0, 0
    with torch.no_grad():
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            sq_err += torch.sum((pred - r) ** 2).item()
            n += r.numel()
    return math.sqrt(sq_err / n)


def train(cfg: Config) -> dict[str, float | list[float]]:
    torch.manual_seed(cfg.seed)
    device = pick_device()
    log.info("device=%s config=%s", device, cfg)

    stats = load_stats()
    train_ds, val_ds, test_ds = load_splits()
    log.info("datasets: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )
    eval_kwargs = dict(batch_size=cfg.batch_size * 4, shuffle=False, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, **eval_kwargs)
    test_loader = DataLoader(test_ds, **eval_kwargs)

    model = MatrixFactorization(
        n_users=stats.n_users,
        n_items=stats.n_items,
        embedding_dim=cfg.embedding_dim,
        rating_min=stats.rating_min,
        rating_max=stats.rating_max,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    history: list[float] = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "mf.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        sq_err, n = 0.0, 0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            loss = torch.mean((pred - r) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sq_err += torch.sum((pred.detach() - r) ** 2).item()
            n += r.numel()
        train_rmse = math.sqrt(sq_err / n)
        val_rmse = rmse(model, val_loader, device)
        history.append(val_rmse)
        log.info(
            "epoch=%d train_rmse=%.4f val_rmse=%.4f elapsed=%.1fs",
            epoch, train_rmse, val_rmse, time.perf_counter() - t0,
        )
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "stats": asdict(stats)}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test_rmse = rmse(model, test_loader, device)
    log.info("best_val_rmse=%.4f @ epoch %d | test_rmse=%.4f", best_val, best_epoch, test_rmse)

    rank_metrics = ranking_eval(model, cfg, stats.n_items, device)

    metrics = {
        "model": "matrix_factorization",
        "best_epoch": best_epoch,
        "best_val_rmse": best_val,
        "test_rmse": test_rmse,
        "val_rmse_per_epoch": history,
        "config": asdict(cfg),
        **rank_metrics,
    }
    (MODELS_DIR / "mf_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def ranking_eval(
    model: MatrixFactorization, cfg: Config, n_items: int, device: torch.device
) -> dict[str, float]:
    """Sample users, score every item, exclude train-seen, score Recall@K + NDCG@K."""
    log.info("ranking eval: building seen/truth maps")
    train_df = pd.read_parquet(PROCESSED / "train.parquet", columns=["user_idx", "item_idx"])
    test_df = pd.read_parquet(PROCESSED / "test.parquet", columns=["user_idx", "item_idx", "rating"])
    test_df = test_df[test_df["rating"] >= cfg.pos_threshold]

    seen: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy(), strict=True):
        seen[int(u)].add(int(i))

    truth_map: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(test_df["user_idx"].to_numpy(), test_df["item_idx"].to_numpy(), strict=True):
        truth_map[int(u)].add(int(i))

    eligible = np.array(sorted(truth_map.keys()))
    rng = np.random.default_rng(cfg.seed)
    sample_n = min(cfg.rank_eval_users, len(eligible))
    sampled = rng.choice(eligible, size=sample_n, replace=False)
    user_ids = torch.from_numpy(sampled).long()
    truth_lists = [truth_map[int(u)] for u in sampled]

    log.info("ranking eval: scoring %d users x %d items @ K=%d", sample_n, n_items, cfg.rank_eval_k)
    t0 = time.perf_counter()
    top_k = rank_top_k(model, user_ids, n_items, cfg.rank_eval_k, seen, device)
    rec = recall_at_k(top_k, truth_lists)
    ndcg = ndcg_at_k(top_k, truth_lists)
    log.info(
        "recall@%d=%.4f ndcg@%d=%.4f (n=%d, %.1fs)",
        cfg.rank_eval_k, rec, cfg.rank_eval_k, ndcg, sample_n, time.perf_counter() - t0,
    )
    return {f"recall@{cfg.rank_eval_k}": rec, f"ndcg@{cfg.rank_eval_k}": ndcg, "rank_eval_n": sample_n}


@click.command()
@click.option("--epochs", default=10, show_default=True)
@click.option("--batch-size", default=4096, show_default=True)
@click.option("--lr", default=5e-3, show_default=True)
@click.option("--embedding-dim", default=64, show_default=True)
def main(epochs: int, batch_size: int, lr: float, embedding_dim: int) -> None:
    """Train the MF baseline (registered as ``recsys-train`` in pyproject)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    train(Config(epochs=epochs, batch_size=batch_size, lr=lr, embedding_dim=embedding_dim))


if __name__ == "__main__":
    main()
