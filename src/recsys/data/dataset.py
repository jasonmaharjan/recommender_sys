"""PyTorch Dataset over the prepared parquet splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from recsys.data.prepare import PROCESSED


@dataclass(frozen=True)
class Stats:
    n_users: int
    n_items: int
    n_train: int
    n_val: int
    n_test: int
    rating_min: float
    rating_max: float


def load_stats(processed_dir: Path = PROCESSED) -> Stats:
    """Read the stats.json written by the prepare pipeline."""
    return Stats(**json.loads((processed_dir / "stats.json").read_text()))


class RatingsDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """(user_idx, item_idx, rating) triples, fully loaded in memory."""

    def __init__(self, parquet_path: Path) -> None:
        df = pd.read_parquet(parquet_path, columns=["user_idx", "item_idx", "rating"])
        self.users = torch.from_numpy(df["user_idx"].to_numpy().copy()).long()
        self.items = torch.from_numpy(df["item_idx"].to_numpy().copy()).long()
        self.ratings = torch.from_numpy(df["rating"].to_numpy().copy()).float()

    def __len__(self) -> int:
        return self.ratings.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]


def load_splits(
    processed_dir: Path = PROCESSED,
) -> tuple[RatingsDataset, RatingsDataset, RatingsDataset]:
    """Return (train, val, test) datasets."""
    return (
        RatingsDataset(processed_dir / "train.parquet"),
        RatingsDataset(processed_dir / "val.parquet"),
        RatingsDataset(processed_dir / "test.parquet"),
    )
