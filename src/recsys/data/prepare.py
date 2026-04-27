"""Download the MovieLens 10M dataset -> clean -> split -> save as parquet
Run ``uv run recsys-prepare`` or ``python -m recsys.data.prepare``
"""

from __future__ import annotations

import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import click
import pandas as pd

log = logging.getLogger("recsys.data")

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
URL = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
ARCHIVE = RAW / "ml-10m.zip"
EXTRACTED = RAW / "ml-10M100K"

MIN_INTERACTIONS = 5
VAL_FRAC = 0.1
TEST_FRAC = 0.1


def download() -> None:
    """Download and extract MovieLens 10M if not already present."""
    ratings_file = EXTRACTED / "ratings.dat"
    if ratings_file.exists():
        log.info("raw files already present at %s", EXTRACTED)
        return

    RAW.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        log.info("downloading %s", URL)
        urllib.request.urlretrieve(URL, ARCHIVE)

    log.info("extracting to %s", RAW)
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(RAW)


def load_ratings() -> pd.DataFrame:
    """Load ratings.dat (userId::movieId::rating::timestamp)."""
    return pd.read_csv(
        EXTRACTED / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "float32", "timestamp": "int64"},
    )


def load_movies() -> pd.DataFrame:
    """Load movies.dat (movieId::title::genres). Latin-1 handles accented titles."""
    return pd.read_csv(
        EXTRACTED / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": "int32", "title": "string", "genres": "string"},
        encoding="latin-1",
    )


def filter_sparse(df: pd.DataFrame, min_count: int = MIN_INTERACTIONS) -> pd.DataFrame:
    """Drop users and items with fewer than ``min_count`` interactions.

    Iterative because removing sparse users can make items sparse and vice versa.
    """
    prev = -1
    while len(df) != prev:
        prev = len(df)
        u_counts = df["user_id"].value_counts()
        i_counts = df["movie_id"].value_counts()
        df = df[
            df["user_id"].isin(u_counts[u_counts >= min_count].index)
            & df["movie_id"].isin(i_counts[i_counts >= min_count].index)
        ]
    return df.reset_index(drop=True)


def reindex(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Map raw ids to contiguous 0..N-1 ids so they can index embedding tables."""
    user_map = {raw: idx for idx, raw in enumerate(sorted(df["user_id"].unique()))}
    item_map = {raw: idx for idx, raw in enumerate(sorted(df["movie_id"].unique()))}
    df = df.assign(
        user_idx=df["user_id"].map(user_map).astype("int32"),
        item_idx=df["movie_id"].map(item_map).astype("int32"),
    )
    return df, user_map, item_map


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by timestamp, then chronologically cut into train / val / test.

    Random splits leak future information; production recsys are evaluated by
    predicting the future from the past.
    """
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - TEST_FRAC))
    val_start = int(n * (1 - TEST_FRAC - VAL_FRAC))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]


def prepare() -> None:
    download()
    ratings = load_ratings()
    movies = load_movies()
    log.info("loaded: %d ratings, %d movies", len(ratings), len(movies))

    ratings = filter_sparse(ratings)
    ratings, user_map, item_map = reindex(ratings)
    log.info(
        "after filtering: %d ratings, %d users, %d items",
        len(ratings),
        len(user_map),
        len(item_map),
    )

    train, val, test = temporal_split(ratings)
    log.info("split: train=%d val=%d test=%d", len(train), len(val), len(test))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED / "train.parquet", index=False)
    val.to_parquet(PROCESSED / "val.parquet", index=False)
    test.to_parquet(PROCESSED / "test.parquet", index=False)
    movies.to_parquet(PROCESSED / "movies.parquet", index=False)

    stats = {
        "n_users": len(user_map),
        "n_items": len(item_map),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
    }
    (PROCESSED / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info("wrote parquet + stats to %s", PROCESSED)


@click.command()
def main() -> None:
    """CLI entry point (registered as ``recsys-prepare`` in pyproject)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare()


if __name__ == "__main__":
    main()
