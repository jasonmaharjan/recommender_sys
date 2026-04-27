"""FastAPI service: top-K movie recommendations from the trained two-tower model."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

# macOS faiss/torch share OpenMP; loading order can SIGSEGV without this flag.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recsys.data.prepare import PROCESSED, ROOT
from recsys.models.two_tower import TwoTower

faiss.omp_set_num_threads(1)
log = logging.getLogger("recsys.api")
MODELS_DIR = ROOT / "models"
DEFAULT_K = 10
MAX_K = 100
CACHE_TTL_SECONDS = 300


@dataclass
class State:
    model: TwoTower
    index: faiss.Index
    device: torch.device
    user_id_to_idx: dict[int, int]
    item_idx_to_movie_id: dict[int, int]
    item_idx_to_title: dict[int, str]
    redis: object | None = None  # redis.asyncio.Redis when available


state = State.__new__(State)


class Recommendation(BaseModel):
    movie_id: int = Field(description="Original MovieLens movieId")
    title: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    k: int
    recommendations: list[Recommendation]
    cache_hit: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_users: int
    n_items: int
    redis_connected: bool


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_state() -> State:
    device = _pick_device()

    ckpt_path = MODELS_DIR / "two_tower.pt"
    if not ckpt_path.exists():
        msg = f"checkpoint missing: {ckpt_path}; run `uv run recsys-train-tt` first"
        raise FileNotFoundError(msg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg, stats = ckpt["config"], ckpt["stats"]

    model = TwoTower(
        n_users=stats["n_users"],
        n_items=stats["n_items"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        out_dim=cfg["out_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    index = faiss.read_index(str(MODELS_DIR / "items.faiss"))

    movies_df = pd.read_parquet(PROCESSED / "movies.parquet", columns=["movie_id", "title"])
    cols = ["user_id", "user_idx", "movie_id", "item_idx"]
    all_splits = pd.concat(
        [pd.read_parquet(PROCESSED / f"{s}.parquet", columns=cols) for s in ("train", "val", "test")],
        ignore_index=True,
    )
    users_unique = all_splits.drop_duplicates(subset=["user_id"])
    items_unique = all_splits.drop_duplicates(subset=["movie_id"])
    user_id_to_idx = dict(
        zip(users_unique["user_id"].astype(int), users_unique["user_idx"].astype(int), strict=True)
    )
    item_idx_to_movie_id = dict(
        zip(items_unique["item_idx"].astype(int), items_unique["movie_id"].astype(int), strict=True)
    )
    title_lookup = dict(zip(movies_df["movie_id"].astype(int), movies_df["title"].astype(str), strict=True))
    item_idx_to_title = {idx: title_lookup.get(mid, "<unknown>") for idx, mid in item_idx_to_movie_id.items()}

    log.info("loaded: device=%s users=%d items=%d", device, stats["n_users"], stats["n_items"])

    return State(
        model=model,
        index=index,
        device=device,
        user_id_to_idx=user_id_to_idx,
        item_idx_to_movie_id=item_idx_to_movie_id,
        item_idx_to_title=item_idx_to_title,
    )


async def _connect_redis() -> object | None:
    url = os.environ.get("REDIS_URL")
    if not url:
        log.info("REDIS_URL not set; running without cache")
        return None
    try:
        from redis.asyncio import Redis

        client = Redis.from_url(url, decode_responses=True)
        await client.ping()
        log.info("redis connected: %s", url)
        return client
    except Exception as e:
        log.warning("redis unavailable (%s); running without cache", e)
        return None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global state
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    state = _load_state()
    state.redis = await _connect_redis()
    yield
    if state.redis is not None:
        await state.redis.close()  # type: ignore[attr-defined]


app = FastAPI(title="recsys", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=hasattr(state, "model"),
        n_users=len(state.user_id_to_idx),
        n_items=len(state.item_idx_to_movie_id),
        redis_connected=state.redis is not None,
    )


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(user_id: int, k: int = DEFAULT_K) -> RecommendResponse:
    if not 1 <= k <= MAX_K:
        raise HTTPException(status_code=400, detail=f"k must be in [1, {MAX_K}]")

    user_idx = state.user_id_to_idx.get(user_id)
    if user_idx is None:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not in training set")

    cache_key = f"rec:{user_id}:{k}"
    if state.redis is not None:
        t0 = time.perf_counter()
        cached = await state.redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            payload = json.loads(cached)
            payload["latency_ms"] = (time.perf_counter() - t0) * 1000
            return RecommendResponse(**payload, cache_hit=True)

    t0 = time.perf_counter()
    with torch.no_grad():
        user_t = torch.tensor([user_idx], device=state.device)
        user_emb = state.model.user_embedding(user_t).cpu().numpy()
    sims, top_ids = state.index.search(user_emb, k)
    latency_ms = (time.perf_counter() - t0) * 1000

    recs = [
        Recommendation(
            movie_id=state.item_idx_to_movie_id[int(idx)],
            title=state.item_idx_to_title[int(idx)],
            score=float(sim),
        )
        for idx, sim in zip(top_ids[0], sims[0], strict=True)
        if int(idx) >= 0
    ]
    payload = {
        "user_id": user_id,
        "k": k,
        "recommendations": [r.model_dump() for r in recs],
        "latency_ms": latency_ms,
    }
    if state.redis is not None:
        await state.redis.setex(  # type: ignore[attr-defined]
            cache_key, CACHE_TTL_SECONDS, json.dumps(payload)
        )
    return RecommendResponse(**payload, cache_hit=False)
