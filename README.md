# recsys

Movie recommender system: two-tower retrieval + FAISS ANN + FastAPI serving, trained on MovieLens 10M.

## Setup

Requires `uv` (install from https://astral.sh/uv) and Python 3.12 (uv handles it).

```bash
uv sync --all-extras
```

## Usage

```bash
uv run recsys-prepare      # download MovieLens 10M, clean, temporal split, save parquet
uv run recsys-train        # train MF baseline (RMSE + Recall@K + NDCG@K)
uv run recsys-train-tt     # train two-tower retrieval, build FAISS index
```

Outputs land in `data/processed/` (splits) and `models/` (checkpoints + metrics + FAISS index).

## Results

Trained on MovieLens 10M (69,878 users × 10,196 items, 80/10/10 chronological split). Ranking eval: 5,000 sampled users, ground truth = test interactions with rating ≥ 4.0.

| Model | Test RMSE | Recall@10 | NDCG@10 | Top-K latency (5k users) |
|---|---|---|---|---|
| Matrix Factorization | **0.930** | 0.0374 | **0.2087** | 7.3s (full forward) |
| Two-Tower + FAISS | n/a (ranking only) | 0.0356 | 0.1515 | **0.2s (ANN)** |

Two-tower trades a small offline-metric gap for ~36× faster retrieval; at billion-item scale this becomes the only viable architecture.

## Development

```bash
uv run pytest              # run tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy src            # type-check
```

## Project layout

```
src/recsys/
├── data/        # dataset download, cleaning, splitting
├── models/      # matrix factorization, two-tower (planned)
├── retrieval/   # FAISS ANN index (planned)
├── ranking/     # re-ranking layer (planned)
├── eval/        # offline metrics: RMSE, Recall@K, NDCG@K (planned)
└── api/         # FastAPI serving (planned)
```
