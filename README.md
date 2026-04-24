# recsys

Movie recommender system: two-tower retrieval + FAISS ANN + FastAPI serving, trained on MovieLens 10M.

## Setup

Requires `uv` (install from https://astral.sh/uv) and Python 3.12 (uv handles it).

```bash
uv sync --all-extras
```

## Usage

Download MovieLens 10M, clean, temporally split, and save as parquet:

```bash
uv run recsys-prepare
```

Outputs land in `data/processed/`:

- `train.parquet`, `val.parquet`, `test.parquet` — 80 / 10 / 10 chronological split
- `movies.parquet` — movie metadata (title, genres)
- `stats.json` — user/item counts, rating range

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
