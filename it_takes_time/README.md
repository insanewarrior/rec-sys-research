# It Takes Time

Temporal collaborative-filtering benchmark on MovieLens-1M: SASRec, BERT4Rec,
GRU4Rec, NARM, FPMC, plus non-sequential baselines (Pop, BPR, ItemKNN), all
under one reproducible methodology with **Optuna** HPO and **on-disk
resumability**.

## Why

Sequential recommenders are routinely benchmarked under inconsistent splits and
hyperparameter budgets, which makes paper-to-paper comparison fragile. This
project's goal is a single notebook where:

- every model sees the same chronological leave-one-out split,
- every model gets the same Optuna budget on the same metric (NDCG@10),
- you can interrupt a long run and resume — partial results survive on disk,
- new variants of SASRec / BERT4Rec / etc. drop into `src/models/variants/`
  and join the comparison table by adding one entry to a registry.

## Layout

```
it_takes_time/
├── pyproject.toml
├── src/
│   ├── config.py        # paths, dataset registry, HPO knobs
│   ├── data.py          # ML-1M download + RecBole atomic-file conversion
│   ├── hpo.py           # Optuna study, persisted to SQLite
│   ├── runner.py        # train + eval, resumable via results/eval/*.json
│   ├── evaluation.py    # aggregate results table, top-K recommend helper
│   └── models/          # MODEL_REGISTRY + variants/ for custom papers
└── notebooks/
    └── 0_movielens_1m_benchmark.ipynb
```

## Setup

```bash
cd it_takes_time
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

(GPU optional. CPU works for ML-1M; SASRec ~5 min/run on a modern laptop.)

## Run

Open `notebooks/0_movielens_1m_benchmark.ipynb` and run top-to-bottom.

Knobs (env vars):

| Var                   | Default | Effect                                  |
|-----------------------|---------|------------------------------------------|
| `N_TRIALS`            | 5       | Optuna trials per model                  |
| `HPO_EPOCHS`          | 10      | Epochs per HPO trial (short)             |
| `FINAL_EPOCHS`        | 50      | Epochs for the final fit on best params  |
| `EARLY_STOP_PATIENCE` | 5       | Early-stopping patience on val NDCG@10   |

## Resumability

- `results/eval/<dataset>__<model>.json` — final metrics. Presence of this file
  means "skip this model".
- `results/hpo/<dataset>__<model>.db` — Optuna SQLite study. Re-running an HPO
  call continues from the last trial.
- `results/checkpoints/` — RecBole-saved best-model `.pth` files.

To re-run a single model from scratch: delete its eval JSON (and optionally its
HPO DB) and re-run the benchmark cell.

## Adding a custom variant

See section 7 of the notebook. TL;DR: subclass a RecBole model under
`src/models/variants/`, append an entry to `MODEL_REGISTRY` in
`src/models/__init__.py`, re-run.

## Switching dataset

Edit `DATASET` in the notebook (currently `'ml-1m'`). `ml-100k` is preconfigured
in `src/config.py:DATASETS`. To add another: append an entry there with its URL,
filename, separator, and column layout — `data.py` will handle the rest.
