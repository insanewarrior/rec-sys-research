# It Takes Time

Sequential-recommender benchmark on **MovieLens-1M**, **MovieLens-100K**,
**Amazon Digital Music 5-core**, and **Amazon Office Products 5-core**,
comparing SOTA
baselines (SASRec, BERT4Rec, GRU4Rec, NARM, FPMC, Pop, BPR, ItemKNN) against
three new **IA-SASRec** (Intensity-Aware SASRec) variants that inject
per-interaction strength (rating) directly into the self-attention
mechanism. One reproducible methodology, **Optuna** HPO, on-disk
resumability, 27 pytest tests.

## Why

Sequential recommenders are routinely benchmarked under inconsistent splits
and hyperparameter budgets, which makes paper-to-paper comparison fragile.
This project's goal is a single notebook where:

- every model sees the same chronological leave-one-out split,
- every model gets the same Optuna budget on the same metric (NDCG@10),
- you can interrupt a long run and resume — partial results survive on disk,
- new SASRec / BERT4Rec variants drop into `src/models/variants/` and join the
  comparison table by adding one entry to a registry,
- the same `.inter` file feeds every model — baselines that don't read the
  intensity column simply ignore it, keeping the comparison apples-to-apples.

## IA-SASRec — what's new

`vanilla SASRec` treats every history item as a binary presence signal.
**IA-SASRec** keeps the per-interaction intensity (rating, hours played,
dwell time) and threads it into the attention computation through one of
three drop-in modifications:

| Variant            | Mechanism                                                   | Extra params |
|--------------------|-------------------------------------------------------------|--------------|
| `IA-SASRec-Add`    | additive logit bias `softmax(QKᵀ/√d + λ·M_W) V`             | 1 scalar `λ` per layer (learnable) |
| `IA-SASRec-Mul`    | multiplicative scaling `softmax((QKᵀ/√d) ⊙ M_W) V`          | none |
| `IA-SASRec-Val`    | value modulation `softmax(QKᵀ/√d) (V ⊙ w)`                  | none |

Full math, motivation, and paper outline: **[ia_sasrec.md](ia_sasrec.md)**.

## Layout

```
it_takes_time/
├── pyproject.toml
├── ia_sasrec.md                # theory + implementation reference for IA-SASRec
├── sasrec_advances.md          # original brainstorm / design doc
├── src/
│   ├── config.py               # paths, dataset registry (ml-1m, ml-100k, amazon-digital-music, amazon-office-products), HPO knobs
│   ├── data.py                 # downloads + writes 4-column .inter (user, item, ts, intensity)
│   ├── hpo.py                  # Optuna study, persisted to SQLite
│   ├── runner.py               # train + eval + invalidate_cache, resumable via results/eval/*.json
│   ├── evaluation.py           # aggregate results table, top-K recommend helper
│   └── models/
│       ├── __init__.py         # MODEL_REGISTRY, search spaces
│       └── variants/
│           └── ia_sasrec.py    # IASASRecBase + Add/Mul/Val + IAMultiHeadAttention
├── tests/                      # 27 pytest tests (math, registry, runner, data)
└── notebooks/
    └── 0_benchmark.ipynb   # loops over all datasets × full registry
```

## Setup

```bash
cd it_takes_time
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

All four datasets download automatically over plain HTTP — no credentials
required.

GPU optional. CPU is fine for ML-100K and Amazon 5-core (~minutes per
model); ML-1M with SASRec ~5 min/run on a modern laptop.

## Run

Open `notebooks/0_benchmark.ipynb` and run top-to-bottom. It loops over
`["ml-1m", "ml-100k", "amazon-digital-music", "amazon-office-products"]` ×
the full model registry (including the three IA-SASRec variants).

Knobs (env vars):

| Var                   | Default | Effect                                  |
|-----------------------|---------|------------------------------------------|
| `N_TRIALS`            | 10      | Optuna trials per model                  |
| `HPO_EPOCHS`          | 10      | Epochs per HPO trial (short)             |
| `FINAL_EPOCHS`        | 50      | Epochs for the final fit on best params  |
| `EARLY_STOP_PATIENCE` | 5       | Early-stopping patience on val NDCG@10   |

### One-time: re-run baselines after schema change

The `.inter` file now carries a fourth column (`intensity:float`). Cached
baseline results from before this change must be regenerated for the
comparison to remain valid:

```bash
python -c "from runner import invalidate_cache; invalidate_cache('ml-100k')"
```

Then re-run the notebook — the baselines retrain on the new file, IA-SASRec
variants train fresh.

## Tests

```bash
pytest -q
```

27 tests covering: intensity normalisation modes, attention math for each
variant (`Add` at `λ=0` matches vanilla bit-for-bit; `Mul` zero-intensity
collapses to uniform; `Val` leaves the probability distribution untouched),
padding-mask correctness across variants, full forward+backward smoke,
registry wiring, two RecBole gotchas (class-object vs. name in `Config`;
inlined SASRec yaml defaults), and data-pipeline format checks.

## Resumability

- `results/eval/<dataset>__<model>.json` — final metrics. Presence of this
  file means "skip this model on next notebook run".
- `results/hpo/<dataset>__<model>.db` — Optuna SQLite study. Re-running an
  HPO call continues from the last trial.
- `results/checkpoints/` — RecBole-saved best-model `.pth` files.

To re-run a single model from scratch:
`from runner import invalidate_cache; invalidate_cache("ml-100k", "SASRec")`,
optionally also delete its HPO DB, then re-run the benchmark cell.

## Adding a custom variant

See section 7 of the notebook, or use IA-SASRec as a worked example:

1. Subclass a RecBole model under `src/models/variants/`.
2. Append an entry to `MODEL_REGISTRY` in `src/models/__init__.py`. If the
   subclass relies on SASRec's internal yaml defaults (`hidden_act`, etc.),
   inline them in the `static` dict — RecBole doesn't load them
   automatically for non-built-in classes.
3. Re-run the benchmark cell. Cached results for other models are reused.

## Datasets

| Key                       | Source                                                  | Intensity signal | Notes                                          |
|---------------------------|---------------------------------------------------------|------------------|------------------------------------------------|
| `ml-1m`                   | GroupLens HTTP zip                                      | Rating 1–5       | Native unix timestamps                         |
| `ml-100k`                 | GroupLens HTTP zip                                      | Rating 1–5       | Native unix timestamps                         |
| `amazon-digital-music`    | snap.stanford.edu JSON-gz (McAuley 2014 5-core)         | Rating 1–5       | ~5.5k users × ~3.6k items × ~64k reviews       |
| `amazon-office-products`  | snap.stanford.edu JSON-gz (McAuley 2014 5-core)         | Rating 1–5       | ~4.9k users × ~2.4k items × ~53k reviews       |

To add another: append an entry to `DATASETS` in `src/config.py` with its
download spec, intensity column, and any pre-filters — `data.py` handles
the rest of the atomic-file conversion.
