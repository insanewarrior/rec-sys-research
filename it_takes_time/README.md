# It Takes Time

Sequential-recommender benchmark on **MovieLens-1M**, **MovieLens-100K**,
**Amazon Digital Music 5-core**, **Amazon Office Products 5-core**, and
**Steam Reviews** (3k / 8k / 15k user subsamples, hours-played as intensity),
comparing SOTA
baselines (SASRec, BERT4Rec, GRU4Rec, NARM, FPMC, Pop, BPR, ItemKNN) against
three new **IA-SASRec** (Intensity-Aware SASRec) variants that inject
per-interaction strength (rating or hours played) directly into the
self-attention mechanism. One reproducible methodology, **Optuna** HPO,
on-disk resumability, full pytest coverage.

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

| Variant            | Mechanism                                                                    | Extra params |
|--------------------|------------------------------------------------------------------------------|--------------|
| `IA-SASRec-Add`    | additive logit bias `softmax(QKᵀ/√d + λ·w_k) V`                              | 1 scalar `λ` per layer (learnable) |
| `IA-SASRec-Mul`    | multiplicative scaling `softmax((QKᵀ/√d) · (1 + λ·(w_k − 1))) V`             | 1 scalar `λ` per layer (learnable) |
| `IA-SASRec-Val`    | post-softmax key reweighting `(softmax(QKᵀ/√d) · (1 + λ·(w_k − 1))) V`       | 1 scalar `λ` per layer (learnable) |

All three variants are now gated by a learnable strength `λ` (initialised to
1.0). At `λ = 0` every variant collapses to vanilla SASRec, giving the model
an explicit escape hatch when intensity is uninformative; the learned `λ` per
layer is recorded alongside metrics in `results/eval/*.json`.

`Val`'s new semantics (post-softmax key reweighting) is a **breaking change**
from earlier versions, where Val multiplied the post-attention context by
query-position intensity — that formulation collapsed to a per-user scalar at
prediction time and could not reorder candidates. See [ia_sasrec.md](ia_sasrec.md)
for the design rationale.

The `minmax` intensity normalisation has a `MINMAX_FLOOR = 0.1`: the
least-intense real item maps to `0.1` rather than `0.0`, so it stays
distinguishable from padding. `log1p_minmax` and `zscore` are unchanged.

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
├── tests/                      # pytest suite (math, registry, runner, data, multi-seed, significance)
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
`["ml-1m", "ml-100k", "amazon-digital-music", "amazon-office-products",
"steam-3k", "steam-8k", "steam-15k"]` × the full model registry (including
the three IA-SASRec variants). A companion notebook
`notebooks/1_significance.ipynb` runs paired significance tests across the
per-seed results produced by the main benchmark.

Knobs (env vars):

| Var                   | Default | Effect                                  |
|-----------------------|---------|------------------------------------------|
| `N_TRIALS`            | 25      | Optuna trials per model                  |
| `HPO_EPOCHS`          | 10      | Epochs per HPO trial (short)             |
| `FINAL_EPOCHS`        | 50      | Epochs for the final fit on best params  |
| `EARLY_STOP_PATIENCE` | 5       | Early-stopping patience on val NDCG@10   |

## Evaluation protocol

Every model follows the same train / valid / test discipline:

- **HPO trials** (`HPO_EPOCHS` epochs, `saved=False`) train on train, score on
  valid; Optuna's objective is validation NDCG@10. Test data is never touched.
  HPO runs once per (dataset, model) at a single seed.
- **Final fit** (`FINAL_EPOCHS` epochs, `saved=True`) uses the HPO-best params,
  trains on train only, and evaluates on valid each epoch for
  best-checkpoint selection and early stopping. Repeated across N seeds
  per (dataset, model) — see *Multi-seed final fit* below.
- **Test metrics** come from a single pass on test using the
  best-on-valid checkpoint (`load_best_model=True`). Valid_data is never
  added to training — we do not refit on `train ∪ valid` because (a) it would
  forfeit the early-stopping signal, (b) it matches the convention of the
  sequential-recsys literature (SASRec, BERT4Rec, RecBole benchmarks), and
  (c) the leave-one-out valid split is too small to materially improve fit.

So the `best_valid_score` field in `results/eval/*.json` is the
model-selection score; the `test_result` block is the reported number.

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

Tests cover: intensity normalisation modes (including the `MINMAX_FLOOR`
guard), attention math for each variant (`Add`/`Mul`/`Val` at `λ = 0` match
vanilla bit-for-bit; `Mul` at `λ = 1` with zero intensity collapses to
uniform; `Val` post-softmax key reweighting actually shifts the attention
distribution), padding-mask correctness across variants, full
forward+backward smoke with `λ` gradients for all three variants,
`get_intensity_params()` shape, registry wiring, two RecBole gotchas
(class-object vs. name in `Config`; inlined SASRec yaml defaults), and
data-pipeline format checks.

## Resumability

- `results/eval/<dataset>__<model>__seed<N>.json` — per-seed final metrics.
  Presence of all configured seed files for a (dataset, model) means "skip
  this model on next notebook run".
  - Legacy single-file results `<dataset>__<model>.json` from before the
    multi-seed schema are still read transparently and treated as the
    `LEGACY_SEED = 2020` result.
- `results/hpo/<dataset>__<model>.db` — Optuna SQLite study. Re-running an
  HPO call continues from the last trial. HPO is single-seed by design.
- `results/checkpoints/` — RecBole-saved best-model `.pth` files.

To re-run a single model from scratch:
`from runner import invalidate_cache; invalidate_cache("ml-100k", "SASRec")`
(this clears *all* per-seed files for that model), optionally also delete
its HPO DB, then re-run the benchmark cell.

### Multi-seed final fit

The final fit is repeated across multiple seeds per (dataset, model) so we
can report mean ± std and run paired significance tests. HPO is *not*
repeated — it stays single-seed for cost reasons and to avoid overfitting
hparams to seed noise.

Configured in [src/config.py](src/config.py):

```python
DEFAULT_SEEDS = [2020, 2021, 2022, 2023, 2024]
SEEDS_PER_DATASET = {
    "ml-1m": 5,
    "ml-100k": 5,
    "amazon-digital-music": 5,
    "amazon-office-products": 5,
    "steam-3k": 5,
    "steam-8k": 5,
    "steam-15k": 5,
}
SEEDS_PER_MODEL_DATASET = {}          # optional per-(ds, model) override
```

Each value can be either an int N (use the first N entries of
`DEFAULT_SEEDS`) or an explicit list of seed integers.
`SEEDS_PER_MODEL_DATASET[(ds, model)]` overrides the dataset default for a
single (dataset, model) pair — useful when you want one specific model
fitted to extra seeds without re-running the whole grid. `seeds_for(ds,
model)` is the resolver used by the runner.

The notebook reads `result["test_result"]["ndcg@10"]` for the mean across
seeds (unchanged shape), and additionally `result["test_result_std"][...]`,
`result["seeds_used"]`, `result["n_seeds"]`, and `result["per_seed"]` for
the full per-seed breakdown.

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
| `ml-100k`             | GroupLens HTTP zip                                      | Rating 1–5       | Native unix timestamps                         |
| `amazon-digital-music`    | snap.stanford.edu JSON-gz (McAuley 2014 5-core)         | Rating 1–5       | ~5.5k users × ~3.6k items × ~64k reviews       |
| `amazon-office-products`  | snap.stanford.edu JSON-gz (McAuley 2014 5-core)         | Rating 1–5       | ~4.9k users × ~2.4k items × ~53k reviews       |
| `steam-3k`                | cseweb.ucsd.edu Steam Reviews JSON-gz                   | Hours played     | 3 000-user subsample (seed 2020); `%Y-%m-%d` timestamps |
| `steam-8k`                | cseweb.ucsd.edu Steam Reviews JSON-gz                   | Hours played     | 8 000-user subsample (seed 2020)               |
| `steam-15k`               | cseweb.ucsd.edu Steam Reviews JSON-gz                   | Hours played     | 15 000-user subsample (seed 2020)              |

To add another: append an entry to `DATASETS` in `src/config.py` with its
download spec, intensity column, and any pre-filters — `data.py` handles
the rest of the atomic-file conversion.
