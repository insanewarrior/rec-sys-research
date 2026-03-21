# Weighting Is Fun: Implicit Feedback Weighting Strategies for Collaborative Filtering

Experimental codebase for the paper. Evaluates a range of confidence/weighting strategies for implicit-feedback recommender systems across multiple datasets and algorithms.

---

## Repository Structure

```
.
├── src/                        # Python modules
│   ├── weighting_strategies.py # All weighting functions
│   ├── autorec.py              # I-AutoRec model wrapper
│   ├── utils/                  # Shared utilities
│   │   ├── datasets.py         # Dataset loaders (Last.fm, Steam, Taste Profile)
│   │   └── sparse.py           # DataFrame → sparse matrix conversion
│   └── tests/                  # Pytest suite
│       ├── conftest.py
│       ├── test_weighting_strategies.py
│       ├── test_sparse.py
│       └── test_autorec.py
│
├── notebooks/                  # Experiment notebooks (one per dataset × algorithm)
│   ├── 0_lastfm_360k_als.ipynb
│   ├── 0_lastfm_360k_knn.ipynb
│   ├── 1_movielens_20m_als.ipynb
│   ├── 1_movielens_20m_knn.ipynb
│   ├── 1_movielens_20m_lmf.ipynb
│   ├── 1_movielens_20M_autorec.ipynb
│   ├── 2_movielens_100k_als.ipynb
│   ├── 2_movielens_100k_knn.ipynb
│   ├── 2_movielens_100k_lmf.ipynb
│   ├── 2_movielens_100k_autorec.ipynb
│   ├── 3_steam_als.ipynb
│   ├── 3_steam_knn.ipynb
│   ├── 4_taste_profile_als.ipynb
│   ├── 4_taste_profile_knn.ipynb
│   └── weighting_time_benchmark.ipynb
│
├── results/
│   ├── raw/                    # Atomic per-run CSVs (checkpointing, one file per strategy)
│   │   ├── lastfm_360k_als/
│   │   ├── lastfm_360k_knn/
│   │   └── ...
│   └── final/                  # Combined, paper-ready results (one file per experiment)
│       ├── lastfm_360k_als_results.csv
│       ├── lastfm_360k_knn_results.csv
│       └── ...
│
```

---

## Weighting Strategies

All strategies are implemented in [`src/weighting_strategies.py`](src/weighting_strategies.py) and accept a sparse user–item interaction matrix as input.

| Strategy | Description |
|---|---|
| `no_weighting` | Raw interaction counts (baseline) |
| `log` | `log(1 + r)` — dampens large counts |
| `bm25` | BM25 term weighting adapted for implicit feedback |
| `tfidf` | TF-IDF weighting per item |
| `normalized` | Row-normalizes each user vector to unit length |
| `log_idf` | `1 + α · log(1+r) · IDF` — log dampening + item rarity |
| `power` | `r^p` — generalized power scaling |
| `pmi` | Pointwise Mutual Information — penalizes popular items and active users |
| `power_lift` | Generalized PMI without the log: `(r·N / (Σu·Σi))^p` |
| `robust_user_centric` | Per-user sigmoid-normalized Z-score (Median/IQR) |
| `robust_user_centric_v2` | Variant with configurable quantile spread |
| `sigmoid_propensity` | Sigmoid saturation of log-counts combined with IDF boosting |

---

## Algorithms

| Algorithm | Library |
|---|---|
| ALS (Alternating Least Squares) | `implicit` |
| KNN (Item-based) | `implicit` |
| LMF (Logistic Matrix Factorization) | `implicit` |
| I-AutoRec | [`src/autorec.py`](src/autorec.py) (PyTorch) |

---

## Datasets

| Prefix | Dataset | Users | Items | Interactions | Density | Feedback Type | Access |
|---|---|---|---|---|---|---|---|
| `0_` | Last.fm 360K | 359,347 | 294,015 | 17,559,530 | 0.0166% | Play counts | Auto-download |
| `1_` | MovieLens 20M | 138,493 | 26,744 | 20,000,263 | 0.54% | Capped implicit (1–5) | Auto-download (via cornac) |
| `2_` | MovieLens 100K | 943 | 1,682 | 100,000 | 6.30% | Capped implicit (1–5) | Auto-download (via cornac) |
| `3_` | Steam | 13,781,059 | 37,610 | 41,154,773 | 0.0079% | Hours played | Manual — Kaggle |
| `4_` | Million Song Dataset (Taste Profile) | 1,019,318 | 384,546 | 48,373,586 | 0.0123% | Play counts | Manual — registration required |

Raw dataset files are not committed. Place them under `data/` as described below.

### Getting the data

**Last.fm 360K** — downloaded automatically on first use:

```python
from utils.datasets import load_lastfm_360k
df = load_lastfm_360k()          # downloads to data/ if not present
```

Or trigger the download explicitly:

```python
from utils.datasets import download_lastfm_360k
download_lastfm_360k()
```

**MovieLens 20M / 100K** — fetched automatically by [cornac](https://cornac.preferred.ai/) inside the notebooks; no manual step needed.

**Steam** — requires a free Kaggle account:

1. Download `recommendations.csv` from [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
2. Save it as:

```
data/steam/steam_recommendations.csv
```

**Million Song Dataset — Taste Profile** — requires agreeing to the dataset terms:

1. Visit <http://millionsongdataset.com/tasteprofile/> and agree to the terms
2. Download `train_triplets.txt`
3. Save it as:

```
data/taste_profile/train_triplets.txt
```

After placing the files the expected layout is:

```
data/
├── lastfm_360k/
│   └── usersha1-artmbid-artname-plays.tsv   ← auto-downloaded
├── steam/
│   └── steam_recommendations.csv            ← manual
└── taste_profile/
    └── train_triplets.txt                   ← manual
```

---

## Results

Final results are in [`results/final/`](results/final/). Each CSV contains one row per weighting strategy, sorted by Test NDCG@20.

| Experiment | File |
|---|---|
| Last.fm 360K — ALS | [lastfm_360k_als_results.csv](results/final/lastfm_360k_als_results.csv) |
| Last.fm 360K — KNN | [lastfm_360k_knn_results.csv](results/final/lastfm_360k_knn_results.csv) |
| MovieLens 20M — ALS | [movielens_20m_als_results.csv](results/final/movielens_20m_als_results.csv) |
| MovieLens 20M — KNN | [movielens_20m_knn_results.csv](results/final/movielens_20m_knn_results.csv) |
| MovieLens 20M — LMF | [movielens_20m_lmf_results.csv](results/final/movielens_20m_lmf_results.csv) |
| MovieLens 100K — ALS | [movielens_100k_als_results.csv](results/final/movielens_100k_als_results.csv) |
| MovieLens 100K — KNN | [movielens_100k_knn_results.csv](results/final/movielens_100k_knn_results.csv) |
| MovieLens 100K — LMF | [movielens_100k_lmf_results.csv](results/final/movielens_100k_lmf_results.csv) |
| MovieLens 100K — I-AutoRec | [movielens_100k_autorec_results.csv](results/final/movielens_100k_autorec_results.csv) |
| Steam — ALS | [steam_als_results.csv](results/final/steam_als_results.csv) |
| Steam — KNN | [steam_knn_results.csv](results/final/steam_knn_results.csv) |
| Taste Profile — ALS | [taste_profile_als_results.csv](results/final/taste_profile_als_results.csv) |
| Taste Profile — KNN | [taste_profile_knn_results.csv](results/final/taste_profile_knn_results.csv) |
| Weighting time benchmark | [weighting_time_results.csv](results/final/weighting_time_results.csv) |

Intermediate per-run checkpoints are in [`results/raw/`](results/raw/).

---

## Setup

Install with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                   # installs all dependencies from uv.lock
uv sync --extra test      # also installs pytest
```

Or with pip:

```bash
pip install -r <(uv export --no-hashes)          # core deps
pip install -r <(uv export --no-hashes --extra test)  # + pytest
```

> **PyTorch / CUDA:** `pyproject.toml` pulls the default (CPU) torch build. For GPU support install torch separately following the [official instructions](https://pytorch.org/get-started/locally/) before running `uv sync` or `pip install`.

---

## Running Experiments

Notebooks assume Jupyter is launched from the repo root:

```bash
cd weighting_is_fun/
jupyter lab
```

Each notebook is self-contained: it loads data, runs Optuna hyperparameter search per weighting strategy, and writes atomic results to `results/raw/<experiment>/`. A final cell combines them into `results/final/<experiment>_results.csv`.

---

## Tests

```bash
pytest src/tests/ -v
```
