"""Project-wide configuration: paths, dataset registry, HPO/eval knobs.

Importing this module also installs the NumPy 2.x compatibility shim that
RecBole needs (see ``_numpy_compat.py``). Every other ``src/`` module imports
``config`` before ``recbole``, so this is the single point of patch install.
"""

from __future__ import annotations

import _numpy_compat  # noqa: F401  -- must precede any recbole import
import _cudnn_compat  # noqa: F401  -- must precede `import torch` to win the dlopen race

import os
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RECBOLE_DATA_DIR = PROJECT_ROOT / "recbole_data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
HPO_DIR = RESULTS_DIR / "hpo"
EVAL_DIR = RESULTS_DIR / "eval"

for _d in (DATA_DIR, RECBOLE_DATA_DIR, CHECKPOINT_DIR, HPO_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

TOP_K = [10, 20, 50, 100]
PRIMARY_METRIC = "ndcg@10"

N_TRIALS = int(os.environ.get("N_TRIALS", "25"))
HPO_EPOCHS = int(os.environ.get("HPO_EPOCHS", "10"))
FINAL_EPOCHS = int(os.environ.get("FINAL_EPOCHS", "50"))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "5"))


# ─── Multi-seed final-fit configuration ────────────────────────────────────
#
# HPO runs once at a single seed (cheap and deterministic); the *final fit*
# is repeated across N seeds with the HPO-found best_params so we can
# measure variance and run paired significance tests. Each seed produces
# its own JSON under EVAL_DIR; aggregation (mean/std across seeds) happens
# at read time via runner.load_result / evaluation.aggregate_results.
#
# Seed list resolution for a (dataset, model) pair, in priority order:
#   1. SEEDS_PER_MODEL_DATASET[(dataset, model)] — most specific
#   2. SEEDS_PER_DATASET[dataset]
#   3. DEFAULT_SEEDS[:1]
# A value can be either an int N (use the first N entries of DEFAULT_SEEDS)
# or an explicit list of ints.
DEFAULT_SEEDS: list[int] = [2020, 2021, 2022, 2023, 2024]
SEEDS_PER_DATASET: dict[str, int | list[int]] = {
    "ml-1m": 4,
    "ml-100k-iar": 5,
    "amazon-digital-music": 5,
    "amazon-office-products": 5,
    "steam-3k": 5,
    "steam-8k": 5,
}
# Per-(dataset, model) override. Empty by default; populate when you need to
# run a specific model at a non-default seed count (e.g. ml-1m IA-SASRec-Add
# at 5 seeds while other ml-1m models stay at 3).
SEEDS_PER_MODEL_DATASET: dict[tuple[str, str], int | list[int]] = {}

# Seed used by RecBole before this multi-seed schema existed. Eval JSONs
# saved without a `__seed<N>` suffix correspond to this seed.
LEGACY_SEED: int = 2020


def seeds_for(dataset_name: str, model_name: str) -> list[int]:
    """Resolve the seed list for a (dataset, model) pair.

    Parameters:
        dataset_name: Dataset key (e.g. ``"ml-1m"``).
        model_name: Model key as used in ``MODEL_REGISTRY``.

    Returns:
        Concrete list of seed integers to fit the final model under, ordered.
    """
    spec: int | list[int] | None = SEEDS_PER_MODEL_DATASET.get((dataset_name, model_name))
    if spec is None:
        spec = SEEDS_PER_DATASET.get(dataset_name, 1)
    if isinstance(spec, int):
        if spec < 1:
            raise ValueError(f"Seed count must be >= 1, got {spec}")
        if spec > len(DEFAULT_SEEDS):
            raise ValueError(
                f"Asked for {spec} seeds but DEFAULT_SEEDS only has "
                f"{len(DEFAULT_SEEDS)} entries; extend DEFAULT_SEEDS first."
            )
        return list(DEFAULT_SEEDS[:spec])
    return list(spec)

DATASETS: dict[str, dict] = {
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "raw_subdir": "ml-1m",
        "ratings_file": "ratings.dat",
        "sep": "::",
        "columns": ["user_id", "item_id", "rating", "timestamp"],
        "intensity_col": "rating",
        "min_user_inter": 5,
        "min_item_inter": 5,
        "rating_threshold": 0,
    },
    # NOTE: must not be literally "ml-100k" — RecBole 1.2.0's Configurator
    # hard-codes that name and overrides `data_path` to its bundled
    # `dataset_example/ml-100k/` (a 3-column rating-only file), which silently
    # drops our `intensity:float` column. See
    # site-packages/recbole/config/configurator.py:_set_default_parameters.
    "ml-100k-iar": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "raw_subdir": "ml-100k",
        "ratings_file": "u.data",
        "sep": "\t",
        "columns": ["user_id", "item_id", "rating", "timestamp"],
        "intensity_col": "rating",
        "min_user_inter": 5,
        "min_item_inter": 5,
        "rating_threshold": 0,
    },
    "amazon-digital-music": {
        # McAuley 2014 Amazon Reviews, Digital Music 5-core. Native (rating, unix_timestamp).
        # ~5.5k users, ~3.6k items, ~64k reviews.
        "url": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz",
        "download_format": "gz",
        "raw_subdir": "amazon-digital-music",
        "ratings_file": "reviews_Digital_Music_5.json",
        "format": "jsonl",
        "column_map": {
            "reviewerID": "user_id",
            "asin": "item_id",
            "overall": "rating",
            "unixReviewTime": "timestamp",
        },
        "intensity_col": "rating",
        "min_user_inter": 5,
        "min_item_inter": 5,
        "rating_threshold": 0,
    },
    "amazon-office-products": {
        # McAuley 2014 Amazon Reviews, Office Products 5-core.
        # ~4.9k users, ~2.4k items, ~53k reviews.
        "url": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz",
        "download_format": "gz",
        "raw_subdir": "amazon-office-products",
        "ratings_file": "reviews_Office_Products_5.json",
        "format": "jsonl",
        "column_map": {
            "reviewerID": "user_id",
            "asin": "item_id",
            "overall": "rating",
            "unixReviewTime": "timestamp",
        },
        "intensity_col": "rating",
        "min_user_inter": 5,
        "min_item_inter": 5,
        "rating_threshold": 0,
    },
    # Steam reviews (Wan & McAuley / W. Kang). Native (user, item, date, hours_played).
    # The file is Python-repr jsonl ({u'k': 'v', ...}), not strict JSON — parser uses
    # ast.literal_eval. Two subsample sizes share the same raw download +
    # parsed-parquet cache via raw_subdir="steam":
    #   - steam-3k: ml-100k-size parity (~36k interactions post-5-core)
    #   - steam-8k: ml-1m-size parity, more reliable absolute numbers
    "steam-3k": {
        "url": "https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
        "download_format": "gz_keep",
        "raw_subdir": "steam",                   # shared with steam-8k
        "ratings_file": "steam_reviews.json.gz",
        "format": "pylit_jsonl_gz",
        "column_map": {
            "username": "user_id",
            "product_id": "item_id",
            "date": "timestamp",
            "hours": "intensity",
        },
        "intensity_col": "intensity",
        "timestamp_format": "%Y-%m-%d",
        "subsample_users": 3000,
        "subsample_seed": 2020,
        "min_user_inter": 5,
        "min_item_inter": 5,
        "rating_threshold": 0,
    },
    "steam-8k": {
        "url": "https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
        "download_format": "gz_keep",
        "raw_subdir": "steam",                   # shared with steam-3k
        "ratings_file": "steam_reviews.json.gz",
        "format": "pylit_jsonl_gz",
        "column_map": {
            "username": "user_id",
            "product_id": "item_id",
            "date": "timestamp",
            "hours": "intensity",
        },
        "intensity_col": "intensity",
        "timestamp_format": "%Y-%m-%d",
        "subsample_users": 8000,
        "subsample_seed": 2020,
        "min_user_inter": 5,
        "min_item_inter": 5,
        "rating_threshold": 0,
    },
}


def common_recbole_config(dataset_name: str) -> dict[str, Any]:
    """Return config fields shared by every model on a given dataset.

    Parameters:
        dataset_name: Key into ``DATASETS`` (e.g. ``"ml-1m"``); used to resolve
            per-dataset interaction-count filters.

    Returns:
        Dict of RecBole config keys covering paths, field names, evaluation
        strategy, metrics, top-K list, device, and early-stopping patience.
    """
    return {
        "data_path": str(RECBOLE_DATA_DIR),
        "dataset": dataset_name,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "show_progress": False,
        "save_dataset": True,
        # Caching dataloaders is unsafe across runs: RecBole's cache key only
        # checks dataset_arguments + seed + repeatable + eval_args, so a change
        # to `train_neg_sample_args` (e.g. switching loss_type CE <-> BPR) silently
        # reuses a stale dataloader whose sampler is None, then crashes deep in
        # the training loop. Rebuilding takes a few seconds — worth it.
        "save_dataloaders": False,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        # `intensity` is a per-interaction FLOAT. RecBole's SequentialDataset auto-builds
        # `intensity_list` aligned to `item_id_list`; non-IA models simply ignore it.
        "INTENSITY_FIELD": "intensity",
        "load_col": {"inter": ["user_id", "item_id", "timestamp", "intensity"]},
        "user_inter_num_interval": f"[{DATASETS[dataset_name]['min_user_inter']},inf)",
        "item_inter_num_interval": f"[{DATASETS[dataset_name]['min_item_inter']},inf)",
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
        "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
        "topk": TOP_K,
        "valid_metric": "NDCG@10",
        "device": DEVICE,
        "MAX_ITEM_LIST_LENGTH": 50,
        "stopping_step": EARLY_STOP_PATIENCE,
    }
