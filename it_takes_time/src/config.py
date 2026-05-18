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

N_TRIALS = int(os.environ.get("N_TRIALS", "10"))
HPO_EPOCHS = int(os.environ.get("HPO_EPOCHS", "10"))
FINAL_EPOCHS = int(os.environ.get("FINAL_EPOCHS", "50"))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "5"))

DATASETS: dict[str, dict] = {
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "raw_subdir": "ml-1m",
        "ratings_file": "ratings.dat",
        "sep": "::",
        "columns": ["user_id", "item_id", "rating", "timestamp"],
        "intensity_col": "rating",  # 1..5 explicit; doubles as intensity for IA-SASRec
        "min_user_inter": 5,
        "min_item_inter": 5,
        "rating_threshold": 0,
    },
    "ml-100k": {
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
    "steam": {
        # Steam-200k (Tamber/Kaggle) — true implicit feedback with hours-played intensity.
        # Fetched via kagglehub; needs ``~/.kaggle/kaggle.json`` or the env vars
        # ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` to be set once. The CSV has no
        # timestamps; we synthesize them by stable-sorting on hours so
        # higher-engagement plays appear later in the user's sequence.
        "kaggle_dataset": "tamber/steam-video-games",
        "raw_subdir": "steam",
        "ratings_file": "steam-200k.csv",
        "sep": ",",
        "columns": ["user_id", "item_id", "behavior", "hours", "extra"],
        "intensity_col": "hours",
        "behavior_filter": "play",   # drop "purchase" rows (hours==1.0 sentinel)
        "synthesize_timestamp": True,
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
