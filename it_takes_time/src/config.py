"""Project-wide configuration: paths, dataset registry, HPO/eval knobs."""

from __future__ import annotations

import os
from pathlib import Path

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

N_TRIALS = int(os.environ.get("N_TRIALS", "5"))
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
        "min_user_inter": 5,
        "min_item_inter": 5,
        "rating_threshold": 0,
    },
}


def common_recbole_config(dataset_name: str) -> dict:
    """Config fields shared by every model on a given dataset."""
    return {
        "data_path": str(RECBOLE_DATA_DIR),
        "dataset": dataset_name,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "show_progress": False,
        "save_dataset": True,
        "save_dataloaders": True,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
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
