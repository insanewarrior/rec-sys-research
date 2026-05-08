"""Model registry: name -> spec dict consumed by the runner.

A spec has:
  - ``class``:        either a string (RecBole built-in name) or a Python class
                      (a custom variant in ``src/models/variants/``).
  - ``type``:         "general" or "sequential" (drives RecBole config defaults).
  - ``search_space``: callable ``(optuna.Trial) -> dict[str, Any]`` of hyperparam
                      overrides on top of RecBole defaults.
  - ``static``:       dict of fixed config overrides (optional).

Add custom variants by subclassing in ``variants/`` and inserting an entry below.
"""

from __future__ import annotations

from typing import Any, Callable

import optuna


def _ce_static() -> dict:
    """Sequential models that support full-softmax CE (SASRec, BERT4Rec, GRU4Rec, NARM).

    Disables negative sampling: ``train_neg_sample_args=None`` tells RecBole to
    feed full-vocabulary targets into the cross-entropy loss.
    """
    return {"loss_type": "CE", "train_neg_sample_args": None}


def _bpr_static() -> dict:
    """Models that only support pairwise BPR (e.g. FPMC).

    No need to set ``train_neg_sample_args``: RecBole's ``overall.yaml`` default
    (uniform, sample_num=1) is correct, and overriding with an identical dict
    confuses the Config merge in some versions, leaving the sampler unbuilt.
    """
    return {"loss_type": "BPR"}


def sasrec_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "n_heads": trial.suggest_categorical("n_heads", [1, 2, 4]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "inner_size": trial.suggest_categorical("inner_size", [64, 128, 256]),
        "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.1, 0.5),
        "attn_dropout_prob": trial.suggest_float("attn_dropout_prob", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def bert4rec_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "n_heads": trial.suggest_categorical("n_heads", [1, 2, 4]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "inner_size": trial.suggest_categorical("inner_size", [64, 128, 256]),
        "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.1, 0.5),
        "attn_dropout_prob": trial.suggest_float("attn_dropout_prob", 0.1, 0.5),
        "mask_ratio": trial.suggest_float("mask_ratio", 0.1, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def gru4rec_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "embedding_size": trial.suggest_categorical("embedding_size", [32, 64, 128]),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "dropout_prob": trial.suggest_float("dropout_prob", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def narm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "embedding_size": trial.suggest_categorical("embedding_size", [32, 64, 128]),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "n_layers": trial.suggest_int("n_layers", 1, 2),
        "dropout_probs": trial.suggest_categorical("dropout_probs", [[0.25, 0.5], [0.1, 0.25]]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def fpmc_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "embedding_size": trial.suggest_categorical("embedding_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def bpr_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "embedding_size": trial.suggest_categorical("embedding_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def itemknn_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "k": trial.suggest_int("k", 50, 400, step=50),
        "shrink": trial.suggest_float("shrink", 0.0, 1.0),
    }


def pop_space(trial: optuna.Trial) -> dict[str, Any]:
    return {}  # no hyperparams


ModelSpec = dict[str, Any]
SearchSpaceFn = Callable[[optuna.Trial], dict[str, Any]]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Pop":      {"class": "Pop",      "type": "general",    "search_space": pop_space},
    "BPR":      {"class": "BPR",      "type": "general",    "search_space": bpr_space},
    "ItemKNN":  {"class": "ItemKNN",  "type": "general",    "search_space": itemknn_space},
    "FPMC":     {"class": "FPMC",     "type": "sequential", "search_space": fpmc_space,    "static": _bpr_static()},
    "GRU4Rec":  {"class": "GRU4Rec",  "type": "sequential", "search_space": gru4rec_space, "static": _ce_static()},
    "NARM":     {"class": "NARM",     "type": "sequential", "search_space": narm_space,    "static": _ce_static()},
    "SASRec":   {"class": "SASRec",   "type": "sequential", "search_space": sasrec_space,  "static": _ce_static()},
    "BERT4Rec": {"class": "BERT4Rec", "type": "sequential", "search_space": bert4rec_space,"static": _ce_static()},
}


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def get_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Known: {list_models()}")
    return MODEL_REGISTRY[name]
