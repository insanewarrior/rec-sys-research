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


def _ce_static() -> dict[str, Any]:
    """Return fixed config overrides for models using full-softmax cross-entropy loss.

    Applies to SASRec, BERT4Rec, GRU4Rec, and NARM. Disables negative sampling so
    RecBole feeds full-vocabulary targets into the CE loss.

    Returns:
        Config dict with ``loss_type="CE"`` and ``train_neg_sample_args=None``.
    """
    return {"loss_type": "CE", "train_neg_sample_args": None}


def _bpr_static() -> dict[str, Any]:
    """Return fixed config overrides for models that only support pairwise BPR loss.

    Applies to FPMC. Does not set ``train_neg_sample_args`` because RecBole's
    ``overall.yaml`` default (uniform, sample_num=1) is correct; overriding it with
    an identical dict confuses the Config merge in some versions.

    Returns:
        Config dict with ``loss_type="BPR"``.
    """
    return {"loss_type": "BPR"}


def sasrec_space(trial: optuna.Trial) -> dict[str, Any]:
    """Sample hyperparameters for SASRec.

    Parameters:
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Dict of RecBole config overrides covering architecture and learning-rate
        hyperparameters for SASRec.
    """
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
    """Sample hyperparameters for BERT4Rec.

    Parameters:
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Dict of RecBole config overrides covering architecture, dropout, mask ratio,
        and learning rate for BERT4Rec.
    """
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
    """Sample hyperparameters for GRU4Rec.

    Parameters:
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Dict of RecBole config overrides covering embedding size, hidden size,
        layer count, dropout, and learning rate for GRU4Rec.
    """
    return {
        "embedding_size": trial.suggest_categorical("embedding_size", [32, 64, 128]),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "dropout_prob": trial.suggest_float("dropout_prob", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def narm_space(trial: optuna.Trial) -> dict[str, Any]:
    """Sample hyperparameters for NARM.

    Parameters:
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Dict of RecBole config overrides covering embedding size, hidden size,
        layer count, dropout pair, and learning rate for NARM.
    """
    return {
        "embedding_size": trial.suggest_categorical("embedding_size", [32, 64, 128]),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "n_layers": trial.suggest_int("n_layers", 1, 2),
        "dropout_probs": trial.suggest_categorical("dropout_probs", [[0.25, 0.5], [0.1, 0.25]]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def fpmc_space(trial: optuna.Trial) -> dict[str, Any]:
    """Sample hyperparameters for FPMC.

    Parameters:
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Dict of RecBole config overrides covering embedding size and learning rate
        for FPMC.
    """
    return {
        "embedding_size": trial.suggest_categorical("embedding_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def bpr_space(trial: optuna.Trial) -> dict[str, Any]:
    """Sample hyperparameters for BPR (Bayesian Personalised Ranking MF).

    Parameters:
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Dict of RecBole config overrides covering embedding size and learning rate
        for BPR.
    """
    return {
        "embedding_size": trial.suggest_categorical("embedding_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }


def itemknn_space(trial: optuna.Trial) -> dict[str, Any]:
    """Sample hyperparameters for ItemKNN.

    Parameters:
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Dict of RecBole config overrides covering neighbourhood size *k* and
        Laplacian shrinkage for ItemKNN.
    """
    return {
        "k": trial.suggest_int("k", 50, 400, step=50),
        "shrink": trial.suggest_float("shrink", 0.0, 1.0),
    }


def pop_space(trial: optuna.Trial) -> dict[str, Any]:
    """Return an empty search space for the Popularity baseline (no hyperparameters).

    Parameters:
        trial: Active Optuna trial (unused; present for interface uniformity).

    Returns:
        Empty dict — Pop has no tunable hyperparameters.
    """
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
    """Return the names of all registered models.

    Returns:
        List of model name strings in insertion order (matches ``MODEL_REGISTRY``
        key order).
    """
    return list(MODEL_REGISTRY.keys())


def get_spec(name: str) -> ModelSpec:
    """Look up and return the spec dict for a registered model.

    Parameters:
        name: Model name key (e.g. ``"SASRec"``). Case-sensitive.

    Returns:
        The ``ModelSpec`` dict containing ``class``, ``type``, ``search_space``,
        and optionally ``static``.

    Raises:
        KeyError: If *name* is not present in ``MODEL_REGISTRY``.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Known: {list_models()}")
    return MODEL_REGISTRY[name]
