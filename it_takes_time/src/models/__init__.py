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

from models.variants.ia_sasrec import IASASRecAdd, IASASRecMul, IASASRecVal


def _ce_static() -> dict[str, Any]:
    """Return fixed config overrides for models using full-softmax cross-entropy loss.

    Applies to SASRec, BERT4Rec, GRU4Rec, and NARM. Disables negative sampling so
    RecBole feeds full-vocabulary targets into the CE loss.

    Returns:
        Config dict with ``loss_type="CE"`` and ``train_neg_sample_args=None``.
    """
    return {"loss_type": "CE", "train_neg_sample_args": None}


def _sasrec_yaml_defaults() -> dict[str, Any]:
    """Return SASRec's internal-yaml defaults that custom variants miss.

    When a custom subclass is passed to ``recbole.config.Config`` as a class
    object, RecBole looks up ``<classname>.yaml`` for internal defaults — which
    doesn't exist for our IA-SASRec variants, leaving fields like ``hidden_act``
    set to ``None`` and crashing inside ``FeedForward.get_hidden_act``. We
    inline the values from ``recbole/properties/model/SASRec.yaml`` here. The
    other SASRec fields (``n_layers``, ``hidden_size``, etc.) are always
    supplied by the HPO search space, so they don't need defaults.
    """
    return {
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
    }


def _bpr_static() -> dict[str, Any]:
    """Return fixed config overrides for models that only support pairwise BPR loss.

    Applies to FPMC. Does not set ``train_neg_sample_args`` because RecBole's
    ``overall.yaml`` default (uniform, sample_num=1) is correct; overriding it with
    an identical dict confuses the Config merge in some versions.

    Returns:
        Config dict with ``loss_type="BPR"``.
    """
    return {"loss_type": "BPR"}


# Small / heavy-tailed datasets (currently just Steam at 36k interactions, mean
# seq length 12) expose single-seed-HPO fragility: hidden_size=32 + lr near 5e-3
# converges only from a lucky init, then 4/5 final-fit seeds collapse to the
# noise floor. For these datasets we drop the smallest hidden_size and lower the
# learning-rate ceiling. Other datasets are unchanged.
_NARROW_HPO_DATASETS = {"steam-3k", "steam-8k", "steam-15k"}


def sasrec_space(trial: optuna.Trial, dataset_name: str | None = None) -> dict[str, Any]:
    """Sample hyperparameters for SASRec.

    Parameters:
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Dict of RecBole config overrides covering architecture and learning-rate
        hyperparameters for SASRec.
    """
    narrow = dataset_name in _NARROW_HPO_DATASETS
    hidden_choices = [64, 128] if narrow else [32, 64, 128]
    lr_high = 2e-3 if narrow else 5e-3
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "n_heads": trial.suggest_categorical("n_heads", [1, 2, 4]),
        "hidden_size": trial.suggest_categorical("hidden_size", hidden_choices),
        "inner_size": trial.suggest_categorical("inner_size", [64, 128, 256]),
        "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.1, 0.5),
        "attn_dropout_prob": trial.suggest_float("attn_dropout_prob", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, lr_high, log=True),
    }


def ia_sasrec_space(trial: optuna.Trial, dataset_name: str | None = None) -> dict[str, Any]:
    """Sample hyperparameters for IA-SASRec variants.

    Reuses the vanilla SASRec search space and adds a normalisation-mode knob
    that controls how raw intensity values are squashed before injection into
    the attention mechanism.

    Parameters:
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Dict of RecBole config overrides covering SASRec hyperparameters plus
        ``intensity_norm``.
    """
    base = sasrec_space(trial, dataset_name=dataset_name)
    base["intensity_norm"] = trial.suggest_categorical(
        "intensity_norm", ["log1p_minmax", "minmax", "zscore"]
    )
    return base


def bert4rec_space(trial: optuna.Trial, dataset_name: str | None = None) -> dict[str, Any]:
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


def gru4rec_space(trial: optuna.Trial, dataset_name: str | None = None) -> dict[str, Any]:
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


def narm_space(trial: optuna.Trial, dataset_name: str | None = None) -> dict[str, Any]:
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


def fpmc_space(trial: optuna.Trial, dataset_name: str | None = None) -> dict[str, Any]:
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


def bpr_space(trial: optuna.Trial, dataset_name: str | None = None) -> dict[str, Any]:
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


def itemknn_space(trial: optuna.Trial, dataset_name: str | None = None) -> dict[str, Any]:
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


def pop_space(trial: optuna.Trial, dataset_name: str | None = None) -> dict[str, Any]:
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
    "IA-SASRec-Add": {"class": IASASRecAdd, "type": "sequential",
                      "search_space": ia_sasrec_space,
                      "static": {**_ce_static(), **_sasrec_yaml_defaults()}},
    "IA-SASRec-Mul": {"class": IASASRecMul, "type": "sequential",
                      "search_space": ia_sasrec_space,
                      "static": {**_ce_static(), **_sasrec_yaml_defaults()}},
    "IA-SASRec-Val": {"class": IASASRecVal, "type": "sequential",
                      "search_space": ia_sasrec_space,
                      "static": {**_ce_static(), **_sasrec_yaml_defaults()}},
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
