"""Optuna-driven hyperparameter search per (dataset, model).

Studies are persisted to a SQLite DB under ``results/hpo/`` so HPO itself is also
resumable: re-running with the same study name continues from completed trials.
"""

from __future__ import annotations

import logging
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config import HPO_DIR, HPO_EPOCHS, N_TRIALS, PRIMARY_METRIC
from models import get_spec
from runner import train_one

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _study_storage(dataset_name: str, model_name: str) -> str:
    """Return the SQLite connection string for the Optuna study database.

    Parameters:
        dataset_name: RecBole dataset identifier (e.g. ``"ml-1m"``).
        model_name: Key in ``MODEL_REGISTRY`` (e.g. ``"SASRec"``).

    Returns:
        SQLite URL of the form ``sqlite:///<HPO_DIR>/<dataset>__<model>.db``.
    """
    db_path = HPO_DIR / f"{dataset_name}__{model_name}.db"
    return f"sqlite:///{db_path}"


def _objective(dataset_name: str, model_name: str, trial: optuna.Trial) -> float:
    """Optuna objective: sample hyperparams, train for HPO_EPOCHS, return the primary metric.

    Parameters:
        dataset_name: RecBole dataset identifier passed through to ``train_one``.
        model_name: Key in ``MODEL_REGISTRY``; its ``search_space`` callable is
            used to sample the hyperparameter overrides for this trial.
        trial: Active Optuna trial used for parameter suggestion.

    Returns:
        Value of ``PRIMARY_METRIC`` from the validation result, falling back to
        ``best_valid_score`` if the metric key is not found.
    """
    spec = get_spec(model_name)
    overrides = spec["search_space"](trial)
    overrides["epochs"] = HPO_EPOCHS
    out = train_one(
        dataset_name,
        model_name,
        overrides=overrides,
        epochs=HPO_EPOCHS,
        saved=False,
    )
    valid = out["best_valid_result"]
    metric_key = PRIMARY_METRIC.lower()
    for k, v in valid.items():
        if k.lower() == metric_key:
            return v
    return out["best_valid_score"]


def run_optuna(
    dataset_name: str,
    model_name: str,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Run (or resume) an Optuna study and return the best hyperparameters found.

    Skips HPO entirely if the model's search space is empty. Re-running with the
    same arguments continues from previously completed trials in the persistent DB.

    Parameters:
        dataset_name: RecBole dataset identifier (e.g. ``"ml-1m"``).
        model_name: Key in ``MODEL_REGISTRY`` (e.g. ``"SASRec"``).
        n_trials: Total number of trials to complete (including previously finished
            ones). Defaults to ``N_TRIALS`` from config when ``None``.

    Returns:
        Dictionary with keys:

        - ``best_params``: Dict of best hyperparameter values (empty if HPO was
          skipped or no trial completed successfully).
        - ``best_value``: Best observed metric value, or ``None`` if unavailable.
        - ``n_trials``: Total number of trials in the study after this run.
    """
    n = n_trials if n_trials is not None else N_TRIALS
    spec = get_spec(model_name)

    try:
        probe = spec["search_space"](optuna.trial.FixedTrial({}))
    except Exception:
        probe = {"_": "non-empty"}
    if not probe:
        print(f"[hpo] {model_name}: empty search space, skipping HPO.")
        return {"best_params": {}, "best_value": None, "n_trials": 0}

    study = optuna.create_study(
        study_name=f"{dataset_name}__{model_name}",
        storage=_study_storage(dataset_name, model_name),
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=2),
    )

    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, n - completed)
    if remaining == 0:
        print(f"[hpo] {model_name}: study has {completed} completed trials (>= {n}), skipping.")
    else:
        print(f"[hpo] {model_name}: running {remaining} new trials (already have {completed}).")

        def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
            value = f"{trial.value:.4f}" if trial.value is not None else "n/a"
            print(f"[hpo] {model_name}: trial {done}/{n} ({trial.state.name}) value={value}")

        study.optimize(
            lambda t: _objective(dataset_name, model_name, t),
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=False,
            callbacks=[_log_trial],
        )

    if not study.best_trial or study.best_trial.value is None:
        return {"best_params": {}, "best_value": None, "n_trials": completed}
    return {
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
    }
