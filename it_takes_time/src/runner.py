"""Train + evaluate a single model on a single dataset, with on-disk resumability.

The runner is the only place that talks to RecBole's ``Config``, ``Dataset``,
``data_preparation``, and ``Trainer`` directly. Everything upstream (notebook,
HPO loop) calls into ``train_and_eval`` / ``train_one`` and treats it as a black box.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import config as _config  # installs numpy 2.x shim before recbole import below

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed

from config import (
    CHECKPOINT_DIR,
    EVAL_DIR,
    FINAL_EPOCHS,
    HPO_EPOCHS,
    LEGACY_SEED,
    common_recbole_config,
    seeds_for,
)
from models import get_spec
from models.builtin import resolve


def _build_config(
    dataset_name: str,
    model_name: str,
    overrides: dict[str, Any],
    epochs: int,
    saved: bool,
) -> tuple[Config, type | None]:
    """Construct a RecBole ``Config`` and resolve the model class.

    Merges the common dataset config, the model's static overrides, and any
    caller-supplied *overrides*, then seeds the RNG.

    Parameters:
        dataset_name: RecBole dataset identifier (e.g. ``"ml-1m"``).
        model_name: Key in ``MODEL_REGISTRY`` (e.g. ``"SASRec"``).
        overrides: Arbitrary config key/value pairs that take precedence over
            both the common config and the model's static defaults.
        epochs: Training epoch count written into the config.
        saved: Whether the best checkpoint should be persisted to disk; written
            into ``save_dataset`` to prevent stale dataloader caching.

    Returns:
        A 2-tuple ``(config, model_cls)`` where *model_cls* is ``None`` for
        RecBole built-ins (resolved by name at trainer creation time) or the
        actual Python class for custom variants.
    """
    spec = get_spec(model_name)
    name_str, model_cls = resolve(spec["class"])
    cfg_dict = common_recbole_config(dataset_name)
    cfg_dict.update(spec.get("static", {}) or {})
    cfg_dict.update(overrides or {})
    cfg_dict["epochs"] = epochs
    cfg_dict["save_dataset"] = saved
    # For custom variants, pass the class object directly so RecBole skips its
    # name-based submodule scan (which transitively imports ``lightgbm`` and
    # crashes when libomp isn't installed). Built-ins are still resolved by name.
    model_arg = model_cls if model_cls is not None else name_str
    config = Config(model=model_arg, dataset=dataset_name, config_dict=cfg_dict)
    init_seed(config["seed"], config["reproducibility"])
    return config, model_cls


def _attach_curve_capture(trainer: Any) -> tuple[list[dict[str, Any]], list[float]]:
    """Patch *trainer* in-place to record per-epoch valid scores + train losses.

    RecBole's ``Trainer.fit`` already calls ``_valid_epoch`` and ``_train_epoch``
    once per epoch (subject to ``eval_step``); this wrapper just intercepts
    their return values. **Zero extra compute** — we are reusing work the
    trainer would do anyway.

    Useful for convergence plots, diagnosing under/overfitting, and deciding
    whether ``FINAL_EPOCHS`` / ``HPO_EPOCHS`` are well-tuned per dataset.

    Returns:
        Tuple ``(valid_curve, train_losses)``; both lists are mutated in place
        as ``trainer.fit`` runs. ``valid_curve`` entries look like
        ``{"epoch": 0, "valid_score": 0.09, "valid_result": {"ndcg@10": 0.09, ...}}``.
    """
    valid_curve: list[dict[str, Any]] = []
    train_losses: list[float | list[float]] = []
    orig_valid = trainer._valid_epoch
    orig_train = trainer._train_epoch

    def _wrap_valid(*args, **kwargs):
        valid_score, valid_result = orig_valid(*args, **kwargs)
        valid_curve.append({
            "epoch": len(valid_curve),
            "valid_score": float(valid_score),
            "valid_result": {k: float(v) for k, v in dict(valid_result).items()},
        })
        return valid_score, valid_result

    def _wrap_train(*args, **kwargs):
        loss = orig_train(*args, **kwargs)
        if isinstance(loss, (int, float)):
            train_losses.append(float(loss))
        elif isinstance(loss, (list, tuple)):
            train_losses.append([float(x) for x in loss])
        else:
            # Tensor or other scalar-like — best-effort cast.
            try:
                train_losses.append(float(loss))
            except Exception:
                train_losses.append(None)
        return loss

    trainer._valid_epoch = _wrap_valid
    trainer._train_epoch = _wrap_train
    return valid_curve, train_losses


def _instantiate_model(config: Config, dataset: Any, model_cls: type | None) -> Any:
    """Instantiate and move a model to the configured device.

    Parameters:
        config: RecBole ``Config`` object; provides ``"model"`` name and ``"device"``.
        dataset: RecBole ``Dataset`` passed as the second argument to the model
            constructor.
        model_cls: Explicit Python class to instantiate, or ``None`` to look up
            the built-in class via ``recbole.utils.get_model``.

    Returns:
        Model instance on ``config["device"]``.
    """
    if model_cls is not None:
        return model_cls(config, dataset).to(config["device"])
    cls = get_model(config["model"])
    return cls(config, dataset).to(config["device"])


def train_one(
    dataset_name: str,
    model_name: str,
    overrides: dict[str, Any] | None = None,
    epochs: int | None = None,
    saved: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a single train/evaluate pass and return the metric results.

    Used both for HPO trials (``saved=False``, reduced epochs) and for final
    full-length fits (``saved=True``).

    Parameters:
        dataset_name: RecBole dataset identifier (e.g. ``"ml-1m"``).
        model_name: Key in ``MODEL_REGISTRY`` (e.g. ``"SASRec"``).
        overrides: Config key/value pairs that override both the common config and
            the model's static defaults. Typically supplied by the HPO objective.
        epochs: Number of training epochs. Defaults to ``FINAL_EPOCHS`` when
            ``None``.
        saved: If ``True``, the best checkpoint is written to ``CHECKPOINT_DIR``
            and ``load_best_model=True`` is used during test evaluation.

    Returns:
        Dictionary with keys ``best_valid_score``, ``best_valid_result``,
        ``test_result``, ``train_seconds``, ``checkpoint``, and ``config_dict``.
    """
    overrides = dict(overrides or {})
    if seed is not None:
        # Stamp the requested seed into the RecBole config so init_seed picks
        # it up. Used by multi-seed final fits (HPO trials don't pass this).
        overrides["seed"] = seed
    config, model_cls = _build_config(
        dataset_name,
        model_name,
        overrides,
        epochs if epochs is not None else FINAL_EPOCHS,
        saved=saved,
    )
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = _instantiate_model(config, train_data._dataset, model_cls)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    valid_curve, train_losses = _attach_curve_capture(trainer)

    t0 = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=False
    )
    train_time = time.time() - t0

    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=False)
    # Surface per-layer λ values for IA-SASRec variants. `trainer.evaluate`
    # above reloads the best checkpoint when saved=True, so `model` is in the
    # state we actually care about.
    intensity_params = (
        model.get_intensity_params() if hasattr(model, "get_intensity_params") else None
    )
    return {
        "best_valid_score": float(best_valid_score),
        "best_valid_result": {k: float(v) for k, v in dict(best_valid_result).items()},
        "test_result": {k: float(v) for k, v in dict(test_result).items()},
        "train_seconds": train_time,
        "checkpoint": Path(trainer.saved_model_file).name if saved else None,
        "config_dict": {k: overrides.get(k) for k in overrides},
        "intensity_params": intensity_params,
        "seed": int(config["seed"]),
        # Per-epoch convergence trace. Cheap to capture (no extra compute) and
        # invaluable for figures / tuning epoch counts in retrospect.
        "valid_curve": valid_curve,
        "train_losses": train_losses,
    }


def eval_path(dataset_name: str, model_name: str, seed: int | None = None) -> Path:
    """Return the canonical JSON path for a (dataset, model, seed) result.

    Parameters:
        dataset_name: RecBole dataset identifier (e.g. ``"ml-1m"``).
        model_name: Key in ``MODEL_REGISTRY`` (e.g. ``"SASRec"``).
        seed: When provided, returns the seed-tagged path
            ``<dataset>__<model>__seed<seed>.json``. When ``None``, returns the
            legacy single-seed path ``<dataset>__<model>.json`` — used only for
            backward compatibility when reading pre-multi-seed results.

    Returns:
        Path under ``EVAL_DIR``.
    """
    if seed is None:
        return EVAL_DIR / f"{dataset_name}__{model_name}.json"
    return EVAL_DIR / f"{dataset_name}__{model_name}__seed{seed}.json"


def _legacy_path(dataset_name: str, model_name: str) -> Path:
    """Path of the pre-multi-seed result file (single JSON per dataset/model)."""
    return EVAL_DIR / f"{dataset_name}__{model_name}.json"


def _list_seed_paths(dataset_name: str, model_name: str) -> list[Path]:
    """All per-seed JSONs that exist on disk for a (dataset, model) pair."""
    return sorted(EVAL_DIR.glob(f"{dataset_name}__{model_name}__seed*.json"))


def has_seed_result(dataset_name: str, model_name: str, seed: int) -> bool:
    """Return ``True`` if a result for this specific seed exists on disk.

    Treats the legacy single-file path as the ``LEGACY_SEED`` result so we
    don't redundantly re-train at seed=2020 on already-completed models.
    """
    if eval_path(dataset_name, model_name, seed).exists():
        return True
    if seed == LEGACY_SEED and _legacy_path(dataset_name, model_name).exists():
        return True
    return False


def load_seed_result(dataset_name: str, model_name: str, seed: int) -> dict[str, Any]:
    """Load the saved result for a single seed.

    Falls back to the legacy single-file path when ``seed == LEGACY_SEED`` and
    no seed-tagged file is present yet.
    """
    p = eval_path(dataset_name, model_name, seed)
    if not p.exists() and seed == LEGACY_SEED:
        p = _legacy_path(dataset_name, model_name)
    return json.loads(p.read_text())


def save_seed_result(
    dataset_name: str, model_name: str, seed: int, result: dict[str, Any]
) -> Path:
    """Serialize *result* to the seed-tagged JSON path for the given triple."""
    p = eval_path(dataset_name, model_name, seed)
    p.write_text(json.dumps(result, indent=2, default=str))
    return p


def _aggregate_metric_dicts(dicts: list[dict[str, float] | None]) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and std (sample, ddof=1) of a list of {metric: value} dicts.

    Missing keys / ``None`` entries are skipped silently. Std is 0 when only
    one value is present.
    """
    if not dicts:
        return {}, {}
    keys: set[str] = set()
    for d in dicts:
        if d:
            keys.update(d.keys())
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts if d is not None and k in d]
        if not vals:
            continue
        mean[k] = sum(vals) / len(vals)
        if len(vals) > 1:
            m = mean[k]
            std[k] = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        else:
            std[k] = 0.0
    return mean, std


def aggregate_seed_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-seed result dicts into a single aggregated record.

    The returned dict preserves the legacy single-seed shape (``test_result``,
    ``train_seconds``, ``intensity_params``, ``best_params`` at the top level),
    so notebook code that reads ``result["test_result"]["ndcg@10"]`` keeps
    working. Per-seed details and standard deviations are added alongside.
    """
    if not records:
        raise ValueError("aggregate_seed_records called with no records")
    sample = records[0]
    test_results = [r.get("test_result") for r in records]
    valid_results = [r.get("best_valid_result") for r in records]
    intensity_params_list = [r.get("intensity_params") for r in records]
    train_secs = [r.get("train_seconds", 0.0) for r in records]
    best_valid_scores = [r.get("best_valid_score") for r in records if r.get("best_valid_score") is not None]

    test_mean, test_std = _aggregate_metric_dicts(test_results)
    valid_mean, _valid_std = _aggregate_metric_dicts(valid_results)
    ip_mean, ip_std = _aggregate_metric_dicts(intensity_params_list)

    n = len(records)
    train_mean = sum(train_secs) / n if n else 0.0
    train_std = (sum((t - train_mean) ** 2 for t in train_secs) / (n - 1)) ** 0.5 if n > 1 else 0.0

    return {
        "model": sample.get("model"),
        "dataset": sample.get("dataset"),
        "best_params": sample.get("best_params", {}),
        # Legacy-shape fields (means across seeds) so existing readers work.
        "test_result": test_mean,
        "best_valid_result": valid_mean,
        "best_valid_score": (sum(best_valid_scores) / len(best_valid_scores)) if best_valid_scores else None,
        "train_seconds": train_mean,
        "intensity_params": ip_mean or None,
        # Multi-seed additions.
        "test_result_std": test_std,
        "train_seconds_std": train_std,
        "intensity_params_std": ip_std or None,
        "seeds_used": [r.get("seed") for r in records],
        "n_seeds": n,
        "per_seed": records,
        "completed_at": sample.get("completed_at"),
    }


def has_result(dataset_name: str, model_name: str) -> bool:
    """Return ``True`` if at least one saved result JSON exists for the pair.

    Used by the notebook as a coarse "skip HPO if we already have any result"
    gate. For finer-grained checks see :func:`has_seed_result`.
    """
    if _legacy_path(dataset_name, model_name).exists():
        return True
    return bool(_list_seed_paths(dataset_name, model_name))


def load_result(dataset_name: str, model_name: str) -> dict[str, Any]:
    """Load and aggregate every saved-seed result for a (dataset, model) pair.

    Returns a record with the legacy single-seed shape (so old callers keep
    working) plus the multi-seed extras documented in
    :func:`aggregate_seed_records`.

    Raises:
        FileNotFoundError: If no result file (legacy or seed-tagged) exists.
    """
    records: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for p in _list_seed_paths(dataset_name, model_name):
        rec = json.loads(p.read_text())
        records.append(rec)
        seed = rec.get("seed")
        if seed is not None:
            seen_seeds.add(int(seed))
    legacy = _legacy_path(dataset_name, model_name)
    if legacy.exists() and LEGACY_SEED not in seen_seeds:
        rec = json.loads(legacy.read_text())
        # Stamp the implicit seed so downstream code can treat it uniformly.
        rec.setdefault("seed", LEGACY_SEED)
        records.append(rec)
    if not records:
        raise FileNotFoundError(
            f"No result JSONs for {dataset_name}__{model_name} under {EVAL_DIR}"
        )
    return aggregate_seed_records(records)


def save_result(dataset_name: str, model_name: str, result: dict[str, Any]) -> Path:
    """Compatibility wrapper: route to the seed-tagged path using ``result['seed']``.

    Kept so callers that historically wrote whole records here continue to
    work; ``train_and_eval`` itself uses :func:`save_seed_result` directly.
    """
    seed = result.get("seed", LEGACY_SEED)
    return save_seed_result(dataset_name, model_name, seed, result)


def invalidate_cache(
    dataset_name: str,
    model_name: str | None = None,
    also_recbole_cache: bool = False,
) -> list[Path]:
    """Delete every cached result file (all seeds, plus legacy) for a dataset.

    Use this after changing the underlying ``.inter`` schema (e.g. adding the
    intensity column) or whenever you intentionally want to force re-evaluation.

    Parameters:
        dataset_name: RecBole dataset identifier.
        model_name: If given, only delete that single model's results;
            otherwise delete every cached result for the dataset across all
            models and seeds.
        also_recbole_cache: If ``True``, also remove RecBole's pickled
            ``<dataset>-Dataset.pth`` / ``<dataset>-SequentialDataset.pth``
            under ``CHECKPOINT_DIR``. Required when the ``.inter`` schema or
            ``load_col`` changes, because ``recbole.data.create_dataset``
            silently loads the pickle whenever its ``dataset_arguments`` match,
            regardless of ``save_dataset``.

    Returns:
        List of removed file paths.
    """
    if model_name is not None:
        targets = [_legacy_path(dataset_name, model_name)]
        targets += _list_seed_paths(dataset_name, model_name)
    else:
        targets = sorted(EVAL_DIR.glob(f"{dataset_name}__*.json"))
    if also_recbole_cache:
        targets += [
            CHECKPOINT_DIR / f"{dataset_name}-Dataset.pth",
            CHECKPOINT_DIR / f"{dataset_name}-SequentialDataset.pth",
        ]
    removed: list[Path] = []
    for p in targets:
        if p.exists():
            p.unlink()
            removed.append(p)
    if removed:
        print(f"[runner] Invalidated {len(removed)} cached file(s) for {dataset_name}.")
    return removed


def train_and_eval(
    dataset_name: str,
    model_name: str,
    best_params: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Resumable entry point: train and evaluate across all configured seeds.

    For each seed in ``seeds_for(dataset, model)``:
      - if a result file already exists and ``force`` is ``False``, it is
        reused;
      - otherwise the model is trained from scratch with that seed and the
        per-seed JSON is written to disk.

    The returned record aggregates across seeds (mean + std). The legacy
    single-seed shape (``test_result``, ``train_seconds``, etc.) is preserved
    so existing notebook readers keep working unchanged.

    Parameters:
        dataset_name: RecBole dataset identifier (e.g. ``"ml-1m"``).
        model_name: Key in ``MODEL_REGISTRY`` (e.g. ``"SASRec"``).
        best_params: HPO-derived hyperparameter overrides applied to *every*
            seed. Pass ``None`` or ``{}`` to use defaults only.
        force: Re-train even if cached per-seed results already exist.

    Returns:
        Aggregated record dict as produced by :func:`aggregate_seed_records`.
    """
    seeds = seeds_for(dataset_name, model_name)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        if not force and has_seed_result(dataset_name, model_name, seed):
            rec = load_seed_result(dataset_name, model_name, seed)
            rec.setdefault("seed", seed)
            print(
                f"[runner] {model_name} on {dataset_name} (seed={seed}): cached "
                f"result loaded, NDCG@10="
                f"{rec.get('test_result', {}).get('ndcg@10')}"
            )
            records.append(rec)
            continue
        out = train_one(
            dataset_name,
            model_name,
            overrides=best_params or {},
            epochs=FINAL_EPOCHS,
            saved=True,
            seed=seed,
        )
        rec = {
            "model": model_name,
            "dataset": dataset_name,
            "best_params": best_params or {},
            **out,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_seed_result(dataset_name, model_name, seed, rec)
        records.append(rec)
        print(
            f"[runner] {model_name} on {dataset_name} (seed={seed}): "
            f"saved to {eval_path(dataset_name, model_name, seed)}"
        )
    return aggregate_seed_records(records)
