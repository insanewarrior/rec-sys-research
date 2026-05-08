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

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed

from config import (
    CHECKPOINT_DIR,
    EVAL_DIR,
    FINAL_EPOCHS,
    HPO_EPOCHS,
    common_recbole_config,
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
    spec = get_spec(model_name)
    name_str, model_cls = resolve(spec["class"])
    cfg_dict = common_recbole_config(dataset_name)
    cfg_dict.update(spec.get("static", {}) or {})
    cfg_dict.update(overrides or {})
    cfg_dict["epochs"] = epochs
    cfg_dict["save_dataset"] = saved
    config = Config(model=name_str, dataset=dataset_name, config_dict=cfg_dict)
    init_seed(config["seed"], config["reproducibility"])
    return config, model_cls


def _instantiate_model(config: Config, dataset, model_cls: type | None):
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
) -> dict[str, Any]:
    """Single train/eval pass. Used both for HPO trials and final fits."""
    overrides = overrides or {}
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

    t0 = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=False
    )
    train_time = time.time() - t0

    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=False)
    return {
        "best_valid_score": float(best_valid_score),
        "best_valid_result": {k: float(v) for k, v in dict(best_valid_result).items()},
        "test_result": {k: float(v) for k, v in dict(test_result).items()},
        "train_seconds": train_time,
        "checkpoint": str(trainer.saved_model_file) if saved else None,
        "config_dict": {k: overrides.get(k) for k in overrides},
    }


def eval_path(dataset_name: str, model_name: str) -> Path:
    return EVAL_DIR / f"{dataset_name}__{model_name}.json"


def has_result(dataset_name: str, model_name: str) -> bool:
    return eval_path(dataset_name, model_name).exists()


def load_result(dataset_name: str, model_name: str) -> dict[str, Any]:
    return json.loads(eval_path(dataset_name, model_name).read_text())


def save_result(dataset_name: str, model_name: str, result: dict[str, Any]) -> Path:
    p = eval_path(dataset_name, model_name)
    p.write_text(json.dumps(result, indent=2, default=str))
    return p


def train_and_eval(
    dataset_name: str,
    model_name: str,
    best_params: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Resumable entry point. If a prior result exists on disk, load it."""
    if has_result(dataset_name, model_name) and not force:
        print(f"[runner] {model_name}: cached result loaded, skipping training.")
        return load_result(dataset_name, model_name)

    out = train_one(
        dataset_name,
        model_name,
        overrides=best_params or {},
        epochs=FINAL_EPOCHS,
        saved=True,
    )
    record = {
        "model": model_name,
        "dataset": dataset_name,
        "best_params": best_params or {},
        **out,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_result(dataset_name, model_name, record)
    print(f"[runner] {model_name}: saved to {eval_path(dataset_name, model_name)}")
    return record
