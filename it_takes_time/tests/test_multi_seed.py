"""Tests for the multi-seed result schema (config resolver, aggregation, IO).

These don't exercise the real training loop — they verify the path / loading /
aggregation glue, which is the part most likely to silently corrupt results.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_seeds_for_uses_dataset_default(monkeypatch):
    import config as cfg

    monkeypatch.setitem(cfg.SEEDS_PER_DATASET, "ml-1m", 3)
    assert cfg.seeds_for("ml-1m", "SASRec") == [2020, 2021, 2022]


def test_seeds_for_uses_explicit_list(monkeypatch):
    import config as cfg

    monkeypatch.setitem(cfg.SEEDS_PER_DATASET, "ml-1m", [7, 11, 13])
    assert cfg.seeds_for("ml-1m", "SASRec") == [7, 11, 13]


def test_seeds_for_model_override_wins(monkeypatch):
    import config as cfg

    monkeypatch.setitem(cfg.SEEDS_PER_DATASET, "ml-1m", 3)
    monkeypatch.setitem(cfg.SEEDS_PER_MODEL_DATASET, ("ml-1m", "SASRec"), 5)
    assert cfg.seeds_for("ml-1m", "SASRec") == [2020, 2021, 2022, 2023, 2024]
    # Other models on the same dataset still get the dataset default.
    assert cfg.seeds_for("ml-1m", "IA-SASRec-Add") == [2020, 2021, 2022]


def test_seeds_for_unknown_dataset_defaults_to_one():
    import config as cfg

    assert cfg.seeds_for("brand-new-dataset", "SomeModel") == [2020]


def test_seeds_for_rejects_overdraft(monkeypatch):
    import config as cfg

    monkeypatch.setitem(cfg.SEEDS_PER_DATASET, "ml-1m", 99)
    with pytest.raises(ValueError, match="DEFAULT_SEEDS"):
        cfg.seeds_for("ml-1m", "SASRec")


def test_eval_path_legacy_vs_seed_tagged():
    import runner

    p_legacy = runner.eval_path("ml-1m", "SASRec")
    p_tagged = runner.eval_path("ml-1m", "SASRec", seed=2021)
    assert p_legacy.name == "ml-1m__SASRec.json"
    assert p_tagged.name == "ml-1m__SASRec__seed2021.json"


def test_aggregate_seed_records_mean_and_std():
    import runner

    recs = [
        {
            "model": "SASRec", "dataset": "ml-1m", "seed": 2020,
            "test_result": {"ndcg@10": 0.10, "hit@10": 0.20},
            "best_valid_result": {"ndcg@10": 0.11},
            "best_valid_score": 0.11,
            "train_seconds": 100.0,
            "intensity_params": None,
            "best_params": {"lr": 0.001},
        },
        {
            "model": "SASRec", "dataset": "ml-1m", "seed": 2021,
            "test_result": {"ndcg@10": 0.12, "hit@10": 0.22},
            "best_valid_result": {"ndcg@10": 0.13},
            "best_valid_score": 0.13,
            "train_seconds": 110.0,
            "intensity_params": None,
            "best_params": {"lr": 0.001},
        },
    ]
    agg = runner.aggregate_seed_records(recs)
    assert agg["n_seeds"] == 2
    assert agg["seeds_used"] == [2020, 2021]
    assert agg["test_result"]["ndcg@10"] == pytest.approx(0.11)
    assert agg["test_result"]["hit@10"] == pytest.approx(0.21)
    # Sample std of {0.10, 0.12} = 0.014142...
    assert agg["test_result_std"]["ndcg@10"] == pytest.approx(0.01414213562, rel=1e-4)
    assert agg["train_seconds"] == pytest.approx(105.0)
    assert agg["train_seconds_std"] == pytest.approx((50.0) ** 0.5, rel=1e-4)


def test_aggregate_seed_records_intensity_params_mean():
    import runner

    recs = [
        {
            "model": "IA-SASRec-Add", "dataset": "ml-1m", "seed": 2020,
            "test_result": {"ndcg@10": 0.10},
            "train_seconds": 100.0,
            "intensity_params": {"layer_0_lambda": 0.8, "layer_1_lambda": 0.6},
        },
        {
            "model": "IA-SASRec-Add", "dataset": "ml-1m", "seed": 2021,
            "test_result": {"ndcg@10": 0.12},
            "train_seconds": 100.0,
            "intensity_params": {"layer_0_lambda": 0.9, "layer_1_lambda": 0.7},
        },
    ]
    agg = runner.aggregate_seed_records(recs)
    assert agg["intensity_params"]["layer_0_lambda"] == pytest.approx(0.85)
    assert agg["intensity_params"]["layer_1_lambda"] == pytest.approx(0.65)


def test_load_result_legacy_file_treated_as_legacy_seed(tmp_path, monkeypatch):
    """A pre-multi-seed JSON should be loadable and stamped as LEGACY_SEED."""
    import runner
    import config as cfg

    monkeypatch.setattr(cfg, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(runner, "EVAL_DIR", tmp_path)

    legacy = tmp_path / "ml-1m__SASRec.json"
    legacy.write_text(json.dumps({
        "model": "SASRec",
        "dataset": "ml-1m",
        "test_result": {"ndcg@10": 0.15},
        "train_seconds": 50.0,
        "best_params": {},
    }))

    agg = runner.load_result("ml-1m", "SASRec")
    assert agg["n_seeds"] == 1
    assert agg["seeds_used"] == [cfg.LEGACY_SEED]
    assert agg["test_result"]["ndcg@10"] == pytest.approx(0.15)


def test_has_seed_result_uses_legacy_for_legacy_seed(tmp_path, monkeypatch):
    import runner
    import config as cfg

    monkeypatch.setattr(cfg, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(runner, "EVAL_DIR", tmp_path)

    (tmp_path / "ml-1m__SASRec.json").write_text("{}")
    assert runner.has_seed_result("ml-1m", "SASRec", cfg.LEGACY_SEED) is True
    # Non-legacy seed must not be satisfied by the legacy file alone.
    assert runner.has_seed_result("ml-1m", "SASRec", cfg.LEGACY_SEED + 1) is False


def test_invalidate_cache_removes_seed_and_legacy(tmp_path, monkeypatch):
    import runner
    import config as cfg

    monkeypatch.setattr(cfg, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(runner, "EVAL_DIR", tmp_path)

    (tmp_path / "ml-1m__SASRec.json").write_text("{}")
    (tmp_path / "ml-1m__SASRec__seed2021.json").write_text("{}")
    (tmp_path / "ml-1m__SASRec__seed2022.json").write_text("{}")
    (tmp_path / "ml-1m__OtherModel__seed2020.json").write_text("{}")

    removed = runner.invalidate_cache("ml-1m", "SASRec")
    assert len(removed) == 3
    # Other model's results survive.
    assert (tmp_path / "ml-1m__OtherModel__seed2020.json").exists()
