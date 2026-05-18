"""End-to-end test that intensity actually flows through the RecBole pipeline.

This is the regression test that would have caught the bit-identical HPO
trajectory across IA-SASRec-{Add,Mul,Val}. The existing forward-pass tests in
``test_ia_sasrec_forward.py`` hand-craft an ``Interaction`` containing
``intensity_list`` and verify the model math. They cannot catch a *pipeline*
bug where RecBole silently loads the wrong .inter file (e.g. its bundled
``ml-100k`` example) and the model never sees the field at all.

We build a tiny synthetic dataset under ``tmp_path`` (with a name that is
*not* literally ``ml-100k`` so RecBole's hard-coded override doesn't fire),
let ``recbole.data.create_dataset`` + ``data_preparation`` run end-to-end,
and assert:

1. ``intensity`` and ``intensity_list`` appear in the dataset's ``field2type``.
2. A real training batch contains ``intensity_list`` with non-zero values.
3. The three IA-SASRec variants produce **pairwise different** outputs when
   given the same batch and identical seeded weight init — bit-identical
   outputs would indicate intensity was a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import config
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from models.variants.ia_sasrec import IASASRecAdd, IASASRecMul, IASASRecVal


def _write_synthetic_inter(out_dir: Path, dataset_name: str) -> None:
    """Write a small 4-column .inter with varied intensity values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # Five users, ~12 interactions each. Item IDs spread; intensity ∈ {1..5}.
    n_users, n_items = 5, 30
    t = 1_000_000
    for u in range(1, n_users + 1):
        for k in range(12):
            item = ((u * 7 + k * 3) % n_items) + 1
            intensity = ((u + k) % 5) + 1
            rows.append(f"{u}\t{item}\t{t}\t{intensity}")
            t += 1
    header = "user_id:token\titem_id:token\ttimestamp:float\tintensity:float\n"
    (out_dir / f"{dataset_name}.inter").write_text(header + "\n".join(rows) + "\n")


def _build_recbole_dataset(tmp_path: Path, dataset_name: str):
    """Build a fresh RecBole SequentialDataset from a synthetic .inter."""
    _write_synthetic_inter(tmp_path / dataset_name, dataset_name)

    # Don't go through `config.common_recbole_config` — it indexes the DATASETS
    # registry for per-dataset min-inter thresholds, and we're using a
    # synthetic dataset name that isn't registered. Mirror the fields the
    # pipeline needs inline.
    cfg_dict = {
        "data_path": str(tmp_path),
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "show_progress": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "INTENSITY_FIELD": "intensity",
        "load_col": {"inter": ["user_id", "item_id", "timestamp", "intensity"]},
        "user_inter_num_interval": "[3,inf)",
        "item_inter_num_interval": "[1,inf)",
        "eval_args": {"split": {"LS": "valid_and_test"},
                       "group_by": "user", "order": "TO", "mode": "full"},
        "metrics": ["NDCG"],
        "topk": [10],
        "valid_metric": "NDCG@10",
        "MAX_ITEM_LIST_LENGTH": 6,
        "stopping_step": 5,
        "epochs": 1,
        "loss_type": "CE",
        "train_neg_sample_args": None,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "n_layers": 1,
        "n_heads": 2,
        "hidden_size": 16,
        "inner_size": 32,
        "hidden_dropout_prob": 0.1,
        "attn_dropout_prob": 0.1,
        "learning_rate": 1e-3,
        "intensity_norm": "zscore",
        "device": "cpu",
    }

    cfg = Config(model=IASASRecAdd, dataset=dataset_name, config_dict=cfg_dict)
    init_seed(cfg["seed"], cfg["reproducibility"])
    dataset = create_dataset(cfg)
    train_data, _valid, _test = data_preparation(cfg, dataset)
    return cfg, dataset, train_data


def test_intensity_field_present_in_dataset(tmp_path):
    """RecBole must register `intensity` and create the `intensity_list`
    companion field for our `load_col` configuration."""
    _cfg, ds, _train = _build_recbole_dataset(tmp_path, "tiny-iar")
    assert "intensity" in ds.field2type, (
        f"intensity missing from field2type — RecBole likely loaded the wrong "
        f".inter file. Got: {list(ds.field2type)}"
    )
    assert "intensity_list" in ds.field2type


def test_intensity_list_in_batch_and_nonzero(tmp_path):
    """A real training batch must carry `intensity_list` populated from the
    .inter file (not zeros, not missing)."""
    _cfg, _ds, train = _build_recbole_dataset(tmp_path, "tiny-iar")
    batch = next(iter(train))
    assert "intensity_list" in batch.interaction
    w = batch["intensity_list"]
    assert w.dim() == 2
    # Some positions must carry a real intensity (>=1 in our fixture).
    assert (w > 0).any(), "intensity_list is all zero — values not propagated"


def test_variants_produce_pairwise_different_outputs(tmp_path):
    """Add / Mul / Val must yield different attention outputs on identical
    input + seed. Bit-identical outputs are the smoking gun that intensity
    has been silenced (the production bug this test guards against)."""
    cfg, ds, train = _build_recbole_dataset(tmp_path, "tiny-iar")
    batch = next(iter(train))

    def _seeded(cls):
        init_seed(cfg["seed"], cfg["reproducibility"])
        return cls(cfg, ds).to(cfg["device"]).eval()

    models = {"Add": _seeded(IASASRecAdd),
              "Mul": _seeded(IASASRecMul),
              "Val": _seeded(IASASRecVal)}

    outs = {}
    # `cfg["device"]` may be overridden by RecBole based on CUDA availability;
    # take the actual device from one of the model's parameters.
    device = next(models["Add"].parameters()).device
    batch_dev = batch.to(device)
    item_seq = batch_dev["item_id_list"]
    item_seq_len = batch_dev["item_length"]
    with torch.no_grad():
        for name, m in models.items():
            intensity = m._resolve_intensity(batch_dev, item_seq)
            assert intensity is not None, f"{name}: intensity resolved to None"
            outs[name] = m.forward(item_seq, item_seq_len, intensity=intensity)

    # Pairwise non-equality — the assertion that would have failed loudly
    # on the bug we just fixed.
    pairs = [("Add", "Mul"), ("Add", "Val"), ("Mul", "Val")]
    for a, b in pairs:
        assert not torch.allclose(outs[a], outs[b], atol=1e-6), (
            f"{a} and {b} produced bit-identical outputs — intensity is a no-op."
        )
