"""Forward + backward smoke tests for the three IA-SASRec variants.

These bypass the full RecBole data pipeline by hand-rolling a minimal config
dict and a stub dataset; the model itself is built end-to-end with real
PyTorch parameters.
"""
from __future__ import annotations

import pytest
import torch
from recbole.data.interaction import Interaction

from models.variants.ia_sasrec import IASASRecAdd, IASASRecMul, IASASRecVal


class _StubDataset:
    """Implements just the surface area SASRec.__init__ touches."""

    def __init__(self, n_items: int):
        self._n_items = n_items

    def num(self, _field: str) -> int:
        return self._n_items


def _make_config(**overrides):
    cfg = {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "INTENSITY_FIELD": "intensity",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "NEG_PREFIX": "neg_",
        "MAX_ITEM_LIST_LENGTH": 8,
        "device": "cpu",
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": 16,
        "inner_size": 32,
        "hidden_dropout_prob": 0.1,
        "attn_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "loss_type": "CE",
        "intensity_norm": "log1p_minmax",
    }
    cfg.update(overrides)

    # SASRec accesses config via ``config[key]``; a plain dict works.
    return cfg


def _make_batch(B: int = 3, T: int = 8, n_items: int = 50) -> Interaction:
    torch.manual_seed(0)
    item_seq = torch.randint(1, n_items, (B, T))
    # Pad the tail of each row randomly to mimic variable-length sequences.
    lengths = torch.tensor([T, T - 2, T - 5])
    for i, L in enumerate(lengths.tolist()):
        item_seq[i, L:] = 0
    intensity_list = torch.rand(B, T) * 5.0
    intensity_list[item_seq == 0] = 0.0
    pos_items = torch.randint(1, n_items, (B,))
    return Interaction({
        "item_id_list": item_seq,
        "item_length": lengths,
        "item_id": pos_items,
        "intensity_list": intensity_list,
    })


@pytest.mark.parametrize("cls", [IASASRecAdd, IASASRecMul, IASASRecVal])
def test_forward_shapes(cls):
    n_items = 50
    model = cls(_make_config(), _StubDataset(n_items)).eval()
    batch = _make_batch(B=3, T=8, n_items=n_items)
    item_seq = batch["item_id_list"]
    item_seq_len = batch["item_length"]
    intensity = model._resolve_intensity(batch, item_seq)
    out = model(item_seq, item_seq_len, intensity=intensity)
    assert out.shape == (3, model.hidden_size)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cls", [IASASRecAdd, IASASRecMul, IASASRecVal])
def test_loss_and_backward(cls):
    n_items = 50
    model = cls(_make_config(), _StubDataset(n_items))
    batch = _make_batch(B=3, T=8, n_items=n_items)
    loss = model.calculate_loss(batch)
    assert torch.isfinite(loss), "loss must be finite"
    loss.backward()
    # Item embedding receives gradient.
    assert model.item_embedding.weight.grad is not None
    assert torch.isfinite(model.item_embedding.weight.grad).all()


def test_add_variant_lambda_receives_gradient():
    n_items = 50
    model = IASASRecAdd(_make_config(), _StubDataset(n_items))
    batch = _make_batch(B=3, T=8, n_items=n_items)
    loss = model.calculate_loss(batch)
    loss.backward()
    # First-layer lambda should pick up a non-None gradient.
    lam = model.trm_encoder.layer[0].multi_head_attention.intensity_lambda
    assert lam.grad is not None
    assert torch.isfinite(lam.grad).all()


def test_missing_intensity_field_falls_back_gracefully():
    """If the dataloader is missing ``intensity_list`` the model should still
    produce a finite forward pass (the attention reverts to vanilla)."""
    n_items = 50
    model = IASASRecAdd(_make_config(), _StubDataset(n_items)).eval()
    item_seq = torch.randint(1, n_items, (2, 8))
    item_seq_len = torch.tensor([8, 6])
    batch = Interaction({"item_id_list": item_seq, "item_length": item_seq_len,
                          "item_id": torch.randint(1, n_items, (2,))})
    intensity = model._resolve_intensity(batch, item_seq)
    assert intensity is None
    out = model(item_seq, item_seq_len, intensity=None)
    assert torch.isfinite(out).all()
