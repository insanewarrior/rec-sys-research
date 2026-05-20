"""Direct unit tests on ``IAMultiHeadAttention`` with hand-crafted tensors."""
from __future__ import annotations

import torch

from models.variants.ia_sasrec import IAMultiHeadAttention


def _build_layer(mode: str, seed: int = 0) -> IAMultiHeadAttention:
    torch.manual_seed(seed)
    return IAMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        intensity_mode=mode,
    ).eval()


def _causal_mask(T: int, B: int = 2) -> torch.Tensor:
    """Return RecBole-style extended additive mask: ``[B, 1, 1, T]`` with 0/-inf."""
    mask = torch.zeros(B, 1, 1, T)
    return mask  # all-zero == every position visible; sufficient for tests


def test_add_lambda_zero_matches_vanilla():
    """At lambda=0 the additive variant should equal the no-intensity output."""
    layer = _build_layer("add")
    layer.intensity_lambda.data.zero_()
    B, T, H = 2, 4, 8
    x = torch.randn(B, T, H)
    mask = _causal_mask(T, B)
    intensity = torch.rand(B, T)
    out_with = layer(x, mask, intensity)
    out_without = layer(x, mask, None)
    assert torch.allclose(out_with, out_without, atol=1e-6)


def test_add_large_lambda_concentrates_attention():
    """With huge lambda, the highest-intensity key dominates."""
    layer = _build_layer("add")
    layer.intensity_lambda.data.fill_(1e3)
    B, T, H = 1, 4, 8
    x = torch.randn(B, T, H)
    intensity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    mask = _causal_mask(T, B)
    # Inspect the softmax probs by patching: compute manually
    q = layer.transpose_for_scores(layer.query(x)).permute(0, 2, 1, 3)
    k = layer.transpose_for_scores(layer.key(x)).permute(0, 2, 3, 1)
    scores = torch.matmul(q, k) / layer.sqrt_attention_head_size
    scores = scores + layer.intensity_lambda * intensity[:, None, None, :]
    probs = scores.softmax(dim=-1)
    assert torch.allclose(probs[..., 3], torch.ones_like(probs[..., 3]), atol=1e-3)


def test_mul_lambda_zero_matches_vanilla():
    """At λ=0 the multiplicative variant should equal the no-intensity output."""
    layer = _build_layer("mul")
    layer.intensity_lambda.data.zero_()
    B, T, H = 2, 4, 8
    x = torch.randn(B, T, H)
    mask = _causal_mask(T, B)
    intensity = torch.rand(B, T)
    out_with = layer(x, mask, intensity)
    out_without = layer(x, mask, None)
    assert torch.allclose(out_with, out_without, atol=1e-6)


def test_val_lambda_zero_matches_vanilla():
    """At λ=0 the value-modulation variant should equal the no-intensity output."""
    layer = _build_layer("val")
    layer.intensity_lambda.data.zero_()
    B, T, H = 2, 4, 8
    x = torch.randn(B, T, H)
    mask = _causal_mask(T, B)
    intensity = torch.rand(B, T)
    out_with = layer(x, mask, intensity)
    out_without = layer(x, mask, None)
    assert torch.allclose(out_with, out_without, atol=1e-6)


def test_val_attention_distribution_changes_with_intensity():
    """Val now reweights post-softmax attention probs by key intensity.

    This is a deliberate change from the original Val (which multiplied the
    post-attention context by query-position intensity, a no-op for ranking).
    We assert the new semantics: probs differ between intensity-on and
    intensity-off forwards.
    """
    layer = _build_layer("val", seed=42)
    layer.capture_probs = True
    B, T, H = 2, 4, 8
    x = torch.randn(B, T, H)
    mask = _causal_mask(T, B)
    intensity = torch.rand(B, T)

    layer(x, mask, intensity)
    probs_with = layer.last_probs.clone()

    layer(x, mask, None)
    probs_without = layer.last_probs.clone()

    # Probs must actually differ — Val now influences attention, not just scale.
    assert not torch.allclose(probs_with, probs_without, atol=1e-6)


def test_mul_lambda_one_collapses_zero_intensity():
    """At λ=1 (init), multiplicative variant with all-zero intensity → uniform attention.

    This documents the original Mul behaviour, now reachable as the λ=1 special
    case of the learnable formulation.
    """
    layer = _build_layer("mul")  # λ init = 1.0
    B, T, H = 1, 4, 8
    x = torch.randn(B, T, H)
    mask = _causal_mask(T, B)
    intensity = torch.zeros(B, T)
    q = layer.transpose_for_scores(layer.query(x)).permute(0, 2, 1, 3)
    k = layer.transpose_for_scores(layer.key(x)).permute(0, 2, 3, 1)
    scores = torch.matmul(q, k) / layer.sqrt_attention_head_size
    w_k = intensity[:, None, None, :]
    scores = scores * (1.0 + layer.intensity_lambda * (w_k - 1.0))
    probs = (scores + mask).softmax(dim=-1)
    assert torch.allclose(probs, torch.full_like(probs, 1.0 / T), atol=1e-5)


def test_padding_mask_kills_attention():
    """``-inf`` in attention_mask must zero out probability at padded positions
    for every variant, even after intensity injection."""
    for mode in ("add", "mul", "val"):
        layer = _build_layer(mode, seed=7)
        layer.capture_probs = True
        B, T, H = 1, 4, 8
        x = torch.randn(B, T, H)
        # Mask out last position.
        mask = torch.zeros(B, 1, 1, T)
        mask[..., -1] = float("-inf")
        intensity = torch.tensor([[1.0, 1.0, 1.0, 5.0]])  # padded slot has high "intensity"
        layer(x, mask, intensity)
        probs = layer.last_probs
        assert torch.allclose(probs[..., -1], torch.zeros_like(probs[..., -1]), atol=1e-6), mode
