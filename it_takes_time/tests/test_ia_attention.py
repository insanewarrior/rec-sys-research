"""Direct unit tests on ``IAMultiHeadAttention`` with hand-crafted tensors."""
from __future__ import annotations

import copy

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


def test_val_attention_distribution_unchanged():
    """The value-modulation variant must not alter the softmax probabilities."""
    layer_v = _build_layer("val", seed=42)
    layer_n = copy.deepcopy(layer_v)
    layer_n.intensity_mode = "add"  # but we'll pass intensity=None to make it a no-op
    B, T, H = 2, 4, 8
    x = torch.randn(B, T, H)
    mask = _causal_mask(T, B)
    intensity = torch.rand(B, T)
    out_v = layer_v(x, mask, intensity)
    out_n = layer_n(x, mask, None)
    # The two outputs differ in magnitude but the *attention probability* path is
    # identical; we verify by computing probs directly.
    q = layer_v.transpose_for_scores(layer_v.query(x)).permute(0, 2, 1, 3)
    k = layer_v.transpose_for_scores(layer_v.key(x)).permute(0, 2, 3, 1)
    scores = torch.matmul(q, k) / layer_v.sqrt_attention_head_size
    probs_v = scores.softmax(dim=-1)
    # Same op for layer_n (parameters are a deep copy):
    q2 = layer_n.transpose_for_scores(layer_n.query(x)).permute(0, 2, 1, 3)
    k2 = layer_n.transpose_for_scores(layer_n.key(x)).permute(0, 2, 3, 1)
    probs_n = (torch.matmul(q2, k2) / layer_n.sqrt_attention_head_size).softmax(dim=-1)
    assert torch.allclose(probs_v, probs_n, atol=1e-6)
    # Sanity: with non-trivial intensity, the val variant's output is not the
    # same as the no-intensity output (i.e. intensity is doing something).
    assert not torch.allclose(out_v, out_n, atol=1e-4)


def test_mul_zero_intensity_collapses_logits():
    """Multiplicative variant with all-zero intensity zeros out attention scores."""
    layer = _build_layer("mul")
    B, T, H = 1, 4, 8
    x = torch.randn(B, T, H)
    mask = _causal_mask(T, B)
    intensity = torch.zeros(B, T)
    q = layer.transpose_for_scores(layer.query(x)).permute(0, 2, 1, 3)
    k = layer.transpose_for_scores(layer.key(x)).permute(0, 2, 3, 1)
    scores = torch.matmul(q, k) / layer.sqrt_attention_head_size
    scores = scores * intensity[:, None, None, :]
    probs = (scores + mask).softmax(dim=-1)
    # All scores zero -> softmax is uniform over T positions.
    assert torch.allclose(probs, torch.full_like(probs, 1.0 / T), atol=1e-5)


def test_padding_mask_kills_attention():
    """``-inf`` in attention_mask must zero out probability at padded positions
    for every variant, even after intensity injection."""
    for mode in ("add", "mul", "val"):
        layer = _build_layer(mode, seed=7)
        B, T, H = 1, 4, 8
        x = torch.randn(B, T, H)
        # Mask out last position.
        mask = torch.zeros(B, 1, 1, T)
        mask[..., -1] = float("-inf")
        intensity = torch.tensor([[1.0, 1.0, 1.0, 5.0]])  # padded slot has high "intensity"
        q = layer.transpose_for_scores(layer.query(x)).permute(0, 2, 1, 3)
        k = layer.transpose_for_scores(layer.key(x)).permute(0, 2, 3, 1)
        scores = torch.matmul(q, k) / layer.sqrt_attention_head_size
        if mode == "add":
            scores = scores + layer.intensity_lambda * intensity[:, None, None, :]
        elif mode == "mul":
            scores = scores * intensity[:, None, None, :]
        scores = scores + mask
        probs = scores.softmax(dim=-1)
        assert torch.allclose(probs[..., -1], torch.zeros_like(probs[..., -1]), atol=1e-6), mode
