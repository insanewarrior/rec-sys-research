"""Unit tests for ``normalise_intensity``."""
from __future__ import annotations

import torch

from models.variants.ia_sasrec import normalise_intensity


def _mask(seq):
    return torch.tensor(seq, dtype=torch.bool)


def test_log1p_minmax_range():
    w = torch.tensor([[1.0, 2.0, 5.0, 0.0], [10.0, 100.0, 0.0, 0.0]])
    mask = _mask([[True, True, True, False], [True, True, False, False]])
    out = normalise_intensity(w, "log1p_minmax", mask)
    assert torch.all(out >= 0) and torch.all(out <= 1.0 + 1e-6)
    assert out[0, 3].item() == 0.0  # padding zeroed
    assert out[1, 2:].sum().item() == 0.0


def test_minmax_range():
    w = torch.tensor([[3.0, 1.0, 5.0, 0.0]])
    mask = _mask([[True, True, True, False]])
    out = normalise_intensity(w, "minmax", mask)
    real = out[0, :3]
    assert torch.isclose(real.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(real.max(), torch.tensor(1.0), atol=1e-6)


def test_zscore_no_nan_on_constant():
    w = torch.full((1, 4), 7.0)
    mask = _mask([[True, True, True, True]])
    out = normalise_intensity(w, "zscore", mask)
    assert torch.isfinite(out).all()


def test_log1p_minmax_no_nan_on_all_zero():
    w = torch.zeros(2, 4)
    mask = _mask([[True, True, False, False], [True, True, True, True]])
    out = normalise_intensity(w, "log1p_minmax", mask)
    assert torch.isfinite(out).all()


def test_unknown_mode_raises():
    w = torch.zeros(1, 4)
    mask = _mask([[True, True, True, True]])
    try:
        normalise_intensity(w, "bogus", mask)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")
