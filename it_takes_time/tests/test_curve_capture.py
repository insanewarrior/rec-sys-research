"""Tests for ``runner._attach_curve_capture``.

The capture wrapper is meant to be a transparent observer of a RecBole
``Trainer``'s ``_valid_epoch`` / ``_train_epoch`` methods: it must (a) not
alter what those methods return, (b) append a faithful record per call,
(c) handle the different loss shapes RecBole emits (scalar, tuple).

We don't spin up a real RecBole training loop here — we hand-roll a tiny
trainer stub with the two methods of interest and exercise the wrapper
directly.
"""
from __future__ import annotations

import pytest


class _StubTrainer:
    """Mimics the surface area of ``recbole.trainer.Trainer`` that the
    capture wrapper relies on."""

    def __init__(self, valid_scores, train_losses):
        self._valid_idx = 0
        self._train_idx = 0
        self._valid_scores = valid_scores
        self._train_losses = train_losses

    def _valid_epoch(self, valid_data, show_progress=False):
        score = self._valid_scores[self._valid_idx]
        self._valid_idx += 1
        result = {"ndcg@10": score, "hit@10": score * 2}
        return score, result

    def _train_epoch(self, train_data, epoch_idx, show_progress=False):
        loss = self._train_losses[self._train_idx]
        self._train_idx += 1
        return loss


def _run_fit(trainer, n_epochs):
    """Simulate Trainer.fit's per-epoch loop: train then validate each epoch."""
    for i in range(n_epochs):
        trainer._train_epoch(None, i)
        trainer._valid_epoch(None)


def test_valid_curve_records_each_call():
    import runner

    t = _StubTrainer(
        valid_scores=[0.10, 0.12, 0.15, 0.14],
        train_losses=[1.5, 1.2, 1.0, 0.9],
    )
    valid_curve, train_losses = runner._attach_curve_capture(t)

    _run_fit(t, n_epochs=4)

    assert len(valid_curve) == 4
    assert [r["epoch"] for r in valid_curve] == [0, 1, 2, 3]
    assert [r["valid_score"] for r in valid_curve] == pytest.approx([0.10, 0.12, 0.15, 0.14])
    # Per-epoch valid_result is copied through unchanged.
    assert valid_curve[2]["valid_result"]["ndcg@10"] == pytest.approx(0.15)
    assert valid_curve[2]["valid_result"]["hit@10"] == pytest.approx(0.30)


def test_train_losses_record_each_call():
    import runner

    t = _StubTrainer(valid_scores=[0.1, 0.1], train_losses=[2.5, 1.5])
    _curve, train_losses = runner._attach_curve_capture(t)
    _run_fit(t, n_epochs=2)
    assert train_losses == pytest.approx([2.5, 1.5])


def test_train_losses_handle_tuple_returns():
    """Some RecBole models return a tuple of per-component losses
    (e.g. NeuMF)."""
    import runner

    t = _StubTrainer(valid_scores=[0.1, 0.1], train_losses=[(1.0, 0.2), (0.8, 0.1)])
    _curve, train_losses = runner._attach_curve_capture(t)
    _run_fit(t, n_epochs=2)
    assert train_losses[0] == pytest.approx([1.0, 0.2])
    assert train_losses[1] == pytest.approx([0.8, 0.1])


def test_capture_does_not_alter_return_values():
    """Wrapper must be transparent — calling code (Trainer.fit) gets the
    same values back as if the wrapper weren't there."""
    import runner

    t = _StubTrainer(valid_scores=[0.42], train_losses=[3.14])
    runner._attach_curve_capture(t)

    loss = t._train_epoch(None, 0)
    score, result = t._valid_epoch(None)
    assert loss == pytest.approx(3.14)
    assert score == pytest.approx(0.42)
    assert result == {"ndcg@10": pytest.approx(0.42), "hit@10": pytest.approx(0.84)}


def test_capture_lists_are_independent_per_attach():
    """Re-attaching to a fresh trainer gives a fresh pair of empty lists —
    no leakage between training runs."""
    import runner

    t1 = _StubTrainer(valid_scores=[0.1], train_losses=[1.0])
    curve_a, losses_a = runner._attach_curve_capture(t1)
    _run_fit(t1, n_epochs=1)

    t2 = _StubTrainer(valid_scores=[0.2, 0.3], train_losses=[2.0, 3.0])
    curve_b, losses_b = runner._attach_curve_capture(t2)
    _run_fit(t2, n_epochs=2)

    assert len(curve_a) == 1
    assert len(curve_b) == 2
    assert losses_a == pytest.approx([1.0])
    assert losses_b == pytest.approx([2.0, 3.0])
