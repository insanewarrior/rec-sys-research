"""Unit tests for ``evaluation.paired_significance`` and ``significance_summary``.

These bypass real eval JSONs by monkeypatching ``runner.load_result`` to return
hand-constructed per-seed records. We verify:
  - the math (mean_diff, n, t-test, Wilcoxon) against scipy ground truth,
  - seed alignment when challenger and baseline don't share every seed,
  - metric filtering,
  - the ``significance_summary`` cell formatting / starring logic,
  - graceful behaviour on missing models and self-comparisons.
"""
from __future__ import annotations

import pytest


def _record(model: str, dataset: str, per_seed: list[dict]) -> dict:
    """Mimic the aggregated record shape produced by runner.load_result."""
    return {
        "model": model,
        "dataset": dataset,
        "per_seed": per_seed,
        "test_result": {},  # not used by significance code
    }


def _seed(seed: int, **metrics) -> dict:
    return {"seed": seed, "test_result": dict(metrics)}


def _make_loader(records: dict[tuple[str, str], dict]):
    """Return a ``load_result(ds, model)`` stand-in that reads from a dict."""
    def _load(dataset: str, model: str):
        try:
            return records[(dataset, model)]
        except KeyError:
            raise FileNotFoundError(f"no record for ({dataset!r}, {model!r})")
    return _load


def test_mean_diff_and_n_match_per_seed_data(monkeypatch):
    import runner
    import evaluation as ev

    records = {
        ("ds", "SASRec"): _record("SASRec", "ds", [
            _seed(2020, **{"ndcg@10": 0.10}),
            _seed(2021, **{"ndcg@10": 0.12}),
            _seed(2022, **{"ndcg@10": 0.14}),
        ]),
        ("ds", "IA"): _record("IA", "ds", [
            _seed(2020, **{"ndcg@10": 0.11}),
            _seed(2021, **{"ndcg@10": 0.13}),
            _seed(2022, **{"ndcg@10": 0.13}),
        ]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    df = ev.paired_significance("ds", "SASRec", ["IA"], metrics=["ndcg@10"])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n"] == 3
    # Diffs: +0.01, +0.01, -0.01 → mean = +0.01/3 ≈ 0.003333
    assert row["mean_diff"] == pytest.approx(0.01 / 3, rel=1e-6)
    assert row["baseline_mean"] == pytest.approx(0.12)
    assert row["challenger_mean"] == pytest.approx((0.11 + 0.13 + 0.13) / 3)
    assert row["per_seed_diffs"] == [0.01, 0.01, -0.01]


def test_paired_ttest_matches_scipy(monkeypatch):
    """The reported p-value must equal scipy.stats.ttest_rel on the same arrays."""
    import runner
    import evaluation as ev
    from scipy import stats as sps

    base_vals = [0.10, 0.11, 0.09, 0.12, 0.10]
    chal_vals = [0.11, 0.13, 0.10, 0.13, 0.12]
    records = {
        ("ds", "B"): _record("B", "ds", [
            _seed(2020 + i, **{"m": v}) for i, v in enumerate(base_vals)
        ]),
        ("ds", "C"): _record("C", "ds", [
            _seed(2020 + i, **{"m": v}) for i, v in enumerate(chal_vals)
        ]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    df = ev.paired_significance("ds", "B", ["C"], metrics=["m"])
    row = df.iloc[0]
    expected_t = float(sps.ttest_rel(chal_vals, base_vals).pvalue)
    expected_w = float(sps.wilcoxon(chal_vals, base_vals, zero_method="zsplit").pvalue)
    assert row["t_pvalue"] == pytest.approx(expected_t, rel=1e-9)
    assert row["wilcoxon_pvalue"] == pytest.approx(expected_w, rel=1e-9)


def test_pairs_only_on_common_seeds(monkeypatch):
    """If challenger and baseline share only a subset of seeds, only those pair."""
    import runner
    import evaluation as ev

    records = {
        ("ds", "B"): _record("B", "ds", [
            _seed(2020, **{"m": 0.10}),
            _seed(2021, **{"m": 0.11}),
            _seed(2022, **{"m": 0.12}),
        ]),
        ("ds", "C"): _record("C", "ds", [
            _seed(2020, **{"m": 0.11}),
            _seed(2099, **{"m": 0.30}),  # disjoint seed — should be ignored
            _seed(2021, **{"m": 0.13}),
        ]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    df = ev.paired_significance("ds", "B", ["C"], metrics=["m"])
    row = df.iloc[0]
    assert row["n"] == 2  # only seeds 2020 and 2021 paired
    assert row["mean_diff"] == pytest.approx(((0.11 - 0.10) + (0.13 - 0.11)) / 2)


def test_skips_when_too_few_paired_seeds(monkeypatch):
    """With fewer than 2 paired seeds we can't run a paired test — row is omitted."""
    import runner
    import evaluation as ev

    records = {
        ("ds", "B"): _record("B", "ds", [_seed(2020, **{"m": 0.10})]),
        ("ds", "C"): _record("C", "ds", [_seed(2020, **{"m": 0.11})]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    df = ev.paired_significance("ds", "B", ["C"], metrics=["m"])
    assert df.empty


def test_metric_filter(monkeypatch):
    """When `metrics` is given, only those are tested even if more are present."""
    import runner
    import evaluation as ev

    records = {
        ("ds", "B"): _record("B", "ds", [
            _seed(2020, **{"m1": 0.10, "m2": 0.20}),
            _seed(2021, **{"m1": 0.11, "m2": 0.21}),
        ]),
        ("ds", "C"): _record("C", "ds", [
            _seed(2020, **{"m1": 0.11, "m2": 0.20}),
            _seed(2021, **{"m1": 0.12, "m2": 0.21}),
        ]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    df = ev.paired_significance("ds", "B", ["C"], metrics=["m1"])
    assert set(df["metric"]) == {"m1"}


def test_default_metrics_use_baseline_keys(monkeypatch):
    """When `metrics=None`, every metric present in the baseline is tested."""
    import runner
    import evaluation as ev

    records = {
        ("ds", "B"): _record("B", "ds", [
            _seed(2020, **{"m1": 0.10, "m2": 0.20}),
            _seed(2021, **{"m1": 0.11, "m2": 0.21}),
        ]),
        ("ds", "C"): _record("C", "ds", [
            _seed(2020, **{"m1": 0.11, "m2": 0.20}),
            _seed(2021, **{"m1": 0.12, "m2": 0.21}),
        ]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    df = ev.paired_significance("ds", "B", ["C"])
    assert set(df["metric"]) == {"m1", "m2"}


def test_self_comparison_is_skipped(monkeypatch):
    """Passing the baseline name itself among challengers must not produce a row."""
    import runner
    import evaluation as ev

    records = {
        ("ds", "B"): _record("B", "ds", [
            _seed(2020, **{"m": 0.10}),
            _seed(2021, **{"m": 0.11}),
        ]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    df = ev.paired_significance("ds", "B", ["B"], metrics=["m"])
    assert df.empty


def test_missing_challenger_is_skipped(monkeypatch):
    """A challenger model with no eval record is silently dropped, not raised."""
    import runner
    import evaluation as ev

    records = {
        ("ds", "B"): _record("B", "ds", [
            _seed(2020, **{"m": 0.10}),
            _seed(2021, **{"m": 0.11}),
        ]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    df = ev.paired_significance("ds", "B", ["nope", "also-nope"], metrics=["m"])
    assert df.empty


def test_wilcoxon_handles_identical_arrays(monkeypatch):
    """When all paired diffs are zero, Wilcoxon raises — we must return p=1.0 instead."""
    import runner
    import evaluation as ev

    records = {
        ("ds", "B"): _record("B", "ds", [
            _seed(2020, **{"m": 0.10}),
            _seed(2021, **{"m": 0.11}),
            _seed(2022, **{"m": 0.12}),
        ]),
        ("ds", "C"): _record("C", "ds", [
            _seed(2020, **{"m": 0.10}),
            _seed(2021, **{"m": 0.11}),
            _seed(2022, **{"m": 0.12}),
        ]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    df = ev.paired_significance("ds", "B", ["C"], metrics=["m"])
    row = df.iloc[0]
    assert row["mean_diff"] == 0.0
    assert row["wilcoxon_pvalue"] == 1.0


def test_significance_summary_marks_significant_cells(monkeypatch):
    """`significance_summary` adds a `*` to cells where the paired t-test crosses alpha."""
    import runner
    import evaluation as ev

    # Construct a case where the diff is large and consistent → very small p.
    base_vals = [0.10, 0.10, 0.10, 0.10, 0.10]
    chal_vals = [0.20, 0.20, 0.20, 0.20, 0.20]
    records = {
        ("ds", "B"): _record("B", "ds", [_seed(2020 + i, **{"m": v}) for i, v in enumerate(base_vals)]),
        ("ds", "C_sig"): _record("C_sig", "ds", [_seed(2020 + i, **{"m": v}) for i, v in enumerate(chal_vals)]),
        # And one where diffs are zero → p = 1.0 → no star.
        ("ds", "C_ns"): _record("C_ns", "ds", [_seed(2020 + i, **{"m": v}) for i, v in enumerate(base_vals)]),
    }
    monkeypatch.setattr(runner, "load_result", _make_loader(records))

    wide = ev.significance_summary("ds", "B", ["C_sig", "C_ns"], metrics=["m"], alpha=0.05)
    assert wide.loc["C_sig", "m"].endswith("*")
    assert wide.loc["C_ns", "m"].endswith(" ")
    # Sign in the formatted cell should match the actual diff direction.
    assert "+" in wide.loc["C_sig", "m"]
