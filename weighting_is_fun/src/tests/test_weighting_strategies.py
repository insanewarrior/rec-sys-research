import numpy as np
import pytest
from scipy.sparse import csr_matrix

from weighting_strategies import (
    bm25_weight,
    log_idf_weight,
    log_weight,
    normalized_weight,
    pmi_weight,
    power_lift_weight,
    power_lift_weight_slow,
    power_weight,
    robust_user_centric_weight,
    robust_user_centric_weight_slow,
    robust_user_centric_weight_v2,
    robust_user_centric_weight_v2_slow,
    sigmoid_propensity_weight,
    sigmoid_propensity_weight_slow,
    tfidf_weight,
)


@pytest.fixture
def X():
    """5 users x 6 items with varied interaction counts."""
    data = np.array([1, 5, 2, 3, 10, 4, 7, 1, 2, 8, 3, 6], dtype=np.float64)
    rows = np.array([0, 0, 1, 1,  2, 2, 2, 3, 3, 4, 4, 4])
    cols = np.array([0, 2, 1, 3,  0, 1, 4, 2, 3, 0, 1, 5])
    return csr_matrix((data, (rows, cols)), shape=(5, 6))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_valid_output(result, original):
    assert isinstance(result, csr_matrix)
    assert result.shape == original.shape
    assert result.nnz == original.nnz


# ---------------------------------------------------------------------------
# Output shape / type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    tfidf_weight,
    normalized_weight,
    bm25_weight,
    log_weight,
    log_idf_weight,
    power_weight,
    pmi_weight,
    power_lift_weight,
])
def test_output_is_valid_csr(fn, X):
    assert_valid_output(fn(X.copy()), X)


def test_robust_user_centric_output(X):
    assert_valid_output(robust_user_centric_weight(X.copy()), X)


def test_robust_user_centric_v2_output(X):
    assert_valid_output(robust_user_centric_weight_v2(X.copy()), X)


def test_sigmoid_propensity_output(X):
    assert_valid_output(sigmoid_propensity_weight(X.copy()), X)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_log_weight_values(X):
    result = log_weight(X.copy())
    np.testing.assert_allclose(result.data, np.log1p(X.data))


def test_power_weight_values(X):
    result = power_weight(X.copy(), p=2.0)
    np.testing.assert_allclose(result.data, X.data ** 2)


def test_power_weight_identity_at_p1(X):
    result = power_weight(X.copy(), p=1.0)
    np.testing.assert_allclose(result.data, X.data)


def test_normalized_weight_unit_row_norms(X):
    result = normalized_weight(X.copy())
    row_norms = np.sqrt(np.array(result.power(2).sum(axis=1)).flatten())
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-10)


def test_pmi_weight_non_negative(X):
    result = pmi_weight(X.copy())
    assert (result.data >= 0).all()


def test_robust_user_centric_sigmoid_range(X):
    result = robust_user_centric_weight(X.copy())
    assert (result.data > 0).all()
    assert (result.data < 1).all()


def test_robust_user_centric_v2_sigmoid_range(X):
    result = robust_user_centric_weight_v2(X.copy())
    assert (result.data > 0).all()
    assert (result.data < 1).all()


def test_bm25_custom_params(X):
    assert_valid_output(bm25_weight(X.copy(), K1=50, B=0.5), X)


def test_log_idf_custom_alpha(X):
    assert_valid_output(log_idf_weight(X.copy(), alpha=1.0), X)


# ---------------------------------------------------------------------------
# Fast vs slow equivalence
# ---------------------------------------------------------------------------

def test_power_lift_fast_vs_slow(X):
    fast = power_lift_weight(X.copy())
    slow = power_lift_weight_slow(X.copy())
    np.testing.assert_allclose(fast.toarray(), slow.toarray(), rtol=1e-5)


def test_robust_user_centric_fast_vs_slow(X):
    fast = robust_user_centric_weight(X.copy())
    slow = robust_user_centric_weight_slow(X.copy())
    np.testing.assert_allclose(fast.toarray(), slow.toarray(), atol=1e-10)


def test_robust_user_centric_v2_fast_vs_slow(X):
    fast = robust_user_centric_weight_v2(X.copy())
    slow = robust_user_centric_weight_v2_slow(X.copy())
    np.testing.assert_allclose(fast.toarray(), slow.toarray(), atol=1e-10)


def test_sigmoid_propensity_fast_vs_slow(X):
    fast = sigmoid_propensity_weight(X.copy())
    slow = sigmoid_propensity_weight_slow(X.copy())
    np.testing.assert_allclose(fast.toarray(), slow.toarray(), rtol=1e-5)
