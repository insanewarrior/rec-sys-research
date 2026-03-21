import numpy as np
import pytest
from scipy.sparse import csr_matrix

from autorec import AutoRec


@pytest.fixture
def small_matrix():
    """8 users x 10 items — explicit to keep tests deterministic."""
    data = [3, 7, 1, 5, 9, 2, 4, 8, 6, 1, 3, 7]
    rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7]
    cols = [0, 3, 1, 5, 2, 7, 4, 8, 6, 9, 0, 1]
    return csr_matrix((data, (rows, cols)), shape=(8, 10), dtype=np.float32)


@pytest.fixture
def fitted_model(small_matrix):
    model = AutoRec(hidden_dim=5, epochs=2, batch_size=4)
    model.fit(small_matrix)
    return model, small_matrix


# ---------------------------------------------------------------------------
# Pre-fit guard
# ---------------------------------------------------------------------------

def test_recommend_before_fit_raises():
    model = AutoRec()
    with pytest.raises(ValueError, match="Model must be fit"):
        model.recommend(0, csr_matrix((1, 5)))


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

def test_fit_sets_predictions(small_matrix):
    model = AutoRec(hidden_dim=5, epochs=2, batch_size=4)
    model.fit(small_matrix)
    assert model.predictions is not None
    assert model.predictions.shape == small_matrix.shape


# ---------------------------------------------------------------------------
# recommend() — single user
# ---------------------------------------------------------------------------

def test_recommend_returns_n_items(fitted_model):
    model, matrix = fitted_model
    ids, scores = model.recommend(0, matrix[0], N=5)
    assert len(ids) == 5
    assert len(scores) == 5


def test_recommend_filters_liked_items(fitted_model):
    model, matrix = fitted_model
    user_row = matrix[0]
    liked = set(user_row.indices)
    ids, _ = model.recommend(0, user_row, N=5, filter_already_liked_items=True)
    assert liked.isdisjoint(ids)


def test_recommend_no_filter_may_include_liked(fitted_model):
    model, matrix = fitted_model
    # With filter off, recommend should still return N items without error.
    ids, scores = model.recommend(0, matrix[0], N=5, filter_already_liked_items=False)
    assert len(ids) == 5


def test_recommend_scores_descending(fitted_model):
    model, matrix = fitted_model
    _, scores = model.recommend(0, matrix[0], N=5)
    assert list(scores) == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# recommend() — batch
# ---------------------------------------------------------------------------

def test_recommend_batch_shape(fitted_model):
    model, matrix = fitted_model
    user_ids = [0, 1, 2]
    user_rows = [matrix[i] for i in user_ids]
    ids, scores = model.recommend(user_ids, user_rows, N=3)
    assert ids.shape == (3, 3)
    assert scores.shape == (3, 3)
