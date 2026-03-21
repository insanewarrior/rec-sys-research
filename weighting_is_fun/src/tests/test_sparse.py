import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from utils.sparse import transform_dataframe_to_sparse


@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "user_id":      [1, 2, 1],
        "item_id":      [3, 3, 4],
        "interactions": [5, 3, 1],
    })


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

def test_returns_tuple_of_three(simple_df):
    result = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    assert isinstance(result, tuple) and len(result) == 3


def test_matrix_is_csr(simple_df):
    matrix, _, _ = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    assert isinstance(matrix, sp.csr_matrix)


def test_mappings_are_ndarrays(simple_df):
    _, row_map, col_map = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    assert isinstance(row_map, np.ndarray)
    assert isinstance(col_map, np.ndarray)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_shape_matches_unique_counts(simple_df):
    matrix, row_map, col_map = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    assert matrix.shape == (2, 2)   # 2 unique users, 2 unique items
    assert len(row_map) == 2
    assert len(col_map) == 2


def test_shape_single_row():
    df = pd.DataFrame({"u": [1], "i": [1], "v": [7]})
    matrix, row_map, col_map = transform_dataframe_to_sparse(df, "u", "i", "v")
    assert matrix.shape == (1, 1)


# ---------------------------------------------------------------------------
# Mapping correctness
# ---------------------------------------------------------------------------

def test_row_mapping_contains_all_users(simple_df):
    _, row_map, _ = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    assert set(row_map) == {1, 2}


def test_col_mapping_contains_all_items(simple_df):
    _, _, col_map = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    assert set(col_map) == {3, 4}


def test_mappings_are_sorted(simple_df):
    _, row_map, col_map = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    assert list(row_map) == sorted(row_map)
    assert list(col_map) == sorted(col_map)


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------

def test_matrix_values_match_docstring_example(simple_df):
    matrix, _, _ = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    expected = np.array([[5.0, 1.0], [3.0, 0.0]])
    np.testing.assert_array_equal(matrix.toarray(), expected)


def test_non_zero_count(simple_df):
    matrix, _, _ = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    assert matrix.nnz == 3


def test_data_dtype_is_float(simple_df):
    matrix, _, _ = transform_dataframe_to_sparse(simple_df, "user_id", "item_id", "interactions")
    assert np.issubdtype(matrix.dtype, np.floating)


# ---------------------------------------------------------------------------
# String keys
# ---------------------------------------------------------------------------

def test_string_ids():
    df = pd.DataFrame({
        "user": ["alice", "bob", "alice"],
        "item": ["x",     "x",   "y"],
        "val":  [2,        4,     1],
    })
    matrix, row_map, col_map = transform_dataframe_to_sparse(df, "user", "item", "val")
    assert matrix.shape == (2, 2)
    assert set(row_map) == {"alice", "bob"}
    assert set(col_map) == {"x", "y"}


# ---------------------------------------------------------------------------
# Large / sparse
# ---------------------------------------------------------------------------

def test_sparse_matrix_has_correct_nnz():
    # 100 users x 100 items, only 50 interactions
    rng = np.random.default_rng(0)
    users = rng.integers(0, 100, size=50)
    items = rng.integers(0, 100, size=50)
    vals  = rng.integers(1, 10,  size=50).astype(float)
    df = pd.DataFrame({"u": users, "i": items, "v": vals})
    df = df.drop_duplicates(subset=["u", "i"])  # avoid duplicate entries
    matrix, _, _ = transform_dataframe_to_sparse(df, "u", "i", "v")
    assert matrix.nnz == len(df)
