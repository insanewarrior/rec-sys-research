import numpy as np
import pandas as pd
import scipy.sparse as sp


def transform_dataframe_to_sparse(
    df: pd.DataFrame, row_field: str, col_field: str, data_field: str
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Transform a pandas dataframe into a scipy sparse matrix representation.

    Parameters:
        df: Input DataFrame containing data that satisfies
            the relationship ``sparse_matrix[df[row_field][k], df[col_field][k]] = df[data_field][k]``.
        row_field: Name of the column representing rows in sparse matrix.
        col_field: Name of the column representing columns in sparse matrix.
        data_field: Name of the column representing data values in sparse matrix.

    Returns:
        Tuple of (sp.csr_matrix, np.ndarray, np.ndarray):
            csr_matrix: Scipy sparse matrix of shape (n_rows, n_cols),
                where `n_rows` is the number of unique values in `row_field`
                and `n_cols` is the number of unique values in `col_field`.
            np.ndarray: Array of real row ids corresponding to the indices in the sparse matrix.
            np.ndarray: Array of real column ids corresponding to the indices in the sparse matrix.

    Examples:
        >>> import pandas as pd
        >>> data = {"user_id": [1, 2, 1], "item_id": [3, 3, 4], "interactions": [5, 3, 1]}
        >>> df = pd.DataFrame(data)
        >>> df
           user_id  item_id  interactions
        0        1        3             5
        1        2        3             3
        2        1        4             1
        >>> sparse_matrix, row_mapping, col_mapping = transform_dataframe_to_sparse(
        ...     df, row_field="user_id", col_field="item_id", data_field="interactions"
        ... )
        >>> sparse_matrix.toarray()
        array([[5., 1.],
               [3., 0.]])
        >>> row_mapping
        array([1, 2])
        >>> col_mapping
        array([3, 4])
    """
    row_ind = df[row_field].astype("category").cat
    col_ind = df[col_field].astype("category").cat

    sparse_matrix = sp.csr_matrix(
        (df[data_field], (row_ind.codes, col_ind.codes)),
        shape=(row_ind.categories.size, col_ind.categories.size),
        dtype=np.float64,
    )
    return sparse_matrix, row_ind.categories.to_numpy(), col_ind.categories.to_numpy()
