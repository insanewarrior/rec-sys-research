import numpy as np
from numpy import bincount, log, log1p, sqrt, array
from scipy.sparse import coo_matrix, csr_matrix


def tfidf_weight(X) -> csr_matrix:
    """
    Weighs each row of a sparse matrix X by TF-IDF weighting.

    Parameters:
        X: The item-user interaction matrix.

    Returns:
        The TF-IDF weighted sparse matrix.
    """
    # convert to COO format
    X = coo_matrix(X)

    # calculate IDF
    N = float(X.shape[0])
    idf = log(N) - log1p(bincount(X.col))

    # apply TF-IDF adjustment
    X.data = sqrt(X.data) * idf[X.col]
    return X.tocsr()


def normalized_weight(X) -> csr_matrix:
    """
    Normalizes each row of a sparse matrix X to unit length.

    Parameters:
        X: The item-user interaction matrix.

    Returns:
        The row-normalized sparse matrix.
    """
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data**2))[X.row]
    return X.tocsr()


def bm25_weight(X, K1=100, B=0.8) -> csr_matrix:
    """
    Weighs each row of a sparse matrix X  by BM25 weighting.

    Parameters:
        X: The item-user interaction matrix
        K1: BM25 K1 parameter
        B: BM25 B parameter

    Returns:
        The BM25 weighted sparse matrix.
    """
    # calculate idf per term (user)
    X = coo_matrix(X)

    N = float(X.shape[0])
    idf = log(N) - log1p(bincount(X.col))

    # calculate length_norm per document (artist)
    row_sums = np.ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X.tocsr()


def log_weight(X) -> csr_matrix:
    """
    Applies log(1 + x) weighting to the interaction matrix.

    Parameters:
        X: The item-user interaction matrix.

    Returns:
        The log-weighted sparse matrix.
    """
    X = coo_matrix(X)
    X.data = log1p(X.data)
    return X.tocsr()


def log_idf_weight(X, alpha=40.0) -> csr_matrix:
    """
    Applies sophisticated confidence weighting: 1 + alpha * log(1 + r_ui) * idf_i.

    This addresses naive linear confidence by:
    1. Dampening high interaction counts matching user perception (log).
    2. Penalizing very popular items where interactions are less informative (IDF).

    Parameters:
        X: The item-user interaction matrix.
        alpha: The confidence scaling factor.

    Returns:
        The confidence-weighted sparse matrix.
    """
    X = coo_matrix(X)

    N = float(X.shape[0])
    idf = log(N) - log1p(bincount(X.col))

    # Apply sophisticated weighting
    # log1p(X.data) dampens the raw counts
    # idf[X.col] scales by item informativeness
    X.data = 1 + alpha * log1p(X.data) * idf[X.col]
    return X.tocsr()


def power_weight(X, p=0.5) -> csr_matrix:
    """
    Applies power weighting: x^p.

    Parameters:
        X: The item-user interaction matrix.
        p: The power exponent.

    Returns:
        The power-weighted sparse matrix.
    """
    X = coo_matrix(X)
    X.data = np.power(X.data, p)
    return X.tocsr()


def pmi_weight(X) -> csr_matrix:
    """
    Applies Pointwise Mutual Information (PMI) weighting.
    Penalizes both popular items AND promiscuous users.
    
    w_ui = log( (r_ui * N) / (sum_u * sum_i) )
    """
    X = coo_matrix(X)
    N = X.data.sum()
    
    # Calculate row (user) and col (item) sums
    # Note: sum returns a matrix, we need flattened arrays
    col_sums = array(X.sum(axis=0)).flatten()
    row_sums = array(X.sum(axis=1)).flatten()
    
    # Avoid division by zero
    col_sums[col_sums == 0] = 1
    row_sums[row_sums == 0] = 1
    
    # Calculate denominator per interaction: row_sum[u] * col_sum[i]
    denominator = row_sums[X.row] * col_sums[X.col]
    
    # Calculate PMI
    # We add a small epsilon inside log to avoid log(0) issues if counts are weird,
    # though r_ui should be > 0.
    # max(0, ...) ensures we don't have negative weights for "expected" or "below expected" overlap
    pmi = log((X.data * N) / denominator) # we could use np.power(X.data, p) instead of log for a softer effect
    X.data = np.maximum(0, pmi)

    return X.tocsr()


def power_lift_weight(X, p=0.5, batch_size=100_000) -> csr_matrix:
    """
    Applies Power-Lift weighting (Generalized PMI without the log).
    
    Formula:
        w_ui = ( (r_ui * N) / (sum_u * sum_i) ) ^ p

    Optimized: Batched row processing to drastically reduce peak RAM.
    """
    X = X.tocsr()
    N = float(X.data.sum())

    # 1. Calculate marginals (Pre-computed globally)
    # Axis sums differ for CSR, but logic holds.
    col_sums = array(X.sum(axis=0)).flatten()
    row_sums = array(X.sum(axis=1)).flatten()

    col_sums[col_sums == 0] = 1.0
    row_sums[row_sums == 0] = 1.0

    n_users = X.shape[0]

    # 2. Process rows in batches
    for start_row in range(0, n_users, batch_size):
        end_row = min(start_row + batch_size, n_users)
        
        start_ptr = X.indptr[start_row]
        end_ptr = X.indptr[end_row]
        
        if start_ptr == end_ptr:
            continue
            
        data_batch = X.data[start_ptr:end_ptr]
        col_indices_batch = X.indices[start_ptr:end_ptr]
        
        # We need to broadcast row_sums[u] to every interaction in this batch.
        # Since we are iterating rows, we know which row each interaction belongs to.
        # row_counts tells us how many times to repeat each row_sum value.
        batch_indptr = X.indptr[start_row:end_row+1]
        row_counts = np.diff(batch_indptr)
        
        # Extract sums for just these rows
        current_row_sums = row_sums[start_row:end_row]
        
        # Repeat scalar row sum for every interaction in that row
        # e.g. row A has 2 items -> [SumA, SumA]
        expanded_row_sums = np.repeat(current_row_sums, row_counts)
        
        # Lift calculation for batch
        # expected = (row_sum[u] * col_sum[i]) / N
        expected_batch = (expanded_row_sums * col_sums[col_indices_batch]) / N
        
        lift_batch = data_batch / expected_batch
        
        # 3. Write back
        X.data[start_ptr:end_ptr] = np.power(lift_batch, p)

    return X


def robust_user_centric_weight(X, scale_factor=10.0, batch_size=100_000) -> csr_matrix:
    """
    Weights based on user-specific Z-score (using Median/IQR for robustness).
    Adapts '50 plays' to be high for User A but low for User B.
    
    Vectorized implementation with batch processing to prevent Memory Errors.
    """
    X = X.tocsr() # We will modify X.data in-place
    n_users = X.shape[0]

    # Process in chunks to avoid allocating massive sort arrays
    for start_row in range(0, n_users, batch_size):
        end_row = min(start_row + batch_size, n_users)
        
        # 1. Extract batch info
        start_ptr = X.indptr[start_row]
        end_ptr = X.indptr[end_row]
        
        if start_ptr == end_ptr:
            continue
            
        # These are views/copies of the current batch's data
        data_batch = X.data[start_ptr:end_ptr]
        
        # We need indptr relative to the batch start to index into data_batch
        # batch_indptr[0] will be 0
        batch_indptr = X.indptr[start_row:end_row+1] - start_ptr
        row_counts = np.diff(batch_indptr)
        
        n_batch_rows = end_row - start_row

        # 2. Sort data row-wise efficiently (Local to batch)
        # Create row indices for the batch data
        row_indices = np.repeat(np.arange(n_batch_rows), row_counts)
        
        sort_indices = np.lexsort((data_batch, row_indices))
        sorted_batch = data_batch[sort_indices]
        
        # 3. Compute Median and IQR vectorially
        batch_starts = batch_indptr[:-1]
        has_data = row_counts > 0
        valid_rows = np.where(has_data)[0]

        def get_percentile_batch(p):
            values = np.zeros(n_batch_rows)
            if len(valid_rows) == 0:
                return values
                
            indices_float = (row_counts[valid_rows] - 1) * p
            indices_floor = np.floor(indices_float).astype(int)
            indices_ceil = np.ceil(indices_float).astype(int)
            fraction = indices_float - indices_floor
            
            # Index into sorted_batch using flattened row starts
            p_starts = batch_starts[valid_rows]
            val_floor = sorted_batch[p_starts + indices_floor]
            val_ceil = sorted_batch[p_starts + indices_ceil]
            
            values[valid_rows] = val_floor * (1 - fraction) + val_ceil * fraction
            return values

        medians = get_percentile_batch(0.5)
        q75 = get_percentile_batch(0.75)
        q25 = get_percentile_batch(0.25)
        
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        
        # 4. Apply weights to the original (unsorted) data_batch
        user_medians = medians[row_indices]
        user_iqrs = iqr[row_indices]
        
        z_scores = (data_batch - user_medians) / user_iqrs
        weights = 1 / (1 + np.exp(-z_scores))
        
        # 5. Write back to the main matrix
        X.data[start_ptr:end_ptr] = weights * scale_factor

    return X


def robust_user_centric_weight_v2(X, lower_q=25, upper_q=75, batch_size=100_000) -> csr_matrix:
    """
    Weights based on user-specific robust statistics.
    
    Instead of scaling magnitude, this focuses on the shape of the distribution.
    It squashes user interactions into a [0, 1] confidence range based on 
    where the play count falls relative to their personal history.
    """
    X = X.tocsr() # Modify in-place
    n_users = X.shape[0]

    for start_row in range(0, n_users, batch_size):
        end_row = min(start_row + batch_size, n_users)
        
        start_ptr = X.indptr[start_row]
        end_ptr = X.indptr[end_row]
        
        if start_ptr == end_ptr:
            continue
            
        data_batch = X.data[start_ptr:end_ptr]
        batch_indptr = X.indptr[start_row:end_row+1] - start_ptr
        row_counts = np.diff(batch_indptr)
        n_batch_rows = end_row - start_row

        # Sort batch
        row_indices = np.repeat(np.arange(n_batch_rows), row_counts)
        sort_indices = np.lexsort((data_batch, row_indices))
        sorted_batch = data_batch[sort_indices]
        
        batch_starts = batch_indptr[:-1]
        has_data = row_counts > 0
        valid_rows = np.where(has_data)[0]

        def get_percentile_batch(p):
            values = np.zeros(n_batch_rows)
            if len(valid_rows) == 0: return values
            
            indices_float = (row_counts[valid_rows] - 1) * p
            indices_floor = np.floor(indices_float).astype(int)
            indices_ceil = np.ceil(indices_float).astype(int)
            fraction = indices_float - indices_floor
            
            p_starts = batch_starts[valid_rows]
            val_floor = sorted_batch[p_starts + indices_floor]
            val_ceil = sorted_batch[p_starts + indices_ceil]
            
            values[valid_rows] = val_floor * (1 - fraction) + val_ceil * fraction
            return values

        medians = get_percentile_batch(0.5)
        q_high = get_percentile_batch(upper_q / 100.0)
        q_low = get_percentile_batch(lower_q / 100.0)
        
        spread = q_high - q_low
        spread[spread == 0] = 1.0 

        # Apply
        user_medians = medians[row_indices]
        user_spread = spread[row_indices]
        
        z_scores = (data_batch - user_medians) / user_spread
        weights = 1 / (1 + np.exp(-z_scores))

        X.data[start_ptr:end_ptr] = weights

    return X


def sigmoid_propensity_weight(X, p=10.0, beta=1.0, batch_size=100_000) -> csr_matrix:
    """
    Optimized:
    1. Avoids global np.log(X.data) allocation.
    2. Uses 1/(1+C*x^-p) to remove log/exp from loop.
    """
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float32)

    X = X.tocsr()

    N = float(X.shape[0])
    idf = log(N) - np.log1p(bincount(X.indices, minlength=X.shape[1]))

    # Optimized Mean Calculation (Batched)
    sum_log = 0.0
    count = X.nnz
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        sum_log += np.sum(np.log(X.data[start:end]))
    mean_log_val = sum_log / count

    # Pre-calc constant
    C = np.exp(p * mean_log_val)
    neg_p = -p

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        
        data_chunk = X.data[start:end]
        col_chunk = X.indices[start:end]
        
        # Optimized Math
        term = C * np.power(data_chunk, neg_p)
        sigmoid_counts = 1.0 / (1.0 + term)
        
        X.data[start:end] = sigmoid_counts * (1 + beta * idf[col_chunk])

    return X


def sigmoid_propensity_weight_slow(X, p=10.0, beta=1.0) -> csr_matrix:
    """
    Combines Sigmoid saturation (S-Curve) with Inverse Frequency boosting.
    p: Steepness of the sigmoid
    beta: Strength of the IDF boosting
    """
    X = coo_matrix(X)

    # 1. Calculate IDF (Item rarity)
    N = float(X.shape[0])
    idf = log(N) - np.log1p(bincount(X.col))

    # 2. Sigmoid saturation of raw counts
    # Using log(r_ui) inside sigmoid allows handling massive counts better
    sigmoid_counts = 1 / (1 + np.exp(-p * (log(X.data) - np.mean(log(X.data)))))

    # 3. Combine
    X.data = sigmoid_counts * (1 + beta * idf[X.col])

    return X.tocsr()

def power_lift_weight_slow(X, p=0.5) -> csr_matrix:
    """
    Applies Power-Lift weighting (Generalized PMI without the log).
    
    Formula:
        w_ui = ( (r_ui * N) / (sum_u * sum_i) ) ^ p

    Theoretical Advantage:
    Unlike PMI, this does not produce negative weights for 'below expectation' 
    interactions, avoiding the need for zero-clipping (PPMI).
    """
    # Work with COO for fast row/col indexing
    X = coo_matrix(X)
    N = float(X.data.sum())

    # 1. Calculate row (user) and col (item) marginals
    # Note: .sum() returns a matrix, we flatten to 1D array
    col_sums = array(X.sum(axis=0)).flatten()
    row_sums = array(X.sum(axis=1)).flatten()

    # Safety: replace 0 sums with 1 to avoid DivisionByZero (though rare in cleaned data)
    col_sums[col_sums == 0] = 1.0
    row_sums[row_sums == 0] = 1.0

    # 2. Vectorized computation of the Lift Ratio on non-zero entries only
    # Ratio = Observed / Expected = (r_ui * N) / (row_sum[u] * col_sum[i])
    expected = (row_sums[X.row] * col_sums[X.col]) / N
    lift = X.data / expected

    # 3. Apply Power Scaling
    X.data = np.power(lift, p)

    return X.tocsr()

def robust_user_centric_weight_slow(X, scale_factor=10.0) -> csr_matrix:
    """
    Weights based on user-specific Z-score (using Median/IQR for robustness).
    Adapts '50 plays' to be high for User A but low for User B.
    """
    X = X.tocsr()
    new_data = []

    # We iterate by user (rows)
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i+1]
        if start == end:
            continue

        user_counts = X.data[start:end]
        # Robust statistics
        median = np.median(user_counts)
        q75, q25 = np.percentile(user_counts, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1.0 # Avoid div/0 for users with constant play counts
        # Z-score-ish normalization centered on user's own behavior
        # Sigmoid squash to keep it bounded (0 to 1), then scaled
        z_scores = (user_counts - median) / iqr
        weights = 1 / (1 + np.exp(-z_scores)) # Sigmoid
        new_data.extend(weights * scale_factor)
    X.data = np.array(new_data)
    return X


def robust_user_centric_weight_v2_slow(X, lower_q=25, upper_q=75) -> csr_matrix:
    """
    Weights based on user-specific robust statistics.
    
    Instead of scaling magnitude, this focuses on the shape of the distribution.
    It squashes user interactions into a [0, 1] confidence range based on 
    where the play count falls relative to their personal history.

    Parameters:
        X: Sparse interaction matrix
        lower_q: Lower quantile for spread calc (e.g., 25 for IQR)
        upper_q: Upper quantile for spread calc (e.g., 75 for IQR)
    """
    X = X.tocsr()
    new_data = []

    # Iterate over users (rows)
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i+1]
        
        # Skip empty users to keep data alignment
        if start == end:
            continue

        user_counts = X.data[start:end]
        
        # 1. Calculate Robust Statistics
        # We assume the Median is the "neutral" anchor point
        median = np.median(user_counts)
        
        # Calculate dynamic spread
        q_high, q_low = np.percentile(user_counts, [upper_q, lower_q])
        spread = q_high - q_low
        
        # Safety: If a user listens to everything exactly 10 times, spread is 0.
        # We default to 1.0 to prevent division by zero.
        if spread == 0:
            spread = 1.0 

        # 2. Calculate Robust Z-Score
        # "How many 'spreads' away from the median is this interaction?"
        z_scores = (user_counts - median) / spread

        # 3. Sigmoid Squash
        # Maps Z-scores to (0, 1). 
        # Median -> 0.5
        # High outliers -> 1.0
        # Low outliers -> 0.0
        weights = 1 / (1 + np.exp(-z_scores))

        new_data.extend(weights)

    X.data = np.array(new_data)
    return X
