"""
Feature Engineering and Sparse Interaction Matrix Module.

Constructs memory-efficient SciPy CSR interaction matrices from transaction logs
and supports multiple implicit feedback representations:
  - 'binary': Indicator of purchase (1 or 0)
  - 'raw': Raw aggregate purchase quantity
  - 'log': Sublinear log-scaled quantity: log(1 + q)
  - 'iuf' / 'tfidf': Log-scaled quantity weighted by Inverse User Frequency
"""

import sys
import logging
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class InteractionMatrixBuilder:
    """
    Builds and manages sparse user-item interaction matrices with bidirectional index mappings.
    """

    def __init__(
        self,
        user_col: str = "Customer ID",
        item_col: str = "StockCode",
        value_col: str = "Quantity"
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.value_col = value_col

        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}

        self.n_users: int = 0
        self.n_items: int = 0
        self.is_fitted: bool = False

    def fit_mappings(self, df: pd.DataFrame) -> "InteractionMatrixBuilder":
        """
        Create stable bidirectional index mappings for unique users and items.

        Args:
            df: DataFrame containing user_col and item_col

        Returns:
            Self
        """
        unique_users = sorted(df[self.user_col].astype(str).unique())
        unique_items = sorted(df[self.item_col].astype(str).unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}

        self.item_to_idx = {item: i for i, item in enumerate(unique_items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        self.is_fitted = True

        logger.info(
            f"Fitted index mappings: {self.n_users:,} unique users, "
            f"{self.n_items:,} unique items."
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        representation: str = "log"
    ) -> csr_matrix:
        """
        Transform transaction records into a sparse CSR matrix.

        Args:
            df: DataFrame with transactions
            representation: One of 'binary', 'raw', 'log', 'iuf'

        Returns:
            scipy.sparse.csr_matrix of shape (n_users, n_items)
        """
        if not self.is_fitted:
            raise ValueError("InteractionMatrixBuilder must be fitted before transform().")

        # Aggregate quantities per user-item pair
        grouped = (
            df.groupby([self.user_col, self.item_col])[self.value_col]
            .sum()
            .reset_index()
        )

        # Filter out unknown users/items if transforming an unseen split
        user_series = grouped[self.user_col].astype(str)
        item_series = grouped[self.item_col].astype(str)

        valid_mask = user_series.isin(self.user_to_idx) & item_series.isin(self.item_to_idx)
        valid_df = grouped[valid_mask].copy()

        rows = valid_df[self.user_col].astype(str).map(self.user_to_idx).values
        cols = valid_df[self.item_col].astype(str).map(self.item_to_idx).values
        raw_vals = valid_df[self.value_col].values.astype(np.float64)

        # Apply interaction signal transformation
        rep = representation.lower()
        if rep == "binary":
            data = np.ones_like(raw_vals, dtype=np.float64)
        elif rep == "raw":
            data = raw_vals
        elif rep == "log":
            data = np.log1p(np.maximum(0.0, raw_vals))
        elif rep in ("iuf", "tfidf"):
            # Sublinear TF * Inverse User Frequency (IUF)
            tf = np.log1p(np.maximum(0.0, raw_vals))
            # Calculate item user frequencies
            item_freq = np.bincount(cols, minlength=self.n_items).astype(np.float64)
            # IUF: log((N + 1) / (freq + 1)) + 1
            iuf = np.log((self.n_users + 1.0) / (item_freq + 1.0)) + 1.0
            data = tf * iuf[cols]
        else:
            raise ValueError(
                f"Unknown representation '{representation}'. "
                "Choose from 'binary', 'raw', 'log', 'iuf'."
            )

        matrix = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))
        logger.info(
            f"Built CSR matrix [{representation}]: shape={matrix.shape}, "
            f"non-zero entries={matrix.nnz:,}, "
            f"sparsity={self.calculate_sparsity(matrix):.2f}%"
        )
        return matrix

    def fit_transform(
        self,
        df: pd.DataFrame,
        representation: str = "log"
    ) -> csr_matrix:
        """Fit index mappings and return transformed CSR matrix."""
        return self.fit_mappings(df).transform(df, representation=representation)

    @staticmethod
    def calculate_sparsity(matrix: csr_matrix) -> float:
        """Compute percentage of empty (zero) cells in matrix."""
        total_cells = matrix.shape[0] * matrix.shape[1]
        if total_cells == 0:
            return 0.0
        return (1.0 - (matrix.nnz / total_cells)) * 100.0

    @staticmethod
    def get_memory_stats(matrix: csr_matrix) -> Dict[str, float]:
        """
        Compare sparse CSR memory usage against equivalent dense matrix.

        Returns:
            Dictionary with sparse_mb, dense_mb, and compression_ratio.
        """
        sparse_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        dense_bytes = matrix.shape[0] * matrix.shape[1] * np.dtype(np.float64).itemsize

        sparse_mb = sparse_bytes / (1024.0 * 1024.0)
        dense_mb = dense_bytes / (1024.0 * 1024.0)
        ratio = dense_bytes / max(1, sparse_bytes)

        return {
            "sparse_mb": round(sparse_mb, 2),
            "dense_mb": round(dense_mb, 2),
            "compression_ratio": round(ratio, 1)
        }


def compute_sparse_cosine_similarity(matrix: csr_matrix) -> csr_matrix:
    """
    Compute pairwise cosine similarity between row vectors of a sparse matrix.
    
    Using L2 row normalization:
      norm_matrix = matrix / ||row||_2
      similarity = norm_matrix * norm_matrix.T
    
    Returns:
        Sparse CSR similarity matrix of shape (n_rows, n_rows).
    """
    norm_matrix = normalize(matrix, norm="l2", axis=1, copy=True)
    # Cosine similarity is the dot product of L2-normalized vectors
    similarity = norm_matrix.dot(norm_matrix.T)
    return similarity.tocsr()


def get_top_k_neighbors(
    similarity_matrix: csr_matrix,
    row_idx: int,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficiently extract top-K most similar neighbors for a given row index,
    excluding self-similarity at diagonal.

    Args:
        similarity_matrix: Pairwise CSR similarity matrix
        row_idx: Target row index
        k: Number of neighbors to return

    Returns:
        Tuple of (neighbor_indices, neighbor_similarity_scores)
    """
    row = similarity_matrix.getrow(row_idx)
    indices = row.indices
    data = row.data

    # Exclude self
    mask = indices != row_idx
    indices = indices[mask]
    data = data[mask]

    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if len(indices) <= k:
        sorted_order = np.argsort(-data)
        return indices[sorted_order], data[sorted_order]

    # Partial sort via argpartition: O(N) rather than full O(N log N) sort
    top_k_partition = np.argpartition(-data, k)[:k]
    sorted_top_k = top_k_partition[np.argsort(-data[top_k_partition])]

    return indices[sorted_top_k], data[sorted_top_k]


if __name__ == "__main__":
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    clean_csv_path = os.path.join(project_root, "data", "online_retail_clean.csv.gz")

    if not os.path.exists(clean_csv_path):
        print("Please run src/preprocessing.py first!")
        sys.exit(1)

    print(f"Loading cleaned transactions from: {clean_csv_path}")
    df_clean = pd.read_csv(clean_csv_path)

    builder = InteractionMatrixBuilder()
    builder.fit_mappings(df_clean)

    # Compare representations
    representations = ["raw", "binary", "log", "iuf"]
    comparison = []

    for rep in representations:
        mat = builder.transform(df_clean, representation=rep)
        mem = InteractionMatrixBuilder.get_memory_stats(mat)
        sparsity = InteractionMatrixBuilder.calculate_sparsity(mat)
        min_v = mat.data.min() if mat.nnz > 0 else 0
        max_v = mat.data.max() if mat.nnz > 0 else 0
        mean_v = mat.data.mean() if mat.nnz > 0 else 0

        comparison.append({
            "Representation": rep,
            "Sparsity (%)": round(sparsity, 2),
            "Non-Zero Count": mat.nnz,
            "Min Value": round(float(min_v), 2),
            "Mean Value": round(float(mean_v), 2),
            "Max Value": round(float(max_v), 2),
            "Sparse RAM (MB)": mem["sparse_mb"],
            "Dense RAM (MB)": mem["dense_mb"],
            "RAM Savings": f"{mem['compression_ratio']}x"
        })

    comp_df = pd.DataFrame(comparison)
    print("\n" + "=" * 90)
    print("INTERACTION MATRIX REPRESENTATION BENCHMARK")
    print("=" * 90)
    print(comp_df.to_string(index=False))
    print("=" * 90)

