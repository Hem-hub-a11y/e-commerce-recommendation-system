"""
Collaborative Filtering Recommendation Engines.

Implements high-performance, vectorized:
  1. User-Based Collaborative Filtering (UBCF) with normalized similarity weighting.
  2. Item-Based Collaborative Filtering (IBCF) with item-item similarity.
  3. Integrated cold-start / zero-similarity fallback to Popularity Recommender.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.feature_engineering import (
    InteractionMatrixBuilder,
    compute_sparse_cosine_similarity,
    get_top_k_neighbors
)
from src.evaluation import PopularityRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class UserBasedCF:
    """
    Optimized User-Based Collaborative Filtering (UBCF) Recommender.
    
    Predicts item interest for target user u by aggregating purchase signals from
    the top-K most similar customers, weighted by normalized cosine similarity:
    
      r_hat(u, i) = sum_{v in N_k(u)} [ s(u, v) * r(v, i) ] / ( sum_{v in N_k(u)} s(u, v) + epsilon )
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        representation: str = "log",
        normalize_scores: bool = True,
        min_similarity: float = 0.0
    ):
        self.n_neighbors = n_neighbors
        self.representation = representation
        self.normalize_scores = normalize_scores
        self.min_similarity = min_similarity

        self.matrix_builder = InteractionMatrixBuilder()
        self.interaction_matrix: Optional[csr_matrix] = None
        self.similarity_matrix: Optional[csr_matrix] = None
        self.user_history: Dict[str, Set[str]] = {}
        self.popularity_fallback: Optional[PopularityRecommender] = None
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "UserBasedCF":
        """
        Fit model on training interactions:
          1. Build sparse user-item interaction matrix.
          2. Compute pairwise user cosine similarity.
          3. Fit popularity fallback recommender.
        """
        logger.info(
            f"Fitting UserBasedCF (neighbors={self.n_neighbors}, "
            f"rep='{self.representation}', normalized={self.normalize_scores})..."
        )
        self.interaction_matrix = self.matrix_builder.fit_transform(
            train_df, representation=self.representation
        )

        # Compute sparse pairwise cosine similarity
        self.similarity_matrix = compute_sparse_cosine_similarity(self.interaction_matrix)

        # Build user history
        user_grouped = train_df.groupby("Customer ID")["StockCode"].unique()
        self.user_history = {
            str(u): set(str(item) for item in items)
            for u, items in user_grouped.items()
        }

        # Popularity fallback for cold-start or empty neighborhoods
        self.popularity_fallback = PopularityRecommender().fit(train_df)
        self.is_fitted = True
        logger.info("UserBasedCF fitting complete.")
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_known: bool = True
    ) -> List[str]:
        """
        Generate top-K product recommendations for a single customer.

        Args:
            user_id: Customer identifier string
            k: Number of recommendations
            exclude_known: Exclude items already purchased in training set

        Returns:
            List of recommended StockCode strings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommend().")

        user_str = str(user_id)

        # Fallback if user is unknown (cold-start)
        if user_str not in self.matrix_builder.user_to_idx:
            return self.popularity_fallback.recommend(user_str, k=k, exclude_known=exclude_known)

        u_idx = self.matrix_builder.user_to_idx[user_str]
        neighbor_indices, neighbor_sims = get_top_k_neighbors(
            self.similarity_matrix, row_idx=u_idx, k=self.n_neighbors
        )

        # Filter by minimum similarity threshold
        valid_mask = neighbor_sims > self.min_similarity
        neighbor_indices = neighbor_indices[valid_mask]
        neighbor_sims = neighbor_sims[valid_mask]

        if len(neighbor_indices) == 0:
            return self.popularity_fallback.recommend(user_str, k=k, exclude_known=exclude_known)

        # Extract neighbor submatrix: shape (len(neighbors), n_items)
        neighbor_submat = self.interaction_matrix[neighbor_indices, :]

        # Unnormalized score vector: (n_items, len(neighbors)) * (len(neighbors),) -> (n_items,)
        raw_scores = np.asarray(neighbor_submat.T.dot(neighbor_sims)).ravel()

        if self.normalize_scores:
            # Denominator: sum of similarities of neighbors who interacted with each item
            bin_submat = neighbor_submat.copy()
            bin_submat.data = np.ones_like(bin_submat.data)
            sum_weights = np.asarray(bin_submat.T.dot(neighbor_sims)).ravel()
            scores = raw_scores / (sum_weights + 1e-9)
        else:
            scores = raw_scores

        # Exclude items the user already purchased
        if exclude_known and user_str in self.user_history:
            for item in self.user_history[user_str]:
                if item in self.matrix_builder.item_to_idx:
                    scores[self.matrix_builder.item_to_idx[item]] = -np.inf

        # Find items with positive predicted scores
        positive_indices = np.where(scores > 0.0)[0]
        if len(positive_indices) == 0:
            return self.popularity_fallback.recommend(user_str, k=k, exclude_known=exclude_known)

        if len(positive_indices) <= k:
            sorted_order = positive_indices[np.argsort(-scores[positive_indices])]
            recs = [self.matrix_builder.idx_to_item[i] for i in sorted_order]
            # Fill remaining slots with popularity fallback if needed
            if len(recs) < k:
                fallback_recs = self.popularity_fallback.recommend(
                    user_str, k=k * 2, exclude_known=exclude_known
                )
                for f_item in fallback_recs:
                    if f_item not in recs:
                        recs.append(f_item)
                    if len(recs) >= k:
                        break
            return recs

        # Top-K partition
        top_k_partition = positive_indices[np.argpartition(-scores[positive_indices], k)[:k]]
        sorted_top_k = top_k_partition[np.argsort(-scores[top_k_partition])]
        return [self.matrix_builder.idx_to_item[i] for i in sorted_top_k]

    def recommend_batch(
        self,
        user_ids: List[str],
        k: int = 10,
        exclude_known: bool = True
    ) -> Dict[str, List[str]]:
        """Batch recommendation generation for multiple users."""
        return {
            uid: self.recommend(uid, k=k, exclude_known=exclude_known)
            for uid in user_ids
        }


class ItemBasedCF:
    """
    Optimized Item-Based Collaborative Filtering (IBCF) Recommender.
    
    Predicts item interest for target user u by identifying items structurally
    similar to items user u has previously purchased:
    
      r_hat(u, i) = sum_{j in I_u cap N_k(i)} [ s(i, j) * r(u, j) ] / ( sum_{j in I_u cap N_k(i)} s(i, j) + epsilon )
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        representation: str = "log",
        normalize_scores: bool = True,
        min_similarity: float = 0.0
    ):
        self.n_neighbors = n_neighbors
        self.representation = representation
        self.normalize_scores = normalize_scores
        self.min_similarity = min_similarity

        self.matrix_builder = InteractionMatrixBuilder()
        self.interaction_matrix: Optional[csr_matrix] = None
        self.item_similarity_matrix: Optional[csr_matrix] = None
        self.user_history: Dict[str, Set[str]] = {}
        self.popularity_fallback: Optional[PopularityRecommender] = None
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "ItemBasedCF":
        """
        Fit model on training interactions:
          1. Build sparse user-item interaction matrix R.
          2. Compute item-item cosine similarity S_item = Cosine(R.T).
          3. Fit popularity fallback recommender.
        """
        logger.info(
            f"Fitting ItemBasedCF (neighbors={self.n_neighbors}, "
            f"rep='{self.representation}', normalized={self.normalize_scores})..."
        )
        self.interaction_matrix = self.matrix_builder.fit_transform(
            train_df, representation=self.representation
        )

        # Transpose to get item-by-user matrix: shape (n_items, n_users)
        item_user_matrix = self.interaction_matrix.T.tocsr()

        # Compute item-item cosine similarity: shape (n_items, n_items)
        self.item_similarity_matrix = compute_sparse_cosine_similarity(item_user_matrix)

        # Zero out the diagonal (self-similarity)
        self.item_similarity_matrix.setdiag(0.0)
        self.item_similarity_matrix.eliminate_zeros()

        # Filter top-K neighbors per item to maintain a clean, sparse model
        logger.info(f"Retaining top-{self.n_neighbors} similarities per item...")
        self.item_similarity_matrix = self._prune_top_k(
            self.item_similarity_matrix, k=self.n_neighbors
        )

        # Build user history
        user_grouped = train_df.groupby("Customer ID")["StockCode"].unique()
        self.user_history = {
            str(u): set(str(item) for item in items)
            for u, items in user_grouped.items()
        }

        self.popularity_fallback = PopularityRecommender().fit(train_df)
        self.is_fitted = True
        logger.info("ItemBasedCF fitting complete.")
        return self

    @staticmethod
    def _prune_top_k(matrix: csr_matrix, k: int) -> csr_matrix:
        """Keep only top-k highest similarity values per row in sparse matrix."""
        rows, cols, data = [], [], []
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i)
            if row.nnz == 0:
                continue
            if row.nnz <= k:
                rows.extend([i] * row.nnz)
                cols.extend(row.indices)
                data.extend(row.data)
            else:
                top_k_idx = np.argpartition(-row.data, k)[:k]
                rows.extend([i] * k)
                cols.extend(row.indices[top_k_idx])
                data.extend(row.data[top_k_idx])
        return csr_matrix((data, (rows, cols)), shape=matrix.shape)

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_known: bool = True
    ) -> List[str]:
        """
        Generate top-K product recommendations using item-item similarity.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommend().")

        user_str = str(user_id)

        # Cold start fallback
        if user_str not in self.matrix_builder.user_to_idx:
            return self.popularity_fallback.recommend(user_str, k=k, exclude_known=exclude_known)

        u_idx = self.matrix_builder.user_to_idx[user_str]
        user_vec = self.interaction_matrix.getrow(u_idx)

        if user_vec.nnz == 0:
            return self.popularity_fallback.recommend(user_str, k=k, exclude_known=exclude_known)

        # Score vector: user_vec (1, n_items) * item_similarity_matrix.T (n_items, n_items) -> (1, n_items)
        raw_scores = np.asarray(user_vec.dot(self.item_similarity_matrix.T).toarray()).ravel()

        if self.normalize_scores:
            bin_vec = user_vec.copy()
            bin_vec.data = np.ones_like(bin_vec.data)
            sum_weights = np.asarray(bin_vec.dot(self.item_similarity_matrix.T).toarray()).ravel()
            scores = raw_scores / (sum_weights + 1e-9)
        else:
            scores = raw_scores

        # Exclude previously purchased items
        if exclude_known and user_str in self.user_history:
            for item in self.user_history[user_str]:
                if item in self.matrix_builder.item_to_idx:
                    scores[self.matrix_builder.item_to_idx[item]] = -np.inf

        positive_indices = np.where(scores > 0.0)[0]
        if len(positive_indices) == 0:
            return self.popularity_fallback.recommend(user_str, k=k, exclude_known=exclude_known)

        if len(positive_indices) <= k:
            sorted_order = positive_indices[np.argsort(-scores[positive_indices])]
            recs = [self.matrix_builder.idx_to_item[i] for i in sorted_order]
            if len(recs) < k:
                fallback_recs = self.popularity_fallback.recommend(
                    user_str, k=k * 2, exclude_known=exclude_known
                )
                for f_item in fallback_recs:
                    if f_item not in recs:
                        recs.append(f_item)
                    if len(recs) >= k:
                        break
            return recs

        top_k_partition = positive_indices[np.argpartition(-scores[positive_indices], k)[:k]]
        sorted_top_k = top_k_partition[np.argsort(-scores[top_k_partition])]
        return [self.matrix_builder.idx_to_item[i] for i in sorted_top_k]

    def recommend_batch(
        self,
        user_ids: List[str],
        k: int = 10,
        exclude_known: bool = True
    ) -> Dict[str, List[str]]:
        """Batch recommendation generation for multiple users."""
        return {
            uid: self.recommend(uid, k=k, exclude_known=exclude_known)
            for uid in user_ids
        }
