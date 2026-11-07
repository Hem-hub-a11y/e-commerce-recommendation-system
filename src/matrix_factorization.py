"""
Matrix Factorization (Latent Factor Model) Module.

Decomposes the high-dimensional sparse user-item interaction matrix into
low-rank latent factor embeddings using Truncated SVD:

  R ≈ P * Q^T

Where:
  - P in R^(M x d) represents latent user preference embeddings.
  - Q in R^(N x d) represents latent item characteristic embeddings.
  - r_hat(u, i) = p_u . q_i^T predicts user affinity across all items.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.feature_engineering import InteractionMatrixBuilder
from src.evaluation import PopularityRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MatrixFactorizationSVD:
    """
    Matrix Factorization Recommendation Engine via Truncated Singular Value Decomposition.
    """

    def __init__(
        self,
        n_components: int = 50,
        representation: str = "binary",
        random_state: int = 42
    ):
        self.n_components = n_components
        self.representation = representation
        self.random_state = random_state

        self.matrix_builder = InteractionMatrixBuilder()
        self.svd: Optional[TruncatedSVD] = None
        self.user_factors: Optional[np.ndarray] = None   # P matrix: (n_users, n_components)
        self.item_factors: Optional[np.ndarray] = None   # V matrix: (n_components, n_items)
        self.user_history: Dict[str, Set[str]] = {}
        self.popularity_fallback: Optional[PopularityRecommender] = None
        self.explained_variance_ratio_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "MatrixFactorizationSVD":
        """
        Fit TruncatedSVD on the sparse user-item interaction matrix.

        Args:
            train_df: Training transactions DataFrame

        Returns:
            Self
        """
        logger.info(
            f"Fitting MatrixFactorizationSVD (d={self.n_components}, rep='{self.representation}')..."
        )
        interaction_matrix = self.matrix_builder.fit_transform(
            train_df, representation=self.representation
        )

        # TruncatedSVD uses randomized SVD (Halko et al., 2009) optimized for CSR matrices
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            algorithm="randomized",
            random_state=self.random_state
        )

        # User latent factor embeddings: shape (n_users, n_components)
        self.user_factors = self.svd.fit_transform(interaction_matrix)

        # Item latent factor components: shape (n_components, n_items)
        self.item_factors = self.svd.components_

        self.explained_variance_ratio_ = float(self.svd.explained_variance_ratio_.sum())

        # Build user purchase history for exclusion
        user_grouped = train_df.groupby("Customer ID")["StockCode"].unique()
        self.user_history = {
            str(u): set(str(item) for item in items)
            for u, items in user_grouped.items()
        }

        # Popularity fallback for unseen cold-start customers
        self.popularity_fallback = PopularityRecommender().fit(train_df)
        self.is_fitted = True

        logger.info(
            f"SVD fitting complete. Total explained variance ({self.n_components} factors): "
            f"{self.explained_variance_ratio_ * 100:.2f}%"
        )
        return self

    def predict_scores_for_user(self, u_idx: int) -> np.ndarray:
        """
        Reconstruct predicted affinity scores for all items:
          scores = p_u . V = (1, d) . (d, n_items) -> (n_items,)
        """
        user_vec = self.user_factors[u_idx, :]  # shape (d,)
        scores = user_vec.dot(self.item_factors)  # shape (n_items,)
        return scores

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_known: bool = True
    ) -> List[str]:
        """
        Generate top-K product recommendations for a customer.

        Args:
            user_id: Customer identifier string
            k: Number of recommendations
            exclude_known: Exclude items user previously purchased

        Returns:
            List of recommended StockCode strings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommend().")

        user_str = str(user_id)

        # Cold start fallback
        if user_str not in self.matrix_builder.user_to_idx:
            return self.popularity_fallback.recommend(user_str, k=k, exclude_known=exclude_known)

        u_idx = self.matrix_builder.user_to_idx[user_str]
        scores = self.predict_scores_for_user(u_idx).copy()

        # Exclude previously purchased items
        if exclude_known and user_str in self.user_history:
            for item in self.user_history[user_str]:
                if item in self.matrix_builder.item_to_idx:
                    scores[self.matrix_builder.item_to_idx[item]] = -np.inf

        # Filter candidates (exclude masked items with -inf score)
        candidate_indices = np.where(scores > -np.inf)[0]
        if len(candidate_indices) == 0:
            return self.popularity_fallback.recommend(user_str, k=k, exclude_known=exclude_known)

        if len(candidate_indices) <= k:
            sorted_order = candidate_indices[np.argsort(-scores[candidate_indices])]
            return [self.matrix_builder.idx_to_item[i] for i in sorted_order]

        top_k_partition = candidate_indices[np.argpartition(-scores[candidate_indices], k)[:k]]
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

