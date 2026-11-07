"""
Unit tests for Matrix Factorization (TruncatedSVD) Recommender.
"""

import unittest
import numpy as np
import pandas as pd
from src.matrix_factorization import MatrixFactorizationSVD


class TestMatrixFactorization(unittest.TestCase):
    """Test suite for SVD Latent Factor Model."""

    def setUp(self):
        """Create synthetic multi-user multi-item dataset."""
        self.train_df = pd.DataFrame({
            "Customer ID": ["101", "101", "102", "102", "103", "103", "104", "104", "105"],
            "StockCode":   ["P1",  "P2",  "P1",  "P3",  "P2",  "P3",  "P3",  "P4",  "P1"],
            "Quantity":    [5,     2,     4,     1,     10,    2,     1,     20,    3]
        })

    def test_svd_factor_dimensions(self):
        """Verify SVD generates low-rank user and item embeddings with requested dimension."""
        d = 2
        model = MatrixFactorizationSVD(n_components=d, representation="binary", random_state=42)
        model.fit(self.train_df)

        n_users = model.matrix_builder.n_users
        n_items = model.matrix_builder.n_items

        self.assertEqual(model.user_factors.shape, (n_users, d))
        self.assertEqual(model.item_factors.shape, (d, n_items))
        self.assertTrue(model.explained_variance_ratio_ > 0.0)

    def test_svd_recommendations_and_history_masking(self):
        """Verify recommendations exclude previously purchased items."""
        model = MatrixFactorizationSVD(n_components=2, representation="binary", random_state=42)
        model.fit(self.train_df)

        # User 101 bought P1 and P2
        recs = model.recommend("101", k=2, exclude_known=True)
        self.assertEqual(len(recs), 2)
        self.assertNotIn("P1", recs)
        self.assertNotIn("P2", recs)

    def test_svd_cold_start_fallback(self):
        """Verify unseen customer gracefully receives popularity recommendations."""
        model = MatrixFactorizationSVD(n_components=2, representation="binary", random_state=42)
        model.fit(self.train_df)

        recs = model.recommend("UNKNOWN_999", k=2, exclude_known=True)
        self.assertEqual(len(recs), 2)
        self.assertIn("P1", recs)


if __name__ == "__main__":
    unittest.main()

