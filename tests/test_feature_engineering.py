"""
Unit tests for Feature Engineering & Sparse Interaction Matrix Module.
"""

import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.feature_engineering import (
    InteractionMatrixBuilder,
    compute_sparse_cosine_similarity,
    get_top_k_neighbors
)


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for sparse interaction matrix builder and feature transformations."""

    def setUp(self):
        """Create synthetic multi-transaction history."""
        self.sample_df = pd.DataFrame({
            "Customer ID": ["101", "101", "102", "102", "103", "103", "103"],
            "StockCode":   ["P1",  "P2",  "P1",  "P3",  "P2",  "P3",  "P4"],
            "Quantity":    [10,    2,     5,     1,     20,    4,     1]
        })
        self.builder = InteractionMatrixBuilder()
        self.builder.fit_mappings(self.sample_df)

    def test_matrix_dimensions_and_mappings(self):
        """Ensure correct matrix shape and bidirectional mapping integrity."""
        self.assertEqual(self.builder.n_users, 3)
        self.assertEqual(self.builder.n_items, 4)

        # Mapping reversibility
        for user, idx in self.builder.user_to_idx.items():
            self.assertEqual(self.builder.idx_to_user[idx], user)
        for item, idx in self.builder.item_to_idx.items():
            self.assertEqual(self.builder.idx_to_item[idx], item)

    def test_binary_representation(self):
        """Verify binary transformation sets all purchases to exactly 1.0."""
        mat = self.builder.transform(self.sample_df, representation="binary")
        self.assertIsInstance(mat, csr_matrix)
        self.assertEqual(mat.shape, (3, 4))
        self.assertEqual(mat.nnz, 7)
        np.testing.assert_array_equal(mat.data, np.ones(7))

    def test_log_representation(self):
        """Verify log transformation applies log(1 + q)."""
        mat = self.builder.transform(self.sample_df, representation="log")
        u101_idx = self.builder.user_to_idx["101"]
        p1_idx = self.builder.item_to_idx["P1"]
        expected_val = np.log1p(10.0)
        self.assertAlmostEqual(mat[u101_idx, p1_idx], expected_val, places=4)

    def test_iuf_representation(self):
        """Verify TF-IUF weighting attenuates items purchased by all users."""
        mat = self.builder.transform(self.sample_df, representation="iuf")
        self.assertEqual(mat.shape, (3, 4))
        self.assertTrue(np.all(mat.data > 0))

    def test_sparse_cosine_similarity_and_top_k(self):
        """Verify cosine similarity is normalized and self-similarity is excluded in top-k."""
        mat = self.builder.transform(self.sample_df, representation="binary")
        sim = compute_sparse_cosine_similarity(mat)

        # Diagonal values should be 1.0
        for i in range(3):
            self.assertAlmostEqual(sim[i, i], 1.0, places=5)

        # Check top-k neighbors for user 0
        neighbors, scores = get_top_k_neighbors(sim, row_idx=0, k=2)
        # Should not include user 0 (self)
        self.assertNotIn(0, neighbors)
        self.assertTrue(len(neighbors) <= 2)
        # Scores should be in descending order
        if len(scores) > 1:
            self.assertTrue(scores[0] >= scores[1])

    def test_sparsity_and_memory(self):
        """Verify sparsity calculation and memory statistics."""
        mat = self.builder.transform(self.sample_df, representation="binary")
        # 7 non-zeros out of 12 cells -> sparsity = (1 - 7/12) * 100% = 41.67%
        sparsity = InteractionMatrixBuilder.calculate_sparsity(mat)
        self.assertAlmostEqual(sparsity, (1 - 7 / 12) * 100, places=2)

        mem = InteractionMatrixBuilder.get_memory_stats(mat)
        self.assertIn("sparse_mb", mem)
        self.assertIn("dense_mb", mem)
        self.assertIn("compression_ratio", mem)


if __name__ == "__main__":
    unittest.main()

