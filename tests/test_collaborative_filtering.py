"""
Unit tests for Collaborative Filtering Recommenders (UBCF and IBCF).
"""

import unittest
import pandas as pd
from src.collaborative_filtering import UserBasedCF, ItemBasedCF


class TestCollaborativeFiltering(unittest.TestCase):
    """Test suite for UBCF and IBCF recommendation models."""

    def setUp(self):
        """Create synthetic multi-user, multi-product transaction dataset."""
        self.train_df = pd.DataFrame({
            "Customer ID": ["101", "101", "102", "102", "103", "103", "103", "104"],
            "StockCode":   ["P1",  "P2",  "P1",  "P3",  "P2",  "P3",  "P4",  "P1"],
            "Quantity":    [5,     2,     4,     1,     10,    2,     1,     20]
        })

    def test_ubcf_recommendations(self):
        """Test UserBasedCF recommends expected items and excludes previously owned items."""
        model = UserBasedCF(n_neighbors=2, representation="log", normalize_scores=True)
        model.fit(self.train_df)

        # For user 102 who bought P1 and P3:
        # User 101 bought P1 and P2 -> similarity between 102 and 101 on P1 is high
        # So P2 should be recommended to 102!
        recs = model.recommend("102", k=2, exclude_known=True)

        self.assertTrue(len(recs) <= 2)
        # 102 already owns P1 and P3, so neither should be recommended
        self.assertNotIn("P1", recs)
        self.assertNotIn("P3", recs)
        # P2 should be in recommendations
        self.assertIn("P2", recs)

    def test_ubcf_cold_start_fallback(self):
        """Ensure unknown customers receive popularity fallback recommendations gracefully."""
        model = UserBasedCF(n_neighbors=2, representation="log")
        model.fit(self.train_df)

        recs = model.recommend("UNKNOWN_999", k=2, exclude_known=True)
        self.assertEqual(len(recs), 2)
        # Most popular item is P1
        self.assertIn("P1", recs)

    def test_ibcf_recommendations(self):
        """Test ItemBasedCF recommends items structurally similar to user purchase history."""
        model = ItemBasedCF(n_neighbors=2, representation="log", normalize_scores=True)
        model.fit(self.train_df)

        recs = model.recommend("102", k=2, exclude_known=True)
        self.assertTrue(len(recs) <= 2)
        self.assertNotIn("P1", recs)
        self.assertNotIn("P3", recs)

    def test_ibcf_cold_start_fallback(self):
        """Ensure IBCF gracefully handles cold-start users."""
        model = ItemBasedCF(n_neighbors=2, representation="log")
        model.fit(self.train_df)

        recs = model.recommend("UNKNOWN_999", k=2, exclude_known=True)
        self.assertEqual(len(recs), 2)
        self.assertIn("P1", recs)

    def test_batch_recommendations(self):
        """Ensure batch generation produces recommendations for all requested users."""
        model = UserBasedCF(n_neighbors=2)
        model.fit(self.train_df)

        batch_recs = model.recommend_batch(["101", "102"], k=2)
        self.assertIn("101", batch_recs)
        self.assertIn("102", batch_recs)
        self.assertEqual(len(batch_recs["101"]), 2)


if __name__ == "__main__":
    unittest.main()

