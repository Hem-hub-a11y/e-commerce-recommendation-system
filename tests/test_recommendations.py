"""
End-to-End Unit Tests for Unified Recommendation Engine.
"""

import unittest
import pandas as pd
from src.recommendations import RecommendationEngine


class TestRecommendationEngine(unittest.TestCase):
    """End-to-end test suite for RecommendationEngine facade."""

    def setUp(self):
        """Create sample transaction and catalog data."""
        self.transactions = pd.DataFrame({
            "Customer ID": ["101", "101", "102", "102", "103", "104", "104", "105"],
            "StockCode":   ["P1",  "P2",  "P1",  "P3",  "P2",  "P1",  "P4",  "P5"],
            "Quantity":    [5,     2,     4,     1,     10,    20,    2,     1]
        })

    def test_engine_svd_recommendations(self):
        """Test SVD recommendation generation and output structure."""
        engine = RecommendationEngine(model_type="svd", n_components=2)
        engine.fit(self.transactions)

        recs = engine.recommend("101", n_recommendations=2, exclude_known=True)

        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 2)
        self.assertListEqual(list(recs.columns), ["Rank", "StockCode", "Description", "EstimatedPrice"])
        # Customer 101 owns P1 and P2
        self.assertNotIn("P1", recs["StockCode"].values)
        self.assertNotIn("P2", recs["StockCode"].values)
        # Ranks must be [1, 2]
        self.assertListEqual(list(recs["Rank"]), [1, 2])

    def test_engine_cold_start_fallback(self):
        """Test unknown customer receives popularity recommendations without error."""
        engine = RecommendationEngine(model_type="svd", n_components=2)
        engine.fit(self.transactions)

        recs = engine.recommend("UNKNOWN_CUSTOMER", n_recommendations=2)
        self.assertEqual(len(recs), 2)
        # Most popular is P1
        self.assertIn("P1", recs["StockCode"].values)

    def test_no_duplicate_recommendations(self):
        """Ensure no duplicate items exist in top-K recommendations."""
        engine = RecommendationEngine(model_type="popularity")
        engine.fit(self.transactions)

        recs = engine.recommend("101", n_recommendations=2)
        self.assertEqual(len(recs["StockCode"]), len(set(recs["StockCode"])))


if __name__ == "__main__":
    unittest.main()
