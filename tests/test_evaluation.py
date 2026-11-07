"""
Unit tests for Model Evaluation and Ranking Metrics Module.
"""

import unittest
import pandas as pd
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    hit_rate_at_k,
    evaluate_recommendations,
    split_data_chronological,
    PopularityRecommender
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for evaluation metric calculations."""

    def test_precision_at_k(self):
        """Test precision calculation with exact hand-calculated cases."""
        rec = ["A", "B", "C", "D", "E"]
        actual = {"A", "C", "X"}

        # Top 3: ["A", "B", "C"] -> hits: A, C -> 2 / 3
        self.assertAlmostEqual(precision_at_k(rec, actual, k=3), 2.0 / 3.0)
        # Top 5: ["A", "B", "C", "D", "E"] -> hits: A, C -> 2 / 5
        self.assertAlmostEqual(precision_at_k(rec, actual, k=5), 2.0 / 5.0)
        # Zero hits
        self.assertEqual(precision_at_k(rec, {"Z"}, k=5), 0.0)

    def test_recall_at_k(self):
        """Test recall calculation with exact hand-calculated cases."""
        rec = ["A", "B", "C", "D", "E"]
        actual = {"A", "C", "X", "Y"}  # 4 items total

        # Top 3: hits: A, C -> 2 / 4 = 0.5
        self.assertAlmostEqual(recall_at_k(rec, actual, k=3), 2.0 / 4.0)
        # Top 5: hits: A, C -> 2 / 4 = 0.5
        self.assertAlmostEqual(recall_at_k(rec, actual, k=5), 2.0 / 4.0)
        # Empty actual
        self.assertEqual(recall_at_k(rec, set(), k=5), 0.0)

    def test_average_precision_at_k(self):
        """Test AP@K rewarding higher rank positions."""
        actual = {"A", "B"}

        # Perfect ranking: hits at rank 1 and 2
        rec_perfect = ["A", "B", "C", "D"]
        # i=1: hit, P@1 = 1/1 = 1.0
        # i=2: hit, P@2 = 2/2 = 1.0
        # normalizer = min(4, 2) = 2 -> AP = (1.0 + 1.0) / 2 = 1.0
        self.assertAlmostEqual(average_precision_at_k(rec_perfect, actual, k=4), 1.0)

        # Worse ranking: hits at rank 3 and 4
        rec_worse = ["C", "D", "A", "B"]
        # i=3: hit, P@3 = 1/3
        # i=4: hit, P@4 = 2/4 = 0.5
        # AP = (1/3 + 0.5) / 2 = 0.41667
        self.assertAlmostEqual(average_precision_at_k(rec_worse, actual, k=4), (1.0 / 3.0 + 0.5) / 2.0)

        # Zero hits
        self.assertEqual(average_precision_at_k(["X", "Y"], actual, k=2), 0.0)

    def test_hit_rate_at_k(self):
        """Test hit rate indicator function."""
        rec = ["A", "B", "C"]
        self.assertEqual(hit_rate_at_k(rec, {"B"}, k=3), 1.0)
        self.assertEqual(hit_rate_at_k(rec, {"Z"}, k=3), 0.0)
        self.assertEqual(hit_rate_at_k(rec, {"C"}, k=2), 0.0)  # C is at rank 3, cutoff k=2

    def test_popularity_recommender(self):
        """Test Popularity baseline orders by count and excludes known training items."""
        train_df = pd.DataFrame({
            "Customer ID": ["1", "1", "2", "3", "3"],
            "StockCode":   ["P1", "P2", "P1", "P1", "P3"],
            "Quantity":    [1,    1,    1,    1,    1]
        })
        # Frequencies: P1: 3, P2: 1, P3: 1
        pop = PopularityRecommender().fit(train_df)
        self.assertEqual(pop.popular_items[0], "P1")

        # For user 1 who already bought P1 and P2:
        recs = pop.recommend("1", k=2, exclude_known=True)
        self.assertNotIn("P1", recs)
        self.assertNotIn("P2", recs)
        self.assertIn("P3", recs)

    def test_chronological_split(self):
        """Test temporal split guarantees no future data in train."""
        df = pd.DataFrame({
            "Customer ID": ["1", "1", "2"],
            "StockCode": ["P1", "P2", "P1"],
            "InvoiceDate": pd.to_datetime(["2010-01-01", "2010-03-01", "2010-02-01"]),
            "Quantity": [1, 1, 1]
        })
        train, test, meta = split_data_chronological(df, split_date="2010-02-15")
        self.assertEqual(len(train), 2)  # Jan and Feb rows
        self.assertEqual(len(test), 1)   # Mar row
        self.assertTrue(train["InvoiceDate"].max() < pd.to_datetime("2010-02-15"))
        self.assertTrue(test["InvoiceDate"].min() >= pd.to_datetime("2010-02-15"))


if __name__ == "__main__":
    unittest.main()

