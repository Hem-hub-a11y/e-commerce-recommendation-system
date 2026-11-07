"""
Unit tests for data preprocessing and hygiene pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from src.preprocessing import (
    is_administrative_code,
    clean_transactions,
    build_product_catalog,
)


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing data hygiene rules."""

    def test_administrative_codes_detected(self):
        """Ensure administrative non-stock codes are properly caught."""
        admin_samples = ["POST", "D", "M", "DOT", "C2", "BANK CHARGES", "TEST001", "AMAZONFEE", "ADJUST"]
        for code in admin_samples:
            self.assertTrue(is_administrative_code(code), f"Expected {code} to be administrative.")
            self.assertTrue(is_administrative_code(code.lower()), f"Expected {code.lower()} to be administrative.")

        valid_samples = ["85048", "79323P", "22041", "21232", "DCGS0058"]
        for code in valid_samples:
            self.assertFalse(is_administrative_code(code), f"Expected {code} to be a valid product.")

    def test_clean_transactions_synthetic(self):
        """Test cancellation reconciliation, duplicate dropping, and missing customer handling."""
        synthetic_data = pd.DataFrame({
            "Invoice": ["489434", "489434", "489435", "C489436", "489437", "489438"],
            "StockCode": ["85048", "85048", "22041", "22041", "POST", "21232"],
            "Description": [
                "15CM CHRISTMAS GLASS BALL",
                "15CM CHRISTMAS GLASS BALL",  # Exact duplicate row
                "RECORD FRAME 7\"",
                "RECORD FRAME 7\"",            # Cancellation of 22041
                "POSTAGE",                     # Admin code
                "STRAWBERRY CERAMIC TRINKET",  # Missing customer ID
            ],
            "Quantity": [12, 12, 5, -5, 1, 10],
            "InvoiceDate": pd.to_datetime([
                "2009-12-01 07:45:00",
                "2009-12-01 07:45:00",
                "2009-12-01 08:00:00",
                "2009-12-01 08:30:00",
                "2009-12-01 09:00:00",
                "2009-12-01 09:15:00",
            ]),
            "Price": [6.95, 6.95, 2.10, 2.10, 18.00, 1.25],
            "Customer ID": [13085.0, 13085.0, 13085.0, 13085.0, 13085.0, np.nan],
            "Country": ["United Kingdom"] * 6
        })

        clean_df, stats = clean_transactions(synthetic_data, drop_unregistered_customers=True)

        # Missing customer dropped
        self.assertEqual(stats["missing_customer_ids"], 1)
        # Duplicate dropped
        self.assertEqual(stats["duplicate_records_dropped"], 1)
        # Administrative POST dropped
        self.assertEqual(stats["administrative_records_dropped"], 1)
        # Net quantity for 22041: 5 + (-5) = 0, so should be filtered out
        self.assertNotIn("22041", clean_df["StockCode"].values)
        # 85048 net quantity: 12 > 0, so should be retained
        self.assertIn("85048", clean_df["StockCode"].values)
        self.assertEqual(len(clean_df), 1)

    def test_build_product_catalog(self):
        """Ensure catalog standardizes descriptions to the most frequent clean version."""
        data = pd.DataFrame({
            "StockCode": ["85048", "85048", "85048"],
            "Description": ["Glass Ball", "GLASS BALL 20 LIGHTS", "GLASS BALL 20 LIGHTS"],
            "Price": [6.95, 6.95, 7.50],
            "Quantity": [10, 12, 24]
        })
        catalog = build_product_catalog(data)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog.iloc[0]["Description"], "GLASS BALL 20 LIGHTS")
        self.assertEqual(catalog.iloc[0]["MedianPrice"], 6.95)


if __name__ == "__main__":
    unittest.main()

