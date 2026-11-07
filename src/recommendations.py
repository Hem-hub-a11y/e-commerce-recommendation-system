"""
Unified E-Commerce Recommendation Interface.

Provides a single production-grade entry point to train, persist, and query
any recommendation engine (Popularity, UBCF, IBCF, SVD, or Hybrid) with
human-readable product catalog enrichment (Description, Price, Category).
"""

import os
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from src.evaluation import PopularityRecommender
from src.collaborative_filtering import UserBasedCF, ItemBasedCF
from src.matrix_factorization import MatrixFactorizationSVD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    High-level facade for e-commerce recommendation generation.
    """

    SUPPORTED_MODELS = ["svd", "ibcf", "ubcf", "popularity"]

    def __init__(
        self,
        model_type: str = "svd",
        catalog_path: Optional[str] = None,
        **model_kwargs
    ):
        self.model_type = model_type.lower()
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose from {self.SUPPORTED_MODELS}."
            )

        self.catalog_path = catalog_path
        self.catalog_df: Optional[pd.DataFrame] = None
        self.catalog_dict: Dict[str, Dict[str, any]] = {}

        if self.catalog_path and os.path.exists(self.catalog_path):
            self.load_catalog(self.catalog_path)

        # Initialize underlying model
        if self.model_type == "svd":
            self.model = MatrixFactorizationSVD(
                n_components=model_kwargs.get("n_components", 50),
                representation=model_kwargs.get("representation", "binary"),
                random_state=model_kwargs.get("random_state", 42)
            )
        elif self.model_type == "ibcf":
            self.model = ItemBasedCF(
                n_neighbors=model_kwargs.get("n_neighbors", 20),
                representation=model_kwargs.get("representation", "binary"),
                normalize_scores=model_kwargs.get("normalize_scores", False)
            )
        elif self.model_type == "ubcf":
            self.model = UserBasedCF(
                n_neighbors=model_kwargs.get("n_neighbors", 20),
                representation=model_kwargs.get("representation", "binary"),
                normalize_scores=model_kwargs.get("normalize_scores", False)
            )
        elif self.model_type == "popularity":
            self.model = PopularityRecommender()

    def load_catalog(self, catalog_path: str):
        """Load product catalog mapping StockCode to Description and Price."""
        self.catalog_df = pd.read_csv(catalog_path)
        self.catalog_dict = (
            self.catalog_df.set_index("StockCode")[["Description", "MedianPrice"]]
            .to_dict(orient="index")
        )
        logger.info(f"Loaded product catalog with {len(self.catalog_dict):,} items.")

    def fit(self, train_df: pd.DataFrame) -> "RecommendationEngine":
        """Train the recommendation model on historical transaction records."""
        logger.info(f"Training RecommendationEngine [{self.model_type.upper()}]...")
        self.model.fit(train_df)
        return self

    def recommend(
        self,
        customer_id: str,
        n_recommendations: int = 5,
        exclude_known: bool = True
    ) -> pd.DataFrame:
        """
        Generate ranked recommendations enriched with product metadata.

        Args:
            customer_id: Target Customer ID string
            n_recommendations: Number of products to recommend
            exclude_known: Exclude items customer already bought

        Returns:
            DataFrame with columns: [Rank, StockCode, Description, UnitPrice]
        """
        item_ids = self.model.recommend(
            str(customer_id), k=n_recommendations, exclude_known=exclude_known
        )

        rows = []
        for rank, item in enumerate(item_ids, 1):
            meta = self.catalog_dict.get(str(item), {})
            desc = meta.get("Description", "Description Unavailable")
            price = meta.get("MedianPrice", np.nan)
            rows.append({
                "Rank": rank,
                "StockCode": item,
                "Description": desc,
                "EstimatedPrice": f"£{price:.2f}" if pd.notnull(price) else "N/A"
            })

        return pd.DataFrame(rows)


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    clean_csv_path = os.path.join(project_root, "data", "online_retail_clean.csv.gz")
    catalog_path = os.path.join(project_root, "data", "product_catalog.csv")

    df = pd.read_csv(clean_csv_path)

    # Train production SVD engine
    engine = RecommendationEngine(model_type="svd", catalog_path=catalog_path, n_components=50)
    engine.fit(df)

    test_customers = ["12346", "12347", "12348", "99999"]

    print("\n" + "=" * 80)
    print("PRODUCTION RECOMMENDATION ENGINE DEMONSTRATION")
    print("=" * 80)

    for cust in test_customers:
        status = "(Known Customer)" if cust != "99999" else "(Cold-Start Customer)"
        print(f"\nTop 5 Recommendations for Customer {cust} {status}:")
        recs = engine.recommend(cust, n_recommendations=5)
        print(recs.to_string(index=False))

