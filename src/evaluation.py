"""
Model Evaluation and Metric Validation Module.

Implements temporal train/test splitting to prevent future-data leakage,
and vectorized calculation of standard recommendation ranking metrics:
  - Precision@K
  - Recall@K
  - AP@K (Average Precision)
  - MAP@K (Mean Average Precision)
  - HitRate@K

Also includes the Global Popularity Recommender as an essential baseline.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Callable
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def split_data_chronological(
    df: pd.DataFrame,
    split_date: str = "2010-11-01",
    user_col: str = "Customer ID",
    item_col: str = "StockCode",
    date_col: str = "InvoiceDate"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Split transaction history chronologically based on a global cutoff timestamp.
    
    Transactions before split_date form the training set.
    Transactions on or after split_date form the test evaluation set.
    
    Prevents temporal data leakage (future-data contamination).

    Args:
        df: Cleaned transactions DataFrame
        split_date: Cutoff date string (YYYY-MM-DD)
        user_col: Customer identifier column
        item_col: Product identifier column
        date_col: Timestamp column

    Returns:
        Tuple of (train_df, test_df, split_metadata)
    """
    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
    cutoff = pd.to_datetime(split_date)

    train_mask = df_sorted[date_col] < cutoff
    train_df = df_sorted[train_mask].copy()
    test_df = df_sorted[~train_mask].copy()

    train_users = set(train_df[user_col].astype(str).unique())
    test_users = set(test_df[user_col].astype(str).unique())

    warm_users = train_users & test_users
    cold_users = test_users - train_users

    metadata = {
        "split_date": split_date,
        "train_transactions": len(train_df),
        "test_transactions": len(test_df),
        "train_date_range": (str(train_df[date_col].min()), str(train_df[date_col].max())),
        "test_date_range": (str(test_df[date_col].min()), str(test_df[date_col].max())),
        "train_users_count": len(train_users),
        "test_users_count": len(test_users),
        "warm_users_count": len(warm_users),
        "cold_users_count": len(cold_users),
        "unique_train_items": train_df[item_col].nunique(),
        "unique_test_items": test_df[item_col].nunique(),
    }

    logger.info(
        f"Chronological split at {split_date}: "
        f"{len(train_df):,} train rows, {len(test_df):,} test rows. "
        f"Warm test users: {len(warm_users):,}, Cold-start test users: {len(cold_users):,}."
    )
    return train_df, test_df, metadata


def build_user_ground_truth(
    df: pd.DataFrame,
    user_col: str = "Customer ID",
    item_col: str = "StockCode"
) -> Dict[str, Set[str]]:
    """
    Create a dictionary mapping each user to their set of purchased items.

    Args:
        df: Transactions DataFrame
        user_col: Customer identifier column
        item_col: Product identifier column

    Returns:
        Dict mapping user_id -> set of item_ids
    """
    grouped = df.groupby(user_col)[item_col].unique()
    return {str(user): set(str(item) for item in items) for user, items in grouped.items()}


def precision_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Compute Precision@K:
      Precision@K = |Recommended@K ∩ GroundTruth| / K

    Args:
        recommended: Ordered list of recommended item IDs
        ground_truth: Set of actual items purchased by user in test set
        k: Ranking cutoff

    Returns:
        Precision score in [0.0, 1.0]
    """
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & ground_truth)
    return float(hits) / float(k)


def recall_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Compute Recall@K:
      Recall@K = |Recommended@K ∩ GroundTruth| / |GroundTruth|

    Args:
        recommended: Ordered list of recommended item IDs
        ground_truth: Set of actual items purchased by user in test set
        k: Ranking cutoff

    Returns:
        Recall score in [0.0, 1.0]
    """
    if len(ground_truth) == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & ground_truth)
    return float(hits) / float(len(ground_truth))


def average_precision_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Compute Average Precision at K (AP@K):
      AP@K = (1 / min(K, |GroundTruth|)) * sum_{i=1}^K [P@i * I(rec_i in GroundTruth)]

    Args:
        recommended: Ordered list of recommended item IDs
        ground_truth: Set of actual items purchased by user in test set
        k: Ranking cutoff

    Returns:
        AP@K score in [0.0, 1.0]
    """
    if len(ground_truth) == 0 or k <= 0:
        return 0.0

    rec_k = recommended[:k]
    score = 0.0
    hits = 0.0

    for i, item in enumerate(rec_k, 1):
        if item in ground_truth:
            hits += 1.0
            score += hits / float(i)

    normalizer = min(k, len(ground_truth))
    return float(score / float(normalizer)) if normalizer > 0 else 0.0


def hit_rate_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Compute HitRate@K (1.0 if at least one correct recommendation in top-K, 0.0 otherwise).
    """
    rec_k = recommended[:k]
    return 1.0 if bool(set(rec_k) & ground_truth) else 0.0


def evaluate_recommendations(
    recommendations: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate recommendation dictionary across multiple cutoff values of K.

    Args:
        recommendations: Dict mapping user_id -> list of recommended items
        ground_truth: Dict mapping user_id -> set of true items
        k_values: List of cutoffs (e.g. [5, 10, 20])

    Returns:
        Dictionary of mean metric scores (Precision@K, Recall@K, MAP@K, HitRate@K)
    """
    eval_users = [u for u in ground_truth if len(ground_truth[u]) > 0]
    if not eval_users:
        return {}

    metrics: Dict[str, float] = {}

    for k in k_values:
        precisions = []
        recalls = []
        aps = []
        hit_rates = []

        for user in eval_users:
            rec_items = recommendations.get(user, [])
            true_items = ground_truth[user]

            precisions.append(precision_at_k(rec_items, true_items, k))
            recalls.append(recall_at_k(rec_items, true_items, k))
            aps.append(average_precision_at_k(rec_items, true_items, k))
            hit_rates.append(hit_rate_at_k(rec_items, true_items, k))

        metrics[f"Precision@{k}"] = float(np.mean(precisions))
        metrics[f"Recall@{k}"] = float(np.mean(recalls))
        metrics[f"MAP@{k}"] = float(np.mean(aps))
        metrics[f"HitRate@{k}"] = float(np.mean(hit_rates))

    metrics["evaluated_users_count"] = len(eval_users)
    return metrics


class PopularityRecommender:
    """
    Global Popularity Baseline Recommender.
    
    Ranks items by training set interaction frequency.
    For any user, recommends top items they have not yet purchased.
    Serves as the cold-start fallback and baseline benchmark.
    """

    def __init__(self, item_col: str = "StockCode", user_col: str = "Customer ID"):
        self.item_col = item_col
        self.user_col = user_col
        self.popular_items: List[str] = []
        self.user_history: Dict[str, Set[str]] = {}

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """Fit popularity ordering based on training frequency."""
        counts = train_df[self.item_col].astype(str).value_counts()
        self.popular_items = counts.index.tolist()

        # Build training history to avoid recommending already-purchased items
        self.user_history = build_user_ground_truth(
            train_df, user_col=self.user_col, item_col=self.item_col
        )
        logger.info(
            f"Fitted PopularityRecommender: {len(self.popular_items):,} distinct items ranked."
        )
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_known: bool = True
    ) -> List[str]:
        """
        Recommend top-K popular items for a target user.

        Args:
            user_id: Customer ID string
            k: Number of items
            exclude_known: Whether to filter items user already purchased in train

        Returns:
            List of recommended item IDs
        """
        known = self.user_history.get(str(user_id), set()) if exclude_known else set()
        recs: List[str] = []
        for item in self.popular_items:
            if item not in known:
                recs.append(item)
                if len(recs) >= k:
                    break
        return recs

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


if __name__ == "__main__":
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    clean_csv_path = os.path.join(project_root, "data", "online_retail_clean.csv.gz")

    if not os.path.exists(clean_csv_path):
        print("Please run src/preprocessing.py first!")
        exit(1)

    df_clean = pd.read_csv(clean_csv_path)

    # 1. Perform Chronological Split
    train_df, test_df, meta = split_data_chronological(df_clean, split_date="2010-11-01")

    # 2. Build ground truth sets
    train_history = build_user_ground_truth(train_df)
    test_truth = build_user_ground_truth(test_df)

    # Warm users: Users present in both train and test
    warm_users = [u for u in test_truth if u in train_history]
    warm_test_truth = {u: test_truth[u] for u in warm_users}

    # 3. Fit and evaluate Popularity Baseline on warm users
    pop_model = PopularityRecommender().fit(train_df)
    pop_recs = pop_model.recommend_batch(warm_users, k=20, exclude_known=True)

    metrics = evaluate_recommendations(pop_recs, warm_test_truth, k_values=[5, 10, 20])

    print("\n" + "=" * 65)
    print("POPULARITY BASELINE BENCHMARK (CHRONOLOGICAL SPLIT)")
    print("=" * 65)
    print(f"Evaluated Warm Users: {metrics['evaluated_users_count']:,}")
    print(f"Precision@5 : {metrics['Precision@5']:.4f}  |  Recall@5 : {metrics['Recall@5']:.4f}  |  MAP@5 : {metrics['MAP@5']:.4f}  |  HR@5 : {metrics['HitRate@5']:.4f}")
    print(f"Precision@10: {metrics['Precision@10']:.4f}  |  Recall@10: {metrics['Recall@10']:.4f}  |  MAP@10: {metrics['MAP@10']:.4f}  |  HR@10: {metrics['HitRate@10']:.4f}")
    print(f"Precision@20: {metrics['Precision@20']:.4f}  |  Recall@20: {metrics['Recall@20']:.4f}  |  MAP@20: {metrics['MAP@20']:.4f}  |  HR@20: {metrics['HitRate@20']:.4f}")
    print("=" * 65)

