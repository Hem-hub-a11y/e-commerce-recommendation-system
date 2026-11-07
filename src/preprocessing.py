"""
Data Preprocessing Pipeline for E-Commerce Recommendation System.

Handles raw transaction loading, data hygiene, administrative code filtering,
cancellation reconciliation, canonical product catalog building, and data serialization.
"""

import os
import re
import logging
from typing import Tuple, Dict, Set
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Known non-merchandise administrative StockCodes to exclude
ADMINISTRATIVE_CODES: Set[str] = {
    "POST",          # Postage
    "D",             # Discount
    "M",             # Manual entry
    "DOT",           # DOTCOM Postage
    "C2",            # Carriage
    "BANK CHARGES",  # Bank service charges
    "TEST001",       # System test product 1
    "TEST002",       # System test product 2
    "ADJUST",        # Inventory adjustment
    "ADJUST2",       # Inventory adjustment 2
    "AMAZONFEE",     # Amazon marketplace fee
    "CRUK",          # Cancer Research UK donation/commission
    "S",             # Samples
    "B",             # Bad debt adjustment
    "PADS",          # Packaging pads
}


def load_raw_data(filepath: str, sheet_name: str = "Year 2009-2010") -> pd.DataFrame:
    """
    Load raw Excel transaction dataset and apply initial formatting.

    Args:
        filepath: Path to online_retail_II.xlsx
        sheet_name: Worksheet name to load

    Returns:
        DataFrame with raw transaction records.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")

    logger.info(f"Loading raw data from '{filepath}', sheet '{sheet_name}'...")
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    logger.info(f"Loaded {len(df):,} raw records with {df.shape[1]} columns.")

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Clean string columns
    for col in ["Invoice", "StockCode", "Description", "Country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def is_administrative_code(stock_code: str) -> bool:
    """
    Check if a stock code represents non-merchandise administrative overhead.

    Args:
        stock_code: Product stock code string

    Returns:
        True if code is administrative/system entry, False otherwise.
    """
    code_upper = str(stock_code).strip().upper()
    if code_upper in ADMINISTRATIVE_CODES:
        return True
    # Catch any test or adjustment code variants
    if re.match(r"^(TEST|ADJUST|GIFT_\d)", code_upper):
        return True
    return False


def build_product_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a canonical StockCode -> Description catalog.
    
    Multiple raw records have slight description variations for the same StockCode.
    This selects the most frequent, clean description for each unique valid StockCode.

    Args:
        df: Cleaned transactions DataFrame

    Returns:
        DataFrame indexed by StockCode with canonical Description and median Price.
    """
    logger.info("Building canonical product catalog...")
    valid = df[
        df["StockCode"].notnull()
        & (~df["StockCode"].apply(is_administrative_code))
        & (df["Description"].notnull())
        & (df["Description"] != "nan")
        & (df["Price"] > 0)
    ].copy()

    # Find most frequent description per StockCode
    desc_counts = (
        valid.groupby(["StockCode", "Description"])
        .size()
        .reset_index(name="count")
        .sort_values(["StockCode", "count"], ascending=[True, False])
    )
    canonical_desc = desc_counts.drop_duplicates(subset=["StockCode"], keep="first")[
        ["StockCode", "Description"]
    ]

    # Median unit price and total transactions per StockCode
    price_stats = (
        valid.groupby("StockCode")
        .agg(
            MedianPrice=("Price", "median"),
            TotalTransactions=("Quantity", "count"),
            TotalQuantity=("Quantity", "sum")
        )
        .reset_index()
    )

    catalog = pd.merge(canonical_desc, price_stats, on="StockCode", how="inner")
    catalog["Description"] = catalog["Description"].str.upper().str.strip()
    logger.info(f"Canonical catalog created with {len(catalog):,} unique products.")
    return catalog


def clean_transactions(
    df: pd.DataFrame,
    drop_unregistered_customers: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Perform rigorous e-commerce data hygiene on transaction logs.

    Steps:
      1. Audit and report missing Customer IDs.
      2. Drop exact duplicate transaction rows.
      3. Filter administrative/overhead non-stock codes.
      4. Filter invalid non-positive prices.
      5. Reconcile cancellation invoices ('C' prefix and negative quantities):
         aggregates total purchases per (Customer ID, StockCode) and ensures
         net interaction quantity is positive.

    Args:
        df: Raw transactions DataFrame
        drop_unregistered_customers: Whether to drop transactions with null Customer ID

    Returns:
        Tuple of (clean_df, audit_stats)
    """
    stats: Dict[str, int] = {"raw_records": len(df)}

    # 1. Customer ID check
    missing_cust_mask = df["Customer ID"].isnull() | (df["Customer ID"] == "nan")
    stats["missing_customer_ids"] = int(missing_cust_mask.sum())
    logger.info(
        f"Missing Customer IDs: {stats['missing_customer_ids']:,} "
        f"({stats['missing_customer_ids'] / len(df) * 100:.2f}%)"
    )

    if drop_unregistered_customers:
        df_clean = df[~missing_cust_mask].copy()
        # Format Customer ID as integer string (e.g. '13085' not '13085.0')
        df_clean["Customer ID"] = (
            df_clean["Customer ID"].astype(float).astype(int).astype(str)
        )
    else:
        df_clean = df.copy()

    stats["after_customer_id_filter"] = len(df_clean)

    # 2. Drop duplicate rows
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    stats["duplicate_records_dropped"] = initial_len - len(df_clean)
    logger.info(f"Dropped {stats['duplicate_records_dropped']:,} duplicate rows.")

    # 3. Filter administrative StockCodes
    admin_mask = df_clean["StockCode"].apply(is_administrative_code)
    stats["administrative_records_dropped"] = int(admin_mask.sum())
    df_clean = df_clean[~admin_mask].copy()
    logger.info(
        f"Dropped {stats['administrative_records_dropped']:,} administrative records."
    )

    # 4. Filter invalid prices
    invalid_price_mask = df_clean["Price"] <= 0
    stats["invalid_price_records"] = int(invalid_price_mask.sum())
    df_clean = df_clean[~invalid_price_mask].copy()
    logger.info(f"Dropped {stats['invalid_price_records']:,} non-positive price rows.")

    # 5. Handle cancellations and compute net interaction quantities
    # Canceled invoices start with 'C'
    is_cancellation = df_clean["Invoice"].str.startswith("C") | (df_clean["Quantity"] <= 0)
    stats["cancellation_records"] = int(is_cancellation.sum())
    logger.info(f"Identified {stats['cancellation_records']:,} cancellation/return records.")

    # Calculate net quantity per customer-item interaction across the dataset
    customer_item_summary = (
        df_clean.groupby(["Customer ID", "StockCode"])
        .agg(
            NetQuantity=("Quantity", "sum"),
            TransactionCount=("Quantity", "count"),
            LastInvoiceDate=("InvoiceDate", "max"),
            FirstInvoiceDate=("InvoiceDate", "min")
        )
        .reset_index()
    )

    stats["total_customer_item_pairs"] = len(customer_item_summary)
    net_non_positive = customer_item_summary["NetQuantity"] <= 0
    stats["cancelled_customer_item_pairs"] = int(net_non_positive.sum())

    logger.info(
        f"Total Customer-Product interaction pairs: {len(customer_item_summary):,}. "
        f"Pairs with net quantity <= 0 (fully returned/canceled): "
        f"{stats['cancelled_customer_item_pairs']:,}."
    )

    # Keep only interactions where the customer retained a positive net quantity
    valid_interactions = customer_item_summary[~net_non_positive]
    stats["valid_positive_interaction_pairs"] = len(valid_interactions)

    # Filter raw transactions to only keep positive purchases for valid pairs
    # (Excludes the negative return line items from the finalized transaction history)
    df_positive_transactions = df_clean[
        (~is_cancellation)
        & (df_clean["Quantity"] > 0)
    ].copy()

    # Ensure transaction customer-product matches valid retained purchases
    valid_pair_keys = set(zip(valid_interactions["Customer ID"], valid_interactions["StockCode"]))
    transaction_keys = list(zip(df_positive_transactions["Customer ID"], df_positive_transactions["StockCode"]))
    df_positive_transactions["is_valid_retained"] = [k in valid_pair_keys for k in transaction_keys]
    df_final = df_positive_transactions[df_positive_transactions["is_valid_retained"]].drop(
        columns=["is_valid_retained"]
    ).copy()

    stats["final_clean_transactions"] = len(df_final)
    stats["unique_customers"] = df_final["Customer ID"].nunique()
    stats["unique_products"] = df_final["StockCode"].nunique()

    logger.info(
        f"Cleaned dataset: {stats['final_clean_transactions']:,} transactions, "
        f"{stats['unique_customers']:,} customers, {stats['unique_products']:,} unique products."
    )

    return df_final, stats


def run_preprocessing_pipeline(
    raw_path: str,
    output_dir: str,
    sheet_name: str = "Year 2009-2010"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Execute complete end-to-end preprocessing workflow and persist artifacts.

    Args:
        raw_path: Path to raw online_retail_II.xlsx
        output_dir: Destination directory for processed files
        sheet_name: Sheet to process

    Returns:
        Tuple of (clean_transactions, product_catalog, stats)
    """
    os.makedirs(output_dir, exist_ok=True)

    df_raw = load_raw_data(raw_path, sheet_name=sheet_name)
    df_clean, stats = clean_transactions(df_raw, drop_unregistered_customers=True)
    catalog = build_product_catalog(df_clean)

    # Attach canonical description to clean transactions
    df_clean = df_clean.drop(columns=["Description"], errors="ignore")
    df_clean = pd.merge(
        df_clean,
        catalog[["StockCode", "Description"]],
        on="StockCode",
        how="inner"
    )

    # Save to compressed CSV and standard CSV (plus Parquet if engine available)
    clean_csv_gz = os.path.join(output_dir, "online_retail_clean.csv.gz")
    clean_csv = os.path.join(output_dir, "online_retail_clean.csv")
    catalog_csv = os.path.join(output_dir, "product_catalog.csv")

    logger.info(f"Saving cleaned transactions to {clean_csv_gz} and {clean_csv}...")
    df_clean.to_csv(clean_csv_gz, index=False, compression="gzip")
    df_clean.to_csv(clean_csv, index=False)

    logger.info(f"Saving product catalog to {catalog_csv}...")
    catalog.to_csv(catalog_csv, index=False)

    # Optional parquet export if pyarrow/fastparquet is available
    try:
        clean_parquet = os.path.join(output_dir, "online_retail_clean.parquet")
        catalog_parquet = os.path.join(output_dir, "product_catalog.parquet")
        df_clean.to_parquet(clean_parquet, index=False)
        catalog.to_parquet(catalog_parquet, index=False)
        logger.info(f"Exported Parquet files to {output_dir}.")
    except (ImportError, Exception) as e:
        logger.info(f"Parquet engine not available ({e}); using compressed CSV (.csv.gz).")

    logger.info("Preprocessing pipeline completed successfully.")
    return df_clean, catalog, stats


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_file = os.path.join(project_root, "data", "online_retail_II.xlsx")
    out_dir = os.path.join(project_root, "data")

    clean_df, catalog_df, pipeline_stats = run_preprocessing_pipeline(
        raw_path=raw_file,
        output_dir=out_dir,
        sheet_name="Year 2009-2010"
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE SUMMARY STATISTICS")
    print("=" * 60)
    for key, value in pipeline_stats.items():
        print(f"  {key:<35}: {value:,}")
    print("=" * 60)

