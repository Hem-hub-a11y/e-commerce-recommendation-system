# E-Commerce Product Recommendation System: Production ML Engineering Upgrade

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Unit_Tests-26%2F26_Passed-brightgreen.svg)]()
[![Code Architecture](https://img.shields.io/badge/Architecture-Modular_SciPy_CSR-orange.svg)]()

A production-grade, mathematically verified E-Commerce Product Recommendation Engine built on the **Online Retail II** transaction dataset.

Upgraded from an academic prototype into a production-ready machine learning system featuring:
- Automated data hygiene & cancellation reconciliation.
- High-performance sparse matrices (`scipy.sparse.csr_matrix`, **41.6x memory reduction**).
- Multiple implicit feedback signal representations (`binary`, `log`, `iuf`, `raw`).
- Rigorous out-of-time chronological evaluation (**zero future-data leakage**).
- Comprehensive multi-algorithm benchmarks: **Popularity Baseline**, **User-Based CF**, **Item-Based CF**, **Matrix Factorization (TruncatedSVD)**, and **Hybrid Ensembles**.
- High-level inference facade with canonical catalog enrichment (Description, Unit Price, Rank).
- Complete automated unit test suite (**26/26 tests passing**).

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Dataset Description & Preprocessing Protocol](#2-dataset-description--preprocessing-protocol)
3. [System Architecture](#3-system-architecture)
4. [Mathematical Foundations & ML Algorithms](#4-mathematical-foundations--ml-algorithms)
5. [Evaluation Methodology & Protocol](#5-evaluation-methodology--protocol)
6. [Empirical Experimental Results](#6-empirical-experimental-results)
7. [Project Structure](#7-project-structure)
8. [Installation & Quickstart Guide](#8-installation--quickstart-guide)
9. [Automated Testing Suite](#9-automated-testing-suite)
10. [Limitations & Future Roadmap](#10-limitations--future-roadmap)

---

## 1. Project Overview & Problem Statement

### The Problem
E-commerce platforms track millions of user transactions. Unlike movie streaming or survey platforms where users assign explicit 1-to-5 star ratings, e-commerce data consists purely of **implicit feedback** (purchase logs, quantities, timestamps). 

Naive collaborative filtering models applied to transaction logs suffer from three critical flaws:
1. **Wholesale Outlier Distortion:** Bulk purchasers buying thousands of units distort vector magnitudes and dominate cosine similarity calculations.
2. **Cancellation Signal Inversion:** Dropping negative return entries without reconciling purchase quantities treats canceled/returned items as positive endorsements.
3. **Data Leakage & Evaluation Absence:** Evaluating models without strict chronological train/test splitting leads to catastrophic temporal data leakage (training on future transactions to predict the past).

### The Objective
Transform an academic notebook into an optimized, modular, reproducible AI/ML recommendation engine that maximizes top-$K$ precision, recall, and ranking order (MAP@K) under realistic production constraints.

---

## 2. Dataset Description & Preprocessing Protocol

### Source Dataset
* **Dataset:** [Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) (UCI Machine Learning Repository / Kaggle).
* **Domain:** UK-based registered non-store online retail platform.
* **Timeline:** 01/12/2009 to 09/12/2010 (525,461 raw records).

### Data Preprocessing & Hygiene Pipeline (`src/preprocessing.py`)

| Preprocessing Step | Issue Identified | Engineering Solution | Quantitative Impact |
| :--- | :--- | :--- | :--- |
| **Missing Customer IDs** | 20.54% of transactions lack authenticated User IDs | Dropped rows missing `Customer ID` (guest checkouts) | 107,927 rows removed |
| **Duplicate Transmissions** | Duplicate network log records | Exact row deduplication via `df.drop_duplicates()` | **6,771 duplicate rows dropped** |
| **Administrative Non-Stock Codes** | Accounting and system overhead entries (`POST`, `D`, `M`, `BANK CHARGES`, etc.) | Set/Regex filter for administrative entries | **1,809 non-retail rows filtered** |
| **Cancellation Reversal** | Invoices starting with `'C'` represent returns. Naive filtering kept the positive purchase | Reconciled net quantity: $\sum_{t} q_t$ per customer-item pair | **2,809 false-positive pairs purged** |
| **Product Fragmentation** | 424 StockCodes had multiple differing descriptions (typos/casing) | Built canonical `StockCode -> Description` catalog | Standardized to **3,997 canonical products** |
| **Serialization** | Loading 45 MB Excel via `openpyxl` took ~75 seconds | Compressed Gzip CSV (`.csv.gz`) with typed schema | **1.08s load time (~70x speedup)** |

---

## 3. System Architecture

```
[Raw Transactions: online_retail_II.xlsx]
                   │
                   ▼ (src/preprocessing.py)
[Clean Transactions & Canonical Catalog: online_retail_clean.csv.gz]
                   │
                   ▼ (src/evaluation.py: Chronological Split at 2010-11-01)
     ┌─────────────────────────────┴─────────────────────────────┐
     ▼                                                           ▼
[Train Set: 324,918 rows (Dec 2009 - Oct 2010)]      [Test Set: 72,646 rows (Nov - Dec 2010)]
     │                                                           │
     ▼ (src/feature_engineering.py)                              │
[Sparse CSR Matrix: 3,896 Users × 3,931 Items]                   │
     │ (Sparsity: 98.51% | Memory: 3.13 MB)                      │
     │                                                           │
     ├──► 1. Popularity Baseline Recommender                     │
     │                                                           │
     ├──► 2. User-Based CF (Consensus Dot Product)               │
     │                                                           │
     ├──► 3. Item-Based CF (Pruned Top-K Item Sim)               │
     │                                                           │
     ├──► 4. Matrix Factorization (TruncatedSVD, d=50)           │
     │                                                           │
     └──► 5. Hybrid Ensemble (50% IBCF + 50% SVD)                │
                   │                                             │
                   ▼                                             │
       [Top-K Candidate Generation]                              │
                   │                                             │
                   ▼ (History Masking: exclude I_train)          │
       [Candidate Recommendation Lists]                          │
                   │                                             │
                   └───────────────────────┬─────────────────────┘
                                           │
                                           ▼ (src/evaluation.py)
                  [Ranking Metrics: Precision@K, Recall@K, MAP@K, HitRate@K]
```

---

## 4. Mathematical Foundations & ML Algorithms

### A. Implicit Feedback Transformations
To prevent wholesale outliers from skewing vector similarities, we formulate four signal transformations:
1. **Binary Interaction:**
   $$p_{u, i} = \begin{cases} 1 & \text{if } q_{u, i} > 0 \\ 0 & \text{otherwise} \end{cases}$$
2. **Sublinear Log-Scaling:**
   $$\tilde{r}_{u, i} = \log(1 + q_{u, i})$$
3. **Inverse User Frequency (IUF) / TF-IDF Weighting:**
   $$r_{u, i}^{\text{IUF}} = \log(1 + q_{u, i}) \cdot \left[ \log\left(\frac{|U| + 1}{|\{u \in U : q_{u, i} > 0\}| + 1}\right) + 1 \right]$$

### B. User-Based Collaborative Filtering (UBCF)
Measures peer alignment between users $\mathbf{u}, \mathbf{v} \in \mathbb{R}^N$ via Cosine Similarity:
$$s(u, v) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^N r_{u,i} r_{v,i}}{\sqrt{\sum_{i=1}^N r_{u,i}^2} \sqrt{\sum_{i=1}^N r_{v,i}^2}}$$

Predicted affinity score for unpurchased item $i \notin \mathcal{I}_{\text{train}}(u)$:
$$\hat{r}_{u, i} = \sum_{v \in \mathcal{N}_K(u)} s(u, v) \cdot r_{v, i}$$

### C. Item-Based Collaborative Filtering (IBCF)
Transposes interaction matrix $\mathbf{R}^T \in \mathbb{R}^{N \times M}$ and computes item-item similarity:
$$s(i, j) = \frac{\mathbf{i} \cdot \mathbf{j}}{\|\mathbf{i}\|_2 \|\mathbf{j}\|_2}$$

Predicted affinity for user $u$ on candidate item $i$:
$$\hat{r}_{u, i} = \sum_{j \in \mathcal{I}_{\text{train}}(u) \cap \mathcal{N}_K(i)} s(i, j) \cdot r_{u, j}$$

### D. Matrix Factorization via TruncatedSVD
Decomposes interaction matrix $\mathbf{R} \in \mathbb{R}^{M \times N}$ into low-rank user factors $\mathbf{P} \in \mathbb{R}^{M \times d}$ and item components $\mathbf{Q} \in \mathbb{R}^{N \times d}$:
$$\mathbf{R} \approx \mathbf{P} \mathbf{Q}^T$$
Predicted preference vector for user $u$:
$$\hat{\mathbf{r}}_u = \mathbf{p}_u \cdot \mathbf{Q}^T$$

---

## 5. Evaluation Methodology & Protocol

### Temporal Split (Zero Data Leakage)
* **Cutoff Date:** `2010-11-01 00:00:00`.
* **Training Window:** 324,918 transactions (Dec 1, 2009 – Oct 31, 2010).
* **Test Window:** 72,646 transactions (Nov 1, 2010 – Dec 9, 2010).
* **Evaluated Cohort:** **1,456 warm repeat customers** present in both train and test partitions.
* **History Masking:** Any product purchased during training is strictly masked ($\hat{r}_{u, i} = -\infty$) to measure true discovery.

### Evaluation Metrics
1. **Precision@K:**
   $$\text{Precision}@K(u) = \frac{|\hat{\mathcal{I}}_K(u) \cap \mathcal{I}_{\text{test}}(u)|}{K}$$
2. **Recall@K:**
   $$\text{Recall}@K(u) = \frac{|\hat{\mathcal{I}}_K(u) \cap \mathcal{I}_{\text{test}}(u)|}{|\mathcal{I}_{\text{test}}(u)|}$$
3. **Average Precision at K (AP@K):**
   $$\text{AP}@K(u) = \frac{1}{\min(K, |\mathcal{I}_{\text{test}}(u)|)} \sum_{k=1}^K P@k(u) \cdot \mathbb{I}(i_k \in \mathcal{I}_{\text{test}}(u))$$
4. **Mean Average Precision (MAP@K):**
   $$\text{MAP}@K = \frac{1}{|U_{\text{eval}}|} \sum_{u \in U_{\text{eval}}} \text{AP}@K(u)$$
5. **Hit Rate at K (HR@K):**
   $$\text{HR}@K(u) = \mathbb{I}(|\hat{\mathcal{I}}_K(u) \cap \mathcal{I}_{\text{test}}(u)| > 0)$$

---

## 6. Empirical Experimental Results

### Master Algorithm Comparison (1,456 Warm Test Customers)

| Model Paradigm | Representation | Parameters / Config | Precision@5 | Recall@5 | MAP@5 | Hit Rate@5 | Hit Rate@10 | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | Purchase Counts | Top Frequency | 0.0372 | 0.0058 | 0.0198 | 16.00% | 23.01% | 0.78s |
| **User-Based CF** (Original Notebook) | `raw` | $K=20$, Normalized | 0.0191 | 0.0038 | 0.0095 | 8.93% | 15.45% | 3.82s |
| **User-Based CF** (Optimized) | `binary` | $K=20$, Consensus | 0.0702 | 0.0143 | 0.0422 | 25.96% | 37.02% | 3.75s |
| **Item-Based CF** (Optimized) | `binary` | $K=20$, Consensus | 0.0720 | 0.0148 | 0.0447 | 25.21% | 37.57% | 4.12s |
| **Hybrid Ensemble** | `binary` | $50\%\text{ IBCF} + 50\%\text{ SVD}$ | 0.0804 | 0.0157 | 0.0493 | 27.61% | 39.84% | 5.20s |
| **Matrix Factorization (SVD)** 🏆 | **`binary`** | **$d=50$ components** | **0.0845** | **0.0160** | **0.0507** | **29.95%** | **41.96%** | **2.28s** |

### Key Experimental Insights:
1. **Raw Quantities Fail:** The original notebook's raw quantity approach performed worse than a non-personalized Popularity Baseline ($P@5 = 1.91\%$ vs $3.72\%$) due to wholesale outlier distortion.
2. **Binary Feedback Wins:** Binary transformation increased UBCF Precision@5 from **1.91% to 7.02%** (+267% increase).
3. **Latent SVD Dominates:** Matrix Factorization achieved the highest overall performance ($P@5 = 8.45\%$, $\text{MAP}@5 = 0.0507$, $\text{HitRate}@10 = 41.96\%$). Its dense latent factors effectively bridge the 98.51% sparsity gap where neighborhood models lack exact co-purchases.
4. **Cold-Start Validation:** On 373 brand-new customers with zero prior interactions, our Popularity Fallback captured purchases for **40.21% of users at $K=5$** and **53.62% of users at $K=10$**.

---

## 7. Project Structure

```
e-commerce-recommendation-system/
│
├── data/
│   ├── README.md                      # Dataset provenance & documentation
│   ├── online_retail_clean.csv.gz     # Processed, reconciled transactions (~6 MB)
│   └── product_catalog.csv            # Canonical StockCode -> Description & Price catalog
│
├── notebooks/
│   └── recommendation_analysis.ipynb  # Reference exploratory analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Data hygiene, cancellation & catalog pipeline
│   ├── feature_engineering.py         # SciPy CSR sparse matrix builder & transformers
│   ├── collaborative_filtering.py     # Vectorized UBCF & IBCF recommendation models
│   ├── matrix_factorization.py        # Latent factor TruncatedSVD recommendation engine
│   ├── evaluation.py                  # Chronological split & ranking metrics (P@K, MAP@K)
│   └── recommendations.py             # High-level production inference facade
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py          # Unit tests for data hygiene & cancellations
│   ├── test_feature_engineering.py    # Unit tests for sparse matrix & transformations
│   ├── test_collaborative_filtering.py# Unit tests for UBCF and IBCF engines
│   ├── test_matrix_factorization.py   # Unit tests for SVD factor dimensions & masking
│   └── test_recommendations.py        # End-to-end integration tests for inference facade
│
├── results/
│   ├── master_benchmark.csv           # Full comparative metrics across all 5 paradigms
│   ├── collaborative_filtering_benchmark.csv
│   └── svd_benchmark.csv
│
├── requirements.txt                   # Production dependencies
├── .gitignore                         # Strict exclusion of raw datasets and binaries
└── README.md                          # Technical engineering portfolio documentation
```

---

## 8. Installation & Quickstart Guide

### Prerequisites
* Python 3.10+ (Tested on Python 3.13)
* Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/Hem-hub-a11y/e-commerce-recommendation-system.git
cd e-commerce-recommendation-system

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
.\venv\Scripts\activate        # On Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Preprocessing Pipeline
```bash
python src/preprocessing.py
```

### Running the End-to-End Recommendation Engine
```python
import pandas as pd
from src.recommendations import RecommendationEngine

# Load cleaned transaction history
df = pd.read_csv("data/online_retail_clean.csv.gz")

# Initialize and fit production SVD engine (d=50)
engine = RecommendationEngine(
    model_type="svd",
    catalog_path="data/product_catalog.csv",
    n_components=50
)
engine.fit(df)

# Generate top-5 recommendations for customer 12347
recs = engine.recommend(customer_id="12347", n_recommendations=5)
print(recs)
```

**Output:**
```text
Rank StockCode                         Description EstimatedPrice
   1    84997A GREEN 3 PIECE MINI DOTS CUTLERY SET          £3.75
   2     22624                IVORY KITCHEN SCALES          £8.50
   3     21977  PACK OF 60 PINK PAISLEY CAKE CASES          £0.55
   4     22326  ROUND SNACK BOXES SET OF4 WOODLAND          £2.95
   5     84378       SET OF 3 HEART COOKIE CUTTERS          £1.25
```

---

## 9. Automated Testing Suite

The repository includes a comprehensive unit test suite covering input validation, matrix shapes, neighbor rankings, exclusion of prior purchases, and cold-start fallbacks:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Test Execution Output:**
```text
Ran 26 tests in 0.334s
OK
```

---

## 10. Limitations & Future Roadmap

1. **Explicit Negative Feedback:** E-commerce logs only capture positive implicit feedback (purchases). Customers who view an item but choose not to purchase provide a negative signal that can be modeled via clickstream data.
2. **Implicit Alternating Least Squares (iALS):** Implementing Hu, Koren & Volinsky's weighted matrix factorization ($c_{u,i} = 1 + \alpha r_{u,i}$) using coordinate descent could further enhance latent factor representations.
3. **Session-Based & Sequential Recommenders:** Integrating deep sequential architectures (SASRec, BERT4Rec, or GRU4Rec) to capture immediate intra-session purchase intent.
4. **Real-Time Vector Search:** Deploying user/item embeddings into approximate nearest neighbor (ANN) vector indices (e.g., FAISS or ScaNN) for sub-millisecond retrieval at scale.

---

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
