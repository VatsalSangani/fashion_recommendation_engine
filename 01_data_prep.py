"""
Step 1: Data Exploration & Preparation for H&M Recommendation Engine
=====================================================================
Run: python 01_data_prep.py
Prereqs: pip install pandas numpy scikit-learn pyarrow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gc
import json

DATA_DIR = Path("data")
OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Load & Explore ──────────────────────────────────────────────
print("=" * 60)
print("LOADING DATA...")
print("=" * 60)

articles = pd.read_csv(DATA_DIR / "articles.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")

txn_dtypes = {
    "t_dat": str,
    "customer_id": str,
    "article_id": str,
    "price": np.float32,
    "sales_channel_id": np.int8,
}
transactions = pd.read_csv(DATA_DIR / "transactions_train.csv", dtype=txn_dtypes)
transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])

print(f"\nArticles:      {articles.shape}")
print(f"Customers:     {customers.shape}")
print(f"Transactions:  {transactions.shape}")
print(f"Date range:    {transactions['t_dat'].min()} → {transactions['t_dat'].max()}")
print(f"Unique customers who purchased: {transactions['customer_id'].nunique():,}")
print(f"Unique articles purchased:      {transactions['article_id'].nunique():,}")

# ── 2. Article feature summary ─────────────────────────────────────
print("\n" + "=" * 60)
print("ARTICLE FEATURES")
print("=" * 60)
print(articles.dtypes)
print(f"\nKey categorical columns & cardinality:")
for col in ["product_type_name", "product_group_name", "colour_group_name",
            "department_name", "index_group_name", "section_name", "garment_group_name"]:
    if col in articles.columns:
        print(f"  {col}: {articles[col].nunique()} unique")

# ── 3. Filter to recent data (last 3 months) ──────────────────────
cutoff = transactions["t_dat"].max() - pd.Timedelta(days=90)
txn_recent = transactions[transactions["t_dat"] >= cutoff].copy()
print(f"\n{'=' * 60}")
print(f"FILTERING TO LAST 90 DAYS")
print(f"{'=' * 60}")
print(f"Transactions: {len(transactions):,} → {len(txn_recent):,}")
print(f"Customers:    {txn_recent['customer_id'].nunique():,}")
print(f"Articles:     {txn_recent['article_id'].nunique():,}")

del transactions
gc.collect()

# ── 4. Encode IDs to integers ─────────────────────────────────────
print("\nEncoding IDs...")
customer_ids = txn_recent["customer_id"].unique()
article_ids = txn_recent["article_id"].unique()

cust2idx = {c: i for i, c in enumerate(customer_ids)}
art2idx = {a: i for i, a in enumerate(article_ids)}
idx2cust = {i: c for c, i in cust2idx.items()}
idx2art = {i: a for a, i in art2idx.items()}

txn_recent["customer_idx"] = txn_recent["customer_id"].map(cust2idx)
txn_recent["article_idx"] = txn_recent["article_id"].map(art2idx)

print(f"Customer index range: 0 → {len(cust2idx) - 1}")
print(f"Article index range:  0 → {len(art2idx) - 1}")

# ── 5. Train/validation split (last 7 days = val) ─────────────────
val_cutoff = txn_recent["t_dat"].max() - pd.Timedelta(days=7)
txn_train = txn_recent[txn_recent["t_dat"] <= val_cutoff]
txn_val = txn_recent[txn_recent["t_dat"] > val_cutoff]

print(f"\nTrain: {len(txn_train):,} transactions  ({txn_train['t_dat'].min()} → {txn_train['t_dat'].max()})")
print(f"Val:   {len(txn_val):,} transactions  ({txn_val['t_dat'].min()} → {txn_val['t_dat'].max()})")

# ── 6. Prepare article features — KEEP MEANINGFUL NAMES ───────────
print("\nPreparing article features...")

# Numeric features to keep directly
numeric_feat_cols = [
    "product_type_no", "colour_group_code", "department_no",
    "index_group_no", "section_no", "garment_group_no",
]
numeric_feat_cols = [c for c in numeric_feat_cols if c in articles.columns]

# Categorical features to one-hot encode (low cardinality ones)
cat_feat_cols = ["product_group_name", "index_code"]
cat_feat_cols = [c for c in cat_feat_cols if c in articles.columns]

article_features = articles[["article_id"] + numeric_feat_cols + cat_feat_cols].copy()

# One-hot encode categoricals — pandas will create named columns like
# "product_group_name_Garment Upper body" instead of anonymous feat_0
if cat_feat_cols:
    article_features = pd.get_dummies(
        article_features, columns=cat_feat_cols, drop_first=True, dtype=np.float32
    )

# Clean column names: replace spaces and special chars for XGBoost compatibility
article_features.columns = [
    col.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    for col in article_features.columns
]

feat_cols = [c for c in article_features.columns if c != "article_id"]
print(f"Article feature matrix: {article_features.shape}")
print(f"Feature columns ({len(feat_cols)}):")
for col in feat_cols:
    print(f"  - {col}")

# ── 7. Save everything ────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_DIR}/...")
txn_train.to_parquet(OUTPUT_DIR / "txn_train.parquet", index=False)
txn_val.to_parquet(OUTPUT_DIR / "txn_val.parquet", index=False)
article_features.to_parquet(OUTPUT_DIR / "article_features.parquet", index=False)

with open(OUTPUT_DIR / "cust2idx.json", "w") as f:
    json.dump(cust2idx, f)
with open(OUTPUT_DIR / "art2idx.json", "w") as f:
    json.dump(art2idx, f)
with open(OUTPUT_DIR / "idx2art.json", "w") as f:
    json.dump({str(k): v for k, v in idx2art.items()}, f)
with open(OUTPUT_DIR / "idx2cust.json", "w") as f:
    json.dump({str(k): v for k, v in idx2cust.items()}, f)

print("\n✅ Data preparation complete!")
print(f"   Files saved in ./{OUTPUT_DIR}/")
print("\n   Next step: python 02_train_candidate_gen.py")