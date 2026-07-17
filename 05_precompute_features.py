"""
Pre-compute all behavioral features into small JSON files.
This eliminates the need to load txn_train.parquet at API startup.

Run: python 05_precompute_features.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

PROCESSED_DIR = Path("processed")
DEPLOY_DIR = Path("deploy_data")
DEPLOY_DIR.mkdir(exist_ok=True)

print("Loading transaction data...")
txn_train = pd.read_parquet(PROCESSED_DIR / "txn_train.parquet")
articles_full = pd.read_csv(Path("data") / "articles.csv", dtype=str)

with open(PROCESSED_DIR / "art2idx.json") as f:
    art2idx = json.load(f)

train_end_date = txn_train["t_dat"].max()

# ── Item features ──
print("Computing item features...")
item_purchase_counts = txn_train.groupby("article_idx").size().to_dict()
item_unique_buyers = txn_train.groupby("article_idx")["customer_idx"].nunique().to_dict()
item_last = txn_train.groupby("article_idx")["t_dat"].max()
item_recency_days = {k: (train_end_date - v).days for k, v in item_last.items()}
item_avg_price = txn_train.groupby("article_idx")["price"].mean().to_dict()

item_features = {}
for aidx in item_purchase_counts:
    item_features[str(aidx)] = {
        "pop": round(float(np.log1p(item_purchase_counts.get(aidx, 0))), 4),
        "buyers": round(float(np.log1p(item_unique_buyers.get(aidx, 0))), 4),
        "recency": int(item_recency_days.get(aidx, 90)),
        "price": round(float(item_avg_price.get(aidx, 0)), 6),
    }

with open(DEPLOY_DIR / "item_features.json", "w") as f:
    json.dump(item_features, f)
print(f"  Saved {len(item_features):,} item features")

# ── User features ──
print("Computing user features...")
user_purchase_counts = txn_train.groupby("customer_idx").size().to_dict()
user_unique_items = txn_train.groupby("customer_idx")["article_idx"].nunique().to_dict()
user_avg_price = txn_train.groupby("customer_idx")["price"].mean().to_dict()
user_last = txn_train.groupby("customer_idx")["t_dat"].max()
user_recency_days = {k: (train_end_date - v).days for k, v in user_last.items()}

user_features = {}
for uid in user_purchase_counts:
    user_features[str(uid)] = {
        "total_log": round(float(np.log1p(user_purchase_counts.get(uid, 0))), 4),
        "unique": float(user_unique_items.get(uid, 0)),
        "price": round(float(user_avg_price.get(uid, 0)), 6),
        "recency": int(user_recency_days.get(uid, 90)),
    }

with open(DEPLOY_DIR / "user_features.json", "w") as f:
    json.dump(user_features, f)
print(f"  Saved {len(user_features):,} user features")

# ── Category affinity ──
print("Computing category affinity...")
art_idx_to_group = {}
art_idx_to_dept = {}
for _, row in articles_full.iterrows():
    aid = str(row["article_id"]).zfill(10)
    if aid in art2idx:
        aidx = art2idx[aid]
        art_idx_to_group[aidx] = row.get("product_group_name", "Unknown")
        art_idx_to_dept[aidx] = str(row.get("department_no", "0"))

txn_with_group = txn_train.copy()
txn_with_group["product_group"] = txn_with_group["article_idx"].map(art_idx_to_group)
user_gc = txn_with_group.groupby(["customer_idx", "product_group"]).size().unstack(fill_value=0)
user_group_fracs = user_gc.div(user_gc.sum(axis=1), axis=0)

cat_affinity = {}
for uid in user_group_fracs.index:
    row = user_group_fracs.loc[uid]
    non_zero = row[row > 0]
    if len(non_zero) > 0:
        cat_affinity[str(uid)] = {g: round(float(v), 4) for g, v in non_zero.items()}

with open(DEPLOY_DIR / "cat_affinity.json", "w") as f:
    json.dump(cat_affinity, f)
print(f"  Saved affinity for {len(cat_affinity):,} users")

# ── User department sets ──
print("Computing user department sets...")
user_dept_sets = {}
for uid, group in txn_train.groupby("customer_idx")["article_idx"]:
    depts = set()
    for aidx in group:
        d = art_idx_to_dept.get(aidx)
        if d:
            depts.add(d)
    user_dept_sets[str(uid)] = list(depts)

with open(DEPLOY_DIR / "user_depts.json", "w") as f:
    json.dump(user_dept_sets, f)
print(f"  Saved dept sets for {len(user_dept_sets):,} users")

# ── Mappings ──
with open(DEPLOY_DIR / "art_idx_to_group.json", "w") as f:
    json.dump({str(k): v for k, v in art_idx_to_group.items()}, f)
with open(DEPLOY_DIR / "art_idx_to_dept.json", "w") as f:
    json.dump({str(k): v for k, v in art_idx_to_dept.items()}, f)

# ── Copy files needed for serving ──
for fname in ["art2idx.json", "idx2art.json", "cust2idx.json",
              "article_metadata.json", "article_features.parquet"]:
    src = PROCESSED_DIR / fname
    if src.exists():
        shutil.copy2(src, DEPLOY_DIR / fname)

MODEL_DIR = Path("models")
for fname in ["als_model.pkl", "item_factors.npy", "user_item_matrix.npz",
              "xgb_ranker.json", "ranker_feature_names.json"]:
    src = MODEL_DIR / fname
    if src.exists():
        shutil.copy2(src, DEPLOY_DIR / fname)

# ── Report sizes ──
total = sum(f.stat().st_size for f in DEPLOY_DIR.rglob("*") if f.is_file())
print(f"\n✅ All deploy files saved to {DEPLOY_DIR}/")
print(f"   Total size: {total / 1024 / 1024:.1f} MB")
for f in sorted(DEPLOY_DIR.iterdir()):
    size = f.stat().st_size / 1024 / 1024
    print(f"     {f.name}: {size:.1f} MB")