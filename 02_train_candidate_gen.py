"""
Step 2: Train Candidate Generation Model (ALS Collaborative Filtering)
========================================================================
This is Stage 1 of the two-stage recsys — retrieve ~100 candidate items per user.
Maps to ASOS's "Similar Items" / "People Also Viewed" systems.

Run: python 02_train_candidate_gen.py
Prereqs: pip install implicit scipy pandas numpy
"""

import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from pathlib import Path
import pickle
import time

PROCESSED_DIR = Path("processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ── 1. Load prepared data ─────────────────────────────────────────
print("Loading data...")
txn_train = pd.read_parquet(PROCESSED_DIR / "txn_train.parquet")
txn_val = pd.read_parquet(PROCESSED_DIR / "txn_val.parquet")

with open(PROCESSED_DIR / "art2idx.json") as f:
    art2idx = json.load(f)
with open(PROCESSED_DIR / "cust2idx.json") as f:
    cust2idx = json.load(f)
with open(PROCESSED_DIR / "idx2art.json") as f:
    idx2art = {int(k): v for k, v in json.load(f).items()}

n_customers = len(cust2idx)
n_articles = len(art2idx)
print(f"Customers: {n_customers:,}  |  Articles: {n_articles:,}")

# ── 2. Build user-item interaction matrix ──────────────────────────
# Confidence = 1 + alpha * count (implicit feedback)
print("\nBuilding interaction matrix...")
interaction = txn_train.groupby(["customer_idx", "article_idx"]).size().reset_index(name="count")

# Confidence weighting — more purchases = stronger signal
alpha = 40
interaction["confidence"] = 1 + alpha * np.log1p(interaction["count"])

user_item = csr_matrix(
    (interaction["confidence"].values,
     (interaction["customer_idx"].values, interaction["article_idx"].values)),
    shape=(n_customers, n_articles),
)
print(f"Interaction matrix: {user_item.shape}, nnz={user_item.nnz:,}")
print(f"Sparsity: {1 - user_item.nnz / (n_customers * n_articles):.6f}")

# ── 3. Train ALS model ────────────────────────────────────────────
print("\nTraining ALS model...")
# Hyperparams — tuned for fashion recsys
model = AlternatingLeastSquares(
    factors=128,          # embedding dimension
    regularization=0.01,
    iterations=15,
    use_gpu=False,        # set True if you have CUDA
    random_state=42,
)

start = time.time()
model.fit(user_item)
elapsed = time.time() - start
print(f"Training took {elapsed:.1f}s")

# ── 4. Evaluate: Recall@K on validation set ───────────────────────
print("\nEvaluating on validation set...")

# For each user in validation, check if their actual purchases
# appear in the model's top-K recommendations
K_VALUES = [12, 50, 100]

# Get validation users who also appear in training
val_users = txn_val[txn_val["customer_idx"].notna()]["customer_idx"].unique().astype(int)
# Only evaluate users the model has seen
val_users = val_users[val_users < n_customers]
print(f"Evaluating {len(val_users):,} validation users...")

# Sample if too many (for speed)
if len(val_users) > 5000:
    rng = np.random.default_rng(42)
    val_users = rng.choice(val_users, 5000, replace=False)
    print(f"Sampled down to {len(val_users):,} users for eval speed")

# Build ground truth: what each val user actually bought
val_ground_truth = (
    txn_val[txn_val["customer_idx"].isin(val_users)]
    .groupby("customer_idx")["article_idx"]
    .apply(set)
    .to_dict()
)

recalls = {k: [] for k in K_VALUES}
for user_idx in val_users:
    if user_idx not in val_ground_truth:
        continue
    true_items = val_ground_truth[user_idx]

    # Get recommendations (exclude already purchased in training)
    item_ids, scores = model.recommend(
        user_idx, user_item[user_idx], N=max(K_VALUES), filter_already_liked_items=True
    )

    for k in K_VALUES:
        top_k = set(item_ids[:k])
        hit = len(top_k & true_items)
        recalls[k].append(hit / len(true_items))

print("\n📊 Results:")
print("-" * 40)
for k in K_VALUES:
    mean_recall = np.mean(recalls[k])
    print(f"  Recall@{k:>3}: {mean_recall:.4f}")
print("-" * 40)
print("(Recall@100 is what matters for candidate gen — we just need")
print(" the true item to be IN the candidate set for Stage 2 to rank)")

# ── 5. Save model & artifacts ─────────────────────────────────────
print("\nSaving model...")
with open(MODEL_DIR / "als_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the interaction matrix (needed at inference for user history)
from scipy.sparse import save_npz
save_npz(MODEL_DIR / "user_item_matrix.npz", user_item)

# Save item factors separately (for "similar items" endpoint)
np.save(MODEL_DIR / "item_factors.npy", model.item_factors)

print("\n✅ Candidate generation model saved!")
print(f"   Model:          {MODEL_DIR}/als_model.pkl")
print(f"   Item factors:   {MODEL_DIR}/item_factors.npy")
print(f"   User-item mat:  {MODEL_DIR}/user_item_matrix.npz")
print(f"\n   Next step: python 03_train_ranker.py")