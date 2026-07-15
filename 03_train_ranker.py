"""
Stage 2: Ranking Model (XGBoost LambdaMART)
=============================================
Production-grade ranking pipeline with:
  - Rich user-item interaction features (not just article metadata)
  - Hard negative mining for training
  - Realistic end-to-end evaluation: ALS top-100 → XGBoost re-rank → top-K
  - ALS baseline comparison
  - MLflow experiment tracking

Run: python 03_train_ranker.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
import xgboost as xgb
from sklearn.metrics import ndcg_score
import mlflow
import time

PROCESSED_DIR = Path("processed")
MODEL_DIR = Path("models")
NEG_RATIO = 4
N_TRAIN_USERS = 10000
N_VAL_USERS = 3000
K = 12
N_CANDIDATES = 100

# ── 1. Load Data & Models ─────────────────────────────────────────
print("Loading models and data...")
with open(MODEL_DIR / "als_model.pkl", "rb") as f:
    als_model = pickle.load(f)

user_item = load_npz(MODEL_DIR / "user_item_matrix.npz")
item_factors = np.load(MODEL_DIR / "item_factors.npy")

txn_train = pd.read_parquet(PROCESSED_DIR / "txn_train.parquet")
txn_val = pd.read_parquet(PROCESSED_DIR / "txn_val.parquet")
article_features = pd.read_parquet(PROCESSED_DIR / "article_features.parquet")
articles_full = pd.read_csv(Path("data") / "articles.csv", dtype=str)

with open(PROCESSED_DIR / "idx2art.json") as f:
    idx2art = {int(k): v for k, v in json.load(f).items()}
with open(PROCESSED_DIR / "art2idx.json") as f:
    art2idx = json.load(f)

n_articles = user_item.shape[1]

# Article feature lookup
art_id_col = "article_id"
feat_col_names = [c for c in article_features.columns if c != art_id_col]
art_feat_dict = {
    str(int(row[art_id_col])).zfill(10): row[feat_col_names].values.astype(np.float32)
    for _, row in article_features.iterrows()
}


# ── 2. Build User & Item Behavioral Features ─────────────────────
print("\nBuilding behavioral features from training transactions...")

# --- Item popularity features ---
item_purchase_counts = txn_train.groupby("article_idx").size()
item_unique_buyers = txn_train.groupby("article_idx")["customer_idx"].nunique()

# Item recency: days since last purchase (relative to train end)
train_end_date = txn_train["t_dat"].max()
item_last_purchase = txn_train.groupby("article_idx")["t_dat"].max()
item_recency_days = (train_end_date - item_last_purchase).dt.days

# Item average price
item_avg_price = txn_train.groupby("article_idx")["price"].mean()

print(f"  Item features built for {len(item_purchase_counts):,} articles")

# --- User profile features ---
user_purchase_counts = txn_train.groupby("customer_idx").size()
user_unique_items = txn_train.groupby("customer_idx")["article_idx"].nunique()
user_avg_price = txn_train.groupby("customer_idx")["price"].mean()
user_last_purchase = txn_train.groupby("customer_idx")["t_dat"].max()
user_recency_days = (train_end_date - user_last_purchase).dt.days

# User category preferences: fraction of purchases in each product_group
# Map article_idx → product_group_name
art_idx_to_group = {}
for _, row in articles_full.iterrows():
    aid = str(row["article_id"]).zfill(10)
    if aid in art2idx:
        art_idx_to_group[art2idx[aid]] = row.get("product_group_name", "Unknown")

txn_with_group = txn_train.copy()
txn_with_group["product_group"] = txn_with_group["article_idx"].map(art_idx_to_group)
user_group_counts = txn_with_group.groupby(["customer_idx", "product_group"]).size().unstack(fill_value=0)
user_group_fracs = user_group_counts.div(user_group_counts.sum(axis=1), axis=0)

# Map article_idx → department
art_idx_to_dept = {}
for _, row in articles_full.iterrows():
    aid = str(row["article_id"]).zfill(10)
    if aid in art2idx:
        art_idx_to_dept[art2idx[aid]] = row.get("department_no", "0")

print(f"  User features built for {len(user_purchase_counts):,} customers")


# ── 3. Feature Extraction Function ────────────────────────────────
def extract_features(user_idx, art_idx, als_score):
    """
    Build a rich feature vector for a (user, item) pair.
    
    Features:
      - als_score: collaborative filtering relevance
      - Article metadata: product_type, colour, department, etc.
      - Item popularity: purchase count, unique buyers, recency
      - Item price: average transaction price
      - User profile: activity level, avg price, recency
      - User-item affinity: does this item's category match user preference?
    """
    art_id = idx2art.get(int(art_idx))
    if not art_id or art_id not in art_feat_dict:
        return None

    article_feat = art_feat_dict[art_id]

    # Item behavioral features
    item_pop = float(item_purchase_counts.get(art_idx, 0))
    item_buyers = float(item_unique_buyers.get(art_idx, 0))
    item_rec = float(item_recency_days.get(art_idx, 90))
    item_price = float(item_avg_price.get(art_idx, 0))

    # Popularity features (log-scaled)
    item_pop_log = float(np.log1p(item_pop))
    item_buyers_log = float(np.log1p(item_buyers))

    # User behavioral features
    user_total = float(user_purchase_counts.get(user_idx, 0))
    user_unique = float(user_unique_items.get(user_idx, 0))
    user_price = float(user_avg_price.get(user_idx, 0))
    user_rec = float(user_recency_days.get(user_idx, 90))
    user_total_log = float(np.log1p(user_total))

    # Price affinity: how close is item price to user's average
    price_diff = abs(item_price - user_price) if user_price > 0 else 0
    price_ratio = item_price / user_price if user_price > 0 else 1.0

    # Category affinity: user's historical fraction in this item's product group
    item_group = art_idx_to_group.get(art_idx, None)
    if item_group and user_idx in user_group_fracs.index and item_group in user_group_fracs.columns:
        cat_affinity = float(user_group_fracs.loc[user_idx, item_group])
    else:
        cat_affinity = 0.0

    # Department match: has user bought from this department before?
    item_dept = art_idx_to_dept.get(art_idx, "0")
    # Simple binary: did user buy anything from this department?
    user_depts = txn_train[txn_train["customer_idx"] == user_idx]["article_idx"].map(art_idx_to_dept)
    dept_match = float(item_dept in user_depts.values) if len(user_depts) > 0 else 0.0

    features = {
        "als_score": als_score,
        # Article metadata
        **{feat_col_names[j]: float(article_feat[j]) for j in range(len(article_feat))},
        # Item behavioral
        "item_popularity_log": item_pop_log,
        "item_unique_buyers_log": item_buyers_log,
        "item_recency_days": item_rec,
        "item_avg_price": item_price,
        # User behavioral
        "user_total_purchases_log": user_total_log,
        "user_unique_items": user_unique,
        "user_avg_price": user_price,
        "user_recency_days": user_rec,
        # User-item interaction
        "price_diff": price_diff,
        "price_ratio": min(price_ratio, 10.0),  # cap outliers
        "category_affinity": cat_affinity,
        "department_match": dept_match,
    }
    return features


# Pre-compute user departments to avoid repeated lookups
print("Pre-computing user department sets...")
user_dept_sets = {}
for uid, group in txn_train.groupby("customer_idx")["article_idx"]:
    depts = set()
    for aidx in group:
        d = art_idx_to_dept.get(aidx)
        if d:
            depts.add(d)
    user_dept_sets[uid] = depts


def extract_features_fast(user_idx, art_idx, als_score):
    """Optimized feature extraction using pre-computed lookups."""
    art_id = idx2art.get(int(art_idx))
    if not art_id or art_id not in art_feat_dict:
        return None

    article_feat = art_feat_dict[art_id]

    item_pop_log = float(np.log1p(item_purchase_counts.get(art_idx, 0)))
    item_buyers_log = float(np.log1p(item_unique_buyers.get(art_idx, 0)))
    item_rec = float(item_recency_days.get(art_idx, 90))
    item_price = float(item_avg_price.get(art_idx, 0))

    user_total_log = float(np.log1p(user_purchase_counts.get(user_idx, 0)))
    user_unique = float(user_unique_items.get(user_idx, 0))
    user_price = float(user_avg_price.get(user_idx, 0))
    user_rec = float(user_recency_days.get(user_idx, 90))

    price_diff = abs(item_price - user_price) if user_price > 0 else 0
    price_ratio = min(item_price / user_price if user_price > 0 else 1.0, 10.0)

    item_group = art_idx_to_group.get(art_idx, None)
    if item_group and user_idx in user_group_fracs.index and item_group in user_group_fracs.columns:
        cat_affinity = float(user_group_fracs.loc[user_idx, item_group])
    else:
        cat_affinity = 0.0

    item_dept = art_idx_to_dept.get(art_idx, "0")
    dept_match = 1.0 if item_dept in user_dept_sets.get(user_idx, set()) else 0.0

    return {
        "als_score": als_score,
        **{feat_col_names[j]: float(article_feat[j]) for j in range(len(article_feat))},
        "item_popularity_log": item_pop_log,
        "item_unique_buyers_log": item_buyers_log,
        "item_recency_days": item_rec,
        "item_avg_price": item_price,
        "user_total_purchases_log": user_total_log,
        "user_unique_items": user_unique,
        "user_avg_price": user_price,
        "user_recency_days": user_rec,
        "price_diff": price_diff,
        "price_ratio": price_ratio,
        "category_affinity": cat_affinity,
        "department_match": dept_match,
    }


# ── 4. Build Training Dataset ─────────────────────────────────────
print("\n" + "=" * 60)
print("BUILDING TRAINING DATASET")
print("=" * 60)

rng = np.random.default_rng(42)

train_user_pool = txn_train["customer_idx"].unique()
train_users = rng.choice(train_user_pool, min(N_TRAIN_USERS, len(train_user_pool)), replace=False)

val_user_pool = txn_val["customer_idx"].unique()
val_user_pool = val_user_pool[np.isin(val_user_pool, train_user_pool)]
val_users = rng.choice(val_user_pool, min(N_VAL_USERS, len(val_user_pool)), replace=False)

print(f"\nTrain users: {len(train_users):,}  |  Val users: {len(val_users):,}")

user_purchases_train = txn_train.groupby("customer_idx")["article_idx"].apply(set).to_dict()

rows = []
start = time.time()

for i, user_idx in enumerate(train_users):
    user_idx = int(user_idx)
    purchased = user_purchases_train.get(user_idx, set())
    if not purchased:
        continue

    user_factor = als_model.user_factors[user_idx]

    # Positives
    for art_idx in purchased:
        als_score = float(user_factor @ item_factors[int(art_idx)])
        feat = extract_features_fast(user_idx, int(art_idx), als_score)
        if feat:
            feat["user_idx"] = user_idx
            feat["label"] = 1
            rows.append(feat)

    # Hard negatives
    candidate_ids, _ = als_model.recommend(
        userid=user_idx, user_items=user_item[user_idx],
        N=200, filter_already_liked_items=True,
    )
    hard_negatives = [int(cid) for cid in candidate_ids if int(cid) not in purchased]
    neg_sample = hard_negatives[: NEG_RATIO * len(purchased)]

    for art_idx in neg_sample:
        als_score = float(user_factor @ item_factors[art_idx])
        feat = extract_features_fast(user_idx, art_idx, als_score)
        if feat:
            feat["user_idx"] = user_idx
            feat["label"] = 0
            rows.append(feat)

    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(train_users)} users...")

df_rank_train = pd.DataFrame(rows).sort_values("user_idx").reset_index(drop=True)
elapsed = time.time() - start
print(f"\n  Built {len(df_rank_train):,} samples in {elapsed:.1f}s")
print(f"  Positive: {int((df_rank_train.label==1).sum()):,}  "
      f"Negative: {int((df_rank_train.label==0).sum()):,}")

# Get feature names (everything except user_idx and label)
feature_names = [c for c in df_rank_train.columns if c not in ("user_idx", "label")]
print(f"  Features ({len(feature_names)}): {feature_names}")


# ── 5. Train XGBoost Ranker ───────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING XGBOOST RANKER")
print("=" * 60)

X_train = df_rank_train[feature_names]
y_train = df_rank_train["label"].to_numpy()
groups_train = df_rank_train.groupby("user_idx").size().values

# Inner train/val split for early stopping
split_idx = int(len(groups_train) * 0.8)
train_end = int(np.sum(groups_train[:split_idx]))

dtrain = xgb.DMatrix(X_train.iloc[:train_end], label=y_train[:train_end], feature_names=feature_names)
dtrain.set_group(groups_train[:split_idx])
dval_inner = xgb.DMatrix(X_train.iloc[train_end:], label=y_train[train_end:], feature_names=feature_names)
dval_inner.set_group(groups_train[split_idx:])

# Monotone constraint: als_score (feature 0) must be monotonically increasing
# This means XGBoost can REFINE the ALS ordering but never fully override it
# 1 = increasing constraint, 0 = no constraint
monotone = tuple([1] + [0] * (len(feature_names) - 1))

params = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@12",
    "eta": 0.05,             # slower learning — less aggressive overriding
    "max_depth": 4,          # shallower trees — less overfitting
    "min_child_weight": 20,  # more conservative splits
    "subsample": 0.8,
    "colsample_bytree": 0.6, # see fewer features per tree — regularise
    "seed": 42,
    "monotone_constraints": monotone,
}

ranker = xgb.train(
    params, dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train"), (dval_inner, "holdout")],
    early_stopping_rounds=30,
    verbose_eval=20,
)

best_rounds = ranker.best_iteration + 1
print(f"\nRetraining on full data with {best_rounds} rounds...")
dtrain_full = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtrain_full.set_group(groups_train)
ranker = xgb.train(params, dtrain_full, num_boost_round=best_rounds)


# ── 6. Realistic End-to-End Evaluation ────────────────────────────
print("\n" + "=" * 60)
print(f"END-TO-END EVALUATION (ALS top-{N_CANDIDATES} → re-rank → top-{K})")
print("=" * 60)

val_purchases = txn_val.groupby("customer_idx")["article_idx"].apply(set).to_dict()

als_recalls, als_ndcgs, als_maps, als_precs = [], [], [], []
xgb_recalls, xgb_ndcgs, xgb_maps, xgb_precs = [], [], [], []
# Blended: alpha * als_score + (1-alpha) * xgb_score
ALPHAS = [0.3, 0.5, 0.7, 0.9]
blend_results = {a: {"recalls": [], "precs": [], "maps": [], "ndcgs": []} for a in ALPHAS}
skipped = 0

eval_start = time.time()

for i, user_idx in enumerate(val_users):
    user_idx = int(user_idx)
    true_items = val_purchases.get(user_idx, set())
    if not true_items:
        skipped += 1
        continue

    # ALS candidate generation
    candidate_ids, als_scores = als_model.recommend(
        userid=user_idx, user_items=user_item[user_idx],
        N=N_CANDIDATES, filter_already_liked_items=True,
    )

    candidate_list = [int(cid) for cid in candidate_ids]
    relevance = np.array([1 if cid in true_items else 0 for cid in candidate_list])

    if relevance.sum() == 0:
        skipped += 1
        continue

    n_relevant = int(relevance.sum())
    als_score_array = np.array([float(s) for s in als_scores])

    # ── ALS-only metrics ──
    als_top_k_idx = np.argsort(-als_score_array)[:K]
    als_top_rel = relevance[als_top_k_idx]
    als_hits = int(als_top_rel.sum())
    ak = len(als_top_rel)

    als_recalls.append(als_hits / n_relevant)
    als_precs.append(als_hits / ak)
    als_ndcgs.append(ndcg_score([relevance], [als_score_array], k=min(K, len(relevance))))
    cs = np.cumsum(als_top_rel)
    pai = cs / np.arange(1, ak + 1)
    als_maps.append((pai * als_top_rel).sum() / min(n_relevant, ak))

    # ── XGBoost re-ranking ──
    rank_rows = []
    valid_indices = []

    for j, (cid, als_s) in enumerate(zip(candidate_list, als_scores)):
        feat = extract_features_fast(user_idx, cid, float(als_s))
        if feat:
            row = [feat[fn] for fn in feature_names]
            rank_rows.append(row)
            valid_indices.append(j)

    if rank_rows:
        dmatrix = xgb.DMatrix(np.array(rank_rows), feature_names=feature_names)
        xgb_scores = ranker.predict(dmatrix)

        scored = sorted(zip(valid_indices, xgb_scores), key=lambda x: -x[1])
        xgb_top_k_idx = [idx for idx, _ in scored[:K]]
        xgb_top_rel = relevance[xgb_top_k_idx]
        xgb_hits = int(xgb_top_rel.sum())
        xk = len(xgb_top_rel)

        xgb_recalls.append(xgb_hits / n_relevant)
        xgb_precs.append(xgb_hits / xk)

        full_scores = np.zeros(len(candidate_list))
        for idx, score in scored:
            full_scores[idx] = score
        xgb_ndcgs.append(ndcg_score([relevance], [full_scores], k=min(K, len(relevance))))

        cs_x = np.cumsum(xgb_top_rel)
        pai_x = cs_x / np.arange(1, xk + 1)
        xgb_maps.append((pai_x * xgb_top_rel).sum() / min(n_relevant, xk))

        # ── Blended scores: alpha * ALS + (1-alpha) * XGBoost ──
        # Normalise both score arrays to [0,1] for fair blending
        als_norm = als_score_array.copy()
        als_range = als_norm.max() - als_norm.min()
        if als_range > 0:
            als_norm = (als_norm - als_norm.min()) / als_range

        xgb_full_norm = full_scores.copy()
        xgb_range = xgb_full_norm.max() - xgb_full_norm.min()
        if xgb_range > 0:
            xgb_full_norm = (xgb_full_norm - xgb_full_norm.min()) / xgb_range

        for alpha in ALPHAS:
            blended = alpha * als_norm + (1 - alpha) * xgb_full_norm
            bl_top_k = np.argsort(-blended)[:K]
            bl_rel = relevance[bl_top_k]
            bl_hits = int(bl_rel.sum())
            bk = len(bl_rel)

            blend_results[alpha]["recalls"].append(bl_hits / n_relevant)
            blend_results[alpha]["precs"].append(bl_hits / bk)
            blend_results[alpha]["ndcgs"].append(
                ndcg_score([relevance], [blended], k=min(K, len(relevance)))
            )
            cs_b = np.cumsum(bl_rel)
            pai_b = cs_b / np.arange(1, bk + 1)
            blend_results[alpha]["maps"].append(
                (pai_b * bl_rel).sum() / min(n_relevant, bk)
            )

    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(val_users)} users...")

eval_elapsed = time.time() - eval_start

als_metrics = {
    f"Recall@{K}": np.mean(als_recalls), f"Precision@{K}": np.mean(als_precs),
    f"MAP@{K}": np.mean(als_maps), f"NDCG@{K}": np.mean(als_ndcgs),
}
xgb_metrics = {
    f"Recall@{K}": np.mean(xgb_recalls), f"Precision@{K}": np.mean(xgb_precs),
    f"MAP@{K}": np.mean(xgb_maps), f"NDCG@{K}": np.mean(xgb_ndcgs),
}
n_evaluated = len(als_ndcgs)


# ── 7. MLflow Logging ─────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("hm-fashion-recommendation")

with mlflow.start_run(run_name="als_xgb_rich_features_v5"):
    mlflow.log_params({
        "candidate_generator": "ALS",
        "ranking_model": "XGBoost LambdaMART",
        "negative_strategy": "hard_negatives_from_ALS",
        "evaluation_strategy": "end_to_end_realistic",
        "n_candidates": N_CANDIDATES,
        "evaluation_k": K,
        "neg_ratio": NEG_RATIO,
        "n_train_users": len(train_users),
        "n_val_users": len(val_users),
        "n_val_evaluated": n_evaluated,
        "n_val_skipped": skipped,
        "n_train_samples": len(df_rank_train),
        "feature_count": len(feature_names),
        "best_rounds": best_rounds,
        "features_added": "item_popularity,item_recency,user_profile,price_affinity,category_affinity,dept_match",
    })

    for name, val in als_metrics.items():
        mlflow.log_metric(f"als_{name.replace('@','_at_')}", val)
    for name, val in xgb_metrics.items():
        mlflow.log_metric(f"xgb_{name.replace('@','_at_')}", val)

    importance = ranker.get_score(importance_type="gain")
    mlflow.log_dict(importance, "feature_importance.json")
    mlflow.xgboost.log_model(ranker, artifact_path="xgb_ranker")


# ── 8. Save & Print ──────────────────────────────────────────────
ranker.save_model(str(MODEL_DIR / "xgb_ranker.json"))
with open(MODEL_DIR / "ranker_feature_names.json", "w") as f:
    json.dump(feature_names, f)

print(f"\n{'=' * 60}")
print(f"📊 RESULTS — ALS vs ALS+XGBoost")
print(f"   Pipeline: ALS top-{N_CANDIDATES} → XGBoost re-rank → top-{K}")
print(f"   Validation: temporal (last 7 days)")
print(f"   Evaluation took {eval_elapsed:.1f}s")
print(f"{'=' * 60}")
print(f"{'Metric':<20} {'ALS Only':>12} {'ALS+XGBoost':>12} {'Δ Gain':>10}")
print(f"{'-'*56}")
for name in [f"Recall@{K}", f"Precision@{K}", f"MAP@{K}", f"NDCG@{K}"]:
    a = als_metrics[name]
    x = xgb_metrics[name]
    d = x - a
    pct = (d / a * 100) if a > 0 else 0
    sign = "+" if d >= 0 else ""
    print(f"  {name:<18} {a:>10.4f}   {x:>10.4f}   {sign}{d:.4f} ({sign}{pct:.1f}%)")

# Blended results
print(f"\n  Score Blending: alpha * ALS + (1-alpha) * XGBoost")
print(f"  {'Alpha':<10} {'NDCG@12':>10} {'Recall@12':>10} {'MAP@12':>10}")
print(f"  {'-'*42}")
print(f"  {'ALS only':<10} {als_metrics[f'NDCG@{K}']:>10.4f} {als_metrics[f'Recall@{K}']:>10.4f} {als_metrics[f'MAP@{K}']:>10.4f}")
best_alpha, best_ndcg = None, 0
for alpha in ALPHAS:
    br = blend_results[alpha]
    ndcg_val = np.mean(br["ndcgs"])
    rec_val = np.mean(br["recalls"])
    map_val = np.mean(br["maps"])
    print(f"  {alpha:<10} {ndcg_val:>10.4f} {rec_val:>10.4f} {map_val:>10.4f}")
    if ndcg_val > best_ndcg:
        best_ndcg = ndcg_val
        best_alpha = alpha
print(f"  {'XGB only':<10} {xgb_metrics[f'NDCG@{K}']:>10.4f} {xgb_metrics[f'Recall@{K}']:>10.4f} {xgb_metrics[f'MAP@{K}']:>10.4f}")
print(f"\n  → Best blend: alpha={best_alpha} (NDCG@{K}={best_ndcg:.4f})")

print(f"\n  Users evaluated: {n_evaluated} / {len(val_users)}")
print(f"  Users skipped: {skipped}")

print(f"\n  Top features by gain:")
for feat, gain in sorted(importance.items(), key=lambda x: -x[1])[:15]:
    print(f"    {feat}: {gain:.1f}")

print(f"\n✅ Model saved to {MODEL_DIR}/xgb_ranker.json")