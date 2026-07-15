"""
H&M Recommendation Engine API
================================
Two-stage recommendation: ALS candidate generation + XGBoost re-ranking.
Serves frontend at root (/) and API endpoints.
Supports A/B model comparison: ALS Only vs ALS + XGBoost.

Run:   uvicorn 04_api:app --reload --port 8000
App:   http://localhost:8000
Docs:  http://localhost:8000/docs
"""

import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import load_npz
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL_DIR = Path("models")

app = FastAPI(
    title="H&M Recommendation Engine",
    description="Two-stage personalised fashion recommendations: ALS + XGBoost",
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Startup: Load Models & Pre-compute Features ───────────────────
@app.on_event("startup")
def load_models():
    global als_model, user_item, item_factors, xgb_ranker
    global art2idx, idx2art, cust2idx, feature_names
    global article_features_dict, article_metadata
    global item_purchase_counts, item_unique_buyers, item_recency_days, item_avg_price
    global user_purchase_counts, user_unique_items, user_avg_price, user_recency_days
    global user_group_fracs, art_idx_to_group, user_dept_sets, art_idx_to_dept
    global feat_col_names

    with open(MODEL_DIR / "als_model.pkl", "rb") as f:
        als_model = pickle.load(f)

    user_item = load_npz(MODEL_DIR / "user_item_matrix.npz")
    item_factors = np.load(MODEL_DIR / "item_factors.npy")

    xgb_ranker = xgb.Booster()
    xgb_ranker.load_model(str(MODEL_DIR / "xgb_ranker.json"))

    with open(MODEL_DIR / "ranker_feature_names.json") as f:
        feature_names = json.load(f)

    with open(Path("processed") / "art2idx.json") as f:
        art2idx = json.load(f)
    with open(Path("processed") / "idx2art.json") as f:
        idx2art = {int(k): v for k, v in json.load(f).items()}
    with open(Path("processed") / "cust2idx.json") as f:
        cust2idx = json.load(f)

    af = pd.read_parquet(Path("processed") / "article_features.parquet")
    feat_col_names = [c for c in af.columns if c != "article_id"]
    article_features_dict = {}
    for _, row in af.iterrows():
        article_features_dict[str(int(row["article_id"])).zfill(10)] = row[feat_col_names].values.astype(np.float32)

    articles_full = pd.read_csv(Path("data") / "articles.csv", dtype=str)
    article_metadata = {}
    for _, row in articles_full.iterrows():
        aid = str(row["article_id"]).zfill(10)
        article_metadata[aid] = {
            "prod_name": row.get("prod_name", ""),
            "product_type_name": row.get("product_type_name", ""),
            "product_group_name": row.get("product_group_name", ""),
            "colour_group_name": row.get("colour_group_name", ""),
            "department_name": row.get("department_name", ""),
            "section_name": row.get("section_name", ""),
            "detail_desc": row.get("detail_desc", ""),
        }

    txn_train = pd.read_parquet(Path("processed") / "txn_train.parquet")
    train_end_date = txn_train["t_dat"].max()

    item_purchase_counts = txn_train.groupby("article_idx").size().to_dict()
    item_unique_buyers = txn_train.groupby("article_idx")["customer_idx"].nunique().to_dict()
    item_last = txn_train.groupby("article_idx")["t_dat"].max()
    item_recency_days = {k: (train_end_date - v).days for k, v in item_last.items()}
    item_avg_price = txn_train.groupby("article_idx")["price"].mean().to_dict()

    user_purchase_counts = txn_train.groupby("customer_idx").size().to_dict()
    user_unique_items = txn_train.groupby("customer_idx")["article_idx"].nunique().to_dict()
    user_avg_price = txn_train.groupby("customer_idx")["price"].mean().to_dict()
    user_last = txn_train.groupby("customer_idx")["t_dat"].max()
    user_recency_days = {k: (train_end_date - v).days for k, v in user_last.items()}

    art_idx_to_group = {}
    art_idx_to_dept = {}
    for _, row in articles_full.iterrows():
        aid = str(row["article_id"]).zfill(10)
        if aid in art2idx:
            aidx = art2idx[aid]
            art_idx_to_group[aidx] = row.get("product_group_name", "Unknown")
            art_idx_to_dept[aidx] = row.get("department_no", "0")

    txn_with_group = txn_train.copy()
    txn_with_group["product_group"] = txn_with_group["article_idx"].map(art_idx_to_group)
    user_gc = txn_with_group.groupby(["customer_idx", "product_group"]).size().unstack(fill_value=0)
    user_group_fracs = user_gc.div(user_gc.sum(axis=1), axis=0)

    user_dept_sets = {}
    for uid, group in txn_train.groupby("customer_idx")["article_idx"]:
        depts = set()
        for aidx in group:
            d = art_idx_to_dept.get(aidx)
            if d:
                depts.add(d)
        user_dept_sets[uid] = depts

    del txn_train, txn_with_group
    print(f"✅ Models loaded ({len(article_metadata):,} articles, {len(user_purchase_counts):,} users)")


# ── Feature Builder ───────────────────────────────────────────────
def build_features(user_idx, art_idx, als_score):
    art_id = idx2art.get(int(art_idx))
    if not art_id or art_id not in article_features_dict:
        return None

    article_feat = article_features_dict[art_id]

    item_pop_log = float(np.log1p(item_purchase_counts.get(art_idx, 0)))
    item_buyers_log = float(np.log1p(item_unique_buyers.get(art_idx, 0)))
    item_rec = float(item_recency_days.get(art_idx, 90))
    i_price = float(item_avg_price.get(art_idx, 0))

    user_total_log = float(np.log1p(user_purchase_counts.get(user_idx, 0)))
    u_unique = float(user_unique_items.get(user_idx, 0))
    u_price = float(user_avg_price.get(user_idx, 0))
    u_rec = float(user_recency_days.get(user_idx, 90))

    price_diff = abs(i_price - u_price) if u_price > 0 else 0
    price_ratio = min(i_price / u_price if u_price > 0 else 1.0, 10.0)

    item_group = art_idx_to_group.get(art_idx, None)
    if item_group and user_idx in user_group_fracs.index and item_group in user_group_fracs.columns:
        cat_affinity = float(user_group_fracs.loc[user_idx, item_group])
    else:
        cat_affinity = 0.0

    item_dept = art_idx_to_dept.get(art_idx, "0")
    dept_match = 1.0 if item_dept in user_dept_sets.get(user_idx, set()) else 0.0

    feat_dict = {
        "als_score": als_score,
        **{feat_col_names[j]: float(article_feat[j]) for j in range(len(article_feat))},
        "item_popularity_log": item_pop_log,
        "item_unique_buyers_log": item_buyers_log,
        "item_recency_days": item_rec,
        "item_avg_price": i_price,
        "user_total_purchases_log": user_total_log,
        "user_unique_items": u_unique,
        "user_avg_price": u_price,
        "user_recency_days": u_rec,
        "price_diff": price_diff,
        "price_ratio": price_ratio,
        "category_affinity": cat_affinity,
        "department_match": dept_match,
    }
    return [feat_dict[fn] for fn in feature_names]


# ── Request Schemas ───────────────────────────────────────────────
class RecommendRequest(BaseModel):
    customer_id: str
    n_candidates: int = 100
    n_results: int = 12
    ranking_mode: str = "als_only"

class SimilarItemsRequest(BaseModel):
    article_id: str
    n_results: int = 12


def enrich(article_id, score_dict):
    meta = article_metadata.get(article_id, {})
    return {
        "article_id": article_id,
        **score_dict,
        "product_name": meta.get("prod_name", "Unknown"),
        "product_type": meta.get("product_type_name", ""),
        "color": meta.get("colour_group_name", ""),
        "department": meta.get("department_name", ""),
        "section": meta.get("section_name", ""),
        "description": meta.get("detail_desc", ""),
    }


def calculate_match_percentages(scores, min_pct=70, max_pct=99):
    """Converts raw ML scores to a user-friendly match percentage (70% - 99%)"""
    if not scores:
        return []
    max_s, min_s = max(scores), min(scores)
    if max_s == min_s:
        return [95] * len(scores)
    return [int(min_pct + (max_pct - min_pct) * ((s - min_s) / (max_s - min_s))) for s in scores]


# ── API Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "model": "two-stage-recsys-v2"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """Personalised recommendations with A/B model comparison."""
    if req.customer_id not in cust2idx:
        raise HTTPException(404, f"Customer {req.customer_id} not found")

    user_idx = cust2idx[req.customer_id]

    candidate_ids, als_scores = als_model.recommend(
        user_idx, user_item[user_idx],
        N=req.n_candidates,
        filter_already_liked_items=True,
    )

    valid_candidates = []
    als_score_list = []
    for art_idx, als_score in zip(candidate_ids, als_scores):
        art_id = idx2art.get(int(art_idx))
        if art_id:
            valid_candidates.append((int(art_idx), art_id))
            als_score_list.append(float(als_score))

    if not valid_candidates:
        return {"customer_id": req.customer_id, "recommendations": [], "stage1_candidates": 0, "ranking_mode": req.ranking_mode}

    if req.ranking_mode == "als_only":
        top = list(zip(valid_candidates[:req.n_results], als_score_list[:req.n_results]))
        raw_scores = [s for _, s in top]
        match_pcts = calculate_match_percentages(raw_scores)
        results = [
            enrich(art_id, {"match_percentage": pct})
            for ((_, art_id), _), pct in zip(top, match_pcts)
        ]
    else:
        rank_rows = []
        xgb_candidates = []
        for (art_idx, art_id), als_score in zip(valid_candidates, als_score_list):
            row = build_features(user_idx, art_idx, als_score)
            if row:
                rank_rows.append(row)
                xgb_candidates.append((art_idx, art_id))

        if not rank_rows:
            return {"customer_id": req.customer_id, "recommendations": [], "stage1_candidates": 0, "ranking_mode": req.ranking_mode}

        dmatrix = xgb.DMatrix(np.array(rank_rows), feature_names=feature_names)
        rank_scores = xgb_ranker.predict(dmatrix)
        ranked = sorted(zip(xgb_candidates, rank_scores), key=lambda x: -x[1])
        top = ranked[:req.n_results]
        raw_scores = [float(s) for _, s in top]
        match_pcts = calculate_match_percentages(raw_scores)
        results = [
            enrich(art_id, {"match_percentage": pct})
            for ((_, art_id), _), pct in zip(top, match_pcts)
        ]

    return {
        "customer_id": req.customer_id,
        "recommendations": results,
        "stage1_candidates": len(valid_candidates),
        "ranking_mode": req.ranking_mode,
    }


@app.post("/similar-items")
def similar_items(req: SimilarItemsRequest):
    """Item-to-item similarity — powers 'People Also Viewed' / 'Similar Items'."""
    if req.article_id not in art2idx:
        raise HTTPException(404, f"Article {req.article_id} not found")

    art_idx = art2idx[req.article_id]
    similar_ids, scores = als_model.similar_items(art_idx, N=req.n_results + 1)

    valid_pairs = [(int(sid), float(score)) for sid, score in zip(similar_ids, scores)
                   if int(sid) != art_idx and int(sid) in idx2art]
    top_pairs = valid_pairs[:req.n_results]

    raw_scores = [score for _, score in top_pairs]
    match_percentages = calculate_match_percentages(raw_scores)

    results = [
        enrich(idx2art[sid], {"match_percentage": pct})
        for (sid, score), pct in zip(top_pairs, match_percentages)
    ]

    return {"article_id": req.article_id, "similar_items": results}


# ── Frontend ──────────────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fashion Recommendation Engine</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0a0a0a; --surface: #141414; --border: #2a2a2a;
    --text: #e8e8e8; --text-muted: #888; --accent: #c8ff00; --coral: #ff6b5a;
    --font-display: 'Space Grotesk', sans-serif; --font-body: 'Inter', sans-serif;
  }
  body { font-family: var(--font-body); background: var(--bg); color: var(--text); line-height: 1.5; min-height: 100vh; }

  header { border-bottom: 1px solid var(--border); padding: 1.25rem 2rem; display: flex; align-items: center; justify-content: space-between; }
  .logo { font-family: var(--font-display); font-weight: 700; font-size: 1.25rem; letter-spacing: -0.02em; }
  .logo span { color: var(--accent); }
  .badge { font-size: 0.7rem; font-weight: 500; color: var(--accent); border: 1px solid rgba(200,255,0,0.3); padding: 0.2rem 0.6rem; border-radius: 100px; letter-spacing: 0.05em; text-transform: uppercase; }

  main { max-width: 1100px; margin: 0 auto; padding: 2.5rem 2rem; }
  .hero { margin-bottom: 2.5rem; }
  .hero h1 { font-family: var(--font-display); font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 0.75rem; }
  .hero p { color: var(--text-muted); max-width: 540px; font-size: 0.95rem; }

  .pipeline { display: flex; gap: 0.5rem; margin-top: 1.5rem; flex-wrap: wrap; }
  .pipeline-step { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; color: var(--text-muted); background: var(--surface); border: 1px solid var(--border); padding: 0.4rem 0.8rem; border-radius: 6px; }
  .pipeline-step .num { color: var(--accent); font-family: var(--font-display); font-weight: 700; font-size: 0.75rem; }
  .pipeline-arrow { color: var(--border); font-size: 1.2rem; align-self: center; }

  .tabs { display: flex; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
  .tab { padding: 0.75rem 1.5rem; font-size: 0.85rem; font-weight: 500; color: var(--text-muted); cursor: pointer; border: none; border-bottom: 2px solid transparent; background: none; font-family: var(--font-body); transition: all 0.15s; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .input-section { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .input-section input { flex: 1; min-width: 280px; padding: 0.75rem 1rem; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-family: var(--font-body); font-size: 0.9rem; outline: none; transition: border-color 0.15s; }
  .input-section input:focus { border-color: var(--accent); }
  .input-section input::placeholder { color: #555; }

  .btn { padding: 0.75rem 1.5rem; background: var(--accent); color: #000; border: none; border-radius: 8px; font-family: var(--font-display); font-weight: 700; font-size: 0.85rem; cursor: pointer; transition: opacity 0.15s; white-space: nowrap; }
  .btn:hover { opacity: 0.85; }

  .model-toggle { padding: 0.4rem 0.8rem; font-size: 0.75rem; font-weight: 500; color: var(--text-muted); background: none; border: 1px solid var(--border); border-radius: 6px; cursor: pointer; font-family: var(--font-body); transition: all 0.15s; }
  .model-toggle:hover { border-color: #555; color: var(--text); }
  .model-toggle.active { border-color: var(--accent); color: var(--accent); background: rgba(200,255,0,0.08); }

  .compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
  .compare-col h3 { font-family: var(--font-display); font-size: 0.9rem; margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }
  @media (max-width: 700px) { .compare-grid { grid-template-columns: 1fr; } }

  .results-meta { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 1rem; display: flex; gap: 1.5rem; }
  .results-meta span strong { color: var(--text); }

  .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1rem; }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; transition: border-color 0.15s, transform 0.15s; position: relative; }
  .card:hover { border-color: #444; transform: translateY(-2px); }

  .card-rank { position: absolute; top: 0.75rem; right: 0.75rem; background: var(--accent); color: #000; font-family: var(--font-display); font-weight: 700; font-size: 0.7rem; width: 1.6rem; height: 1.6rem; display: flex; align-items: center; justify-content: center; border-radius: 50%; z-index: 2; }

  .card-swatch { width: 100%; aspect-ratio: 4/3; display: flex; align-items: center; justify-content: center; position: relative; }
  .card-colour-tag { position: absolute; bottom: 0.5rem; left: 0.5rem; background: rgba(0,0,0,0.65); backdrop-filter: blur(4px); padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.65rem; color: #ddd; font-weight: 500; }

  .card-body { padding: 0.75rem 1rem 1rem; }
  .card-name { font-family: var(--font-display); font-weight: 500; font-size: 0.85rem; letter-spacing: -0.01em; margin-bottom: 0.35rem; line-height: 1.3; }
  .card-desc { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.5rem; line-height: 1.4; font-style: italic; background: rgba(255,255,255,0.03); padding: 0.4rem; border-radius: 4px; }
  .card-tags { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.5rem; }
  .card-tag { font-size: 0.65rem; color: var(--text-muted); background: rgba(255,255,255,0.05); padding: 0.15rem 0.4rem; border-radius: 3px; }
  .card-id { font-size: 0.7rem; color: #555; font-family: monospace; margin-bottom: 0.25rem; }
  .card-score { font-size: 0.75rem; color: var(--text-muted); }
  .card-score strong { color: var(--accent); }
  .card-action { margin-top: 0.5rem; font-size: 0.75rem; color: var(--coral); font-weight: 500; cursor: pointer; }
  .card-action:hover { text-decoration: underline; }

  .empty-state { text-align: center; padding: 4rem 2rem; color: var(--text-muted); }

  .loading { display: flex; align-items: center; gap: 0.75rem; padding: 2rem; color: var(--text-muted); font-size: 0.9rem; }
  .spinner { width: 18px; height: 18px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error { background: rgba(255,107,90,0.08); border: 1px solid rgba(255,107,90,0.2); color: var(--coral); padding: 0.75rem 1rem; border-radius: 8px; font-size: 0.85rem; margin-bottom: 1.5rem; }

  footer { border-top: 1px solid var(--border); padding: 1.5rem 2rem; text-align: center; color: #444; font-size: 0.75rem; margin-top: 4rem; }

  @media (max-width: 600px) { .hero h1 { font-size: 1.75rem; } main { padding: 1.5rem 1rem; } .results-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>

<header>
  <div class="logo">recsys<span>.</span>engine</div>
  <div class="badge">Two-Stage ML Pipeline</div>
</header>

<main>
  <div class="hero">
    <h1>Fashion Recommendation Engine</h1>
    <p>Personalised product discovery powered by collaborative filtering and learning-to-rank. Built on 31M transactions.</p>
    <div class="pipeline">
      <div class="pipeline-step"><span class="num">1</span> User interactions</div>
      <span class="pipeline-arrow">&rarr;</span>
      <div class="pipeline-step"><span class="num">2</span> ALS candidate retrieval</div>
      <span class="pipeline-arrow">&rarr;</span>
      <div class="pipeline-step"><span class="num">3</span> XGBoost re-ranking</div>
      <span class="pipeline-arrow">&rarr;</span>
      <div class="pipeline-step"><span class="num">4</span> Top-K results</div>
    </div>
  </div>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('recommend')">Personalised Picks</button>
    <button class="tab" onclick="switchTab('similar')">Similar Items</button>
  </div>

  <div id="tab-recommend">
    <div class="input-section">
      <input type="text" id="customer-id" placeholder="Customer ID (e.g. 0001b012...)">
      <input type="number" id="n-results" value="12" min="1" max="50" style="max-width:100px">
      <button class="btn" onclick="getRecommendations()">Get Picks</button>
    </div>
    <div style="display:flex;gap:0.5rem;margin-bottom:1.5rem;align-items:center;">
      <span style="font-size:0.8rem;color:var(--text-muted);">Model:</span>
      <button class="model-toggle active" id="btn-als" onclick="setMode('als_only')">ALS Only</button>
      <button class="model-toggle" id="btn-xgb" onclick="setMode('als_xgboost')">ALS + XGBoost</button>
      <button class="model-toggle" id="btn-compare" onclick="compareModels()" style="border-color:var(--coral);color:var(--coral);">Compare Both</button>
    </div>
  </div>

  <div id="tab-similar" style="display:none">
    <div class="input-section">
      <input type="text" id="article-id" placeholder="Article ID (e.g. 0739590032)">
      <input type="number" id="n-similar" value="12" min="1" max="50" style="max-width:100px">
      <button class="btn" onclick="getSimilarItems()">Find Similar</button>
    </div>
  </div>

  <div id="error-box" class="error" style="display:none"></div>
  <div id="results-info" class="results-meta" style="display:none"></div>
  <div id="results"></div>
</main>

<footer>Two-stage recommendation engine &middot; ALS (implicit) + XGBoost (LambdaMART) &middot; FastAPI</footer>

<script>
const COLOUR_HUE = {'black':0,'white':0,'grey':0,'beige':40,'brown':25,'red':0,'pink':340,'blue':220,'dark blue':230,'light blue':200,'green':140,'yellow':55,'orange':30,'purple':280,'turquoise':175,'gold':45,'cream':45};

function colourFromName(n) {
  if (!n) return 'hsl(0,0%,20%)';
  const l = n.toLowerCase();
  if (l.includes('white')||l.includes('cream')) return 'hsl(45,15%,85%)';
  if (l.includes('black')) return 'hsl(0,0%,12%)';
  if (l.includes('grey')||l.includes('gray')||l.includes('silver')) return 'hsl(0,0%,35%)';
  for (const [k,h] of Object.entries(COLOUR_HUE)) { if (l.includes(k)) return `hsl(${h},40%,30%)`; }
  let hash=0; for(let i=0;i<n.length;i++) hash=n.charCodeAt(i)+((hash<<5)-hash);
  return `hsl(${Math.abs(hash)%360},30%,25%)`;
}

let currentMode = 'als_only';

function setMode(mode) {
  currentMode = mode;
  document.getElementById('btn-als').classList.toggle('active', mode==='als_only');
  document.getElementById('btn-xgb').classList.toggle('active', mode==='als_xgboost');
  document.getElementById('btn-compare').classList.remove('active');
  document.getElementById('btn-compare').style.background='none';
}

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-recommend').style.display=tab==='recommend'?'block':'none';
  document.getElementById('tab-similar').style.display=tab==='similar'?'block':'none';
  document.querySelectorAll('.tab')[tab==='recommend'?0:1].classList.add('active');
  document.getElementById('results').innerHTML='';
  document.getElementById('results-info').style.display='none';
  document.getElementById('error-box').style.display='none';
}

function showError(m){const b=document.getElementById('error-box');b.textContent=m;b.style.display='block';}
function showLoading(){document.getElementById('error-box').style.display='none';document.getElementById('results-info').style.display='none';document.getElementById('results').innerHTML='<div class="loading"><div class="spinner"></div>Running inference pipeline...</div>';}

function renderCardsTo(container, items, scoreLabel, showAction) {
  if(!items.length){container.innerHTML='<p style="color:var(--text-muted)">No results found</p>';return;}
  const grid=document.createElement('div');grid.className='results-grid';
  items.forEach((item,i)=>{
    const score = item.match_percentage;
    const card=document.createElement('div');card.className='card';
    card.innerHTML=`
      <div class="card-rank">${i+1}</div>
      <div class="card-swatch" style="background:${colourFromName(item.color)}">
        ${item.color?`<span class="card-colour-tag">${item.color}</span>`:''}
      </div>
      <div class="card-body">
        <div class="card-name">${item.product_name||'Unknown product'}</div>
        ${item.description?`<div class="card-desc">${item.description}</div>`:''}
        <div class="card-tags">
          ${item.product_type?`<span class="card-tag">${item.product_type}</span>`:''}
          ${item.department?`<span class="card-tag">${item.department}</span>`:''}
          ${item.section?`<span class="card-tag">${item.section}</span>`:''}
        </div>
        <div class="card-id">${item.article_id}</div>
        <div class="card-score">${scoreLabel}: <strong>${score}%</strong></div>
        ${showAction?`<div class="card-action" onclick="event.stopPropagation();findSimilarFromCard('${item.article_id}')">Find similar &rarr;</div>`:''}
      </div>`;
    grid.appendChild(card);
  });
  container.appendChild(grid);
}

async function fetchRecommend(id, n, mode) {
  const r = await fetch('/recommend', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({customer_id:id, n_candidates:100, n_results:n, ranking_mode:mode})
  });
  if (!r.ok) { const e = await r.json(); throw new Error(e.detail||`HTTP ${r.status}`); }
  return r.json();
}

async function getRecommendations(){
  const id=document.getElementById('customer-id').value.trim();
  const n=parseInt(document.getElementById('n-results').value)||12;
  if(!id){showError('Enter a customer ID');return;}
  showLoading();
  try{
    const d = await fetchRecommend(id, n, currentMode);
    const info=document.getElementById('results-info');
    const modeLabel = d.ranking_mode==='als_only' ? 'ALS Only' : 'ALS + XGBoost';
    info.innerHTML=`<span>Candidates: <strong>${d.stage1_candidates}</strong></span><span>Returned: <strong>${d.recommendations.length}</strong></span><span>Mode: <strong>${modeLabel}</strong></span>`;
    info.style.display='flex';
    const container=document.getElementById('results');
    container.innerHTML='';
    renderCardsTo(container, d.recommendations, 'Match', true);
  }catch(e){showError(e.message);}
}

async function compareModels(){
  const id=document.getElementById('customer-id').value.trim();
  const n=parseInt(document.getElementById('n-results').value)||12;
  if(!id){showError('Enter a customer ID first');return;}
  document.getElementById('btn-als').classList.remove('active');
  document.getElementById('btn-xgb').classList.remove('active');
  document.getElementById('btn-compare').classList.add('active');
  document.getElementById('btn-compare').style.background='rgba(255,107,90,0.08)';
  showLoading();
  try{
    const [alsData, xgbData] = await Promise.all([
      fetchRecommend(id, n, 'als_only'),
      fetchRecommend(id, n, 'als_xgboost'),
    ]);
    const alsIds = new Set(alsData.recommendations.map(r=>r.article_id));
    const xgbIds = new Set(xgbData.recommendations.map(r=>r.article_id));
    const overlap = [...alsIds].filter(id=>xgbIds.has(id)).length;

    const info=document.getElementById('results-info');
    info.innerHTML=`<span>Comparing side-by-side</span><span>Overlap: <strong>${overlap}/${n}</strong> items</span><span><strong>${n-overlap}</strong> differ</span>`;
    info.style.display='flex';

    const container=document.getElementById('results');
    container.innerHTML=`
      <div class="compare-grid">
        <div class="compare-col">
          <h3 style="color:var(--accent)">ALS Only (Stage 1)</h3>
          <div id="compare-als"></div>
        </div>
        <div class="compare-col">
          <h3 style="color:var(--coral)">ALS + XGBoost (Stage 1+2)</h3>
          <div id="compare-xgb"></div>
        </div>
      </div>`;
    renderCardsTo(document.getElementById('compare-als'), alsData.recommendations, 'Match', true);
    renderCardsTo(document.getElementById('compare-xgb'), xgbData.recommendations, 'Match', true);
  }catch(e){showError(e.message);}
}

async function getSimilarItems(){
  const id=document.getElementById('article-id').value.trim();
  const n=parseInt(document.getElementById('n-similar').value)||12;
  if(!id){showError('Enter an article ID');return;}
  showLoading();
  try{
    const r=await fetch('/similar-items',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({article_id:id,n_results:n})});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||`HTTP ${r.status}`);}
    const d=await r.json();
    document.getElementById('results-info').style.display='none';
    const container=document.getElementById('results');
    container.innerHTML='';
    renderCardsTo(container, d.similar_items, 'Match', false);
  }catch(e){showError(e.message);}
}

function findSimilarFromCard(id){switchTab('similar');document.getElementById('article-id').value=id;getSimilarItems();}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)