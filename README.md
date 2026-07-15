# Fashion Recommendation Engine

A production-grade two-stage recommendation system built on **31M transactions** from the H&M Personalized Fashion Recommendations dataset. The system combines collaborative filtering (ALS) for candidate retrieval with a learning-to-rank model (XGBoost LambdaMART) for re-ranking, served through a FastAPI application with an interactive A/B comparison frontend.

Built as a portfolio project targeting ML Engineer roles in Search & Recommendations.

---

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────┐
│  Stage 1: Candidate Gen     │
│  ALS (Alternating Least     │
│  Squares) — implicit lib    │
│  128-dim embeddings         │
│  525K users × 42K items     │
│  Top 100 candidates         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 2: Re-ranking        │
│  XGBoost LambdaMART         │
│  30 features:               │
│  · ALS score                │
│  · Item popularity/recency  │
│  · User profile features    │
│  · Category affinity        │
│  · Price compatibility      │
│  · Article metadata         │
│  Top 12 returned            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  FastAPI + Frontend          │
│  A/B comparison:            │
│  ALS Only vs ALS+XGBoost   │
│  Similar Items endpoint     │
│  Match % scoring            │
└─────────────────────────────┘
```

## Key Results

End-to-end evaluation on temporal validation (last 7 days), ranking ALS top-100 candidates:

| Metric | ALS Only | ALS + XGBoost | Δ |
|---|---|---|---|
| Recall@12 | **0.2975** | 0.1772 | -0.1203 |
| Precision@12 | **0.0336** | 0.0200 | -0.0137 |
| MAP@12 | **0.1207** | 0.0512 | -0.0695 |
| NDCG@12 | **0.1678** | 0.0844 | -0.0833 |

**Why ALS outperforms the re-ranker (and why this matters):**

The XGBoost re-ranker learned to prioritize `department_match` (288.9 gain) and `item_popularity` (194.1) over the personalised ALS signal (166.0). This means it replaced collaborative filtering with a generic popularity-based recommender — which is strictly worse for personalised ranking.

This is a well-documented failure mode in two-stage systems: coarse re-ranking features can override the retrieval stage's personalised ordering. The fix requires finer-grained features that ALS can't capture — real-time session signals, sequential purchase patterns, or visual similarity embeddings.

The project includes an A/B comparison frontend that demonstrates this finding interactively.

## Features

**ML Pipeline**
- ALS collaborative filtering with implicit feedback (confidence weighting)
- XGBoost LambdaMART with `rank:ndcg` objective and monotone constraints
- Hard negative mining from ALS candidates (not random sampling)
- 30 engineered features: ALS score, item popularity/recency, user profile, price affinity, category affinity, department match
- Temporal train/val split (last 7 days held out)
- Realistic end-to-end evaluation matching the production inference pipeline
- Score blending analysis (alpha sweep)

**Experiment Tracking**
- MLflow integration with full parameter, metric, and artifact logging
- Feature importance tracking across experiment iterations
- Model versioning with `mlflow.xgboost.log_model`

**Serving**
- FastAPI with two endpoints: `/recommend` (personalised) and `/similar-items` (item-to-item)
- A/B model comparison: ALS Only vs ALS + XGBoost with side-by-side frontend
- Match percentage scoring for user-friendly presentation
- Pre-computed behavioral features loaded at startup
- Interactive frontend served at root `/`

## Project Structure

```
fashion-recommendation-engine/
├── 01_data_prep.py            # Data exploration, filtering, train/val split
├── 02_train_candidate_gen.py  # ALS model training + Recall@K evaluation
├── 03_train_ranker.py         # XGBoost ranker with hard negatives, MLflow
├── 04_api.py                  # FastAPI server + embedded frontend
├── pyproject.toml             # Project config (uv)
├── requirements.txt           # Python dependencies
└── processed/                 # Pre-computed artifacts for serving
    ├── art2idx.json           # Article ID → index mapping
    ├── idx2art.json           # Index → article ID mapping
    ├── cust2idx.json          # Customer ID → index mapping
    ├── article_metadata.json  # Product names, colours, descriptions
    └── article_features.parquet  # Article features for ranking
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://astral.sh/uv) package manager (recommended)
- H&M dataset from [Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

### Setup

```bash
git clone https://github.com/VatsalSangani/fashion_recommendation_engine.git
cd fashion_recommendation_engine

# Install dependencies
uv init --name hm-recsys
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv add fastapi "uvicorn[standard]" numpy scipy pandas implicit xgboost scikit-learn joblib pyarrow mlflow

# Download H&M data from Kaggle and place in data/
mkdir data models
# Place transactions_train.csv, articles.csv, customers.csv in data/
```

### Train

```bash
uv run python 01_data_prep.py            # ~2 min — prepares data
uv run python 02_train_candidate_gen.py   # ~1 min — trains ALS
uv run python 03_train_ranker.py          # ~5 min — trains XGBoost + evaluates
```

### Serve

```bash
uv run uvicorn 04_api:app --port 8000
# Open http://localhost:8000
```

### MLflow Dashboard

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

## API Endpoints

### POST /recommend
Personalised recommendations with model selection.

```json
{
  "customer_id": "0001b0127d3e5ff8...",
  "n_candidates": 100,
  "n_results": 12,
  "ranking_mode": "als_only"
}
```

`ranking_mode`: `"als_only"` (Stage 1 only) or `"als_xgboost"` (Stage 1 + 2)

### POST /similar-items
Item-to-item similarity (powers "People Also Viewed").

```json
{
  "article_id": "0739590032",
  "n_results": 12
}
```

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (Kaggle)

| | Count |
|---|---|
| Transactions | 31,788,324 |
| Customers | 1,371,980 |
| Articles | 105,542 |
| Date range | Sep 2018 – Sep 2020 |
| Training window | Last 90 days (3.9M txns) |
| Validation | Last 7 days (240K txns) |

## Tech Stack

| Component | Technology |
|---|---|
| Candidate generation | ALS via [implicit](https://github.com/benfred/implicit) |
| Re-ranking | XGBoost LambdaMART |
| Experiment tracking | MLflow |
| API | FastAPI |
| Package management | uv |
| Language | Python 3.12 |

## What I'd Do Next

1. **Sequential modeling** — Replace ALS with a transformer-based model (SASRec / BERT4Rec) to capture purchase sequences
2. **Real-time features** — Session-level signals (clicks, dwell time) that ALS can't capture, giving the ranker a reason to improve over retrieval
3. **Visual similarity** — CNN embeddings from product images as ranking features
4. **Feature store** — Move pre-computed user/item features to Redis for production-grade serving latency
5. **Online evaluation** — A/B testing framework with business metrics (CTR, conversion, revenue per session)

## License

This project is for educational and portfolio purposes. The H&M dataset is subject to [Kaggle's competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules).
