"""
Lightweight API for Azure deployment.
Loads pre-computed feature files instead of raw transaction data.
RAM usage: ~500-800MB vs ~3GB for the full version.
"""

import json, pickle, numpy as np, pandas as pd, xgboost as xgb
from scipy.sparse import load_npz
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

DATA_DIR = Path("deploy_data")
app = FastAPI(title="Fashion Recommendation Engine", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def load_all():
    global als_model, user_item, item_factors, xgb_ranker
    global art2idx, idx2art, cust2idx, feature_names, feat_col_names
    global article_features_dict, article_metadata
    global item_feat, user_feat, cat_affinity, user_depts, art_idx_to_group, art_idx_to_dept

    with open(DATA_DIR/"als_model.pkl","rb") as f: als_model=pickle.load(f)
    user_item=load_npz(DATA_DIR/"user_item_matrix.npz")
    item_factors=np.load(DATA_DIR/"item_factors.npy")
    xgb_ranker=xgb.Booster(); xgb_ranker.load_model(str(DATA_DIR/"xgb_ranker.json"))
    with open(DATA_DIR/"ranker_feature_names.json") as f: feature_names=json.load(f)
    with open(DATA_DIR/"art2idx.json") as f: art2idx=json.load(f)
    with open(DATA_DIR/"idx2art.json") as f: idx2art={int(k):v for k,v in json.load(f).items()}
    with open(DATA_DIR/"cust2idx.json") as f: cust2idx=json.load(f)

    af=pd.read_parquet(DATA_DIR/"article_features.parquet")
    feat_col_names=[c for c in af.columns if c!="article_id"]
    article_features_dict={str(int(row["article_id"])).zfill(10):row[feat_col_names].values.astype(np.float32) for _,row in af.iterrows()}

    with open(DATA_DIR/"article_metadata.json") as f: article_metadata=json.load(f)
    with open(DATA_DIR/"item_features.json") as f: item_feat=json.load(f)
    with open(DATA_DIR/"user_features.json") as f: user_feat=json.load(f)
    with open(DATA_DIR/"cat_affinity.json") as f: cat_affinity=json.load(f)
    with open(DATA_DIR/"user_depts.json") as f: user_depts=json.load(f)
    with open(DATA_DIR/"art_idx_to_group.json") as f: art_idx_to_group=json.load(f)
    with open(DATA_DIR/"art_idx_to_dept.json") as f: art_idx_to_dept=json.load(f)
    print(f"Ready ({len(article_metadata):,} articles, {len(user_feat):,} users)")

def build_features(user_idx,art_idx,als_score):
    art_id=idx2art.get(int(art_idx))
    if not art_id or art_id not in article_features_dict: return None
    af=article_features_dict[art_id]; si=str(art_idx); su=str(user_idx)
    ifeat=item_feat.get(si,{"pop":0,"buyers":0,"recency":90,"price":0})
    ufeat=user_feat.get(su,{"total_log":0,"unique":0,"price":0,"recency":90})
    up,ip=ufeat["price"],ifeat["price"]
    fd={
        "als_score":als_score,
        **{feat_col_names[j]:float(af[j]) for j in range(len(af))},
        "item_popularity_log":ifeat["pop"],"item_unique_buyers_log":ifeat["buyers"],
        "item_recency_days":ifeat["recency"],"item_avg_price":ip,
        "user_total_purchases_log":ufeat["total_log"],"user_unique_items":ufeat["unique"],
        "user_avg_price":up,"user_recency_days":ufeat["recency"],
        "price_diff":abs(ip-up) if up>0 else 0,
        "price_ratio":min(ip/up if up>0 else 1.0,10.0),
        "category_affinity":cat_affinity.get(su,{}).get(art_idx_to_group.get(si,""),0.0),
        "department_match":1.0 if art_idx_to_dept.get(si,"0") in user_depts.get(su,[]) else 0.0,
    }
    return [fd[fn] for fn in feature_names]

class RecommendRequest(BaseModel):
    customer_id:str; n_candidates:int=100; n_results:int=12; ranking_mode:str="als_only"
class SimilarItemsRequest(BaseModel):
    article_id:str; n_results:int=12

def enrich(aid,sd):
    m=article_metadata.get(aid,{})
    return {"article_id":aid,**sd,"product_name":m.get("name",m.get("prod_name","Unknown")),"product_type":m.get("product_type_name",""),"color":m.get("color",m.get("colour_group_name","")),"department":m.get("department_name",""),"section":m.get("section_name",""),"description":m.get("description",m.get("detail_desc",""))}

def calc_pct(scores,lo=70,hi=99):
    if not scores: return []
    mx,mn=max(scores),min(scores)
    if mx==mn: return [95]*len(scores)
    return [int(lo+(hi-lo)*((s-mn)/(mx-mn))) for s in scores]

@app.get("/health")
def health(): return {"status":"healthy"}

@app.post("/recommend")
def recommend(req:RecommendRequest):
    if req.customer_id not in cust2idx: raise HTTPException(404,"Customer not found")
    uidx=cust2idx[req.customer_id]
    cids,ascores=als_model.recommend(uidx,user_item[uidx],N=req.n_candidates,filter_already_liked_items=True)
    vc,asl=[],[]
    for ai,asc in zip(cids,ascores):
        aid=idx2art.get(int(ai))
        if aid: vc.append((int(ai),aid)); asl.append(float(asc))
    if not vc: return {"customer_id":req.customer_id,"recommendations":[],"stage1_candidates":0,"ranking_mode":req.ranking_mode}

    if req.ranking_mode=="als_only":
        top=list(zip(vc[:req.n_results],asl[:req.n_results]))
        rs=[s for _,s in top]; mp=calc_pct(rs)
        results=[enrich(aid,{"match_percentage":p}) for ((_,aid),_),p in zip(top,mp)]
    else:
        rr,xc=[],[]
        for (ai,aid),asc in zip(vc,asl):
            row=build_features(uidx,ai,asc)
            if row: rr.append(row); xc.append((ai,aid))
        if not rr: return {"customer_id":req.customer_id,"recommendations":[],"stage1_candidates":0,"ranking_mode":req.ranking_mode}
        dm=xgb.DMatrix(np.array(rr),feature_names=feature_names)
        xs=xgb_ranker.predict(dm)
        ranked=sorted(zip(xc,xs),key=lambda x:-x[1])[:req.n_results]
        rs=[float(s) for _,s in ranked]; mp=calc_pct(rs)
        results=[enrich(aid,{"match_percentage":p}) for ((_,aid),_),p in zip(ranked,mp)]
    return {"customer_id":req.customer_id,"recommendations":results,"stage1_candidates":len(vc),"ranking_mode":req.ranking_mode}

@app.post("/similar-items")
def similar_items(req:SimilarItemsRequest):
    if req.article_id not in art2idx: raise HTTPException(404,"Article not found")
    aidx=art2idx[req.article_id]
    sids,scores=als_model.similar_items(aidx,N=req.n_results+1)
    vp=[(int(s),float(sc)) for s,sc in zip(sids,scores) if int(s)!=aidx and int(s) in idx2art][:req.n_results]
    rs=[sc for _,sc in vp]; mp=calc_pct(rs)
    results=[enrich(idx2art[s],{"match_percentage":p}) for (s,_),p in zip(vp,mp)]
    return {"article_id":req.article_id,"similar_items":results}

@app.get("/",response_class=HTMLResponse)
def frontend():
    p=Path(__file__).parent/"index.html"
    return p.read_text() if p.exists() else "<h1>Fashion Rec Engine</h1><p>Use /docs</p>"

if __name__=="__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)