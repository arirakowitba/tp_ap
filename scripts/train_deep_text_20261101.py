"""Paired strong-feature CatBoost ablation of fold-safe char text probability."""
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import zipfile
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features
from scripts.train_text_condition_20261101 import ZIP, OUT, split, text_features

SEED = int(sys.argv[1]) if len(sys.argv)>1 else 20261104
DROP_AGG = ("freq_", "seller_", "sell_mean_", "rel_price_cat")

def split_testlike(sellers, seed):
    rng=np.random.default_rng(seed)
    unique=np.unique(sellers)
    rng.shuffle(unique)
    k=int(.0175*len(unique))
    tune_new=set(unique[:k])
    confirm_new=set(unique[k:2*k])
    seen_rows=np.flatnonzero(~np.isin(sellers,list(tune_new | confirm_new)))
    rng.shuffle(seen_rows)
    n_seen=int(.14*len(sellers))
    tune=np.r_[np.flatnonzero(np.isin(sellers,list(tune_new))),seen_rows[:n_seen]]
    confirm=np.r_[np.flatnonzero(np.isin(sellers,list(confirm_new))),seen_rows[n_seen:2*n_seen]]
    fit=np.setdiff1d(np.arange(len(sellers)),np.r_[tune,confirm])
    return fit,tune,confirm



def read_raw():
    with zipfile.ZipFile(ZIP) as z:
        return [json.loads(line) for line in z.open("train_data.jsonlines")]


def add_fit_aggregates(a, cat, fit):
    a = a.copy()
    for c in ("seller_id", "category_id", "city"):
        cnt = cat.iloc[fit][c].value_counts()
        a["fit_freq_" + c] = cat[c].map(cnt).fillna(1).astype(np.float32).values
    med = a.iloc[fit].groupby(cat.iloc[fit].category_id).log_price.median()
    a["fit_rel_price_cat"] = a.log_price - cat.category_id.map(med).fillna(a.iloc[fit].log_price.median())
    return a


def fit_model(a, y, fit, tune, cats, seed):
    model = CatBoostClassifier(iterations=1250, depth=6, learning_rate=.05, l2_leaf_reg=5,
                               loss_function="Logloss", thread_count=7, random_seed=seed, verbose=False)
    model.fit(a.iloc[fit], y[fit], cat_features=cats,
              eval_set=(a.iloc[tune], y[tune]), early_stopping_rounds=100, use_best_model=True)
    return model


def main():
    rows = read_raw()
    y = np.array([int(r["condition"] == "used") for r in rows])
    seller = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])
    fit, tune, confirm = split_testlike(seller, SEED) if len(sys.argv)>2 and sys.argv[2]=="testlike" else split(y, seller, SEED)
    features, cat, titles, warranties, combos = extract_deep_features(rows)
    keep = [c for c in features if not c.startswith(DROP_AGG)]
    a = add_fit_aggregates(features[keep], cat, fit)
    cats = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]
    a["seller_id_cat"] = cat.seller_id
    cats += ["seller_id_cat"]
    text = np.array([str(r.get("title") or "") + " " + str(r.get("warranty") or "") +
                     " " + " ".join(str(v.get("value_name") or "") for v in r.get("attributes") or [])
                     for r in rows], dtype=object)
    oof, tu_text, co_text = text_features(text, y, seller, fit, tune, confirm)
    base = fit_model(a,y,fit,tune,cats,SEED)
    bp_tu=base.predict_proba(a.iloc[tune])[:,1]
    bp_co=base.predict_proba(a.iloc[confirm])[:,1]
    b=a.copy()
    b["text_oof_p"]=np.nan
    b.loc[fit,"text_oof_p"]=oof
    b.loc[tune,"text_oof_p"]=tu_text
    b.loc[confirm,"text_oof_p"]=co_text
    cand=fit_model(b,y,fit,tune,cats,SEED)
    cp_tu=cand.predict_proba(b.iloc[tune])[:,1]
    cp_co=cand.predict_proba(b.iloc[confirm])[:,1]
    thresholds=np.arange(.44,.561,.005)
    th_b=max(thresholds,key=lambda t:accuracy_score(y[tune],bp_tu>=t))
    th_c=max(thresholds,key=lambda t:accuracy_score(y[tune],cp_tu>=t))
    pred_b=bp_co>=th_b
    pred_c=cp_co>=th_c
    seen=np.isin(seller[confirm],seller[fit])
    paired=pd.DataFrame({"item_id":ids[confirm],"seller_id":seller[confirm],
                         "is_seen":seen.astype(int),"y_true":y[confirm],"p_base":bp_co,
                         "p_candidate":cp_co,"pred_base":pred_b.astype(int),
                         "pred_candidate":pred_c.astype(int)})
    out=OUT/f"paired_deep_{SEED}.csv"
    paired.to_csv(out,index=False)
    delta=(pred_c==y[confirm]).astype(int)-(pred_b==y[confirm]).astype(int)
    by=paired.assign(delta=delta).groupby("seller_id").delta.sum().to_numpy()
    rng=np.random.default_rng(SEED)
    boot=by[rng.integers(0,len(by),(2000,len(by)))].sum(axis=1)
    report={
        "seed":SEED,"fit_tune_confirm":[len(fit),len(tune),len(confirm)],
        "hash_ids":{name:hashlib.sha256(ids[idx].astype("U").tobytes()).hexdigest()
                    for name,idx in (("fit",fit),("tune",tune),("confirm",confirm))},
        "iterations":[base.best_iteration_,cand.best_iteration_],
        "thresholds":[th_b,th_c],
        "tune_acc":[accuracy_score(y[tune],bp_tu>=th_b),accuracy_score(y[tune],cp_tu>=th_c)],
        "confirm_acc":[accuracy_score(y[confirm],pred_b),accuracy_score(y[confirm],pred_c)],
        "confirm_logloss":[log_loss(y[confirm],bp_co),log_loss(y[confirm],cp_co)],
        "net":int(delta.sum()),"seller_ci95":np.quantile(boot,[.025,.975]).tolist(),
        "seen_new":{str(k):{"n":int(sum(seen==k)),"net":int(delta[seen==k].sum())} for k in (0,1)},
        "paired_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (OUT/f"report_deep_{SEED}.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
