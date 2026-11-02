import hashlib
import os
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
ZIP = Path(os.environ.get("AP_ZIP", str(ROOT / "data/ap-2026-q-2.zip")))
OUT = ROOT / "results/text_condition_20261101"
OUT.mkdir(exist_ok=True)


def clean(x):
    return str(x or "").lower() if x is not None else ""


def extract(d):
    shipping = d.get("shipping") or {}
    address = d.get("seller_address") or {}
    pics = d.get("pictures") or []
    attrs = d.get("attributes") or []
    names = []
    for a in attrs:
        if isinstance(a, dict):
            names.append(clean(a.get("name")) + " " + clean(a.get("value_name")))
    text = " ".join([clean(d.get("title")), clean(d.get("warranty")), " ".join(names)])
    title = clean(d.get("title"))
    warranty = clean(d.get("warranty"))
    date = clean(d.get("date_created"))[:10]
    seller_city = (address.get("city") or {}).get("id") if isinstance(address, dict) else ""
    seller_state = (address.get("state") or {}).get("id") if isinstance(address, dict) else ""
    f = {
        "seller_id": str(d.get("seller_id") or ""),
        "category_id": str(d.get("category_id") or ""),
        "listing_type_id": str(d.get("listing_type_id") or ""),
        "seller_city": str(seller_city or ""),
        "seller_state": str(seller_state or ""),
        "status": str(d.get("status") or ""),
        "shipping_mode": str(shipping.get("mode") or "") if isinstance(shipping, dict) else "",
        "price": float(d.get("price") or 0),
        "base_price": float(d.get("base_price") or 0),
        "initial_quantity": float(d.get("initial_quantity") or 0),
        "sold_quantity": float(d.get("sold_quantity") or 0),
        "available_quantity": float(d.get("available_quantity") or 0),
        "picture_count": len(pics),
        "attr_count": len(attrs),
        "title_len": len(clean(d.get("title"))),
        "warranty_len": len(clean(d.get("warranty"))),
        "date_month": int(date[5:7]) if len(date) >= 7 and date[5:7].isdigit() else 0,
        "free_shipping": int(bool(shipping.get("free_shipping"))) if isinstance(shipping, dict) else 0,
        "local_pick_up": int(bool(shipping.get("local_pick_up"))) if isinstance(shipping, dict) else 0,
        "accepts_mercadopago": int(bool(d.get("accepts_mercadopago"))),
        "official_store": int(d.get("official_store_id") is not None),
        "id": str(d.get("id")),
        "text": text,
    }
    f["log_price"] = np.log1p(max(f["price"], 0))
    f["sold_ratio"] = f["sold_quantity"] / (1 + f["initial_quantity"])
    f["avail_ratio"] = f["available_quantity"] / (1 + f["initial_quantity"])
    f["price_cents"] = round(f["price"] * 100) % 100
    f["price_round100"] = int(f["price"] % 100 == 0)
    f["title_words"] = len(title.split())
    f["title_digits"] = sum(ch.isdigit() for ch in title)
    f["title_upper_frac"] = sum(ch.isupper() for ch in str(d.get("title") or "")) / (1 + len(title))
    f["warranty_null"] = int(not warranty)
    for name, pattern in {
        "t_usado": r"usad", "t_nuevo": r"nuev", "t_antiguo": r"antig|vintage|retro|colecci",
        "t_original": r"original", "t_lote": r"lote", "t_repuesto": r"repuest",
        "w_sin": r"\bsin\b", "w_garantia": r"garant", "w_meses": r"mes",
        "w_usado": r"usad", "w_nuevo": r"nuev",
    }.items():
        f[name] = int(bool(re.search(pattern, warranty if name.startswith("w_") else title)))
    f["n_payments"] = len(d.get("non_mercado_pago_payment_methods") or [])
    f["n_tags"] = len(d.get("tags") or [])
    f["n_variations"] = len(d.get("variations") or [])
    for tag in ("dragged_bids_and_visits", "good_quality_thumbnail", "poor_quality_thumbnail", "free_relist"):
        f["tag_" + tag] = int(tag in (d.get("tags") or []))
    for pcode in ("MLATB", "MLAWC", "MLAMO", "MLAOT", "MLACD", "MLABC"):
        f["payment_" + pcode] = int(pcode in [x.get("id") for x in d.get("non_mercado_pago_payment_methods") or []])
    f["automatic_relist"] = int(bool(d.get("automatic_relist")))
    f["has_parent"] = int(d.get("parent_item_id") is not None)
    f["has_catalog"] = int(d.get("catalog_product_id") is not None)
    f["has_video"] = int(d.get("video_id") is not None)
    f["has_original_price"] = int(d.get("original_price") is not None)
    f["created_days"] = (pd.Timestamp(date) - pd.Timestamp("2013-01-01")).days if date else 0
    f["created_hour"] = int(clean(d.get("date_created"))[11:13] or 0)
    f["n_shipping_methods"] = len(shipping.get("methods") or []) if isinstance(shipping, dict) else 0
    f["n_shipping_tags"] = len(shipping.get("tags") or []) if isinstance(shipping, dict) else 0
    return f


def load():
    records = []
    labels = []
    with zipfile.ZipFile(ZIP) as z:
        for name in ("train_data.jsonlines", "test_data.jsonlines"):
            with z.open(name) as stream:
                for line in stream:
                    d = json.loads(line)
                    records.append(extract(d))
                    if name.startswith("train"):
                        labels.append(int(d["condition"] == "used"))
    df = pd.DataFrame(records)
    assert len(labels) == 70000 and len(df) == 100000
    assert df.id.is_unique
    return df, np.asarray(labels, dtype=np.int8)


def split(y, sellers, seed):
    # New sellers are held out in whole groups; seen sellers contribute disjoint rows.
    rng = np.random.default_rng(seed)
    uniq = np.unique(sellers)
    rng.shuffle(uniq)
    new_tune = set(uniq[: int(.12 * len(uniq))])
    new_confirm = set(uniq[int(.12 * len(uniq)):int(.24 * len(uniq))])
    seen = np.flatnonzero(~np.isin(sellers, list(new_tune | new_confirm)))
    rng.shuffle(seen)
    nt = int(.11 * len(sellers))
    tune = np.r_[np.flatnonzero(np.isin(sellers, list(new_tune))), seen[:nt]]
    confirm = np.r_[np.flatnonzero(np.isin(sellers, list(new_confirm))), seen[nt:2 * nt]]
    fit = np.setdiff1d(np.arange(len(sellers)), np.r_[tune, confirm])
    assert len(set(tune) & set(confirm)) == 0
    return fit, tune, confirm


def text_features(text, y, groups, fit, tune, confirm):
    oof = np.empty(len(fit), np.float32)
    folds = GroupKFold(n_splits=4)
    def vectorizer():
        return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3,
                               max_features=180000, sublinear_tf=True, dtype=np.float32)
    for tr, va in folds.split(fit, y[fit], groups[fit]):
        vec = vectorizer()
        xtr = vec.fit_transform(text[fit[tr]])
        m = LogisticRegression(C=2.0, max_iter=180, solver="liblinear")
        m.fit(xtr, y[fit[tr]])
        oof[va] = m.predict_proba(vec.transform(text[fit[va]]))[:, 1]
    vec = vectorizer()
    xfit = vec.fit_transform(text[fit])
    m = LogisticRegression(C=2.0, max_iter=180, solver="liblinear")
    m.fit(xfit, y[fit])
    return oof, m.predict_proba(vec.transform(text[tune]))[:, 1], m.predict_proba(vec.transform(text[confirm]))[:, 1]


def metrics(y, p, th):
    return {"accuracy": float(accuracy_score(y, p >= th)), "logloss": float(log_loss(y, p)),
            "threshold": float(th)}


def run(df, y, seed):
    n = len(y)
    sellers = df.seller_id.to_numpy()[:n]
    fit, tune, confirm = split(y, sellers, seed)
    cat = ["seller_id", "category_id", "listing_type_id", "seller_city",
           "seller_state", "status", "shipping_mode"]
    features = [c for c in df.columns if c not in ("id", "text")]
    a = df.iloc[:n][features].copy()
    # Unsupervised maps learned only from fit; never from tune/confirm rows.
    for col in ("seller_id", "category_id"):
        counts = df.iloc[fit][col].value_counts()
        a["freq_" + col] = a[col].map(counts).fillna(1).astype(float)
    med = df.iloc[fit].groupby("category_id").log_price.median()
    a["rel_price_category"] = a.log_price - a.category_id.map(med).fillna(df.iloc[fit].log_price.median())
    for c in cat:
        a[c] = a[c].fillna("").astype(str)
    oof, tu_text, co_text = text_features(df.text.to_numpy(), y, sellers, fit, tune, confirm)
    baseline = CatBoostClassifier(iterations=900, depth=6, learning_rate=.06, l2_leaf_reg=5,
                                  loss_function="Logloss", verbose=False, thread_count=7,
                                  random_seed=seed)
    baseline.fit(a.iloc[fit], y[fit], cat_features=cat, eval_set=(a.iloc[tune], y[tune]),
                 early_stopping_rounds=90, use_best_model=True)
    bp_tu = baseline.predict_proba(a.iloc[tune])[:, 1]
    bp_co = baseline.predict_proba(a.iloc[confirm])[:, 1]
    b = a.copy()
    b["text_oof_p"] = np.nan
    b.loc[fit, "text_oof_p"] = oof
    b.loc[tune, "text_oof_p"] = tu_text
    b.loc[confirm, "text_oof_p"] = co_text
    candidate = CatBoostClassifier(iterations=900, depth=6, learning_rate=.06, l2_leaf_reg=5,
                                   loss_function="Logloss", verbose=False, thread_count=7,
                                   random_seed=seed)
    candidate.fit(b.iloc[fit], y[fit], cat_features=cat, eval_set=(b.iloc[tune], y[tune]),
                  early_stopping_rounds=90, use_best_model=True)
    cp_tu = candidate.predict_proba(b.iloc[tune])[:, 1]
    cp_co = candidate.predict_proba(b.iloc[confirm])[:, 1]
    thresholds = np.arange(.44, .561, .005)
    th_b = max(thresholds, key=lambda t: accuracy_score(y[tune], bp_tu >= t))
    th_c = max(thresholds, key=lambda t: accuracy_score(y[tune], cp_tu >= t))
    report = {"seed":seed,"sizes":[len(fit),len(tune),len(confirm)],
              "ids_hashes":{name:hashlib.sha256(df.id.iloc[idx].str.cat(sep="|").encode()).hexdigest()
                            for name,idx in [("fit",fit),("tune",tune),("confirm",confirm)]},
              "iterations":[baseline.best_iteration_, candidate.best_iteration_],
              "tune_baseline":metrics(y[tune],bp_tu,th_b),
              "tune_candidate":metrics(y[tune],cp_tu,th_c),
              "confirm_baseline":metrics(y[confirm],bp_co,th_b),
              "confirm_candidate":metrics(y[confirm],cp_co,th_c)}
    seen=np.isin(sellers[confirm],sellers[fit])
    for name,mask in [("seen",seen),("new",~seen)]:
        report[name]={"n":int(mask.sum()),"base_acc":float(accuracy_score(y[confirm][mask],bp_co[mask]>=th_b)),
                      "cand_acc":float(accuracy_score(y[confirm][mask],cp_co[mask]>=th_c))}
    pd.DataFrame({"id":df.id.iloc[confirm].values,"seller_id":sellers[confirm],
                  "seen":seen.astype(int),"y":y[confirm],"p_base":bp_co,
                  "p_candidate":cp_co,"pred_base":(bp_co>=th_b).astype(int),
                  "pred_candidate":(cp_co>=th_c).astype(int)}).to_csv(OUT/f"paired_{seed}.csv",index=False)
    (OUT/f"report_{seed}.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    df,y=load()
    print("data loaded",len(df), "used", y.mean(), flush=True)
    run(df,y,20261103)
