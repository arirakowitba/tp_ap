import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

print("Testing nested inner-OOF text specialist on Fold 1...")
t0 = time.time()

# 1. Load Datasets
raw_path = "data/ap_2026_q_2/raw/train_data.jsonlines"
rows = []
with open(raw_path, encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df_tr = pd.DataFrame(rows)
y_all = (df_tr["condition"] == "used").astype(int).values
n_train = len(y_all)

cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
trn_idx, val_idx = next(cv_outer.split(df_tr, y_all))
y_train = y_all[trn_idx]
y_val = y_all[val_idx]

tr_text_corpus = (df_tr["title"].fillna("") + " " + df_tr["tags"].astype(str)).tolist()
tr_text_f = [tr_text_corpus[i] for i in trn_idx]
val_text_f = [tr_text_corpus[i] for i in val_idx]

# Strict TF-IDF fit on outer-train only
print("TF-IDF 100k fitted strictly on outer-train (56k)...", flush=True)
vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
vec_c = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)

X_tr_w = vec_w.fit_transform(tr_text_f)
X_val_w = vec_w.transform(val_text_f)

X_tr_c = vec_c.fit_transform(tr_text_f)
X_val_c = vec_c.transform(val_text_f)

X_tr_tfidf = hstack([X_tr_w, X_tr_c]).tocsr()
X_val_tfidf = hstack([X_val_w, X_val_c]).tocsr()

svd = TruncatedSVD(n_components=64, random_state=42)
X_tr_svd = svd.fit_transform(X_tr_tfidf)
X_val_svd = svd.transform(X_val_tfidf)

# Inner-OOF for text specialist on 56k outer-train
print("Computing 5-fold inner-OOF text predictions on outer-train...", flush=True)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_text_inner_oof = np.zeros(len(trn_idx), dtype=np.float32)

for in_tr_idx, in_val_idx in cv_inner.split(X_tr_tfidf, y_train):
    lr_in = LogisticRegression(C=1.5, max_iter=250, solver="lbfgs", random_state=42)
    lr_in.fit(X_tr_tfidf[in_tr_idx], y_train[in_tr_idx])
    p_text_inner_oof[in_val_idx] = lr_in.predict_proba(X_tr_tfidf[in_val_idx])[:, 1]

# Fit text specialist on all 56k to predict outer-val
lr_outer = LogisticRegression(C=1.5, max_iter=250, solver="lbfgs", random_state=42)
lr_outer.fit(X_tr_tfidf, y_train)
p_text_val = lr_outer.predict_proba(X_val_tfidf)[:, 1].astype(np.float32)

print(f"Inner-OOF Text Acc on Outer-Train: {accuracy_score(y_train, (p_text_inner_oof >= 0.5).astype(int)):.5f}")
print(f"Text Acc on Outer-Val:             {accuracy_score(y_val, (p_text_val >= 0.5).astype(int)):.5f}")

# Tabular features
used_kws = ["antiguo", "antigua", "antiguos", "antiguas", "vintage", "retro", "impecable", "impecables", "poco uso", "como nuevo", "despiece", "usado", "usada"]
new_kws = ["nuevo", "nueva", "nuevos", "nuevas", "sellado", "en caja", "0km", "cero km", "garantia oficial", "de fabrica"]

def extract_text_indicators(df_sub):
    titles = df_sub["title"].fillna("").astype(str)
    t_lower = titles.str.lower()
    has_used_kw = t_lower.apply(lambda s: sum(1 for kw in used_kws if kw in s))
    has_new_kw = t_lower.apply(lambda s: sum(1 for kw in new_kws if kw in s))
    char_len = titles.apply(len)
    word_count = titles.apply(lambda s: len(s.split()))
    digit_count = titles.apply(lambda s: sum(c.isdigit() for c in s))
    upper_count = titles.apply(lambda s: sum(c.isupper() for c in s))
    return np.column_stack([has_used_kw, has_new_kw, char_len, word_count, digit_count, upper_count])

text_ind_trn = extract_text_indicators(df_tr.iloc[trn_idx])
text_ind_val = extract_text_indicators(df_tr.iloc[val_idx])

prices_trn = df_tr.iloc[trn_idx]["price"].fillna(0).values
prices_val = df_tr.iloc[val_idx]["price"].fillna(0).values

log_price_trn = np.log1p(np.maximum(0, prices_trn))
log_price_val = np.log1p(np.maximum(0, prices_val))

cats_trn = df_tr.iloc[trn_idx]["category_id"].astype(str).values
cats_val = df_tr.iloc[val_idx]["category_id"].astype(str).values

sellers_trn = df_tr.iloc[trn_idx]["seller_id"].astype(str).values
sellers_val = df_tr.iloc[val_idx]["seller_id"].astype(str).values

cat_med_price = pd.Series(prices_trn).groupby(cats_trn).median().to_dict()
glob_med_price = float(np.median(prices_trn))

price_ratio_trn = np.clip(prices_trn / np.array([cat_med_price.get(c, glob_med_price) + 1.0 for c in cats_trn]), 0, 50)
price_ratio_val = np.clip(prices_val / np.array([cat_med_price.get(c, glob_med_price) + 1.0 for c in cats_val]), 0, 50)

attrs_trn = df_tr.iloc[trn_idx]["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values
attrs_val = df_tr.iloc[val_idx]["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values

def parse_warr(w):
    if not w or pd.isna(w):
        return 0
    w_str = str(w).lower()
    return 1 if ("mes" in w_str or "año" in w_str or "garant" in w_str) else 0

warr_trn = df_tr.iloc[trn_idx]["warranty"].apply(parse_warr).values
warr_val = df_tr.iloc[val_idx]["warranty"].apply(parse_warr).values

init_qty_trn = np.log1p(df_tr.iloc[trn_idx]["initial_quantity"].fillna(1).values)
init_qty_val = np.log1p(df_tr.iloc[val_idx]["initial_quantity"].fillna(1).values)

# Inner-OOF Target Encoding for Category and Seller on 56k
global_prior = float(y_train.mean())
cat_te_inner_oof = np.zeros(len(trn_idx))
seller_te_inner_oof = np.zeros(len(trn_idx))

df_trn_target = pd.DataFrame({"cat": cats_trn, "seller": sellers_trn, "target": y_train})

for in_tr_idx, in_val_idx in cv_inner.split(df_trn_target, y_train):
    f_tr = df_trn_target.iloc[in_tr_idx]
    c_st = f_tr.groupby("cat")["target"].agg(["count", "sum"])
    c_sm = (c_st["sum"] + 10.0 * global_prior) / (c_st["count"] + 10.0)
    cat_te_inner_oof[in_val_idx] = df_trn_target.iloc[in_val_idx]["cat"].map(c_sm).fillna(global_prior).values

    s_st = f_tr.groupby("seller")["target"].agg(["count", "sum"])
    s_sm = (s_st["sum"] + 5.0 * global_prior) / (s_st["count"] + 5.0)
    seller_te_inner_oof[in_val_idx] = df_trn_target.iloc[in_val_idx]["seller"].map(s_sm).fillna(global_prior).values

# Full 56k stats for outer-val transform
full_c_st = df_trn_target.groupby("cat")["target"].agg(["count", "sum"])
full_c_sm = ((full_c_st["sum"] + 10.0 * global_prior) / (full_c_st["count"] + 10.0)).to_dict()
cat_te_val = np.array([full_c_sm.get(c, global_prior) for c in cats_val])

full_s_st = df_trn_target.groupby("seller")["target"].agg(["count", "sum"])
full_s_sm = ((full_s_st["sum"] + 5.0 * global_prior) / (full_s_st["count"] + 5.0)).to_dict()
seller_te_val = np.array([full_s_sm.get(s, full_c_sm.get(c, global_prior)) for s, c in zip(sellers_val, cats_val)])

X_tr_tab = np.column_stack([
    log_price_trn, price_ratio_trn, attrs_trn, warr_trn, init_qty_trn,
    cat_te_inner_oof, seller_te_inner_oof, text_ind_trn, p_text_inner_oof, X_tr_svd
])
X_val_tab = np.column_stack([
    log_price_val, price_ratio_val, attrs_val, warr_val, init_qty_val,
    cat_te_val, seller_te_val, text_ind_val, p_text_val, X_val_svd
])

# LightGBM Fold 1
print("Training LightGBM on strict inner-nested features...", flush=True)
clf = lgb.LGBMClassifier(n_estimators=900, learning_rate=0.025, num_leaves=55, subsample=0.85, colsample_bytree=0.8, min_child_samples=20, n_jobs=8, random_state=42, verbose=-1)
clf.fit(X_tr_tab, y_train)
val_preds = clf.predict_proba(X_val_tab)[:, 1]

acc = accuracy_score(y_val, (val_preds >= 0.5).astype(int))
hits = int(round(acc * len(val_idx)))
ll = log_loss(y_val, val_preds)

print(f"\nStrict Inner-Nested LightGBM Fold 1: Acc = {acc:.5f} ({hits} hits), LL = {ll:.5f}")

# Compare with Podium target
podium_oof = pd.read_csv("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")
p_target = podium_oof.iloc[val_idx]["oof_lgb"].values
acc_target = accuracy_score(y_val, (p_target >= 0.5).astype(int))
print(f"Podium Target OOF LightGBM Fold 1:    Acc = {acc_target:.5f} ({int(round(acc_target*len(val_idx)))} hits), LL = {log_loss(y_val, p_target):.5f}")
