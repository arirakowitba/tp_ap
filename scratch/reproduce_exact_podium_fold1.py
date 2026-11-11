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

print("Testing exact replication of Podium LightGBM on Fold 1...")
t0 = time.time()

# 1. Load Datasets
raw_path = "data/ap_2026_q_2/raw/train_data.jsonlines"
rows = []
with open(raw_path, encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df_tr = pd.DataFrame(rows)
y_train = (df_tr["condition"] == "used").astype(int).values
n_train = len(y_train)

# Exact text corpus: title + tags
used_kws = ["antiguo", "antigua", "antiguos", "antiguas", "vintage", "retro", "impecable", "impecables", "poco uso", "como nuevo", "despiece", "usado", "usada"]
new_kws = ["nuevo", "nueva", "nuevos", "nuevas", "sellado", "en caja", "0km", "cero km", "garantia oficial", "de fabrica"]

def extract_text_indicators(df):
    titles = df["title"].fillna("").astype(str)
    t_lower = titles.str.lower()
    has_used_kw = t_lower.apply(lambda s: sum(1 for kw in used_kws if kw in s))
    has_new_kw = t_lower.apply(lambda s: sum(1 for kw in new_kws if kw in s))
    char_len = titles.apply(len)
    word_count = titles.apply(lambda s: len(s.split()))
    digit_count = titles.apply(lambda s: sum(c.isdigit() for c in s))
    upper_count = titles.apply(lambda s: sum(c.isupper() for c in s))
    return np.column_stack([has_used_kw, has_new_kw, char_len, word_count, digit_count, upper_count])

text_ind_train = extract_text_indicators(df_tr)
tr_text_corpus = (df_tr["title"].fillna("") + " " + df_tr["tags"].astype(str)).tolist()

print("TF-IDF 100k...", flush=True)
vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
vec_c = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)

X_tr_w = vec_w.fit_transform(tr_text_corpus)
X_tr_c = vec_c.fit_transform(tr_text_corpus)
X_tr_tfidf = hstack([X_tr_w, X_tr_c]).tocsr()

svd = TruncatedSVD(n_components=64, random_state=42)
X_tr_svd = svd.fit_transform(X_tr_tfidf)

print("Text Logistic Specialist (in-sample fit as in Podium)...", flush=True)
lr_text = LogisticRegression(C=1.5, max_iter=300, solver="lbfgs", random_state=42)
lr_text.fit(X_tr_tfidf, y_train)
p_text_tr = lr_text.predict_proba(X_tr_tfidf)[:, 1]

# Tabular features
price_tr = df_tr["price"].fillna(0).values
log_price_tr = np.log1p(np.maximum(0, price_tr))

cat_price_median = df_tr.groupby("category_id")["price"].median()
tr_cat_med = df_tr["category_id"].map(cat_price_median).fillna(100.0).values
tr_price_ratio = np.clip(price_tr / (tr_cat_med + 1.0), 0, 50)

tr_num_attrs = df_tr["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values

def parse_warranty(w):
    if not w or pd.isna(w):
        return 0
    w_str = str(w).lower()
    if "mes" in w_str or "año" in w_str or "garant" in w_str:
        return 1
    return 0

tr_warr = df_tr["warranty"].apply(parse_warranty).values
tr_init_qty = np.log1p(df_tr["initial_quantity"].fillna(1).values)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_te_oof = np.zeros(n_train)
seller_te_oof = np.zeros(n_train)
global_prior = float(y_train.mean())

df_tr_target = df_tr.copy()
df_tr_target["target"] = y_train

for tr_idx_f, val_idx_f in cv.split(df_tr, y_train):
    fold_tr = df_tr_target.iloc[tr_idx_f]
    cat_stats = fold_tr.groupby("category_id")["target"].agg(["count", "sum"])
    smoothed = (cat_stats["sum"] + 10.0 * global_prior) / (cat_stats["count"] + 10.0)
    cat_te_oof[val_idx_f] = df_tr.iloc[val_idx_f]["category_id"].map(smoothed).fillna(global_prior).values

    seller_stats = fold_tr.groupby("seller_id")["target"].agg(["count", "sum"])
    s_smoothed = (seller_stats["sum"] + 5.0 * global_prior) / (seller_stats["count"] + 5.0)
    seller_te_oof[val_idx_f] = df_tr.iloc[val_idx_f]["seller_id"].map(s_smoothed).fillna(global_prior).values

X_tr_tab = np.column_stack([
    log_price_tr, tr_price_ratio, tr_num_attrs, tr_warr, tr_init_qty,
    cat_te_oof, seller_te_oof, text_ind_train, p_text_tr, X_tr_svd
])

trn_idx, val_idx = next(cv.split(X_tr_tab, y_train))
y_val = y_train[val_idx]

# LightGBM Fold 1
lgb_params = {
    "n_estimators": 900,
    "learning_rate": 0.025,
    "num_leaves": 55,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "min_child_samples": 20,
    "n_jobs": 8,
    "verbose": -1,
    "random_state": 42
}
clf = lgb.LGBMClassifier(**lgb_params)
clf.fit(X_tr_tab[trn_idx], y_train[trn_idx])
val_preds = clf.predict_proba(X_tr_tab[val_idx])[:, 1]

acc = accuracy_score(y_val, (val_preds >= 0.5).astype(int))
hits = int(round(acc * len(val_idx)))
ll = log_loss(y_val, val_preds)

print(f"Replicated LightGBM Fold 1: Acc = {acc:.5f} ({hits} hits), LL = {ll:.5f}")

# Target reference from oof file
podium_oof = pd.read_csv("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")
p_target = podium_oof.iloc[val_idx]["oof_lgb"].values
acc_target = accuracy_score(y_val, (p_target >= 0.5).astype(int))
print(f"Target OOF File LightGBM Fold 1: Acc = {acc_target:.5f} ({int(round(acc_target*len(val_idx)))} hits), LL = {log_loss(y_val, p_target):.5f}")
