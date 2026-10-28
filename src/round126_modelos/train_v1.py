"""v1: OOF text logistic + OOF seller/category target encoding + LightGBM. 5-fold stratified random CV
(test is a random split of the same period: same dates, 72% seen sellers), so random CV mirrors test."""
import pandas as pd, numpy as np, scipy.sparse as sp, sys, json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

D = "/home/claude/data/"
P = pd.read_pickle(D + "feats.pkl")
F, y, is_test, CAT = P["F"], P["y"], P["is_test"], P["CAT"]
ntr = int((is_test == 0).sum())
SEED = 2026
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds = list(skf.split(np.zeros(ntr), y))

# ---------- OOF text model ----------
def text_oof(texts, name, **kw):
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", lowercase=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, strip_accents="unicode", lowercase=True, max_features=300000)
    Xw = vw.fit_transform(texts); Xc = vc.fit_transform(texts)
    X = sp.hstack([Xw, Xc]).tocsr()
    Xtr, Xte = X[:ntr], X[ntr:]
    oof = np.zeros(ntr); te = np.zeros(X.shape[0] - ntr)
    for k, (a, b) in enumerate(folds):
        m = LogisticRegression(C=kw.get("C", 4.0), max_iter=2000, solver="liblinear")
        m.fit(Xtr[a], y[a]); oof[b] = m.predict_proba(Xtr[b])[:, 1]; te += m.predict_proba(Xte)[:, 1] / 5
    print(name, "oof acc", ((oof > .5) == y).mean().round(4), flush=True)
    return np.r_[oof, te]

title = P["text"]
F["txt_title"] = text_oof(title, "title")
combo = np.array([f"{a} || {b} || {c}" for a, b, c in zip(title, P["warranty"], P["attr"])])
F["txt_combo"] = text_oof(combo, "title+warranty+attr")

# ---------- OOF target encoding (seller, category, city) ----------
def te_oof(col, alpha=5):
    keys = F[col].astype(str).values
    prior = y.mean(); out = np.zeros(len(F)); cnt = np.zeros(len(F))
    ktr = keys[:ntr]
    for a, b in folds:
        s = pd.DataFrame({"k": ktr[a], "y": y[a]}).groupby("k").y.agg(["sum", "count"])
        mp = s.reindex(ktr[b]); out[b] = ((mp["sum"].fillna(0) + alpha * prior) / (mp["count"].fillna(0) + alpha)).values
        cnt[b] = mp["count"].fillna(0).values
    s = pd.DataFrame({"k": ktr, "y": y}).groupby("k").y.agg(["sum", "count"])
    mp = s.reindex(keys[ntr:]); out[ntr:] = ((mp["sum"].fillna(0) + alpha * prior) / (mp["count"].fillna(0) + alpha)).values
    cnt[ntr:] = mp["count"].fillna(0).values * 0.8   # rescale count to fold size (4/5 of train)
    return out, cnt
for c in ["seller_id", "category_id", "city"]:
    F["te_" + c], F["tecnt_" + c] = te_oof(c)

feat_cols = [c for c in F.columns if c != "seller_id"]
X = F[feat_cols]
Xtr, Xte = X.iloc[:ntr], X.iloc[ntr:]
params = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_child_samples=20, feature_fraction=0.6, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, cat_smooth=20, max_cat_to_onehot=8, verbose=-1, seed=SEED)
oof = np.zeros(ntr); tep = np.zeros(len(Xte)); its = []
for k, (a, b) in enumerate(folds):
    dtr = lgb.Dataset(Xtr.iloc[a], y[a]); dva = lgb.Dataset(Xtr.iloc[b], y[b])
    m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[b] = m.predict(Xtr.iloc[b], num_iteration=m.best_iteration); its.append(m.best_iteration)
    tep += m.predict(Xte, num_iteration=m.best_iteration) / 5
    print(f"fold {k} acc {((oof[b] > .5) == y[b]).mean():.4f} it {m.best_iteration}", flush=True)
acc = ((oof > .5) == y).mean()
print("OOF acc", round(acc, 5))
imp = pd.Series(m.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
print(imp.head(25).round(0).to_dict())
np.savez(D + "pred_v1.npz", oof=oof, te=tep)
