"""Generic GBM trainer (LightGBM / CatBoost) on feats.pkl + OOF stacking inputs.
Same 5 stratified folds (seed 2026) for every model so OOF predictions are comparable/blendable.
usage: python train_gbm.py NAME lgb|cat [stack1,stack2,...] [extra1.pkl,extra2.pkl]"""
import pandas as pd, numpy as np, sys, json, time
from sklearn.model_selection import StratifiedKFold

import os
D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
name, algo = sys.argv[1], sys.argv[2]
stacks = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []
extras = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else []
P = pd.read_pickle(D + "feats.pkl")
F, y, is_test, CAT = P["F"].copy(), P["y"], P["is_test"], list(P["CAT"])
ntr = int((is_test == 0).sum())
folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(ntr), y))

# ---- seller-level unsupervised aggregates (train+test) ----
g = F.groupby("seller_id", observed=True)
for c in ["warranty_null", "t_usado", "t_nuevo", "w_usado", "w_nuevo", "ship_free", "automatic_relist", "n_pics", "pic_mean_area",
          "sold_quantity", "dur_days", "accepts_mp", "n_pm", "title_upper_frac", "has_video", "price_round100"]:
    F["sell_mean_" + c] = g[c].transform("mean")
for lt in ["free", "bronze", "silver", "gold_special", "gold", "gold_premium", "gold_pro"]:
    F["sell_frac_lt_" + lt] = (F.listing_type_id.astype(str) == lt).groupby(F.seller_id, observed=True).transform("mean")

# ---- OOF target encodings (fold-safe) ----
def te_oof(keys, alpha=5):
    keys = np.asarray(keys).astype(str); prior = y.mean(); out = np.zeros(len(F)); cnt = np.zeros(len(F)); ktr = keys[:ntr]
    for a, b in folds:
        s = pd.DataFrame({"k": ktr[a], "y": y[a]}).groupby("k").y.agg(["sum", "count"]).reindex(ktr[b])
        out[b] = ((s["sum"].fillna(0) + alpha * prior) / (s["count"].fillna(0) + alpha)).values; cnt[b] = s["count"].fillna(0).values
    s = pd.DataFrame({"k": ktr, "y": y}).groupby("k").y.agg(["sum", "count"]).reindex(keys[ntr:])
    out[ntr:] = ((s["sum"].fillna(0) + alpha * prior) / (s["count"].fillna(0) + alpha)).values; cnt[ntr:] = s["count"].fillna(0).values * 0.8
    return out, cnt
F["te_seller"], F["tecnt_seller"] = te_oof(F.seller_id)
F["te_cat"], F["tecnt_cat"] = te_oof(F.category_id)
F["te_city"], _ = te_oof(F.city)
F["te_seller_cat"], F["tecnt_seller_cat"] = te_oof(F.seller_id.astype(str) + "|" + F.category_id.astype(str))
F["te_seller_lt"], _ = te_oof(F.seller_id.astype(str) + "|" + F.listing_type_id.astype(str))

# ---- stacking inputs (OOF probs of other models) ----
for s in stacks:
    d = np.load(D + f"pred_{s}.npz"); v = np.r_[d["oof"], d["te"]]
    F["stk_" + s] = v
    # seller mean of the stacked prob over OTHER listings (train OOF + test preds)
    df = pd.DataFrame({"s": F.seller_id.astype(str).values, "v": v})
    sm = df.groupby("s").v.transform("sum"); sc = df.groupby("s").v.transform("count")
    F["stk_sellerother_" + s] = ((sm - v) / (sc - 1)).where(sc > 1, np.nan).values

for e in extras:
    E = pd.read_pickle(D + e)
    for c in E.columns: F[c] = E[c].values
CAT_TEXT = os.environ.get("CAT_TEXT", "0") == "1"
TEXTC = []
if CAT_TEXT and algo == "cat":
    F["title_raw"] = P["text"]; F["warranty_raw"] = P["warranty"]; TEXTC = ["title_raw", "warranty_raw"]
SEED = int(os.environ.get("SEED", "2026"))
F = F.copy()
drop = ["seller_id"] if algo == "lgb" else []
cols = [c for c in F.columns if c not in drop]
Xtr, Xte = F[cols].iloc[:ntr], F[cols].iloc[ntr:]
oof = np.zeros(ntr); tep = np.zeros(len(Xte)); t0 = time.time(); its = []
for k, (a, b) in enumerate(folds):
    if algo == "lgb":
        import lightgbm as lgb
        params = dict(objective="binary", learning_rate=0.02, num_leaves=63, min_child_samples=30, feature_fraction=0.5, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=3.0, cat_smooth=30, cat_l2=10, max_cat_to_onehot=8, verbose=-1, seed=SEED + k, num_threads=int(os.environ.get("THREADS", "2")))
        m = lgb.train(params, lgb.Dataset(Xtr.iloc[a], y[a]), 8000, valid_sets=[lgb.Dataset(Xtr.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[b] = m.predict(Xtr.iloc[b], num_iteration=m.best_iteration); tep += m.predict(Xte, num_iteration=m.best_iteration) / 5; its.append(m.best_iteration)
    else:
        from catboost import CatBoostClassifier, Pool
        catc = [c for c in cols if c in CAT]
        def pool(X, yy=None):
            X = X.copy()
            for c in catc: X[c] = X[c].astype(str)
            return Pool(X, yy, cat_features=catc, text_features=TEXTC or None)
        m = CatBoostClassifier(iterations=6000, learning_rate=float(os.environ.get("CAT_LR", "0.04")), depth=int(os.environ.get("CAT_DEPTH", "8")),
                               l2_leaf_reg=float(os.environ.get("CAT_L2", "5")), eval_metric=os.environ.get("CAT_OD_METRIC", "Accuracy"), od_type="Iter", od_wait=400,
                               random_seed=SEED + k, thread_count=int(os.environ.get("THREADS", "2")), verbose=0, one_hot_max_size=8, train_dir=D + "catboost_info")
        m.fit(pool(Xtr.iloc[a], y[a]), eval_set=pool(Xtr.iloc[b], y[b]), use_best_model=True)
        oof[b] = m.predict_proba(pool(Xtr.iloc[b]))[:, 1]; tep += m.predict_proba(pool(Xte))[:, 1] / 5; its.append(m.get_best_iteration())
    print(f"{name} fold {k} acc {((oof[b] > .5) == y[b]).mean():.4f} it {its[-1]} t {time.time()-t0:.0f}s", flush=True)
acc = ((oof > .5) == y).mean()
ll = -np.mean(y * np.log(np.clip(oof, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - oof, 1e-6, 1)))
print(f"{name} OOF acc {acc:.5f} logloss {ll:.5f}", flush=True)
np.savez(D + f"pred_{name}.npz", oof=oof, te=tep)
json.dump({"name": name, "algo": algo, "stacks": stacks, "extras": extras, "env": {k: os.environ.get(k) for k in ["CAT_TEXT", "SEED", "CAT_LR", "CAT_DEPTH", "CAT_L2", "CAT_OD_METRIC"]}, "oof_acc": acc, "logloss": ll, "iters": its, "n_features": len(cols)},
          open(D + f"report_{name}.json", "w"))
