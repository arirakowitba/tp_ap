"""Round 200 - fully nested, leak-free CV pipeline (AP 2026 Q2, new vs used).

Why: earlier OOF numbers (round126/127) had subtle optimism sources:
  * early stopping / best-iteration chosen on the same fold that is scored;
  * target encodings and text stacks for *training* rows built with folds that
    contain the validation rows (second-order leak);
  * stk_sellerother leak (documented in round127).
Here every supervised feature is rebuilt inside each outer fold using only the
outer-train labels (inner K-fold for the outer-train rows, full outer-train fit
for the outer-validation and test rows). Early stopping uses an inner split of
outer-train, never the outer-validation fold.

The same function produces the final test prediction (outer-train = all 70k).

usage: python src/round200_clean/clean_cv.py NAME [--algo cat|lgb] [--depth 6] [--lr 0.05]
       [--seed 2026] [--folds 5] [--threads 16] [--skip-cv]
Outputs: results/round200_clean/pred_NAME.npz (oof, te) + report_NAME.json
"""
import argparse, json, os, time, hashlib
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--algo", default="cat")
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--l2", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--cvseed", type=int, default=2026)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--inner", type=int, default=5)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--skip-cv", action="store_true")
ap.add_argument("--no-sellerother", action="store_true")
args = ap.parse_args()

D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2/raw").rstrip("/") + "/"
OUT = "results/round200_clean/"
CACHE = D + "r200_cache/"
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

P = pd.read_pickle(D + "feats.pkl")
F, y, is_test, CAT = P["F"].copy(), np.asarray(P["y"]), P["is_test"], list(P["CAT"])
ntr = int((is_test == 0).sum()); N = len(F)
seller = F.seller_id.astype(str).values

# ---------------- unsupervised seller aggregates (train+test, no labels) ----------------
g = F.groupby("seller_id", observed=True)
for c in ["warranty_null", "t_usado", "t_nuevo", "w_usado", "w_nuevo", "ship_free", "automatic_relist", "n_pics", "pic_mean_area",
          "sold_quantity", "dur_days", "accepts_mp", "n_pm", "title_upper_frac", "has_video", "price_round100", "log_price", "initial_quantity"]:
    F["sell_mean_" + c] = g[c].transform("mean")
for lt in ["free", "bronze", "silver", "gold_special", "gold", "gold_premium", "gold_pro"]:
    F["sell_frac_lt_" + lt] = (F.listing_type_id.astype(str) == lt).groupby(F.seller_id, observed=True).transform("mean")
for bm in ["auction", "classified"]:
    F["sell_frac_bm_" + bm] = (F.buying_mode.astype(str) == bm).groupby(F.seller_id, observed=True).transform("mean")
F["seller_logprice_dev"] = F.log_price - F.sell_mean_log_price

# ---------------- sparse design for the linear text/categorical model ----------------
sp_path = CACHE + "X_sparse.npz"
if os.path.exists(sp_path):
    Xs = sparse.load_npz(sp_path).tocsr()
else:
    t = pd.Series(P["text"]).fillna("").str.lower()
    w = pd.Series(P["warranty"]).fillna("").str.lower()
    a = pd.Series(P["attr"]).fillna("")
    Xt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True).fit_transform(t)
    Xc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300000, sublinear_tf=True).fit_transform(t)
    Xw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True).fit_transform(w)
    Xa = TfidfVectorizer(token_pattern=r"\S+", min_df=2).fit_transform(a)
    catdf = F[["listing_type_id", "buying_mode", "status", "category_id", "seller_id", "sub_status", "ship_mode", "ship_tags", "state", "city"]].astype(str)
    num = F[["log_price", "initial_quantity", "sold_quantity", "available_quantity", "n_pics", "dur_days", "n_pm", "title_len"]].copy()
    for c in num.columns:
        num[c] = pd.qcut(num[c].rank(method="first"), 20, labels=False, duplicates="drop").astype(str)
    bincols = [c for c in F.columns if c.startswith(("pm_", "tag_", "w_", "t_", "ship_local", "ship_free", "has_", "official", "accepts", "automatic", "price_"))]
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=1).fit_transform(pd.concat([catdf, num, F[bincols].astype(str)], axis=1))
    Xs = sparse.hstack([Xt, Xc, Xw, Xa, oh]).tocsr()
    sparse.save_npz(sp_path, Xs)
print("sparse", Xs.shape, flush=True)

LINEAR_C = 2.0

def fit_lin(idx):
    m = LogisticRegression(C=LINEAR_C, solver="liblinear", max_iter=2000)
    m.fit(Xs[idx], y[idx]); return m

def sellerother(pred, rows_mask=None):
    """mean of `pred` over the OTHER listings of the same seller (all rows train+test)."""
    df = pd.DataFrame({"s": seller, "v": pred})
    sm = df.groupby("s").v.transform("sum").values; sc = df.groupby("s").v.transform("count").values
    out = np.where(sc > 1, (sm - pred) / np.maximum(sc - 1, 1), np.nan)
    return out, sc - 1

def te_feats(tr_idx, ap_idx, inner_folds, keys_list, alpha=5.0):
    """Target encodings: inner-OOF for tr_idx rows, full tr_idx fit for ap_idx rows."""
    out = {}
    prior = y[tr_idx].mean()
    for name, keys in keys_list:
        v = np.full(N, np.nan); cnt = np.zeros(N)
        ktr = keys[tr_idx]
        for a, b in inner_folds:  # positions into tr_idx
            s = pd.DataFrame({"k": ktr[a], "y": y[tr_idx[a]]}).groupby("k").y.agg(["sum", "count"]).reindex(ktr[b])
            v[tr_idx[b]] = ((s["sum"].fillna(0) + alpha * prior) / (s["count"].fillna(0) + alpha)).values
            cnt[tr_idx[b]] = s["count"].fillna(0).values
        s = pd.DataFrame({"k": ktr, "y": y[tr_idx]}).groupby("k").y.agg(["sum", "count"]).reindex(keys[ap_idx])
        # scale counts so their distribution matches the inner-OOF one ((k-1)/k of data)
        v[ap_idx] = ((s["sum"].fillna(0) + alpha * prior) / (s["count"].fillna(0) + alpha)).values
        cnt[ap_idx] = s["count"].fillna(0).values * (len(inner_folds) - 1) / len(inner_folds)
        out["te_" + name] = v; out["tecnt_" + name] = cnt
    return out

KEYS = [("seller", seller), ("cat", F.category_id.astype(str).values), ("city", F.city.astype(str).values),
        ("seller_cat", seller + "|" + F.category_id.astype(str).values),
        ("seller_lt", seller + "|" + F.listing_type_id.astype(str).values),
        ("seller_bm", seller + "|" + F.buying_mode.astype(str).values)]

def supervised_block(tr_idx, ap_idx, seed):
    """Return DataFrame (N rows; only tr_idx & ap_idx filled) of supervised features."""
    inner = list(StratifiedKFold(args.inner, shuffle=True, random_state=seed).split(tr_idx, y[tr_idx]))
    S = te_feats(tr_idx, ap_idx, inner, KEYS)
    lin = np.full(N, np.nan); lin_so = np.full(N, np.nan); lin_so_n = np.zeros(N)
    for a, b in inner:
        m = fit_lin(tr_idx[a])
        pall = m.predict_proba(Xs)[:, 1]            # model never saw y of tr_idx[b]
        lin[tr_idx[b]] = pall[tr_idx[b]]
        so, n = sellerother(pall)
        lin_so[tr_idx[b]] = so[tr_idx[b]]; lin_so_n[tr_idx[b]] = n[tr_idx[b]]
    m = fit_lin(tr_idx)
    pall = m.predict_proba(Xs)[:, 1]
    lin[ap_idx] = pall[ap_idx]
    so, n = sellerother(pall)
    lin_so[ap_idx] = so[ap_idx]; lin_so_n[ap_idx] = n[ap_idx]
    S["lin"] = lin
    if not args.no_sellerother:
        S["lin_sellerother"] = lin_so; S["lin_sellerother_n"] = lin_so_n
    return pd.DataFrame(S)

def fit_predict_gbm(Xtr, ytr, Xap_list, seed):
    """Early stopping on an inner 10% split of the training rows only; then refit on all rows with best_it*1.1."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr)); nv = len(Xtr) // 10
    va, fi = perm[:nv], perm[nv:]
    catc = [c for c in Xtr.columns if c in CAT]
    if args.algo == "cat":
        from catboost import CatBoostClassifier, Pool
        def pool(X, yy=None):
            X = X.copy()
            for c in catc: X[c] = X[c].astype(str)
            return Pool(X, yy, cat_features=catc)
        kw = dict(learning_rate=args.lr, depth=args.depth, l2_leaf_reg=args.l2, random_seed=seed, thread_count=args.threads,
                  verbose=0, one_hot_max_size=8, allow_writing_files=False)
        m = CatBoostClassifier(iterations=8000, eval_metric="Logloss", od_type="Iter", od_wait=300, **kw)
        m.fit(pool(Xtr.iloc[fi], ytr[fi]), eval_set=pool(Xtr.iloc[va], ytr[va]), use_best_model=True)
        best = max(200, m.get_best_iteration())
        m2 = CatBoostClassifier(iterations=int(best * 1.1), **kw)
        m2.fit(pool(Xtr, ytr))
        return [m2.predict_proba(pool(X))[:, 1] for X in Xap_list], best
    else:
        import lightgbm as lgb
        Xtr = Xtr.drop(columns=["seller_id"]); Xap_list = [X.drop(columns=["seller_id"]) for X in Xap_list]
        params = dict(objective="binary", learning_rate=args.lr, num_leaves=2 ** args.depth - 1, min_child_samples=30, feature_fraction=0.5,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=args.l2, cat_smooth=30, cat_l2=10, max_cat_to_onehot=8,
                      verbose=-1, seed=seed, num_threads=args.threads)
        m = lgb.train(params, lgb.Dataset(Xtr.iloc[fi], ytr[fi]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        best = max(100, m.best_iteration)
        m2 = lgb.train(params, lgb.Dataset(Xtr, ytr), int(best * 1.1))
        return [m2.predict(X) for X in Xap_list], best

def run_fold(tr_idx, ap_idx, seed):
    S = supervised_block(tr_idx, ap_idx, seed)
    X = pd.concat([F.reset_index(drop=True), S], axis=1)
    preds, best = fit_predict_gbm(X.iloc[tr_idx], y[tr_idx], [X.iloc[ap_idx]], seed)
    return preds[0], best, S

t0 = time.time()
rep = {"name": args.name, "args": vars(args)}
te_idx = np.arange(ntr, N)
if not args.skip_cv:
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.cvseed).split(np.zeros(ntr), y))
    oof = np.zeros(ntr); oof_lin = np.zeros(ntr); its = []
    for k, (a, b) in enumerate(folds):
        p, best, S = run_fold(a, b, args.seed + k)
        oof[b] = p; oof_lin[b] = S["lin"].values[b]; its.append(best)
        print(f"{args.name} fold {k} acc {((p > .5) == y[b]).mean():.4f} lin {((oof_lin[b] > .5) == y[b]).mean():.4f} best {best} t {time.time()-t0:.0f}s", flush=True)
    acc = float(((oof > .5) == y).mean())
    ll = float(-np.mean(y * np.log(np.clip(oof, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - oof, 1e-6, 1))))
    rep.update(oof_acc=acc, oof_logloss=ll, lin_acc=float(((oof_lin > .5) == y).mean()), iters=its,
               fold_acc=[float(((oof[b] > .5) == y[b]).mean()) for _, b in folds])
    print(f"{args.name} OOF acc {acc:.5f} logloss {ll:.5f}", flush=True)
else:
    oof = None
# final fit on all 70k -> test
p_te, best, _ = run_fold(np.arange(ntr), te_idx, args.seed + 99)
rep["final_best_it"] = best; rep["elapsed_s"] = time.time() - t0
np.savez(OUT + f"pred_{args.name}.npz", oof=oof if oof is not None else np.array([]), te=p_te)
rep["te_sha256"] = hashlib.sha256(p_te.tobytes()).hexdigest()
json.dump(rep, open(OUT + f"report_{args.name}.json", "w"), indent=1)
print("done", json.dumps({k: rep[k] for k in rep if k != "args"}), flush=True)
