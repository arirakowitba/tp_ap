"""True outer seller-group nested validation of the Podium Engine recipe vs Dual 50/50 vs logistic stack.

Design (frozen before execution, see manifest):
- Outer: StratifiedGroupKFold(5, shuffle=True, random_state=42), groups = seller_id. Sellers disjoint (asserted).
- For every outer fold k, ALL components (TF-IDF, SVD, text LR, category price medians, category/seller TE,
  LightGBM x3 seeds, CatBoost) are refit exclusively on outer-train.
- Inner: StratifiedGroupKFold(4, shuffle=True, random_state=1000+k) on outer-train, groups = seller_id.
  The full pipeline is refit on each inner-train and predicts its inner-val -> authentic inner-OOF.
- Policies fixed in advance; their free parameters are fit only on inner-OOF, written to disk
  (freeze file), and only then are outer-val labels used for metrics:
    A  Podium recipe: convex blend of LGB/CB/Text, weights by L-BFGS-B logloss on inner-OOF,
       same bounds/initial point as train_ultimate_podium_engine_23h.py.
    B  Fixed 0.5*LGB + 0.5*CB (no free parameters).
    C  LogisticRegression(C=1.0) on logits of [LGB, CB, Text] fit on inner-OOF.
  Threshold fixed at 0.5 for all policies.
- Deviations from the original Podium script (declared): CatBoost uses a fixed iteration count with no
  early stopping (original early-stopped on the evaluation fold, which is leakage); thread counts.
  Text specialist in-sample prediction on the training set is kept as a feature (original recipe; it uses
  only training labels of the current fit set).
- No reuse of results/ultimate_podium_engine_23h OOF files anywhere.
Status of every output: DEVELOPMENT_ONLY. No submission is generated.
"""
import os, sys, json, time, hashlib, warnings, subprocess, platform
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import sklearn, lightgbm, catboost
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / os.environ.get("SG_OUT", "seller_group_nested_podium_20261111")
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

N_THREADS = int(os.environ.get("SG_THREADS", os.cpu_count() or 8))
CB_ITERS = int(os.environ.get("SG_CB_ITERS", 1400))
LGB_TREES = int(os.environ.get("SG_LGB_TREES", 900))
MAX_FOLDS = int(os.environ.get("SG_MAX_FOLDS", 5))
CONFIG = {
    "outer": {"cv": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True, "random_state": 42, "groups": "seller_id"},
    "inner": {"cv": "StratifiedGroupKFold", "n_splits": 4, "shuffle": True, "random_state": "1000+outer_fold", "groups": "seller_id"},
    "tfidf_word": {"ngram_range": [1, 2], "max_features": 40000, "min_df": 3, "sublinear_tf": True},
    "tfidf_char": {"ngram_range": [3, 5], "analyzer": "char_wb", "max_features": 60000, "min_df": 5, "sublinear_tf": True},
    "svd": {"n_components": 64, "random_state": 42},
    "text_lr": {"C": 1.5, "max_iter": 800, "solver": "saga", "random_state": 42},
    "te": {"cv": "StratifiedKFold(5, shuffle, 42) within fit set", "m_cat": 10.0, "m_seller": 5.0},
    "lgb": {"n_estimators": LGB_TREES, "learning_rate": 0.025, "num_leaves": 55, "subsample": 0.85,
            "colsample_bytree": 0.80, "min_child_samples": 20, "seeds": [42, 2026, 777], "early_stopping": None},
    "cb": {"iterations": CB_ITERS, "depth": 8, "learning_rate": 0.035, "loss_function": "Logloss",
           "text_features": ["title"], "cat_features": ["seller_id", "category_id"], "random_seed": "42+outer_fold",
           "early_stopping": None},
    "policy_A": {"type": "convex LGB/CB/Text", "fit_on": "inner-OOF", "x0": [0.45, 0.40, 0.15],
                 "bounds": [[0.05, 0.80], [0.05, 0.80], [0.05, 0.50]], "objective": "logloss"},
    "policy_B": {"type": "fixed 0.5*LGB+0.5*CB", "free_params": 0},
    "policy_C": {"type": "LogisticRegression(C=1.0) on logit[LGB,CB,Text]", "fit_on": "inner-OOF"},
    "threshold": 0.5,
    "bootstrap": {"type": "two-way crossed cluster bootstrap seller x category (Poisson(1) product weights)",
                  "n_boot": 2000, "seed": 20261111},
    "n_threads": N_THREADS,
}
USED_KWS = ["antiguo", "antigua", "antiguos", "antiguas", "vintage", "retro", "impecable", "impecables", "poco uso",
            "como nuevo", "despiece", "usado", "usada"]
NEW_KWS = ["nuevo", "nueva", "nuevos", "nuevas", "sellado", "en caja", "0km", "cero km", "garantia oficial", "de fabrica"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load():
    rows = [json.loads(l) for l in open(ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8")]
    df = pd.DataFrame(rows)
    y = (df["condition"] == "used").astype(int).values
    titles = df["title"].fillna("").astype(str)
    tl = titles.str.lower()
    ind = np.column_stack([
        tl.apply(lambda s: sum(kw in s for kw in USED_KWS)), tl.apply(lambda s: sum(kw in s for kw in NEW_KWS)),
        titles.str.len(), titles.apply(lambda s: len(s.split())),
        titles.apply(lambda s: sum(c.isdigit() for c in s)), titles.apply(lambda s: sum(c.isupper() for c in s))]).astype(float)

    def warr(w):
        if not w or pd.isna(w):
            return 0
        s = str(w).lower()
        return int("mes" in s or "año" in s or "garant" in s)

    base = pd.DataFrame({
        "price": df["price"].fillna(0).values,
        "num_attrs": df["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values,
        "warranty": df["warranty"].apply(warr).values,
        "init_qty": np.log1p(df["initial_quantity"].fillna(1).values),
    })
    corpus = (df["title"].fillna("") + " " + df["tags"].astype(str)).tolist()
    meta = pd.DataFrame({"id": df["id"].values, "seller_id": df["seller_id"].astype(str).values,
                         "category_id": df["category_id"].astype(str).values, "title": titles.values})
    return y, ind, base, corpus, meta


def te_maps(cats, y, m, prior):
    s = pd.DataFrame({"k": cats, "y": y}).groupby("k")["y"].agg(["count", "sum"])
    return (s["sum"] + m * prior) / (s["count"] + m)


def fit_predict(T, P, y, ind, base, corpus, meta, seed_cb):
    """Refit every component on rows T only; return predictions for P and diagnostics."""
    t0 = time.time()
    yT = y[T]
    prior = float(yT.mean())
    cT = [corpus[i] for i in T]; cP = [corpus[i] for i in P]
    vw = TfidfVectorizer(**{**CONFIG["tfidf_word"], "ngram_range": (1, 2)})
    vc = TfidfVectorizer(**{**CONFIG["tfidf_char"], "ngram_range": (3, 5)})
    XT = hstack([vw.fit_transform(cT), vc.fit_transform(cT)]).tocsr()
    XP = hstack([vw.transform(cP), vc.transform(cP)]).tocsr()
    svd = TruncatedSVD(**CONFIG["svd"])
    sT = svd.fit_transform(XT); sP = svd.transform(XP)
    lr = LogisticRegression(**CONFIG["text_lr"])
    lr.fit(XT, yT)
    ptxT = lr.predict_proba(XT)[:, 1]  # in-sample on fit set (Podium recipe)
    ptxP = lr.predict_proba(XP)[:, 1]

    catT = meta["category_id"].values[T]; catP = meta["category_id"].values[P]
    selT = meta["seller_id"].values[T]; selP = meta["seller_id"].values[P]
    price = base["price"].values
    med = pd.Series(price[T]).groupby(catT).median()
    medT = pd.Series(catT).map(med).fillna(100.0).values
    medP = pd.Series(catP).map(med).fillna(100.0).values
    prT = np.clip(price[T] / (medT + 1.0), 0, 50); prP = np.clip(price[P] / (medP + 1.0), 0, 50)
    lpT = np.log1p(np.maximum(0, price[T])); lpP = np.log1p(np.maximum(0, price[P]))

    cteT = np.zeros(len(T)); steT = np.zeros(len(T))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(T, yT):
        cm = te_maps(catT[a], yT[a], 10.0, prior); sm = te_maps(selT[a], yT[a], 5.0, prior)
        cteT[b] = pd.Series(catT[b]).map(cm).fillna(prior).values
        steT[b] = pd.Series(selT[b]).map(sm).fillna(prior).values
    cteP = pd.Series(catP).map(te_maps(catT, yT, 10.0, prior)).fillna(prior).values
    steP = pd.Series(selP).map(te_maps(selT, yT, 5.0, prior)).fillna(prior).values

    na, nw, ni = base["num_attrs"].values, base["warranty"].values, base["init_qty"].values
    tabT = np.column_stack([lpT, prT, na[T], nw[T], ni[T], cteT, steT, ind[T], ptxT, sT])
    tabP = np.column_stack([lpP, prP, na[P], nw[P], ni[P], cteP, steP, ind[P], ptxP, sP])

    plgb = np.zeros(len(P))
    for sd in CONFIG["lgb"]["seeds"]:
        prm = {k: v for k, v in CONFIG["lgb"].items() if k not in ("seeds", "early_stopping")}
        m = LGBMClassifier(**prm, random_state=sd, n_jobs=N_THREADS, verbose=-1)
        m.fit(tabT, yT)
        plgb += m.predict_proba(tabP)[:, 1] / len(CONFIG["lgb"]["seeds"])

    def cbdf(idx, lp, pr, ptx, cte):
        return pd.DataFrame({"title": meta["title"].values[idx], "price": lp, "price_ratio": pr,
                             "num_attrs": na[idx], "warranty": nw[idx], "p_text": ptx, "cat_te": cte,
                             "seller_id": meta["seller_id"].values[idx], "category_id": meta["category_id"].values[idx]})
    cb = CatBoostClassifier(iterations=CB_ITERS, depth=8, learning_rate=0.035, loss_function="Logloss",
                            text_features=["title"], cat_features=["seller_id", "category_id"],
                            random_seed=seed_cb, thread_count=N_THREADS, verbose=0)
    cb.fit(cbdf(T, lpT, prT, ptxT, cteT), yT)
    pcb = cb.predict_proba(cbdf(P, lpP, prP, ptxP, cteP))[:, 1]
    diag = {"n_fit": int(len(T)), "n_pred": int(len(P)), "n_fit_sellers": int(len(set(selT))),
            "vocab_word": int(len(vw.vocabulary_)), "vocab_char": int(len(vc.vocabulary_)),
            "svd_var": float(svd.explained_variance_ratio_.sum()), "text_lr_n_iter": int(lr.n_iter_[0]),
            "text_in_sample_acc_fit_set": float(accuracy_score(yT, ptxT >= 0.5)),
            "cb_iterations_used": int(cb.tree_count_), "seconds": round(time.time() - t0, 1)}
    return plgb, pcb, ptxP, diag


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cached(name, fn):
    f = CACHE / f"{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        log(f"  cache hit {name}")
        return z["lgb"], z["cb"], z["txt"], json.loads(str(z["diag"]))
    a, b, c, d = fn()
    np.savez(f, lgb=a, cb=b, txt=c, diag=json.dumps(d))
    log(f"  saved {name} ({d['seconds']}s)")
    return a, b, c, d


def boot_two_way(diff, sellers, cats, n_boot, seed):
    rng = np.random.default_rng(seed)
    su, si = np.unique(sellers, return_inverse=True)
    cu, ci = np.unique(cats, return_inverse=True)
    out = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.poisson(1.0, len(su))[si] * rng.poisson(1.0, len(cu))[ci]
        out[b] = (w * diff).sum() / w.sum()
    return out


def main():
    t_start = time.time()
    y, ind, base, corpus, meta = load()
    n = len(y)
    groups = meta["seller_id"].values
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y, groups))
    fold_of = np.empty(n, int)
    for k, (tr, va) in enumerate(outer):
        assert not (set(groups[tr]) & set(groups[va])), "seller overlap in outer fold"
        fold_of[va] = k
    log(f"outer folds sizes {[len(v) for _, v in outer]}, sellers disjoint asserted")

    rec = {p: np.zeros(n) for p in ["lgb", "cb", "txt", "A", "B", "C"]}
    fold_rows, provenance, frozen_all = [], [], []
    for k, (tr, va) in enumerate(outer[:MAX_FOLDS]):
        log(f"=== outer fold {k+1}/5: train {len(tr)} val {len(va)}")
        # ---- inner seller-group OOF on outer-train only
        inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + k).split(np.zeros(len(tr)), y[tr], groups[tr]))
        io = {m: np.zeros(len(tr)) for m in ["lgb", "cb", "txt"]}
        for j, (a, b) in enumerate(inner):
            Ti, Pi = tr[a], tr[b]
            assert not (set(groups[Ti]) & set(groups[Pi]))
            l, c, t, d = cached(f"outer{k}_inner{j}", lambda: fit_predict(Ti, Pi, y, ind, base, corpus, meta, 42 + k))
            io["lgb"][b], io["cb"][b], io["txt"][b] = l, c, t
            provenance.append({"model": f"outer{k}_inner{j}", "fit_rows_sha": sha_arr(np.sort(Ti)),
                               "pred_rows_sha": sha_arr(np.sort(Pi)), **d})
        yi = y[tr]
        # ---- fit free parameters on inner-OOF only, then freeze
        def ll(w):
            w = np.asarray(w) / np.sum(w)
            return log_loss(yi, np.clip(w[0] * io["lgb"] + w[1] * io["cb"] + w[2] * io["txt"], 1e-7, 1 - 1e-7))
        res = minimize(ll, CONFIG["policy_A"]["x0"], bounds=CONFIG["policy_A"]["bounds"])
        wA = (res.x / res.x.sum()).tolist()
        Zi = np.column_stack([logit(io["lgb"]), logit(io["cb"]), logit(io["txt"])])
        stk = LogisticRegression(C=1.0, max_iter=1000).fit(Zi, yi)
        frozen = {"outer_fold": k + 1, "wA_lgb_cb_txt": wA, "C_coef": stk.coef_[0].tolist(),
                  "C_intercept": float(stk.intercept_[0]), "threshold": 0.5,
                  "inner_oof_acc": {m: float(accuracy_score(yi, io[m] >= 0.5)) for m in io},
                  "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (OUT / f"freeze_outer{k}.json").write_text(json.dumps(frozen, indent=2))
        frozen_all.append(frozen)
        log(f"  frozen: wA={np.round(wA, 3)}, C coef={np.round(stk.coef_[0], 3)}")
        # ---- outer refit on outer-train, predict outer-val
        l, c, t, d = cached(f"outer{k}_full", lambda: fit_predict(tr, va, y, ind, base, corpus, meta, 42 + k))
        provenance.append({"model": f"outer{k}_full", "fit_rows_sha": sha_arr(np.sort(tr)),
                           "pred_rows_sha": sha_arr(np.sort(va)), **d})
        rec["lgb"][va], rec["cb"][va], rec["txt"][va] = l, c, t
        rec["A"][va] = wA[0] * l + wA[1] * c + wA[2] * t
        rec["B"][va] = 0.5 * l + 0.5 * c
        rec["C"][va] = stk.predict_proba(np.column_stack([logit(l), logit(c), logit(t)]))[:, 1]
        # ---- fold metrics (labels of outer-val used only here)
        yv = y[va]
        seen = np.isin(groups[va], np.unique(groups[tr]))
        row = {"outer_fold": k + 1, "n_val": int(len(va)), "n_val_sellers": int(len(set(groups[va]))),
               "val_seen_rows_fold_local": int(seen.sum()), "val_new_rows_fold_local": int((~seen).sum())}
        for p in ["lgb", "cb", "txt", "A", "B", "C"]:
            pr = np.clip(rec[p][va], 1e-7, 1 - 1e-7)
            row[f"hits_{p}"] = int(((pr >= 0.5) == yv).sum()); row[f"ll_{p}"] = round(float(log_loss(yv, pr)), 5)
        cA = (rec["A"][va] >= 0.5) == yv
        for p in ["B", "C"]:
            cp = (rec[p][va] >= 0.5) == yv
            row[f"fixes_{p}_vs_A"] = int((cp & ~cA).sum()); row[f"breaks_{p}_vs_A"] = int((~cp & cA).sum())
            row[f"delta_{p}_vs_A"] = row[f"hits_{p}"] - row["hits_A"]
        fold_rows.append(row)
        log(f"  fold {k+1}: hits A={row['hits_A']} B={row['hits_B']} ({row['delta_B_vs_A']:+d}) C={row['hits_C']} ({row['delta_C_vs_A']:+d}); seen rows={row['val_seen_rows_fold_local']}")
        pd.DataFrame(fold_rows).to_csv(OUT / "per_fold_metrics.csv", index=False)

    # ---- aggregate
    folds = pd.DataFrame(fold_rows)
    corr = {p: ((rec[p] >= 0.5) == y).astype(int) for p in rec}
    summary = {}
    for p in ["lgb", "cb", "txt", "A", "B", "C"]:
        summary[p] = {"hits": int(corr[p].sum()), "accuracy": round(float(corr[p].mean()), 5),
                      "logloss": round(float(log_loss(y, np.clip(rec[p], 1e-7, 1 - 1e-7))), 5)}
    comps = {}
    for p in ["B", "C"]:
        diff = corr[p] - corr["A"]
        bs = boot_two_way(diff.astype(float), groups, meta["category_id"].values, 2000, 20261111) * n
        lld = (-(y * np.log(np.clip(rec[p], 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - rec[p], 1e-7, 1)))
               + (y * np.log(np.clip(rec["A"], 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - rec["A"], 1e-7, 1))))
        bl = boot_two_way(lld, groups, meta["category_id"].values, 2000, 20261112)
        d = folds[f"delta_{p}_vs_A"]
        comps[f"{p}_vs_A"] = {
            "delta_hits_70k": int(diff.sum()), "fixes": int(((diff == 1)).sum()), "breaks": int((diff == -1).sum()),
            "folds_positive": int((d > 0).sum()), "folds_zero": int((d == 0).sum()), "worst_fold_delta": int(d.min()),
            "boot_delta_hits_median": float(np.median(bs)), "boot_delta_hits_q05": float(np.quantile(bs, 0.05)),
            "boot_delta_hits_q95": float(np.quantile(bs, 0.95)), "boot_frac_gt0": float((bs > 0).mean()),
            "delta_logloss": round(float(lld.mean()), 6), "boot_delta_logloss_q05": float(np.quantile(bl, 0.05)),
            "boot_delta_logloss_q95": float(np.quantile(bl, 0.95))}
    paired = pd.DataFrame({"row": np.arange(n), "item_id": meta["id"].values, "seller_id": groups,
                           "category_id": meta["category_id"].values, "outer_fold": fold_of + 1, "y_true": y,
                           "p_lgb": rec["lgb"], "p_cb": rec["cb"], "p_text": rec["txt"],
                           "p_A_podium_recipe": rec["A"], "p_B_dual_50_50": rec["B"], "p_C_logistic_stack": rec["C"]})
    pf = OUT / "paired_outer_val_predictions.csv"
    paired.to_csv(pf, index=False, lineterminator="\n")
    pd.DataFrame(provenance).to_csv(OUT / "model_provenance.csv", index=False)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        head = "unknown"
    manifest = {
        "status": "DEVELOPMENT_ONLY", "submission_generated": False, "kaggle_api_used": False,
        "git_head_at_run": head, "script": "scripts/seller_group_nested_podium_validation.py",
        "script_sha256": sha_file(Path(__file__)), "config": CONFIG,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__, "catboost": catboost.__version__, "numpy": np.__version__},
        "reused_stratified_oof": False,
        "seen_new_note": "Outer folds are seller-disjoint, so fold-local seen rows in outer-val are 0 by construction; "
                         "this experiment measures transport to unseen sellers only. Test has ~69% seen-seller rows; "
                         "no public-score gain is estimated here.",
        "frozen_policies": frozen_all, "summary": summary, "comparisons": comps,
        "paired_predictions_file": pf.name, "paired_predictions_sha256": sha_file(pf),
        "per_fold_sha256": sha_file(OUT / "per_fold_metrics.csv"),
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(json.dumps({"summary": summary, "comparisons": comps}, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
