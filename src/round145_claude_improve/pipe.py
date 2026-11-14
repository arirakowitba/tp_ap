"""Round145: B10's lgb_tpe_2 feature pipeline + new features (seller-free text OOF, seller-pooled text prob,
transductive seller aggregates, category-relative price, keyword flags). One thread.
usage: python pipe.py MODE OUTDIR
  MODE inner   : train = 56k minus inner fold, eval = inner fold (StratifiedGroupKFold(5, seed 145) fold 0 by parent groups)
  MODE holdout : train = 56k, eval = 14k holdout test
  MODE full    : train = 70k, eval = 30k real test
Writes OUTDIR/<variant>.npy (eval probabilities) and OUTDIR/feats_<mode>.pkl (stack frames) ."""
import os, sys, time, json, re, unicodedata, gc
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
sys.path.insert(0, "/home/claude/wt_main/src")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from optimize_leaderboard import numeric_text
from optimize_stack import group_rates, text_predict

MODE, OUTD = sys.argv[1], Path(sys.argv[2]); OUTD.mkdir(parents=True, exist_ok=True)
VARIANTS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["base", "plus"]
H2 = Path("/home/claude/work/holdout2")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

d = pd.read_pickle(H2 / ("feats_full.pkl" if MODE == "full" else "feats_holdout.pkl"))
if MODE == "inner":
    X, E, y, g = d["xa"], d["ea"], d["y"], d["groups"]
    f = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=145).split(np.zeros(len(y)), y, g))
    it, iv = f[0]
    x, z, ex, ez, ytr, gtr = X.iloc[it].reset_index(drop=True), X.iloc[iv].reset_index(drop=True), E.iloc[it].reset_index(drop=True), E.iloc[iv].reset_index(drop=True), y[it], g[it]
    np.save(OUTD / "inner_eval_rows.npy", iv); np.save(OUTD / "inner_eval_y.npy", y[iv])
else:
    x, z, ex, ez, ytr, gtr = d["xa"], d["xb"], d["ea"], d["eb"], d["y"], d["groups"]
del d; gc.collect()
ntr, nev = len(x), len(z)
log("mode", MODE, ntr, nev)

# ---------- B10 lgb_tpe_2 stack features (identical calls to confirm_and_submit.fit_models) ----------
both = pd.concat([x, z], ignore_index=True)
tr, va, cats, tf, svd = numeric_text(both, np.arange(ntr), np.arange(ntr, ntr + nev)); del both, tf, svd; gc.collect()
log("numeric_text done", tr.shape)
st, sv = group_rates(tr, va, ytr, gtr)
oof = np.zeros(ntr)
for k, (tt, vv) in enumerate(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026).split(np.zeros(ntr), ytr, gtr)):
    oof[vv] = text_predict(x.iloc[tt], ytr[tt], x.iloc[vv]); log("text oof", k)
st["text_oof_probability"] = oof; sv["text_oof_probability"] = text_predict(x, ytr, z)
log("B10 stack done", st.shape)

# ---------- new features ----------
def strip(s): return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
both = pd.concat([x, z], ignore_index=True); eb = pd.concat([ex, ez], ignore_index=True)
title = eb.title_raw.map(strip)
def qb(v, n=10):
    v = pd.to_numeric(v, errors="coerce")
    return pd.qcut(v.rank(method="first"), n, labels=False).astype(str).where(v.notna(), "na")
meta = ("cat_" + both.category_id.astype(str) + " lt_" + both.listing_type_id.astype(str) + " bm_" + both.buying_mode.astype(str)
        + " sm_" + both.shipping_mode.astype(str) + " fs_" + both.shipping_free_shipping.astype(str) + " lp_" + both.shipping_local_pick_up.astype(str)
        + " " + both.tags_joined.str.replace("|", " tag_", regex=False).radd("tag_") + " " + both.sub_status_joined.radd("ss_")
        + " " + eb.nonmp_ids.str.split().map(lambda l: " ".join("pm_" + t for t in l)) + " wm_" + both.warranty_missing.astype(str)
        + " pr_" + qb(both.price_raw) + " iq_" + qb(both.initial_quantity_raw) + " aq_" + qb(both.available_quantity_raw)
        + " sq_" + qb(both.sold_quantity_raw, 5) + " np_" + eb.n_pictures.clip(upper=8).astype(str) + " ar_" + both.automatic_relist.astype(str)
        + " os_" + both.official_store_id.astype(str).ne("__MISSING__").astype(str) + " nv_" + (eb.n_variations > 0).astype(str))
both_text = both.full_text
def ns_matrix(fit_idx, ev_idx):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000, sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=150000, sublinear_tf=True, dtype=np.float32)
    m = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2, binary=True, dtype=np.float32)
    A = sparse.hstack([w.fit_transform(both_text.iloc[fit_idx]), c.fit_transform(title.iloc[fit_idx]), m.fit_transform(meta.iloc[fit_idx])], format="csr")
    B = sparse.hstack([w.transform(both_text.iloc[ev_idx]), c.transform(title.iloc[ev_idx]), m.transform(meta.iloc[ev_idx])], format="csr")
    return A, B
def ns_predict(fit_idx, yfit, ev_idx, C=2.0):
    A, B = ns_matrix(fit_idx, ev_idx); lr = LogisticRegression(C=C, solver="liblinear", max_iter=300); lr.fit(A, yfit)
    return lr.predict_proba(B)[:, 1]
# seller-grouped OOF so that pooled seller means never contain a model trained on that seller's labels
sell = eb.seller_id.to_numpy(); sell_tr = sell[:ntr]
oof_ns = np.zeros(ntr)
for k, (tt, vv) in enumerate(GroupKFold(n_splits=3).split(np.zeros(ntr), ytr, sell_tr)):
    oof_ns[vv] = ns_predict(tt, ytr[tt], vv); log("ns oof", k)
ev_ns = ns_predict(np.arange(ntr), ytr, np.arange(ntr, ntr + nev)); log("ns full")
pall = np.r_[oof_ns, ev_ns]
s = pd.Series(sell); grp = pd.DataFrame({"s": sell, "p": pall}).groupby("s").p
ssum, scnt = grp.transform("sum").to_numpy(), grp.transform("count").to_numpy()
pool = np.where(scnt > 1, (ssum - pall) / np.maximum(scnt - 1, 1), np.nan)
new = pd.DataFrame({"ns_text_prob": pall, "ns_seller_pool_mean": pool, "seller_n_all": np.log1p(scnt)})
# pooled within seller x category
sc = s + "_" + both.category_id.astype(str).values
g2 = pd.DataFrame({"s": sc.values, "p": pall}).groupby("s").p
s2, c2 = g2.transform("sum").to_numpy(), g2.transform("count").to_numpy()
new["ns_sellercat_pool_mean"] = np.where(c2 > 1, (s2 - pall) / np.maximum(c2 - 1, 1), np.nan); new["sellercat_n_all"] = np.log1p(c2)
# transductive (unlabeled) seller aggregates over train+eval
agg = pd.DataFrame({"s": sell, "lp": both.log_price.values, "fs": (both.shipping_free_shipping.astype(str)=="True").astype(float).values,
                    "wm": both.warranty_missing.astype(float).values, "iq": both.log_initial_quantity.values,
                    "bin": (both.buying_mode.astype(str) == "buy_it_now").astype(float).values, "tu": both.title_mentions_used.astype(float).values,
                    "np": eb.n_pictures.values.astype(float), "nv": (eb.n_variations > 0).astype(float).values,
                    "free": (both.listing_type_id.astype(str) == "free").astype(float).values, "bronze": (both.listing_type_id.astype(str) == "bronze").astype(float).values,
                    "gold": both.listing_type_id.astype(str).str.startswith("gold").astype(float).values, "sold0": (both.sold_quantity_raw.fillna(0) == 0).astype(float).values})
for c in agg.columns[1:]: new["seller_mean_" + c] = agg.groupby("s")[c].transform("mean").values
new["seller_nuniq_cat"] = agg.assign(c=both.category_id.astype(str).values).groupby("s").c.transform("nunique").values
new["price_dev_seller"] = both.log_price.values - new["seller_mean_lp"].values
cat = both.category_id.astype(str)
new["price_rel_cat"] = both.log_price.values - both.log_price.groupby(cat).transform("median").values
new["price_pct_cat"] = both.log_price.groupby(cat).rank(pct=True).values
new["cat_n_all"] = np.log1p(cat.map(cat.value_counts()).values)
new["iq_rel_cat"] = both.log_initial_quantity.values - both.log_initial_quantity.groupby(cat).transform("median").values
kw = {"kw_nuevo": r"\bnuev[oa]s?\b", "kw_usado": r"\busad[oa]s?\b", "kw_sinuso": r"\bsin uso\b", "kw_antiguo": r"\bantigu[oa]s?\b|\bantiguedad", "kw_original": r"\boriginal",
      "kw_estrenar": r"estrenar", "kw_impecable": r"impecable|excelente estado|muy buen estado|buen estado", "kw_comonuevo": r"como nuev", "kw_vintage": r"vintage|retro|coleccion",
      "kw_reacond": r"reacondicion|refurb", "kw_liquido": r"liquido|oferta|promo", "kw_lote": r"\blote\b", "kw_garantia": r"garantia", "kw_sellado": r"sellad|blister|caja cerrada",
      "kw_importado": r"importad", "kw_envio": r"envio", "kw_unico": r"unic[oa]\b|unico dueno", "kw_detalle": r"detalle", "kw_libro": r"\blibro|\brevista", "kw_disco": r"\b(cd|lp|vinilo|dvd)\b"}
for k, pat in kw.items(): new[k] = title.str.contains(pat, regex=True).astype("int8").values
warr = both.warranty_text.map(strip)
for k, pat in {"w_nuevo": r"nuev", "w_usado": r"usad|sin garantia|no tiene", "w_fabrica": r"fabric|oficial", "w_meses": r"\d+\s*(mes|ano)"}.items(): new[k] = warr.str.contains(pat, regex=True).astype("int8").values
new_tr, new_ev = new.iloc[:ntr].reset_index(drop=True), new.iloc[ntr:].reset_index(drop=True)
np.save(OUTD / "ns_text_eval.npy", ev_ns); np.save(OUTD / "ns_text_oof.npy", oof_ns)
del both, eb; gc.collect(); log("new feats", new.shape)

GROUPS = {"base": [], "plus": list(new.columns),
          "plus_noagg": [c for c in new.columns if not c.startswith(("seller_mean", "seller_nuniq", "price_dev_seller"))],
          "ns_only": ["ns_text_prob", "ns_seller_pool_mean", "seller_n_all", "ns_sellercat_pool_mean", "sellercat_n_all"]}
params = {"n_estimators": 430, "learning_rate": 0.0228919761888824, "num_leaves": 94, "min_child_samples": 30, "reg_lambda": 0.6140703790090526,
          "reg_alpha": 0.07617361848894641, "colsample_bytree": 0.8889021191105235, "subsample": 0.9638149477566389, "subsample_freq": 1,
          "max_cat_threshold": 8, "cat_smooth": 22.90498629058358, "n_jobs": 1, "verbosity": -1, "random_state": 42}
pd.to_pickle(dict(st=st, sv=sv, new_tr=new_tr, new_ev=new_ev, cats=cats, ytr=ytr), OUTD / f"stack_{MODE}.pkl")
for v in VARIANTS:
    a = pd.concat([st, new_tr[GROUPS[v]]], axis=1); b = pd.concat([sv, new_ev[GROUPS[v]]], axis=1)
    for c in cats:
        dt = pd.CategoricalDtype(categories=pd.Index(a[c].unique())); a[c] = a[c].astype(dt); b[c] = b[c].astype(dt)
    m = LGBMClassifier(**params); m.fit(a, ytr); p = m.predict_proba(b)[:, 1]
    np.save(OUTD / f"lgb_{v}.npy", p); log("lgb", v, "done")
    if v == "plus":
        imp = pd.Series(m.booster_.feature_importance("gain"), index=a.columns).sort_values(ascending=False)
        imp.head(60).to_csv(OUTD / "lgb_plus_importance.csv")
log("ALL DONE")
