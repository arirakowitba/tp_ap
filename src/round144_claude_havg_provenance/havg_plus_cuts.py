"""Validate the exact H-AVG-Plus recipe (AG-0017) on the 3 seller-group cuts with FIXED iterations (no early stopping,
so tune is clean): variant_c features as in seller_te_ablation.py (fit-only), CatBoost 600 it x seeds {2026,314159,20261030}
averaged, LightGBM 227 it random_state 2026; vc_blend = .75 CB_bag + .25 LGB. Saves per-row tune+confirm p.
usage: python havg_plus_cuts.py AG_WORKTREE OUTDIR"""
import sys, time, json
from pathlib import Path
wt, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(wt))
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from src.benchmark_20261030.feature_extractor import compute_txt_sparse_p
from src.benchmark_20261030.split_utils import split_seller_groups_r151
from src.intensive_20261030.seller_te_ablation import compute_target_encodings_ablation
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache
cache = build_or_load_cache(); F = cache["F_r151_tr"]; cat_tr = cache["cat_tr"]; y_tr = cache["y_tr"]; sellers_tr = cache["sellers_tr"]; ids_tr = cache["ids_tr"]
for cut in [2026, 314159, 20261030]:
    t0 = time.time(); fit, tune, confirm = split_seller_groups_r151(sellers_tr, cut); y_fit = y_tr[fit]; sf = sellers_tr[fit]
    oof_sp, (p_sp_tune, p_sp_conf) = compute_txt_sparse_p(cache["combos_hist_tr"], cache["titles_tr"], fit, [tune, confirm], y_fit, sf, seed=cut)
    fit_te, tune_te, conf_te = compute_target_encodings_ablation(cat_tr, fit, tune, confirm, y_fit, sf, variant="variant_a")
    X = {k: F.iloc[ix].copy().reset_index(drop=True) for k, ix in (("fit", fit), ("tu", tune), ("co", confirm))}
    for c in ["seller_id", "category_id", "city"]:
        X["fit"][f"te_{c}"] = fit_te[c]; X["tu"][f"te_{c}"] = tune_te[c]; X["co"][f"te_{c}"] = conf_te[c]
    X["fit"]["txt_sparse_p"] = oof_sp; X["tu"]["txt_sparse_p"] = p_sp_tune; X["co"]["txt_sparse_p"] = p_sp_conf
    cbf = list(CAT_FEATURES) + ["seller_id"]
    for k, ix in (("fit", fit), ("tu", tune), ("co", confirm)):
        X[k]["seller_id"] = cat_tr["seller_id"].iloc[ix].astype(str).values
        for c in CAT_FEATURES: X[k][c] = cat_tr[c].iloc[ix].astype(str).values
    ev = pd.concat([X["tu"], X["co"]], ignore_index=True); pf = Pool(X["fit"], y_fit, cat_features=cbf); pe = Pool(ev, cat_features=cbf)
    cbs = {}
    for s in [2026, 314159, 20261030]:
        m = CatBoostClassifier(iterations=600, learning_rate=0.07, depth=6, l2_leaf_reg=6.0, random_seed=s, thread_count=4, allow_writing_files=False, verbose=0).fit(pf)
        cbs[s] = m.predict_proba(pe)[:, 1]
    L = {"fit": X["fit"].copy(), "ev": ev.copy()}
    for c in CAT_FEATURES + ["seller_id"]:
        cats = pd.Index(L["fit"][c].unique())
        for k in L: L[k][c] = pd.Categorical(L[k][c], categories=cats)
    lg = lgb.LGBMClassifier(n_estimators=227, learning_rate=0.04, num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=1.0, reg_lambda=5.0, random_state=2026, n_jobs=4, verbose=-1).fit(L["fit"], y_fit)
    p_lgb = lg.predict_proba(L["ev"])[:, 1]
    nt = len(tune); rows = np.r_[tune, confirm]
    df = pd.DataFrame(dict(row_idx=rows, item_id=ids_tr[rows], split=["tune"] * nt + ["confirm"] * len(confirm), y_true=y_tr[rows],
                           **{f"p_cb_s{s}": cbs[s] for s in cbs}, p_lgb=p_lgb))
    df["p_cb_bag"] = np.mean([cbs[s] for s in cbs], axis=0); df["p_vc_blend"] = .75 * df.p_cb_bag + .25 * df.p_lgb
    df.to_csv(out / f"havg_plus_recipe_cut{cut}.csv.gz", index=False); print(cut, "done", round(time.time() - t0), "s", flush=True)
print("HPLUS DONE", flush=True)
