"""CC-0006: replicate AG-0011 variant_c (seller_te_ablation.py lines 175-268, variant_c only) to recover the per-row
0.75*CB + 0.25*LGB probabilities that AG-0011 reported but did not save. Uses Antigravity's own modules from a worktree
of antigravity/overnight-recovery-coordination-20261030 (898ffcd), identical hyperparameters and seeds.
usage: python replicate_vc_blend.py AG_WORKTREE OUTDIR"""
import sys, time, json
from pathlib import Path
wt, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(wt))
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss
from src.benchmark_20261030.feature_extractor import compute_txt_sparse_p
from src.benchmark_20261030.split_utils import split_seller_groups_r151
from src.intensive_20261030.seller_te_ablation import compute_target_encodings_ablation
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache
cache = build_or_load_cache(); F = cache["F_r151_tr"]; cat_tr = cache["cat_tr"]; y_tr = cache["y_tr"]; sellers_tr = cache["sellers_tr"]; ids_tr = cache["ids_tr"]
rep = {}
for seed in [2026, 314159, 20261030]:
    t0 = time.time(); fit, tune, confirm = split_seller_groups_r151(sellers_tr, seed); y_fit, y_tu, y_co = y_tr[fit], y_tr[tune], y_tr[confirm]; sf = sellers_tr[fit]
    oof_sp, (p_sp_tune, p_sp_conf) = compute_txt_sparse_p(cache["combos_hist_tr"], cache["titles_tr"], fit, [tune, confirm], y_fit, sf, seed=seed)
    fit_te, tune_te, conf_te = compute_target_encodings_ablation(cat_tr, fit, tune, confirm, y_fit, sf, variant="variant_a")
    X = {k: F.iloc[ix].copy().reset_index(drop=True) for k, ix in (("fit", fit), ("tu", tune), ("co", confirm))}
    for c in ["seller_id", "category_id", "city"]:
        X["fit"][f"te_{c}"] = fit_te[c]; X["tu"][f"te_{c}"] = tune_te[c]; X["co"][f"te_{c}"] = conf_te[c]
    X["fit"]["txt_sparse_p"] = oof_sp; X["tu"]["txt_sparse_p"] = p_sp_tune; X["co"]["txt_sparse_p"] = p_sp_conf
    cbf = list(CAT_FEATURES) + ["seller_id"]
    for k, ix in (("fit", fit), ("tu", tune), ("co", confirm)):
        X[k]["seller_id"] = cat_tr["seller_id"].iloc[ix].astype(str).values
        for c in CAT_FEATURES: X[k][c] = cat_tr[c].iloc[ix].astype(str).values
    cb = CatBoostClassifier(iterations=900, learning_rate=0.07, depth=6, l2_leaf_reg=6.0, random_seed=seed, thread_count=4, verbose=0)
    cb.fit(Pool(X["fit"], y_fit, cat_features=cbf), eval_set=Pool(X["tu"], y_tu, cat_features=cbf), early_stopping_rounds=80, verbose=0)
    p_cb_tu = cb.predict_proba(Pool(X["tu"], cat_features=cbf))[:, 1]; p_cb_co = cb.predict_proba(Pool(X["co"], cat_features=cbf))[:, 1]
    L = {k: v.copy() for k, v in X.items()}
    for c in CAT_FEATURES + ["seller_id"]:
        cats = pd.Index(L["fit"][c].unique())
        for k in L: L[k][c] = pd.Categorical(L[k][c], categories=cats)
    lg = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.04, num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=4, verbose=-1)
    lg.fit(L["fit"], y_fit, eval_set=[(L["tu"], y_tu)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lg_tu = lg.predict_proba(L["tu"])[:, 1]; p_lg_co = lg.predict_proba(L["co"])[:, 1]
    df = pd.concat([pd.DataFrame(dict(row_idx=tune, item_id=ids_tr[tune], split="tune", y_true=y_tu, p_cb=p_cb_tu, p_lgb=p_lg_tu)),
                    pd.DataFrame(dict(row_idx=confirm, item_id=ids_tr[confirm], split="confirm", y_true=y_co, p_cb=p_cb_co, p_lgb=p_lg_co))], ignore_index=True)
    df["p_blend"] = .75 * df.p_cb + .25 * df.p_lgb
    df.to_csv(out / f"vc_blend_replica_seed{seed}.csv.gz", index=False)
    pb = .75 * p_cb_co + .25 * p_lg_co
    rep[seed] = dict(best_iter_cb=int(cb.get_best_iteration()), best_iter_lgb=int(lg.best_iteration_ or -1), acc_cb=float(accuracy_score(y_co, p_cb_co >= .5)),
                     acc_blend=float(accuracy_score(y_co, pb >= .5)), ll_blend=float(log_loss(y_co, pb)), sec=round(time.time() - t0))
    print(seed, rep[seed], flush=True)
(out / "replica_report.json").write_text(json.dumps(rep, indent=2)); print("REPLICA DONE", flush=True)
