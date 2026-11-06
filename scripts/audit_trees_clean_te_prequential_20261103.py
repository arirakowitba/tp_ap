"""Audit of Tree Structural Diversity (CatBoost + LightGBM) with Clean TE Prior and Prequential Validation.

Implements all requirements of CX-0093, CX-0094, and CX-0095:
1. Clean fold-specific prior in make_te: fold_prior = y[fit[train_fold]].mean().
2. Direct empirical audit of LightGBM seed diversity:
   - Records individual seed probabilities, best_iteration, pairwise Pearson/Spearman correlations,
     max/median absolute differences, and binary disagreement count at threshold 0.50.
3. 3 Historical Splits evaluated prequentially:
   - Cut 1: seed 20261141
   - Cut 2: seed 20261143 (transports threshold from Cut 1)
   - Cut 3: seed 20261144 (transports threshold from Cut 2; compares vs AG-0047 baseline)
4. Frozen predeclared rules evaluated on Confirm:
   - CatBoost Bag3 (Control, w=0.0)
   - Blend 90/10 (w_LGB = 0.10)
   - Blend 80/20 (w_LGB = 0.20)
   Evaluated at:
   - Common threshold 0.50
   - Transported threshold from previous cut's tune
   - Own tune threshold
5. Seller-level bootstrap 95% CI (3,000 resamples), seen/new seller breakdown.
6. Prequential deduplicated summary across confirms.
"""

import os, sys, json, hashlib, zipfile, time
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_deep_text_20261101 import split_testlike
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
OUT_DIR = ROOT / "results" / "multimodel_trees_lgbm_catboost"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUT_SEEDS = [20261141, 20261143, 20261144]

def make_clean_te(a, cat, y, fit, tune, confirm, sellers, m=10.0):
    global_prior = float(y[fit].mean())
    te_cols = ["category_id", "listing_type_id", "state"]
    gkf = GroupKFold(n_splits=4)
    for col in te_cols:
        col_name = f"te_{col}"
        a[col_name] = global_prior
        oof_vals = np.full(len(fit), global_prior, dtype=np.float32)
        for tr, va in gkf.split(fit, y[fit], sellers[fit]):
            tr_rows = fit[tr]
            va_rows = fit[va]
            fold_prior = float(y[tr_rows].mean())
            stats = pd.DataFrame({
                "cat": cat.iloc[tr_rows][col].to_numpy(),
                "y": y[tr_rows]
            }).groupby("cat")["y"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + m * fold_prior) / (stats["count"] + m)
            oof_vals[va] = cat.iloc[va_rows][col].map(smooth).fillna(fold_prior).astype(np.float32)
        a.loc[fit, col_name] = oof_vals
        fit_stats = pd.DataFrame({
            "cat": cat.iloc[fit][col].to_numpy(),
            "y": y[fit]
        }).groupby("cat")["y"].agg(["count", "mean"])
        fit_smooth = (fit_stats["count"] * fit_stats["mean"] + m * global_prior) / (fit_stats["count"] + m)
        a.loc[tune, col_name] = cat.iloc[tune][col].map(fit_smooth).fillna(global_prior).astype(np.float32)
        a.loc[confirm, col_name] = cat.iloc[confirm][col].map(fit_smooth).fillna(global_prior).astype(np.float32)

def generate_hist_text_oof(combos, titles, y, fit, tune, confirm, sellers, seed):
    print("  Generating GroupKFold OOF predictions for Historical Text Model...", flush=True)
    gkf = GroupKFold(n_splits=4)
    oof_fit = np.empty(len(fit), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(fit, y[fit], sellers[fit])):
        f_tr = fit[tr_idx]
        f_va = fit[va_idx]
        w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True, strip_accents="unicode", lowercase=True)
        c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=120000, sublinear_tf=True)

        X_tr = sp.hstack([w.fit_transform([combos[i] for i in f_tr]), c.fit_transform([titles[i] for i in f_tr])], format="csr")
        X_va = sp.hstack([w.transform([combos[i] for i in f_va]), c.transform([titles[i] for i in f_va])], format="csr")

        lr = LogisticRegression(C=3.0, max_iter=200, solver="liblinear", random_state=seed)
        lr.fit(X_tr, y[f_tr])
        oof_fit[va_idx] = lr.predict_proba(X_va)[:, 1]

    w_full = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True, strip_accents="unicode", lowercase=True)
    c_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=120000, sublinear_tf=True)

    X_full_fit = sp.hstack([w_full.fit_transform([combos[i] for i in fit]), c_full.fit_transform([titles[i] for i in fit])], format="csr")
    X_tune = sp.hstack([w_full.transform([combos[i] for i in tune]), c_full.transform([titles[i] for i in tune])], format="csr")
    X_conf = sp.hstack([w_full.transform([combos[i] for i in confirm]), c_full.transform([titles[i] for i in confirm])], format="csr")

    lr_full = LogisticRegression(C=3.0, max_iter=200, solver="liblinear", random_state=seed)
    lr_full.fit(X_full_fit, y[fit])

    p_tune = lr_full.predict_proba(X_tune)[:, 1]
    p_conf = lr_full.predict_proba(X_conf)[:, 1]

    return oof_fit, p_tune, p_conf

def run_bootstrap_seller(y_true, pred_a, pred_b, sellers, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    unique_sellers = np.unique(sellers)
    seller_to_indices = {s: np.where(sellers == s)[0] for s in unique_sellers}
    n_sellers = len(unique_sellers)

    diffs = []
    for _ in range(n_boot):
        sample_sellers = rng.choice(unique_sellers, size=n_sellers, replace=True)
        idx = np.concatenate([seller_to_indices[s] for s in sample_sellers])
        acc_a = (pred_a[idx] == y_true[idx]).sum()
        acc_b = (pred_b[idx] == y_true[idx]).sum()
        diffs.append(acc_b - acc_a)

    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def compute_diversity_stats(preds_list, labels, name_prefix=""):
    """preds_list: list of 3 arrays of probabilities."""
    n_models = len(preds_list)
    stats = {
        "n_models": n_models,
        "pairwise": {}
    }
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        p_i, p_j = preds_list[i], preds_list[j]
        pearson = float(np.corrcoef(p_i, p_j)[0, 1])
        spearman = float(pd.Series(p_i).corr(pd.Series(p_j), method="spearman"))
        diff = np.abs(p_i - p_j)
        max_diff = float(np.max(diff))
        median_diff = float(np.median(diff))
        p95_diff = float(np.percentile(diff, 95))
        disag_05 = int(((p_i >= 0.5) != (p_j >= 0.5)).sum())
        disag_rate_05 = float(disag_05 / len(p_i))
        key = f"model_{i+1}_vs_{j+1}"
        stats["pairwise"][key] = {
            "pearson_corr": pearson,
            "spearman_corr": spearman,
            "max_abs_diff": max_diff,
            "median_abs_diff": median_diff,
            "p95_abs_diff": p95_diff,
            "disagreements_05": disag_05,
            "disagreement_rate_05": disag_rate_05
        }
    return stats

def evaluate_prediction(y_true, p_pred, th, p_control, th_control, sellers, fit_sellers, seed=42):
    pred = (p_pred >= th).astype(int)
    pred_ctl = (p_control >= th_control).astype(int)
    n = len(y_true)
    hits = int((pred == y_true).sum())
    hits_ctl = int((pred_ctl == y_true).sum())
    acc = float(hits / n)
    ll = float(log_loss(y_true, p_pred))

    delta_hits = hits - hits_ctl
    fixes = int(((pred_ctl != y_true) & (pred == y_true)).sum())
    breaks = int(((pred_ctl == y_true) & (pred != y_true)).sum())
    discrepancies = int((pred_ctl != pred).sum())

    seen_mask = np.isin(sellers, fit_sellers)
    net_seen = int(((pred[seen_mask] == y_true[seen_mask]).sum()) - ((pred_ctl[seen_mask] == y_true[seen_mask]).sum()))
    net_new = int(((pred[~seen_mask] == y_true[~seen_mask]).sum()) - ((pred_ctl[~seen_mask] == y_true[~seen_mask]).sum()))

    ci_low, ci_high = run_bootstrap_seller(y_true, pred_ctl, pred, sellers, n_boot=3000, seed=seed)

    return {
        "th": float(th),
        "hits": hits,
        "total": n,
        "acc": acc,
        "logloss": ll,
        "delta_hits": delta_hits,
        "fixes": fixes,
        "breaks": breaks,
        "discrepancies": discrepancies,
        "net_seen": net_seen,
        "net_new": net_new,
        "ci_95": [ci_low, ci_high]
    }

def main():
    t0 = time.time()
    print("=== AUDIT: TREE DIVERSITY (CATBOOST + LIGHTGBM) WITH CLEAN TE PRIOR & PREQUENTIAL VALIDATION ===", flush=True)

    with zipfile.ZipFile(ZIP_PATH) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000

    y_all = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers_all = np.array([str(r["seller_id"]) for r in rows])
    ids_all = np.array([str(r["id"]) for r in rows])

    print("Extracting deep tabular feature base (shared across cuts)...", flush=True)
    a_base, cat_df, titles_raw, _, combos_raw = extract_deep_features(rows)
    cat_cols = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]

    grid_th = np.arange(0.35, 0.651, 0.005)

    cut_results = {}
    transported_thresholds = {}  # key: cut_seed -> dict of tune thresholds

    # To store predictions across cuts for prequential deduplicated evaluation
    all_confirm_records = []

    for cut_idx, cut_seed in enumerate(CUT_SEEDS):
        t_cut0 = time.time()
        print(f"\n=======================================================", flush=True)
        print(f"CUT {cut_idx + 1}/{len(CUT_SEEDS)}: SPLIT SEED = {cut_seed}", flush=True)
        print(f"=======================================================", flush=True)

        fit, tune, confirm = split_testlike(sellers_all, cut_seed)
        print(f"Sizes: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}", flush=True)

        a = a_base.copy()
        a["seller_id_cat"] = cat_df.seller_id.astype(str)

        print("Applying clean fold-specific TE prior...", flush=True)
        make_clean_te(a, cat_df, y_all, fit, tune, confirm, sellers_all)

        oof_hist, pt_hist, pc_hist = generate_hist_text_oof(combos_raw, titles_raw, y_all, fit, tune, confirm, sellers_all, seed=cut_seed)
        a["txt_sparse_p"] = np.nan
        for idx, pred in zip((fit, tune, confirm), (oof_hist, pt_hist, pc_hist)):
            a.loc[idx, "txt_sparse_p"] = pred

        # LightGBM categorical format
        a_lgb = a.copy()
        for col in cat_cols:
            a_lgb[col] = a_lgb[col].astype("category")

        model_seeds = [cut_seed, cut_seed + 997, cut_seed + 1994]

        # --- Train CatBoost Bag3 ---
        print("\n--- Fitting CatBoost Bag3 (3 seeds) ---", flush=True)
        cb_tune_probs = []
        cb_conf_probs = []
        cb_best_iters = []
        for s_idx, s in enumerate(model_seeds):
            print(f"  Fitting CatBoost seed={s} ({s_idx+1}/3)...", flush=True)
            cb = CatBoostClassifier(
                iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                random_seed=s, thread_count=14, verbose=False, loss_function="Logloss"
            )
            cb.fit(
                a.loc[fit], y_all[fit],
                cat_features=cat_cols,
                eval_set=(a.loc[tune], y_all[tune]),
                early_stopping_rounds=100,
                use_best_model=True
            )
            cb_best_iters.append(int(cb.get_best_iteration()))
            cb_tune_probs.append(cb.predict_proba(a.loc[tune])[:, 1].astype(np.float32))
            cb_conf_probs.append(cb.predict_proba(a.loc[confirm])[:, 1].astype(np.float32))

        cb_diversity_stats = compute_diversity_stats(cb_conf_probs, y_all[confirm], "CatBoost")
        cb_diversity_stats["best_iterations"] = cb_best_iters
        p_tune_cb = np.mean(cb_tune_probs, axis=0)
        p_conf_cb = np.mean(cb_conf_probs, axis=0)

        th_cb_tune = float(max(grid_th, key=lambda t: accuracy_score(y_all[tune], p_tune_cb >= t)))
        acc_cb_tune = float(accuracy_score(y_all[tune], p_tune_cb >= th_cb_tune))
        ll_cb_tune = float(log_loss(y_all[tune], p_tune_cb))
        print(f"CatBoost Bag3 Tune: Acc={acc_cb_tune:.5f}, th={th_cb_tune:.3f}, LL={ll_cb_tune:.5f}")

        # --- Train LightGBM Bag3 ---
        print("\n--- Fitting LightGBM Bag3 (3 seeds) ---", flush=True)
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": 14
        }
        d_fit = lgb.Dataset(a_lgb.loc[fit], label=y_all[fit])
        d_tune = lgb.Dataset(a_lgb.loc[tune], label=y_all[tune], reference=d_fit)

        lgb_tune_probs = []
        lgb_conf_probs = []
        lgb_best_iters = []
        for s_idx, s in enumerate(model_seeds):
            print(f"  Fitting LightGBM seed={s} ({s_idx+1}/3)...", flush=True)
            p_dict = lgb_params.copy()
            p_dict["random_state"] = s
            m_lgb = lgb.train(
                p_dict,
                d_fit,
                num_boost_round=1000,
                valid_sets=[d_tune],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
            lgb_best_iters.append(int(m_lgb.best_iteration))
            lgb_tune_probs.append(m_lgb.predict(a_lgb.loc[tune]).astype(np.float32))
            lgb_conf_probs.append(m_lgb.predict(a_lgb.loc[confirm]).astype(np.float32))

        lgb_diversity_stats = compute_diversity_stats(lgb_conf_probs, y_all[confirm], "LightGBM")
        lgb_diversity_stats["best_iterations"] = lgb_best_iters
        p_tune_lgb = np.mean(lgb_tune_probs, axis=0)
        p_conf_lgb = np.mean(lgb_conf_probs, axis=0)

        th_lgb_tune = float(max(grid_th, key=lambda t: accuracy_score(y_all[tune], p_tune_lgb >= t)))
        acc_lgb_tune = float(accuracy_score(y_all[tune], p_tune_lgb >= th_lgb_tune))
        ll_lgb_tune = float(log_loss(y_all[tune], p_tune_lgb))
        print(f"LightGBM Bag3 Tune: Acc={acc_lgb_tune:.5f}, th={th_lgb_tune:.3f}, LL={ll_lgb_tune:.5f}")

        # --- Predeclared Blends on Tune ---
        p_tune_b90 = 0.9 * p_tune_cb + 0.1 * p_tune_lgb
        p_conf_b90 = 0.9 * p_conf_cb + 0.1 * p_conf_lgb
        th_b90_tune = float(max(grid_th, key=lambda t: accuracy_score(y_all[tune], p_tune_b90 >= t)))

        p_tune_b80 = 0.8 * p_tune_cb + 0.2 * p_tune_lgb
        p_conf_b80 = 0.8 * p_conf_cb + 0.2 * p_conf_lgb
        th_b80_tune = float(max(grid_th, key=lambda t: accuracy_score(y_all[tune], p_tune_b80 >= t)))

        tune_thresholds = {
            "cb": th_cb_tune,
            "lgb": th_lgb_tune,
            "b90": th_b90_tune,
            "b80": th_b80_tune
        }
        transported_thresholds[cut_seed] = tune_thresholds

        # Determine transported threshold from previous cut (if available)
        prev_tune_th = None
        if cut_idx > 0:
            prev_seed = CUT_SEEDS[cut_idx - 1]
            prev_tune_th = transported_thresholds[prev_seed]
            print(f"Transported Thresholds from Cut {prev_seed}: CB={prev_tune_th['cb']:.3f}, B90={prev_tune_th['b90']:.3f}, B80={prev_tune_th['b80']:.3f}")

        # --- Confirm Evaluations ---
        print("\n--- Confirm Evaluations ---", flush=True)
        fit_sellers = set(sellers_all[fit])
        y_conf = y_all[confirm]
        sellers_conf = sellers_all[confirm]

        # 1. Neutral Threshold 0.50
        eval_cb_050 = evaluate_prediction(y_conf, p_conf_cb, 0.50, p_conf_cb, 0.50, sellers_conf, fit_sellers, seed=cut_seed)
        eval_lgb_050 = evaluate_prediction(y_conf, p_conf_lgb, 0.50, p_conf_cb, 0.50, sellers_conf, fit_sellers, seed=cut_seed)
        eval_b90_050 = evaluate_prediction(y_conf, p_conf_b90, 0.50, p_conf_cb, 0.50, sellers_conf, fit_sellers, seed=cut_seed)
        eval_b80_050 = evaluate_prediction(y_conf, p_conf_b80, 0.50, p_conf_cb, 0.50, sellers_conf, fit_sellers, seed=cut_seed)

        # 2. Own Tune Thresholds
        eval_cb_own = evaluate_prediction(y_conf, p_conf_cb, th_cb_tune, p_conf_cb, th_cb_tune, sellers_conf, fit_sellers, seed=cut_seed)
        eval_b90_own = evaluate_prediction(y_conf, p_conf_b90, th_b90_tune, p_conf_cb, th_cb_tune, sellers_conf, fit_sellers, seed=cut_seed)
        eval_b80_own = evaluate_prediction(y_conf, p_conf_b80, th_b80_tune, p_conf_cb, th_cb_tune, sellers_conf, fit_sellers, seed=cut_seed)

        # 3. Transported Thresholds (from previous cut)
        eval_cb_trans = None
        eval_b90_trans = None
        eval_b80_trans = None
        if prev_tune_th is not None:
            eval_cb_trans = evaluate_prediction(y_conf, p_conf_cb, prev_tune_th["cb"], p_conf_cb, prev_tune_th["cb"], sellers_conf, fit_sellers, seed=cut_seed)
            eval_b90_trans = evaluate_prediction(y_conf, p_conf_b90, prev_tune_th["b90"], p_conf_cb, prev_tune_th["cb"], sellers_conf, fit_sellers, seed=cut_seed)
            eval_b80_trans = evaluate_prediction(y_conf, p_conf_b80, prev_tune_th["b80"], p_conf_cb, prev_tune_th["cb"], sellers_conf, fit_sellers, seed=cut_seed)

        print(f"Confirm @ 0.50:  CB={eval_cb_050['acc']:.5f} ({eval_cb_050['hits']}), B80={eval_b80_050['acc']:.5f} ({eval_b80_050['hits']}), Delta={eval_b80_050['delta_hits']:+d}, CI={eval_b80_050['ci_95']}, LL={eval_b80_050['logloss']:.5f}")
        print(f"Confirm @ Own:   CB={eval_cb_own['acc']:.5f} ({eval_cb_own['hits']}), B80={eval_b80_own['acc']:.5f} ({eval_b80_own['hits']}), Delta={eval_b80_own['delta_hits']:+d}, CI={eval_b80_own['ci_95']}")
        if eval_b80_trans is not None:
            print(f"Confirm @ Trans: CB={eval_cb_trans['acc']:.5f} ({eval_cb_trans['hits']}), B80={eval_b80_trans['acc']:.5f} ({eval_b80_trans['hits']}), Delta={eval_b80_trans['delta_hits']:+d}, CI={eval_b80_trans['ci_95']}")

        # Save paired predictions CSV for this cut
        paired_df = pd.DataFrame({
            "item_id": ids_all[confirm],
            "y_true": y_conf,
            "seller_id": sellers_conf,
            "p_cb_bag3": p_conf_cb,
            "p_lgb_bag3": p_conf_lgb,
            "p_blend_90_10": p_conf_b90,
            "p_blend_80_20": p_conf_b80,
            "p_lgb_s1": lgb_conf_probs[0],
            "p_lgb_s2": lgb_conf_probs[1],
            "p_lgb_s3": lgb_conf_probs[2],
            "p_cb_s1": cb_conf_probs[0],
            "p_cb_s2": cb_conf_probs[1],
            "p_cb_s3": cb_conf_probs[2]
        })
        paired_csv_path = OUT_DIR / f"paired_trees_clean_te_{cut_seed}.csv"
        paired_df.to_csv(paired_csv_path, index=False)
        paired_hash = hashlib.sha256(paired_csv_path.read_bytes()).hexdigest()

        cut_data = {
            "seed": cut_seed,
            "sizes": {"fit": len(fit), "tune": len(tune), "confirm": len(confirm)},
            "tune_thresholds": tune_thresholds,
            "cb_diversity": cb_diversity_stats,
            "lgb_diversity": lgb_diversity_stats,
            "eval_050": {
                "cb": eval_cb_050,
                "lgb": eval_lgb_050,
                "b90": eval_b90_050,
                "b80": eval_b80_050
            },
            "eval_own_tune": {
                "cb": eval_cb_own,
                "b90": eval_b90_own,
                "b80": eval_b80_own
            },
            "eval_transported": {
                "cb": eval_cb_trans,
                "b90": eval_b90_trans,
                "b80": eval_b80_trans
            } if prev_tune_th is not None else None,
            "paired_csv": str(paired_csv_path.relative_to(ROOT)),
            "paired_sha256": paired_hash,
            "duration_sec": time.time() - t_cut0
        }
        cut_results[str(cut_seed)] = cut_data

        # Append records for prequential deduplicated pool
        # For transported rule: if cut 0, use 0.50; otherwise use transported threshold
        th_b80_preq = prev_tune_th["b80"] if prev_tune_th is not None else 0.50
        th_cb_preq = prev_tune_th["cb"] if prev_tune_th is not None else 0.50
        for i_conf in range(len(confirm)):
            all_confirm_records.append({
                "item_id": ids_all[confirm[i_conf]],
                "cut_seed": cut_seed,
                "y_true": int(y_conf[i_conf]),
                "seller_id": sellers_conf[i_conf],
                "p_cb": float(p_conf_cb[i_conf]),
                "p_b80": float(p_conf_b80[i_conf]),
                "p_b90": float(p_conf_b90[i_conf]),
                "pred_cb_050": int(p_conf_cb[i_conf] >= 0.50),
                "pred_b80_050": int(p_conf_b80[i_conf] >= 0.50),
                "pred_cb_preq": int(p_conf_cb[i_conf] >= th_cb_preq),
                "pred_b80_preq": int(p_conf_b80[i_conf] >= th_b80_preq),
            })

    # --- Prequential Deduplicated Analysis ---
    print("\n=======================================================", flush=True)
    print("PREQUENTIAL DEDUPLICATED CONFIRM ANALYSIS", flush=True)
    print("=======================================================", flush=True)
    df_all_conf = pd.DataFrame(all_confirm_records)
    total_evals = len(df_all_conf)
    df_dedup = df_all_conf.drop_duplicates(subset=["item_id"], keep="first").copy()
    n_unique = len(df_dedup)
    print(f"Total confirm instances across 3 cuts: {total_evals}; Unique items: {n_unique}", flush=True)

    # Metrics on deduplicated set @ 0.50
    hits_cb_050 = int((df_dedup["pred_cb_050"] == df_dedup["y_true"]).sum())
    hits_b80_050 = int((df_dedup["pred_b80_050"] == df_dedup["y_true"]).sum())
    delta_dedup_050 = hits_b80_050 - hits_cb_050
    fixes_dedup_050 = int(((df_dedup["pred_cb_050"] != df_dedup["y_true"]) & (df_dedup["pred_b80_050"] == df_dedup["y_true"])).sum())
    breaks_dedup_050 = int(((df_dedup["pred_cb_050"] == df_dedup["y_true"]) & (df_dedup["pred_b80_050"] != df_dedup["y_true"])).sum())
    ll_cb_dedup = float(log_loss(df_dedup["y_true"], df_dedup["p_cb"]))
    ll_b80_dedup = float(log_loss(df_dedup["y_true"], df_dedup["p_b80"]))
    ci_dedup_050 = run_bootstrap_seller(
        df_dedup["y_true"].to_numpy(),
        df_dedup["pred_cb_050"].to_numpy(),
        df_dedup["pred_b80_050"].to_numpy(),
        df_dedup["seller_id"].to_numpy(),
        n_boot=3000, seed=42
    )

    # Metrics on deduplicated set @ transported
    hits_cb_preq = int((df_dedup["pred_cb_preq"] == df_dedup["y_true"]).sum())
    hits_b80_preq = int((df_dedup["pred_b80_preq"] == df_dedup["y_true"]).sum())
    delta_dedup_preq = hits_b80_preq - hits_cb_preq
    fixes_dedup_preq = int(((df_dedup["pred_cb_preq"] != df_dedup["y_true"]) & (df_dedup["pred_b80_preq"] == df_dedup["y_true"])).sum())
    breaks_dedup_preq = int(((df_dedup["pred_cb_preq"] == df_dedup["y_true"]) & (df_dedup["pred_b80_preq"] != df_dedup["y_true"])).sum())
    ci_dedup_preq = run_bootstrap_seller(
        df_dedup["y_true"].to_numpy(),
        df_dedup["pred_cb_preq"].to_numpy(),
        df_dedup["pred_b80_preq"].to_numpy(),
        df_dedup["seller_id"].to_numpy(),
        n_boot=3000, seed=42
    )

    dedup_summary = {
        "total_evaluations": total_evals,
        "unique_items": n_unique,
        "eval_050": {
            "hits_cb": hits_cb_050,
            "acc_cb": float(hits_cb_050 / n_unique),
            "hits_b80": hits_b80_050,
            "acc_b80": float(hits_b80_050 / n_unique),
            "delta_hits": delta_dedup_050,
            "fixes": fixes_dedup_050,
            "breaks": breaks_dedup_050,
            "ci_95": ci_dedup_050,
            "logloss_cb": ll_cb_dedup,
            "logloss_b80": ll_b80_dedup,
            "logloss_delta": ll_b80_dedup - ll_cb_dedup
        },
        "eval_transported_presequential": {
            "hits_cb": hits_cb_preq,
            "acc_cb": float(hits_cb_preq / n_unique),
            "hits_b80": hits_b80_preq,
            "acc_b80": float(hits_b80_preq / n_unique),
            "delta_hits": delta_dedup_preq,
            "fixes": fixes_dedup_preq,
            "breaks": breaks_dedup_preq,
            "ci_95": ci_dedup_preq
        }
    }
    print(f"Dedup Confirm @ 0.50:  CB={hits_cb_050}/{n_unique} ({hits_cb_050/n_unique:.5f}), B80={hits_b80_050}/{n_unique} ({hits_b80_050/n_unique:.5f}), Delta={delta_dedup_050:+d}, CI={ci_dedup_050}, LL={ll_b80_dedup:.5f} (delta={ll_b80_dedup-ll_cb_dedup:+.5f})")
    print(f"Dedup Confirm @ Preq:  CB={hits_cb_preq}/{n_unique} ({hits_cb_preq/n_unique:.5f}), B80={hits_b80_preq}/{n_unique} ({hits_b80_preq/n_unique:.5f}), Delta={delta_dedup_preq:+d}, CI={ci_dedup_preq}")

    # --- Delta vs AG-0047 Baseline on 20261144 ---
    ag0047_path = OUT_DIR / "paired_trees_diversity_20261144.csv"
    te_audit_comparison = None
    if ag0047_path.exists():
        print("\n--- Comparing Clean TE vs AG-0047 Contaminated TE on Split 20261144 ---", flush=True)
        df_old = pd.read_csv(ag0047_path)
        df_clean = pd.read_csv(OUT_DIR / "paired_trees_clean_te_20261144.csv")
        assert len(df_old) == len(df_clean)
        assert (df_clean["item_id"].to_numpy() == df_old["item_id"].to_numpy()).all()

        diff_cb_p = np.abs(df_clean["p_cb_bag3"] - df_old["p_cb"])
        diff_b80_p = np.abs(df_clean["p_blend_80_20"] - df_old["p_ens"])

        te_audit_comparison = {
            "p_cb_mean_abs_diff": float(diff_cb_p.mean()),
            "p_cb_max_abs_diff": float(diff_cb_p.max()),
            "p_b80_mean_abs_diff": float(diff_b80_p.mean()),
            "p_b80_max_abs_diff": float(diff_b80_p.max()),
            "ag0047_cb_acc_own": 0.955887,
            "ag0047_cb_ll": 0.120795,
            "ag0047_ens_acc_own": 0.956621,
            "ag0047_ens_ll": 0.119169,
            "clean_cb_acc_own": cut_results["20261144"]["eval_own_tune"]["cb"]["acc"],
            "clean_cb_ll": cut_results["20261144"]["eval_050"]["cb"]["logloss"],
            "clean_b80_acc_own": cut_results["20261144"]["eval_own_tune"]["b80"]["acc"],
            "clean_b80_ll": cut_results["20261144"]["eval_050"]["b80"]["logloss"],
            "clean_b80_delta_hits_own": cut_results["20261144"]["eval_own_tune"]["b80"]["delta_hits"],
            "clean_b80_ci_own": cut_results["20261144"]["eval_own_tune"]["b80"]["ci_95"]
        }
        print(f"Mean prob diff from clean TE: CB={diff_cb_p.mean():.6f}, Blend={diff_b80_p.mean():.6f}")

    # Final combined report
    total_elapsed = time.time() - t0
    full_report = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "total_elapsed_sec": total_elapsed,
        "cuts": cut_results,
        "deduplicated_prequential": dedup_summary,
        "te_clean_audit_vs_ag0047": te_audit_comparison
    }

    report_json_path = OUT_DIR / "report_trees_diversity_clean_te_prequential.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved JSON report: {report_json_path.relative_to(ROOT)}", flush=True)

if __name__ == "__main__":
    main()
