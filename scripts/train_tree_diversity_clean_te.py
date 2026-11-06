"""Tree Structural Diversity (CatBoost + LightGBM) with Clean Fold-Specific TE.

Implements CX-0093, CX-0094, and CX-0095 requirements:
1. make_te_clean: uses y[fit[tr]].mean() strictly inside each GroupKFold split
   for fit OOF, and y[fit].mean() for tune/confirm.
2. LightGBM seed diversity audit: logs seed-level predictions, best_iteration,
   correlations (Pearson/Spearman), absolute differences (median/max), and binary
   disagreements across the 3 seeds.
3. Evaluates predeclared rules (w_lgb=0.20, w_lgb=0.10) with frozen thresholds:
   - Common threshold 0.500
   - Tune-selected thresholds
   - Transported thresholds from previous historical cut
4. Bootstrap 95% CI over sellers, seen/new seller breakdown, paired CSV.
"""

import os, sys, json, hashlib, zipfile, argparse
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
OUT_DIR = ROOT / "results" / "tree_diversity_clean_te"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def split_testlike(sellers, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(sellers)
    rng.shuffle(unique)
    k = int(.05 * len(unique))
    tune_new, confirm_new = set(unique[:k]), set(unique[k:2 * k])
    grouped = {}
    for i, seller in enumerate(sellers):
        grouped.setdefault(seller, []).append(i)
    eligible = []
    for seller in unique[2 * k:]:
        ix = np.asarray(grouped[seller], dtype=int)
        rng.shuffle(ix)
        eligible.extend(ix[1:])
    eligible = np.asarray(eligible, dtype=int)
    rng.shuffle(eligible)
    n_seen = min(9000, len(eligible) // 2)
    tune = np.r_[np.flatnonzero(np.isin(sellers, list(tune_new))), eligible[:n_seen]]
    confirm = np.r_[np.flatnonzero(np.isin(sellers, list(confirm_new))), eligible[n_seen:2*n_seen]]
    fit = np.setdiff1d(np.arange(len(sellers)), np.r_[tune, confirm])
    return fit, tune, confirm


def make_te_clean(a, cat, y, fit, tune, confirm, sellers, m=10.0):
    te_cols = ["category_id", "listing_type_id", "state"]
    gkf = GroupKFold(n_splits=4)
    global_fit_prior = float(y[fit].mean())
    for col in te_cols:
        col_name = f"te_{col}"
        a[col_name] = global_fit_prior
        oof_vals = np.full(len(fit), global_fit_prior, dtype=np.float32)
        for tr, va in gkf.split(fit, y[fit], sellers[fit]):
            s_tr = cat.iloc[fit[tr]][col]
            fold_prior = float(y[fit[tr]].mean())
            stats = pd.DataFrame({"cat": s_tr, "y": y[fit[tr]]}).groupby("cat")["y"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + m * fold_prior) / (stats["count"] + m)
            oof_vals[va] = cat.iloc[fit[va]][col].map(smooth).fillna(fold_prior).astype(np.float32)
        a.loc[fit, col_name] = oof_vals

        # For tune and confirm: strictly use full-fit statistics and global_fit_prior
        fit_stats = pd.DataFrame({"cat": cat.iloc[fit][col], "y": y[fit]}).groupby("cat")["y"].agg(["count", "mean"])
        fit_smooth = (fit_stats["count"] * fit_stats["mean"] + m * global_fit_prior) / (fit_stats["count"] + m)
        a.loc[tune, col_name] = cat.iloc[tune][col].map(fit_smooth).fillna(global_fit_prior).astype(np.float32)
        a.loc[confirm, col_name] = cat.iloc[confirm][col].map(fit_smooth).fillna(global_fit_prior).astype(np.float32)


def generate_hist_text_oof(combos, titles, y, fit, tune, confirm, sellers, seed):
    print("Generating GroupKFold OOF predictions for Historical Text Model...", flush=True)
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
    n_sellers = len(unique_sellers)
    seller_to_indices = {s: np.where(sellers == s)[0] for s in unique_sellers}

    diffs = []
    for _ in range(n_boot):
        sample_sellers = rng.choice(unique_sellers, size=n_sellers, replace=True)
        idx = np.concatenate([seller_to_indices[s] for s in sample_sellers])
        acc_a = (pred_a[idx] == y_true[idx]).sum()
        acc_b = (pred_b[idx] == y_true[idx]).sum()
        diffs.append(acc_b - acc_a)

    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def evaluate_predictions(y_true, pred_cand, pred_ctrl, sellers, seen_mask, seed=42):
    hits_cand = int((pred_cand == y_true).sum())
    hits_ctrl = int((pred_ctrl == y_true).sum())
    delta = hits_cand - hits_ctrl
    fixes = int(((pred_ctrl != y_true) & (pred_cand == y_true)).sum())
    breaks = int(((pred_ctrl == y_true) & (pred_cand != y_true)).sum())
    discrepancies = int((pred_ctrl != pred_cand).sum())

    net_seen = int(((pred_cand[seen_mask] == y_true[seen_mask]).sum()) - ((pred_ctrl[seen_mask] == y_true[seen_mask]).sum()))
    net_new = int(((pred_cand[~seen_mask] == y_true[~seen_mask]).sum()) - ((pred_ctrl[~seen_mask] == y_true[~seen_mask]).sum()))
    ci_low, ci_high = run_bootstrap_seller(y_true, pred_ctrl, pred_cand, sellers, n_boot=3000, seed=seed)

    return {
        "hits": hits_cand,
        "delta": delta,
        "fixes": fixes,
        "breaks": breaks,
        "discrepancies": discrepancies,
        "net_seen": net_seen,
        "net_new": net_new,
        "ci_95": [ci_low, ci_high]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261144)
    parser.add_argument("--bagging_freq", type=int, default=0, help="LightGBM bagging_freq (0=disabled, 1=active)")
    parser.add_argument("--prev_th_cb", type=float, default=None, help="Transported CatBoost threshold from previous cut")
    parser.add_argument("--prev_th_ens8020", type=float, default=None, help="Transported Ens80/20 threshold from previous cut")
    parser.add_argument("--prev_th_ens9010", type=float, default=None, help="Transported Ens90/10 threshold from previous cut")
    args = parser.parse_args()

    SEED = args.seed
    print(f"=== TREE STRUCTURAL DIVERSITY WITH CLEAN TE (SEED {SEED}, LGB BAGGING_FREQ {args.bagging_freq}) ===", flush=True)

    with zipfile.ZipFile(ZIP_PATH) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000

    y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])

    fit, tune, confirm = split_testlike(sellers, SEED)
    print(f"Split sizes: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}", flush=True)

    # 1. Feature Extraction & Clean TE
    print("Extracting deep tabular features...", flush=True)
    a, cat, titles_raw, _, combos_raw = extract_deep_features(rows)
    make_te_clean(a, cat, y, fit, tune, confirm, sellers)

    oof_hist, pt_hist, pc_hist = generate_hist_text_oof(combos_raw, titles_raw, y, fit, tune, confirm, sellers, SEED)
    a["txt_sparse_p"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof_hist, pt_hist, pc_hist)):
        a.loc[idx, "txt_sparse_p"] = pred

    a["seller_id_cat"] = cat.seller_id.astype(str)
    cat_cols = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]

    a_lgb = a.copy()
    for col in cat_cols:
        a_lgb[col] = a_lgb[col].astype("category")

    cb_seeds = [SEED, SEED + 997, SEED + 1994]
    grid_th = np.arange(0.35, 0.651, 0.005)

    # 2. CatBoost Bag3
    print("\n--- Training CatBoost Bag3 (3 seeds) with Clean TE ---", flush=True)
    cb_tune_probs = []
    cb_conf_probs = []
    cb_best_iters = []
    for s_idx, s in enumerate(cb_seeds):
        print(f"Fitting CatBoost seed={s} ({s_idx+1}/3)...", flush=True)
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                random_seed=s, thread_count=14, verbose=False, loss_function="Logloss")
        cb.fit(a.loc[fit], y[fit], cat_features=cat_cols, eval_set=(a.loc[tune], y[tune]), early_stopping_rounds=100, use_best_model=True)
        cb_best_iters.append(int(cb.get_best_iteration()))
        cb_tune_probs.append(cb.predict_proba(a.loc[tune])[:, 1])
        cb_conf_probs.append(cb.predict_proba(a.loc[confirm])[:, 1])

    p_tune_cb_bag3 = np.mean(cb_tune_probs, axis=0)
    p_conf_cb_bag3 = np.mean(cb_conf_probs, axis=0)
    th_cb_tune = float(max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_cb_bag3 >= t)))
    acc_cb_tune = float(accuracy_score(y[tune], p_tune_cb_bag3 >= th_cb_tune))
    ll_cb_tune = float(log_loss(y[tune], p_tune_cb_bag3))
    print(f"CatBoost Bag3 Tune: Acc={acc_cb_tune:.5f}, th={th_cb_tune:.3f}, LL={ll_cb_tune:.5f}, best_iters={cb_best_iters}")

    # 3. LightGBM Bag3 with Diversity Logging
    print(f"\n--- Training LightGBM Bag3 (3 seeds, bagging_freq={args.bagging_freq}) ---", flush=True)
    lgb_tune_probs = []
    lgb_conf_probs = []
    lgb_best_iters = []
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "bagging_freq": args.bagging_freq,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": 14
    }

    d_fit = lgb.Dataset(a_lgb.loc[fit], label=y[fit])
    d_tune = lgb.Dataset(a_lgb.loc[tune], label=y[tune], reference=d_fit)

    for s_idx, s in enumerate(cb_seeds):
        print(f"Fitting LightGBM seed={s} ({s_idx+1}/3)...", flush=True)
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
        p_tu = m_lgb.predict(a_lgb.loc[tune])
        p_co = m_lgb.predict(a_lgb.loc[confirm])
        lgb_tune_probs.append(p_tu)
        lgb_conf_probs.append(p_co)

    # Audit LightGBM Seed Diversity on confirm
    lgb_conf_arr = np.array(lgb_conf_probs) # shape: (3, n_confirm)
    pearson_corr = np.corrcoef(lgb_conf_arr).tolist()
    spearman_corr = [
        [1.0, float(spearmanr(lgb_conf_arr[0], lgb_conf_arr[1]).statistic), float(spearmanr(lgb_conf_arr[0], lgb_conf_arr[2]).statistic)],
        [float(spearmanr(lgb_conf_arr[1], lgb_conf_arr[0]).statistic), 1.0, float(spearmanr(lgb_conf_arr[1], lgb_conf_arr[2]).statistic)],
        [float(spearmanr(lgb_conf_arr[2], lgb_conf_arr[0]).statistic), float(spearmanr(lgb_conf_arr[2], lgb_conf_arr[1]).statistic), 1.0]
    ]

    pairwise_seed_stats = {}
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        diff = np.abs(lgb_conf_arr[i] - lgb_conf_arr[j])
        disagreements = int(((lgb_conf_arr[i] >= 0.5) != (lgb_conf_arr[j] >= 0.5)).sum())
        pairwise_seed_stats[f"seeds_{cb_seeds[i]}_vs_{cb_seeds[j]}"] = {
            "mean_abs_diff": float(diff.mean()),
            "median_abs_diff": float(np.median(diff)),
            "max_abs_diff": float(diff.max()),
            "binary_disagreements_at_05": disagreements
        }

    p_tune_lgb_bag3 = np.mean(lgb_tune_probs, axis=0)
    p_conf_lgb_bag3 = np.mean(lgb_conf_probs, axis=0)
    th_lgb_tune = float(max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_lgb_bag3 >= t)))
    acc_lgb_tune = float(accuracy_score(y[tune], p_tune_lgb_bag3 >= th_lgb_tune))
    ll_lgb_tune = float(log_loss(y[tune], p_tune_lgb_bag3))
    print(f"LightGBM Bag3 Tune: Acc={acc_lgb_tune:.5f}, th={th_lgb_tune:.3f}, LL={ll_lgb_tune:.5f}, best_iters={lgb_best_iters}")

    # 4. Ensembles on Tune
    p_tune_ens8020 = 0.8 * p_tune_cb_bag3 + 0.2 * p_tune_lgb_bag3
    th_ens8020_tune = float(max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_ens8020 >= t)))
    acc_ens8020_tune = float(accuracy_score(y[tune], p_tune_ens8020 >= th_ens8020_tune))
    ll_ens8020_tune = float(log_loss(y[tune], p_tune_ens8020))

    p_tune_ens9010 = 0.9 * p_tune_cb_bag3 + 0.1 * p_tune_lgb_bag3
    th_ens9010_tune = float(max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_ens9010 >= t)))
    acc_ens9010_tune = float(accuracy_score(y[tune], p_tune_ens9010 >= th_ens9010_tune))
    ll_ens9010_tune = float(log_loss(y[tune], p_tune_ens9010))

    print(f"Ens 0.8/0.2 Tune: Acc={acc_ens8020_tune:.5f}, th={th_ens8020_tune:.3f}, LL={ll_ens8020_tune:.5f}")
    print(f"Ens 0.9/0.1 Tune: Acc={acc_ens9010_tune:.5f}, th={th_ens9010_tune:.3f}, LL={ll_ens9010_tune:.5f}")

    # 5. Confirm Predictions
    p_conf_ens8020 = 0.8 * p_conf_cb_bag3 + 0.2 * p_conf_lgb_bag3
    p_conf_ens9010 = 0.9 * p_conf_cb_bag3 + 0.1 * p_conf_lgb_bag3

    seen_mask = np.isin(sellers[confirm], sellers[fit])

    # LogLoss on Confirm
    ll_cb_conf = float(log_loss(y[confirm], p_conf_cb_bag3))
    ll_lgb_conf = float(log_loss(y[confirm], p_conf_lgb_bag3))
    ll_ens8020_conf = float(log_loss(y[confirm], p_conf_ens8020))
    ll_ens9010_conf = float(log_loss(y[confirm], p_conf_ens9010))

    print("\n=== CONFIRM EVALUATION ===", flush=True)
    print(f"Confirm LogLoss: CB={ll_cb_conf:.5f}, LGB={ll_lgb_conf:.5f}, Ens8020={ll_ens8020_conf:.5f} (delta={ll_ens8020_conf - ll_cb_conf:+.5f}), Ens9010={ll_ens9010_conf:.5f} (delta={ll_ens9010_conf - ll_cb_conf:+.5f})")

    # Evaluations:
    # A. Common threshold 0.500
    pred_cb_05 = (p_conf_cb_bag3 >= 0.5).astype(int)
    pred_ens8020_05 = (p_conf_ens8020 >= 0.5).astype(int)
    pred_ens9010_05 = (p_conf_ens9010 >= 0.5).astype(int)
    pred_lgb_05 = (p_conf_lgb_bag3 >= 0.5).astype(int)

    eval_05_ens8020 = evaluate_predictions(y[confirm], pred_ens8020_05, pred_cb_05, sellers[confirm], seen_mask, seed=SEED)
    eval_05_ens9010 = evaluate_predictions(y[confirm], pred_ens9010_05, pred_cb_05, sellers[confirm], seen_mask, seed=SEED)
    eval_05_lgb = evaluate_predictions(y[confirm], pred_lgb_05, pred_cb_05, sellers[confirm], seen_mask, seed=SEED)

    # B. Tune-selected thresholds
    pred_cb_tune = (p_conf_cb_bag3 >= th_cb_tune).astype(int)
    pred_ens8020_tune = (p_conf_ens8020 >= th_ens8020_tune).astype(int)
    pred_ens9010_tune = (p_conf_ens9010 >= th_ens9010_tune).astype(int)

    eval_tune_ens8020 = evaluate_predictions(y[confirm], pred_ens8020_tune, pred_cb_tune, sellers[confirm], seen_mask, seed=SEED)
    eval_tune_ens9010 = evaluate_predictions(y[confirm], pred_ens9010_tune, pred_cb_tune, sellers[confirm], seen_mask, seed=SEED)

    # C. Transported thresholds (if provided)
    eval_trans_ens8020 = None
    eval_trans_ens9010 = None
    if args.prev_th_cb is not None and args.prev_th_ens8020 is not None:
        pred_cb_trans = (p_conf_cb_bag3 >= args.prev_th_cb).astype(int)
        pred_ens8020_trans = (p_conf_ens8020 >= args.prev_th_ens8020).astype(int)
        eval_trans_ens8020 = evaluate_predictions(y[confirm], pred_ens8020_trans, pred_cb_trans, sellers[confirm], seen_mask, seed=SEED)
        eval_trans_ens8020["th_cb_transported"] = args.prev_th_cb
        eval_trans_ens8020["th_ens_transported"] = args.prev_th_ens8020

    if args.prev_th_cb is not None and args.prev_th_ens9010 is not None:
        pred_cb_trans = (p_conf_cb_bag3 >= args.prev_th_cb).astype(int)
        pred_ens9010_trans = (p_conf_ens9010 >= args.prev_th_ens9010).astype(int)
        eval_trans_ens9010 = evaluate_predictions(y[confirm], pred_ens9010_trans, pred_cb_trans, sellers[confirm], seen_mask, seed=SEED)
        eval_trans_ens9010["th_cb_transported"] = args.prev_th_cb
        eval_trans_ens9010["th_ens_transported"] = args.prev_th_ens9010

    print(f"At Common Th 0.500: Ens8020 delta={eval_05_ens8020['delta']:+d} ({eval_05_ens8020['fixes']} fixes, {eval_05_ens8020['breaks']} breaks), CI={eval_05_ens8020['ci_95']}")
    print(f"At Tune Th ({th_ens8020_tune:.3f} vs {th_cb_tune:.3f}): Ens8020 delta={eval_tune_ens8020['delta']:+d} ({eval_tune_ens8020['fixes']} fixes, {eval_tune_ens8020['breaks']} breaks), CI={eval_tune_ens8020['ci_95']}")
    if eval_trans_ens8020:
        print(f"At Transported Th ({args.prev_th_ens8020:.3f} vs {args.prev_th_cb:.3f}): Ens8020 delta={eval_trans_ens8020['delta']:+d} ({eval_trans_ens8020['fixes']} fixes, {eval_trans_ens8020['breaks']} breaks), CI={eval_trans_ens8020['ci_95']}")

    # Build report dict
    report = {
        "experiment": "tree_diversity_clean_te",
        "split_seed": SEED,
        "n_confirm": len(confirm),
        "bagging_freq": args.bagging_freq,
        "catboost_seeds": cb_seeds,
        "catboost_best_iters": cb_best_iters,
        "lightgbm_best_iters": lgb_best_iters,
        "lightgbm_diversity_confirm": {
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr,
            "pairwise_stats": pairwise_seed_stats
        },
        "tune_thresholds": {
            "th_cb": th_cb_tune,
            "th_lgb": th_lgb_tune,
            "th_ens8020": th_ens8020_tune,
            "th_ens9010": th_ens9010_tune
        },
        "logloss_confirm": {
            "cb_bag3": ll_cb_conf,
            "lgb_bag3": ll_lgb_conf,
            "ens8020": ll_ens8020_conf,
            "ens9010": ll_ens9010_conf,
            "delta_ll_8020": ll_ens8020_conf - ll_cb_conf,
            "delta_ll_9010": ll_ens9010_conf - ll_cb_conf
        },
        "eval_common_th_050": {
            "hits_cb": int((pred_cb_05 == y[confirm]).sum()),
            "hits_lgb": int((pred_lgb_05 == y[confirm]).sum()),
            "ens8020": eval_05_ens8020,
            "ens9010": eval_05_ens9010
        },
        "eval_tune_thresholds": {
            "hits_cb": int((pred_cb_tune == y[confirm]).sum()),
            "ens8020": eval_tune_ens8020,
            "ens9010": eval_tune_ens9010
        },
        "eval_transported_thresholds": {
            "ens8020": eval_trans_ens8020,
            "ens9010": eval_trans_ens9010
        }
    }

    out_file = OUT_DIR / f"report_trees_diversity_clean_te_{SEED}_bf{args.bagging_freq}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    df_pairs = pd.DataFrame({
        "item_id": ids[confirm],
        "seller_id": sellers[confirm],
        "is_seen": seen_mask.astype(int),
        "y_true": y[confirm],
        "p_cb_bag3": p_conf_cb_bag3,
        "p_lgb_s1": lgb_conf_arr[0],
        "p_lgb_s2": lgb_conf_arr[1],
        "p_lgb_s3": lgb_conf_arr[2],
        "p_lgb_bag3": p_conf_lgb_bag3,
        "p_ens8020": p_conf_ens8020,
        "p_ens9010": p_conf_ens9010,
        "pred_cb_tune": pred_cb_tune,
        "pred_ens8020_tune": pred_ens8020_tune,
        "pred_ens9010_tune": pred_ens9010_tune
    })
    pairs_file = OUT_DIR / f"paired_trees_diversity_clean_te_{SEED}_bf{args.bagging_freq}.csv"
    df_pairs.to_csv(pairs_file, index=False)

    print(f"\nReport saved to: {out_file}")
    print(f"Paired predictions saved to: {pairs_file}")


if __name__ == "__main__":
    main()
