"""Model Diversity: CatBoost (Oblivious Trees) + LightGBM (Leaf-Wise Trees).

Hypothesis:
Post-hoc linear text blends failed because linear models lack context conditioning.
Collinear text OOF features failed because they inject split noise into CatBoost.
In contrast, tree structural diversity—combining symmetric oblivious trees (CatBoost)
with leaf-wise asymmetric trees (LightGBM)—captures fundamentally different decision
boundaries on the identical high-signal tabular + seller + text feature space.

Protocol:
- Split SEED = 20261144 (fit, tune, confirm).
- Identical feature matrix: 101 deep features + target encodings + historical text OOF.
- Model A: CatBoost Bag3 (iterations 900, depth 6, lr 0.07, 3 seeds).
- Model B: LightGBM (num_leaves 31, max_depth -1, learning_rate 0.05, 3 seeds).
- Model C: Predeclared Blend (weights chosen strictly on tune).
- Frozen evaluation on confirm:
  - Accuracy, log-loss, delta vs CatBoost Bag3.
  - Discrepancies, fixes, breaks, seen vs new sellers.
  - 3,000 bootstrap seller resamples for 95% CI.
  - Paired CSV with item_id and probabilities.
"""

import os, sys, json, hashlib, zipfile, re
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

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261144
ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
OUT_DIR = ROOT / "results" / "multimodel_trees_lgbm_catboost"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_te(a, cat, y, fit, tune, confirm, sellers, m=10.0):
    g_mean = float(y[fit].mean())
    te_cols = ["category_id", "listing_type_id", "state"]
    gkf = GroupKFold(n_splits=4)
    for col in te_cols:
        col_name = f"te_{col}"
        a[col_name] = g_mean
        oof_vals = np.full(len(fit), g_mean, dtype=np.float32)
        for tr, va in gkf.split(fit, y[fit], sellers[fit]):
            s_tr = cat.iloc[fit[tr]][col]
            stats = pd.DataFrame({"cat": s_tr, "y": y[fit[tr]]}).groupby("cat")["y"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + m * g_mean) / (stats["count"] + m)
            oof_vals[va] = cat.iloc[fit[va]][col].map(smooth).fillna(g_mean).astype(np.float32)
        a.loc[fit, col_name] = oof_vals
        fit_stats = pd.DataFrame({"cat": cat.iloc[fit][col], "y": y[fit]}).groupby("cat")["y"].agg(["count", "mean"])
        fit_smooth = (fit_stats["count"] * fit_stats["mean"] + m * g_mean) / (fit_stats["count"] + m)
        a.loc[tune, col_name] = cat.iloc[tune][col].map(fit_smooth).fillna(g_mean).astype(np.float32)
        a.loc[confirm, col_name] = cat.iloc[confirm][col].map(fit_smooth).fillna(g_mean).astype(np.float32)

def generate_hist_text_oof(combos, titles, y, fit, tune, confirm, sellers):
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

        lr = LogisticRegression(C=3.0, max_iter=200, solver="liblinear", random_state=SEED)
        lr.fit(X_tr, y[f_tr])
        oof_fit[va_idx] = lr.predict_proba(X_va)[:, 1]

    w_full = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True, strip_accents="unicode", lowercase=True)
    c_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=120000, sublinear_tf=True)

    X_full_fit = sp.hstack([w_full.fit_transform([combos[i] for i in fit]), c_full.fit_transform([titles[i] for i in fit])], format="csr")
    X_tune = sp.hstack([w_full.transform([combos[i] for i in tune]), c_full.transform([titles[i] for i in tune])], format="csr")
    X_conf = sp.hstack([w_full.transform([combos[i] for i in confirm]), c_full.transform([titles[i] for i in confirm])], format="csr")

    lr_full = LogisticRegression(C=3.0, max_iter=200, solver="liblinear", random_state=SEED)
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

def main():
    print(f"=== MULTI-MODEL DIVERSITY: CATBOOST + LIGHTGBM (SEED {SEED}) ===", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000

    y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])

    fit, tune, confirm = split_testlike(sellers, SEED)

    # 1. Feature Extraction
    print("Extracting deep tabular features...", flush=True)
    a, cat, titles_raw, _, combos_raw = extract_deep_features(rows)
    make_te(a, cat, y, fit, tune, confirm, sellers)

    oof_hist, pt_hist, pc_hist = generate_hist_text_oof(combos_raw, titles_raw, y, fit, tune, confirm, sellers)
    a["txt_sparse_p"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof_hist, pt_hist, pc_hist)):
        a.loc[idx, "txt_sparse_p"] = pred

    a["seller_id_cat"] = cat.seller_id.astype(str)
    cat_cols = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]

    # Encode categorical columns as category dtype for LightGBM
    a_lgb = a.copy()
    for col in cat_cols:
        a_lgb[col] = a_lgb[col].astype("category")

    cb_seeds = [SEED, SEED + 997, SEED + 1994]
    grid_th = np.arange(0.35, 0.651, 0.005)

    # --- 2. Train CatBoost Bag3 ---
    print("\n--- Training CatBoost Bag3 (3 seeds) ---", flush=True)
    cb_tune_probs = []
    cb_conf_probs = []
    for s_idx, s in enumerate(cb_seeds):
        print(f"Fitting CatBoost seed={s} ({s_idx+1}/3)...", flush=True)
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0, random_seed=s, thread_count=14, verbose=False, loss_function="Logloss")
        cb.fit(a.loc[fit], y[fit], cat_features=cat_cols, eval_set=(a.loc[tune], y[tune]), early_stopping_rounds=100, use_best_model=True)
        cb_tune_probs.append(cb.predict_proba(a.loc[tune])[:, 1])
        cb_conf_probs.append(cb.predict_proba(a.loc[confirm])[:, 1])

    p_tune_cb_bag3 = np.mean(cb_tune_probs, axis=0)
    p_conf_cb_bag3 = np.mean(cb_conf_probs, axis=0)
    th_cb = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_cb_bag3 >= t))
    acc_cb_tune = accuracy_score(y[tune], p_tune_cb_bag3 >= th_cb)
    print(f"CatBoost Bag3 Tune: Acc={acc_cb_tune:.5f}, th={th_cb:.3f}, LL={log_loss(y[tune], p_tune_cb_bag3):.5f}")

    # --- 3. Train LightGBM Bag3 ---
    print("\n--- Training LightGBM Bag3 (3 seeds) ---", flush=True)
    lgb_tune_probs = []
    lgb_conf_probs = []
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
        p_tu = m_lgb.predict(a_lgb.loc[tune])
        p_co = m_lgb.predict(a_lgb.loc[confirm])
        lgb_tune_probs.append(p_tu)
        lgb_conf_probs.append(p_co)

    p_tune_lgb_bag3 = np.mean(lgb_tune_probs, axis=0)
    p_conf_lgb_bag3 = np.mean(lgb_conf_probs, axis=0)
    th_lgb = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_lgb_bag3 >= t))
    acc_lgb_tune = accuracy_score(y[tune], p_tune_lgb_bag3 >= th_lgb)
    print(f"LightGBM Bag3 Tune: Acc={acc_lgb_tune:.5f}, th={th_lgb:.3f}, LL={log_loss(y[tune], p_tune_lgb_bag3):.5f}")

    # --- 4. Predeclared Ensemble Selection Strictly on TUNE ---
    print("\n--- Selecting Ensemble Weight on Tune ---", flush=True)
    best_w_lgb = 0.0
    best_th_ens = 0.5
    best_acc_ens_tune = 0.0

    for w_lgb in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p_ens_tu = (1.0 - w_lgb) * p_tune_cb_bag3 + w_lgb * p_tune_lgb_bag3
        th_w = max(grid_th, key=lambda t: accuracy_score(y[tune], p_ens_tu >= t))
        acc_w = accuracy_score(y[tune], p_ens_tu >= th_w)
        ll_w = log_loss(y[tune], p_ens_tu)
        print(f"w_lgb={w_lgb:.1f}: tune_acc={acc_w:.5f}, th={th_w:.3f}, LL={ll_w:.5f}")
        if acc_w > best_acc_ens_tune:
            best_acc_ens_tune = acc_w
            best_w_lgb = w_lgb
            best_th_ens = th_w

    print(f"\nOptimal Ensemble Selected on Tune: w_cb={1.0-best_w_lgb:.2f}, w_lgb={best_w_lgb:.2f}, th={best_th_ens:.3f}")

    # --- 5. Frozen Evaluation on Confirm ---
    print("\n=== FROZEN EVALUATION ON CONFIRM ===", flush=True)
    p_conf_ens = (1.0 - best_w_lgb) * p_conf_cb_bag3 + best_w_lgb * p_conf_lgb_bag3

    pred_conf_cb = (p_conf_cb_bag3 >= th_cb).astype(int)
    pred_conf_lgb = (p_conf_lgb_bag3 >= th_lgb).astype(int)
    pred_conf_ens = (p_conf_ens >= best_th_ens).astype(int)

    acc_cb = float(accuracy_score(y[confirm], pred_conf_cb))
    ll_cb = float(log_loss(y[confirm], p_conf_cb_bag3))

    acc_lgb = float(accuracy_score(y[confirm], pred_conf_lgb))
    ll_lgb = float(log_loss(y[confirm], p_conf_lgb_bag3))

    acc_ens = float(accuracy_score(y[confirm], pred_conf_ens))
    ll_ens = float(log_loss(y[confirm], p_conf_ens))

    hits_cb = int((pred_conf_cb == y[confirm]).sum())
    hits_lgb = int((pred_conf_lgb == y[confirm]).sum())
    hits_ens = int((pred_conf_ens == y[confirm]).sum())

    delta_ens_cb = hits_ens - hits_cb
    fixes_ens = int(((pred_conf_cb != y[confirm]) & (pred_conf_ens == y[confirm])).sum())
    breaks_ens = int(((pred_conf_cb == y[confirm]) & (pred_conf_ens != y[confirm])).sum())
    discrepancies = int((pred_conf_cb != pred_conf_ens).sum())

    seen_mask = np.isin(sellers[confirm], sellers[fit])
    net_seen = int(((pred_conf_ens[seen_mask] == y[confirm][seen_mask]).sum()) - ((pred_conf_cb[seen_mask] == y[confirm][seen_mask]).sum()))
    net_new = int(((pred_conf_ens[~seen_mask] == y[confirm][~seen_mask]).sum()) - ((pred_conf_cb[~seen_mask] == y[confirm][~seen_mask]).sum()))

    ci_low, ci_high = run_bootstrap_seller(y[confirm], pred_conf_cb, pred_conf_ens, sellers[confirm], 3000, SEED)

    print(f"CatBoost Bag3 Confirm: Acc={acc_cb:.5f} ({hits_cb}/{len(confirm)}), LL={ll_cb:.5f}, th={th_cb:.3f}")
    print(f"LightGBM Bag3 Confirm: Acc={acc_lgb:.5f} ({hits_lgb}/{len(confirm)}), LL={ll_lgb:.5f}, th={th_lgb:.3f}")
    print(f"Ensemble Confirm:      Acc={acc_ens:.5f} ({hits_ens}/{len(confirm)}), LL={ll_ens:.5f}, th={best_th_ens:.3f}")
    print(f"Delta Ensemble vs CatBoost: {delta_ens_cb:+d} hits ({fixes_ens} fixes, {breaks_ens} breaks, {discrepancies} discrepancies)")
    print(f"Seen delta: {net_seen:+d}, New delta: {net_new:+d}")
    print(f"Bootstrap 95% CI: [{ci_low:+.1f}, {ci_high:+.1f}]")

    report = {
        "experiment": "multimodel_trees_lgbm_catboost",
        "split_seed": SEED,
        "n_confirm": len(confirm),
        "best_w_lgb": float(best_w_lgb),
        "th_cb": float(th_cb),
        "th_lgb": float(th_lgb),
        "th_ens": float(best_th_ens),
        "acc_cb": acc_cb,
        "ll_cb": ll_cb,
        "acc_lgb": acc_lgb,
        "ll_lgb": ll_lgb,
        "acc_ens": acc_ens,
        "ll_ens": ll_ens,
        "delta_hits": delta_ens_cb,
        "fixes": fixes_ens,
        "breaks": breaks_ens,
        "discrepancies": discrepancies,
        "net_seen": net_seen,
        "net_new": net_new,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high
    }

    out_file = OUT_DIR / f"report_trees_diversity_{SEED}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    df_pairs = pd.DataFrame({
        "item_id": ids[confirm],
        "seller_id": sellers[confirm],
        "is_seen": seen_mask.astype(int),
        "y_true": y[confirm],
        "p_cb": p_conf_cb_bag3,
        "p_lgb": p_conf_lgb_bag3,
        "p_ens": p_conf_ens,
        "pred_cb": pred_conf_cb,
        "pred_lgb": pred_conf_lgb,
        "pred_ens": pred_conf_ens
    })
    pairs_file = OUT_DIR / f"paired_trees_diversity_{SEED}.csv"
    df_pairs.to_csv(pairs_file, index=False)

    print(f"Report saved to {out_file}")
    print(f"Pairs saved to {pairs_file}")

if __name__ == "__main__":
    main()
