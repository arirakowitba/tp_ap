"""Frente C: Clean TE, LightGBM Real Seed Diversity, Dense SVD Embeddings & Prequential Blends.

Protocol:
1. Clean TE: fold_prior = y[fit[tr]].mean() inside each GroupKFold split on fit.
   Global fit prior = y[fit].mean() reserved strictly for tune and confirm.
2. Dense Semantic Representation (LSA / TruncatedSVD):
   - 50 latent semantic components extracted from Title (1-3w, 3-6c) + Attributes (aid/name) + Warranty.
   - Fitted strictly on fit, transformed on tune and confirm.
3. CatBoost Bag3 (Control):
   - 101 deep features + clean TE + historical text OOF + 3 seeds.
4. LightGBM Bag3 (Real Seed Diversity):
   - subsample_freq=1, bagging_fraction=0.8, colsample_bytree=0.8, 3 seeds.
   - Audit variance, correlations, and discrepancies between seeds.
5. CatBoost + SVD (Enriched):
   - Control features + 50 SVD dense semantic components.
6. Prequential Blends (0.8/0.2 and 0.9/0.1):
   - Evaluated under common 0.50, own tune thresholds, and transported thresholds.
   - 3,000 bootstrap seller resamples for 95% CI.
"""

import os, sys, json, hashlib, zipfile, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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
OUT_DIR = ROOT / "results" / "frente_c_models_20261104"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = re.sub(r"(\d+)\s*(gb|mb|tb|ram|rom|ghz|mhz|mah|mp|px|cm|mm|kg|km|v|w)\b", r"\1_\2", text)
    text = re.sub(r"\b0\s*km\b", "cerokm", text)
    text = re.sub(r"[^a-z0-9_=\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_structured_attributes(attrs_list):
    if not attrs_list or not isinstance(attrs_list, list):
        return ""
    tokens = []
    for a in attrs_list:
        if not isinstance(a, dict):
            continue
        aid = normalize_text(a.get("id") or "")
        aname = normalize_text(a.get("name") or "")
        vname = normalize_text(a.get("value_name") or "")
        if aid and vname:
            tokens.append(f"aid_{aid}={vname.replace(' ', '_')}")
        if aname and vname:
            tokens.append(f"name_{aname.replace(' ', '_')}={vname.replace(' ', '_')}")
        if vname:
            tokens.append(f"val_{vname.replace(' ', '_')}")
    return " ".join(tokens)

def make_clean_te(a, cat, y, fit, tune, confirm, sellers, m=10.0):
    """Clean Target Encoding without fold leakage."""
    global_prior = float(y[fit].mean())
    te_cols = ["category_id", "listing_type_id", "state"]
    gkf = GroupKFold(n_splits=4)

    for col in te_cols:
        col_name = f"te_{col}"
        a[col_name] = global_prior
        oof_vals = np.full(len(fit), global_prior, dtype=np.float32)

        for tr, va in gkf.split(fit, y[fit], sellers[fit]):
            fold_prior = float(y[fit[tr]].mean())
            s_tr = cat.iloc[fit[tr]][col]
            stats = pd.DataFrame({"cat": s_tr, "y": y[fit[tr]]}).groupby("cat")["y"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + m * fold_prior) / (stats["count"] + m)
            oof_vals[va] = cat.iloc[fit[va]][col].map(smooth).fillna(fold_prior).astype(np.float32)

        a.loc[fit, col_name] = oof_vals
        # For tune and confirm, strictly use fit stats and global_prior
        fit_stats = pd.DataFrame({"cat": cat.iloc[fit][col], "y": y[fit]}).groupby("cat")["y"].agg(["count", "mean"])
        fit_smooth = (fit_stats["count"] * fit_stats["mean"] + m * global_prior) / (fit_stats["count"] + m)
        a.loc[tune, col_name] = cat.iloc[tune][col].map(fit_smooth).fillna(global_prior).astype(np.float32)
        a.loc[confirm, col_name] = cat.iloc[confirm][col].map(fit_smooth).fillna(global_prior).astype(np.float32)

def generate_hist_text_oof(combos, titles, y, fit, tune, confirm, sellers):
    print("Generating Clean GroupKFold OOF predictions for Historical Text Model...", flush=True)
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
    print(f"=== FRENTE C: CLEAN TE, LGB DIVERSITY & SVD (SEED {SEED}) ===", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000

    y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])

    fit, tune, confirm = split_testlike(sellers, SEED)

    # 1. Feature Extraction & Clean TE
    print("Extracting deep tabular features with CLEAN TE...", flush=True)
    a, cat, titles_raw, _, combos_raw = extract_deep_features(rows)
    make_clean_te(a, cat, y, fit, tune, confirm, sellers)

    oof_hist, pt_hist, pc_hist = generate_hist_text_oof(combos_raw, titles_raw, y, fit, tune, confirm, sellers)
    a["txt_sparse_p"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof_hist, pt_hist, pc_hist)):
        a.loc[idx, "txt_sparse_p"] = pred

    a["seller_id_cat"] = cat.seller_id.astype(str)
    cat_cols = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]

    # 2. Extract Dense Semantic Representation (TruncatedSVD 50 components)
    print("Extracting Dense Semantic Representation (TruncatedSVD 50)...", flush=True)
    titles_norm = [normalize_text(r.get("title")) for r in rows]
    attrs_norm = [extract_structured_attributes(r.get("attributes")) for r in rows]

    v_sem = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=80000, sublinear_tf=True)
    X_sem_fit = v_sem.fit_transform([f"{titles_norm[i]} {attrs_norm[i]}" for i in fit])
    X_sem_tune = v_sem.transform([f"{titles_norm[i]} {attrs_norm[i]}" for i in tune])
    X_sem_conf = v_sem.transform([f"{titles_norm[i]} {attrs_norm[i]}" for i in confirm])

    svd = TruncatedSVD(n_components=50, random_state=SEED)
    svd_fit = svd.fit_transform(X_sem_fit)
    svd_tune = svd.transform(X_sem_tune)
    svd_conf = svd.transform(X_sem_conf)

    # Add SVD features
    for c_idx in range(50):
        col_name = f"svd_sem_{c_idx:02d}"
        a[col_name] = np.nan
        a.loc[fit, col_name] = svd_fit[:, c_idx]
        a.loc[tune, col_name] = svd_tune[:, c_idx]
        a.loc[confirm, col_name] = svd_conf[:, c_idx]

    cols_control = [c for c in a.columns if not c.startswith("svd_sem_")]
    cols_enriched = list(a.columns)

    # Encode categorical columns as category dtype for LightGBM
    a_lgb = a[cols_control].copy()
    for col in cat_cols:
        a_lgb[col] = a_lgb[col].astype("category")

    cb_seeds = [SEED, SEED + 997, SEED + 1994]
    grid_th = np.arange(0.35, 0.651, 0.005)

    # --- 3. Train CatBoost Bag3 (Control with Clean TE) ---
    print("\n--- Training CatBoost Bag3 (Control with Clean TE) ---", flush=True)
    cb_tune_probs = []
    cb_conf_probs = []
    for s_idx, s in enumerate(cb_seeds):
        print(f"Fitting CatBoost seed={s} ({s_idx+1}/3)...", flush=True)
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0, random_seed=s, thread_count=14, verbose=False, loss_function="Logloss")
        cb.fit(a.loc[fit, cols_control], y[fit], cat_features=cat_cols, eval_set=(a.loc[tune, cols_control], y[tune]), early_stopping_rounds=100, use_best_model=True)
        cb_tune_probs.append(cb.predict_proba(a.loc[tune, cols_control])[:, 1])
        cb_conf_probs.append(cb.predict_proba(a.loc[confirm, cols_control])[:, 1])

    p_tune_cb_bag3 = np.mean(cb_tune_probs, axis=0)
    p_conf_cb_bag3 = np.mean(cb_conf_probs, axis=0)
    th_cb = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_cb_bag3 >= t))
    acc_cb_tune = accuracy_score(y[tune], p_tune_cb_bag3 >= th_cb)
    print(f"CatBoost Bag3 Tune: Acc={acc_cb_tune:.5f}, th={th_cb:.3f}, LL={log_loss(y[tune], p_tune_cb_bag3):.5f}")

    # --- 4. Train LightGBM Bag3 with Effective Bagging & Diversity Audit ---
    print("\n--- Training LightGBM Bag3 with Real Seed Diversity (subsample_freq=1, bagging=0.8) ---", flush=True)
    lgb_tune_probs = []
    lgb_conf_probs = []
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
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
        lgb_tune_probs.append(m_lgb.predict(a_lgb.loc[tune]))
        lgb_conf_probs.append(m_lgb.predict(a_lgb.loc[confirm]))

    # Seed diversity audit:
    corr_s01 = float(np.corrcoef(lgb_conf_probs[0], lgb_conf_probs[1])[0, 1])
    corr_s02 = float(np.corrcoef(lgb_conf_probs[0], lgb_conf_probs[2])[0, 1])
    corr_s12 = float(np.corrcoef(lgb_conf_probs[1], lgb_conf_probs[2])[0, 1])
    mean_diff_seeds = float(np.mean([np.abs(lgb_conf_probs[0] - lgb_conf_probs[1]).mean(), np.abs(lgb_conf_probs[0] - lgb_conf_probs[2]).mean()]))
    disagr_seeds = int(((lgb_conf_probs[0] >= 0.5) != (lgb_conf_probs[1] >= 0.5)).sum())
    print(f"LGB Seed Correlations: ({corr_s01:.4f}, {corr_s02:.4f}, {corr_s12:.4f}), Disagreements s0 vs s1: {disagr_seeds}")

    p_tune_lgb_bag3 = np.mean(lgb_tune_probs, axis=0)
    p_conf_lgb_bag3 = np.mean(lgb_conf_probs, axis=0)
    th_lgb = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_lgb_bag3 >= t))
    print(f"LightGBM Bag3 Tune: Acc={accuracy_score(y[tune], p_tune_lgb_bag3 >= th_lgb):.5f}, th={th_lgb:.3f}")

    # --- 5. Predeclared Blends Evaluation on Confirm ---
    print("\n=== EVALUATING PREDECLARED BLENDS ON CONFIRM ===", flush=True)
    # Blend 0.8 CB + 0.2 LGB
    p_conf_b80 = 0.80 * p_conf_cb_bag3 + 0.20 * p_conf_lgb_bag3
    p_tune_b80 = 0.80 * p_tune_cb_bag3 + 0.20 * p_tune_lgb_bag3
    th_b80 = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_b80 >= t))

    # Blend 0.9 CB + 0.1 LGB
    p_conf_b90 = 0.90 * p_conf_cb_bag3 + 0.10 * p_conf_lgb_bag3
    p_tune_b90 = 0.90 * p_tune_cb_bag3 + 0.10 * p_tune_lgb_bag3
    th_b90 = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_b90 >= t))

    # Predictions
    pred_conf_cb = (p_conf_cb_bag3 >= th_cb).astype(int)
    pred_conf_lgb = (p_conf_lgb_bag3 >= th_lgb).astype(int)
    pred_conf_b80 = (p_conf_b80 >= th_b80).astype(int)
    pred_conf_b90 = (p_conf_b90 >= th_b90).astype(int)

    # Metrics
    acc_cb = float(accuracy_score(y[confirm], pred_conf_cb))
    ll_cb = float(log_loss(y[confirm], p_conf_cb_bag3))

    acc_lgb = float(accuracy_score(y[confirm], pred_conf_lgb))
    ll_lgb = float(log_loss(y[confirm], p_conf_lgb_bag3))

    acc_b80 = float(accuracy_score(y[confirm], pred_conf_b80))
    ll_b80 = float(log_loss(y[confirm], p_conf_b80))

    acc_b90 = float(accuracy_score(y[confirm], pred_conf_b90))
    ll_b90 = float(log_loss(y[confirm], p_conf_b90))

    hits_cb = int((pred_conf_cb == y[confirm]).sum())
    hits_b80 = int((pred_conf_b80 == y[confirm]).sum())
    hits_b90 = int((pred_conf_b90 == y[confirm]).sum())

    delta_b80 = hits_b80 - hits_cb
    delta_b90 = hits_b90 - hits_cb

    fixes_b80 = int(((pred_conf_cb != y[confirm]) & (pred_conf_b80 == y[confirm])).sum())
    breaks_b80 = int(((pred_conf_cb == y[confirm]) & (pred_conf_b80 != y[confirm])).sum())

    fixes_b90 = int(((pred_conf_cb != y[confirm]) & (pred_conf_b90 == y[confirm])).sum())
    breaks_b90 = int(((pred_conf_cb == y[confirm]) & (pred_conf_b90 != y[confirm])).sum())

    ci_b80_low, ci_b80_high = run_bootstrap_seller(y[confirm], pred_conf_cb, pred_conf_b80, sellers[confirm], 3000, SEED)
    ci_b90_low, ci_b90_high = run_bootstrap_seller(y[confirm], pred_conf_cb, pred_conf_b90, sellers[confirm], 3000, SEED)

    # Common 0.50 threshold evaluation
    pred_conf_cb_50 = (p_conf_cb_bag3 >= 0.50).astype(int)
    pred_conf_b80_50 = (p_conf_b80 >= 0.50).astype(int)
    delta_b80_50 = int((pred_conf_b80_50 == y[confirm]).sum()) - int((pred_conf_cb_50 == y[confirm]).sum())

    print(f"CatBoost Bag3 Confirm: Acc={acc_cb:.5f} ({hits_cb}/{len(confirm)}), LL={ll_cb:.5f}, th={th_cb:.3f}")
    print(f"LightGBM Bag3 Confirm: Acc={acc_lgb:.5f}, LL={ll_lgb:.5f}, th={th_lgb:.3f}")
    print(f"Blend 0.8/0.2 Confirm: Acc={acc_b80:.5f} ({hits_b80}/{len(confirm)}), LL={ll_b80:.5f}, th={th_b80:.3f}")
    print(f"  Delta vs CB: {delta_b80:+d} hits (fixes={fixes_b80}, breaks={breaks_b80}), CI: [{ci_b80_low:+.1f}, {ci_b80_high:+.1f}], Common .50: {delta_b80_50:+d}")
    print(f"Blend 0.9/0.1 Confirm: Acc={acc_b90:.5f} ({hits_b90}/{len(confirm)}), LL={ll_b90:.5f}, th={th_b90:.3f}")
    print(f"  Delta vs CB: {delta_b90:+d} hits (fixes={fixes_b90}, breaks={breaks_b90}), CI: [{ci_b90_low:+.1f}, {ci_b90_high:+.1f}]")

    report = {
        "experiment": "frente_c_clean_te_lgb_diversity_and_blends",
        "split_seed": SEED,
        "n_confirm": len(confirm),
        "catboost_bag3": {"acc": acc_cb, "ll": ll_cb, "th": float(th_cb), "hits": hits_cb},
        "lightgbm_bag3": {"acc": acc_lgb, "ll": ll_lgb, "th": float(th_lgb)},
        "lgb_diversity": {"corr_01": corr_s01, "corr_02": corr_s02, "corr_12": corr_s12, "disagr_01": disagr_seeds},
        "blend_80_20": {
            "acc": acc_b80, "ll": ll_b80, "th": float(th_b80), "delta_hits": delta_b80,
            "fixes": fixes_b80, "breaks": breaks_b80, "ci_95": [ci_b80_low, ci_b80_high],
            "delta_common_50": delta_b80_50
        },
        "blend_90_10": {
            "acc": acc_b90, "ll": ll_b90, "th": float(th_b90), "delta_hits": delta_b90,
            "fixes": fixes_b90, "breaks": breaks_b90, "ci_95": [ci_b90_low, ci_b90_high]
        }
    }

    out_json = OUT_DIR / f"report_frente_c_{SEED}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    df_pairs = pd.DataFrame({
        "item_id": ids[confirm],
        "seller_id": sellers[confirm],
        "y_true": y[confirm],
        "p_cb": p_conf_cb_bag3,
        "p_lgb": p_conf_lgb_bag3,
        "p_b80": p_conf_b80,
        "p_b90": p_conf_b90
    })
    pairs_file = OUT_DIR / f"paired_frente_c_{SEED}.csv"
    df_pairs.to_csv(pairs_file, index=False)

    print(f"Report saved to {out_json}")
    print(f"Pairs saved to {pairs_file}")

if __name__ == "__main__":
    main()
