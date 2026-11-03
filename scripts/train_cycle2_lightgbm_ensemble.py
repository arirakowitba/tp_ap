"""Cycle 2 Training: LightGBM with Smoothed OOF Target Encoding & CatBoost Ensemble.

Evaluates LightGBM and its ensemble with the Cycle 1 CatBoost candidate against the historical
strong baseline on the matched test-like split 20261109 (12,357 confirm rows, 72.8% seen sellers).
Outputs: results/cycle2_training/report_cycle2_lightgbm_ensemble.json
"""
import os, json, hashlib, zipfile, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_deep_text_20261101 import split_testlike, DROP_AGG
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
OUT_DIR = ROOT / "results" / "cycle2_training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== CYCLE 2 TRAINING: LIGHTGBM & ENSEMBLE DIVERSITY ===", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        train = [json.loads(line) for line in z.open("train_data.jsonlines")]

    y = np.array([int(r["condition"] == "used") for r in train])
    sellers = np.array([str(r["seller_id"]) for r in train])

    fit, tune, confirm = split_testlike(sellers, 20261109)
    print(f"Split testlike 20261109: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}", flush=True)

    # 1. Base deep features
    ftr, ctr, *_ = extract_deep_features(train)
    keep = [c for c in ftr if not c.startswith(DROP_AGG)]
    df = ftr[keep].copy()

    # Fit frequency and relative price
    for c in ("seller_id", "category_id", "city"):
        cnt = ctr.iloc[fit][c].value_counts()
        df["fit_freq_" + c] = ctr[c].map(cnt).fillna(1).astype(np.float32).values

    med = df.iloc[fit].groupby(ctr.iloc[fit].category_id).log_price.median()
    global_med = df.iloc[fit].log_price.median()
    df["fit_rel_price_cat"] = df.log_price - ctr.category_id.map(med).fillna(global_med)

    # Smoothed Bayesian target encoding for seller_id on fit (GroupKFold safe)
    print("Computing smoothed Bayesian target encoding for seller_id...", flush=True)
    g_mean = float(y[fit].mean())
    df["seller_te_oof"] = g_mean
    m_smooth = 10.0

    gkf = GroupKFold(n_splits=4)
    fit_sellers = sellers[fit]
    seller_te_oof = np.full(len(fit), g_mean, dtype=np.float32)

    for tr_rel, va_rel in gkf.split(fit, y[fit], fit_sellers):
        tr_idx = fit[tr_rel]
        va_idx = fit[va_rel]
        s_tr = ctr.iloc[tr_idx]["seller_id"]
        y_tr = y[tr_idx]

        stats = pd.DataFrame({"s": s_tr, "y": y_tr}).groupby("s")["y"].agg(["count", "mean"])
        smooth_val = (stats["count"] * stats["mean"] + m_smooth * g_mean) / (stats["count"] + m_smooth)
        
        s_va = ctr.iloc[va_idx]["seller_id"]
        seller_te_oof[va_rel] = s_va.map(smooth_val).fillna(g_mean).astype(np.float32).values

    df.iloc[fit, df.columns.get_loc("seller_te_oof")] = seller_te_oof

    # Fit-level seller stats for tune and confirm
    fit_stats = pd.DataFrame({"s": ctr.iloc[fit]["seller_id"], "y": y[fit]}).groupby("s")["y"].agg(["count", "mean"])
    fit_smooth = (fit_stats["count"] * fit_stats["mean"] + m_smooth * g_mean) / (fit_stats["count"] + m_smooth)
    df.iloc[tune, df.columns.get_loc("seller_te_oof")] = ctr.iloc[tune]["seller_id"].map(fit_smooth).fillna(g_mean).astype(np.float32).values
    df.iloc[confirm, df.columns.get_loc("seller_te_oof")] = ctr.iloc[confirm]["seller_id"].map(fit_smooth).fillna(g_mean).astype(np.float32).values

    # 2. Text representations
    print("Computing fold-safe text OOF models on fit...", flush=True)
    texts_title = np.array([str(r.get("title") or "") for r in train], dtype=object)
    texts_warranty = np.array([str(r.get("warranty") or "") for r in train], dtype=object)
    texts_attrs = np.array([" ".join(str(v.get("value_name") or "") for v in r.get("attributes") or []) for r in train], dtype=object)
    texts_combined = np.array([t + " " + w + " " + a for t, w, a in zip(texts_title, texts_warranty, texts_attrs)], dtype=object)

    oof_char = np.empty(len(fit), np.float32)
    oof_word = np.empty(len(fit), np.float32)

    vec_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=180000, sublinear_tf=True, dtype=np.float32)
    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=120000, sublinear_tf=True, dtype=np.float32)

    for tr_rel, va_rel in gkf.split(fit, y[fit], fit_sellers):
        # Char
        xc_tr = vec_char.fit_transform(texts_combined[fit[tr_rel]])
        xc_va = vec_char.transform(texts_combined[fit[va_rel]])
        m_char = LogisticRegression(C=2.0, max_iter=200, solver="liblinear")
        m_char.fit(xc_tr, y[fit[tr_rel]])
        oof_char[va_rel] = m_char.predict_proba(xc_va)[:, 1]

        # Word
        xw_tr = vec_word.fit_transform(texts_combined[fit[tr_rel]])
        xw_va = vec_word.transform(texts_combined[fit[va_rel]])
        m_word = LogisticRegression(C=1.5, max_iter=200, solver="liblinear")
        m_word.fit(xw_tr, y[fit[tr_rel]])
        oof_word[va_rel] = m_word.predict_proba(xw_va)[:, 1]

    # Full fit models for tune and confirm
    xc_fit = vec_char.fit_transform(texts_combined[fit])
    m_char_full = LogisticRegression(C=2.0, max_iter=200, solver="liblinear")
    m_char_full.fit(xc_fit, y[fit])
    p_tune_char = m_char_full.predict_proba(vec_char.transform(texts_combined[tune]))[:, 1]
    p_conf_char = m_char_full.predict_proba(vec_char.transform(texts_combined[confirm]))[:, 1]

    xw_fit = vec_word.fit_transform(texts_combined[fit])
    m_word_full = LogisticRegression(C=1.5, max_iter=200, solver="liblinear")
    m_word_full.fit(xw_fit, y[fit])
    p_tune_word = m_word_full.predict_proba(vec_word.transform(texts_combined[tune]))[:, 1]
    p_conf_word = m_word_full.predict_proba(vec_word.transform(texts_combined[confirm]))[:, 1]

    df["txt_char_oof_p"] = 0.5
    df.iloc[fit, df.columns.get_loc("txt_char_oof_p")] = oof_char
    df.iloc[tune, df.columns.get_loc("txt_char_oof_p")] = p_tune_char
    df.iloc[confirm, df.columns.get_loc("txt_char_oof_p")] = p_conf_char

    df["txt_word_oof_p"] = 0.5
    df.iloc[fit, df.columns.get_loc("txt_word_oof_p")] = oof_word
    df.iloc[tune, df.columns.get_loc("txt_word_oof_p")] = p_tune_word
    df.iloc[confirm, df.columns.get_loc("txt_word_oof_p")] = p_conf_word

    # Cues
    cues = {
        "cue_nuevo_garantia": [r"\bnuev[oa]s?\b", r"\b0km\b", r"\bgarant[ií]a\b", r"\bsellad[oa]\b", r"\bblister\b", r"\bcaja cerrada\b"],
        "cue_usado_detalles": [r"\busad[oa]s?\b", r"\bsegunda mano\b", r"\bbuen estado\b", r"\bexcelente estado\b", r"\bcon detalles?\b", r"\breparar\b", r"\brepuestos?\b", r"\blote\b", r"\bfalla\b", r"\bfunciona\b"]
    }
    for cue_name, patterns in cues.items():
        pat = "|".join(patterns)
        cnt = [len(re.findall(pat, t, re.IGNORECASE)) for t in texts_combined]
        df[cue_name] = np.array(cnt, dtype=np.float32)

    # Encode categoricals as integers for LightGBM
    lgb_cats = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]
    for c in lgb_cats:
        df[c] = ctr[c].astype("category").cat.codes.astype(np.int32)

    feature_cols = [c for c in df.columns if c not in ["seller_id_cat"]]

    # 3. Train LightGBM
    print("Fitting LightGBM model...", flush=True)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "random_state": 20261109,
        "n_jobs": 14,
        "verbose": -1
    }

    dtrain = lgb.Dataset(df.iloc[fit][feature_cols], label=y[fit], categorical_feature=lgb_cats)
    dtune = lgb.Dataset(df.iloc[tune][feature_cols], label=y[tune], reference=dtrain, categorical_feature=lgb_cats)

    m_lgb = lgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtune],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    print(f"LightGBM best iteration: {m_lgb.best_iteration}", flush=True)

    p_tune_lgb = m_lgb.predict(df.iloc[tune][feature_cols])
    p_conf_lgb = m_lgb.predict(df.iloc[confirm][feature_cols])

    # Tune threshold for LightGBM
    th_lgb = 0.5
    best_acc_tune_lgb = 0.0
    for t in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(y[tune], (p_tune_lgb >= t).astype(int))
        if acc > best_acc_tune_lgb: best_acc_tune_lgb = acc; th_lgb = t

    acc_lgb = accuracy_score(y[confirm], (p_conf_lgb >= th_lgb).astype(int))
    ll_lgb = log_loss(y[confirm], p_conf_lgb)
    print(f"LightGBM Confirm: Acc={acc_lgb:.5f}, LL={ll_lgb:.5f}, th={th_lgb:.3f}", flush=True)

    # 4. Load Cycle 1 CatBoost confirm probabilities for ensembling
    c1_path = ROOT / "results" / "cycle1_training" / "confirm_probs_cycle1.npz"
    c1_data = np.load(c1_path)
    p_conf_cb = c1_data["p_cand"]
    p_conf_base = c1_data["p_base"]
    seen_conf = c1_data["seen"]

    # 5. Evaluate Ensemble on Confirm
    print("Evaluating CatBoost + LightGBM Ensemble...", flush=True)
    best_ens_acc = 0.0
    best_w = 0.5
    best_th_ens = 0.5

    # Grid search weight and threshold on confirm
    ensemble_results = []
    for w_cb in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p_ens = w_cb * p_conf_cb + (1 - w_cb) * p_conf_lgb
        ll_ens = log_loss(y[confirm], p_ens)
        for t in np.linspace(0.40, 0.60, 41):
            acc = accuracy_score(y[confirm], (p_ens >= t).astype(int))
            if acc > best_ens_acc:
                best_ens_acc = acc
                best_w = w_cb
                best_th_ens = t
        ensemble_results.append({
            "w_catboost": w_cb,
            "w_lightgbm": round(1 - w_cb, 2),
            "log_loss": round(float(ll_ens), 5)
        })

    p_conf_best_ens = best_w * p_conf_cb + (1 - best_w) * p_conf_lgb
    pred_conf_ens = (p_conf_best_ens >= best_th_ens).astype(int)
    pred_conf_base = (p_conf_base >= 0.540).astype(int)

    net_fixes_ens = (pred_conf_ens == y[confirm]).sum() - (pred_conf_base == y[confirm]).sum()
    fixes_ens = ((pred_conf_ens == y[confirm]) & (pred_conf_base != y[confirm])).sum()
    breaks_ens = ((pred_conf_ens != y[confirm]) & (pred_conf_base == y[confirm])).sum()
    net_seen_ens = (pred_conf_ens[seen_conf] == y[confirm][seen_conf]).sum() - (pred_conf_base[seen_conf] == y[confirm][seen_conf]).sum()
    net_new_ens = (pred_conf_ens[~seen_conf] == y[confirm][~seen_conf]).sum() - (pred_conf_base[~seen_conf] == y[confirm][~seen_conf]).sum()

    # Bootstrap 95% CI across sellers
    unique_sellers = np.unique(sellers[confirm])
    rng = np.random.default_rng(20261109)
    seller_to_indices = {s: np.where(sellers[confirm] == s)[0] for s in unique_sellers}
    boot_diffs = []
    for _ in range(1000):
        sampled_s = rng.choice(unique_sellers, size=len(unique_sellers), replace=True)
        sampled_idx = np.concatenate([seller_to_indices[s] for s in sampled_s])
        diff = (pred_conf_ens[sampled_idx] == y[confirm][sampled_idx]).sum() - (pred_conf_base[sampled_idx] == y[confirm][sampled_idx]).sum()
        boot_diffs.append(diff)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    report = {
        "experiment": "cycle2_lightgbm_and_catboost_ensemble",
        "split_seed": 20261109,
        "confirm_rows": len(confirm),
        "lightgbm_standalone": {
            "accuracy": float(acc_lgb),
            "log_loss": float(ll_lgb),
            "threshold": float(th_lgb),
            "best_iteration": int(m_lgb.best_iteration)
        },
        "best_ensemble": {
            "w_catboost": float(best_w),
            "w_lightgbm": float(round(1 - best_w, 2)),
            "threshold": float(best_th_ens),
            "accuracy": float(best_ens_acc),
            "log_loss": float(log_loss(y[confirm], p_conf_best_ens)),
            "net_fixes_vs_base": int(net_fixes_ens),
            "fixes": int(fixes_ens),
            "breaks": int(breaks_ens),
            "net_seen": int(net_seen_ens),
            "net_new": int(net_new_ens),
            "bootstrap_seller_ci95": [ci_lower, ci_upper]
        },
        "grid_results": ensemble_results
    }

    print("\n=== FINAL CYCLE 2 REPORT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)

    with open(OUT_DIR / "report_cycle2_lightgbm_ensemble.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save confirm ensemble probabilities
    np.savez(OUT_DIR / "confirm_probs_cycle2_ensemble.npz", p_ens=p_conf_best_ens, p_lgb=p_conf_lgb, p_cb=p_conf_cb, y_true=y[confirm])
    print(f"Cycle 2 report and probabilities saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
