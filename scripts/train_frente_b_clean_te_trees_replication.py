"""Frente B: Réplica Científica Rigurosa de Modelos de Árboles con TE Limpio.

Protocolo Ariel Directiva 2 & CX-0098..0100:
1. Partición por vendedor no contaminada (fit, tune, confirm con semilla fija 20261144).
2. Clean Target Encoding: prior y estadísticas calculados estrictamente dentro del subconjunto de entrenamiento de cada fold.
3. CatBoost Bag3: depth=6, lr=0.07, l2=6.0, 3 semillas controladas (20261144, 20262141, 20263138).
4. LightGBM Bag3: subsample_freq=1, bagging_fraction=0.8, colsample=0.8, 3 semillas.
5. Auditoría de diversidad de semillas: correlaciones pareadas y tasa de desacuerdos.
6. Ensamble predeclarado: selección de pesos y umbral exclusivamente en tune; evaluación única en confirm congelado.
7. Desacoplamiento explícito: efecto del ensamble evaluado a umbral optimizado en tune y a umbral común 0.50.
8. 3.000 remuestreos bootstrap agrupados por vendedor para intervalo de confianza al 95%.
"""

import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_deep_text_20261101 import split_testlike
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

OUT_DIR = ROOT / "results" / "frente_b_clean_te_replication_20261104"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261144

def make_clean_te(a, cat, y, fit, tune, confirm, sellers, m=10.0):
    te_cols = ["category_id", "listing_type_id", "state"]
    gkf = GroupKFold(n_splits=4)
    for col in te_cols:
        col_name = f"te_{col}"
        # Compute clean OOF TE on fit
        oof_vals = np.empty(len(fit), dtype=np.float32)
        for tr_rel, va_rel in gkf.split(fit, y[fit], sellers[fit]):
            tr_idx = fit[tr_rel]
            va_idx = fit[va_rel]
            # Prior strictly from train fold of fit
            fold_prior = float(y[tr_idx].mean())
            stats = pd.DataFrame({"cat": cat.iloc[tr_idx][col], "y": y[tr_idx]}).groupby("cat")["y"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + m * fold_prior) / (stats["count"] + m)
            oof_vals[va_rel] = cat.iloc[va_idx][col].map(smooth).fillna(fold_prior).astype(np.float32)
        a.loc[fit, col_name] = oof_vals

        # For tune and confirm, use global prior from full fit
        fit_prior = float(y[fit].mean())
        stats_fit = pd.DataFrame({"cat": cat.iloc[fit][col], "y": y[fit]}).groupby("cat")["y"].agg(["count", "mean"])
        smooth_fit = (stats_fit["count"] * stats_fit["mean"] + m * fit_prior) / (stats_fit["count"] + m)
        a.loc[tune, col_name] = cat.iloc[tune][col].map(smooth_fit).fillna(fit_prior).astype(np.float32)
        a.loc[confirm, col_name] = cat.iloc[confirm][col].map(smooth_fit).fillna(fit_prior).astype(np.float32)

def generate_clean_text_oof(combos_raw, titles_raw, y, fit, tune, confirm, sellers, C=3.0):
    vec_c = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    vec_t = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)

    X_c_fit = vec_c.fit_transform([combos_raw[i] for i in fit])
    X_t_fit = vec_t.fit_transform([titles_raw[i] for i in fit])
    import scipy.sparse as sp
    X_fit = sp.hstack([X_c_fit, X_t_fit], format="csr")

    X_c_tune = vec_c.transform([combos_raw[i] for i in tune])
    X_t_tune = vec_t.transform([titles_raw[i] for i in tune])
    X_tune = sp.hstack([X_c_tune, X_t_tune], format="csr")

    X_c_conf = vec_c.transform([combos_raw[i] for i in confirm])
    X_t_conf = vec_t.transform([titles_raw[i] for i in confirm])
    X_conf = sp.hstack([X_c_conf, X_t_conf], format="csr")

    oof = np.empty(len(fit), dtype=np.float32)
    gkf = GroupKFold(n_splits=4)
    for tr, va in gkf.split(fit, y[fit], sellers[fit]):
        lr = LogisticRegression(C=C, max_iter=200, solver="liblinear", random_state=SEED)
        lr.fit(X_fit[tr], y[fit[tr]])
        oof[va] = lr.predict_proba(X_fit[va])[:, 1]

    lr_full = LogisticRegression(C=C, max_iter=200, solver="liblinear", random_state=SEED)
    lr_full.fit(X_fit, y[fit])
    pt = lr_full.predict_proba(X_tune)[:, 1]
    pc = lr_full.predict_proba(X_conf)[:, 1]
    return oof, pt, pc

def bootstrap_seller_delta(y_true, pred_base, pred_challenger, sellers, n_boot=3000, seed=20261144):
    rng = np.random.RandomState(seed)
    unique_sellers = np.unique(sellers)
    n_sellers = len(unique_sellers)
    seller_to_indices = {s: np.where(sellers == s)[0] for s in unique_sellers}

    deltas = np.empty(n_boot, dtype=np.float32)
    for b in range(n_boot):
        sample_sellers = rng.choice(unique_sellers, size=n_sellers, replace=True)
        idx_sample = np.concatenate([seller_to_indices[s] for s in sample_sellers])
        y_b = y_true[idx_sample]
        hits_base = (pred_base[idx_sample] == y_b).sum()
        hits_chall = (pred_challenger[idx_sample] == y_b).sum()
        deltas[b] = hits_chall - hits_base

    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

def main():
    print("=== FRENTE B: RIGOROUS TREES REPLICATION (CLEAN TE & DIVERSITY) ===", flush=True)
    t0 = time.time()

    # 1. Load Data
    print("Loading 70k training dataset...", flush=True)
    rows = []
    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    assert len(rows) == 70000

    y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    fit, tune, confirm = split_testlike(sellers, SEED)

    print(f"Split sizes: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}")

    # 2. Extract Tabular Features & Clean TE
    print("Extracting features with strictly fold-specific TE...", flush=True)
    a, cat, titles_raw, _, combos_raw = extract_deep_features(rows)
    make_clean_te(a, cat, y, fit, tune, confirm, sellers)

    oof_txt, pt_txt, pc_txt = generate_clean_text_oof(combos_raw, titles_raw, y, fit, tune, confirm, sellers)
    a["txt_sparse_p"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof_txt, pt_txt, pc_txt)):
        a.loc[idx, "txt_sparse_p"] = pred

    a["seller_id_cat"] = cat.seller_id.astype(str)
    cat_cols = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]
    feature_cols = [c for c in a.columns if c not in ["seller_id_cat"] or c in cat_cols]

    # LightGBM dataset copy with categorical types
    a_lgb = a[feature_cols].copy()
    for col in cat_cols:
        a_lgb[col] = a_lgb[col].astype("category")

    cb_seeds = [SEED, SEED + 997, SEED + 1994]
    grid_th = np.arange(0.35, 0.651, 0.005)

    # 3. Train CatBoost Bag3
    print("\n--- Training CatBoost Bag3 (3 seeds, 4 threads) ---", flush=True)
    cb_tune_probs = []
    cb_conf_probs = []
    for s_idx, s in enumerate(cb_seeds):
        t_cb0 = time.time()
        cb = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                random_seed=s, thread_count=4, verbose=False, loss_function="Logloss")
        cb.fit(a.loc[fit, feature_cols], y[fit], cat_features=cat_cols,
               eval_set=(a.loc[tune, feature_cols], y[tune]), early_stopping_rounds=80, use_best_model=True)
        p_tu = cb.predict_proba(a.loc[tune, feature_cols])[:, 1]
        p_co = cb.predict_proba(a.loc[confirm, feature_cols])[:, 1]
        cb_tune_probs.append(p_tu)
        cb_conf_probs.append(p_co)
        print(f"CatBoost seed {s} fitted in {time.time()-t_cb0:.1f}s. Tune Acc: {accuracy_score(y[tune], p_tu >= 0.5):.5f}")

    p_tune_cb_bag3 = np.mean(cb_tune_probs, axis=0)
    p_conf_cb_bag3 = np.mean(cb_conf_probs, axis=0)
    th_cb = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_cb_bag3 >= t))
    print(f"CatBoost Bag3 Tune: Acc={accuracy_score(y[tune], p_tune_cb_bag3 >= th_cb):.5f}, th={th_cb:.3f}, LL={log_loss(y[tune], p_tune_cb_bag3):.5f}")

    # 4. Train LightGBM Bag3 with Effective Subsampling
    print("\n--- Training LightGBM Bag3 (subsample_freq=1, bagging_fraction=0.8, 4 threads) ---", flush=True)
    lgb_tune_probs = []
    lgb_conf_probs = []
    d_fit = lgb.Dataset(a_lgb.loc[fit], label=y[fit])
    d_tune = lgb.Dataset(a_lgb.loc[tune], label=y[tune], reference=d_fit)

    for s_idx, s in enumerate(cb_seeds):
        t_lgb0 = time.time()
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 45,
            "max_depth": 7,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "subsample_freq": 1,
            "seed": s,
            "num_threads": 4,
            "verbose": -1
        }
        m_lgb = lgb.train(
            params,
            d_fit,
            num_boost_round=900,
            valid_sets=[d_tune],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
        )
        p_tu_lgb = m_lgb.predict(a_lgb.loc[tune])
        p_co_lgb = m_lgb.predict(a_lgb.loc[confirm])
        lgb_tune_probs.append(p_tu_lgb)
        lgb_conf_probs.append(p_co_lgb)
        print(f"LightGBM seed {s} fitted in {time.time()-t_lgb0:.1f}s. Tune Acc: {accuracy_score(y[tune], p_tu_lgb >= 0.5):.5f}")

    p_tune_lgb_bag3 = np.mean(lgb_tune_probs, axis=0)
    p_conf_lgb_bag3 = np.mean(lgb_conf_probs, axis=0)
    th_lgb = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_lgb_bag3 >= t))

    # Diversity Audit Between Seeds
    corr_s01 = float(np.corrcoef(lgb_conf_probs[0], lgb_conf_probs[1])[0, 1])
    corr_s02 = float(np.corrcoef(lgb_conf_probs[0], lgb_conf_probs[2])[0, 1])
    corr_s12 = float(np.corrcoef(lgb_conf_probs[1], lgb_conf_probs[2])[0, 1])
    disagr_seeds = int(((lgb_conf_probs[0] >= 0.5) != (lgb_conf_probs[1] >= 0.5)).sum())
    print(f"LGB Seed Correlations: ({corr_s01:.4f}, {corr_s02:.4f}, {corr_s12:.4f}), Disagreements s0 vs s1: {disagr_seeds} ({disagr_seeds/len(confirm)*100:.2f}%)")

    # 5. Predeclared Blend (80% CB + 20% LGB)
    print("\n=== EVALUATING PREDECLARED BLENDS ON FROZEN CONFIRM ===", flush=True)
    p_tune_b80 = 0.80 * p_tune_cb_bag3 + 0.20 * p_tune_lgb_bag3
    p_conf_b80 = 0.80 * p_conf_cb_bag3 + 0.20 * p_conf_lgb_bag3
    th_b80 = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_b80 >= t))

    p_tune_b90 = 0.90 * p_tune_cb_bag3 + 0.10 * p_tune_lgb_bag3
    p_conf_b90 = 0.90 * p_conf_cb_bag3 + 0.10 * p_conf_lgb_bag3
    th_b90 = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_b90 >= t))

    # Evaluation on Frozen Confirm
    pred_cb = (p_conf_cb_bag3 >= th_cb).astype(int)
    pred_lgb = (p_conf_lgb_bag3 >= th_lgb).astype(int)
    pred_b80 = (p_conf_b80 >= th_b80).astype(int)
    pred_b90 = (p_conf_b90 >= th_b90).astype(int)

    acc_cb = float(accuracy_score(y[confirm], pred_cb))
    ll_cb = float(log_loss(y[confirm], p_conf_cb_bag3))
    hits_cb = int((pred_cb == y[confirm]).sum())

    acc_b80 = float(accuracy_score(y[confirm], pred_b80))
    ll_b80 = float(log_loss(y[confirm], p_conf_b80))
    hits_b80 = int((pred_b80 == y[confirm]).sum())
    delta_b80 = hits_b80 - hits_cb

    fixes_b80 = int(((pred_cb != y[confirm]) & (pred_b80 == y[confirm])).sum())
    breaks_b80 = int(((pred_cb == y[confirm]) & (pred_b80 != y[confirm])).sum())
    ci_b80_low, ci_b80_high = bootstrap_seller_delta(y[confirm], pred_cb, pred_b80, sellers[confirm], 3000, SEED)

    # Common threshold 0.50 evaluation (pure model comparison)
    pred_cb_50 = (p_conf_cb_bag3 >= 0.50).astype(int)
    pred_b80_50 = (p_conf_b80 >= 0.50).astype(int)
    hits_cb_50 = int((pred_cb_50 == y[confirm]).sum())
    hits_b80_50 = int((pred_b80_50 == y[confirm]).sum())
    delta_common_50 = hits_b80_50 - hits_cb_50

    print(f"CatBoost Bag3 Confirm: Acc={acc_cb:.5f} ({hits_cb}/{len(confirm)}), LL={ll_cb:.5f}, th={th_cb:.3f}")
    print(f"Blend 80/20 Confirm:   Acc={acc_b80:.5f} ({hits_b80}/{len(confirm)}), LL={ll_b80:.5f}, th={th_b80:.3f}")
    print(f"  Delta vs CB: {delta_b80:+d} hits (fixes={fixes_b80}, breaks={breaks_b80}), 95% CI: [{ci_b80_low:+.1f}, {ci_b80_high:+.1f}]")
    print(f"  Pure Model Delta (Common 0.50): {delta_common_50:+d} hits")

    # 6. Save Paired Predictions and Report
    df_paired = pd.DataFrame({
        "confirm_idx": range(len(confirm)),
        "seller_id": sellers[confirm],
        "y_true": y[confirm],
        "p_cb": p_conf_cb_bag3,
        "p_lgb": p_conf_lgb_bag3,
        "p_blend80": p_conf_b80,
        "pred_cb": pred_cb,
        "pred_blend80": pred_b80
    })
    df_paired.to_csv(OUT_DIR / "paired_confirm_predictions_frente_b.csv", index=False)

    report_trees = {
        "experiment": "frente_b_clean_te_trees_replication",
        "split_seed": SEED,
        "split_sizes": {"fit": len(fit), "tune": len(tune), "confirm": len(confirm)},
        "duration_seconds": float(time.time() - t0),
        "catboost_bag3": {
            "seeds": cb_seeds,
            "accuracy": acc_cb,
            "log_loss": ll_cb,
            "hits": hits_cb,
            "optimal_threshold": float(th_cb)
        },
        "lightgbm_bag3": {
            "seeds": cb_seeds,
            "bagging_params": {"subsample_freq": 1, "bagging_fraction": 0.8, "feature_fraction": 0.8},
            "seed_diversity": {
                "correlations": [corr_s01, corr_s02, corr_s12],
                "disagreements_s0_s1": disagr_seeds
            }
        },
        "blend_80_20": {
            "accuracy": acc_b80,
            "log_loss": ll_b80,
            "hits": hits_b80,
            "delta_hits_tuned_th": delta_b80,
            "delta_hits_common_50": delta_common_50,
            "fixes": fixes_b80,
            "breaks": breaks_b80,
            "ci_95_seller_bootstrap": [ci_b80_low, ci_b80_high]
        }
    }
    with open(OUT_DIR / "report_frente_b_trees_replication.json", "w", encoding="utf-8") as f:
        json.dump(report_trees, f, indent=2)

    print(f"\nFrente B completed in {time.time()-t0:.1f}s! Artifacts saved to {OUT_DIR.name}")

if __name__ == "__main__":
    main()
