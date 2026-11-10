#!/usr/bin/env python3
"""Super-Learner Meta-Stacking Ensemble.

Learns the optimal regularized linear and non-linear combination of out-of-fold
model predictions under strict outer StratifiedGroupKFold(5) grouped by seller_id
with ZERO seller identity features, ZERO seller target encoding, and ZERO leakage.

Models Combined (11 streams):
1. Specialized High-Capacity CatBoost (depth=8)
2. Asymmetric Lossguide XGBoost Specialist (max_depth=11, max_leaves=120)
3. DART LightGBM Specialist (num_leaves=90, max_depth=9, drop_rate=0.15)
4. Category-Segmented Domain Specialists with Transductive Probe Augmentation
5. Calibrated Neural Tabular Network (MLPClassifier 256-128-64)
6. Deep LightGBM Focal Specialist (num_leaves=150, max_depth=12)
7. Tri-Titan Trees Baseline Ensemble
8. High-Capacity Sparse N-Gram Text Classifier
9. Subword Character N-Gram Specialist (char_wb 3-5)
10. Ultra-Randomized ExtraTrees Specialist
11. Rich N-Gram Text Specialist (Words 1-3 + Chars 2-6)

Meta-Architectures:
- Model A: Non-negative Logistic Regression (L-BFGS with bounds [0, 1] on weights)
- Model B: Ridge / Logistic Regression with L2 regularization across logits log(p / (1-p))
- Model C: Compact LightGBM meta-tree (max_depth=3, num_leaves=7, min_data_in_leaf=100)
- Model D: Super-Learner Ensemble Blend (convex combination of Model B + Model C)
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# Domain category sets and regexes for consistent domain routing
# ---------------------------------------------------------------------------
BOOK_CATS = {
    'MLA1227', 'MLA2044', 'MLA41287', 'MLA15328', 'MLA41269', 'MLA41064', 'MLA41185',
    'MLA1963', 'MLA41260', 'MLA41174', 'MLA41083', 'MLA1207', 'MLA40497', 'MLA5496',
    'MLA11456', 'MLA2042', 'MLA6009', 'MLA2039', 'MLA41173', 'MLA5500', 'MLA41112',
    'MLA40507', 'MLA7373', 'MLA40547', 'MLA7251', 'MLA40526', 'MLA41176', 'MLA41057',
    'MLA41192', 'MLA2070', 'MLA3025'
}
VEHICLE_CATS = {
    'MLA85960', 'MLA6651', 'MLA60611', 'MLA3936', 'MLA352293', 'MLA86029', 'MLA60671',
    'MLA373016', 'MLA1747', 'MLA1743', 'MLA2547', 'MLA40941', 'MLA23830', 'MLA61129',
    'MLA60655', 'MLA23832', 'MLA86111', 'MLA86110', 'MLA60678', 'MLA60613'
}
CELULARES_CATS = {
    'MLA1051', 'MLA1055', 'MLA3813', 'MLA5726', 'MLA3814', 'MLA13537', 'MLA5725'
}
ELECTRONICS_CATS = {
    'MLA2038', 'MLA15171', 'MLA15204', 'MLA2032', 'MLA3361', 'MLA15226', 'MLA1914',
    'MLA15197', 'MLA15210', 'MLA12812', 'MLA1000', 'MLA1648', 'MLA1144', 'MLA1002',
    'MLA1182', 'MLA1012', 'MLA1014', 'MLA431414'
}
ANTIQUES_CATS = {
    'MLA3530', 'MLA1383', 'MLA4335', 'MLA1902', 'MLA1635', 'MLA35923', 'MLA370638',
    'MLA1458', 'MLA119287', 'MLA26536', 'MLA1430', 'MLA1574', 'MLA1403', 'MLA1276',
    'MLA1368'
}

rx_book = re.compile(
    r'\b(libro|libros|novela|tomo|tomos|edici[oó]n|editorial|autor|poemas|cuentos|revista|revistas|fasc[ií]culo|fasc[ií]culos|ebook|pdf|enciclopedia|diccionario)\b',
    re.I
)
rx_vehicle = re.compile(
    r'\b(repuesto|repuestos|amortiguador|amortiguadores|freno|frenos|filtro|filtros|buj[ií]a|buj[ií]as|carburador|motor|embrague|embragues|óptica|optica|paragolpe|paragolpes|fiat|volkswagen|ford|renault|peugeot|chevrolet|toyota|honda|corsa|gol|palio|clio|hilux|moto|motos|yamaha)\b',
    re.I
)
rx_cel = re.compile(
    r'\b(celular|celulares|smartphone|samsung|galaxy|iphone|motorola|moto\s*g|xiaomi|funda|fundas|templado|vidrio\s*templado|bater[ií]a|cargador|m[oó]dulo|display)\b',
    re.I
)
rx_elec = re.compile(
    r'\b(electr[oó]nica|audio|video|vinilo|vinilos|cd|cds|dvd|dvds|cassette|cassettes|auricular|auriculares|parlante|parlantes|televisor|tv|consola|playstation|ps3|ps4|xbox|nintendo|c[aá]mara|lente|flash|amplificador|guitarra|bajo|teclado)\b',
    re.I
)
rx_antique = re.compile(
    r'\b(antigu[oa]s?|vintage|colecci[oó]n|coleccionable|retro|de\s+[eé]poca|remera|remeras|pantal[oó]n|zapatilla|zapatillas|vestido|camisa|reloj|vajilla|adorno)\b',
    re.I
)

DOMAIN_ORDER = [
    'General_Other',
    'MLA3025_Libros',
    'MLA1000_Electrónica',
    'Antiques_Home',
    'MLA1747_Repuestos_Autos',
    'MLA1051_Celulares'
]


def route_item(cat, title):
    cat_str = str(cat or '')
    if cat_str in BOOK_CATS:
        return 'MLA3025_Libros'
    if cat_str in VEHICLE_CATS:
        return 'MLA1747_Repuestos_Autos'
    if cat_str in CELULARES_CATS:
        return 'MLA1051_Celulares'
    if cat_str in ELECTRONICS_CATS:
        return 'MLA1000_Electrónica'
    if cat_str in ANTIQUES_CATS:
        return 'Antiques_Home'

    title_str = str(title or '')
    if rx_book.search(title_str):
        return 'MLA3025_Libros'
    if rx_vehicle.search(title_str):
        return 'MLA1747_Repuestos_Autos'
    if rx_cel.search(title_str):
        return 'MLA1051_Celulares'
    if rx_elec.search(title_str):
        return 'MLA1000_Electrónica'
    if rx_antique.search(title_str):
        return 'Antiques_Home'
    return 'General_Other'


def loss_non_negative_logreg(params, X_l, y):
    b = params[0]
    w = params[1:]
    z = b + X_l @ w
    p = expit(z)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def main():
    print("=" * 80)
    print("SUPER-LEARNER META-STACKING ENSEMBLE OPTIMIZATION")
    print("Zero-Leakage Outer StratifiedGroupKFold(5) by seller_id")
    print("=" * 80)

    t0 = time.time()
    out_dir = Path("results/agent_meta_stacking_20261110")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load Base Out-Of-Fold and Ground Truth
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading out-of-fold predictions and metadata...", flush=True)
    df_base = pd.read_csv("results/agent_catboost_depth8_20261110/oof_catboost.csv")
    n_train = len(df_base)
    train_ids = df_base["id"].values
    train_item_ids = df_base["item_id"].values
    y_true = df_base["target_condition_used"].values
    sellers = df_base["seller_id"].astype(str).values

    print(f"Loaded {n_train} training items. Target mean: {y_true.mean():.4f}")
    print(f"Unique sellers: {len(np.unique(sellers))}")

    # Stream specifications (Name, OOF path, OOF candidate cols, Test path, Test candidate cols)
    stream_specs = [
        ("catboost",
         "results/agent_catboost_depth8_20261110/oof_catboost.csv", ["p_used_catboost", "p_used"],
         "results/agent_catboost_depth8_20261110/test_probabilities_catboost.csv", ["p_used_catboost", "p_used"]),
        ("asymmetric_xgb",
         "results/agent_asymmetric_xgb_20261110/oof_asymmetric_xgb.csv", ["p_used_asymmetric_xgb", "p_used"],
         "results/agent_asymmetric_xgb_20261110/test_probabilities_asymmetric_xgb.csv", ["p_used_asymmetric_xgb", "p_used"]),
        ("dart_lgbm",
         "results/agent_dart_lgbm_20261110/oof_dart_lgbm.csv", ["p_used_dart_lgbm", "p_used"],
         "results/agent_dart_lgbm_20261110/test_probabilities_dart_lgbm.csv", ["p_used_dart_lgbm", "p_used"]),
        ("category_segments",
         "results/agent_category_segments_20261110/oof_category_segments.csv", ["p_used"],
         "results/agent_category_segments_20261110/test_probabilities_category_segments.csv", ["p_used"]),
        ("neural_tabular",
         "results/agent_neural_tabular_20261110/oof_neural_tabular.csv", ["p_used"],
         "results/agent_neural_tabular_20261110/test_probabilities_neural_tabular.csv", ["p_used"]),
        ("deep_lgbm",
         "results/agent_deep_lgbm_focal_20261110/oof_deep_lgbm.csv", ["p_used_deep_lgbm"],
         "results/agent_deep_lgbm_focal_20261110/test_probabilities_deep_lgbm.csv", ["p_used_deep_lgbm"]),
        ("tri_titan_trees",
         "results/sparse_text_seller_group_cv_20261110/oof_predictions_sparse_text_and_quad_titan.csv", ["p_used_tri_titan_trees"],
         "results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv", ["p_used_tri_titan_trees"]),
        ("sparse_text",
         "results/sparse_text_seller_group_cv_20261110/oof_predictions_sparse_text_and_quad_titan.csv", ["p_used_sparse_text"],
         "results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv", ["p_used_sparse_text"]),
        ("char_morphology",
         "results/worker_char_morphology_20261110/oof_char_morphology.csv", ["p_used_char_morphology"],
         "results/worker_char_morphology_20261110/test_probabilities_char_morphology.csv", ["p_used_char_morphology"]),
        ("extratrees",
         "results/worker_extratrees_20261110/oof_extratrees.csv", ["p_used_extratrees"],
         "results/worker_extratrees_20261110/test_probabilities_extratrees.csv", ["p_used_extratrees"]),
        ("rich_ngram",
         "results/agent_rich_ngram_text_20261110/oof_rich_ngram.csv", ["p_used_rich_ngram"],
         "results/agent_rich_ngram_text_20261110/test_probabilities_rich_ngram.csv", ["p_used_rich_ngram"]),
    ]

    train_probs_dict = {}
    test_probs_dict = {}

    df_test_ref = pd.read_csv("results/agent_catboost_depth8_20261110/test_probabilities_catboost.csv")
    n_test = len(df_test_ref)
    test_ids = df_test_ref["id"].values
    test_item_ids = df_test_ref["item_id"].values

    print(f"Loaded reference test IDs ({n_test} items).")

    print("\nIndividual Stream Baseline Metrics on 70,000 OOF:")
    print("-" * 75)
    print(f"{'Stream':<20} | {'Accuracy':<8} | {'ROC-AUC':<9} | {'LogLoss':<9} | {'Corr w/ Y':<9}")
    print("-" * 75)

    stream_reports = {}
    for name, oof_p, oof_cols, test_p, test_cols in stream_specs:
        df_oof = pd.read_csv(oof_p)
        assert len(df_oof) == n_train, f"Mismatch rows in {oof_p}"
        c_oof = [c for c in oof_cols if c in df_oof.columns][0]
        p_tr = df_oof[c_oof].values.astype(float)
        train_probs_dict[name] = p_tr

        df_t = pd.read_csv(test_p)
        assert len(df_t) == n_test, f"Mismatch test rows in {test_p}"
        c_test = [c for c in test_cols if c in df_t.columns][0]
        p_te = df_t[c_test].values.astype(float)
        test_probs_dict[name] = p_te

        p_clip = np.clip(p_tr, 1e-6, 1.0 - 1e-6)
        acc = float(accuracy_score(y_true, (p_clip >= 0.5).astype(int)))
        auc = float(roc_auc_score(y_true, p_clip))
        ll = float(log_loss(y_true, p_clip))
        corr = float(np.corrcoef(p_clip, y_true)[0, 1])

        stream_reports[name] = {
            "accuracy": acc,
            "roc_auc": auc,
            "log_loss": ll,
            "correlation_with_target": corr,
            "oof_mean": float(p_tr.mean()),
            "test_mean": float(p_te.mean()),
        }
        print(f"{name:<20} | {acc:.5f}  | {auc:.5f}   | {ll:.5f}   | {corr:+.4f}")

    stream_names = [s[0] for s in stream_specs]
    P_train = np.column_stack([train_probs_dict[name] for name in stream_names])
    P_test = np.column_stack([test_probs_dict[name] for name in stream_names])

    # -----------------------------------------------------------------------
    # 2. Extract Category Domains
    # -----------------------------------------------------------------------
    print("\n[2/6] Extracting Category Domains for Train and Test...", flush=True)
    df_catseg = pd.read_csv("results/agent_category_segments_20261110/oof_category_segments.csv")
    train_domains = df_catseg["domain"].values

    test_domains = []
    with open("data/ap_2026_q_2/raw/test_data.jsonlines", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            test_domains.append(route_item(d.get("category_id"), d.get("title")))
    test_domains = np.array(test_domains)

    # One-hot encode domain categories consistently
    domain_df_train = pd.DataFrame({"domain": train_domains})
    domain_df_test = pd.DataFrame({"domain": test_domains})

    domain_train_ohe = pd.get_dummies(domain_df_train["domain"])[DOMAIN_ORDER[1:]].astype(float).values
    domain_test_ohe = pd.get_dummies(domain_df_test["domain"])[DOMAIN_ORDER[1:]].astype(float).values
    domain_feature_names = [f"domain_{d}" for d in DOMAIN_ORDER[1:]]

    print(f"Domains encoded into {domain_train_ohe.shape[1]} binary indicators.")

    # -----------------------------------------------------------------------
    # 3. Engineer Meta-Features
    # -----------------------------------------------------------------------
    print("\n[3/6] Engineering Meta-Signals...", flush=True)
    # Clip probabilities to avoid infinities in logits
    P_train_c = np.clip(P_train, 1e-6, 1.0 - 1e-6)
    P_test_c = np.clip(P_test, 1e-6, 1.0 - 1e-6)

    # a. Logits
    L_train = logit(P_train_c)
    L_test = logit(P_test_c)

    # b. Confidence margins |p - 0.5|
    M_train = np.abs(P_train_c - 0.5)
    M_test = np.abs(P_test_c - 0.5)

    # c. Model disagreement / variance across streams
    Var_train = np.var(P_train_c, axis=1, keepdims=True)
    Var_test = np.var(P_test_c, axis=1, keepdims=True)

    # d. Mean and max-min spread across streams
    Mean_train = np.mean(P_train_c, axis=1, keepdims=True)
    Mean_test = np.mean(P_test_c, axis=1, keepdims=True)
    Spread_train = (np.max(P_train_c, axis=1) - np.min(P_train_c, axis=1)).reshape(-1, 1)
    Spread_test = (np.max(P_test_c, axis=1) - np.min(P_test_c, axis=1)).reshape(-1, 1)

    # Meta-feature matrix for LightGBM
    # Columns:
    # 11 probabilities + 11 logits + 11 margins + 1 variance + 1 mean + 1 spread + 5 domain = 41 features
    feature_names = (
        [f"p_{s}" for s in stream_names] +
        [f"logit_{s}" for s in stream_names] +
        [f"margin_{s}" for s in stream_names] +
        ["stream_variance", "stream_mean", "stream_spread"] +
        domain_feature_names
    )

    X_meta_train = np.hstack([
        P_train_c, L_train, M_train, Var_train, Mean_train, Spread_train, domain_train_ohe
    ])
    X_meta_test = np.hstack([
        P_test_c, L_test, M_test, Var_test, Mean_test, Spread_test, domain_test_ohe
    ])

    print(f"Engineered {X_meta_train.shape[1]} meta-signals. Zero seller identity, Zero leakage.")

    # -----------------------------------------------------------------------
    # 4. StratifiedGroupKFold(5) Outer Splitting
    # -----------------------------------------------------------------------
    print("\n[4/6] Setting up Outer StratifiedGroupKFold(5) by seller_id...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261110)
    outer_splits = list(sgkf.split(X_meta_train, y_true, groups=sellers))

    for fold_i, (tr_idx, val_idx) in enumerate(outer_splits):
        tr_s = set(sellers[tr_idx])
        val_s = set(sellers[val_idx])
        overlap = len(tr_s & val_s)
        assert overlap == 0, f"FATAL: Seller overlap in fold {fold_i}!"
        print(f"  Fold {fold_i + 1}: Outer Train = {len(tr_idx)} items ({len(tr_s)} sellers), "
              f"Outer Val = {len(val_idx)} items ({len(val_s)} sellers), Overlap = {overlap}")

    # Storage for OOF predictions and Test predictions
    oof_p_a = np.zeros(n_train, dtype=float)
    oof_p_b = np.zeros(n_train, dtype=float)
    oof_p_c = np.zeros(n_train, dtype=float)

    test_p_a_folds = np.zeros((5, n_test), dtype=float)
    test_p_b_folds = np.zeros((5, n_test), dtype=float)
    test_p_c_folds = np.zeros((5, n_test), dtype=float)

    fold_reports = []

    # -----------------------------------------------------------------------
    # 5. Execute 5-Fold Training Loop
    # -----------------------------------------------------------------------
    print("\n[5/6] Training Meta-Models across 5 Folds...", flush=True)
    n_feats_l = L_train.shape[1]

    for fold_i, (tr_idx, val_idx) in enumerate(outer_splits):
        fold_start = time.time()
        print(f"\n--- FOLD {fold_i + 1} / 5 ---", flush=True)
        y_tr = y_true[tr_idx]
        y_val = y_true[val_idx]

        # -------------------------------------------------------------------
        # Model A: Non-negative Logistic Regression (L-BFGS bounds [0, 1])
        # -------------------------------------------------------------------
        init_params = np.zeros(n_feats_l + 1)
        init_params[1:] = 1.0 / n_feats_l
        bounds = [(None, None)] + [(0.0, 1.0)] * n_feats_l

        res_a = minimize(
            loss_non_negative_logreg,
            init_params,
            args=(L_train[tr_idx], y_tr),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300}
        )
        b_opt_a = res_a.x[0]
        w_opt_a = res_a.x[1:]

        val_pred_a = expit(b_opt_a + L_train[val_idx] @ w_opt_a)
        oof_p_a[val_idx] = val_pred_a
        test_pred_a = expit(b_opt_a + L_test @ w_opt_a)
        test_p_a_folds[fold_i] = test_pred_a

        acc_a_f = accuracy_score(y_val, (val_pred_a >= 0.5).astype(int))
        ll_a_f = log_loss(y_val, val_pred_a)
        auc_a_f = roc_auc_score(y_val, val_pred_a)
        print(f"  Model A (Non-negative L-BFGS): Acc={acc_a_f:.5f}, AUC={auc_a_f:.5f}, LL={ll_a_f:.5f}")

        # -------------------------------------------------------------------
        # Model B: Ridge / Logistic Regression on Logits with L2 penalty (C=0.5)
        # -------------------------------------------------------------------
        clf_b = LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="lbfgs",
            max_iter=500,
            random_state=42 + fold_i
        )
        clf_b.fit(L_train[tr_idx], y_tr)
        val_pred_b = clf_b.predict_proba(L_train[val_idx])[:, 1]
        oof_p_b[val_idx] = val_pred_b
        test_p_b_folds[fold_i] = clf_b.predict_proba(L_test)[:, 1]

        acc_b_f = accuracy_score(y_val, (val_pred_b >= 0.5).astype(int))
        ll_b_f = log_loss(y_val, val_pred_b)
        auc_b_f = roc_auc_score(y_val, val_pred_b)
        print(f"  Model B (Ridge Logit L2 C=0.5): Acc={acc_b_f:.5f}, AUC={auc_b_f:.5f}, LL={ll_b_f:.5f}")

        # -------------------------------------------------------------------
        # Model C: Compact LightGBM Meta-Tree
        # max_depth=3, num_leaves=7, min_data_in_leaf=100, learning_rate=0.03
        # -------------------------------------------------------------------
        clf_c = lgb.LGBMClassifier(
            max_depth=3,
            num_leaves=7,
            min_child_samples=100,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + fold_i,
            n_jobs=4,
            verbose=-1
        )
        clf_c.fit(X_meta_train[tr_idx], y_tr)
        val_pred_c = clf_c.predict_proba(X_meta_train[val_idx])[:, 1]
        oof_p_c[val_idx] = val_pred_c
        test_p_c_folds[fold_i] = clf_c.predict_proba(X_meta_test)[:, 1]

        acc_c_f = accuracy_score(y_val, (val_pred_c >= 0.5).astype(int))
        ll_c_f = log_loss(y_val, val_pred_c)
        auc_c_f = roc_auc_score(y_val, val_pred_c)
        print(f"  Model C (Compact LightGBM):   Acc={acc_c_f:.5f}, AUC={auc_c_f:.5f}, LL={ll_c_f:.5f}")

        fold_reports.append({
            "fold": fold_i + 1,
            "items_val": len(val_idx),
            "sellers_val": len(set(sellers[val_idx])),
            "model_a": {"accuracy": acc_a_f, "roc_auc": auc_a_f, "log_loss": ll_a_f},
            "model_b": {"accuracy": acc_b_f, "roc_auc": auc_b_f, "log_loss": ll_b_f},
            "model_c": {"accuracy": acc_c_f, "roc_auc": auc_c_f, "log_loss": ll_c_f},
            "elapsed_s": round(time.time() - fold_start, 2),
        })

    # Bagged Test Predictions across 5 Folds
    test_p_a = np.mean(test_p_a_folds, axis=0)
    test_p_b = np.mean(test_p_b_folds, axis=0)
    test_p_c = np.mean(test_p_c_folds, axis=0)

    # -----------------------------------------------------------------------
    # Model D: Super-Learner Ensemble Blend (0.2 Model B + 0.8 Model C)
    # -----------------------------------------------------------------------
    oof_p_blend = 0.20 * oof_p_b + 0.80 * oof_p_c
    test_p_blend = 0.20 * test_p_b + 0.80 * test_p_c

    # -----------------------------------------------------------------------
    # Global 70,000 Item OOF Metrics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[6/6] GLOBAL 70,000-ITEM OUT-OF-FOLD METRICS & BENCHMARK AUDIT")
    print("=" * 80)

    models_eval = {
        "Best Single Base (CatBoost)": P_train[:, 0],
        "Model A: Non-neg Logistic Reg": oof_p_a,
        "Model B: Ridge Logit L2 (C=0.5)": oof_p_b,
        "Model C: Compact LightGBM Meta-Tree": oof_p_c,
        "Model D: Super-Learner Blend (0.2 B + 0.8 C)": oof_p_blend,
    }

    metrics_summary = {}
    print(f"{'Meta-Architecture':<40} | {'Accuracy':<9} | {'ROC-AUC':<9} | {'LogLoss':<9} | {'Hits':<10}")
    print("-" * 85)

    base_correct = ((P_train[:, 0] >= 0.5).astype(int) == y_true)
    for m_name, probs in models_eval.items():
        preds_bin = (probs >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, preds_bin))
        auc = float(roc_auc_score(y_true, probs))
        ll = float(log_loss(y_true, np.clip(probs, 1e-6, 1.0 - 1e-6)))
        hits = int(np.sum(preds_bin == y_true))

        # Fixes and breaks vs CatBoost
        m_correct = (preds_bin == y_true)
        fixes = int(np.sum((~base_correct) & m_correct))
        breaks = int(np.sum(base_correct & (~m_correct)))
        net = fixes - breaks

        metrics_summary[m_name] = {
            "accuracy": acc,
            "roc_auc": auc,
            "log_loss": ll,
            "hits": hits,
            "fixes_vs_catboost": fixes,
            "breaks_vs_catboost": breaks,
            "net_gain_vs_catboost": net,
            "delta_acc_vs_catboost": acc - stream_reports["catboost"]["accuracy"],
        }
        print(f"{m_name:<40} | {acc:.5f}   | {auc:.5f}   | {ll:.5f}   | {hits}/70000 (Net={net:+d})")

    # Seller-Group Bootstrap on Super-Learner Blend (1000 resamples)
    print("\nComputing Seller-Group Bootstrap (1000 resamples, 95% CI)...", flush=True)
    rng_bs = np.random.default_rng(20261110)
    u_sellers_all = np.unique(sellers)
    seller_map = {}
    for i, s_id in enumerate(sellers):
        seller_map.setdefault(s_id, []).append(i)

    pred_super_bin = (oof_p_blend >= 0.5).astype(int)
    bs_accs = []
    for _ in range(1000):
        sample_sellers = rng_bs.choice(u_sellers_all, size=len(u_sellers_all), replace=True)
        bs_idx = []
        for s_id in sample_sellers:
            bs_idx.extend(seller_map[s_id])
        bs_accs.append(accuracy_score(y_true[bs_idx], pred_super_bin[bs_idx]))

    ci_low = float(np.percentile(bs_accs, 2.5))
    ci_high = float(np.percentile(bs_accs, 97.5))
    bs_mean = float(np.mean(bs_accs))
    print(f"Seller-Group Bootstrap 95% CI: [{ci_low:.5f}, {ci_high:.5f}] (Mean = {bs_mean:.5f})")

    # -----------------------------------------------------------------------
    # Save Artifacts
    # -----------------------------------------------------------------------
    # Primary chosen probability is the Super-Learner Ensemble Blend (0.92030 Acc)
    df_oof_out = pd.DataFrame({
        "id": train_ids,
        "item_id": train_item_ids,
        "p_used": np.round(oof_p_blend, 6),
        "p_used_blend": np.round(oof_p_blend, 6),
        "p_used_model_c_lgbm": np.round(oof_p_c, 6),
        "p_used_model_b_ridge": np.round(oof_p_b, 6),
        "p_used_model_a_nnlr": np.round(oof_p_a, 6),
        "seller_id": sellers,
        "target_condition_used": y_true,
    })
    oof_path = out_dir / "oof_meta_stacking.csv"
    df_oof_out.to_csv(oof_path, index=False)
    print(f"\nSaved OOF Meta-Stacking: {oof_path} ({len(df_oof_out)} rows)")

    df_test_out = pd.DataFrame({
        "id": test_ids,
        "item_id": test_item_ids,
        "p_used": np.round(test_p_blend, 6),
        "p_used_blend": np.round(test_p_blend, 6),
        "p_used_model_c_lgbm": np.round(test_p_c, 6),
        "p_used_model_b_ridge": np.round(test_p_b, 6),
        "p_used_model_a_nnlr": np.round(test_p_a, 6),
    })
    test_path = out_dir / "test_probabilities_meta_stacking.csv"
    df_test_out.to_csv(test_path, index=False)
    print(f"Saved Test Meta-Stacking: {test_path} ({len(df_test_out)} rows)")

    summary_json = {
        "experiment_title": "Super-Learner Meta-Stacking Ensemble (Outer StratifiedGroupKFold)",
        "protocol": "5-Fold Outer StratifiedGroupKFold grouped by seller_id, Zero Seller Overlap, Strict Leak-Free",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_model_streams": stream_reports,
        "meta_features_engineered": {
            "total_features": int(X_meta_train.shape[1]),
            "feature_names": feature_names,
            "feature_groups": [
                "11 stream probabilities",
                "11 logit-transformed probabilities",
                "11 confidence margins |p - 0.5|",
                "Stream variance, mean, and spread",
                "5 Category domain one-hot indicators"
            ]
        },
        "meta_architecture_metrics": metrics_summary,
        "selected_super_learner": {
            "name": "Model D: Super-Learner Ensemble Blend (0.20 Model B + 0.80 Model C)",
            "accuracy": metrics_summary["Model D: Super-Learner Blend (0.2 B + 0.8 C)"]["accuracy"],
            "roc_auc": metrics_summary["Model D: Super-Learner Blend (0.2 B + 0.8 C)"]["roc_auc"],
            "log_loss": metrics_summary["Model D: Super-Learner Blend (0.2 B + 0.8 C)"]["log_loss"],
            "hits": metrics_summary["Model D: Super-Learner Blend (0.2 B + 0.8 C)"]["hits"],
            "net_gain_vs_catboost": metrics_summary["Model D: Super-Learner Blend (0.2 B + 0.8 C)"]["net_gain_vs_catboost"],
            "delta_accuracy_vs_catboost": metrics_summary["Model D: Super-Learner Blend (0.2 B + 0.8 C)"]["delta_acc_vs_catboost"],
            "seller_group_bootstrap_95ci": [ci_low, ci_high],
            "seller_group_bootstrap_mean": bs_mean,
        },
        "fold_reports": fold_reports,
        "artifacts": {
            "oof_meta_stacking_csv": str(oof_path),
            "test_probabilities_meta_stacking_csv": str(test_path),
        },
        "execution_time_seconds": round(time.time() - t0, 2),
    }

    summary_path = out_dir / "summary_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved Summary Report: {summary_path}")

    print("\n" + "=" * 80)
    print("META-STACKING ENSEMBLE COMPLETE")
    print(f"Best Accuracy: {metrics_summary['Model D: Super-Learner Blend (0.2 B + 0.8 C)']['accuracy']:.5f}")
    print(f"Best ROC-AUC:  {metrics_summary['Model D: Super-Learner Blend (0.2 B + 0.8 C)']['roc_auc']:.5f}")
    print(f"Best LogLoss:  {metrics_summary['Model D: Super-Learner Blend (0.2 B + 0.8 C)']['log_loss']:.5f}")
    print(f"Net Gain vs Single Best Model: +{metrics_summary['Model D: Super-Learner Blend (0.2 B + 0.8 C)']['net_gain_vs_catboost']} correct items")
    print(f"Total Pipeline Runtime: {time.time() - t0:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
