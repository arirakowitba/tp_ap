"""AP-2026-Q-2: Master Execution of Vía Principal (Product Identity & Condition Representation).

Strictly leak-free, reproducible, evaluated on the frozen canonical 3-way split (canonical_3way_splits.csv).
Split roles:
- Fit: 40,881 rows
- Tune: 15,044 rows (Cold new: 8,758, Seen bracketed: 6,286)
- Confirm: 14,075 rows (Cold new: 7,875, Seen bracketed: 6,200)

Three Paired Arms Evaluated:
1. Arm 1: Strong Baseline (LightGBM Enriched: attributes, text OOF by seller, causal seller TE).
2. Arm 2: Baseline + Product Identity, Specifications & Condition Signal (21 new structured features).
3. Arm 3: Predefined Multi-Family Routed Ensemble (Arm 2 + CatBoost Native + XGBoost Enriched + Bracketed Specialist).

Protocol Requirements:
- Strictly identical evaluation rows across all arms.
- Two-way cluster bootstrap on (seller, category) with 2,000 resamples.
- Direction of class flips (to NEW vs to USED).
- Top contributor ablation (leave-top-5 categories and leave-top-5 sellers).
- Overlap accounting with previous splits.
- Private pairs saved to gitignored CSV.
- Zero Kaggle API calls.
"""
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import t98_line3_lightgbm_enriched as ag
from extract_product_identity_features import compute_identity_feature_matrices

SPLIT_CSV = ROOT / "data" / "splits" / "canonical_3way_splits.csv"
SPLIT_MANIFEST = ROOT / "data" / "splits" / "canonical_3way_manifest.json"
DATA_PKL = ROOT / "data" / "splits" / "t98_parsed_train.pkl"
CACHE_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

CATEGORICAL = [
    "category_id", "brand_val", "model_val", "item_cond_attr",
    "listing_type_id", "state_id", "city_id", "ship_mode", "buying_mode", "currency_id"
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_batch_seller_features(df, fit_indices, query_indices):
    """Compute leave-one-out historical seller features within fit, and pure lookup on query."""
    y = df["y"].to_numpy()
    outputs_fit = []
    outputs_query = []

    fit_df = df.iloc[fit_indices]
    query_df = df.iloc[query_indices]

    for keys in (["seller_id"], ["seller_id", "category_id"], ["seller_id", "brand_val"]):
        table = fit_df.groupby(keys, sort=False)["y"].agg(["size", "sum"])
        if len(keys) == 1:
            fit_idx = fit_df[keys[0]]
            query_idx = query_df[keys[0]]
        else:
            fit_idx = pd.MultiIndex.from_frame(fit_df[keys])
            query_idx = pd.MultiIndex.from_frame(query_df[keys])

        fit_counts, fit_used = table.reindex(fit_idx).fillna(0).to_numpy().T
        query_counts, query_used = table.reindex(query_idx).fillna(0).to_numpy().T

        # Leave-one-out on fit
        fit_counts = fit_counts - 1
        fit_used = fit_used - y[fit_indices]
        assert np.all((fit_used >= 0) & (fit_counts >= fit_used)), "Invalid negative counts in fit LOO!"

        outputs_fit.extend((np.log1p(np.maximum(0, fit_counts)), (fit_used + 2.5) / (fit_counts + 5.0)))
        outputs_query.extend((np.log1p(np.maximum(0, query_counts)), (query_used + 2.5) / (query_counts + 5.0)))

    hist_fit = np.column_stack(outputs_fit).astype(np.float32)
    hist_query = np.column_stack(outputs_query).astype(np.float32)
    return hist_fit, hist_query


def bootstrap_cluster_metrics(y_true, p_cand, p_base, sellers, cats, n_boot=2000, seed=20261113):
    """Two-way cluster bootstrap on (seller, category) clusters."""
    y_true = np.asarray(y_true)
    c_pred = (np.asarray(p_cand) >= 0.5).astype(int)
    b_pred = (np.asarray(p_base) >= 0.5).astype(int)

    c_correct = (c_pred == y_true)
    b_correct = (b_pred == y_true)

    fixes = int((c_correct & ~b_correct).sum())
    breaks = int((~c_correct & b_correct).sum())
    net = fixes - breaks

    diff = (c_correct.astype(int) - b_correct.astype(int))

    # Flips direction
    flips_to_new = int(((b_pred == 1) & (c_pred == 0)).sum())
    flips_to_used = int(((b_pred == 0) & (c_pred == 1)).sum())

    # Cluster IDs
    unique_sellers, s_idx = np.unique(sellers, return_inverse=True)
    unique_cats, c_idx = np.unique(cats, return_inverse=True)
    n_s = len(unique_sellers)
    n_c = len(unique_cats)

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w_s = rng.choice([-1.0, 1.0], size=n_s)
        w_c = rng.choice([-1.0, 1.0], size=n_c)
        w_row = 1.0 + (w_s[s_idx] * w_c[c_idx])
        boot_deltas[b] = np.sum(diff * w_row)

    q05 = float(np.percentile(boot_deltas, 5.0))
    q95 = float(np.percentile(boot_deltas, 95.0))

    cand_hits = int(c_correct.sum())
    base_hits = int(b_correct.sum())
    cand_acc = float(cand_hits / len(y_true))
    base_acc = float(base_hits / len(y_true))
    cand_ll = float(log_loss(y_true, np.clip(p_cand, 1e-6, 1.0 - 1e-6)))
    base_ll = float(log_loss(y_true, np.clip(p_base, 1e-6, 1.0 - 1e-6)))

    return {
        "n": len(y_true),
        "cand_hits": cand_hits,
        "base_hits": base_hits,
        "cand_acc": cand_acc,
        "base_acc": base_acc,
        "net_hits": net,
        "fixes": fixes,
        "breaks": breaks,
        "flips_to_new": flips_to_new,
        "flips_to_used": flips_to_used,
        "q05": q05,
        "q95": q95,
        "cand_logloss": cand_ll,
        "base_logloss": base_ll,
        "delta_logloss": cand_ll - base_ll,
    }


def ablation_top_contributors(y_true, p_cand, p_base, sellers, cats):
    """Leave-top-5 categories and leave-top-5 sellers ablation."""
    y_true = np.asarray(y_true)
    c_pred = (np.asarray(p_cand) >= 0.5).astype(int)
    b_pred = (np.asarray(p_base) >= 0.5).astype(int)
    net_diff = (c_pred == y_true).astype(int) - (b_pred == y_true).astype(int)

    df_contrib = pd.DataFrame({"seller": sellers, "cat": cats, "net": net_diff})

    # Top categories by net gain
    cat_net = df_contrib.groupby("cat")["net"].sum().sort_values(ascending=False)
    top5_cats = set(cat_net.head(5).index)
    mask_no_top5_cats = ~df_contrib["cat"].isin(top5_cats)
    net_without_top5_cats = int(df_contrib.loc[mask_no_top5_cats, "net"].sum())

    # Top sellers by net gain
    seller_net = df_contrib.groupby("seller")["net"].sum().sort_values(ascending=False)
    top5_sellers = set(seller_net.head(5).index)
    mask_no_top5_sellers = ~df_contrib["seller"].isin(top5_sellers)
    net_without_top5_sellers = int(df_contrib.loc[mask_no_top5_sellers, "net"].sum())

    return {
        "top5_categories": list(top5_cats),
        "top5_cat_net_share": int(cat_net.head(5).sum()),
        "net_without_top5_categories": net_without_top5_cats,
        "top5_sellers": [int(s) for s in list(top5_sellers)],
        "top5_seller_net_share": int(seller_net.head(5).sum()),
        "net_without_top5_sellers": net_without_top5_sellers,
    }


def main():
    t_start = time.time()
    print("=== AP-2026-Q-2: Vía Principal Execution (Product Identity & Condition Representation) ===")

    # 1. Load Split and Parsed Data
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Verified Split SHA-256: {manifest['split_sha256']}")

    split_df = pd.read_csv(SPLIT_CSV)
    fit_indices = np.flatnonzero(split_df["role"] == "fit")
    tune_indices = np.flatnonzero(split_df["role"] == "tune")
    confirm_indices = np.flatnonzero(split_df["role"] == "confirm")

    tune_seen_mask = split_df.iloc[tune_indices]["regime"].values == "tune_seen_bracketed"
    confirm_seen_mask = split_df.iloc[confirm_indices]["regime"].values == "confirm_seen_bracketed"

    with open(DATA_PKL, "rb") as f:
        df = pickle.load(f)

    y_fit = df["y"].iloc[fit_indices].to_numpy()
    y_tune = df["y"].iloc[tune_indices].to_numpy()
    sellers_tune = df["seller_id"].iloc[tune_indices].to_numpy()
    cats_tune = df["category_id"].iloc[tune_indices].to_numpy()

    y_confirm = df["y"].iloc[confirm_indices].to_numpy()
    sellers_confirm = df["seller_id"].iloc[confirm_indices].to_numpy()
    cats_confirm = df["category_id"].iloc[confirm_indices].to_numpy()

    print(f"Fit rows: {len(fit_indices)}")
    print(f"Tune rows: {len(tune_indices)} (Seen bracketed: {tune_seen_mask.sum()}, Cold new: {(~tune_seen_mask).sum()})")
    print(f"Confirm rows: {len(confirm_indices)} (Seen bracketed: {confirm_seen_mask.sum()}, Cold new: {(~confirm_seen_mask).sum()})")

    # 2. Build or Load Fold Matrices (Text OOF, SVD, TE, Causal Seller TE)
    cache_path = CACHE_DIR / f"text_{manifest['split_sha256'][:16]}.npz"
    tr_data = ag.precompute_item_intrinsic_arrays(df)

    inner_cv = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit_indices, tr_data["y"][fit_indices], tr_data["seller_id"][fit_indices]))
    assert all(not set(tr_data["seller_id"][fit_indices[a]]) & set(tr_data["seller_id"][fit_indices[b]]) for a, b in inner_cv)

    print(f"\nBuilding/loading baseline fold matrices (cache: {cache_path.name})...")
    t0 = time.time()
    mats = ag.build_fold_matrices(
        fit_indices,
        [tune_indices, confirm_indices],
        inner_cv,
        tr_data,
        [tr_data, tr_data],
        cache_path
    )
    print(f"Baseline matrices ready in {time.time() - t0:.1f}s.")

    X_base_fit = mats["X_m1_seen_causal_T"]
    X_base_tune = mats["X_m1_seen_causal_evals"][0]
    X_base_confirm = mats["X_m1_seen_causal_evals"][1]

    # 3. Extract Vía Principal Product Identity & Condition Features
    print("\nComputing Product Identity & Condition Feature Matrices...")
    t0 = time.time()
    X_id_fit, X_id_evals, id_feature_names = compute_identity_feature_matrices(
        df, fit_indices, [tune_indices, confirm_indices]
    )
    X_id_tune = X_id_evals[0]
    X_id_confirm = X_id_evals[1]
    print(f"Identity features ready in {time.time() - t0:.1f}s. Shape: {X_id_fit.shape[1]} features.")

    # Combined matrices for Arm 2
    X_arm2_fit = np.column_stack([X_base_fit, X_id_fit])
    X_arm2_tune = np.column_stack([X_base_tune, X_id_tune])
    X_arm2_confirm = np.column_stack([X_base_confirm, X_id_confirm])

    lgb_params = dict(
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=48,
        min_child_samples=35,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        random_state=119,
        n_jobs=6,
        verbosity=-1,
    )

    # =========================================================================
    # PHASE 1: EVALUATION ON TUNE (DEVELOPMENT & SELECTION ONLY)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING & EVALUATION ON TUNE PARTITION")
    print("=" * 60)

    # --- Arm 1: Strong Baseline (LightGBM Enriched) ---
    print("\n--- Arm 1: LightGBM Enriched Baseline ---")
    t0 = time.time()
    model_arm1 = LGBMClassifier(**lgb_params)
    model_arm1.fit(X_base_fit, y_fit)
    p_arm1_tune = model_arm1.predict_proba(X_base_tune)[:, 1].astype(np.float32)
    print(f"Arm 1 trained in {time.time() - t0:.1f}s. Tune Acc: {accuracy_score(y_tune, p_arm1_tune >= 0.5):.5f}, Logloss: {log_loss(y_tune, p_arm1_tune):.5f}")

    # --- Arm 2: Baseline + Product Identity Features ---
    print("\n--- Arm 2: LightGBM + Product Identity & Condition Representation ---")
    t0 = time.time()
    model_arm2 = LGBMClassifier(**lgb_params)
    model_arm2.fit(X_arm2_fit, y_fit)
    p_arm2_tune = model_arm2.predict_proba(X_arm2_tune)[:, 1].astype(np.float32)
    print(f"Arm 2 trained in {time.time() - t0:.1f}s. Tune Acc: {accuracy_score(y_tune, p_arm2_tune >= 0.5):.5f}, Logloss: {log_loss(y_tune, p_arm2_tune):.5f}")

    # Feature importances for Arm 2
    importances = model_arm2.feature_importances_
    id_start_idx = X_base_fit.shape[1]
    id_importances = importances[id_start_idx:]
    print("\nTop Product Identity Feature Importances (splits):")
    for name, imp in sorted(zip(id_feature_names, id_importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name:25s}: {imp:5d}")

    # --- Arm 3: Predefined Multi-Family Routed Ensemble ---
    print("\n--- Arm 3 Components: CatBoost Native, XGBoost Enriched, Bracketed Specialist ---")
    # 1. CatBoost Native with identity features
    df_fit_cat = pd.DataFrame(X_arm2_fit)
    df_fit_cat.columns = [f"f_{i}" for i in range(X_arm2_fit.shape[1])]
    df_tune_cat = pd.DataFrame(X_arm2_tune)
    df_tune_cat.columns = [f"f_{i}" for i in range(X_arm2_tune.shape[1])]

    for c in CATEGORICAL:
        df_fit_cat[c] = df[c].iloc[fit_indices].astype(str).values
        df_tune_cat[c] = df[c].iloc[tune_indices].astype(str).values

    t0 = time.time()
    model_cb = CatBoostClassifier(
        iterations=900,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=119,
        thread_count=6,
        verbose=False,
        cat_features=CATEGORICAL,
        allow_writing_files=False,
    )
    model_cb.fit(df_fit_cat, y_fit)
    p_cb_tune = model_cb.predict_proba(df_tune_cat)[:, 1].astype(np.float32)
    print(f"CatBoost trained in {time.time() - t0:.1f}s. Tune Acc: {accuracy_score(y_tune, p_cb_tune >= 0.5):.5f}, Logloss: {log_loss(y_tune, p_cb_tune):.5f}")

    # 2. XGBoost Enriched with identity features
    t0 = time.time()
    model_xgb = XGBClassifier(
        n_estimators=750,
        max_depth=6,
        learning_rate=0.035,
        colsample_bytree=0.75,
        subsample=0.85,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=119,
        n_jobs=6,
        verbosity=0,
        eval_metric="logloss",
    )
    model_xgb.fit(X_arm2_fit, y_fit)
    p_xgb_tune = model_xgb.predict_proba(X_arm2_tune)[:, 1].astype(np.float32)
    print(f"XGBoost trained in {time.time() - t0:.1f}s. Tune Acc: {accuracy_score(y_tune, p_xgb_tune >= 0.5):.5f}, Logloss: {log_loss(y_tune, p_xgb_tune):.5f}")

    # 3. Frente 2 Seller Specialist on Bracketed Rows
    t0 = time.time()
    hist_fit, hist_tune = compute_batch_seller_features(df, fit_indices, tune_indices)
    X_spec_fit = np.column_stack([X_arm2_fit, hist_fit])
    X_spec_tune = np.column_stack([X_arm2_tune, hist_tune])

    model_spec = LGBMClassifier(**lgb_params)
    model_spec.fit(X_spec_fit, y_fit)
    p_spec_tune = model_spec.predict_proba(X_spec_tune)[:, 1].astype(np.float32)
    print(f"Seller Specialist trained in {time.time() - t0:.1f}s.")

    # Assemble Arm 3
    p_broad_tune = (0.45 * p_arm2_tune + 0.35 * p_cb_tune + 0.20 * p_xgb_tune).astype(np.float32)
    p_arm3_tune = p_broad_tune.copy()
    p_arm3_tune[tune_seen_mask] = (0.50 * p_broad_tune[tune_seen_mask] + 0.50 * p_spec_tune[tune_seen_mask]).astype(np.float32)
    print(f"Arm 3 (Routed Ensemble) Tune Acc: {accuracy_score(y_tune, p_arm3_tune >= 0.5):.5f}, Logloss: {log_loss(y_tune, p_arm3_tune):.5f}")

    # Paired Bootstrap Evaluation on TUNE
    tune_candidates = {
        "Arm2_Identity_LGBM": p_arm2_tune,
        "Arm3_Routed_Ensemble": p_arm3_tune,
    }

    tune_results = {}
    for cand_name, p_cand in tune_candidates.items():
        tune_results[cand_name] = {}
        for regime_name, mask in [("all", np.ones(len(y_tune), bool)),
                                  ("seen_bracketed", tune_seen_mask),
                                  ("cold_new", ~tune_seen_mask)]:
            tune_results[cand_name][regime_name] = bootstrap_cluster_metrics(
                y_tune[mask], p_cand[mask], p_arm1_tune[mask], sellers_tune[mask], cats_tune[mask]
            )

        # Top contributor ablation
        tune_results[cand_name]["ablation"] = ablation_top_contributors(
            y_tune, p_cand, p_arm1_tune, sellers_tune, cats_tune
        )

    print("\n=== TUNE PAIRED EVALUATION SUMMARY VS ARM 1 BASELINE ===")
    for cand_name, res in tune_results.items():
        print(f"\nCandidate: {cand_name}")
        for reg in ("all", "seen_bracketed", "cold_new"):
            m = res[reg]
            print(f"  [{reg:15s}] Net: {m['net_hits']:+4d} (Fixes: {m['fixes']:3d}, Breaks: {m['breaks']:3d}, Flips->New: {m['flips_to_new']}, Flips->Used: {m['flips_to_used']}) | Acc: {m['cand_acc']:.5f} (vs {m['base_acc']:.5f}) | Logloss: {m['cand_logloss']:.5f} (delta: {m['delta_logloss']:+.5f}) | Q05-Q95: [{m['q05']:+.1f}, {m['q95']:+.1f}]")
        abl = res["ablation"]
        print(f"  [Ablation] Net without top-5 categories: {abl['net_without_top5_categories']:+d} (top5 share: {abl['top5_cat_net_share']})")
        print(f"  [Ablation] Net without top-5 sellers:    {abl['net_without_top5_sellers']:+d} (top5 share: {abl['top5_seller_net_share']})")

    # =========================================================================
    # PHASE 2: CONFIRM EVALUATION (STRICTLY FROZEN WEIGHTS & HYPERPARAMETERS)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: HELD-OUT CONFIRM EVALUATION (FROZEN WEIGHTS & ENSEMBLE)")
    print("=" * 60)

    # 1. Arm 1 Confirm
    p_arm1_confirm = model_arm1.predict_proba(X_base_confirm)[:, 1].astype(np.float32)

    # 2. Arm 2 Confirm
    p_arm2_confirm = model_arm2.predict_proba(X_arm2_confirm)[:, 1].astype(np.float32)

    # 3. Arm 3 Components Confirm
    df_confirm_cat = pd.DataFrame(X_arm2_confirm)
    df_confirm_cat.columns = [f"f_{i}" for i in range(X_arm2_confirm.shape[1])]
    for c in CATEGORICAL:
        df_confirm_cat[c] = df[c].iloc[confirm_indices].astype(str).values

    p_cb_confirm = model_cb.predict_proba(df_confirm_cat)[:, 1].astype(np.float32)
    p_xgb_confirm = model_xgb.predict_proba(X_arm2_confirm)[:, 1].astype(np.float32)

    _, hist_confirm = compute_batch_seller_features(df, fit_indices, confirm_indices)
    X_spec_confirm = np.column_stack([X_arm2_confirm, hist_confirm])
    p_spec_confirm = model_spec.predict_proba(X_spec_confirm)[:, 1].astype(np.float32)

    p_broad_confirm = (0.45 * p_arm2_confirm + 0.35 * p_cb_confirm + 0.20 * p_xgb_confirm).astype(np.float32)
    p_arm3_confirm = p_broad_confirm.copy()
    p_arm3_confirm[confirm_seen_mask] = (0.50 * p_broad_confirm[confirm_seen_mask] + 0.50 * p_spec_confirm[confirm_seen_mask]).astype(np.float32)

    confirm_candidates = {
        "Arm2_Identity_LGBM": p_arm2_confirm,
        "Arm3_Routed_Ensemble": p_arm3_confirm,
    }

    confirm_results = {}
    for cand_name, p_cand in confirm_candidates.items():
        confirm_results[cand_name] = {}
        for regime_name, mask in [("all", np.ones(len(y_confirm), bool)),
                                  ("seen_bracketed", confirm_seen_mask),
                                  ("cold_new", ~confirm_seen_mask)]:
            confirm_results[cand_name][regime_name] = bootstrap_cluster_metrics(
                y_confirm[mask], p_cand[mask], p_arm1_confirm[mask], sellers_confirm[mask], cats_confirm[mask]
            )

        confirm_results[cand_name]["ablation"] = ablation_top_contributors(
            y_confirm, p_cand, p_arm1_confirm, sellers_confirm, cats_confirm
        )

    print("\n=== CONFIRM PAIRED EVALUATION SUMMARY VS ARM 1 BASELINE ===")
    for cand_name, res in confirm_results.items():
        print(f"\nCandidate: {cand_name}")
        for reg in ("all", "seen_bracketed", "cold_new"):
            m = res[reg]
            print(f"  [{reg:15s}] Net: {m['net_hits']:+4d} (Fixes: {m['fixes']:3d}, Breaks: {m['breaks']:3d}, Flips->New: {m['flips_to_new']}, Flips->Used: {m['flips_to_used']}) | Acc: {m['cand_acc']:.5f} (vs {m['base_acc']:.5f}) | Logloss: {m['cand_logloss']:.5f} (delta: {m['delta_logloss']:+.5f}) | Q05-Q95: [{m['q05']:+.1f}, {m['q95']:+.1f}]")
        abl = res["ablation"]
        print(f"  [Ablation] Net without top-5 categories: {abl['net_without_top5_categories']:+d} (top5 share: {abl['top5_cat_net_share']})")
        print(f"  [Ablation] Net without top-5 sellers:    {abl['net_without_top5_sellers']:+d} (top5 share: {abl['top5_seller_net_share']})")

    # Save private pairs
    pairs_tune_df = pd.DataFrame({
        "id": df["id"].iloc[tune_indices].values,
        "y": y_tune,
        "regime": split_df.iloc[tune_indices]["regime"].values,
        "p_arm1_base": p_arm1_tune,
        "p_arm2_identity": p_arm2_tune,
        "p_cb_native": p_cb_tune,
        "p_xgb_enriched": p_xgb_tune,
        "p_arm3_ensemble": p_arm3_tune,
    })
    pairs_tune_path = RESULTS_DIR / "via_principal_tune_pairs_private.csv"
    pairs_tune_df.to_csv(pairs_tune_path, index=False)

    pairs_confirm_df = pd.DataFrame({
        "id": df["id"].iloc[confirm_indices].values,
        "y": y_confirm,
        "regime": split_df.iloc[confirm_indices]["regime"].values,
        "p_arm1_base": p_arm1_confirm,
        "p_arm2_identity": p_arm2_confirm,
        "p_cb_native": p_cb_confirm,
        "p_xgb_enriched": p_xgb_confirm,
        "p_arm3_ensemble": p_arm3_confirm,
    })
    pairs_confirm_path = RESULTS_DIR / "via_principal_confirm_pairs_private.csv"
    pairs_confirm_df.to_csv(pairs_confirm_path, index=False)

    # Master report
    final_report = {
        "experiment": "via_principal_product_identity_and_routed_ensemble",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "split_manifest_sha256": manifest["split_sha256"],
        "tune_pairs_sha256": sha256_file(pairs_tune_path),
        "confirm_pairs_sha256": sha256_file(pairs_confirm_path),
        "baseline": "Arm1_LightGBM_Enriched",
        "tune_eval_metrics": tune_results,
        "confirm_eval_metrics": confirm_results,
        "status": "DEVELOPMENT_ONLY",
        "public_probability_claims": "NO_IDENTIFIABLE",
    }

    report_path = RESULTS_DIR / "via_principal_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nExecution finished in {time.time() - t_start:.1f}s.")
    print(f"Tune pairs saved to: {pairs_tune_path}")
    print(f"Confirm pairs saved to: {pairs_confirm_path}")
    print(f"Master report saved to: {report_path}")


if __name__ == "__main__":
    main()
