"""AP-2026-Q-2: Master Execution of Frente 1 (Broad Model) and Frente 2 (Seller Specialist).

Strictly leak-free, reproducible, evaluated on the canonical test-representative split (9c7a6a9a1305ec).
Frente 1: Broad multi-family ensemble (LightGBM Enriched + CatBoost Native + XGBoost Enriched).
Frente 2: Bracketed Seller Specialist strictly routed to bracketed rows (avoiding new seller penalty).
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

SPLIT_CSV = ROOT / "data" / "splits" / "frente3_test_representative_split.csv"
SPLIT_MANIFEST = ROOT / "data" / "splits" / "frente3_manifest.json"
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


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


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
    
    # Cluster IDs
    unique_sellers, s_idx = np.unique(sellers, return_inverse=True)
    unique_cats, c_idx = np.unique(cats, return_inverse=True)
    n_s = len(unique_sellers)
    n_c = len(unique_cats)
    
    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        # Two-way cluster weights via Rademacher / normal product
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
        "q05": q05,
        "q95": q95,
        "cand_logloss": cand_ll,
        "base_logloss": base_ll,
        "delta_logloss": cand_ll - base_ll,
    }


def main():
    t_start = time.time()
    print("=== Master Execution: Frente 1 (Broad Model) + Frente 2 (Seller Specialist) ===")
    
    # 1. Load Split and Parsed Data
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Verified Split SHA-256: {manifest['split_sha256']}")
    
    split_df = pd.read_csv(SPLIT_CSV)
    fit_indices = np.flatnonzero(split_df["role"] == "fit")
    val_indices = np.flatnonzero(split_df["role"] == "val")
    seen_mask = split_df.iloc[val_indices]["regime"].values == "seen_bracketed"
    
    with open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
        
    y_val = df["y"].iloc[val_indices].to_numpy()
    sellers_val = df["seller_id"].iloc[val_indices].to_numpy()
    cats_val = df["category_id"].iloc[val_indices].to_numpy()
    
    print(f"Fit: {len(fit_indices)} | Val: {len(val_indices)} (Seen bracketed: {seen_mask.sum()}, Cold new: {(~seen_mask).sum()})")
    
    # 2. Build or Load Fold Matrices
    cache_path = CACHE_DIR / f"text_{manifest['split_sha256'][:16]}.npz"
    tr_data = ag.precompute_item_intrinsic_arrays(df)
    
    inner_cv = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit_indices, tr_data["y"][fit_indices], tr_data["seller_id"][fit_indices]))
    assert all(not set(tr_data["seller_id"][fit_indices[a]]) & set(tr_data["seller_id"][fit_indices[b]]) for a, b in inner_cv)
    
    ag.compute_causal_seller_features = ag.compute_causal_seller_features  # default causal
    
    print(f"Building/loading fold matrices (cache: {cache_path})...")
    t0 = time.time()
    mats = ag.build_fold_matrices(fit_indices, [val_indices], inner_cv, tr_data, [tr_data], cache_path)
    print(f"Matrices ready in {time.time() - t0:.1f}s. Sha ptx1: {mats['sha_ptx1_oof'][:12]}")
    
    # Feature blocks
    x_base_fit = mats["X_m1_seen_causal_T"]
    x_base_val = mats["X_m1_seen_causal_evals"][0]
    y_fit = mats["y_T"]
    
    # 3. Train Baseline: LightGBM Enriched
    print("\n--- Training Model 1: LightGBM Enriched (Strong Baseline) ---")
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
    t0 = time.time()
    model_lgb = LGBMClassifier(**lgb_params)
    model_lgb.fit(x_base_fit, y_fit)
    p_lgb = model_lgb.predict_proba(x_base_val)[:, 1].astype(np.float32)
    print(f"LightGBM trained in {time.time() - t0:.1f}s. Val Acc: {accuracy_score(y_val, p_lgb >= 0.5):.5f}, Logloss: {log_loss(y_val, p_lgb):.5f}")
    
    # 4. Train Frente 1 Challenger: CatBoost Native
    print("\n--- Training Model 2: CatBoost Native (Frente 1 Challenger) ---")
    # Assemble CatBoost native DataFrame with categoricals
    df_fit_cat = pd.DataFrame(x_base_fit)
    df_fit_cat.columns = [f"f_{i}" for i in range(x_base_fit.shape[1])]
    df_val_cat = pd.DataFrame(x_base_val)
    df_val_cat.columns = [f"f_{i}" for i in range(x_base_val.shape[1])]
    
    for c in CATEGORICAL:
        df_fit_cat[c] = df[c].iloc[fit_indices].astype(str).values
        df_val_cat[c] = df[c].iloc[val_indices].astype(str).values
        
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
    p_cb = model_cb.predict_proba(df_val_cat)[:, 1].astype(np.float32)
    print(f"CatBoost trained in {time.time() - t0:.1f}s. Val Acc: {accuracy_score(y_val, p_cb >= 0.5):.5f}, Logloss: {log_loss(y_val, p_cb):.5f}")
    
    # 5. Train Frente 1 Challenger: XGBoost Enriched
    print("\n--- Training Model 3: XGBoost Enriched (Frente 1 Challenger) ---")
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
    model_xgb.fit(x_base_fit, y_fit)
    p_xgb = model_xgb.predict_proba(x_base_val)[:, 1].astype(np.float32)
    print(f"XGBoost trained in {time.time() - t0:.1f}s. Val Acc: {accuracy_score(y_val, p_xgb >= 0.5):.5f}, Logloss: {log_loss(y_val, p_xgb):.5f}")
    
    # 6. Frente 1 Broad Blend
    p_broad = (0.45 * p_lgb + 0.35 * p_cb + 0.20 * p_xgb).astype(np.float32)
    print(f"\nBroad Blend Val Acc: {accuracy_score(y_val, p_broad >= 0.5):.5f}, Logloss: {log_loss(y_val, p_broad):.5f}")
    
    # 7. Frente 2: Seller Specialist on Bracketed Rows
    print("\n--- Training Model 4: Frente 2 Bracketed Seller Specialist ---")
    t0 = time.time()
    hist_fit, hist_val = compute_batch_seller_features(df, fit_indices, val_indices)
    x_spec_fit = np.column_stack([x_base_fit, hist_fit])
    x_spec_val = np.column_stack([x_base_val, hist_val])
    
    model_spec = LGBMClassifier(**lgb_params)
    model_spec.fit(x_spec_fit, y_fit)
    p_spec = model_spec.predict_proba(x_spec_val)[:, 1].astype(np.float32)
    print(f"Seller Specialist trained in {time.time() - t0:.1f}s.")
    
    # Strict Routing: apply specialist only to seen bracketed rows!
    p_routed = p_broad.copy()
    p_routed[seen_mask] = (0.50 * p_broad[seen_mask] + 0.50 * p_spec[seen_mask]).astype(np.float32)
    print(f"Strictly Routed Hybrid Val Acc: {accuracy_score(y_val, p_routed >= 0.5):.5f}, Logloss: {log_loss(y_val, p_routed):.5f}")
    
    # 8. Evaluate all candidates with two-way cluster bootstrap
    candidates = {
        "Frente1_CatBoost_Native": p_cb,
        "Frente1_XGBoost_Enriched": p_xgb,
        "Frente1_Broad_Blend": p_broad,
        "Frente2_Pure_Specialist_Unrouted": p_spec,
        "Frente2_Strictly_Routed_Hybrid": p_routed,
    }
    
    metrics_report = {}
    for cand_name, p_cand in candidates.items():
        metrics_report[cand_name] = {}
        for regime_name, mask in [("all", np.ones(len(y_val), bool)), 
                                  ("seen_bracketed", seen_mask), 
                                  ("cold_new", ~seen_mask)]:
            metrics_report[cand_name][regime_name] = bootstrap_cluster_metrics(
                y_val[mask], p_cand[mask], p_lgb[mask], sellers_val[mask], cats_val[mask])
            
    print("\n=== SUMMARY OF PAIRED EVALUATION VS BASELINE ===")
    for cand_name, res in metrics_report.items():
        print(f"\nCandidate: {cand_name}")
        for reg in ("all", "seen_bracketed", "cold_new"):
            m = res[reg]
            print(f"  [{reg:15s}] Net: {m['net_hits']:+4d} (Fixes: {m['fixes']:3d}, Breaks: {m['breaks']:3d}) | Acc: {m['cand_acc']:.5f} (vs {m['base_acc']:.5f}) | Logloss: {m['cand_logloss']:.5f} (delta: {m['delta_logloss']:+.5f}) | Q05-Q95: [{m['q05']:+.1f}, {m['q95']:+.1f}]")
            
    # Save private pairs
    pairs_df = pd.DataFrame({
        "id": df["id"].iloc[val_indices].values,
        "y": y_val,
        "regime": split_df.iloc[val_indices]["regime"].values,
        "p_lgb_baseline": p_lgb,
        "p_cb_native": p_cb,
        "p_xgb_enriched": p_xgb,
        "p_broad_blend": p_broad,
        "p_seller_specialist": p_spec,
        "p_routed_hybrid": p_routed,
    })
    pairs_path = RESULTS_DIR / "frentes_validation_pairs_private.csv"
    pairs_df.to_csv(pairs_path, index=False)
    
    # Save full report
    final_report = {
        "experiment": "frente1_broad_model_and_frente2_seller_specialist",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "split_manifest_sha256": manifest["split_sha256"],
        "pairs_sha256": sha256_file(pairs_path),
        "baseline": "LightGBM_Enriched_m1",
        "eval_metrics": metrics_report,
        "status": "DEVELOPMENT_ONLY",
        "public_probability_claims": "NO_IDENTIFIABLE",
    }
    
    report_path = RESULTS_DIR / "frentes_master_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
        
    print(f"\nExecution finished in {time.time() - t_start:.1f}s.")
    print(f"Private pairs saved to: {pairs_path}")
    print(f"Master report saved to: {report_path}")


if __name__ == "__main__":
    main()
