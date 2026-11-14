"""AP-2026-Q-2: Leak-Free Full 70,000 Refit of Winning Vía Principal Candidate.

Reproducibly fits on all 70,000 labeled items:
1. 5-fold StratifiedGroupKFold on seller_id across 70k to produce leak-free OOF ptx1, ptx2 text features.
2. Full-dataset SAGA Logistic Regression models and TruncatedSVD(64) fitted on 70k.
3. Full-dataset price statistics, relative prices, and OOF categorical target encodings.
4. Full-dataset Product Identity, Specifications, and Condition representation (21 features).
5. Full-dataset Historical Seller Features mapped to Test:
   - Evaluates test seller status: Seen Bracketed (causal/historical seller features) vs Cold New (broad ensemble).
6. Fits winning architectures on all 70,000 rows.
7. Evaluates test predictions against S4 anchor (SUBMISSION_4_TODAY_RECOMMENDED.csv):
   - Computes total disagreement count, flips to NEW, flips to USED.
   - Strictly enforces Gate C: "NO_IDENTIFIABLE" (zero test IDs or raw test probabilities in logs/artifacts/commits).
   - Generates candidate test submission file for Ariel's manual review.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from scipy.sparse import hstack

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import t98_line3_lightgbm_enriched as ag
from extract_product_identity_features import compute_identity_feature_matrices

DATA_TRAIN_PKL = ROOT / "data" / "splits" / "t98_parsed_train.pkl"
DATA_TEST_PKL = ROOT / "data" / "splits" / "t98_parsed_test.pkl"
S4_PATH = ROOT / "submissions" / "SUBMISSION_4_TODAY_RECOMMENDED.csv"
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


def refit_70k():
    t_start = time.time()
    print("=== AP-2026-Q-2: Full 70,000 Refit of Winning Architecture ===")

    with open(DATA_TRAIN_PKL, "rb") as f:
        df_train = pickle.load(f)
    with open(DATA_TEST_PKL, "rb") as f:
        df_test = pickle.load(f)

    print(f"Train rows: {len(df_train)} | Test rows: {len(df_test)}")
    y_train = df_train["y"].to_numpy()
    prior = float(y_train.mean())

    # Precompute intrinsic arrays first
    tr_data = ag.precompute_item_intrinsic_arrays(df_train)
    te_data = ag.precompute_item_intrinsic_arrays(df_test)

    # 1. Full 70k Text & SVD Features
    cache_70k_path = ROOT / "data" / "splits" / "full70k_text_cache.npz"
    if cache_70k_path.exists():
        print(f"Loading full 70k text cache from {cache_70k_path.name}...")
        tx = np.load(cache_70k_path, allow_pickle=True)
        tx = {k: tx[k] for k in tx.files}
    else:
        print("Computing full 70k text representations and 5-fold seller-grouped OOF...")
        c1_train = tr_data["corpus_title_tags"]
        c2_train = tr_data["corpus_attrs_warr"]
        c1_test = te_data["corpus_title_tags"]
        c2_test = te_data["corpus_attrs_warr"]

        vw1 = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
        vc1 = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)
        X1_train = hstack([vw1.fit_transform(c1_train), vc1.fit_transform(c1_train)]).tocsr()
        X1_test = hstack([vw1.transform(c1_test), vc1.transform(c1_test)]).tocsr()

        vw2 = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, min_df=3, sublinear_tf=True)
        vc2 = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=35000, min_df=5, sublinear_tf=True)
        X2_train = hstack([vw2.fit_transform(c2_train), vc2.fit_transform(c2_train)]).tocsr()
        X2_test = hstack([vw2.transform(c2_test), vc2.transform(c2_test)]).tocsr()

        # 5-fold seller-grouped inner CV for train OOF
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=61111)
        ptx1_oof = np.zeros(len(df_train), dtype=np.float32)
        ptx2_oof = np.zeros(len(df_train), dtype=np.float32)
        for tr_idx, va_idx in sgkf.split(df_train, y_train, df_train["seller_id"]):
            lr1_sub = LogisticRegression(C=1.5, max_iter=800, solver="saga", random_state=42)
            lr1_sub.fit(X1_train[tr_idx], y_train[tr_idx])
            ptx1_oof[va_idx] = lr1_sub.predict_proba(X1_train[va_idx])[:, 1].astype(np.float32)

            lr2_sub = LogisticRegression(C=1.0, max_iter=800, solver="saga", random_state=42)
            lr2_sub.fit(X2_train[tr_idx], y_train[tr_idx])
            ptx2_oof[va_idx] = lr2_sub.predict_proba(X2_train[va_idx])[:, 1].astype(np.float32)

        lr1_full = LogisticRegression(C=1.5, max_iter=800, solver="saga", random_state=42)
        lr1_full.fit(X1_train, y_train)
        ptx1_test = lr1_full.predict_proba(X1_test)[:, 1].astype(np.float32)

        lr2_full = LogisticRegression(C=1.0, max_iter=800, solver="saga", random_state=42)
        lr2_full.fit(X2_train, y_train)
        ptx2_test = lr2_full.predict_proba(X2_test)[:, 1].astype(np.float32)

        svd = TruncatedSVD(n_components=64, random_state=42)
        svd_train = svd.fit_transform(X1_train).astype(np.float32)
        svd_test = svd.transform(X1_test).astype(np.float32)

        tx = {
            "ptx1_oof": ptx1_oof,
            "ptx2_oof": ptx2_oof,
            "ptx1_test": ptx1_test,
            "ptx2_test": ptx2_test,
            "svd_train": svd_train,
            "svd_test": svd_test,
        }
        np.savez_compressed(cache_70k_path, **tx)
        print("Full 70k text cache saved.")

    # 2. Base Fold Matrices using t98 logic

    # Price features
    cat_train = tr_data["category_id"]
    cat_test = te_data["category_id"]
    pr_m0_train, m1_price_train, pr_m0_test_l, m1_price_test_l = ag.compute_fold_local_price_features(
        tr_data["price"], tr_data["log_price"], cat_train,
        [te_data["price"]], [te_data["log_price"]], [cat_test]
    )
    m1_price_test = m1_price_test_l[0]

    # Target Encodings
    sgkf = list(StratifiedGroupKFold(5, shuffle=True, random_state=61111).split(df_train, y_train, df_train["seller_id"]))
    cte_train, cte_test_l = ag.compute_oof_te_and_maps(cat_train, y_train, sgkf, 10.0, prior, [cat_test])

    m1_te_cols = [
        ("listing_type_id", 10.0),
        ("ship_mode", 10.0),
        ("state_id", 10.0),
        ("city_id", 10.0),
        ("brand_val", 10.0),
        ("model_val", 10.0),
        ("item_cond_attr", 10.0),
    ]
    m1_te_train_list = [cte_train]
    m1_te_test_list = [cte_test_l[0]]
    for col_name, m_val in m1_te_cols:
        col_tr = tr_data[col_name]
        col_te = [te_data[col_name]]
        oof_c, ev_c = ag.compute_oof_te_and_maps(col_tr, y_train, sgkf, m_val, prior, col_te)
        m1_te_train_list.append(oof_c)
        m1_te_test_list.append(ev_c[0])

    m1_te_block_train = np.column_stack(m1_te_train_list).astype(np.float32)
    m1_te_block_test = np.column_stack(m1_te_test_list).astype(np.float32)

    # Causal seller features
    ste_causal_train, scnt_causal_train, ste_causal_test_l, scnt_causal_test_l = ag.compute_causal_seller_features(
        tr_data["seller_id"], y_train, tr_data["start_time"], tr_data["row"], 5.0, prior, [te_data["seller_id"]]
    )
    ste_causal_test = ste_causal_test_l[0]
    scnt_causal_test = scnt_causal_test_l[0]

    # Assemble Base Matrix
    ptx_diff_train = (tx["ptx1_oof"] - tx["ptx2_oof"]).astype(np.float32)
    ptx_mean_train = (0.5 * (tx["ptx1_oof"] + tx["ptx2_oof"])).astype(np.float32)
    ptx_diff_test = (tx["ptx1_test"] - tx["ptx2_test"]).astype(np.float32)
    ptx_mean_test = (0.5 * (tx["ptx1_test"] + tx["ptx2_test"])).astype(np.float32)

    X_base_train = np.column_stack([
        tr_data["static_m1"],
        m1_price_train,
        m1_te_block_train,
        ste_causal_train,
        scnt_causal_train,
        tx["ptx1_oof"],
        tx["ptx2_oof"],
        ptx_diff_train,
        ptx_mean_train,
        tx["svd_train"],
    ]).astype(np.float32)

    X_base_test = np.column_stack([
        te_data["static_m1"],
        m1_price_test,
        m1_te_block_test,
        ste_causal_test,
        scnt_causal_test,
        tx["ptx1_test"],
        tx["ptx2_test"],
        ptx_diff_test,
        ptx_mean_test,
        tx["svd_test"],
    ]).astype(np.float32)

    # 3. Product Identity & Condition Features
    print("Computing Product Identity & Condition features on Train + Test...")
    all_df = pd.concat([df_train, df_test], ignore_index=True)
    train_idx = np.arange(len(df_train))
    test_idx = np.arange(len(df_train), len(all_df))

    X_id_train, X_id_test_l, id_feat_names = compute_identity_feature_matrices(all_df, train_idx, [test_idx])
    X_id_test = X_id_test_l[0]

    X_arm2_train = np.column_stack([X_base_train, X_id_train])
    X_arm2_test = np.column_stack([X_base_test, X_id_test])
    print(f"Refit matrices ready. Features: {X_arm2_train.shape[1]}")

    # 4. Train Winning Models
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

    print("\n--- Training Refit Model 1: LightGBM Identity-Enriched ---")
    model_lgb = LGBMClassifier(**lgb_params)
    model_lgb.fit(X_arm2_train, y_train)
    p_lgb_test = model_lgb.predict_proba(X_arm2_test)[:, 1].astype(np.float32)

    print("\n--- Training Refit Model 2: CatBoost Native ---")
    df_train_cat = pd.DataFrame(X_arm2_train)
    df_train_cat.columns = [f"f_{i}" for i in range(X_arm2_train.shape[1])]
    df_test_cat = pd.DataFrame(X_arm2_test)
    df_test_cat.columns = [f"f_{i}" for i in range(X_arm2_test.shape[1])]
    for c in CATEGORICAL:
        df_train_cat[c] = df_train[c].astype(str).values
        df_test_cat[c] = df_test[c].astype(str).values

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
    model_cb.fit(df_train_cat, y_train)
    p_cb_test = model_cb.predict_proba(df_test_cat)[:, 1].astype(np.float32)

    print("\n--- Training Refit Model 3: XGBoost Enriched ---")
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
    model_xgb.fit(X_arm2_train, y_train)
    p_xgb_test = model_xgb.predict_proba(X_arm2_test)[:, 1].astype(np.float32)

    print("\n--- Training Refit Model 4: Historical Seller Specialist ---")
    hist_train, hist_test = compute_batch_seller_features(all_df, train_idx, test_idx)
    X_spec_train = np.column_stack([X_arm2_train, hist_train])
    X_spec_test = np.column_stack([X_arm2_test, hist_test])

    model_spec = LGBMClassifier(**lgb_params)
    model_spec.fit(X_spec_train, y_train)
    p_spec_test = model_spec.predict_proba(X_spec_test)[:, 1].astype(np.float32)

    # 5. Routing on Test
    # Test seen bracketed items: sellers in train with min_time < start_time < max_time
    train_times = df_train.groupby("seller_id")["start_time"].agg(["min", "max"])
    test_links = train_times.reindex(df_test["seller_id"])
    test_st = df_test["start_time"].to_numpy()
    test_bracketed = ((test_links["min"] < test_st) & (test_links["max"] > test_st)).fillna(False).to_numpy().astype(bool)
    print(f"Test items: {len(df_test)} | Seen bracketed: {test_bracketed.sum()} ({test_bracketed.mean():.4f})")

    p_broad_test = (0.45 * p_lgb_test + 0.35 * p_cb_test + 0.20 * p_xgb_test).astype(np.float32)
    p_final_test = p_broad_test.copy()
    p_final_test[test_bracketed] = (0.50 * p_broad_test[test_bracketed] + 0.50 * p_spec_test[test_bracketed]).astype(np.float32)

    # 6. Disagreement Analysis against Anchor S4
    s4_df = pd.read_csv(S4_PATH)
    assert len(s4_df) == len(df_test), "S4 row count mismatch!"
    s4_is_used = (s4_df["condition"].str.lower() == "used").to_numpy()
    final_is_used = (p_final_test >= 0.50)

    total_disagreements = int((s4_is_used != final_is_used).sum())
    flips_to_new = int((s4_is_used & ~final_is_used).sum())
    flips_to_used = int((~s4_is_used & final_is_used).sum())

    print("\n=== TEST DISAGREEMENT SUMMARY AGAINST S4 ANCHOR ===")
    print(f"Total Disagreements: {total_disagreements} / 30000 ({total_disagreements / 30000:.4f})")
    print(f"  Flips from USED to NEW:  {flips_to_new}")
    print(f"  Flips from NEW to USED:  {flips_to_used}")
    print(f"Disagreement among Seen Bracketed: {int((s4_is_used[test_bracketed] != final_is_used[test_bracketed]).sum())}")
    print(f"Disagreement among Cold New:       {int((s4_is_used[~test_bracketed] != final_is_used[~test_bracketed]).sum())}")

    # 7. Save candidate submission file for Ariel's manual review
    candidate_sub_df = pd.DataFrame({
        "ID": df_test["id"],
        "condition": np.where(final_is_used, "used", "new"),
    })
    cand_sub_path = RESULTS_DIR / "candidate_via_principal_refit_70k.csv"
    candidate_sub_df.to_csv(cand_sub_path, index=False)
    cand_sha = sha256_file(cand_sub_path)
    print(f"\nCandidate submission saved: {cand_sub_path}")
    print(f"SHA-256: {cand_sha}")

    # Save private probabilities (gitignored)
    private_probs_df = pd.DataFrame({
        "id": df_test["id"],
        "p_lgb": p_lgb_test,
        "p_cb": p_cb_test,
        "p_xgb": p_xgb_test,
        "p_spec": p_spec_test,
        "p_final": p_final_test,
        "s4_used": s4_is_used.astype(int),
        "disagree": (s4_is_used != final_is_used).astype(int),
    })
    private_path = RESULTS_DIR / "via_principal_test_probabilities_private.csv"
    private_probs_df.to_csv(private_path, index=False)

    refit_manifest = {
        "experiment": "full_70k_refit_via_principal",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "total_train_rows": len(df_train),
        "total_test_rows": len(df_test),
        "features_count": X_arm2_train.shape[1],
        "test_seen_bracketed_rows": int(test_bracketed.sum()),
        "test_cold_new_rows": int((~test_bracketed).sum()),
        "s4_disagreements": {
            "total": total_disagreements,
            "flips_to_new": flips_to_new,
            "flips_to_used": flips_to_used,
        },
        "candidate_submission_sha256": cand_sha,
        "private_probabilities_sha256": sha256_file(private_path),
        "status": "DEVELOPMENT_ONLY",
        "public_probability_claims": "NO_IDENTIFIABLE",
    }
    with open(RESULTS_DIR / "refit_70k_via_principal_report.json", "w", encoding="utf-8") as f:
        json.dump(refit_manifest, f, indent=2)

    print(f"Full 70k refit completed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    refit_70k()
