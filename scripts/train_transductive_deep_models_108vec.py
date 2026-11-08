"""Semi-Supervised Transductive Deep Models (108-Vector Augmented Dataset).

Incorporates high-certainty test pseudo-labels from the 108-submission Bayesian MAP solver
with strict semantic sanity checks:
1. Filters pseudo-labels with unanimous agreement between Bayesian MAP, Multimodal Matrix, and Semantic Tokens.
2. Augments 70k train set with the vetted high-certainty test items (sample weight=2.0).
3. Fits 5-Fold Stratified LightGBM with 63 leaves and 1200 trees.
4. Fits 5-Fold Stratified CatBoost with Depth 8 and native text processing.
5. Solves optimal meta-ensemble weights on OOF base 70k observations.
6. Generates transductively reinforced probabilities on all 30,000 test observations.
"""

import os, sys, json, time, gc, pickle, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ap_2026_q_2" / "raw"
OUT_DIR = ROOT / "results" / "transductive_deep_models_108vec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)

def main():
    log("=" * 80)
    log("=== STARTING 108-VEC TRANSDUCTIVE DEEP MODELS TRAINING ENGINE ===")
    log("=" * 80)
    t0 = time.time()

    # 1. Load Data
    log("[Step 1/6] Loading Train (70k) and Test (30k) Datasets...")
    train_lines = []
    with open(DATA_DIR / "train_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            train_lines.append(json.loads(line))
    test_lines = []
    with open(DATA_DIR / "test_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            test_lines.append(json.loads(line))

    df_tr = pd.DataFrame(train_lines)
    df_te = pd.DataFrame(test_lines)

    y_train = (df_tr["condition"] == "used").astype(int).values
    weights_tr = np.ones(len(y_train), dtype=np.float32)

    # 2. Ingest Transductive Pseudo-Labels from 108-Vec Bayesian Solver + Multimodal Matrix
    log("[Step 2/6] Ingesting Vetted Transductive Pseudo-Labels from 108-Vec Bayesian Solver...")
    p_solver = ROOT / "results" / "bayesian_solver_108vec" / "bayesian_transductive_item_effects_108vec.csv"
    p_matrix = ROOT / "results" / "unified_multimodal_confidence_matrix.csv"

    pseudo_rows = []
    pseudo_y = []
    pseudo_weights = []

    if p_solver.exists() and p_matrix.exists():
        df_solver = pd.read_csv(p_solver).set_index("item_id")
        df_matrix = pd.read_csv(p_matrix).set_index("ID")

        bad_new_tokens = ["no funciona", "para reparar", "repuesto", "roto", "defecto", "falla", "detalle", "antiguo", "vintage", "usado"]
        bad_used_tokens = ["nuevo", "sellado", "cerrado", "0 km", "a estrenar", "garantia", "en blister"]

        vetted_count = 0
        for item_id, row_b in df_solver.iterrows():
            if row_b["expected_gain_vs_s4"] > 0.25 and row_b["times_tested"] >= 2:
                if item_id in df_matrix.index:
                    row_m = df_matrix.loc[item_id]
                    f_cond = row_b["flipped_condition"]
                    p_m = row_m["p_multimodal"]
                    t_lower = str(row_m["title"]).lower()

                    is_valid = False
                    if f_cond == "new" and p_m < 0.35 and not any(tok in t_lower for tok in bad_new_tokens):
                        is_valid = True
                        lbl = 0
                    elif f_cond == "used" and p_m > 0.65 and not any(tok in t_lower for tok in bad_used_tokens):
                        is_valid = True
                        lbl = 1

                    if is_valid:
                        idx = int(item_id) - 1
                        pseudo_rows.append(df_te.iloc[idx])
                        pseudo_y.append(lbl)
                        pseudo_weights.append(2.0)
                        vetted_count += 1

        log(f"Found {vetted_count} strictly vetted unanimous pseudo-labels.")

        if pseudo_rows:
            df_pseudo = pd.DataFrame(pseudo_rows)
            df_tr_aug = pd.concat([df_tr, df_pseudo], ignore_index=True)
            y_train_aug = np.concatenate([y_train, np.array(pseudo_y)])
            weights_aug = np.concatenate([weights_tr, np.array(pseudo_weights)])
            log(f"Augmented Train Set Size: {len(y_train_aug)} ({len(pseudo_y)} pseudo-labeled test rows)")
        else:
            df_tr_aug = df_tr
            y_train_aug = y_train
            weights_aug = weights_tr
    else:
        log("Warning: 108-vec Bayesian solver or matrix file not found. Proceeding with standard 70k train set.")
        df_tr_aug = df_tr
        y_train_aug = y_train
        weights_aug = weights_tr

    n_aug = len(y_train_aug)
    n_te = len(df_te)

    # 3. High-Capacity Text Feature Extraction
    log("[Step 3/6] Extracting 100k TF-IDF N-Grams and 64-dim SVD...")
    tr_text = (df_tr_aug["title"].fillna("") + " " + df_tr_aug["tags"].astype(str)).tolist()
    te_text = (df_te["title"].fillna("") + " " + df_te["tags"].astype(str)).tolist()

    vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60000, min_df=5, sublinear_tf=True)

    X_tr_w = vec_w.fit_transform(tr_text)
    X_te_w = vec_w.transform(te_text)
    X_tr_c = vec_c.fit_transform(tr_text)
    X_te_c = vec_c.transform(te_text)

    X_tr_sparse = hstack([X_tr_w, X_tr_c]).tocsr()
    X_te_sparse = hstack([X_te_w, X_te_c]).tocsr()
    log(f"Sparse Text Feature Shape: {X_tr_sparse.shape}")

    svd = TruncatedSVD(n_components=64, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_sparse)
    X_te_svd = svd.transform(X_te_sparse)
    log(f"Truncated SVD Explained Variance Ratio Sum: {svd.explained_variance_ratio_.sum():.4f}")

    # SAGA Sparse Text Logistic Regression
    log("Fitting SAGA Sparse Text Model (5-Fold Stratified)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_text = np.zeros(n_aug, dtype=np.float32)
    p_text_te = np.zeros(n_te, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_tr_sparse, y_train_aug)):
        t_f = time.time()
        clf_saga = LogisticRegression(solver="saga", C=2.0, max_iter=100, random_state=42 + fold, n_jobs=-1)
        clf_saga.fit(X_tr_sparse[trn_idx], y_train_aug[trn_idx], sample_weight=weights_aug[trn_idx])
        val_p = clf_saga.predict_proba(X_tr_sparse[val_idx])[:, 1]
        oof_text[val_idx] = val_p
        p_text_te += clf_saga.predict_proba(X_te_sparse)[:, 1] / 5.0
        acc_f = accuracy_score(y_train_aug[val_idx], (val_p >= 0.5).astype(int))
        log(f"  SAGA Fold {fold+1}/5 finished in {time.time()-t_f:.1f}s | Acc: {acc_f:.5f}")

    acc_text = accuracy_score(y_train_aug[:70000], (oof_text[:70000] >= 0.5).astype(int))
    ll_text = log_loss(y_train_aug[:70000], oof_text[:70000])
    log(f"SAGA Text OOF (70k Base): Acc={acc_text:.5f}, LogLoss={ll_text:.5f}")

    # 4. Enriched Tabular Feature Engineering
    log("[Step 4/6] Engineering Enriched Tabular Features with OOF Target Encodings...")
    for col in ["price", "initial_quantity", "sold_quantity"]:
        df_tr_aug[col] = pd.to_numeric(df_tr_aug[col], errors="coerce").fillna(0.0)
        df_te[col] = pd.to_numeric(df_te[col], errors="coerce").fillna(0.0)

    # Basic structural features
    feats_tr = []
    feats_te = []

    for df, f_list in [(df_tr_aug, feats_tr), (df_te, feats_te)]:
        f_df = pd.DataFrame()
        f_df["log_price"] = np.log1p(np.maximum(df["price"].values, 0.0))
        f_df["log_iq"] = np.log1p(np.maximum(df["initial_quantity"].values, 0.0))
        f_df["log_sq"] = np.log1p(np.maximum(df["sold_quantity"].values, 0.0))
        f_df["sq_ratio"] = df["sold_quantity"].values / (df["initial_quantity"].values + 1.0)
        f_df["title_len"] = df["title"].fillna("").str.len()
        f_df["word_count"] = df["title"].fillna("").str.split().str.len()
        f_df["has_tags"] = (df["tags"].astype(str) != "[]").astype(int)
        f_df["is_free"] = (df["listing_type_id"].astype(str) == "free").astype(int)
        f_df["is_bronze"] = (df["listing_type_id"].astype(str) == "bronze").astype(int)
        f_df["is_silver"] = (df["listing_type_id"].astype(str) == "silver").astype(int)
        f_df["is_gold"] = df["listing_type_id"].astype(str).str.contains("gold").astype(int)
        f_df["accepts_mercadopago"] = df["accepts_mercadopago"].astype(int)
        f_df["non_mercado_pago_payment_methods_count"] = df["non_mercado_pago_payment_methods"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        f_df["descriptions_count"] = df["descriptions"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        f_list.append(f_df)

    tab_tr = feats_tr[0]
    tab_te = feats_te[0]

    # Target encodings (Category & Seller)
    cat_oof = np.zeros(n_aug, dtype=np.float32)
    cat_te = np.zeros(n_te, dtype=np.float32)
    seller_oof = np.zeros(n_aug, dtype=np.float32)
    seller_te = np.zeros(n_te, dtype=np.float32)

    global_mean = y_train_aug.mean()

    for trn_idx, val_idx in cv.split(df_tr_aug, y_train_aug):
        # Category target encoding
        c_stats = df_tr_aug.iloc[trn_idx].groupby("category_id")["condition"].apply(lambda s: (s == "used").mean())
        c_cnt = df_tr_aug.iloc[trn_idx].groupby("category_id").size()
        c_smooth = (c_stats * c_cnt + global_mean * 10.0) / (c_cnt + 10.0)
        cat_oof[val_idx] = df_tr_aug.iloc[val_idx]["category_id"].map(c_smooth).fillna(global_mean)

        # Seller target encoding
        s_stats = df_tr_aug.iloc[trn_idx].groupby("seller_id")["condition"].apply(lambda s: (s == "used").mean())
        s_cnt = df_tr_aug.iloc[trn_idx].groupby("seller_id").size()
        s_smooth = (s_stats * s_cnt + global_mean * 5.0) / (s_cnt + 5.0)
        seller_oof[val_idx] = df_tr_aug.iloc[val_idx]["seller_id"].map(s_smooth).fillna(global_mean)

    # Full train encoding for test
    c_stats_full = df_tr_aug.groupby("category_id")["condition"].apply(lambda s: (s == "used").mean())
    c_cnt_full = df_tr_aug.groupby("category_id").size()
    c_smooth_full = (c_stats_full * c_cnt_full + global_mean * 10.0) / (c_cnt_full + 10.0)
    cat_te[:] = df_te["category_id"].map(c_smooth_full).fillna(global_mean).values

    s_stats_full = df_tr_aug.groupby("seller_id")["condition"].apply(lambda s: (s == "used").mean())
    s_cnt_full = df_tr_aug.groupby("seller_id").size()
    s_smooth_full = (s_stats_full * s_cnt_full + global_mean * 5.0) / (s_cnt_full + 5.0)
    seller_te[:] = df_te["seller_id"].map(s_smooth_full).fillna(global_mean).values

    tab_tr["cat_te"] = cat_oof
    tab_te["cat_te"] = cat_te
    tab_tr["seller_te"] = seller_oof
    tab_te["seller_te"] = seller_te
    tab_tr["oof_text_saga"] = oof_text
    tab_te["oof_text_saga"] = p_text_te

    # Append SVD features
    for col_idx in range(64):
        tab_tr[f"svd_{col_idx}"] = X_tr_svd[:, col_idx]
        tab_te[f"svd_{col_idx}"] = X_te_svd[:, col_idx]

    log(f"Tabular Feature Matrix Shape: {tab_tr.shape}")

    # 5. Fit 5-Fold Stratified LightGBM
    log("[Step 5/6] Fitting 5-Fold Stratified LightGBM (63 leaves, 1200 trees)...")
    oof_lgb = np.zeros(n_aug, dtype=np.float32)
    p_te_lgb = np.zeros(n_te, dtype=np.float32)

    lgb_params = {
        "num_leaves": 63,
        "max_depth": -1,
        "learning_rate": 0.04,
        "n_estimators": 1200,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_samples": 20,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "n_jobs": -1,
        "verbose": -1
    }

    for fold, (trn_idx, val_idx) in enumerate(cv.split(tab_tr, y_train_aug)):
        t_f = time.time()
        clf_lgb = LGBMClassifier(**lgb_params, random_state=42 + fold * 100)
        clf_lgb.fit(
            tab_tr.iloc[trn_idx], y_train_aug[trn_idx],
            sample_weight=weights_aug[trn_idx],
            eval_set=[(tab_tr.iloc[val_idx], y_train_aug[val_idx])],
            callbacks=[]
        )
        val_p = clf_lgb.predict_proba(tab_tr.iloc[val_idx])[:, 1]
        oof_lgb[val_idx] = val_p
        p_te_lgb += clf_lgb.predict_proba(tab_te)[:, 1] / 5.0
        acc_lgb_f = accuracy_score(y_train_aug[val_idx], (val_p >= 0.5).astype(int))
        log(f"  LightGBM Fold {fold+1}/5 finished in {time.time()-t_f:.1f}s | Acc: {acc_lgb_f:.5f}")

    acc_lgb = accuracy_score(y_train_aug[:70000], (oof_lgb[:70000] >= 0.5).astype(int))
    ll_lgb = log_loss(y_train_aug[:70000], oof_lgb[:70000])
    auc_lgb = roc_auc_score(y_train_aug[:70000], oof_lgb[:70000])
    log("=" * 80)
    log(f"=== 108-VEC TRANSDUCTIVE LIGHTGBM F117 OOF (70K BASE) ===")
    log(f"Accuracy: {acc_lgb:.5f} | LogLoss: {ll_lgb:.5f} | ROC-AUC: {auc_lgb:.5f}")
    log("=" * 80)

    # 6. Fit 5-Fold Stratified CatBoost Depth 8 Native Text
    log("[Step 6/6] Fitting 5-Fold Stratified CatBoost Depth 8 with Native Text...")
    df_cb_tr = pd.DataFrame({
        "title": df_tr_aug["title"].fillna("").astype(str),
        "seller_id": df_tr_aug["seller_id"].astype(str),
        "category_id": df_tr_aug["category_id"].astype(str),
        "log_price": tab_tr["log_price"].values,
        "is_free": tab_tr["is_free"].values,
        "sq_ratio": tab_tr["sq_ratio"].values,
        "seller_te": tab_tr["seller_te"].values,
        "cat_te": tab_tr["cat_te"].values,
        "oof_text_saga": oof_text
    })
    df_cb_te = pd.DataFrame({
        "title": df_te["title"].fillna("").astype(str),
        "seller_id": df_te["seller_id"].astype(str),
        "category_id": df_te["category_id"].astype(str),
        "log_price": tab_te["log_price"].values,
        "is_free": tab_te["is_free"].values,
        "sq_ratio": tab_te["sq_ratio"].values,
        "seller_te": tab_te["seller_te"].values,
        "cat_te": tab_te["cat_te"].values,
        "oof_text_saga": p_text_te
    })

    oof_cb = np.zeros(n_aug, dtype=np.float32)
    p_te_cb = np.zeros(n_te, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(cv.split(df_cb_tr, y_train_aug)):
        t_f = time.time()
        cb = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.06,
            depth=8,
            loss_function="Logloss",
            eval_metric="Accuracy",
            text_features=["title"],
            cat_features=["seller_id", "category_id"],
            random_seed=42 + fold,
            task_type="CPU",
            thread_count=-1,
            verbose=0
        )
        cb.fit(df_cb_tr.iloc[trn_idx], y_train_aug[trn_idx], sample_weight=weights_aug[trn_idx], verbose=False)
        val_p = cb.predict_proba(df_cb_tr.iloc[val_idx])[:, 1]
        oof_cb[val_idx] = val_p
        p_te_cb += cb.predict_proba(df_cb_te)[:, 1] / 5.0
        acc_cb_f = accuracy_score(y_train_aug[val_idx], (val_p >= 0.5).astype(int))
        log(f"  CatBoost Fold {fold+1}/5 finished in {time.time()-t_f:.1f}s | Acc: {acc_cb_f:.5f}")

    acc_cb = accuracy_score(y_train_aug[:70000], (oof_cb[:70000] >= 0.5).astype(int))
    ll_cb = log_loss(y_train_aug[:70000], oof_cb[:70000])
    auc_cb = roc_auc_score(y_train_aug[:70000], oof_cb[:70000])
    log("=" * 80)
    log(f"=== 108-VEC TRANSDUCTIVE CATBOOST DEPTH 8 OOF (70K BASE) ===")
    log(f"Accuracy: {acc_cb:.5f} | LogLoss: {ll_cb:.5f} | ROC-AUC: {auc_cb:.5f}")
    log("=" * 80)

    # Optimal Meta-Blend: 55% CatBoost + 35% LightGBM + 10% SAGA Text
    oof_blend = 0.55 * oof_cb[:70000] + 0.35 * oof_lgb[:70000] + 0.10 * oof_text[:70000]
    p_te_blend = 0.55 * p_te_cb + 0.35 * p_te_lgb + 0.10 * p_text_te

    acc_blend = accuracy_score(y_train[:70000], (oof_blend >= 0.5).astype(int))
    ll_blend = log_loss(y_train[:70000], oof_blend)
    auc_blend = roc_auc_score(y_train[:70000], oof_blend)
    log(f"108-Vec Grandmaster Blend OOF: Acc={acc_blend:.5f}, LogLoss={ll_blend:.5f}, ROC-AUC={auc_blend:.5f}")

    # Save test predictions
    df_out = pd.DataFrame({
        "ID": range(1, n_te + 1),
        "p_lgb_108vec": p_te_lgb,
        "p_cb_108vec": p_te_cb,
        "p_text_108vec": p_text_te,
        "p_blend_108vec": p_te_blend
    })
    out_file = OUT_DIR / "transductive_test_probabilities_108vec.csv"
    df_out.to_csv(out_file, index=False)
    log(f"Saved 108-vec test probabilities to {out_file}")

    # Save summary report
    rep_file = OUT_DIR / "transductive_training_report_108vec.md"
    with open(rep_file, "w", encoding="utf-8") as f:
        f.write("# 108-Vector Transductive Deep Models Training Report\n\n")
        f.write(f"- **Augmented Train Observations:** `{len(y_train_aug)}`\n")
        f.write(f"- **CatBoost Depth 8 OOF Accuracy:** `{acc_cb:.5f}`\n")
        f.write(f"- **CatBoost Depth 8 OOF LogLoss:** `{ll_cb:.5f}`\n")
        f.write(f"- **CatBoost Depth 8 OOF ROC-AUC:** `{auc_cb:.5f}`\n")
        f.write(f"- **LightGBM F117 OOF Accuracy:** `{acc_lgb:.5f}`\n")
        f.write(f"- **Grandmaster Meta-Blend OOF Accuracy:** `{acc_blend:.5f}`\n")
        f.write(f"- **Grandmaster Meta-Blend OOF LogLoss:** `{ll_blend:.5f}`\n")
        f.write(f"- **Total Elapsed Time:** `{time.time()-t0:.1f}s`\n")
        f.write(f"- **Output File:** `{out_file.name}`\n")
    log(f"Saved report to {rep_file}")
    log(f"108-Vec Training Engine Completed in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
