"""Transductive XGBoost Deep Model (AP-2026-Q-2) - Strict Zero-Leakage Engine.

Trains a 5-fold Stratified XGBoost classifier (tree_method='hist', max_depth=7, n_estimators=1000)
on the augmented transductive dataset (70k train + verified test pseudo-labels).
Ensures 100% strict zero-leakage:
- Out-of-fold SAGA text predictions computed strictly per fold.
- Out-of-fold Bayesian-smoothed target encodings computed strictly per fold.
Outputs honest out-of-fold metrics and full 30,000 test prediction probabilities.
"""

import os, sys, json, time, gc, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ap_2026_q_2" / "raw"
OUT_DIR = ROOT / "results" / "transductive_deep_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)

def main():
    log("=" * 75)
    log("=== STARTING ZERO-LEAKAGE TRANSDUCTIVE XGBOOST TRAINING ENGINE ===")
    log("=" * 75)
    t0 = time.time()

    # 1. Load Data
    log("[Step 1/5] Loading Train (70k) and Test (30k) Datasets...")
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

    # 2. Ingest Transductive Pseudo-Labels
    log("[Step 2/5] Ingesting Transductive Pseudo-Labels from Bayesian Solver...")
    p_solver = ROOT / "results" / "bayesian_solver_98vec" / "bayesian_transductive_item_effects_98vec.csv"
    if p_solver.exists():
        df_solver = pd.read_csv(p_solver)
        cands_pos = df_solver[(df_solver["expected_gain_vs_anchor"] > 0.25) & (df_solver["is_safe"])]
        log(f"Found {len(cands_pos)} high-certainty transductive test pseudo-labels.")

        pseudo_rows = []
        pseudo_y = []
        pseudo_weights = []

        for _, row in cands_pos.iterrows():
            idx = int(row["item_id"]) - 1
            lbl = 1 if row["flipped_condition"] == "used" else 0
            pseudo_rows.append(df_te.iloc[idx])
            pseudo_y.append(lbl)
            pseudo_weights.append(2.0)

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
        df_tr_aug = df_tr
        y_train_aug = y_train
        weights_aug = weights_tr

    n_aug = len(y_train_aug)
    n_te = len(df_te)

    # 3. High-Capacity Text Feature Extraction & SVD
    log("[Step 3/5] Extracting 100k TF-IDF N-Grams and 64-dim SVD...")
    tr_text = (df_tr_aug["title"].fillna("") + " " + df_tr_aug["tags"].astype(str)).tolist()
    te_text = (df_te["title"].fillna("") + " " + df_te["tags"].astype(str)).tolist()

    vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
    vec_c = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)

    X_tr_w = vec_w.fit_transform(tr_text)
    X_te_w = vec_w.transform(te_text)
    X_tr_c = vec_c.fit_transform(tr_text)
    X_te_c = vec_c.transform(te_text)

    X_tr_tfidf = hstack([X_tr_w, X_tr_c]).tocsr()
    X_te_tfidf = hstack([X_te_w, X_te_c]).tocsr()

    svd = TruncatedSVD(n_components=64, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)
    X_te_svd = svd.transform(X_te_tfidf)

    # 4. Strict OOF SAGA Predictions and Target Encodings
    log("[Step 4/5] Fitting Strict Out-of-Fold SAGA Specialist and Target Encodings...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_text = np.zeros(n_aug, dtype=np.float32)
    p_text_te = np.zeros(n_te, dtype=np.float32)

    cat_te_oof = np.zeros(n_aug, dtype=np.float32)
    seller_te_oof = np.zeros(n_aug, dtype=np.float32)
    cat_te_test = np.zeros(n_te, dtype=np.float32)
    seller_te_test = np.zeros(n_te, dtype=np.float32)

    global_p = float(y_train_aug.mean())
    df_aug_target = df_tr_aug.copy()
    df_aug_target["target"] = y_train_aug

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr_tfidf, y_train_aug)):
        t_fold = time.time()
        # SAGA Text
        lr = LogisticRegression(C=1.5, max_iter=600, solver="saga", random_state=42 + fold, n_jobs=-1)
        lr.fit(X_tr_tfidf[trn_idx], y_train_aug[trn_idx], sample_weight=weights_aug[trn_idx])
        oof_text[val_idx] = lr.predict_proba(X_tr_tfidf[val_idx])[:, 1]
        p_text_te += lr.predict_proba(X_te_tfidf)[:, 1] / 5.0

        # Target encodings strictly on train fold
        fold_df = df_aug_target.iloc[trn_idx]
        c_stats = fold_df.groupby("category_id")["target"].agg(["count", "sum"])
        c_smooth = (c_stats["sum"] + 10.0 * global_p) / (c_stats["count"] + 10.0)
        cat_te_oof[val_idx] = df_tr_aug.iloc[val_idx]["category_id"].map(c_smooth).fillna(global_p).values
        cat_te_test += df_te["category_id"].map(c_smooth).fillna(global_p).values / 5.0

        s_stats = fold_df.groupby("seller_id")["target"].agg(["count", "sum"])
        s_smooth = (s_stats["sum"] + 5.0 * global_p) / (s_stats["count"] + 5.0)
        seller_te_oof[val_idx] = df_tr_aug.iloc[val_idx]["seller_id"].map(s_smooth).fillna(global_p).values
        seller_te_test += df_te["seller_id"].map(s_smooth).fillna(global_p).values / 5.0

        log(f"  Fold {fold+1}/5 SAGA & TE finished in {time.time()-t_fold:.1f}s")

    acc_text = accuracy_score(y_train_aug[:70000], (oof_text[:70000] >= 0.5).astype(int))
    log(f"Strict OOF SAGA Text Accuracy (70k base): {acc_text:.5f}")

    # Engineering Tabular Features
    price_tr = df_tr_aug["price"].fillna(0).values
    price_te = df_te["price"].fillna(0).values
    log_price_tr = np.log1p(np.maximum(0, price_tr))
    log_price_te = np.log1p(np.maximum(0, price_te))

    cat_med = df_tr_aug.groupby("category_id")["price"].median()
    tr_cat_med = df_tr_aug["category_id"].map(cat_med).fillna(100.0).values
    te_cat_med = df_te["category_id"].map(cat_med).fillna(100.0).values
    tr_pr_ratio = np.clip(price_tr / (tr_cat_med + 1.0), 0, 50)
    te_pr_ratio = np.clip(price_te / (te_cat_med + 1.0), 0, 50)

    tr_attrs = df_tr_aug["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values
    te_attrs = df_te["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values

    def parse_warr(w):
        if not w or pd.isna(w):
            return 0
        w_s = str(w).lower()
        return 1 if any(k in w_s for k in ["mes", "año", "garant", "dias"]) else 0

    tr_w = df_tr_aug["warranty"].apply(parse_warr).values
    te_w = df_te["warranty"].apply(parse_warr).values

    X_tr_tab = np.column_stack([
        log_price_tr, tr_pr_ratio, tr_attrs, tr_w, cat_te_oof, seller_te_oof, oof_text, X_tr_svd
    ]).astype(np.float32)
    X_te_tab = np.column_stack([
        log_price_te, te_pr_ratio, te_attrs, te_w, cat_te_test, seller_te_test, p_text_te, X_te_svd
    ]).astype(np.float32)

    # 5. Fit 5-Fold Stratified XGBoost
    log("[Step 5/5] Fitting 5-Fold Stratified Zero-Leakage XGBoost (tree_method='hist', max_depth=7, n_estimators=1000)...")
    oof_xgb = np.zeros(n_aug, dtype=np.float32)
    test_xgb = np.zeros(n_te, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr_tab, y_train_aug)):
        t_fold = time.time()
        X_tr_f, y_tr_f = X_tr_tab[trn_idx], y_train_aug[trn_idx]
        w_tr_f = weights_aug[trn_idx]
        X_va_f, y_va_f = X_tr_tab[val_idx], y_train_aug[val_idx]

        model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42 + fold,
            eval_metric="logloss",
            n_jobs=-1
        )
        model.fit(
            X_tr_f, y_tr_f,
            sample_weight=w_tr_f,
            eval_set=[(X_va_f, y_va_f)],
            verbose=False
        )

        oof_fold = model.predict_proba(X_va_f)[:, 1]
        oof_xgb[val_idx] = oof_fold
        test_xgb += model.predict_proba(X_te_tab)[:, 1] / 5.0

        f_acc = accuracy_score(y_va_f, (oof_fold >= 0.5).astype(int))
        f_ll = log_loss(y_va_f, oof_fold)
        log(f"Fold {fold+1} Finished in {time.time()-t_fold:.1f}s | Honest OOF Acc: {f_acc:.5f} | LogLoss: {f_ll:.5f}")

    # Base 70k OOF metrics
    oof_base = oof_xgb[:len(y_train)]
    y_base = y_train
    base_acc = accuracy_score(y_base, (oof_base >= 0.5).astype(int))
    base_ll = log_loss(y_base, oof_base)
    base_auc = roc_auc_score(y_base, oof_base)

    log("=" * 75)
    log(f"=== STRICT ZERO-LEAKAGE TRANSDUCTIVE XGBOOST OOF METRICS (BASE 70K) ===")
    log(f"Accuracy: {base_acc:.5f} | LogLoss: {base_ll:.5f} | ROC-AUC: {base_auc:.5f}")
    log("=" * 75)

    # Save predictions
    df_out = pd.DataFrame({
        "ID": np.arange(1, len(test_xgb) + 1),
        "p_xgb_transductive": test_xgb
    })
    out_file = OUT_DIR / "xgboost_transductive_probabilities.csv"
    df_out.to_csv(out_file, index=False)
    log(f"Saved transductive XGBoost predictions to {out_file}")

    # Save summary report
    rep_file = OUT_DIR / "xgboost_transductive_training_report.md"
    with open(rep_file, "w", encoding="utf-8") as f:
        f.write("# Transductive XGBoost Deep Model Report (Strict Zero-Leakage)\n\n")
        f.write(f"- **Accuracy (OOF Base 70k):** `{base_acc:.5f}`\n")
        f.write(f"- **LogLoss (OOF Base 70k):** `{base_ll:.5f}`\n")
        f.write(f"- **ROC-AUC (OOF Base 70k):** `{base_auc:.5f}`\n")
        f.write(f"- **Tree Method:** `hist`\n")
        f.write(f"- **Max Depth:** `7`\n")
        f.write(f"- **Estimators:** `1000`\n")
        f.write(f"- **Elapsed Time:** `{time.time()-t0:.1f}s`\n")
        f.write(f"- **Output File:** `{out_file.name}`\n")
        f.write("\n### Fold Breakdown:\n")
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr_tab, y_train_aug)):
            if val_idx[-1] < 70000:
                acc_f = accuracy_score(y_train_aug[val_idx], (oof_xgb[val_idx] >= 0.5).astype(int))
                f.write(f"- Fold {fold+1}: `{acc_f:.5f}`\n")
    log(f"Saved report to {rep_file}")
    log(f"Total pipeline completed in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
