"""Train High-Capacity LightGBM on Full 117-Feature Engineered Matrix F.

AP-2026-Q-2 Iteration 3:
Trains multi-seed LightGBM on the full 117-column tabular feature matrix F (feats.pkl),
computes 5-fold cross-validation metrics, and generates test probability predictions.
"""

import os, sys, json, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ap_2026_q_2" / "raw"
OUT_DIR = ROOT / "results" / "full_f117_lgbm_20261105"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== TRAINING FULL 117-FEATURE LIGHTGBM ENSEMBLE ===", flush=True)
    t0 = time.time()

    with open(DATA_DIR / "feats.pkl", "rb") as f:
        feats = pickle.load(f)

    F = feats["F"]
    y = feats["y"]
    if isinstance(y, pd.Series):
        y = y.values
    y_train = (y == 1).astype(int) if set(np.unique(y)).issubset({0, 1}) else (y == "used").astype(int)

    X_train = F.iloc[:70000].copy()
    X_test = F.iloc[70000:].copy()

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train mean: {y_train.mean():.4f}", flush=True)

    # 5-fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(X_test))

    params = {
        "n_estimators": 900,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": 14,
        "verbose": -1
    }

    print("\nStarting 5-Fold Cross-Validation...", flush=True)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        t_f = time.time()
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[]
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(X_test)[:, 1] / 5.0

        fold_acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
        fold_ll = log_loss(y_val, val_probs)
        fold_auc = roc_auc_score(y_val, val_probs)
        print(f"Fold {fold+1}/5 - Acc: {fold_acc:.5f}, LogLoss: {fold_ll:.5f}, AUC: {fold_auc:.5f} ({time.time()-t_f:.1f}s)", flush=True)

    oof_acc = accuracy_score(y_train, (oof_preds >= 0.5).astype(int))
    oof_ll = log_loss(y_train, oof_preds)
    oof_auc = roc_auc_score(y_train, oof_preds)
    print("\n" + "="*50, flush=True)
    print(f"OOF Overall - Accuracy: {oof_acc:.5f}, LogLoss: {oof_ll:.5f}, AUC: {oof_auc:.5f}", flush=True)
    print("="*50, flush=True)

    # Save OOF and Test probabilities
    np.save(OUT_DIR / "oof_probs_f117_lgbm.npy", oof_preds)
    np.save(OUT_DIR / "test_probs_f117_lgbm.npy", test_preds)

    df_test_out = pd.DataFrame({
        "ID": range(1, 30001),
        "p_f117_lgbm": test_preds
    })
    df_test_out.to_csv(OUT_DIR / "test_probabilities_f117_lgbm.csv", index=False)
    print(f"Saved test probabilities to {OUT_DIR / 'test_probabilities_f117_lgbm.csv'}", flush=True)

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "model": "LGBMClassifier-F117",
        "n_features": X_train.shape[1],
        "n_folds": 5,
        "oof_accuracy": float(oof_acc),
        "oof_logloss": float(oof_ll),
        "oof_auc": float(oof_auc),
        "test_prob_mean": float(test_preds.mean()),
        "test_prob_std": float(test_preds.std()),
        "elapsed_seconds": time.time() - t0
    }
    with open(OUT_DIR / "f117_lgbm_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Script completed in {time.time()-t0:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
