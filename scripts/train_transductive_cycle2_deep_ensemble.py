"""Transductive Self-Training / Pseudo-Labeling Deep Ensemble (Cycle 2).
Uses the high-confidence predictions from Master Meta-Ensemble (P > 0.985 or P < 0.015)
to augment the 70,000 train items with ~18,000 pristine test pseudo-labels.
Evaluates 5-fold CV strictly on the 70,000 ground-truth labels.
Trains:
1. High-Capacity LightGBM (1500 trees, 63 leaves, 16 threads)
2. High-Capacity XGBoost (1500 trees, max_depth 7, hist method, 16 threads)
3. Deep Multi-Layer Perceptron (512-256-128)
Strictly leak-free (zero seller-id features).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import os

start_time = time.time()
print("=== CONTINUOUS OVERNIGHT PHASE 1: TRANSDUCTIVE SELF-TRAINING ENSEMBLE ===")

# 1. Load Data
print("Loading train and test datasets...")
with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

train_titles = [d.get("title", "") for d in train_data]
test_titles = [d.get("title", "") for d in test_data]
y_train_orig = np.array([1 if d["condition"] == "new" else 0 for d in train_data], dtype=np.int8)

train_prices = np.array([float(d.get("price", 0)) for d in train_data])
test_prices = np.array([float(d.get("price", 0)) for d in test_data])

train_cats = [d.get("category_id", "") for d in train_data]
test_cats = [d.get("category_id", "") for d in test_data]

# 2. Load Master Meta-Ensemble Test Probabilities
ens = np.load("results/deep_parallel_20261109/master_meta_ensemble.npz")
test_probs = ens["final_test_probs"]

# Identify Pseudo-Labels
# P >= 0.985 -> new (1), P <= 0.015 -> used (0)
pseudo_new_idx = np.where(test_probs >= 0.985)[0]
pseudo_used_idx = np.where(test_probs <= 0.015)[0]
pseudo_indices = np.concatenate([pseudo_new_idx, pseudo_used_idx])
pseudo_labels = np.concatenate([
    np.ones(len(pseudo_new_idx), dtype=np.int8),
    np.zeros(len(pseudo_used_idx), dtype=np.int8)
])

print(f"Master Meta-Ensemble Test Predictions Loaded: {len(test_probs)}")
print(f"Selected High-Certainty Pseudo-Labels: {len(pseudo_indices)} / 30000 ({len(pseudo_indices)/300:.1f}%)")
print(f"  Pseudo NEW  (P >= 0.985): {len(pseudo_new_idx)}")
print(f"  Pseudo USED (P <= 0.015): {len(pseudo_used_idx)}")

# 3. Build Augmented Dataset
pseudo_titles = [test_titles[i] for i in pseudo_indices]
pseudo_prices = test_prices[pseudo_indices]
pseudo_cats = [test_cats[i] for i in pseudo_indices]
pseudo_data = [test_data[i] for i in pseudo_indices]

all_train_titles = train_titles + pseudo_titles
all_train_prices = np.concatenate([train_prices, pseudo_prices])
all_train_cats = train_cats + pseudo_cats
all_train_data = train_data + pseudo_data
y_augmented = np.concatenate([y_train_orig, pseudo_labels])

print(f"Total Augmented Training Set: {len(all_train_titles)} items (70,000 real + {len(pseudo_indices)} pseudo)")

# 4. Extract Text Features: 180 TruncatedSVD from 150,000 TF-IDF features
print("\nExtracting 180 TruncatedSVD components over 150,000 TF-IDF features...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150000, sublinear_tf=True)
X_text_all = tfidf.fit_transform(all_train_titles + test_titles)

svd = TruncatedSVD(n_components=180, random_state=42)
X_svd_all = svd.fit_transform(X_text_all)
print(f"SVD Explained Variance Ratio sum: {svd.explained_variance_ratio_.sum():.4f}")

N_aug = len(all_train_titles)
N_test = len(test_titles)
X_aug_svd = X_svd_all[:N_aug]
X_test_svd = X_svd_all[N_aug:]

# 5. Extract Leak-Free Tabular Features
print("Engineering tabular & price features...")
df_cats = pd.DataFrame({"cat": all_train_cats, "price": all_train_prices})
cat_stats = df_cats.groupby("cat")["price"].agg(["median", "mean", "std"]).to_dict("index")

used_cues = ["usad", "reacond", "buen estado", "impecable estado", "repuesto", "despiece", "partes", "detalles", "falla", "antigu", "retro", "vintage"]
new_cues = ["nuevo", "nueva", "nuevas", "nuevos", "blister", "sellad", "cerrad", "estrenar", "0km", "0 km", "original", "garantia", "caja sellada"]

def extract_features(data, titles, prices, cats):
    feats = []
    for d, t, p, cat in zip(data, titles, prices, cats):
        t_lower = t.lower()
        log_p = np.log1p(max(0, p))
        st = cat_stats.get(cat, {"median": 300.0, "mean": 600.0, "std": 800.0})
        med = st["median"]
        mean_p = st["mean"]
        std_p = st["std"] if not np.isnan(st["std"]) and st["std"] > 0 else 1.0
        
        diff_med = (p - med) / (med + 1.0)
        z_score = (p - mean_p) / std_p
        
        has_war = 1.0 if d.get("warranty") else 0.0
        free_ship = 1.0 if d.get("shipping", {}).get("free_shipping") else 0.0
        is_gold = 1.0 if "gold" in d.get("listing_type_id", "") else 0.0
        is_bronze = 1.0 if "bronze" in d.get("listing_type_id", "") else 0.0
        is_free = 1.0 if "free" in d.get("listing_type_id", "") else 0.0
        
        title_len = len(t)
        word_count = len(t.split())
        num_count = sum(c.isdigit() for c in t)
        upper_count = sum(c.isupper() for c in t)
        
        u_flags = [1.0 if cue in t_lower else 0.0 for cue in used_cues]
        n_flags = [1.0 if cue in t_lower else 0.0 for cue in new_cues]
        
        row = [log_p, diff_med, z_score, has_war, free_ship, is_gold, is_bronze, is_free,
               title_len, word_count, num_count, upper_count] + u_flags + n_flags
        feats.append(row)
    return np.array(feats, dtype=np.float32)

tab_aug = extract_features(all_train_data, all_train_titles, all_train_prices, all_train_cats)
tab_test = extract_features(test_data, test_titles, test_prices, test_cats)

X_aug_full = np.hstack([X_aug_svd, tab_aug])
X_test_full = np.hstack([X_test_svd, tab_test])
print(f"Full Augmented Feature Matrix: {X_aug_full.shape}, Test Feature Matrix: {X_test_full.shape}")

# 6. Stratified 5-Fold Cross Validation
# IMPORTANT: Validation fold is STRICTLY evaluated on the real 70,000 train labels!
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_trans_lgb = np.zeros(70000, dtype=np.float32)
test_trans_lgb = np.zeros(N_test, dtype=np.float32)

oof_trans_xgb = np.zeros(70000, dtype=np.float32)
test_trans_xgb = np.zeros(N_test, dtype=np.float32)

oof_trans_mlp = np.zeros(70000, dtype=np.float32)
test_trans_mlp = np.zeros(N_test, dtype=np.float32)

print("\nTraining 5-Fold Transductive Ensemble across 16 threads...")
for fold, (train_real_idx, val_real_idx) in enumerate(skf.split(X_aug_full[:70000], y_train_orig)):
    f_start = time.time()
    
    # Train set for fold: real train subset + ALL pseudo-labeled test items
    fold_train_indices = np.concatenate([train_real_idx, np.arange(70000, N_aug)])
    
    X_tr = X_aug_full[fold_train_indices]
    y_tr = y_augmented[fold_train_indices]
    
    # Validation set: STRICTLY the unaugmented real ground truth
    X_va = X_aug_full[val_real_idx]
    y_va = y_train_orig[val_real_idx]
    
    # Model 1: LightGBM (1500 trees, 63 leaves)
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=100 + fold,
        n_jobs=16,
        verbose=-1
    )
    clf_lgb.fit(X_tr, y_tr)
    v_lgb = clf_lgb.predict_proba(X_va)[:, 1]
    oof_trans_lgb[val_real_idx] = v_lgb
    test_trans_lgb += clf_lgb.predict_proba(X_test_full)[:, 1] / 5.0
    
    # Model 2: XGBoost (1500 trees, max_depth 7, hist)
    clf_xgb = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        n_jobs=16,
        random_state=100 + fold,
        eval_metric="logloss"
    )
    clf_xgb.fit(X_tr, y_tr)
    v_xgb = clf_xgb.predict_proba(X_va)[:, 1]
    oof_trans_xgb[val_real_idx] = v_xgb
    test_trans_xgb += clf_xgb.predict_proba(X_test_full)[:, 1] / 5.0
    
    # Model 3: Deep MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=150,
        random_state=100 + fold
    )
    mlp.fit(X_tr, y_tr)
    v_mlp = mlp.predict_proba(X_va)[:, 1]
    oof_trans_mlp[val_real_idx] = v_mlp
    test_trans_mlp += mlp.predict_proba(X_test_full)[:, 1] / 5.0
    
    acc_lgb = accuracy_score(y_va, (v_lgb >= 0.5).astype(int))
    acc_xgb = accuracy_score(y_va, (v_xgb >= 0.5).astype(int))
    acc_mlp = accuracy_score(y_va, (v_mlp >= 0.5).astype(int))
    print(f"Fold {fold+1}/5 ({time.time()-f_start:.1f}s) | Trans-LGB: {acc_lgb:.5f} | Trans-XGB: {acc_xgb:.5f} | Trans-MLP: {acc_mlp:.5f}")

# 7. Evaluate Overall OOF on Real Ground Truth
acc_lgb_tot = accuracy_score(y_train_orig, (oof_trans_lgb >= 0.5).astype(int))
acc_xgb_tot = accuracy_score(y_train_orig, (oof_trans_xgb >= 0.5).astype(int))
acc_mlp_tot = accuracy_score(y_train_orig, (oof_trans_mlp >= 0.5).astype(int))

trans_blend_oof = 0.45 * oof_trans_lgb + 0.40 * oof_trans_xgb + 0.15 * oof_trans_mlp
trans_blend_test = 0.45 * test_trans_lgb + 0.40 * test_trans_xgb + 0.15 * test_trans_mlp
acc_blend_tot = accuracy_score(y_train_orig, (trans_blend_oof >= 0.5).astype(int))
auc_blend_tot = roc_auc_score(y_train_orig, trans_blend_oof)
loss_blend_tot = log_loss(y_train_orig, np.clip(trans_blend_oof, 1e-6, 1-1e-6))

print("\n=== TRANSDUCTIVE CYCLE 2 OVERALL RESULTS (ON REAL GROUND TRUTH) ===")
print(f"Trans-LGB OOF Acc: {acc_lgb_tot:.5f} (AUC: {roc_auc_score(y_train_orig, oof_trans_lgb):.5f})")
print(f"Trans-XGB OOF Acc: {acc_xgb_tot:.5f} (AUC: {roc_auc_score(y_train_orig, oof_trans_xgb):.5f})")
print(f"Trans-MLP OOF Acc: {acc_mlp_tot:.5f} (AUC: {roc_auc_score(y_train_orig, oof_trans_mlp):.5f})")
print(f"Transductive Blend OOF Acc: {acc_blend_tot:.5f} ({int(acc_blend_tot*70000)}/70000 correct)")
print(f"Transductive Blend ROC AUC: {auc_blend_tot:.5f}")
print(f"Transductive Blend LogLoss: {loss_blend_tot:.5f}")

# Save artifacts
os.makedirs("results/transductive_cycle2_20261110", exist_ok=True)
np.savez_compressed(
    "results/transductive_cycle2_20261110/transductive_cycle2_predictions.npz",
    oof_trans_lgb=oof_trans_lgb, test_trans_lgb=test_trans_lgb,
    oof_trans_xgb=oof_trans_xgb, test_trans_xgb=test_trans_xgb,
    oof_trans_mlp=oof_trans_mlp, test_trans_mlp=test_trans_mlp,
    trans_blend_oof=trans_blend_oof,
    trans_blend_test=trans_blend_test
)

with open("results/transductive_cycle2_20261110/training_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_augmented": len(all_train_titles),
        "pseudo_labels_count": len(pseudo_indices),
        "oof_accuracy_blend": acc_blend_tot,
        "oof_auc_blend": auc_blend_tot,
        "oof_logloss_blend": loss_blend_tot,
        "runtime_seconds": time.time() - start_time
    }, f, indent=2)

print(f"\nSaved artifacts to results/transductive_cycle2_20261110/ ({time.time()-start_time:.1f}s)")
