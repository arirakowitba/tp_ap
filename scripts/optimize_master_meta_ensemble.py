"""Master Multi-Modal Meta-Ensemble Optimizer.
Combines:
1. High-Dimensional Sparse Sub-Word Linear Model (122k n-grams)
2. Category & Domain Specialists (4 macro clusters)
3. Deep Multi-Layer Perceptron (512-256-128)
4. High-Capacity Gradient Boosted Trees (LightGBM, XGBoost, CatBoost)
Optimizes blending weights via Nelder-Mead on OOF log-loss/accuracy.
Generates final meta-ensemble predictions for test set.
"""
import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from scipy.optimize import minimize
import os

print("=== BUILDING MASTER MULTI-MODAL META-ENSEMBLE ===")

# 1. Load Ground Truth y_train
with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]
y_train = np.array([1 if d["condition"] == "new" else 0 for d in train_data], dtype=np.int8)

# 2. Load all 4 prediction streams
p1 = np.load("results/deep_parallel_20261109/sparse_linear_predictions.npz")
p2 = np.load("results/deep_parallel_20261109/category_specialist_predictions.npz")
p3 = np.load("results/deep_parallel_20261109/mlp_predictions.npz")
p4 = np.load("results/deep_parallel_20261109/tree_predictions.npz")

oof_models = {
    "sparse_linear": p1["blend_oof"],
    "cat_specialist": p2["oof_cluster"],
    "deep_mlp": p3["oof_mlp"],
    "lightgbm": p4["oof_lgb"],
    "xgboost": p4["oof_xgb"],
    "catboost": p4["oof_cat"]
}

test_models = {
    "sparse_linear": p1["blend_test"],
    "cat_specialist": p2["test_cluster"],
    "deep_mlp": p3["test_mlp"],
    "lightgbm": p4["test_lgb"],
    "xgboost": p4["test_xgb"],
    "catboost": p4["test_cat"]
}

print("Individual Model Performance (OOF):")
for name, pred in oof_models.items():
    acc = accuracy_score(y_train, (pred >= 0.5).astype(int))
    auc = roc_auc_score(y_train, pred)
    loss = log_loss(y_train, np.clip(pred, 1e-6, 1-1e-6))
    print(f"  {name:15s} | Acc: {acc:.5f} ({int(acc*70000)}/70000) | AUC: {auc:.5f} | LogLoss: {loss:.5f}")

# 3. Optimize Weights via Minimizing Log Loss
model_names = list(oof_models.keys())
OOF_matrix = np.column_stack([oof_models[m] for m in model_names])
TEST_matrix = np.column_stack([test_models[m] for m in model_names])

def loss_func(weights):
    w = np.exp(weights) / np.sum(np.exp(weights)) # softmax to enforce sum to 1, positive
    pred = OOF_matrix @ w
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return log_loss(y_train, pred)

init_weights = np.zeros(len(model_names))
res = minimize(loss_func, init_weights, method="Nelder-Mead", options={"maxiter": 2000})

opt_w = np.exp(res.x) / np.sum(np.exp(res.x))

print("\nOptimized Meta-Ensemble Weights:")
for name, w in zip(model_names, opt_w):
    print(f"  {name:15s}: {w*100:5.2f}%")

final_oof_probs = OOF_matrix @ opt_w
final_test_probs = TEST_matrix @ opt_w

final_acc = accuracy_score(y_train, (final_oof_probs >= 0.5).astype(int))
final_auc = roc_auc_score(y_train, final_oof_probs)
final_loss = log_loss(y_train, np.clip(final_oof_probs, 1e-6, 1-1e-6))

print("\n=== FINAL MASTER META-ENSEMBLE RESULTS ===")
print(f"Final OOF Accuracy: {final_acc:.5f} ({int(final_acc*70000)}/70000 correct)")
print(f"Final OOF ROC AUC : {final_auc:.5f}")
print(f"Final OOF Log Loss: {final_loss:.5f}")

# Save master ensemble
os.makedirs("results/deep_parallel_20261109", exist_ok=True)
np.savez_compressed(
    "results/deep_parallel_20261109/master_meta_ensemble.npz",
    final_oof_probs=final_oof_probs,
    final_test_probs=final_test_probs,
    model_names=model_names,
    weights=opt_w
)

# 4. Compare vs EVT-143 (Current Best, Score 0.94761)
anchor = pd.read_csv("submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv")
p_anc = anchor.set_index("ID")["condition"].to_dict()

# Calculate consensus and disagreement vs anchor
ensemble_preds = np.where(final_test_probs >= 0.5, "new", "used")
disagreements = []
with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_items = [json.loads(line) for line in f]

for i in range(30000):
    iid = i + 1
    anc_val = p_anc[iid]
    ens_val = ensemble_preds[i]
    prob = final_test_probs[i]
    margin = abs(prob - 0.5)
    if ens_val != anc_val:
        disagreements.append({
            "id": iid,
            "anchor": anc_val,
            "ensemble": ens_val,
            "p_new": float(prob),
            "margin": float(margin),
            "price": test_items[i].get("price", 0),
            "title": test_items[i].get("title", ""),
            "category_id": test_items[i].get("category_id", "")
        })

print(f"\nTotal Disagreements between Master Meta-Ensemble and EVT-143: {len(disagreements)} / 30000")
disagreements.sort(key=lambda x: x["margin"], reverse=True)

print("\nTop 15 Highest-Confidence Disagreements (Ensemble Certainty > 90%):")
for d in disagreements[:15]:
    print(f"ID {d['id']:5d} | Anc: {d['anchor']:4s} -> Ens: {d['ensemble']:4s} (P_new: {d['p_new']:.4f}, Margin: {d['margin']:.4f}) | ${d['price']:8.1f} | {d['title'][:45]}")

with open("results/deep_parallel_20261109/ensemble_disagreements_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "total_disagreements": len(disagreements),
        "disagreements": disagreements
    }, f, indent=2)

print("\nWrote summary to results/deep_parallel_20261109/ensemble_disagreements_summary.json")
