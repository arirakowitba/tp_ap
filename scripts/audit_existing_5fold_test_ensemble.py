"""Audit existing 16 5-fold test probabilities across models."""
import numpy as np
from pathlib import Path
import pandas as pd

ROOT = Path(".")
results_dir = ROOT / "results"

models = [
    "catboost_depth8_5fold",
    "expanded_feedback_catboost_5fold",
    "expanded_feedback_extratrees_5fold",
    "expanded_feedback_lgbm_5fold",
    "expanded_feedback_mlp_5fold",
    "expanded_feedback_xgb_5fold",
    "extratrees_5fold",
    "high_capacity_lgbm_5fold",
    "high_capacity_lgbm_feedback_5fold",
    "iterative_feedback_lgbm_5fold",
    "lgbm_feedback_5fold",
    "meta_stacking_ensemble",
    "mlp_neural_5fold",
    "sparse_linear_5fold",
    "xgboost_diversity_5fold",
    "xgboost_feedback_5fold",
]

loaded = {}
for m in models:
    p = results_dir / m / "test_probabilities.npy"
    if p.exists():
        arr = np.load(p)
        # Check shape
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr[:, 1]
        elif arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        loaded[m] = arr
        print(f"Loaded {m:35s}: shape {arr.shape}, mean {arr.mean():.4f}, min {arr.min():.4f}, max {arr.max():.4f}")

# Load test data and EVT-132
df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
y_evt = (df_evt["condition"] == "used").astype(int).values

mat = np.column_stack([loaded[m] for m in loaded])
mean_p = np.mean(mat, axis=1)
median_p = np.median(mat, axis=1)
min_p = np.min(mat, axis=1)
std_p = np.std(mat, axis=1)

print("\nEnsemble Matrix Shape:", mat.shape)
print(f"Number of models in meta-ensemble: {len(loaded)}")

# Load test jsonlines to inspect titles
import json
with open("data/ap_2026_q_2/raw/test_data.jsonlines", "r", encoding="utf-8") as f:
    test_items = [json.loads(line) for line in f]
test_titles = [item.get("title", "") for item in test_items]
test_prices = [item.get("price", 0) for item in test_items]

# Check Slot 1 items
s1 = pd.read_csv("submissions/portfolio_20261109_podium_ladder/submission_slot1_top20_rank6_probe.csv")
diff_indices = s1[s1['condition'] != df_evt['condition']].index.tolist()

print("\n=== SLOT 1 (TOP 20 PROBE) VALIDATION ACROSS 16 DIVERSE MODELS ===")
for idx in diff_indices:
    row_id = df_evt.loc[idx, "ID"]
    title = test_titles[idx]
    p_mean = mean_p[idx]
    p_min = min_p[idx]
    p_std = std_p[idx]
    n_votes_used = np.sum(mat[idx] > 0.5)
    print(f"Index {idx:5d} | ID {row_id:6d} | Mean_P: {p_mean:.4f} | Min_P: {p_min:.4f} | Std: {p_std:.4f} | Votes: {n_votes_used}/16 | {title[:55]}")

# Now let's search across ALL test items where EVT-132 predicted NEW (0), but models say USED
# Sorted by ensemble mean probability
evt_new = (y_evt == 0)
candidates = []
for idx in np.where(evt_new)[0]:
    candidates.append({
        "index": idx,
        "ID": df_evt.loc[idx, "ID"],
        "title": test_titles[idx],
        "price": test_prices[idx],
        "mean_p": mean_p[idx],
        "median_p": median_p[idx],
        "min_p": min_p[idx],
        "std_p": std_p[idx],
        "votes": int(np.sum(mat[idx] > 0.5)),
        "in_slot1": idx in diff_indices
    })

df_cands = pd.DataFrame(candidates)
df_cands_sorted = df_cands.sort_values(by=["mean_p", "votes"], ascending=[False, False])

print("\n=== TOP 30 STRONGEST 'USED' ACCORDING TO 16-MODEL ENSEMBLE (EVT=NEW) ===")
for i, (_, r) in enumerate(df_cands_sorted.head(30).iterrows(), 1):
    in_s1 = " [IN SLOT 1]" if r['in_slot1'] else ""
    print(f"#{i:2d} | Idx {int(r['index']):5d} | ID {int(r['ID']):6d} | Mean: {r['mean_p']:.4f} | Min: {r['min_p']:.4f} | Votes: {int(r['votes'])}/16 | ${r['price']:7.1f} | {r['title'][:45]}{in_s1}")

df_cands_sorted.to_csv("results/top_candidates_16_model_ensemble.csv", index=False)
print("\nSaved all ranked candidates to results/top_candidates_16_model_ensemble.csv")

