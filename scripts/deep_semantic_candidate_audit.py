"""Deep Semantic Candidate Audit and Ultra-Pure Selection."""
import json
import re
from pathlib import Path
import numpy as np
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
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr[:, 1]
        elif arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        loaded[m] = arr

# Add the 5 newly trained full-data models from task-7437
npz_path = results_dir / "intensive_training_20261109" / "intensive_specialist_test_predictions.npz"
if npz_path.exists():
    npz_data = np.load(npz_path)
    loaded["new_ridge_70k"] = npz_data["p_ridge_test"]
    loaded["new_logreg_70k"] = npz_data["p_lr_test"]
    loaded["new_cnb_70k"] = npz_data["p_cnb_test"]
    loaded["new_lgb_1000"] = npz_data["p_lgb_test"]
    loaded["new_cat_1000"] = npz_data["p_cat_test"]

total_models = len(loaded)
print(f"Total Combined Diverse Models Loaded: {total_models}")

mat = np.column_stack([loaded[m] for m in loaded])
mean_p = np.mean(mat, axis=1)
median_p = np.median(mat, axis=1)
min_p = np.min(mat, axis=1)
std_p = np.std(mat, axis=1)

# Load test data
with open("data/ap_2026_q_2/raw/test_data.jsonlines", "r", encoding="utf-8") as f:
    test_items = [json.loads(line) for line in f]

df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
y_evt = (df_evt["condition"] == "used").astype(int).values

# Identify toxic patterns
TOXIC_PATTERNS = [
    r"\b0\s*km\b",
    r"\bokm\b",
    r"\bsin uso\b",
    r"\ba estrenar\b",
    r"\bnuevo sellado\b",
    r"\bcaja cerrada\b",
    r"\bcon garantia\b",
    r"\bcon etiqueta\b",
    r"\bcon etiquetas\b",
    r"\bmaracuya\b",
    r"\bcongelad[oa]\b",
    r"\brecinto industrial\b",
    r"\bterreno\b",
    r"\blote de \d+[\.,]\d+m2\b",
]
toxic_regex = re.compile("|".join(TOXIC_PATTERNS), re.IGNORECASE)

records = []
for idx, item in enumerate(test_items):
    if y_evt[idx] == 1:
        continue  # EVT-132 already predicted used
    
    title = item.get("title", "")
    price = item.get("price", 0)
    cat = item.get("category_id", "")
    attrs = item.get("attributes", [])
    
    # Check toxic patterns
    if toxic_regex.search(title):
        continue
    
    # Check car / real estate by price
    if price > 80000:
        continue  # skip high priced vehicles / real estate / high-end hardware
    
    # Compute metrics
    votes = int(np.sum(mat[idx] > 0.5))
    p_m = float(mean_p[idx])
    p_min = float(min_p[idx])
    p_std = float(std_p[idx])
    
    # Positive signals for used
    is_book_or_cd = any(w in title.lower() for w in ["libro", "tomo", "editorial", "ed.", "autor", "novela", "cd", "vinilo", "lp", "cassette"])
    is_vintage = any(w in title.lower() for w in ["antiguo", "antigua", "vintage", "retro", "coleccion", "ano 19", "año 19", "lote de monedas", "de epoca"])
    has_condition_cue = any(w in title.lower() for w in ["impecable", "excelente estado", "muy buen estado", "como nuevo", "usado", "buen estado", "detalle"])
    
    records.append({
        "index": idx,
        "ID": item.get("id"),
        "title": title,
        "price": price,
        "category_id": cat,
        "mean_p": p_m,
        "median_p": float(median_p[idx]),
        "min_p": p_min,
        "std_p": p_std,
        "votes": votes,
        "is_book_or_cd": is_book_or_cd,
        "is_vintage": is_vintage,
        "has_condition_cue": has_condition_cue,
        "score_pure": p_m * (votes / float(total_models)) - 0.5 * p_std + (0.05 if (is_book_or_cd or is_vintage or has_condition_cue) else 0.0)
    })

df_clean = pd.DataFrame(records)
df_clean = df_clean.sort_values(by="score_pure", ascending=False)
df_clean.to_csv("results/clean_pure_candidates_16models.csv", index=False)

print(f"Total Purified Candidates Evaluated: {len(df_clean)}")
print(f"\n=== TOP 30 ABSOLUTELY PURE CANDIDATES ACROSS {total_models} SPECIALIST MODELS ===")
for i, (_, r) in enumerate(df_clean.head(30).iterrows(), 1):
    print(f"#{i:2d} | Idx {int(r['index']):5d} | ID {r['ID']:14s} | Mean_P: {r['mean_p']:.3f} | Min_P: {r['min_p']:.3f} | Votes: {int(r['votes']):2d}/{total_models} | ${r['price']:6.1f} | {r['title'][:55]}")

