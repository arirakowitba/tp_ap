#!/usr/bin/env python3
"""Forensic Deconvolution across All Historical Submissions and Pattern Extraction."""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

ROOT = Path(".")
OUT_DIR = ROOT / "results" / "deconvolution_20261109"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Anchor EVT-132
ANCHOR_PATH = ROOT / "results" / "canonical_ledger_rebuild_20261108" / "evaluated_vectors_git_pinned" / "EVT-132_23ff472_11aabc76.csv"
assert ANCHOR_PATH.exists(), "Anchor missing"
df_anchor = pd.read_csv(ANCHOR_PATH)
anchor_preds = df_anchor["condition"].to_numpy()
ANCHOR_HITS = 19896

print(f"Loaded Anchor EVT-132: {len(anchor_preds)} rows, {ANCHOR_HITS} public hits.")

# 2. Load Pinned Vectors (EVT-129 to EVT-138)
CATALOG_PATH = ROOT / "results" / "canonical_ledger_rebuild_20261108" / "canonical_events_catalog.csv"
df_cat = pd.read_csv(CATALOG_PATH)
events_map = {}
for _, r in df_cat.iterrows():
    if pd.notnull(r.get("public_hits")):
        events_map[r["event_id"]] = {
            "name": r.get("display_name"),
            "score": float(r["public_score"]),
            "hits": int(r["public_hits"]),
            "delta_hits": int(r["public_hits"]) - ANCHOR_HITS,
        }

pinned_files = sorted(list((ROOT / "results" / "canonical_ledger_rebuild_20261108" / "evaluated_vectors_git_pinned").glob("EVT-*.csv")))
print(f"Found {len(pinned_files)} pinned vector files.")

vectors = []
for p in pinned_files:
    evt_id = p.name.split("_")[0]
    if evt_id in events_map:
        df_v = pd.read_csv(p)
        cond_v = df_v["condition"].to_numpy()
        vectors.append({
            "event_id": evt_id,
            "name": events_map[evt_id]["name"],
            "hits": events_map[evt_id]["hits"],
            "delta_hits": events_map[evt_id]["delta_hits"],
            "preds": cond_v,
        })
        print(f"  {evt_id} ({events_map[evt_id]['name'][:40]}): Hits={events_map[evt_id]['hits']}, Delta={events_map[evt_id]['delta_hits']}")

# 3. Build Difference Matrix vs Anchor
# Find all test indices where at least one vector differs from anchor
all_diff_indices = set()
for v in vectors:
    diff = np.where(v["preds"] != anchor_preds)[0]
    all_diff_indices.update(diff)

intervened_indices = sorted(list(all_diff_indices))
n_intervened = len(intervened_indices)
idx_to_col = {idx: col for col, idx in enumerate(intervened_indices)}
print(f"\nTotal distinct test items intervened across pinned vectors: {n_intervened}")

# Design matrix Z: (n_vectors x n_intervened)
# Z[s, c] = +1 if flipped new -> used, -1 if flipped used -> new, 0 if unchanged
Z = np.zeros((len(vectors), n_intervened), dtype=np.float64)
y_delta = np.zeros(len(vectors), dtype=np.float64)

for row, v in enumerate(vectors):
    y_delta[row] = v["delta_hits"]
    diff = np.where(v["preds"] != anchor_preds)[0]
    for d in diff:
        col = idx_to_col[d]
        if anchor_preds[d] == "new" and v["preds"][d] == "used":
            Z[row, col] = 1.0
        elif anchor_preds[d] == "used" and v["preds"][d] == "new":
            Z[row, col] = -1.0

# 4. Solve for Item Posterior Effects theta
# theta_i = E[delta_hits if item i is flipped from new -> used]
# For item i:
# If true used: theta_i = +0.70
# If true new:  theta_i = -0.70
print(f"Design Matrix Shape: {Z.shape}")

# L2 regularized dual solve: theta = Z.T @ (Z @ Z.T + alpha * I)^(-1) @ y_delta
alpha_reg = 0.5
K_mat = Z @ Z.T
dual = np.linalg.solve(K_mat + alpha_reg * np.eye(len(vectors)), y_delta)
theta_est = Z.T @ dual

# Also solve bounded least squares: bounds [-0.75, +0.75]
res_lsq = lsq_linear(Z, y_delta, bounds=(-0.75, 0.75), lsmr_tol='auto', verbose=0)
theta_bounded = res_lsq.x

appearances = (Z != 0).sum(axis=0)

# 5. Load Test Metadata to Inspect Intervened Items
test_metadata = []
with open("data/ap_2026_q_2/raw/test_data.jsonlines", "r", encoding="utf-8") as f:
    for line in f:
        test_metadata.append(json.loads(line))

results_list = []
for c, test_idx in enumerate(intervened_indices):
    item_id = test_idx + 1
    t_dual = float(theta_est[c])
    t_bnd = float(theta_bounded[c])
    app = int(appearances[c])
    row = test_metadata[test_idx]
    
    # Classify confidence
    if t_bnd > 0.30:
        inferred = "HIGH_CONFIDENCE_TRUE_USED"
    elif t_bnd < -0.30:
        inferred = "HIGH_CONFIDENCE_TRUE_NEW"
    else:
        inferred = "UNCERTAIN"
        
    results_list.append({
        "item_id": item_id,
        "test_idx": test_idx,
        "anchor_pred": str(anchor_preds[test_idx]),
        "appearances": app,
        "theta_dual": round(t_dual, 4),
        "theta_bounded": round(t_bnd, 4),
        "inferred_label": inferred,
        "title": str(row.get("title") or ""),
        "price": float(row.get("price") or 0.0),
        "warranty": str(row.get("warranty") or ""),
        "tags": row.get("tags") or [],
        "category_id": str(row.get("category_id") or ""),
        "seller_id": str(row.get("seller_id") or ""),
    })

df_deconv = pd.DataFrame(results_list)
df_deconv = df_deconv.sort_values(by="theta_bounded", ascending=False)
df_deconv.to_csv(OUT_DIR / "deconvolved_intervened_items.csv", index=False)
print(f"Saved deconvolution results to {OUT_DIR / 'deconvolved_intervened_items.csv'}")

# Print summary
print("\n=== DECONVOLUTION SUMMARY ===")
print(df_deconv["inferred_label"].value_counts())

print("\n--- TOP RESOLVED TRUE USED ITEMS (theta > +0.30) ---")
for _, r in df_deconv[df_deconv["theta_bounded"] > 0.30].head(15).iterrows():
    print(f"  ID {r['item_id']:<5} | theta={r['theta_bounded']:+.4f} (apps={r['appearances']}) | \"{r['title'][:55]}\"")

print("\n--- TOP RESOLVED TRUE NEW ITEMS (theta < -0.30) ---")
for _, r in df_deconv[df_deconv["theta_bounded"] < -0.30].head(15).iterrows():
    print(f"  ID {r['item_id']:<5} | theta={r['theta_bounded']:+.4f} (apps={r['appearances']}) | \"{r['title'][:55]}\"")
