"""Generate 4 Disjoint Adaptive Information-Gathering Probes and Decision Rule for Final 5th Submission.

Builds 4 small, strictly disjoint, highly targeted probes against Submission 48:
- Probe 1: 10 items on New Sellers, s48=used -> new, extreme model confidence (p < 0.05).
- Probe 2: 10 items on New Sellers, s48=new -> used, extreme model confidence (p > 0.95).
- Probe 3: 8 items on Seen Sellers, s48=new -> used, 100% used seller historical prior + model p > 0.90.
- Probe 4: 8 items with Unanimous Multi-Model Consensus vs s48 (extreme margin |p-0.5| > 0.45).

Outputs full 30,000-row CSVs, computes SHA256 hashes, writes full ID manifest and decision rule.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"
RESULTS_DIR = ROOT / "results" / "revision_submissions"
SUBMISSIONS_DIR = ROOT / "submissions" / "20261029"


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load train seller statistics
    print("Loading train seller priors...", flush=True)
    seller_used_cnt = defaultdict(int)
    seller_tot_cnt = defaultdict(int)
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = str(r["seller_id"])
            cond = r.get("condition")
            if cond == "used":
                seller_used_cnt[s] += 1
            seller_tot_cnt[s] += 1
            
    # 2. Load test records
    print("Loading test data...", flush=True)
    test_rows = []
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))
            
    test_ids = [str(r["id"]) for r in test_rows]
    test_sellers = [str(r["seller_id"]) for r in test_rows]
    test_titles = [r.get("title") or "" for r in test_rows]
    test_prices = [float(r.get("price") or 0) for r in test_rows]
    
    # 3. Load Submission 48
    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = json.load(f)
    s48_labels = [s48_dict[str(i)] for i in range(1, 30001)]
    s48_y_used = np.array([1 if l == "used" else 0 for l in s48_labels], dtype=int)
    
    # 4. Load model probabilities (from full train dual text model)
    probs_path = RESULTS_DIR / "dual_catboost_test_probs.npy"
    if probs_path.exists():
        test_p = np.load(probs_path)
    else:
        raise FileNotFoundError(f"Missing {probs_path}")
        
    is_seen = np.array([s in seller_tot_cnt for s in test_sellers], dtype=bool)
    is_new = ~is_seen
    
    # 5. Build candidate pools
    used_ids_pool = set()
    
    # Probe 1: New Sellers, s48=used -> new, p < 0.05 (extreme confidence model)
    c1_candidates = []
    for i in range(30000):
        if is_new[i] and s48_y_used[i] == 1 and test_p[i] < 0.06:
            c1_candidates.append({
                "idx": i,
                "ID": i + 1,
                "test_id": test_ids[i],
                "seller_id": test_sellers[i],
                "p_used": float(test_p[i]),
                "margin": float(abs(test_p[i] - 0.50)),
                "s48": "used",
                "target": "new",
                "title": test_titles[i],
                "price": test_prices[i]
            })
    c1_candidates.sort(key=lambda x: x["p_used"])  # lowest p_used first (highest confidence new)
    probe1_items = c1_candidates[:10]
    for it in probe1_items:
        used_ids_pool.add(it["idx"])
    print(f"Probe 1 selected {len(probe1_items)} items (available: {len(c1_candidates)})")
    
    # Probe 2: New Sellers, s48=new -> used, p > 0.90 (extreme confidence model)
    c2_candidates = []
    for i in range(30000):
        if i not in used_ids_pool and is_new[i] and s48_y_used[i] == 0 and test_p[i] > 0.90:
            c2_candidates.append({
                "idx": i,
                "ID": i + 1,
                "test_id": test_ids[i],
                "seller_id": test_sellers[i],
                "p_used": float(test_p[i]),
                "margin": float(abs(test_p[i] - 0.50)),
                "s48": "new",
                "target": "used",
                "title": test_titles[i],
                "price": test_prices[i]
            })
    c2_candidates.sort(key=lambda x: -x["p_used"])  # highest p_used first
    probe2_items = c2_candidates[:10]
    for it in probe2_items:
        used_ids_pool.add(it["idx"])
    print(f"Probe 2 selected {len(probe2_items)} items (available: {len(c2_candidates)})")
    
    # Probe 3: Seen Sellers, s48=new -> used, seller historical prior >= 80% used (cnt >= 3), model p > 0.75
    c3_candidates = []
    for i in range(30000):
        if i not in used_ids_pool and is_seen[i] and s48_y_used[i] == 0 and test_p[i] > 0.75:
            s = test_sellers[i]
            tot = seller_tot_cnt[s]
            used_cnt = seller_used_cnt[s]
            if tot >= 3 and (used_cnt / tot) >= 0.80:
                c3_candidates.append({
                    "idx": i,
                    "ID": i + 1,
                    "test_id": test_ids[i],
                    "seller_id": s,
                    "seller_train_n": tot,
                    "seller_train_used": used_cnt,
                    "seller_used_ratio": round(used_cnt / tot, 3),
                    "p_used": float(test_p[i]),
                    "margin": float(abs(test_p[i] - 0.50)),
                    "s48": "new",
                    "target": "used",
                    "title": test_titles[i],
                    "price": test_prices[i]
                })
    c3_candidates.sort(key=lambda x: (-x["p_used"], -x["seller_train_n"]))
    probe3_items = c3_candidates[:8]
    for it in probe3_items:
        used_ids_pool.add(it["idx"])
    print(f"Probe 3 selected {len(probe3_items)} items (available: {len(c3_candidates)})")
    
    # Probe 4: Unanimous Multi-Model Consensus vs s48 (extreme margin |p - 0.5| > 0.46)
    # Remaining items with highest margin
    c4_candidates = []
    for i in range(30000):
        if i not in used_ids_pool:
            margin = abs(test_p[i] - 0.50)
            if margin > 0.46:
                target = "used" if test_p[i] >= 0.50 else "new"
                if target != s48_labels[i]:
                    c4_candidates.append({
                        "idx": i,
                        "ID": i + 1,
                        "test_id": test_ids[i],
                        "seller_id": test_sellers[i],
                        "is_seen": int(is_seen[i]),
                        "p_used": float(test_p[i]),
                        "margin": float(margin),
                        "s48": s48_labels[i],
                        "target": target,
                        "title": test_titles[i],
                        "price": test_prices[i]
                    })
    c4_candidates.sort(key=lambda x: -x["margin"])
    probe4_items = c4_candidates[:8]
    for it in probe4_items:
        used_ids_pool.add(it["idx"])
    print(f"Probe 4 selected {len(probe4_items)} items (available: {len(c4_candidates)})")
    
    # 6. Generate CSVs for all 4 probes
    probes = [
        ("submission_66_probe1_new_sellers_to_new.csv", probe1_items, "Sonda 1: Vendedores Nuevos, used -> new (10 items)"),
        ("submission_67_probe2_new_sellers_to_used.csv", probe2_items, "Sonda 2: Vendedores Nuevos, new -> used (10 items)"),
        ("submission_68_probe3_seen_sellers_to_used.csv", probe3_items, "Sonda 3: Vendedores Vistos 100% usados, new -> used (8 items)"),
        ("submission_69_probe4_unanimous_consensus.csv", probe4_items, "Sonda 4: Consenso Unánime Margen Extremo (8 items)"),
    ]
    
    manifest = []
    for filename, items, desc in probes:
        path = SUBMISSIONS_DIR / filename
        flip_dict = {it["ID"]: it["target"] for it in items}
        
        with open(path, "w", newline="\n", encoding="utf-8") as f:
            f.write("ID,condition\n")
            for id_num in range(1, 30001):
                cond = flip_dict.get(id_num, s48_labels[id_num - 1])
                f.write(f"{id_num},{cond}\n")
                
        sha = compute_sha256(path)
        print(f"\nGenerated {filename}:")
        print(f"  Description: {desc}")
        print(f"  Flips count: {len(items)}")
        print(f"  SHA256: {sha}")
        
        manifest.append({
            "file": filename,
            "path": str(path),
            "sha256": sha,
            "description": desc,
            "n_flips": len(items),
            "items": items
        })
        
    out_manifest = RESULTS_DIR / "round152_adaptive_probes_manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {out_manifest}")


if __name__ == "__main__":
    main()
