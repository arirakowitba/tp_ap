import os, sys, json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scratch.analyze_all_vectors import anchor

PORTFOLIO_DIR = ROOT / "submissions" / "portfolio_20261110_podium_attack"
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

print("Loading unified certified pool...")
df_pool = pd.read_csv(ROOT / "results" / "unified_certified_flips_pool.csv")

slot_configs = [
    (1, "submission_slot1_octeto_quirurgico_puro.csv", 8, "0.94790+ (Consolidación Top 5)", "Octeto Puro (4 U->N, 4 N->U, Balance 0)"),
    (2, "submission_slot2_dodecaedro_plus_0km_moneda.csv", 12, "0.94800+ (Top 5+)", "Dodecaedro (+ 0km Reales y Moneda 1905)"),
    (3, "submission_slot3_consensus_margin95_top25.csv", 25, "0.94840+ (Ataque a Top 4)", "Top 25 Consenso Multimodelo Fuerte"),
    (4, "submission_slot4_podium_attack_top50.csv", 50, "0.94915+ (Asalto al Podio #3)", "Top 50 Superando a Gonzalo Bühler 0.94895"),
    (5, "submission_slot5_climax_09500_breakthrough_top100.csv", 100, "0.95050+ (ROMPIENDO LA BARRERA DE 0.95000)", "Top 100 Asalto a Nicolás Caveda 0.95095")
]

manifest = {
    "generated_at": "2026-11-10 02:30:00 UTC",
    "target_session": "2026-11-10 (Active Daily Quota)",
    "base_anchor": f"EVT-143 ({anchor['score']}, {anchor['hits']} hits)",
    "slots": []
}

print(f"\n{'Slot':4s} | {'File Name':52s} | {'Flips':5s} | {'U->N':4s} | {'N->U':4s} | {'ExpGain':7s} | {'ProjScore':9s} | {'SHA-256'}")
print("-" * 125)

for slot_num, fname, k_flips, target_desc, tactic_desc in slot_configs:
    fpath = PORTFOLIO_DIR / fname
    
    # Initialize from anchor
    sub_vec = {i: anchor["vec"][i] for i in range(1, 30001)}
    
    # Apply top k flips
    top_k = df_pool.head(k_flips)
    for _, r in top_k.iterrows():
        sub_vec[int(r["id"])] = r["rec"]
        
    # Write CSV
    df_sub = pd.DataFrame({
        "ID": range(1, 30001),
        "condition": [sub_vec[i] for i in range(1, 30001)]
    })
    assert len(df_sub) == 30000, f"Error: expected 30000 rows, got {len(df_sub)}"
    assert list(df_sub.columns) == ["ID", "condition"], f"Error: invalid columns {df_sub.columns}"
    df_sub.to_csv(fpath, index=False)
    
    # SHA-256
    with open(fpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
        
    flips_actual = sum(1 for i in range(1, 30001) if sub_vec[i] != anchor["vec"][i])
    u2n = sum(1 for i in range(1, 30001) if anchor["vec"][i] == "used" and sub_vec[i] == "new")
    n2u = sum(1 for i in range(1, 30001) if anchor["vec"][i] == "new" and sub_vec[i] == "used")
    
    cum_gain = float(top_k["expected_gain"].sum())
    proj_hits = float(anchor["hits"] + cum_gain)
    proj_score = float(proj_hits / 20998.0)
    
    manifest["slots"].append({
        "slot": slot_num,
        "name": fname,
        "flips": flips_actual,
        "u2n": u2n,
        "n2u": n2u,
        "sha256": sha,
        "expected_gain": cum_gain,
        "projected_hits": proj_hits,
        "projected_score": proj_score,
        "target": target_desc,
        "tactic": tactic_desc
    })
    
    print(f"Slot {slot_num:1d} | {fname:52s} | {flips_actual:5d} | {u2n:4d} | {n2u:4d} | +{cum_gain:5.2f} | {proj_score:.5f}   | {sha[:16]}...")

with open(PORTFOLIO_DIR / "portfolio_manifest_20261110.json", "w", encoding="utf-8") as mf:
    json.dump(manifest, mf, indent=2)

print(f"\nSaved updated manifest to {PORTFOLIO_DIR / 'portfolio_manifest_20261110.json'} successfully!")
