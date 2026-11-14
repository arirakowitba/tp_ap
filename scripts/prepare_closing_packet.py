"""Prepare and audit the Closing Execution Packet for AP-2026-Q-2.
Generates candidate summary tables, decision branches, hashes, and timing bounds.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

N_PUB = 20998
N_TEST = 30000
ANCHOR_SCORE = 0.94766
ANCHOR_HITS = 19899

s4_path = Path("submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv")
s4_df = pd.read_csv(s4_path).sort_values("ID")
s4_used = s4_df["condition"].eq("used").to_numpy()
s4_sha256 = hashlib.sha256(s4_path.read_bytes()).hexdigest()

def get_target_hash(df):
    lines = [f"{i}:{c}\n" for i, c in zip(df["ID"], df["condition"])]
    return hashlib.sha256("".join(lines).encode()).hexdigest()

s4_map_hash = get_target_hash(s4_df)

# Candidate files
candidates = {
    "Slot1_Primary_C5": {
        "name": "candidate_via_principal_pruned_historical_bounds.csv",
        "path": Path("results/candidate_via_principal_pruned_historical_bounds.csv"),
        "role": "Primer Cupo / Apuesta Principal para superar 0.95000",
        "model": "Multi-Family Routed Ensemble (LGBM + CB + XGB + Especialista) podado contra historial",
        "anchor": "S4 (0.94766)",
        "expected_gain": 53.25,
        "ci90": [29.0, 77.0],
        "p_gt_09500": 0.6028,
        "p_gt_095095": 0.1503
    },
    "Slot_Contrast_C6": {
        "name": "candidate_via_principal_high_margin_contrast.csv",
        "path": Path("results/candidate_via_principal_high_margin_contrast.csv"),
        "role": "Contraste de Riesgo Mínimo / Alta Certeza",
        "model": "Vía Principal sobre soporte limpio con margen |p - 0.5| >= 0.20",
        "anchor": "S4 (0.94766)",
        "expected_gain": 22.68,
        "ci90": [13.0, 32.0],
        "p_gt_09500": 0.0,
        "p_gt_095095": 0.0
    },
    "Fallback_S4": {
        "name": "SUBMISSION_4_TODAY_RECOMMENDED.csv",
        "path": s4_path,
        "role": "Fallback Seguro / Línea Base Confirmada",
        "model": "Ensamble Histórico S4",
        "anchor": "S4 (Auto)",
        "expected_gain": 0.0,
        "ci90": [0.0, 0.0],
        "p_gt_09500": 0.0,
        "p_gt_095095": 0.0
    }
}

# Verify and extract file metrics
packet = {}
for cid, info in candidates.items():
    p = info["path"]
    df = pd.read_csv(p).sort_values("ID")
    assert len(df) == 30000
    assert df["ID"].tolist() == list(range(1, 30001))
    assert df["condition"].isin(["new", "used"]).all()
    
    cand_used = df["condition"].eq("used").to_numpy()
    flips_mask = (cand_used != s4_used)
    flips_total = int(flips_mask.sum())
    u2n = int((s4_used & ~cand_used).sum())
    n2u = int((~s4_used & cand_used).sum())
    
    file_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    map_hash = get_target_hash(df)
    
    packet[cid] = {
        "id": cid,
        "filename": info["name"],
        "filepath": str(p),
        "file_sha256": file_sha,
        "target_map_hash": map_hash,
        "role": info["role"],
        "model": info["model"],
        "anchor": info["anchor"],
        "flips_total": flips_total,
        "flips_u2n": u2n,
        "flips_n2u": n2u,
        "expected_net_gain": info["expected_gain"],
        "ci90": info["ci90"],
        "p_gt_09500": info["p_gt_09500"],
        "p_gt_095095": info["p_gt_095095"]
    }

# Save closing execution packet
with open("results/closing_execution_packet.json", "w") as f:
    json.dump(packet, f, indent=2)

print("Closing execution packet successfully created and verified.")
for cid, d in packet.items():
    print(f"[{cid}] {d['filename']}: Flips={d['flips_total']} (U2N={d['flips_u2n']}, N2U={d['flips_n2u']}), SHA={d['file_sha256'][:16]}...")
