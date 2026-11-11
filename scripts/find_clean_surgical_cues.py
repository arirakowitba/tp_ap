import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")

with open(ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_raw = [json.loads(line) for line in f]

evt = pd.read_csv(ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv")
y_evt = evt["condition"].values

s1 = pd.read_csv(ROOT / "submissions/SUBMISSION_1_TODAY_RECOMMENDED.csv")
diff_s1 = set(np.where(s1["condition"] != y_evt)[0])

# Search for indisputable candidates in EVT-143
# 1. Non-functional dummies or parts currently predicted 'new' in EVT-143
print("=== Candidates for NEW -> USED ===")
for i, r in enumerate(test_raw):
    if i in diff_s1:
        continue
    t = (r.get("title") or "").lower()
    p = float(r.get("price") or 0.0)
    c = y_evt[i]
    if c == "new":
        if "no funcional" in t or "para desarme" in t or "a reparar" in t or "desarme" in t or "roto" in t:
            print(f"ID {i+1}: ${p:.1f} | {r.get('title')}")

# 2. Genuinely brand new sealed video games or CDs or electronics currently predicted 'used' in EVT-143
print("\n=== Candidates for USED -> NEW ===")
for i, r in enumerate(test_raw):
    if i in diff_s1:
        continue
    t = (r.get("title") or "").lower()
    p = float(r.get("price") or 0.0)
    c = y_evt[i]
    if c == "used":
        if "original sellado" in t or "sellado sin abrir" in t or "nuevo cerrado" in t or "en caja sellada" in t or "blister cerrado" in t or "sellado de fabrica" in t:
            # Filter out antiques, stamps, vintage
            if not any(w in t for w in ["antiguo", "antigua", "vintage", "icono", "vajilla", "plata", "de época"]):
                print(f"ID {i+1}: ${p:.1f} | {r.get('title')}")
