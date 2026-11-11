import json
from pathlib import Path
import pandas as pd

ROOT = Path(".")

with open(ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_raw = [json.loads(line) for line in f]

evt = pd.read_csv(ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv")
y_evt = evt["condition"].values
s1 = pd.read_csv(ROOT / "submissions/SUBMISSION_1_TODAY_RECOMMENDED.csv")
diff_s1 = set(i for i in range(30000) if s1.loc[i, "condition"] != y_evt[i])

queries = ["para partes", "sin funcionar", "le falta", "falta cargador", "solo repuesto", "repuesto usado", "con faltantes"]
for q in queries:
    matches = [i for i, r in enumerate(test_raw) if q in (r.get("title") or "").lower() and y_evt[i] == "new" and i not in diff_s1]
    print(f"Query '{q}': {len(matches)} matches")
    for idx in matches:
        print(f"  ID {idx+1}: ${test_raw[idx].get('price')} | {test_raw[idx].get('title')}")
