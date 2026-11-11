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

print(f"Total rows: {len(test_raw)}, S1 diffs: {len(diff_s1)}")

# N2U: currently predicted NEW in EVT-143, but title clearly indicates USED
repuestos_words = [
    "para repuesto", "para repuestos", "para desarme", "para piezas", 
    "no funciona", "a reparar", "desarme", "pantalla rota", "bloqueado para repuesto"
]

n2u_found = []
for w in repuestos_words:
    matches = [i for i, r in enumerate(test_raw) if w in (r.get("title") or "").lower() and y_evt[i] == "new" and i not in diff_s1]
    print(f"Keyword '{w}' in EVT=new: {len(matches)} matches")
    for idx in matches:
        title = test_raw[idx].get("title", "")
        price = float(test_raw[idx].get("price") or 0.0)
        n2u_found.append((idx, w, price, title))

# Deduplicate
n2u_dict = {x[0]: x for x in n2u_found}
print(f"Total unique N2U candidates: {len(n2u_dict)}")
for idx, (i, w, p, t) in enumerate(list(n2u_dict.values())[:10]):
    print(f"  N2U {idx+1}: ID {i+1} | ${p:.1f} | '{w}' | {t}")

# U2N: currently predicted USED in EVT-143, but title clearly indicates NEW
sellado_words = [
    "sellado", "en caja sellada", "cerrado de fabrica", "nuevo cerrado", 
    "blister cerrado", "sin uso en caja", "0km", "sellada de fabrica",
    "nuevo en blister", "nuevo de fabrica", "cerrada de fabrica"
]

u2n_found = []
for w in sellado_words:
    matches = [i for i, r in enumerate(test_raw) if w in (r.get("title") or "").lower() and y_evt[i] == "used" and i not in diff_s1]
    print(f"Keyword '{w}' in EVT=used: {len(matches)} matches")
    for idx in matches:
        title = test_raw[idx].get("title", "")
        price = float(test_raw[idx].get("price") or 0.0)
        # Avoid vintage books/antiques that might say "nuevo" ironically
        u2n_found.append((idx, w, price, title))

u2n_dict = {x[0]: x for x in u2n_found}
print(f"Total unique U2N candidates: {len(u2n_dict)}")
for idx, (i, w, p, t) in enumerate(list(u2n_dict.values())[:10]):
    print(f"  U2N {idx+1}: ID {i+1} | ${p:.1f} | '{w}' | {t}")
