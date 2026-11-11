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

print("--- Searching for N2U (EVT=new, clearly used) ---")
n2u_phrases = [
    "para repuesto", "para repuestos", "para desarme", "para despiece",
    "bloqueado icloud", "bloqueado para", "no funciona", "a reparar",
    "con detalles de uso", "funciona perfectamente usado", "segunda mano",
    "en buen estado usado", "muy poco uso", "excelente estado usado"
]

for p in n2u_phrases:
    for i, r in enumerate(test_raw):
        if i in diff_s1:
            continue
        t = (r.get("title") or "").lower()
        if p in t and y_evt[i] == "new":
            print(f"ID {i+1}: Phrase '{p}' | Price: ${r.get('price')} | Title: {r.get('title')}")

print("\n--- Searching for U2N (EVT=used, clearly new) ---")
u2n_phrases = [
    "original sellado", "sellado sin abrir", "nuevo en caja sellada",
    "sellado de fabrica", "en blister cerrado", "blister cerrado original",
    "0km nuevo", "a estrenar con etiqueta", "nuevo con etiqueta original"
]

for p in u2n_phrases:
    for i, r in enumerate(test_raw):
        if i in diff_s1:
            continue
        t = (r.get("title") or "").lower()
        if p in t and y_evt[i] == "used":
            # skip antiques
            if not any(w in t for w in ["antiguo", "antigua", "vintage", "coleccion", "icono"]):
                print(f"ID {i+1}: Phrase '{p}' | Price: ${r.get('price')} | Title: {r.get('title')}")
