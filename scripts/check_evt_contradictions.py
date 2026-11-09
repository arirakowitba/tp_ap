"""Inspect unambiguous semantic contradictions in EVT-132."""

import pandas as pd
import numpy as np
import re
from pathlib import Path

DATA_DIR = Path("data/ap_2026_q_2/raw")
df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)

evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
df_test["evt_condition"] = evt["condition"]

# 1. Obvious USED cues where EVT-132 says 'new'
used_patterns = [
    r"\bcomo nuevo\b",
    r"\bimpecable\b",
    r"\bmuy buen estado\b",
    r"\bexcelente estado\b",
    r"\bbuen estado\b",
    r"\bdetalles de uso\b",
    r"\bcon detalles\b",
    r"\bpara repuesto\b",
    r"\bpara reparar\b",
    r"\busado\b",
    r"\busada\b",
    r"\bsegunda mano\b",
    r"\blote de libros usados\b",
    r"\bvintage\b",
    r"\bantiguo\b",
    r"\bantigua\b",
]

df_test["title_clean"] = df_test["title"].fillna("").astype(str).str.lower()

used_regex = re.compile("|".join(used_patterns))
df_test["has_used_cue"] = df_test["title_clean"].apply(lambda t: bool(used_regex.search(t)))

contradictions_used = df_test[(df_test["evt_condition"] == "new") & df_test["has_used_cue"]]
print(f"Total test items with obvious USED cue where EVT-132 said NEW: {len(contradictions_used)}")

# 2. Obvious NEW cues where EVT-132 says 'used'
new_patterns = [
    r"\bnuevo sellado\b",
    r"\bcaja cerrada\b",
    r"\bblister cerrado\b",
    r"\b100% nuevo\b",
    r"\bde fabrica con garantia\b",
    r"\bgarantia oficial\b",
    r"\bgarantia de fabrica\b",
]
new_regex = re.compile("|".join(new_patterns))
df_test["has_new_cue"] = df_test["title_clean"].apply(lambda t: bool(new_regex.search(t)))

contradictions_new = df_test[(df_test["evt_condition"] == "used") & df_test["has_new_cue"]]
print(f"Total test items with obvious NEW cue where EVT-132 said USED: {len(contradictions_new)}")

print("\nSample contradictions (EVT-132 says NEW, title says USED):")
for _, r in contradictions_used.head(10).iterrows():
    print(f"  ID {r['id']}: {r['title']}")

print("\nSample contradictions (EVT-132 says USED, title says NEW):")
for _, r in contradictions_new.head(10).iterrows():
    print(f"  ID {r['id']}: {r['title']}")
