#!/usr/bin/env python3
"""Analyze pattern signals discovered from deconvolution on 70k Train Dataset."""

import json
import re
import zipfile
import numpy as np
import pandas as pd

# Load train dataset
train_rows = []
with zipfile.ZipFile("data/train_data.zip") as z:
    for line in z.open("train_data.jsonlines"):
        train_rows.append(json.loads(line))

y_true = np.array([int(r["condition"] == "used") for r in train_rows], dtype=np.int32)
print(f"Loaded {len(train_rows)} train rows. Global Used rate: {y_true.mean():.4f}")

# Define pattern regexes to test
PATTERNS = {
    # Patterns that indicate NEW despite low seller volume or no warranty
    "cue_new_a_estrenar": r"\b(a estrenar)\b",
    "cue_new_0km": r"\b(0km|0 km)\b",
    "cue_new_sin_uso": r"\b(sin uso)\b",
    "cue_new_sellado": r"\b(sellad[oa]s?|en blister|cerrad[oa] en caja|caja cerrada)\b",
    "cue_new_impecable_nuevo": r"\b(impecable sin uso|nuevo sellado|nuevo sin uso)\b",
    "cue_new_amiibo": r"\b(amiibo)\b",
    "cue_new_dummy": r"\b(dummy|ficticio)\b",
    "cue_new_artesanal": r"\b(artesanal|hech[oa] a mano|souvenirs?|cotill[oó]n|centro de mesa)\b",
    "cue_new_disfraz": r"\b(disfraz|disfraces)\b",
    
    # Patterns that indicate USED despite warranty or positive text
    "cue_used_cubiertas": r"\b(cubiertas?|neumaticos?)\b",
    "cue_used_antiguo": r"\b(antigu[oa]s?|antiguedades|vintage|retro)\b",
    "cue_used_lote_revistas": r"\b(lote de|colecci[oó]n de|revistas?|fasc[ií]culos?)\b",
    "cue_used_desarme": r"\b(desarme|despiece|repuestos?|para repuesto)\b",
    "cue_used_detalles": r"\b(detalles? de uso|leves detalles|buen estado|muy buen estado|impecable estado)\b",
    "cue_used_segunda_mano": r"\b(segunda mano|usad[oa]s?)\b",
}

print(f"\n{'Pattern Name':<25} | {'Count':<6} | {'Used %':<8} | {'New %':<8} | {'Dominant Class':<15}")
print("-" * 70)

for name, pat_str in PATTERNS.items():
    regex = re.compile(pat_str, re.IGNORECASE)
    mask = np.array([bool(regex.search(str(r.get("title") or "") + " " + str(r.get("warranty") or ""))) for r in train_rows])
    count = int(mask.sum())
    if count > 0:
        used_p = float(y_true[mask].mean())
        new_p = 1.0 - used_p
        dom = "USED (CONFIRMED)" if used_p >= 0.70 else ("NEW (CONFIRMED)" if new_p >= 0.70 else "MIXED")
        print(f"{name:<25} | {count:<6d} | {used_p*100:<7.1f}% | {new_p*100:<7.1f}% | {dom:<15}")
