import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

print("Testing Domain-Gated Residual Corrections on Full 70,000 Podium OOF...")

# 1. Load Podium OOF
podium_oof = pd.read_csv("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")
y_true = podium_oof["y_true"].values
p_base = podium_oof["oof_blend"].values

acc_base = accuracy_score(y_true, (p_base >= 0.5).astype(int))
hits_base = int(round(acc_base * 70000))
ll_base = log_loss(y_true, p_base)
print(f"Base Podium Engine: Acc = {acc_base:.5f} ({hits_base} / 70.000 hits), LL = {ll_base:.5f}")

# 2. Load Raw Metadata
raw_path = "data/ap_2026_q_2/raw/train_data.jsonlines"
titles = []
sellers = []
categories = []
with open(raw_path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        titles.append(str(r.get("title") or ""))
        sellers.append(str(r.get("seller_id") or "MISSING"))
        categories.append(str(r.get("category_id") or "MISSING"))

from collections import Counter
sc = Counter(sellers)
is_singleton = np.array([sc[s] == 1 for s in sellers])

# Regexes
sin_uso_re = re.compile(r"\b(sin uso|a estrenar|nunca usad[oa]s?|en caja cerrada|sellad[oa]s?|0\s*km|cero\s*km|de fabrica|impecable sin uso)\b", re.IGNORECASE)
reparar_re = re.compile(r"\b(para reparar|para repuesto|despiece|no funciona|repuestos?|a revisar|para arreglo|dañad[oa])\b", re.IGNORECASE)

r_sin_uso = np.array([bool(sin_uso_re.search(t)) for t in titles])
r_reparar = np.array([bool(reparar_re.search(t)) for t in titles])

# Evaluate Rule 1: High certainty sin_uso on borderline used
# If r_sin_uso is True and p_base is borderline used [0.50, 0.75]
mask_rule1 = r_sin_uso & (p_base >= 0.50) & (p_base <= 0.72)
fixes_r1 = ((y_true == 0) & mask_rule1).sum()
breaks_r1 = ((y_true == 1) & mask_rule1).sum()
print(f"Rule 1 ('sin uso' on borderline used [0.50, 0.72]): {mask_rule1.sum()} items -> Fixes={fixes_r1}, Breaks={breaks_r1}, Net={fixes_r1 - breaks_r1:+d}")

# Evaluate Rule 2: High certainty para_reparar on borderline new
mask_rule2 = r_reparar & (p_base >= 0.28) & (p_base < 0.50)
fixes_r2 = ((y_true == 1) & mask_rule2).sum()
breaks_r2 = ((y_true == 0) & mask_rule2).sum()
print(f"Rule 2 ('para reparar' on borderline new [0.28, 0.50)): {mask_rule2.sum()} items -> Fixes={fixes_r2}, Breaks={breaks_r2}, Net={fixes_r2 - breaks_r2:+d}")

# Combine into a smooth probability nudge
p_challenger = p_base.copy()
# Nudge borderline sin_uso towards new
p_challenger[mask_rule1] -= 0.25
# Nudge borderline para_reparar towards used
p_challenger[mask_rule2] += 0.25

p_challenger = np.clip(p_challenger, 0.0001, 0.9999)

acc_chal = accuracy_score(y_true, (p_challenger >= 0.5).astype(int))
hits_chal = int(round(acc_chal * 70000))
ll_chal = log_loss(y_true, p_challenger)

pred_base = (p_base >= 0.5).astype(int)
pred_chal = (p_challenger >= 0.5).astype(int)

fixes = int(((pred_base != y_true) & (pred_chal == y_true)).sum())
breaks = int(((pred_base == y_true) & (pred_chal != y_true)).sum())

print(f"\nCHALLENGER RESULT ON 70,000 OOF:")
print(f"Podium Base:       Acc = {acc_base:.5f} ({hits_base} hits), LL = {ll_base:.5f}")
print(f"Challenger Gated:  Acc = {acc_chal:.5f} ({hits_chal} hits), LL = {ll_chal:.5f}")
print(f"NET GAIN:          {hits_chal - hits_base:+d} HITS (+{fixes} fixes, -{breaks} breaks)")
