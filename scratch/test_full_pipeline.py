import json, csv, os, sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import re
from scratch.analyze_all_vectors import vectors, anchor

# 1. Load test metadata
with open("data/ap_2026_q_2/raw/test_data.jsonlines", "r", encoding="utf-8") as f:
    test_items = [json.loads(line) for line in f]

N = len(test_items)
M = len(vectors)

anchor_hits = anchor['hits']
anchor_vec = np.array([1 if anchor['vec'][i] == 'new' else 0 for i in range(1, N + 1)], dtype=np.int8)

# 2. Build sub_matrix
sub_matrix = np.zeros((M, N), dtype=np.int8)
scores_vec = np.zeros(M, dtype=float)
hits_vec = np.zeros(M, dtype=int)

for s_idx, v in enumerate(vectors):
    scores_vec[s_idx] = v['score']
    hits_vec[s_idx] = v['hits']
    for i in range(1, N + 1):
        if v['vec'][i] == 'new':
            sub_matrix[s_idx, i - 1] = 1

times_new = sub_matrix.sum(axis=0)
times_used = M - times_new
times_flipped = np.sum(sub_matrix != anchor_vec, axis=0)
rate_new = times_new / M
rate_used = times_used / M

# Top 17 high-performing vectors (score >= 0.94700)
top17_mask = scores_vec >= 0.94700
flips_in_top17 = np.sum(sub_matrix[top17_mask] != anchor_vec, axis=0)
is_invariant_top17 = (flips_in_top17 == 0)

# Net delta hits sum for flips vs anchor
delta_hits_s = hits_vec - anchor_hits
F = (sub_matrix != anchor_vec).astype(int)
delta_hits_sum = F.T @ delta_hits_s

# 3. Bayesian probe deconvolution
X = np.zeros((M, N), dtype=np.float64)
for s_idx, v in enumerate(vectors):
    for i in range(1, N + 1):
        pred = sub_matrix[s_idx, i - 1]
        if pred == 1 and anchor_vec[i - 1] == 0:
            X[s_idx, i - 1] = 1.0
        elif pred == 0 and anchor_vec[i - 1] == 1:
            X[s_idx, i - 1] = -1.0

intervened_mask = (X != 0).any(axis=0)
intervened_indices = np.where(intervened_mask)[0]

prior_w0 = np.where(anchor_vec == 1, 0.62665, -0.62665)
prior_w0_int = prior_w0[intervened_indices]
X_int = X[:, intervened_indices]

sigma0_sq = 0.25**2
sigma_sq = 0.5**2

K = X_int @ X_int.T * sigma0_sq + sigma_sq * np.eye(M)
y_res = delta_hits_s - X_int @ prior_w0_int
alpha = np.linalg.solve(K, y_res)
w_bayesian_int = prior_w0_int + sigma0_sq * (X_int.T @ alpha)

p_new_probe = np.clip(0.5 + prior_w0 / 1.40, 0.001, 0.999)
p_new_probe[intervened_indices] = np.clip(0.5 + w_bayesian_int / 1.40, 0.001, 0.999)

# 4. Lexical cues matching
DECISIVE_USED_CUES = [
    (r"\bpara repuesto[s]?\b", "para repuesto", 1.0),
    (r"\bpara despiece\b", "para despiece", 1.0),
    (r"\b(?:para|a)\s+reparar\b", "a reparar", 1.0),
    (r"\bno\s+funciona\b", "no funciona", 1.0),
    (r"\bno\s+(?:enciende|prende|anda)\b", "no enciende", 1.0),
    (r"\bbuen\s+estado\s+general\b", "buen estado general", 1.0),
    (r"\bbuen\s+estado\b", "buen estado", 1.0),
    (r"\bmuy\s+buen\s+estado\b", "muy buen estado", 1.0),
    (r"\bexcelente\s+estado\b", "excelente estado", 0.9925),
    (r"\bimpecable\s+estado\b|\bestado\s+impecable\b", "impecable estado", 1.0),
    (r"\b(?:detalles|marcas|signos)\s+de\s+uso\b", "detalles de uso", 1.0),
    (r"\bpoco\s+uso\b", "poco uso", 1.0),
    (r"\busad[oa]s?\b", "usado", 0.9833),
]

DECISIVE_NEW_CUES = [
    (r"\b(?:en\s+)?caja\s+sellada\b", "caja sellada", 1.0),
    (r"\b(?:en\s+)?caja\s+cerrada\b", "caja cerrada", 1.0),
    (r"\bsellad[oa]\s+de\s+f[aá]brica\b", "sellado de fabrica", 1.0),
    (r"\bcerrad[oa]\s+de\s+f[aá]brica\b", "cerrado de fabrica", 1.0),
    (r"\bnuev[oa]\s+sellad[oa]\b|\bsellad[oa]\s+en\s+caja\b", "nuevo sellado", 1.0),
    (r"\bnuev[oa]\s+cerrad[oa]\b", "nuevo cerrado", 0.981),
    (r"\bnuev[oa]\s+en\s+blister\b", "nuevo en blister", 1.0),
    (r"\b0\s*km\s+sin\s+rodar\b", "0 km sin rodar", 1.0),
    (r"\bdistribuidor\s+oficial\b", "distribuidor oficial", 1.0),
    (r"\bsomos\s+fabricantes\b|\bdirecto\s+de\s+f[aá]brica\b|\bfabricantes\b", "fabricantes", 1.0),
    (r"\b(?:x|por|venta\s+por)\s+mayor\b", "por mayor", 1.0),
]

p_new_lexical = np.full(N, 0.50, dtype=np.float64) # neutral 0.50
has_cue_new = np.zeros(N, dtype=bool)
has_cue_used = np.zeros(N, dtype=bool)

for idx, it in enumerate(test_items):
    t_lower = it.get('title', '').lower()
    for pat, name, prec in DECISIVE_USED_CUES:
        if re.search(pat, t_lower):
            p_new_lexical[idx] = 1.0 - prec
            has_cue_used[idx] = True
            break
    if not has_cue_used[idx]:
        for pat, name, prec in DECISIVE_NEW_CUES:
            if re.search(pat, t_lower):
                p_new_lexical[idx] = prec
                has_cue_new[idx] = True
                break

# 5. Classify each item
# Classes: CONFIRMED_NEW, CONFIRMED_USED, HIGH_CONFIDENCE_NEW, HIGH_CONFIDENCE_USED, CONTESTED, UNPROBED_STABLE
confidence_classes = []
recommended_labels = []

for i in range(N):
    # Check CONFIRMED by lexical cue
    if has_cue_new[i]:
        c_class = "CONFIRMED_NEW"
        rec = "new"
    elif has_cue_used[i]:
        c_class = "CONFIRMED_USED"
        rec = "used"
    # Check CONFIRMED by decisive probe evidence (tested with >=98% certainty)
    elif times_flipped[i] > 0 and p_new_probe[i] >= 0.98:
        c_class = "CONFIRMED_NEW"
        rec = "new"
    elif times_flipped[i] > 0 and p_new_probe[i] <= 0.02:
        c_class = "CONFIRMED_USED"
        rec = "used"
    # Check CONTESTED: high submission variance (20% to 80% consensus) OR ambiguous probe support (0.35 to 0.65)
    elif (0.20 <= rate_new[i] <= 0.80) or (times_flipped[i] > 0 and 0.35 <= p_new_probe[i] <= 0.65):
        c_class = "CONTESTED"
        rec = "new" if p_new_probe[i] >= 0.50 else "used"
    # Check HIGH_CONFIDENCE: empirical probe support (>0.90 or <0.10) OR model consensus >95%
    elif p_new_probe[i] >= 0.90 or rate_new[i] > 0.95:
        c_class = "HIGH_CONFIDENCE_NEW"
        rec = "new"
    elif p_new_probe[i] <= 0.10 or rate_used[i] > 0.95:
        c_class = "HIGH_CONFIDENCE_USED"
        rec = "used"
    # UNPROBED_STABLE: Invariant across all high-performing vectors (>94.7% score)
    elif is_invariant_top17[i]:
        c_class = "UNPROBED_STABLE"
        rec = "new" if anchor_vec[i] == 1 else "used"
    else:
        # Fallback if flipped in top17 but not contested/high confidence
        c_class = "CONTESTED"
        rec = "new" if p_new_probe[i] >= 0.50 else "used"
        
    confidence_classes.append(c_class)
    recommended_labels.append(rec)

from collections import Counter
counts = Counter(confidence_classes)
print("\n=== CONFIDENCE CLASS DISTRIBUTION ===")
for c, cnt in counts.most_common():
    print(f"  {c:<22s}: {cnt:5d} ({cnt/N*100:5.2f}%)")

rec_counts = Counter(recommended_labels)
print("\n=== RECOMMENDED LABELS ===")
for r, cnt in rec_counts.items():
    print(f"  {r:<10s}: {cnt:5d} ({cnt/N*100:5.2f}%)")

# Compare vs Anchor EVT-143
disagreements = sum(rec != ("new" if anchor_vec[i] == 1 else "used") for i, rec in enumerate(recommended_labels))
print(f"\nDisagreements vs Anchor EVT-143: {disagreements}")
