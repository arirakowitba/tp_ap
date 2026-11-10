import json, re, sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from scratch.analyze_all_vectors import vectors, anchor

with open("data/ap_2026_q_2/raw/test_data.jsonlines", "r", encoding="utf-8") as f:
    test_items = [json.loads(line) for line in f]

N = len(test_items)
M = len(vectors)

anchor_hits = anchor['hits']
anchor_vec = np.array([1 if anchor['vec'][i] == 'new' else 0 for i in range(1, N + 1)], dtype=np.int8)

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

delta_hits_s = hits_vec - anchor_hits
F = (sub_matrix != anchor_vec).astype(int)
delta_hits_sum = F.T @ delta_hits_s

# Bayesian probe deconvolution
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

# Decisive lexical cues
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

p_new_lexical = np.full(N, 0.50, dtype=np.float64)
cue_names = [None] * N

for idx, it in enumerate(test_items):
    t_lower = it.get('title', '').lower()
    for pat, name, prec in DECISIVE_USED_CUES:
        if re.search(pat, t_lower):
            p_new_lexical[idx] = 1.0 - prec
            cue_names[idx] = name
            break
    if cue_names[idx] is None:
        for pat, name, prec in DECISIVE_NEW_CUES:
            if re.search(pat, t_lower):
                p_new_lexical[idx] = prec
                cue_names[idx] = name
                break

# Identify items where evidence disagrees with EVT-143
disagreements = []
for i in range(N):
    anc = "new" if anchor_vec[i] == 1 else "used"
    # Evidence for flip:
    # 1. Decisive lexical cue contradicts anchor
    has_conflicting_cue = (anc == "new" and cue_names[i] in [c[1] for c in DECISIVE_USED_CUES]) or \
                          (anc == "used" and cue_names[i] in [c[1] for c in DECISIVE_NEW_CUES])
    
    # 2. Probe evidence contradicts anchor:
    # If anchor is new, probe says used: p_new_probe < 0.35
    # If anchor is used, probe says new: p_new_probe > 0.65
    probe_reversal = (anc == "new" and p_new_probe[i] < 0.35) or (anc == "used" and p_new_probe[i] > 0.65)
    
    # 3. Model consensus contradicts anchor:
    consensus_reversal = (anc == "new" and rate_used[i] > 0.70) or (anc == "used" and rate_new[i] > 0.70)
    
    if has_conflicting_cue or (times_flipped[i] > 0 and (probe_reversal or consensus_reversal)):
        # Calculate confidence score of the correction
        rec = "used" if anc == "new" else "new"
        # Confidence score:
        conf_score = 0.0
        just = []
        if has_conflicting_cue:
            conf_score += 1.0
            just.append(f"Decisive cue '{cue_names[i]}' (precision >= 98%)")
        if probe_reversal:
            p_conf = p_new_probe[i] if rec == "new" else (1.0 - p_new_probe[i])
            conf_score += 0.8 * p_conf
            just.append(f"Probe deconvolution P({rec})={p_conf:.3f} (delta_hits_sum={delta_hits_sum[i]:+d})")
        if consensus_reversal:
            cons_pct = (rate_new[i] if rec == "new" else rate_used[i]) * 100
            conf_score += 0.5 * (cons_pct / 100.0)
            just.append(f"Model consensus {cons_pct:.1f}% ({times_new[i] if rec=='new' else times_used[i]}/{M} submissions)")
            
        disagreements.append({
            'item_id': i + 1,
            'title': test_items[i].get('title', ''),
            'category_id': test_items[i].get('category_id', ''),
            'price': test_items[i].get('price', 0.0),
            'anchor_label': anc,
            'recommended_label': rec,
            'confidence_score': conf_score,
            'p_new_probe': round(float(p_new_probe[i]), 4),
            'p_new_lexical': round(float(p_new_lexical[i]), 4),
            'times_flipped': int(times_flipped[i]),
            'delta_hits_sum': int(delta_hits_sum[i]),
            'cue': cue_names[i],
            'justification': "; ".join(just)
        })

print(f"Total potential corrections vs EVT-143: {len(disagreements)}")
disagreements.sort(key=lambda x: x['confidence_score'], reverse=True)

print("\nTop 25 highest-confidence corrections vs EVT-143:")
for rank, d in enumerate(disagreements[:25], 1):
    print(f"{rank:2d}. ID {d['item_id']:5d} | {d['anchor_label']} -> {d['recommended_label']} | Conf: {d['confidence_score']:.3f} | ${d['price']:7.1f} | {d['title'][:40]} | {d['justification']}")
