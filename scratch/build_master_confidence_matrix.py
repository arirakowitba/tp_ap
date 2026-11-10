"""Build complete 30,000 item confidence matrix combining 116 vectors, Quad-Titan probabilities, and verified patterns.
"""
import json, os, re
import numpy as np
import pandas as pd

print("1. Loading test items and metadata...", flush=True)
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]
n_test = len(test_rows)

# 2. Load EVT-143 anchor
anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
col_cond = 'condition' if 'condition' in anchor_df.columns else anchor_df.columns[1]
anchor_labels = anchor_df[col_cond].values

# 3. Load Quad-Titan probabilities
quad_df = pd.read_csv('results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv')
p_quad = quad_df['p_used_quad_titan'].values
p_tree = quad_df['p_used_tri_titan_trees'].values
p_txt = quad_df['p_used_sparse_text'].values
quad_labels = quad_df['pred_label_quad_titan'].values

# 4. Load scored vectors for consensus
with open('results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json') as f:
    cat = json.load(f)

scored_events = [e for e in cat if e.get('public_score') is not None and e.get('vector_path') and os.path.exists(e.get('vector_path'))]
elite_events = [e for e in scored_events if e.get('public_score', 0) >= 0.94700]

print(f"Loaded {len(scored_events)} total scored vectors, {len(elite_events)} elite vectors.")

matrix_all = np.zeros((len(scored_events), n_test), dtype=np.int8)
for i, ev in enumerate(scored_events):
    df_v = pd.read_csv(ev['vector_path'])
    c_col = 'condition' if 'condition' in df_v.columns else df_v.columns[1]
    matrix_all[i, :] = (df_v[c_col].values == 'used').astype(np.int8)

elite_indices = [i for i, ev in enumerate(scored_events) if ev.get('public_score', 0) >= 0.94700]
matrix_elite = matrix_all[elite_indices, :]

used_all_counts = np.sum(matrix_all, axis=0)
pct_all_used = used_all_counts / len(scored_events)

used_elite_counts = np.sum(matrix_elite, axis=0)
pct_elite_used = used_elite_counts / len(elite_indices)

# Invariance tier
invariance_tier = []
for idx in range(n_test):
    u_all = used_all_counts[idx]
    u_el = used_elite_counts[idx]
    if u_all == 0 or u_all == len(scored_events):
        invariance_tier.append("INVARIANT_ALL")
    elif u_el == 0 or u_el == len(elite_indices):
        invariance_tier.append("INVARIANT_ELITE")
    else:
        invariance_tier.append("ELITE_DISPUTED")

# 5. Lexical cues matching
patterns = [
    ("repuesto", re.compile(r"\bpara\s+repuestos?\b|\bpara\s+desarme\b|\bno\s+funciona\b|\brot[oa]s?\b|\bcon\s+falla\b", re.I)),
    ("caja_sellada", re.compile(r"\bcaja\s+sellada\b|\bcerrad[oa]\s+de\s+fabrica\b|\bprecintad[oa]\b|\ben\s+blister\b", re.I)),
    ("0km", re.compile(r"\b0\s*km\b|\bzero\s*km\b|\bsin\s+rodar\b|\b0km\b", re.I)),
    ("digital", re.compile(r"\blibro\s+digital\b|\bformato\s+digital\b|\bdescarga\s+digital\b|\bdigital\s+oferta\b|\bebook\b|\bpdf\b", re.I)),
    ("usado_explicit", re.compile(r"\busad[oa]s?\b|\bsegunda\s+mano\b", re.I)),
    ("impecable", re.compile(r"\bimpecable\b|\bexcelente\s+estado\b|\bcomo\s+nuev[oa]\b|\bmuy\s+buen\s+estado\b", re.I)),
    ("antiguo", re.compile(r"\bantigu[oa]s?\b|\bvintage\b|\bretro\b|\bcolecci[oó]n\b", re.I)),
    ("reacondicionado", re.compile(r"\breacondicionad[oa]\b|\breacond\b", re.I)),
]

# Load certified 42 flips
cert_df = pd.read_csv('results/certified_quad_titan_flips_20261110.csv')
cert_ids = set(cert_df['ID'].values)

print("2. Computing item-by-item confidence scores and actions...", flush=True)
records = []
for idx in range(n_test):
    item_id = idx + 1
    r = test_rows[idx]
    title = str(r.get('title') or '').strip()
    cat = str(r.get('category_id') or '')
    price = float(r.get('price') or 0.0)
    anc = anchor_labels[idx]
    quad_l = quad_labels[idx]
    
    p_q = p_quad[idx]
    p_tr = p_tree[idx]
    p_tx = p_txt[idx]
    tier = invariance_tier[idx]
    
    # Check lexical cue
    matched_cue = ""
    for name, pat in patterns:
        if pat.search(title):
            matched_cue = name
            break
            
    # Consensus agreement
    tree_text_agree = (p_tr >= 0.5 and p_tx >= 0.5) or (p_tr < 0.5 and p_tx < 0.5)
    
    # Calibrated confidence of anchor label
    p_anchor_correct = (p_q if anc == 'used' else (1.0 - p_q))
    
    # Adjust confidence by empirical vector consensus
    if tier == "INVARIANT_ALL":
        conf = 0.90 + 0.09 * p_anchor_correct
    elif tier == "INVARIANT_ELITE":
        conf = 0.85 + 0.12 * p_anchor_correct
    else: # ELITE_DISPUTED
        conf = 0.50 + 0.40 * p_anchor_correct
    conf = min(0.999, max(0.50, conf))
    
    # Recommended Action
    if item_id in cert_ids:
        action = "CERTIFIED_FLIP"
    elif anc != quad_l and abs(p_q - 0.5) > 0.30 and tree_text_agree:
        action = "EXPLORATORY_CANDIDATE"
    elif anc == quad_l:
        action = "KEEP_ANCHOR"
    else:
        action = "UNCERTAIN_DISPUTED"
        
    records.append({
        'ID': item_id,
        'title': title,
        'category_id': cat,
        'price': price,
        'anchor_evt143': anc,
        'quad_titan_pred': quad_l,
        'p_used_quad_titan': round(float(p_q), 4),
        'p_used_trees': round(float(p_tr), 4),
        'p_used_sparse_text': round(float(p_tx), 4),
        'tree_text_agreement': bool(tree_text_agree),
        'pct_all_vectors_used': round(float(pct_all_used[idx]), 3),
        'pct_elite_vectors_used': round(float(pct_elite_used[idx]), 3),
        'invariance_tier': tier,
        'decisive_lexical_cue': matched_cue,
        'confidence_in_anchor': round(float(conf), 4),
        'recommended_action': action,
    })

df_out = pd.DataFrame(records)
out_csv = 'results/master_item_confidences_30k_20261110.csv'
df_out.to_csv(out_csv, index=False)
print(f"3. Saved master confidence matrix to {out_csv} ({len(df_out)} rows)")

# Summary distributions
summary = {
    'total_items': n_test,
    'invariance_tier_distribution': {
        tier: int(count) for tier, count in df_out['invariance_tier'].value_counts().items()
    },
    'action_distribution': {
        act: int(count) for act, count in df_out['recommended_action'].value_counts().items()
    },
    'confidence_percentiles': {
        'p10': float(df_out['confidence_in_anchor'].quantile(0.10)),
        'p25': float(df_out['confidence_in_anchor'].quantile(0.25)),
        'p50': float(df_out['confidence_in_anchor'].quantile(0.50)),
        'p75': float(df_out['confidence_in_anchor'].quantile(0.75)),
        'p90': float(df_out['confidence_in_anchor'].quantile(0.90)),
        'p99': float(df_out['confidence_in_anchor'].quantile(0.99)),
    },
    'certified_flips_count': len(cert_ids),
    'exploratory_candidates_count': int((df_out['recommended_action'] == 'EXPLORATORY_CANDIDATE').sum()),
}

out_json = 'results/master_prediction_confidences_summary_20261110.json'
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"4. Saved summary to {out_json}")
print("\n--- SUMMARY STATS ---")
print(json.dumps(summary, indent=2))
