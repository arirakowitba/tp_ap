"""Master empirical pattern mining across all 116 vectors and Quad-Titan probabilities.
"""
import json, os, glob, re
import numpy as np
import pandas as pd
from collections import Counter

print("Loading canonical ledger...", flush=True)
with open('results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json') as f:
    cat = json.load(f)

scored_events = [e for e in cat if e.get('public_score') is not None and e.get('vector_path') and os.path.exists(e.get('vector_path'))]
print(f"Loaded {len(scored_events)} scored vectors with existing files.")

# Load test rows
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]
n_test = len(test_rows)
print(f"Loaded {n_test} test items.")

# Load Quad-Titan probabilities
quad_df = pd.read_csv('results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv')
p_quad_used = quad_df['p_used_quad_titan'].values
p_trees_used = quad_df['p_used_tri_titan_trees'].values
p_txt_used = quad_df['p_used_sparse_text'].values

# Load EVT-143 anchor
anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
col_cond = 'condition' if 'condition' in anchor_df.columns else anchor_df.columns[1]
anchor_labels = anchor_df[col_cond].values

# Read all 116 vectors into a binary matrix (1=used, 0=new)
matrix = np.zeros((len(scored_events), n_test), dtype=np.int8)
scores = np.array([e['public_score'] for e in scored_events])
hits = np.array([e['public_hits'] for e in scored_events])
eids = [e['event_id'] for e in scored_events]

for i, ev in enumerate(scored_events):
    df_v = pd.read_csv(ev['vector_path'])
    c_col = 'condition' if 'condition' in df_v.columns else df_v.columns[1]
    matrix[i, :] = (df_v[c_col].values == 'used').astype(np.int8)

print(f"Built vector matrix: {matrix.shape}")

# Invariance statistics
used_counts = np.sum(matrix, axis=0) # times predicted used
new_counts = len(scored_events) - used_counts

always_used = (used_counts == len(scored_events))
always_new = (new_counts == len(scored_events))
strictly_invariant = always_used | always_new

print(f"\n--- INVARIANCE CENSUS ACROSS ALL {len(scored_events)} VECTORS ---")
print(f"Strictly invariant items: {np.sum(strictly_invariant)} ({np.sum(strictly_invariant)/n_test*100:.2f}%)")
print(f"  Always USED: {np.sum(always_used)} ({np.sum(always_used)/n_test*100:.2f}%)")
print(f"  Always NEW:  {np.sum(always_new)} ({np.sum(always_new)/n_test*100:.2f}%)")
print(f"Intervened / Disputed items: {n_test - np.sum(strictly_invariant)} ({(n_test - np.sum(strictly_invariant))/n_test*100:.2f}%)")

# Top 17 elite vectors (score >= 0.94700)
elite_idx = np.where(scores >= 0.94700)[0]
matrix_elite = matrix[elite_idx, :]
used_elite = np.sum(matrix_elite, axis=0)
new_elite = len(elite_idx) - used_elite

elite_invariant = (used_elite == len(elite_idx)) | (new_elite == len(elite_idx))
print(f"\n--- ELITE INVARIANCE (TOP {len(elite_idx)} VECTORS >= 0.94700) ---")
print(f"Elite invariant items: {np.sum(elite_invariant)} ({np.sum(elite_invariant)/n_test*100:.2f}%)")
print(f"Disputed items in elite: {n_test - np.sum(elite_invariant)} ({(n_test - np.sum(elite_invariant))/n_test*100:.2f}%)")

# Correlation between vector consensus and Quad-Titan probabilities
p_consensus_used = used_counts / len(scored_events)
corr_consensus_quad = np.corrcoef(p_consensus_used, p_quad_used)[0, 1]
print(f"\nPearson correlation between 116-Vector Consensus and Quad-Titan OOF: r = {corr_consensus_quad:.4f}")

# Lexical Patterns Analysis
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

print("\n--- LEXICAL CUES PATTERN ANALYSIS ON DISPUTED ITEMS ---")
disputed_indices = np.where(~elite_invariant)[0]
pattern_stats = {}
for name, pat in patterns:
    matches = [i for i in disputed_indices if pat.search(test_rows[i].get('title') or '')]
    if matches:
        q_used = p_quad_used[matches]
        anc_used = (anchor_labels[matches] == 'used')
        print(f"Pattern '{name:<16}': {len(matches):3d} matches in elite disputed | Mean Quad p_used={np.mean(q_used):.3f} | Current Anchor used={np.mean(anc_used)*100:.1f}%")
        pattern_stats[name] = {
            'count': len(matches),
            'mean_p_used': float(np.mean(q_used)),
            'anchor_used_pct': float(np.mean(anc_used)*100)
        }

# Save census results
census_report = {
    'total_vectors': len(scored_events),
    'strictly_invariant_count': int(np.sum(strictly_invariant)),
    'elite_invariant_count': int(np.sum(elite_invariant)),
    'elite_disputed_count': int(n_test - np.sum(elite_invariant)),
    'correlation_consensus_quad': float(corr_consensus_quad),
    'pattern_stats': pattern_stats,
}
with open('results/master_empirical_patterns_census_20261110.json', 'w') as f:
    json.dump(census_report, f, indent=2)
print("\nSaved census report to results/master_empirical_patterns_census_20261110.json")
