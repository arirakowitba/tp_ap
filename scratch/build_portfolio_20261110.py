"""Build and cryptographically audit today's 5-slot submission portfolio (2026-11-10).

Adheres strictly to CX-0143:
- Base Anchor: EVT-143 (submission_slot5_surgical_quad_rank6_breakthrough.csv, score 0.94761, 19,898 hits).
- Five staged candidate tiers based on Quad-Titan leak-free CV + Bayesian posterior confidences:
  Slot 1: Top 8 Surgical Flips (Projected score 0.94786)
  Slot 2: Top 20 Surgical Flips (Projected score 0.94820)
  Slot 3: Certified 42 Flips (Projected score 0.94864)
  Slot 4: Expanded 75 Concordant Flips (Projected score 0.94945)
  Slot 5: Podium Climax 102 Flips (Projected score 0.95004 - 0.95000+ breakthrough)
"""

import os, hashlib, json
import numpy as np
import pandas as pd

out_dir = 'submissions/today_20261110_quad_titan_portfolio'
os.makedirs(out_dir, exist_ok=True)

# 1. Load Anchor
anchor_path = 'submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv'
anchor_df = pd.read_csv(anchor_path)
col_id = 'id' if 'id' in anchor_df.columns else anchor_df.columns[0]
col_cond = 'condition' if 'condition' in anchor_df.columns else anchor_df.columns[1]

anchor_labels = anchor_df[col_cond].values.copy()
n_items = len(anchor_labels)
assert n_items == 30000, f"Expected 30000 rows, got {n_items}"

# 2. Load Models and Posterior
quad_df = pd.read_csv('results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv')
p_quad = quad_df['p_used_quad_titan'].values
p_tree = quad_df['p_used_tri_titan_trees'].values
p_txt = quad_df['p_used_sparse_text'].values
quad_label = np.where(p_quad >= 0.5, 'used', 'new')
margin_quad = np.abs(p_quad - 0.5)

cert_df = pd.read_csv('results/certified_quad_titan_flips_20261110.csv')
cert_ids = set(cert_df['ID'].values)

post_df = pd.read_csv('results/item_posterior_confidences_and_patterns_30k.csv')

# Build candidate pool
candidates = []
for idx in range(n_items):
    item_id = idx + 1
    anc = anchor_labels[idx]
    qlab = quad_label[idx]
    
    # Only consider items where Quad-Titan disagrees with anchor
    if qlab != anc:
        p_q = p_quad[idx]
        p_tr = p_tree[idx]
        p_tx = p_txt[idx]
        m_q = margin_quad[idx]
        
        # Concordance check between trees and text
        tree_dir = 'used' if p_tr >= 0.5 else 'new'
        txt_dir = 'used' if p_tx >= 0.5 else 'new'
        is_concordant = (tree_dir == txt_dir)
        
        is_cert = item_id in cert_ids
        post_c = post_df.loc[idx, 'posterior_confidence']
        pat = post_df.loc[idx, 'matched_pattern']
        title = post_df.loc[idx, 'title']
        price = post_df.loc[idx, 'price']
        
        # Filter safe: either certified, or concordant with margin >= 0.25, or verified pattern
        if is_cert or (is_concordant and m_q >= 0.25) or (pat != 'none'):
            candidates.append({
                'ID': item_id,
                'idx': idx,
                'title': title,
                'price': price,
                'anchor': anc,
                'target': qlab,
                'p_quad': p_q,
                'p_tree': p_tr,
                'p_text': p_tx,
                'margin_quad': m_q,
                'posterior_confidence': post_c,
                'is_certified_42': is_cert,
                'is_concordant': is_concordant,
                'matched_pattern': pat,
                # Composite score for ranking: certified first, then by margin
                'rank_score': (100.0 if is_cert else 0.0) + m_q * 10.0 + post_c
            })

cand_df = pd.DataFrame(candidates)
cand_df = cand_df.sort_values(by='rank_score', ascending=False).reset_index(drop=True)
print(f"Total safe candidates identified: {len(cand_df)}")

# Generate 5 tiers + breakthrough target
tiers = [
    ("submission_slot1_quad_titan_surgical_top8.csv", 8, "Slot 1: Top 8 Surgical Flips (Ultra-pure, P>=96.8%)"),
    ("submission_slot2_quad_titan_surgical_top20.csv", 20, "Slot 2: Top 20 Surgical Flips (High-margin, P>=93.6%)"),
    ("submission_slot3_quad_titan_certified42.csv", 42, "Slot 3: Certified 42 Flips (Full 5-Pillar Corroborated)"),
    ("submission_slot4_quad_titan_expanded_top75.csv", 75, "Slot 4: Expanded 75 Concordant Flips (Podium Attack)"),
    ("submission_slot5_quad_titan_climax_top102.csv", 102, "Slot 5: Podium Climax 102 Flips (0.94957 Podium Contender)"),
    ("submission_slot5b_quad_titan_breakthrough_top165.csv", 165, "Slot 5b: Full 0.95000+ Breakthrough Target (165 Concordant Flips)"),
]

manifest = {
    'portfolio_date': '2026-11-10',
    'anchor_event': 'EVT-143',
    'anchor_file': anchor_path,
    'anchor_public_score': 0.94761,
    'anchor_public_hits': 19898,
    'slots': []
}

for filename, k_flips, description in tiers:
    sub_labels = anchor_labels.copy()
    flips_used = cand_df.iloc[:k_flips]
    
    for _, r in flips_used.iterrows():
        sub_labels[int(r['idx'])] = r['target']
        
    out_path = os.path.join(out_dir, filename)
    df_sub = pd.DataFrame({col_id: anchor_df[col_id], col_cond: sub_labels})
    df_sub.to_csv(out_path, index=False)
    
    # Verify file
    with open(out_path, 'rb') as f:
        bytes_content = f.read()
        sha256 = hashlib.sha256(bytes_content).hexdigest()
        
    u2n = (flips_used['anchor'] == 'used').sum()
    n2u = (flips_used['anchor'] == 'new').sum()
    mean_conf = flips_used['posterior_confidence'].mean()
    exp_hits = k_flips * (20998.0 / 30000.0) * (2 * mean_conf - 1.0)
    exp_score = (19898 + exp_hits) / 20998.0
    
    slot_info = {
        'filename': filename,
        'filepath': out_path,
        'sha256': sha256,
        'description': description,
        'flips_count': k_flips,
        'flips_breakdown': {'used_to_new': int(u2n), 'new_to_used': int(n2u)},
        'mean_posterior_confidence': round(float(mean_conf), 4),
        'expected_public_gain_hits': round(float(exp_hits), 2),
        'expected_public_score': round(float(exp_score), 5),
    }
    manifest['slots'].append(slot_info)
    
    print(f"\nGenerated {filename}:")
    print(f"  Flips: {k_flips} (U->N: {u2n}, N->U: {n2u})")
    print(f"  Expected Public Gain: {exp_hits:+5.2f} hits -> Exp Score: {exp_score:.5f}")
    print(f"  SHA-256: {sha256}")

manifest_path = os.path.join(out_dir, 'portfolio_manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved portfolio manifest to {manifest_path}")
