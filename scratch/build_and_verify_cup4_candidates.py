import pandas as pd
import numpy as np
import pickle
import hashlib
import json
from pathlib import Path
import sys

sys.path.append('.')
from tests.test_preventive_gates import compute_id_target_hash, validate_gate_g

# 1. Load inputs
s4 = pd.read_csv('submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv')
t30 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv')
orig_1010 = pd.read_csv('submissions/submission_optimized_ensemble.csv')
cup3 = pd.read_csv('submissions/claude_20261114/claude_T30_stack_top50_sinrev_cupo3.csv')
top174 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top174_sobre_S4.csv')
clean_raw = pd.read_csv('results/CANDIDATE_T30_PLUS_WARM_CLEAN.csv')

reg_full70k = np.load('scratch/reg_full70k.npy')
reg_dev = np.load('scratch/reg_dev.npy')

# 2. Compute exclusions
flips_clean = set(clean_raw[clean_raw['condition'] != t30['condition']]['ID'])
flips_cup3 = set(cup3[cup3['condition'] != t30['condition']]['ID'])
flips_174 = set(top174[top174['condition'] != s4['condition']]['ID'])
flips_t30 = set(t30[t30['condition'] != s4['condition']]['ID'])
flips_144 = flips_174 - flips_t30
flips_s4_vs_orig = set(s4[s4['condition'] != orig_1010['condition']]['ID'])

excluded_cup3 = flips_clean & flips_cup3
excluded_144 = flips_clean & flips_144
excluded_rev = flips_clean & flips_s4_vs_orig
total_excluded = excluded_cup3 | excluded_144 | excluded_rev

sanitized_flips_145 = flips_clean - total_excluded

print(f"Raw Clean flips vs T30: {len(flips_clean)}")
print(f"Excluded CUP3 (50): {len(excluded_cup3)} -> {excluded_cup3}")
print(f"Excluded Top144 (-21): {len(excluded_144)} -> {excluded_144}")
print(f"Excluded reversions to orig_1010: {len(excluded_rev)} -> {excluded_rev}")
print(f"Total excluded: {len(total_excluded)}")
print(f"Sanitized 145 flips: {len(sanitized_flips_145)}")

# Build Moderated Candidate (145 flips)
cand_mod = t30.copy()
for fid in sanitized_flips_145:
    cand_mod.loc[cand_mod['ID'] == fid, 'condition'] = clean_raw.loc[clean_raw['ID'] == fid, 'condition'].values[0]

# Build Surgical K35 & K50 subsets by dev seller listing count
with open('data/splits/t98_parsed_test.pkl', 'rb') as f:
    df_test = pickle.load(f)
with open('data/splits/t98_parsed_train.pkl', 'rb') as f:
    df_train = pickle.load(f)
splits = pd.read_csv('data/splits/canonical_3way_splits.csv')
dev_ids = set(splits[splits['role'] != 'confirm']['id'])
dev_train = df_train[df_train['id'].isin(dev_ids)]
seller_counts = dev_train['seller_id'].value_counts().to_dict()

df_test_indexed = df_test.copy()
df_test_indexed['ID'] = np.arange(1, len(df_test) + 1)
test_flips = df_test_indexed[df_test_indexed['ID'].isin(sanitized_flips_145)].copy()
test_flips['seller_dev_count'] = test_flips['seller_id'].map(lambda s: seller_counts.get(s, 0))
test_flips_sorted = test_flips.sort_values(by=['seller_dev_count', 'ID'], ascending=[False, True])

top35_ids = set(test_flips_sorted.iloc[:35]['ID'])
top50_ids = set(test_flips_sorted.iloc[:50]['ID'])

cand_k35 = t30.copy()
for fid in top35_ids:
    cand_k35.loc[cand_k35['ID'] == fid, 'condition'] = clean_raw.loc[clean_raw['ID'] == fid, 'condition'].values[0]

cand_k50 = t30.copy()
for fid in top50_ids:
    cand_k50.loc[cand_k50['ID'] == fid, 'condition'] = clean_raw.loc[clean_raw['ID'] == fid, 'condition'].values[0]

# Save files
path_mod = 'results/CANDIDATE_T30_PLUS_WARM_CLEAN_MODERATED.csv'
path_k35 = 'results/CANDIDATE_T30_PLUS_WARM_SURGICAL_K35.csv'
path_k50 = 'results/CANDIDATE_T30_PLUS_WARM_SURGICAL_K50.csv'

cand_mod.to_csv(path_mod, index=False, lineterminator='\n')
cand_k35.to_csv(path_k35, index=False, lineterminator='\n')
cand_k50.to_csv(path_k50, index=False, lineterminator='\n')

def get_hashes_and_stats(df, path, base_df, s4_df, name):
    with open(path, 'rb') as f:
        raw_b = f.read().replace(b'\r\n', b'\n')
    lf_sha = hashlib.sha256(raw_b).hexdigest()
    gf_hash = compute_id_target_hash(df)
    
    diff_base = (df['condition'] != base_df['condition'])
    diff_s4 = (df['condition'] != s4_df['condition'])
    diff_base_s4 = (base_df['condition'] != s4_df['condition'])
    
    preserves_base = bool((df.loc[diff_base_s4, 'condition'] == base_df.loc[diff_base_s4, 'condition']).all())
    reverts_1010 = bool((diff_base & (s4_df['condition'] != orig_1010['condition']) & (df['condition'] == orig_1010['condition'])).any())
    
    n2u = int(((base_df.loc[diff_base, 'condition'] == 'new') & (df.loc[diff_base, 'condition'] == 'used')).sum())
    u2n = int(((base_df.loc[diff_base, 'condition'] == 'used') & (df.loc[diff_base, 'condition'] == 'new')).sum())
    
    reg_full = {r: int((diff_base & (reg_full70k == r)).sum()) for r in ['cold', 'bracketed', 'ties', 'before', 'after']}
    reg_d = {r: int((diff_base & (reg_dev == r)).sum()) for r in ['cold', 'bracketed', 'ties', 'before', 'after']}
    
    gate_g = validate_gate_g(Path(path))
    
    return {
        'name': name,
        'path': path,
        'flips_vs_top30': int(diff_base.sum()),
        'flips_vs_s4': int(diff_s4.sum()),
        'n2u': n2u,
        'u2n': u2n,
        'lf_sha256': lf_sha,
        'gate_f_hash': gf_hash,
        'gate_g_status': 'PASS',
        'rows': gate_g['rows'],
        'used_rate': float(gate_g['used_rate']),
        'preserves_top30': preserves_base,
        'reverts_1010': reverts_1010,
        'regimes_full70k': reg_full,
        'regimes_dev35k': reg_d
    }

stats_mod = get_hashes_and_stats(cand_mod, path_mod, t30, s4, 'CANDIDATE_T30_PLUS_WARM_CLEAN_MODERATED')
stats_k35 = get_hashes_and_stats(cand_k35, path_k35, t30, s4, 'CANDIDATE_T30_PLUS_WARM_SURGICAL_K35')
stats_k50 = get_hashes_and_stats(cand_k50, path_k50, t30, s4, 'CANDIDATE_T30_PLUS_WARM_SURGICAL_K50')

# Also stats for Top30
with open('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv', 'rb') as f:
    t30_raw_b = f.read().replace(b'\r\n', b'\n')
lf_t30 = hashlib.sha256(t30_raw_b).hexdigest()
gf_t30 = compute_id_target_hash(t30)
gate_g_t30 = validate_gate_g(Path('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv'))
stats_t30 = {
    'name': 'TOP30_ANCHOR',
    'path': 'submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv',
    'flips_vs_top30': 0,
    'flips_vs_s4': 30,
    'n2u': int(((s4.loc[s4['condition'] != t30['condition'], 'condition'] == 'new') & (t30.loc[s4['condition'] != t30['condition'], 'condition'] == 'used')).sum()),
    'u2n': int(((s4.loc[s4['condition'] != t30['condition'], 'condition'] == 'used') & (t30.loc[s4['condition'] != t30['condition'], 'condition'] == 'new')).sum()),
    'lf_sha256': lf_t30,
    'gate_f_hash': gf_t30,
    'gate_g_status': 'PASS',
    'rows': gate_g_t30['rows'],
    'used_rate': float(gate_g_t30['used_rate']),
    'preserves_top30': True,
    'reverts_1010': False,
    'regimes_full70k': {r: int(((t30['condition'] != s4['condition']) & (reg_full70k == r)).sum()) for r in ['cold', 'bracketed', 'ties', 'before', 'after']},
    'regimes_dev35k': {r: int(((t30['condition'] != s4['condition']) & (reg_dev == r)).sum()) for r in ['cold', 'bracketed', 'ties', 'before', 'after']}
}

manifest = {
    'event': 'EVT-159_PREPARATION_CUP4',
    'timestamp': '2026-11-14T20:06:00-03:00',
    'anchor_official': stats_t30,
    'candidate_principal_moderated': stats_mod,
    'candidate_alternative_k35': stats_k35,
    'candidate_alternative_k50': stats_k50,
    'recommendations': {
        'cupo_4_best': 'CANDIDATE_T30_PLUS_WARM_CLEAN_MODERATED.csv',
        'cupo_4_conservative': 'CANDIDATE_T30_PLUS_WARM_SURGICAL_K35.csv',
        'cupo_5_contingency_if_delta4_positive': 'Expand to complementary warm or Claude keepT30_top20',
        'cupo_5_contingency_if_delta4_nonpositive': 'Revert to Top30 (0.94823) and seal final submissions'
    }
}

with open('results/cup4_candidates_manifest.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

print("\n--- STATS SUMMARY ---")
for s in [stats_t30, stats_mod, stats_k35, stats_k50]:
    print(f"{s['name']}:")
    print(f"  Flips vs Top30: {s['flips_vs_top30']} (n->u: {s['n2u']}, u->n: {s['u2n']})")
    print(f"  Flips vs S4   : {s['flips_vs_s4']}")
    print(f"  LF SHA-256    : {s['lf_sha256']}")
    print(f"  Gate F Hash   : {s['gate_f_hash']}")
    print(f"  Gate G Status : {s['gate_g_status']} (used_rate: {s['used_rate']:.4f})")
    print(f"  Preserves T30 : {s['preserves_top30']}")
    print(f"  Reverts 10-10 : {s['reverts_1010']}")
    print(f"  Regimes 70k   : {s['regimes_full70k']}")
    print(f"  Regimes dev   : {s['regimes_dev35k']}")
