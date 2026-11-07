import json, re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('.')

# 1. Load Slot 4 anchor (0.94752 = 19,896 hits)
p_s4 = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_04_podium_breakthrough_top85.csv'
s4_df = pd.read_csv(p_s4).set_index('ID')['condition']

# 2. Identify all items flipped in Slot 5 vs Slot 4 (the 19 items that dropped -3 hits)
p_s5 = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_05_podium_climax_top105.csv'
s5_df = pd.read_csv(p_s5).set_index('ID')['condition']
slot5_dropped_ids = set(s5_df[s5_df != s4_df].index)
print(f"Slot 5 dropped IDs (proven toxic flips): {len(slot5_dropped_ids)}")

# 3. Identify items flipped in Slot 3 that were NOT in Slot 4 (the items that dropped -6 hits in Slot 3)
p_s3 = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_03_podium_attack_top65.csv'
s3_df = pd.read_csv(p_s3).set_index('ID')['condition']
slot3_toxic_ids = set(s3_df[(s3_df == 'used') & (s4_df == 'new')].index)
print(f"Slot 3 toxic IDs (proven toxic flips): {len(slot3_toxic_ids)}")

# Combined toxic / negative probe blacklist
PROVEN_TOXIC_IDS = slot5_dropped_ids.union(slot3_toxic_ids)
print(f"Total proven toxic IDs to blackball: {len(PROVEN_TOXIC_IDS)}")

# 4. Load Bayesian item effects & Transductive probabilities
p_bayes = ROOT / 'results' / 'bayesian_solver_103vec' / 'bayesian_transductive_item_effects_103vec.csv'
p_probs = ROOT / 'results' / 'transductive_deep_models_103vec' / 'transductive_test_probabilities_103vec.csv'

df_bayes = pd.read_csv(p_bayes).set_index('item_id')
df_probs = pd.read_csv(p_probs).set_index('ID')

df = df_bayes.join(df_probs)

# Strict semantic filter
BAD_PATTERNS = [
    r'\b0\s*km\b', r'\bcero\s*km\b', r'\bnuevo\b', r'\bnueva\b', r'\bnuevos\b', r'\bnuevas\b',
    r'\bsellad[oas]*\b', r'\bcerrad[oas]*\b', r'\bblister\b', r'\bestrenar\b',
    r'\ben caja\b', r'\bde fabrica\b', r'\bsin uso\b', r'\bposter\b', r'\bconcesionari[oa]\b',
    r'\bmoto\b', r'\bmotosport\b', r'\bplanta\b', r'\bsuculenta\b', r'\baeonium\b',
    r'\blote de\b', r'\bmayorista\b', r'\bdistribuidor\b', r'\bfabrica\b'
]

def text_toxic(t):
    s = str(t).lower()
    for p in BAD_PATTERNS:
        if re.search(p, s):
            return True
    return False

df['is_text_toxic'] = df['title'].apply(text_toxic)

# Filter for pure candidates
clean_cands = df[
    (df['anchor_condition_s4'] == 'new') &
    (df['is_safe']) &
    (~df.index.isin(PROVEN_TOXIC_IDS)) &
    (~df['is_text_toxic']) &
    (df['price'] <= 15000) & # Filter out luxury / expensive items ($>15k)
    (df['expected_gain_vs_s4'] > 0.15) &
    (df['p_blend_103vec'] > 0.72) &
    (df['times_tested'] >= 3)
].copy()

# Sort by test count desc, then expected gain desc
clean_cands = clean_cands.sort_values(by=['times_tested', 'expected_gain_vs_s4', 'p_blend_103vec'], ascending=[False, False, False])

print(f"\nTotal Ultra-Clean Candidates Available: {len(clean_cands)}")
print("\n" + "=" * 110)
print(f"{'#':2s} | {'ID':5s} | {'Tests':5s} | {'Gain':6s} | {'P_blend':7s} | {'Price':7s} | {'Title'}")
print("=" * 110)
for i, (idx, r) in enumerate(clean_cands.iterrows(), 1):
    print(f"{i:2d} | {idx:5d} | {r['times_tested']:5.0f} | {r['expected_gain_vs_s4']:+6.3f} | {r['p_blend_103vec']:7.3f} | ${r['price']:6.0f} | {r['title'][:60]}")
