from pathlib import Path
import pandas as pd, json, re

ROOT = Path('.')
p_bayes = ROOT / 'results' / 'bayesian_solver_103vec' / 'bayesian_transductive_item_effects_103vec.csv'
p_probs = ROOT / 'results' / 'transductive_deep_models_103vec' / 'transductive_test_probabilities_103vec.csv'

df_bayes = pd.read_csv(p_bayes).set_index('item_id')
df_probs = pd.read_csv(p_probs).set_index('ID')
df = df_bayes.join(df_probs)

p_s4 = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_04_podium_breakthrough_top85.csv'
s4_df = pd.read_csv(p_s4).set_index('ID')['condition']

p_s5 = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_05_podium_climax_top105.csv'
s5_df = pd.read_csv(p_s5).set_index('ID')['condition']
p_s3 = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_03_podium_attack_top65.csv'
s3_df = pd.read_csv(p_s3).set_index('ID')['condition']

PROVEN_TOXIC_IDS = set(s5_df[s5_df != s4_df].index).union(set(s3_df[(s3_df == 'used') & (s4_df == 'new')].index))

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

t2 = df[
    (df['anchor_condition_s4'] == 'new') &
    (df['is_safe']) &
    (~df.index.isin(PROVEN_TOXIC_IDS)) &
    (~df['is_text_toxic']) &
    (df['price'] <= 10000) &
    (df['expected_gain_vs_s4'] > 0.40) &
    (df['p_blend_103vec'] > 0.80) &
    (df['times_tested'] == 2)
].sort_values(by='expected_gain_vs_s4', ascending=False)

print(f"Candidates with times_tested == 2, Gain > 0.40, P_blend > 0.80: {len(t2)}")
for i, (idx, r) in enumerate(t2.iterrows(), 1):
    print(f"{i:2d} | ID={idx:5d} | Gain={r['expected_gain_vs_s4']:+6.3f} | Pb={r['p_blend_103vec']:7.3f} | ${r['price']:5.0f} | {r['title'][:60]}")
