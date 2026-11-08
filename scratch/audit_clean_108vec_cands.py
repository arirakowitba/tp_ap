import pandas as pd, json, re
from pathlib import Path

ROOT = Path('.')
p_effects = ROOT / 'results' / 'bayesian_solver_108vec' / 'bayesian_transductive_item_effects_108vec.csv'
df = pd.read_csv(p_effects)

p_s4 = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_04_podium_breakthrough_top85.csv'
s4 = pd.read_csv(p_s4).set_index('ID')['condition']

EXCLUDE_CATS = {
    "MLA1459", "MLA1466", "MLA1467", "MLA1468", "MLA1470", "MLA1471", 
    "MLA1472", "MLA1473", "MLA1474", "MLA1475", "MLA1495", # Real Estate
    "MLA50279", "MLA50280", "MLA50281", # Vacation rentals
    "MLA1743", "MLA1744", "MLA2547", "MLA1745", "MLA1746", # Vehicles
    "MLA5632", "MLA1071", "MLA1072", # Pets
    "MLA1540" # Services
}

BAD_PATTERNS = [
    r'\b0\s*km\b', r'\bcero\s*km\b', r'\bnuevo\b', r'\bnueva\b', r'\bnuevos\b', r'\bnuevas\b',
    r'\bsellad[oas]*\b', r'\bcerrad[oas]*\b', r'\bblister\b', r'\bestrenar\b',
    r'\ben caja\b', r'\bde fabrica\b', r'\bsin uso\b', r'\bposter\b', r'\bconcesionari[oa]\b',
    r'\bmoto\b', r'\bmotosport\b', r'\bplanta\b', r'\bsuculenta\b', r'\baeonium\b',
    r'\blote de\b', r'\bmayorista\b', r'\bdistribuidor\b', r'\bfabrica\b'
]

def is_clean(r):
    t = str(r['title']).lower()
    cat = str(r['category_id'])
    p = float(r['price'] or 0)
    
    if cat in EXCLUDE_CATS:
        return False
    if p > 15000:
        return False
    if r['flipped_condition'] == 'used':
        if any(re.search(bp, t) for bp in BAD_PATTERNS):
            return False
    return True

df['clean'] = df.apply(is_clean, axis=1)
clean_pos = df[(df['clean']) & (df['expected_gain_vs_s4'] > 0.25)].sort_values(by=['times_tested', 'expected_gain_vs_s4'], ascending=[False, False])

print(f"Total Clean Positive Candidates vs Slot 4 anchor: {len(clean_pos)}")
print(f"Tested >= 5: {len(clean_pos[clean_pos['times_tested'] >= 5])}")
print(f"Tested 2 to 4: {len(clean_pos[clean_pos['times_tested'].between(2, 4)])}")

print("\n" + "=" * 120)
print(f"{'#':2s} | {'ID':5s} | {'Dir':11s} | {'Tests':5s} | {'Gain':6s} | {'P_used':6s} | {'Price':5s} | {'Title'}")
print("=" * 120)
for i, (idx, r) in enumerate(clean_pos.iloc[:40].iterrows(), 1):
    d = f"{r['anchor_condition_s4']}->{r['flipped_condition']}"
    print(f"{i:2d} | {int(r['item_id']):5d} | {d:11s} | {int(r['times_tested']):5d} | {r['expected_gain_vs_s4']:+6.3f} | {r['model_p_used']:6.3f} | ${r['price']:5.0f} | {r['title'][:55]}")
