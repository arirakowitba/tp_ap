import sys
sys.path.insert(0, '.')
import pandas as pd
from scratch.analyze_all_vectors import anchor

df_sub5 = pd.read_csv('submissions/portfolio_20261110_podium_attack/submission_slot5_climax_09500_breakthrough_top110.csv')
df_cert = pd.read_csv('results/certified_five_pillar_candidates.csv')

sub5_flips = []
for idx, r in df_sub5.iterrows():
    iid = int(r['ID'])
    if r['condition'] != anchor['vec'][iid]:
        sub5_flips.append((iid, anchor['vec'][iid], r['condition']))

sub5_ids = set(x[0] for x in sub5_flips)
cert_ids = set(df_cert['id'].values)

overlap = sub5_ids.intersection(cert_ids)
in_sub5_not_cert = sub5_ids - cert_ids
in_cert_not_sub5 = cert_ids - sub5_ids

print(f"Slot 5 flips total: {len(sub5_ids)}")
print(f"Certified candidates total: {len(cert_ids)}")
print(f"Overlap (Slot 5 flips that are certified): {len(overlap)} / {len(sub5_ids)} ({len(overlap)/len(sub5_ids)*100:.1f}%)")

if in_sub5_not_cert:
    print(f"\nItems in Slot 5 but quarantined by 5-Pillar filter ({len(in_sub5_not_cert)} items):")
    for iid in sorted(in_sub5_not_cert):
        title = df_sub5.loc[df_sub5['ID'] == iid, 'condition'].values[0]
        print(f"  ID {iid:5d}: {anchor['vec'][iid]} -> {title}")
