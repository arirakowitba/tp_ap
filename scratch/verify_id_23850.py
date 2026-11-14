import pandas as pd

s4 = pd.read_csv('submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv').sort_values('ID').reset_index(drop=True)
b10 = pd.read_csv('results/round141_claude_baseline/full70k/submission_b10_full.csv').sort_values('ID').reset_index(drop=True)
u37 = pd.read_csv('results/CANDIDATE_T30_PLUS_U37.csv').sort_values('ID').reset_index(drop=True)
# check if u36 exists or checkout from claude
import os
import subprocess
import io

if not os.path.exists('submissions/claude_20261114/claude_T30_mas_U36_estricto.csv'):
    c36_bytes = subprocess.check_output(['git', 'show', 'origin/claude/posterior-y-modelos-20261027:submissions/claude_20261114/claude_T30_mas_U36_estricto.csv'])
    u36 = pd.read_csv(io.BytesIO(c36_bytes)).sort_values('ID').reset_index(drop=True)
else:
    u36 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U36_estricto.csv').sort_values('ID').reset_index(drop=True)

t30 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv').sort_values('ID').reset_index(drop=True)

# check 10-10 original
orig = pd.read_csv('submissions/submission_optimized_ensemble.csv').sort_values('ID').reset_index(drop=True)

diff_37 = set(u37[u37['condition'] != t30['condition']]['ID'])
diff_36 = set(u36[u36['condition'] != t30['condition']]['ID'])

print(f"Count diff_37: {len(diff_37)}")
print(f"Count diff_36: {len(diff_36)}")
print(f"Intersection: {len(diff_37.intersection(diff_36))}")
extra_id = list(diff_37 - diff_36)
print(f"Extra ID in AG U37: {extra_id}")

for i in extra_id:
    print(f"ID {i}:")
    print(f"  orig 10-10 : {orig.loc[orig['ID'] == i, 'condition'].values[0]}")
    print(f"  S4         : {s4.loc[s4['ID'] == i, 'condition'].values[0]}")
    print(f"  B10        : {b10.loc[b10['ID'] == i, 'condition'].values[0]}")
    print(f"  Top30      : {t30.loc[t30['ID'] == i, 'condition'].values[0]}")
    print(f"  U36        : {u36.loc[u36['ID'] == i, 'condition'].values[0]}")
    print(f"  U37        : {u37.loc[u37['ID'] == i, 'condition'].values[0]}")
