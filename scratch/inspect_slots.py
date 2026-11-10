import sys
sys.path.insert(0, '.')
import glob, os
import pandas as pd
from scratch.analyze_all_vectors import anchor

for slot in range(1, 6):
    matches = glob.glob(f"submissions/portfolio_20261110_podium_attack/submission_slot{slot}_*.csv")
    if not matches:
        continue
    fpath = matches[0]
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath)
    diffs = []
    for idx, row in df.iterrows():
        iid = int(row['ID'])
        pred = row['condition']
        anch = anchor['vec'][iid]
        if pred != anch:
            diffs.append((iid, anch, pred))
    print(f"=== Slot {slot}: {fname} ({len(diffs)} flips vs EVT-143) ===")
    for d in diffs:
        print(f"   ID {d[0]:5d}: {d[1]} -> {d[2]}")
