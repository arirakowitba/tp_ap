import pandas as pd
import numpy as np
import hashlib

# Load files
s4 = pd.read_csv('submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv')
t30 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv')
orig_1010 = pd.read_csv('submissions/submission_optimized_ensemble.csv')

reg_full70k = np.load('scratch/reg_full70k.npy')
reg_dev = np.load('scratch/reg_dev.npy')

cand_files = {
    'stack_top50_sinrev_cupo3': 'submissions/claude_20261114/claude_T30_stack_top50_sinrev_cupo3.csv',
    'stack_top50_cupo3': 'submissions/claude_20261114/claude_T30_stack_top50_cupo3.csv',
    'stack_top20': 'submissions/claude_20261114/claude_T30_stack_top20.csv',
    'stack_top100': 'submissions/claude_20261114/claude_T30_stack_top100.csv',
    'post99_cal_top50': 'submissions/claude_20261114/claude_T30_post99_cal_top50.csv',
}

regimes_list = ['cold', 'bracketed', 'ties', 'before', 'after']

print("=== AUDIT OF CLAUDE LATEST CANDIDATES (CL-0030 / CL-0031) ===")

t30_flips_mask = (t30['condition'] != s4['condition']).values

for name, path in cand_files.items():
    df = pd.read_csv(path)
    
    # Hash check
    with open(path, 'rb') as f:
        raw = f.read().replace(b'\r\n', b'\n')
    lf_sha = hashlib.sha256(raw).hexdigest()
    
    df_sorted = df.sort_values('ID')
    payload = '\n'.join(f'{int(i)}:{str(t)}' for i, t in zip(df_sorted['ID'], df_sorted['condition'])) + '\n'
    gate_f = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    
    # Flips
    flips_t30 = (df['condition'] != t30['condition']).values
    flips_s4 = (df['condition'] != s4['condition']).values
    
    # Preservation of top30 30 flips
    preserves_t30 = (df.loc[t30_flips_mask, 'condition'] == t30.loc[t30_flips_mask, 'condition']).all()
    
    # Check reversions to 10-10 original (where S4 != orig_1010, does df revert to orig_1010?)
    s4_corrected = (s4['condition'] != orig_1010['condition']).values
    # In rows where df != t30, how many match orig_1010 where S4 was different?
    reversions_1010 = (flips_t30 & s4_corrected & (df['condition'] == orig_1010['condition'])).sum()
    
    # Directions
    n2u = ((t30.loc[flips_t30, 'condition'] == 'new') & (df.loc[flips_t30, 'condition'] == 'used')).sum()
    u2n = ((t30.loc[flips_t30, 'condition'] == 'used') & (df.loc[flips_t30, 'condition'] == 'new')).sum()
    
    print(f"\n--- {name} ---")
    print(f"  Path              : {path}")
    print(f"  Rows              : {len(df)}")
    print(f"  Flips vs Top30    : {flips_t30.sum()} (new->used: {n2u}, used->new: {u2n})")
    print(f"  Flips vs S4       : {flips_s4.sum()}")
    print(f"  Preserves Top30   : {preserves_t30}")
    print(f"  Reversions to 1010: {reversions_1010}")
    print(f"  Raw LF SHA-256    : {lf_sha}")
    print(f"  Gate F Hash       : {gate_f}")
    
    # Regimes under full 70k
    reg_counts = {r: int((flips_t30 & (reg_full70k == r)).sum()) for r in regimes_list}
    print(f"  Regimes (Full 70k): {reg_counts}")
