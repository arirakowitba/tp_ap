import pandas as pd
import numpy as np
import hashlib

# Load files
s4 = pd.read_csv('submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv').sort_values('ID').reset_index(drop=True)
t30 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv').sort_values('ID').reset_index(drop=True)
t30_u58 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U58_blendfix_noprobado.csv').sort_values('ID').reset_index(drop=True)
t30_u58_f86 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U58_mas_F86_bfix_noprobado.csv').sort_values('ID').reset_index(drop=True)
b10 = pd.read_csv('results/round141_claude_baseline/full70k/submission_b10_full.csv').sort_values('ID').reset_index(drop=True)

# Build U37: exactly Top30 + the 37 concordant flips
f_u58 = t30_u58['condition'] != t30['condition']
is_concordant = (s4['condition'] == b10['condition'])
mask_u37 = f_u58 & is_concordant
mask_r21 = f_u58 & (~is_concordant)

print(f"Mask U37 count: {mask_u37.sum()}")
print(f"Mask R21 count: {mask_r21.sum()}")

df_u37 = t30.copy()
df_u37.loc[mask_u37, 'condition'] = t30_u58.loc[mask_u37, 'condition']

# Verify df_u37 properties
flips_vs_t30 = (df_u37['condition'] != t30['condition']).sum()
flips_vs_s4 = (df_u37['condition'] != s4['condition']).sum()
flips_t30_vs_s4 = (t30['condition'] != s4['condition']).sum()

# Check preservation of Top30's 30 flips
t30_flips_mask = (t30['condition'] != s4['condition'])
preserves_t30 = (df_u37.loc[t30_flips_mask, 'condition'] == t30.loc[t30_flips_mask, 'condition']).all()

print(f"df_u37 flips vs Top30: {flips_vs_t30}")
print(f"df_u37 flips vs S4: {flips_vs_s4}")
print(f"Preserves Top30 30 flips: {preserves_t30}")

# Save CANDIDATE_T30_PLUS_U37.csv
out_path = 'results/CANDIDATE_T30_PLUS_U37.csv'
df_u37.to_csv(out_path, index=False, lineterminator='\n')

with open(out_path, 'rb') as f:
    raw_bytes = f.read()
lf_sha = hashlib.sha256(raw_bytes).hexdigest()

# Canonical Gate F hash
norm_content = b'\n'.join([line.strip() for line in raw_bytes.splitlines() if line.strip()]) + b'\n'
gate_f = hashlib.sha256(norm_content).hexdigest()

print(f"\nCANDIDATE_T30_PLUS_U37.csv:")
print(f"  LF SHA-256: {lf_sha}")
print(f"  Gate F Hash: {gate_f}")
print(f"  Row count: {len(df_u37)}")
print(f"  Used rate: {(df_u37['condition'] == 'used').mean():.4f}")

# Check F86 properties
f_f86 = t30_u58_f86['condition'] != t30_u58['condition']
print(f"\nF86 flips vs T30_U58: {f_f86.sum()}")
print(f"In F86 rows, baseline T30_U58 == B10: {(t30_u58.loc[f_f86, 'condition'] == b10.loc[f_f86, 'condition']).sum()} / {f_f86.sum()}")
print(f"In F86 rows, baseline T30_U58 == S4: {(t30_u58.loc[f_f86, 'condition'] == s4.loc[f_f86, 'condition']).sum()} / {f_f86.sum()}")
