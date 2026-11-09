import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# Let's inspect the 35 items and build the linear system
from analyze_todays_5_subs import subs, df_anchor, all_diff_ids, sub_diffs

all_ids = sorted(list(all_diff_ids), key=lambda x: int(x))
id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}
n_items = len(all_ids)
print(f"Total variables: {n_items}")

A = np.zeros((len(subs), n_items), dtype=float)
b = np.zeros(len(subs), dtype=float)

# In each sub, if item was flipped from anchor:
# If anchor was 'new' and sub is 'used': flip = +1 (we tested flipping to used)
# If anchor was 'used' and sub is 'new': flip = -1 (we tested flipping to new)
for row_idx, (name, path, score, hits) in enumerate(subs):
    delta = hits - 19896
    b[row_idx] = delta
    for item_id, anc, cur in sub_diffs[name]:
        col_idx = id_to_idx[item_id]
        if anc == 'new' and cur == 'used':
            A[row_idx, col_idx] = 1.0  # +1 means test effect of new->used
        elif anc == 'used' and cur == 'new':
            A[row_idx, col_idx] = -1.0 # -1 means test effect of used->new

print("Matrix A shape:", A.shape)
print("Vector b:", b)

# Let's print the items and in which submissions they appear
df_appear = pd.DataFrame(A, index=[s[0] for s in subs], columns=all_ids)
print("\nAppearances in submissions:")
print(df_appear.T)
