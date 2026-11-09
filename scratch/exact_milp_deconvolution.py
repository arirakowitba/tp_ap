import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from solve_todays_sub_matrix import A, b, all_ids, id_to_idx

print(f"Solving integer system with Scipy MILP for {len(all_ids)} variables...")
# Variables x_i in {-1, 0, 1}
# Let's map x_i = y_i - 1, where y_i in {0, 1, 2}
# A @ (y - 1) = b  =>  A @ y = b + A @ 1

A_y = A
b_y = b + A.sum(axis=1)

constraints = LinearConstraint(A_y, b_y, b_y)
integrality = np.ones(len(all_ids)) # all integer
bounds = Bounds(0, 2) # y in [0, 2]

# Let's solve min and max for each variable to get exact bounds!
min_bounds = {}
max_bounds = {}

for i, id_ in enumerate(all_ids):
    c = np.zeros(len(all_ids))
    # minimize y_i
    c[i] = 1.0
    res_min = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    if res_min.success:
        min_bounds[id_] = int(round(res_min.x[i])) - 1
    else:
        min_bounds[id_] = None
        
    # maximize y_i
    c[i] = -1.0
    res_max = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    if res_max.success:
        max_bounds[id_] = int(round(res_max.x[i])) - 1
    else:
        max_bounds[id_] = None

df_bounds = pd.DataFrame({
    'ID': all_ids,
    'min_effect': [min_bounds[i] for i in all_ids],
    'max_effect': [max_bounds[i] for i in all_ids]
})

df_meta = pd.read_csv('scratch/todays_35_items_metadata.csv')
df_meta['ID'] = df_meta['ID'].astype(str)
merged = df_bounds.merge(df_meta, on='ID', how='left')
merged['width'] = merged['max_effect'] - merged['min_effect']
print("\nExact integer bounds for each item's public test effect (-1 = miss, 0 = private, +1 = hit):")
for _, r in merged.sort_values(['width', 'min_effect'], ascending=[True, False]).iterrows():
    print(f"ID {r['ID']:5s} | bounds: [{r['min_effect']:2d}, {r['max_effect']:2d}] (width={r['width']}) | {str(r['title'])[:45]:45s} | ${r['price']:7.2f}")
