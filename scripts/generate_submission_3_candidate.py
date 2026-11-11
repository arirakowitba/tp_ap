import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
evt143_path = ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv"
assert evt143_path.exists()
df_evt143 = pd.read_csv(evt143_path)

df_s3 = df_evt143.copy()
# 3 U2N items: IDs 5176, 8560, 23469
u2n_ids = [5176, 8560, 23469]
for item_id in u2n_ids:
    idx = item_id - 1
    assert df_s3.loc[idx, "ID"] == item_id
    assert df_s3.loc[idx, "condition"] == "used"
    df_s3.loc[idx, "condition"] = "new"

out_path = ROOT / "submissions/SUBMISSION_3_TODAY_RECOMMENDED.csv"
df_s3.to_csv(out_path, index=False)

def sha256_lf(p):
    data = p.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()

def hash_id_target_map(df):
    mapping_str = "\n".join(f"{row['ID']},{row['condition']}" for _, row in df.iterrows())
    return hashlib.sha256(mapping_str.encode("utf-8")).hexdigest()

print("File Created:", out_path)
print("SHA-256 (LF):", sha256_lf(out_path))
print("ID/Target Hash:", hash_id_target_map(df_s3))
print("Total Flips vs EVT-143:", int(np.sum(df_s3["condition"] != df_evt143["condition"])))
print("U2N Flips:", int(np.sum((df_s3["condition"] == "new") & (df_evt143["condition"] == "used"))))
print("N2U Flips:", int(np.sum((df_s3["condition"] == "used") & (df_evt143["condition"] == "new"))))
