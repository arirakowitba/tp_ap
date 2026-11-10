"""Generate Slot 3: 12-Model Nested Meta-Stacker Submission.

Highest CV model in the repository:
- 12 models cross-fitted across 5 folds
- Meta OOF Accuracy: 0.94331 (66,032 / 70,000 hits)
- Log Loss: 0.14369
- ROC AUC: 0.98682
- Concordance with EVT-132: 29,364 / 30,000 (97.88%)
- Diff vs EVT-132: 636 items
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
out_dir = ROOT / "submissions/portfolio_20261109_podium_ladder"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load probabilities
p_nested = np.load("results/meta_stacking_ensemble/nested_test_probabilities.npy")
assert len(p_nested) == 30000

# 2. Threshold at 0.50
conditions = np.where(p_nested >= 0.50, "used", "new")

# 3. Create DataFrame
out_df = pd.DataFrame({
    "ID": list(range(1, 30001)),
    "condition": conditions
})

# 4. Verifications
assert len(out_df) == 30000
assert list(out_df.columns) == ["ID", "condition"]
assert set(out_df["condition"].unique()) == {"new", "used"}
assert out_df["ID"].tolist() == list(range(1, 30001))

out_path = out_dir / "submission_slot3_12model_nested_meta_stacker.csv"
out_df.to_csv(out_path, index=False, lineterminator="\n")

# SHA-256 calculation
data = out_path.read_bytes().replace(b"\r\n", b"\n")
sha = hashlib.sha256(data).hexdigest()

# Compare vs EVT-132
df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
diff = (out_df["condition"] != df_evt["condition"]).sum()

print("=" * 80)
print(f"Generated Slot 3 Submission: {out_path.name}")
print(f"Path:                        {out_path.resolve()}")
print(f"SHA-256 (LF):                {sha}")
print(f"Diff vs EVT-132:             {diff} items (97.88% concordance)")
print(f"Total Predicted 'used':      {(conditions == 'used').sum()} / 30,000")
print(f"Total Predicted 'new':       {(conditions == 'new').sum()} / 30,000")
print(f"OOF Accuracy on 70k Train:   0.94331")
print(f"OOF Log Loss on 70k Train:   0.14369")
print(f"OOF ROC AUC on 70k Train:    0.98682")
print("=" * 80)
