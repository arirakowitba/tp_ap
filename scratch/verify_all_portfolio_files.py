import json, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(".")
p_manifest = ROOT / "submissions" / "portfolio_20261107_target_09490" / "portfolio_manifest.json"
with open(p_manifest, encoding="utf-8") as f:
    manifest = json.load(f)

p_anc = ROOT / "submissions" / "submission_20261106_05_climax_consumer_unanimous_top15.csv"
df_anc = pd.read_csv(p_anc).set_index("ID")["condition"]

for item in manifest:
    path = Path(item["path"])
    df = pd.read_csv(path)
    assert len(df) == 30000, f"Bad len {len(df)}"
    assert list(df.columns) == ["ID", "condition"]
    assert set(df["condition"].unique()).issubset({"new", "used"})
    
    diff = df.set_index("ID")["condition"] != df_anc
    n_flips = int(diff.sum())
    exp_flips = int(item["flips_vs_anchor"])
    assert n_flips == exp_flips, f"Flip count mismatch: got {n_flips}, expected {exp_flips}"
    
    lf_h = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert lf_h == item["sha256_lf"], f"Hash mismatch: got {lf_h}, expected {item['sha256_lf']}"
    
    # Check that all flips are strictly new -> used
    sub_diff = df.set_index("ID")["condition"][diff]
    anc_diff = df_anc[diff]
    assert (anc_diff == "new").all(), "Found flip not from new!"
    assert (sub_diff == "used").all(), "Found flip not to used!"
    
    print(f"Slot {item['slot']}: {item['filename']} PASSED ALL AUDITS! Flips: {n_flips} (all new->used), SHA: {lf_h[:16]}...")
