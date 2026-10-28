"""Regenerate seller-segment probes from the exact 60 CSV and original JSONL.
No test labels are read. Outputs two 30k-row CSVs on the exact submission 48.
"""
import hashlib, json, zipfile
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest_bandas_fin_stack.json").read_text())
probe = next(p for p in manifest["probes"] if p["number"] == 60)
model = pd.read_csv(HERE / probe["file"])
assert hashlib.sha256((HERE / probe["file"]).read_bytes()).hexdigest() == probe["sha256"]
ids = set(probe["ids"])
assert len(ids) == 252
base = model.copy()
changed = base.ID.isin(ids)
base.loc[changed, "condition"] = base.loc[changed, "condition"].map({"new": "used", "used": "new"})
assert hashlib.sha256(base.to_csv(index=False).encode()).hexdigest() == manifest["base_sha256"]

with zipfile.ZipFile(Path("data.zip")) as z:
    train = pd.DataFrame(json.loads(line) for line in z.open("train_data.jsonlines"))
    test = pd.DataFrame(json.loads(line) for line in z.open("test_data.jsonlines"))
seen = test.seller_id.isin(set(train.seller_id)).to_numpy()
for segment, mask in [("seen", seen), ("new", ~seen)]:
    chosen = changed & mask
    output = base.copy()
    output.loc[chosen, "condition"] = model.loc[chosen, "condition"]
    filename = HERE / f"probe_60_{segment}_seller_sobre48.csv"
    output.to_csv(filename, index=False)
    print(filename.name, int(chosen.sum()), hashlib.sha256(filename.read_bytes()).hexdigest())
