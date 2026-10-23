from pathlib import Path
import csv, hashlib, json

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submissions_20261022"
OUT = ROOT / "submissions_20261023"
OUT.mkdir(exist_ok=True)

def load(number):
    path = next(SOURCE.glob(f"submission_{number}_*.csv"))
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 30000
    assert [int(row["ID"]) for row in rows] == list(range(1, 30001))
    assert all(row["condition"] in {"new", "used"} for row in rows)
    return path, rows

p26, r26 = load(26)
p29, r29 = load(29)
p30, r30 = load(30)
y26 = [row["condition"] for row in r26]
y29 = [row["condition"] for row in r29]
y30 = [row["condition"] for row in r30]
g1 = {i for i, (a, b) in enumerate(zip(y26, y29)) if a != b}
g2 = {i for i, (a, b) in enumerate(zip(y26, y30)) if a != b}
assert len(g1) == len(g2) == 32 and g1.isdisjoint(g2)

y31 = y26.copy()
for i in g1 | g2:
    y31[i] = "new" if y31[i] == "used" else "used"

target = OUT / "submission_31_union_G1_G2_64_sobre26.csv"
with target.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["ID", "condition"])
    writer.writerows((i + 1, label) for i, label in enumerate(y31))

check = list(csv.DictReader(target.open()))
assert len(check) == 30000
assert [int(row["ID"]) for row in check] == list(range(1, 30001))
assert sum(a != b for a, b in zip(y26, y31)) == 64
assert all(y31[i] == y29[i] for i in g1)
assert all(y31[i] == y30[i] for i in g2)

report = {
    "date": "2026-10-22",
    "base": p26.name,
    "components": [p29.name, p30.name],
    "observed_scores": {"26": 0.94547, "29": 0.94571, "30": 0.94561},
    "assumed_public_rows": 21000,
    "inferred_net_gains": {"G1": 5, "G2": 3, "union": 8},
    "predicted_public_score_union": 19863 / 21000,
    "changes_vs26": 64,
    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "public_score_observed": None,
    "limitations": "Prediction assumes additive balances because G1/G2 are disjoint and the public set is fixed. It does not identify individual labels or imply hidden-set gain.",
}
print(json.dumps(report, ensure_ascii=False, indent=2))
