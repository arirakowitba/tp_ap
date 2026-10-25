"""Replay every numerated submission with an exact available CSV (04-09, 11-41)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASES = {}
for number in (7, 15, 33):
    bits = (ROOT / f"base{number:02d}_labels_1_to_30000.txt").read_text().strip()
    assert len(bits) == 30000 and set(bits) <= {"0", "1"}
    BASES[number] = ["used" if b == "1" else "new" for b in bits]

def replay(path, base, base_col, target_col):
    changes = {}
    with (ROOT / path).open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            n, i = int(r["submission"]), int(r["ID"])
            b = int(r.get("base", base))
            assert b in BASES and 1 <= i <= 30000
            assert BASES[b][i - 1] == r[base_col]
            assert r[target_col] != r[base_col] and r[target_col] in {"new", "used"}
            assert i not in changes.setdefault((n, b), {})
            changes[(n, b)][i] = r[target_col]
    return changes

sets = [
    replay("flips_04_09.tsv", 7, "from_07", "to"),
    replay("flips_11_14_27_35.tsv", 15, "from", "to"),
    replay("flips_16_26.tsv", 15, "from_15", "to"),
    replay("flips_36_41.tsv", 33, "from_33", "to"),
]
all_changes = {(7, 7): {}, (15, 15): {}, (33, 33): {}}
for cohort in sets:
    assert not (all_changes.keys() & cohort.keys())
    all_changes.update(cohort)
assert len(all_changes) == 37
out = ROOT / "replayed"
out.mkdir(exist_ok=True)
for (n, b), changed in sorted(all_changes.items()):
    with (out / f"submission_{n:02d}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "condition"])
        writer.writerows((i, changed.get(i, label)) for i, label in enumerate(BASES[b], 1))
    print(n, b, len(changed))
