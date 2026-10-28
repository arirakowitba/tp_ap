"""Rebuild exact 30k predictions from baseline 33 and an event ledger."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
bits = (ROOT / "base33_labels_1_to_30000.txt").read_text().strip()
assert len(bits) == 30000 and set(bits) <= {"0", "1"}
base = {i: ("used" if bit == "1" else "new") for i, bit in enumerate(bits, 1)}
events = {}
with (ROOT / "flips_36_41.tsv").open(newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        sub, idx = row["submission"], int(row["ID"])
        assert base[idx] == row["from_33"]
        assert row["to"] in {"used", "new"} and row["to"] != base[idx]
        assert idx not in events.setdefault(sub, {})
        events[sub][idx] = row["to"]

assert {s: len(e) for s, e in events.items()} == {
    "36": 18, "37": 146, "38": 201, "39": 1228, "40": 4, "41": 6
}
assert events["40"].items() <= events["36"].items()
assert events["37"].items() <= events["38"].items()
assert events["38"].items() <= events["39"].items()
out = ROOT / "replayed"
out.mkdir(exist_ok=True)
for sub, changes in events.items():
    with (out / f"submission_{sub}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "condition"])
        writer.writerows((i, changes.get(i, base[i])) for i in base)
    print(sub, len(changes))
