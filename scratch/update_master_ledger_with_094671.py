import json
from pathlib import Path

ROOT = Path('.')
p_master = ROOT / "results" / "forensic_ledger_20261104" / "submission_ledger_master_85vec.json"
with open(p_master, encoding="utf-8") as f:
    master = json.load(f)

new_entry = {
    "key": "20261105_01_upper10",
    "name": "submission_20261105_01_upper10_on_d1d2.csv",
    "sha256_lf": "ac35697bef44bcd679dfbae59011d72fcf558af5acf1661f5818b8629129d13d",
    "score": 0.94671,
    "public_hits": 19879,
    "flipped_ids_vs_d1d2": [4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453],
    "delta_vs_d1d2": 4,
    "delta_vs_48": 8,
    "n_flips_vs_d1d2": 10,
    "n_flips_vs_48": 33,
    "notes": "Confirmed by Ariel on Kaggle 2026-11-05. Exact +4 public hits over D1+D2 (19875 -> 19879).",
    "status": "verified_recoverable"
}

# Check if already present
existing_keys = [e.get("key") for e in master["entries"]]
if new_entry["key"] not in existing_keys:
    master["entries"].append(new_entry)
    master["n_entries_total"] = len(master["entries"])
    master["n_recoverable_vectors"] = len([e for e in master["entries"] if e.get("status") == "verified_recoverable"])
    master["best_public_score"] = 0.94671
    master["best_public_hits"] = 19879

    with open(p_master, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    print("Master ledger updated successfully with 20261105_01_upper10 = 0.94671!")
else:
    print("Entry already present in master ledger.")
