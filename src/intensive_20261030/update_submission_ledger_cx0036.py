"""Update Submission Ledger with S1, S2, S3, C1 and Final Reconfirmation (CX-0036..CX-0042).

Maintains:
1. Exact sha256_lf and sha256_vector.
2. Verified flips_dict and flipped_ids vs S48 reference baseline.
3. Actual LB scores and deltas.
"""
import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "intensive_20261030"
LEDGER_PATH = OUT_DIR / "submission_ledger_04_current.json"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def sha256_lf(path):
    b = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()

def main():
    print("=== UPDATING SUBMISSION LEDGER (CX-0036..CX-0042) ===", flush=True)

    with open(S48_JSON, "r") as f:
        s48_raw = json.load(f)
    s48 = {int(k): str(v) for k, v in s48_raw.items()}
    assert len(s48) == 30000

    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    existing_keys = {e["key"] for e in ledger["entries"]}

    new_subs = [
        {
            "key": "probe_s1_seen_virgin25",
            "file": ROOT / "submissions" / "probe_s1_seen_virgin25.csv",
            "score": 0.94628,
            "delta_vs_48": round(0.94628 - 0.94633, 5),
            "description": "Probe S1: 25 seen virgin flips on D1+D2 (score: 0.94628, delta vs best: -0.00024)"
        },
        {
            "key": "probe_s2_new_virgin25",
            "file": ROOT / "submissions" / "probe_s2_new_virgin25.csv",
            "score": 0.94585,
            "delta_vs_48": round(0.94585 - 0.94633, 5),
            "description": "Probe S2: 25 new virgin flips on D1+D2 (score: 0.94585, delta vs best: -0.00067)"
        },
        {
            "key": "probe_s3_lexical_virgin25",
            "file": ROOT / "submissions" / "probe_s3_lexical_virgin25.csv",
            "score": 0.94628,
            "delta_vs_48": round(0.94628 - 0.94633, 5),
            "description": "Probe S3: 25 lexical virgin flips on D1+D2 (score: 0.94628, delta vs best: -0.00024)"
        },
        {
            "key": "candidate_c1_sub79_new_clean20",
            "file": ROOT / "submissions" / "20261030" / "candidate_sub79_new_sellers_clean_text.csv",
            "score": 0.94623,
            "delta_vs_48": round(0.94623 - 0.94633, 5),
            "description": "Probe C1: 20 new seller clean text flips on S48 (score: 0.94623, delta vs best: -0.00029)"
        },
        {
            "key": "submission_final_reconfirmation_d1_d2",
            "file": ROOT / "submissions" / "candidate_d1_d2_combined_on48.csv",
            "score": 0.94652,
            "delta_vs_48": round(0.94652 - 0.94633, 5),
            "description": "Reconfirmation submission of D1+D2 on S48 (exact duplicate score: 0.94652)"
        }
    ]

    added = 0
    for sub in new_subs:
        k = sub["key"]
        p = sub["file"]
        if k in existing_keys:
            print(f"Key {k} already in ledger, skipping duplicate key addition")
            continue
        if not p.exists():
            print(f"Warning: {p} not found, skipping")
            continue
        df = pd.read_csv(p)
        preds = dict(zip(df["ID"].astype(int), df["condition"].astype(str)))
        flips = {int(i): preds[i] for i in range(1, 30001) if preds[i] != s48[i]}
        raw_hash = sha256_file(p)
        lf_hash = sha256_lf(p)
        vec = "".join("1" if preds[i] == "used" else "0" for i in range(1, 30001))
        vec_hash = hashlib.sha256(vec.encode("ascii")).hexdigest()

        entry = {
            "key": k,
            "score": sub["score"],
            "delta_vs_48": sub["delta_vs_48"],
            "n_flips_vs_48": len(flips),
            "sha256_raw": raw_hash,
            "sha256_lf": lf_hash,
            "sha256_vector": vec_hash,
            "flipped_ids": sorted(list(flips.keys())),
            "flips_dict": {str(fid): tgt for fid, tgt in flips.items()},
            "description": sub["description"]
        }
        ledger["entries"].append(entry)
        added += 1
        print(f"Added {k}: flips_vs_48={len(flips)}, score={sub['score']}, sha256_lf={lf_hash[:12]}...")

    ledger["metadata"]["last_updated_utc"] = "2026-11-01T06:30:00Z"
    ledger["metadata"]["total_entries"] = len(ledger["entries"])

    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    print(f"Saved updated ledger to {LEDGER_PATH} (total entries: {len(ledger['entries'])}, added: {added})")

if __name__ == "__main__":
    main()
