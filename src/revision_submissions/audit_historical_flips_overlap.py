"""Audit 35 probe IDs against all previously tested submissions on Kaggle.

Checks whether any of the 35 IDs in submissions 66-69 were already flipped in:
- submission 57 / 58 (.94633 tied)
- submission 59 (.94576 dropped)
- submissions 60, 61, 62, 63 (fin_stack dropped)
- probes 150 nuevos (.94495) / 150 vistos (.94257)
- earlier submissions
"""
import glob
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_JSON = ROOT / "results" / "revision_submissions" / "round152_adaptive_probes_manifest.json"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"


def main():
    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    probe_items = []
    for p in manifest:
        for it in p["items"]:
            it["probe_file"] = p["file"]
            probe_items.append(it)

    with open(S48_JSON) as f:
        s48 = json.load(f)

    tested_flips = {}

    # Check submissions 57, 58, 59, 60, 61, 62, 63
    sub_patterns = [
        ("submissions/20261027/submission_57_*.csv", "sub_57", "0.94633 (tied)"),
        ("submissions/20261027/submission_58_*.csv", "sub_58", "0.94633 (tied)"),
        ("submissions/20261028/submission_59_*.csv", "sub_59", "0.94576 (dropped)"),
        ("submissions/20261028/submission_60_*.csv", "sub_60", "fin_stack 0.70-0.85 (dropped)"),
        ("submissions/20261028/submission_61_*.csv", "sub_61", "fin_stack 0.60-0.70 (dropped)"),
        ("submissions/20261028/submission_62_*.csv", "sub_62", "fin_stack 0.50-0.60 (dropped)"),
        ("submissions/20261028/submission_63_*.csv", "sub_63", "fin_stack modelo puro (dropped)"),
    ]

    for pat, label, score_res in sub_patterns:
        matches = glob.glob(str(ROOT / pat))
        if matches:
            df = pd.read_csv(matches[0])
            for _, r in df.iterrows():
                pid = int(r["ID"])
                cond = str(r["condition"])
                s48_c = str(s48[str(pid)])
                if cond != s48_c:
                    tested_flips[pid] = {
                        "sub": label,
                        "file": Path(matches[0]).name,
                        "direction": f"{s48_c}->{cond}",
                        "score_result": score_res,
                    }

    print(f"Total historical flipped IDs collected from post-48 tested sets: {len(tested_flips)}")

    overlaps = []
    for it in probe_items:
        pid = int(it["ID"])
        if pid in tested_flips:
            t = tested_flips[pid]
            overlaps.append({
                "ID": pid,
                "probe": it["probe_file"],
                "probe_direction": f"{it['s48']}->{it['target']}",
                "tested_in": t["sub"],
                "tested_file": t["file"],
                "tested_direction": t["direction"],
                "tested_result": t["score_result"],
                "title": it.get("title", ""),
                "price": it.get("price", 0),
            })

    print(f"\nTotal overlapping IDs found in our 35 probe items: {len(overlaps)}")
    for o in overlaps:
        print(f"  ID {o['ID']:5d} | In Probe: {o['probe']} ({o['probe_direction']}) | Tested In: {o['tested_in']} ({o['tested_direction']}) -> Result: {o['tested_result']}")

    # Save report
    out_file = ROOT / "results" / "revision_submissions" / "round154_probes_historical_overlap_audit.json"
    with open(out_file, "w") as f:
        json.dump(overlaps, f, indent=2)
    print(f"\nSaved overlap audit to {out_file}")


if __name__ == "__main__":
    main()
