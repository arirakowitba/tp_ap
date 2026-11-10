"""Search all evaluation summaries in results/ to find the best CV models."""
import json
from pathlib import Path

ROOT = Path(".")
results_dir = ROOT / "results"

summaries = []
for p in results_dir.glob("**/evaluation_summary.json"):
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["file"] = str(p)
            summaries.append(data)
    except Exception:
        pass

for p in results_dir.glob("**/*report*.json"):
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and any("acc" in k.lower() or "score" in k.lower() for k in data.keys()):
                data["file"] = str(p)
                summaries.append(data)
    except Exception:
        pass

print(f"Total summary files found: {len(summaries)}")

# Print items with accuracy
for s in summaries:
    fn = s.get("file", "")
    acc = s.get("accuracy") or s.get("acc_total") or s.get("cv_accuracy") or s.get("mean_cv_accuracy")
    if acc:
        print(f"File: {fn[:60]:60s} | Acc: {acc:.5f}")
