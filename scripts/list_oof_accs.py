"""Print all oof accuracies."""
import json
from pathlib import Path

for p in Path("results").glob("**/evaluation_summary.json"):
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            val = data.get("oof_accuracy") or data.get("accuracy") or data.get("test_accuracy")
            name = data.get("model") or p.parent.name
            print(f"{name:45s} | OOF Acc: {val} | Path: {p}")
    except Exception as e:
        pass
