"""Reconcile the historical ledger with the Nov 6 catalog through Nov 2."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def key(value):
    return str(value or "").strip().casefold()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ledger", type=Path)
    p.add_argument("catalog", type=Path)
    p.add_argument("--cutoff", default="2026-11-02")
    args = p.parse_args()
    ledger = json.loads(args.ledger.read_text())
    catalog = [x for x in json.loads(args.catalog.read_text()) if x["date"] <= args.cutoff]
    scored = [x for x in catalog if x["category"].startswith("scored")]
    entries = ledger["entries"]
    names = Counter(key(row["key"]) for row in entries)
    rows = {}
    for x in scored:
        for name in {key(x["legacy_identifier"]), key(x["display_name"])} - {""}:
            rows.setdefault(name, set()).add(x["public_score"])
    extra = [x for x in scored if not ({key(x["legacy_identifier"]), key(x["display_name"])} & names.keys())]
    counts = Counter(x["category"] for x in catalog)
    exact = sum(x["score"] in rows.get(key(x["key"]), ()) for x in entries)
    report = {
        "cutoff_inclusive": args.cutoff,
        "ledger_entry_count": len(entries), "ledger_distinct_keys": len(names),
        "ledger_metadata_total_entries": ledger["metadata"].get("total_entries"),
        "ledger_rows_matching_catalog_key_and_score": exact,
        "duplicate_ledger_keys": sorted(k for k, n in names.items() if n > 1),
        "catalog_counts": counts, "scored_catalog_rows_outside_ledger": len(extra),
        "outside_aliases": sum(bool(x["is_alias"]) for x in extra),
        "outside_non_aliases_without_vectors": sum(
            not x["is_alias"] and x["category"] == "scored_no_vector" for x in extra),
        "catalog_scored_or_error_rows": len(scored) + counts["error_no_score"],
        "raw_kaggle_106_reconciled": False,
        "limitation": "Catalog aliases and dated rows cannot establish distinct Kaggle submission IDs.",
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
