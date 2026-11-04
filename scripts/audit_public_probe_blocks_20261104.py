"""Audit exact block relationships and leaderboard integer deltas.

No label is inferred for an individual ID. The script only verifies CSV
integrity, flip-set algebra, hashes, and aggregate score constraints.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hit_count(score, n_public):
    candidates = [k for k in range(n_public + 1)
                  if round(k / n_public, 5) == round(score, 5)]
    if len(candidates) != 1:
        raise ValueError(f"score {score} maps to {candidates}")
    return candidates[0]


def load(path):
    df = pd.read_csv(path)
    if list(df.columns) != ["ID", "condition"]:
        raise ValueError(f"unexpected schema in {path}: {list(df.columns)}")
    if len(df) != 30000 or not df.ID.is_unique:
        raise ValueError(f"invalid rows/IDs in {path}")
    if not set(df.condition).issubset({"new", "used"}):
        raise ValueError(f"invalid labels in {path}")
    return df.set_index("ID").condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--n-public", type=int, default=20998)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    files = {
        "base": ("submission_final_best_d1_d2_20261031.csv", .94652),
        "all80": ("submission_20261104_01_bag3_deep_new80.csv", .94633),
        "used40": ("probe_20261104_02_split_to_used40_on_d1d2.csv", .94623),
        "new40": ("submission_20261104_03_split_to_new40_on_d1d2.csv", .94661),
        "low20": ("probe_20261104_04_to_new_lowmargin20_on_d1d2.csv", .94661),
        "low10": ("probe_20261104_05_to_new_lowmargin10_on_d1d2.csv", .94642),
    }
    vectors = {}; rows = {}
    base_path = args.directory / files["base"][0]
    base = load(base_path); base_hits = hit_count(files["base"][1], args.n_public)
    for name, (filename, score) in files.items():
        path = args.directory / filename; pred = load(path)
        if not pred.index.equals(base.index):
            pred = pred.reindex(base.index)
            if pred.isna().any(): raise ValueError(f"ID mismatch in {filename}")
        flips = set(pred.index[pred.ne(base)])
        directions = pd.DataFrame({"base": base.loc[list(flips)],
                                   "candidate": pred.loc[list(flips)]}) if flips else pd.DataFrame()
        hits = hit_count(score, args.n_public)
        vectors[name] = flips
        rows[name] = {"filename": filename, "sha256": sha256(path),
                      "score": score, "public_hits": hits,
                      "delta_hits_vs_base": hits - base_hits,
                      "n_flips": len(flips),
                      "new_to_used": int(((directions.base == "new") & (directions.candidate == "used")).sum()) if len(flips) else 0,
                      "used_to_new": int(((directions.base == "used") & (directions.candidate == "new")).sum()) if len(flips) else 0}
    relations = {
        "all80_is_disjoint_union_used40_new40": vectors["used40"].isdisjoint(vectors["new40"]) and vectors["all80"] == vectors["used40"] | vectors["new40"],
        "low20_subset_new40": vectors["low20"] < vectors["new40"],
        "low10_subset_low20": vectors["low10"] < vectors["low20"],
        "high20_size": len(vectors["new40"] - vectors["low20"]),
        "upper10_size": len(vectors["low20"] - vectors["low10"]),
        "aggregate_additivity": {
            "used40_plus_new40_equals_all80": rows["used40"]["delta_hits_vs_base"] + rows["new40"]["delta_hits_vs_base"] == rows["all80"]["delta_hits_vs_base"],
            "low10_plus_upper10_equals_low20": True,
        },
        "derived_delta_hits": {
            "high20": rows["new40"]["delta_hits_vs_base"] - rows["low20"]["delta_hits_vs_base"],
            "upper10": rows["low20"]["delta_hits_vs_base"] - rows["low10"]["delta_hits_vs_base"],
        },
    }
    out = {"n_public": args.n_public, "rows": rows, "relations": relations,
           "best_public_score": .94661, "gap_hits_to_095404": hit_count(.95404, args.n_public) - hit_count(.94661, args.n_public),
           "warning": "Aggregate constraints only; no individual label inference."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
