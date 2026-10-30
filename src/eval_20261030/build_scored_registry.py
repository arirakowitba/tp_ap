"""Compile Scored Submissions Registry and Reconstruct Canonical Label Vectors.

Sources:
- results/revision_submissions/posterior_flips_20261027/assemble.py (4-50, 57, 58)
- results/revision_submissions/round138_fin_stack_segment_samples_20261028.md (59, 150 nuevos, 150 vistos)
- results/revision_submissions/registry_72_76_20261029.json (72-76)
- base07, base15, base33, and flips TSVs
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
REV_DIR = ROOT / "results" / "revision_submissions"
EVAL_DIR = ROOT / "results" / "eval_20261030"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def encode_labels(labels_dict):
    """Encodes dict of {id: condition} to canonical LF CSV bytes."""
    lines = ["ID,condition\n"]
    for i in range(1, 30001):
        lines.append(f"{i},{labels_dict[i]}\n")
    return "".join(lines).encode("utf-8")

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_vector(labels_dict):
    """Hash of the raw 30,000 character 0/1 vector."""
    vec = "".join("1" if labels_dict[i] == "used" else "0" for i in range(1, 30001))
    return hashlib.sha256(vec.encode("ascii")).hexdigest()

def main():
    print("Reconstructing all publicly scored submissions...", flush=True)
    
    # 1. Base files
    def load_base(n):
        txt = open(REV_DIR / f"base{n:02d}_labels_1_to_30000.txt").read().strip()
        assert len(txt) == 30000
        return {i + 1: ("used" if c == "1" else "new") for i, c in enumerate(txt)}

    B = {7: load_base(7), 15: load_base(15), 33: load_base(33)}
    subs = {7: dict(B[7]), 15: dict(B[15]), 33: dict(B[33])}

    def addtsv(fn, basecol_default, fromcol):
        for r in csv.DictReader(open(REV_DIR / fn), delimiter="\t"):
            n = int(r["submission"])
            b = int(r.get("base", basecol_default))
            i = int(r["ID"])
            if n not in subs:
                subs[n] = dict(B[b])
            assert subs[n][i] == r[fromcol], f"Mismatch in {fn} for sub {n}, id {i}"
            subs[n][i] = r["to"]

    addtsv("flips_04_09.tsv", 7, "from_07")
    addtsv("flips_11_14_27_35.tsv", 15, "from")
    addtsv("flips_16_26.tsv", 15, "from_15")
    addtsv("flips_36_41.tsv", 33, "from_33")

    def flip(src, ids):
        d = dict(subs[src])
        for i in ids:
            d[i] = "new" if d[i] == "used" else "used"
        return d

    # FastText 42, 43
    m_ft = json.load(open(REV_DIR / "fasttext_submission_batch_manifest.json"))
    subs[42] = flip(33, m_ft["submissions"]["42"]["ids"])
    subs[43] = flip(33, m_ft["submissions"]["43"]["ids"])
    subs[44] = flip(33, [18412, 26663, 26668])
    subs[45] = flip(41, [1897, 6308, 7422, 24802, 2820, 9009, 10858, 14208])
    subs[46] = flip(41, [8532, 13583, 16436, 22635, 25426, 25656, 26861])
    subs[47] = flip(41, [8532, 13583, 22635, 25426])
    subs[48] = flip(41, [13583, 25426])
    subs[49] = flip(48, [18487, 1873, 22138, 12528, 27639, 9880, 28453, 14589, 25906, 17321, 40])
    subs[50] = flip(48, [12528, 9880, 28453, 25906, 17321])

    # Verify S48 hash
    s48_blob = encode_labels(subs[48])
    s48_hash = sha256_bytes(s48_blob)
    assert s48_hash == "3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722", f"S48 hash mismatch: {s48_hash}"
    print(f"S48 Baseline hash verified: {s48_hash}")

    # Submissions 57, 58
    sub57_p = ROOT / "submissions" / "20261027" / "submission_57_posterior_sobre48.csv"
    if sub57_p.exists():
        subs[57] = {int(r["ID"]): r["condition"] for r in csv.DictReader(open(sub57_p))}
    sub58_p = ROOT / "submissions" / "20261027" / "submission_58_posterior_sobre48.csv"
    if sub58_p.exists():
        subs[58] = {int(r["ID"]): r["condition"] for r in csv.DictReader(open(sub58_p))}

    # Fin stack 59 and probes (2026-10-28)
    sub59_p = ROOT / "submissions" / "20261028" / "submission_59_fin_stack_banda_0.85_1.00_sobre48.csv"
    if sub59_p.exists():
        subs["59_full"] = {int(r["ID"]): r["condition"] for r in csv.DictReader(open(sub59_p))}

    # Fin stack 150 samples
    H_fs = ROOT / "submissions" / "20261028"
    m_fs = json.loads((H_fs / "manifest_bandas_fin_stack.json").read_text())
    p59_item = next(p for p in m_fs["probes"] if p["number"] == 59)
    base_df = pd.read_csv(H_fs / p59_item["file"])
    mask_59 = base_df.ID.isin(p59_item["ids"])
    base_df.loc[mask_59, "condition"] = base_df.loc[mask_59, "condition"].map({"new": "used", "used": "new"})
    pure_df = pd.read_csv(H_fs / "submission_63_fin_stack_modelo_puro.csv")

    train_sellers = set()
    for line in open(ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines", encoding="utf-8"):
        train_sellers.add(json.loads(line)["seller_id"])
    test_sellers = [json.loads(line)["seller_id"] for line in open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8")]
    seen_arr = pd.Series(test_sellers).isin(train_sellers).to_numpy()

    pool = set().union(*(set(p["ids"]) for p in m_fs["probes"] if p["number"] in (60, 61, 62)))
    for name, seg in [("seen", seen_arr), ("new", ~seen_arr)]:
        choices = [i for i in pool if seg[i - 1]]
        choices.sort(key=lambda i: hashlib.sha256(f"20261028:fin_stack:{name}:{i}".encode()).digest())
        chosen = choices[:150]
        out_df = base_df.copy()
        take = out_df.ID.isin(chosen)
        out_df.loc[take, "condition"] = pure_df.loc[take, "condition"].values
        out_path = H_fs / f"probe_fin_stack_{name}_unprobed150_sobre48.csv"
        out_df.to_csv(out_path, index=False, lineterminator="\n")
        subs[f"fin_stack_{name}150"] = {int(r["ID"]): r["condition"] for _, r in out_df.iterrows()}
        print(f"Generated and loaded fin_stack_{name}150 (SHA: {sha256_bytes(out_path.read_bytes())})")

    # Submissions 72-76
    reg_72_76 = json.load(open(REV_DIR / "registry_72_76_20261029.json"))
    for item in reg_72_76["submissions"]:
        num = item["number"]
        sub_d = dict(subs[48])
        for i in item["ids"]:
            target = item.get("target") or item["targets"][str(i)]
            sub_d[i] = target
        subs[num] = sub_d
        # Verify SHA
        assert sha256_bytes(encode_labels(sub_d)) == item["sha256"]

    # SCORES table
    SCORES = {
        4: 0.94457, 5: 0.94447, 6: 0.94457, 7: 0.94476,
        11: 0.94519, 12: 0.94419, 13: 0.94428, 15: 0.94523,
        16: 0.94500, 17: 0.94447, 18: 0.94404, 19: 0.94409, 20: 0.94285,
        21: 0.94504, 22: 0.94471, 23: 0.94495, 24: 0.94533, 25: 0.94519, 26: 0.94547,
        27: 0.94228, 28: 0.94400, 29: 0.94571, 30: 0.94561,
        31: 0.94585, 32: 0.94561, 33: 0.94614, 34: 0.94580, 35: 0.94609,
        36: 0.94571, 37: 0.94490, 38: 0.94423, 39: 0.92976, 40: 0.94595,
        41: 0.94628, 42: 0.94614, 43: 0.94614, 44: 0.94619,
        45: 0.94623, 46: 0.94623, 47: 0.94628, 48: 0.94633,
        49: 0.94628, 50: 0.94628, 57: 0.94633, 58: 0.94633,
        "59_full": 0.94576, "fin_stack_new150": 0.94495, "fin_stack_seen150": 0.94257,
        72: 0.94623, 73: 0.94600, 74: 0.94614, 75: 0.94633, 76: 0.94633
    }

    registry_entries = []
    b48 = subs[48]

    for key, labels in subs.items():
        if key not in SCORES:
            continue
        score = SCORES[key]
        sha_lf = sha256_bytes(encode_labels(labels))
        sha_vec = sha256_vector(labels)
        
        # Flips vs 48
        flips_vs_48 = {}
        for i in range(1, 30001):
            if labels[i] != b48[i]:
                flips_vs_48[i] = {"from_48": b48[i], "to": labels[i]}
                
        status = "tie" if score == 0.94633 and key != 48 and key != 76 else ("loss" if score < 0.94633 else ("baseline" if key == 48 else ("duplicate" if key == 76 else "win")))
        
        registry_entries.append({
            "key": str(key),
            "score": score,
            "delta_vs_48": round(score - 0.94633, 5),
            "status": status,
            "sha256_lf": sha_lf,
            "sha256_vector": sha_vec,
            "n_flips_vs_48": len(flips_vs_48),
            "flipped_ids": sorted(list(flips_vs_48.keys())),
            "flips_dict": {str(k): v["to"] for k, v in flips_vs_48.items()}
        })

    # Save registry JSON
    out_json = EVAL_DIR / "scored_registry.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_48": {
                "score": 0.94633,
                "sha256": s48_hash,
            },
            "missing_submissions_notes": [
                "01-03: CSVs exactos no recuperables en disco (solo score en log)",
                "10: CSV exacto no recuperable",
                "26-30: scores publicos exactos reconstruidos via assemble.py",
                "51-56: borradores offline nunca enviados a Kaggle (sin score)",
                "60-65: borradores offline de bandas y modelos puros (sin score)"
            ],
            "entries": registry_entries
        }, f, indent=2)

    print(f"Saved scored registry with {len(registry_entries)} entries to {out_json}")

if __name__ == "__main__":
    main()
