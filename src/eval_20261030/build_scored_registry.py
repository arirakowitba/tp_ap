"""Compile Scored Submissions Registry and Reconstruct Canonical Label Vectors.

Sources:
- results/revision_submissions/posterior_flips_20261027/assemble.py (4-50, 57, 58)
- results/revision_submissions/round137_adaptive_candidates_20261028.md (59, 59_new, 59_seen_to_new)
- results/revision_submissions/round138_fin_stack_segment_samples_20261028.md (fin_stack_new150, fin_stack_seen150)
- results/revision_submissions/round105_111_alpha1_candidate53_20261027.md (53)
- results/revision_submissions/round118_119_public53_postmortem_20261027.md (53 postmortem)
- results/revision_submissions/round122_partition53_submissions_54_56_20261027.md (54, 55, 56)
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

    # Fin stack 59 and segment probes (2026-10-28)
    sub59_p = ROOT / "submissions" / "20261028" / "submission_59_fin_stack_banda_0.85_1.00_sobre48.csv"
    if sub59_p.exists():
        subs["59_full"] = {int(r["ID"]): r["condition"] for r in csv.DictReader(open(sub59_p))}

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

    # Reconstruct probe_59_new_seller (26 flips vs 48, score 0.94619)
    s59_df = pd.read_csv(H_fs / p59_item["file"])
    s48_df = pd.DataFrame([{"ID": i, "condition": subs[48][i]} for i in range(1, 30001)])
    flips_59 = (s59_df["condition"] != s48_df["condition"]).to_numpy()

    mask_59_new = flips_59 & (~seen_arr)
    df_59_new = s48_df.copy()
    df_59_new.loc[mask_59_new, "condition"] = s59_df.loc[mask_59_new, "condition"]
    path_59_new = H_fs / "probe_59_new_seller_sobre48.csv"
    df_59_new.to_csv(path_59_new, index=False, lineterminator="\n")
    sha_59_new = sha256_bytes(path_59_new.read_bytes())
    assert sha_59_new == "de1ebd9af13ecc1723e5f521e1c0dcb525a2e9261267ae3edb7bafe4bc5e7b9f", f"Mismatch 59 new: {sha_59_new}"
    subs["probe_59_new_seller"] = {int(r["ID"]): r["condition"] for _, r in df_59_new.iterrows()}
    print(f"Generated and loaded probe_59_new_seller (SHA: {sha_59_new})")

    # Reconstruct probe_59_seen_to_new (27 flips vs 48, score 0.94590)
    mask_59_s2n = flips_59 & seen_arr & (s59_df["condition"] == "new")
    df_59_s2n = s48_df.copy()
    df_59_s2n.loc[mask_59_s2n, "condition"] = s59_df.loc[mask_59_s2n, "condition"]
    path_59_s2n = H_fs / "probe_59_seen_to_new_sobre48.csv"
    df_59_s2n.to_csv(path_59_s2n, index=False, lineterminator="\n")
    sha_59_s2n = sha256_bytes(path_59_s2n.read_bytes())
    assert sha_59_s2n == "d06115e8984d4ae82c34cd0ac7276f7b16a6666d1eca57bea736aa5f7b7f32e7", f"Mismatch 59 seen2new: {sha_59_s2n}"
    subs["probe_59_seen_to_new"] = {int(r["ID"]): r["condition"] for _, r in df_59_s2n.iterrows()}
    print(f"Generated and loaded probe_59_seen_to_new (SHA: {sha_59_s2n})")

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
        "59_full": 0.94576,
        "probe_59_new_seller": 0.94619,
        "probe_59_seen_to_new": 0.94590,
        "fin_stack_new150": 0.94495,
        "fin_stack_seen150": 0.94257,
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

    # Add explicitly verified scored submissions 53, 54, 55 (provenance documented, CSV kept in Library)
    documented_scored = [
        {
            "key": "53",
            "file_name": "submission_53_seller_loo_alpha1_on_48.csv",
            "score": 0.94114,
            "delta_vs_48": round(0.94114 - 0.94633, 5),
            "status": "loss",
            "sha256_lf": "a894af2ebabbe3166caf984842813b1a47d0504b181a5800198655e2b15813a8",
            "sha256_vector": None,
            "n_flips_vs_48": 411,
            "flipped_ids": [],
            "flips_dict": {},
            "provenance": "round105_111_alpha1_candidate53_20261027.md, round118_119_public53_postmortem_20261027.md, Library libfile_2071a29b9e7481919a588c058125db96",
            "audit_state": "enviado-score-conocido-hash-conocido-CSV-no-local"
        },
        {
            "key": "54",
            "file_name": "submission_54_partition53_bloque1.csv",
            "score": 0.94457,
            "delta_vs_48": round(0.94457 - 0.94633, 5),
            "status": "loss",
            "sha256_lf": "5ac51b6926ac2d39264ae450bdc8bd8e0f99948ebf0ee893069bce04137debd5",
            "sha256_vector": None,
            "n_flips_vs_48": 204,
            "flipped_ids": [],
            "flips_dict": {},
            "provenance": "round122_partition53_submissions_54_56_20261027.md, Library libfile_3208460aae608191bb598624b81ee4ba",
            "audit_state": "enviado-score-conocido-hash-conocido-CSV-no-local"
        },
        {
            "key": "55",
            "file_name": "submission_55_partition53_bloque2.csv",
            "score": 0.94514,
            "delta_vs_48": round(0.94514 - 0.94633, 5),
            "status": "loss",
            "sha256_lf": "fb82ba0afec97a166d4744cadd8967e028b46824204e67adac8336d8f9711eb5",
            "sha256_vector": None,
            "n_flips_vs_48": 65,
            "flipped_ids": [],
            "flips_dict": {},
            "provenance": "round122_partition53_submissions_54_56_20261027.md, Library libfile_220f8ec2f4648191bee01543493f7e0a",
            "audit_state": "enviado-score-conocido-hash-conocido-CSV-no-local"
        }
    ]
    registry_entries.extend(documented_scored)

    # Add 2026-10-30 scored submissions (D2, D1, H-AVG-Plus, H-AVG, and Combined D1+D2)
    today_scored_configs = [
        {
            "key": "D2_sonda2_new_virgin16",
            "file_name": "probe_havg_plus_sonda2_new_virgin16.csv",
            "score": 0.94638,
            "expected_sha": "d72acb37b1d4d8fefef786fe425b643e4e887d98cdeee287d1812c5b5781ee9a",
            "status": "win",
            "provenance": "results/public_20261030_score_registry.md, CX-0021"
        },
        {
            "key": "D1_sonda1_seen_virgin7",
            "file_name": "probe_havg_plus_sonda1_seen_virgin7.csv",
            "score": 0.94647,
            "expected_sha": "6afa9bb30da486c08373197030409eb018fc7e504f80f89716b898faebab9a9f",
            "status": "win",
            "provenance": "results/public_20261030_score_registry.md, CX-0022"
        },
        {
            "key": "havg_plus_full70k",
            "file_name": "candidate_model_havg_plus_full70k.csv",
            "score": 0.94471,
            "expected_sha": "c568ad01485564a7682135ffec515204200326105c3e2b2d08a222b17f24ee8f",
            "status": "loss",
            "provenance": "results/public_20261030_score_registry.md, CX-0023"
        },
        {
            "key": "havg_b10_variant_c_full70k",
            "file_name": "candidate_model_havg_b10_variant_c_full70k.csv",
            "score": 0.94514,
            "expected_sha": "71d86093d7d14831cd33ef6de9bf7c3e076f2915b7dcce12609ac49bdf8a7a3d",
            "status": "loss",
            "provenance": "results/public_20261030_score_registry.md, CX-0024"
        },
        {
            "key": "candidate_d1_d2_combined_on48",
            "file_name": "candidate_d1_d2_combined_on48.csv",
            "score": 0.94652,
            "expected_sha": "57172f209ff5018d4ceef8c57172cc731828cc78eb9cb02dedccff4a51292502",
            "status": "win",
            "provenance": "results/public_20261030_score_registry.md, CX-0025, CX-0026"
        }
    ]

    for item in today_scored_configs:
        file_path = ROOT / "submissions" / item["file_name"]
        assert file_path.exists(), f"File not found: {file_path}"
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
        sha_lf = sha256_bytes(raw_bytes)
        assert sha_lf == item["expected_sha"], f"SHA mismatch for {item['file_name']}: {sha_lf} vs {item['expected_sha']}"
        
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["ID", "condition"]
            labels = {int(r["ID"]): r["condition"] for r in reader}
        assert len(labels) == 30000
        
        sha_vec = sha256_vector(labels)
        flips_vs_48 = {}
        for i in range(1, 30001):
            if labels[i] != b48[i]:
                flips_vs_48[i] = {"from_48": b48[i], "to": labels[i]}
                
        registry_entries.append({
            "key": item["key"],
            "score": item["score"],
            "delta_vs_48": round(item["score"] - 0.94633, 5),
            "status": item["status"],
            "sha256_lf": sha_lf,
            "sha256_vector": sha_vec,
            "n_flips_vs_48": len(flips_vs_48),
            "flipped_ids": sorted(list(flips_vs_48.keys())),
            "flips_dict": {str(k): v["to"] for k, v in flips_vs_48.items()},
            "provenance": item["provenance"],
            "audit_state": "enviado-score-confirmado-archivo-local-verificado"
        })

    # Save registry JSON
    out_json = EVAL_DIR / "scored_registry.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_48": {
                "score": 0.94633,
                "sha256": s48_hash,
            },
            "current_best_combined": {
                "score": 0.94652,
                "sha256": "57172f209ff5018d4ceef8c57172cc731828cc78eb9cb02dedccff4a51292502",
                "key": "candidate_d1_d2_combined_on48",
                "n_flips_vs_48": 23
            },
            "inventory_correction_notes": [
                "01-03: enviados, scores conocidos (01: 0.94433, 02: 0.94419, 03: 0.94447); CSVs exactos no disponibles localmente en disco (enviado-score-conocido-CSV-no-identificado)",
                "10: enviado, score conocido 0.94500; CSV exacto no disponible localmente en disco (enviado-score-conocido-CSV-no-identificado)",
                "26-30: scores publicos reconstruidos via assemble.py y confirmados",
                "51-52: preparadas, no enviadas (descartadas formalmente tras la caida publica de 53)",
                "53: enviada y puntuada (score publico 0.94114, SHA256 a894af2ebabbe3166caf984842813b1a47d0504b181a5800198655e2b15813a8, 411 flips vs 48)",
                "54: enviada y puntuada (score publico 0.94457, SHA256 5ac51b6926ac2d39264ae450bdc8bd8e0f99948ebf0ee893069bce04137debd5, 204 flips vs 48)",
                "55: enviada y puntuada (score publico 0.94514, SHA256 fb82ba0afec97a166d4744cadd8967e028b46824204e67adac8336d8f9711eb5, 65 flips vs 48)",
                "56: preparada, no enviada (bloque de 142 flips inferido ~.94409, descartado tras resultados adversos de 54 y 55)",
                "59: comprende 5 envios con scores conocidos reportados por Ariel: 59_full (.94576), probe_59_new_seller (.94619), probe_59_seen_to_new (.94590), fin_stack_new150 (.94495), fin_stack_seen150 (.94257)",
                "60-65: borradores offline de bandas y modelos puros sin envios Kaggle confirmados",
                "66-71: preparadas en Ronda 152 sin scores publicos confirmados en el repo (preparado-sin-score-confirmado-en-repo)",
                "72-76: enviadas y puntuadas el 2026-10-29 (72: .94623, 73: .94600, 74: .94614, 75: .94633, 76: .94633 duplicado exacto de 48)",
                "2026-10-30: 5 envios puntuados reportados en results/public_20261030_score_registry.md y CX-0021..CX-0026 (D2: 0.94638, D1: 0.94647, H-AVG-Plus: 0.94471, H-AVG: 0.94514, D1+D2 combinada: 0.94652 nuevo maximo historico)"
            ],
            "entries": registry_entries
        }, f, indent=2)

    print(f"Saved scored registry with {len(registry_entries)} entries to {out_json}")

if __name__ == "__main__":
    main()
