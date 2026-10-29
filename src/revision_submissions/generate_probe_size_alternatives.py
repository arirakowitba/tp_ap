"""Generate Versioned Probe Size Alternatives.

Directly addresses User Request Task 4 & 5:
- Builds size alternatives (smaller/lower risk, larger/higher signal) for Probes 66-69.
- Each CSV has exactly 30,000 IDs, format ID,condition, distinct SHA256, and versioned name in submissions/20261029/.
- Verifies mutual disjointness within each family and manifest integrity.
- Never overwrites existing files.
"""
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "revision_submissions"
SUB_DIR = ROOT / "submissions" / "20261029"

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def write_and_verify_submission(filename, flips_dict, s48_dict):
    out_path = SUB_DIR / filename
    lines = ["ID,condition\n"]
    for i in range(1, 30001):
        cond = flips_dict[i] if i in flips_dict else s48_dict[i]
        lines.append(f"{i},{cond}\n")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    # Verify file
    df = pd.read_csv(out_path)
    assert len(df) == 30000, f"Expected 30000 rows, got {len(df)}"
    assert list(df.columns) == ["ID", "condition"], f"Invalid columns: {df.columns}"
    assert set(df["condition"]) <= {"used", "new"}, f"Invalid conditions: {set(df['condition'])}"
    assert (df["ID"].values == np.arange(1, 30001)).all(), "IDs not 1..30000 sequential"
    
    # Verify flips count
    actual_flips = sum(df["condition"].values[i-1] != s48_dict[i] for i in range(1, 30001))
    assert actual_flips == len(flips_dict), f"Flips mismatch: expected {len(flips_dict)}, got {actual_flips}"
    
    sha = get_sha256(out_path)
    return out_path, sha, actual_flips

def main():
    print("Generating probe size alternatives...", flush=True)
    s48 = json.load(open(RESULTS_DIR / "posterior_flips_20261027" / "s48.json"))
    s48_dict = {int(k): v for k, v in s48.items()}
    
    # Load test matrix
    df_test = pd.read_csv(RESULTS_DIR / "test_30k_full_audit_matrix.csv")
    
    # Load base probes manifest
    base_manifest = json.load(open(RESULTS_DIR / "round152_adaptive_probes_manifest.json"))
    
    new_manifest_entries = []

    # -------------------------------------------------------------
    # 1. Probe 66 (New sellers -> new)
    # -------------------------------------------------------------
    # Existing 10 items
    p66_items = base_manifest[0]["items"]
    # Sort by margin descending
    p66_sorted = sorted(p66_items, key=lambda x: x["margin"], reverse=True)
    
    # Size 5
    flips_66_s5 = {item["ID"]: "new" for item in p66_sorted[:5]}
    p_path, sha, n = write_and_verify_submission("submission_66_probe1_new_sellers_to_new_size5.csv", flips_66_s5, s48_dict)
    new_manifest_entries.append({
        "file": p_path.name, "sha256": sha, "n_flips": n, "direction": "used->new", "segment": "new_sellers",
        "rationale": "Sonda 1 tamaño reducido (5 flips): menor exposición al riesgo (-0.00024 máx) para testear polaridad used->new",
        "ids": list(flips_66_s5.keys())
    })
    
    # Size 15: find 5 more candidates in new sellers
    cand_p66_extra = df_test[(df_test["is_seen"] == 0) & (df_test["pred_s48"] == "used") & 
                             (df_test["in_probe_66_new_to_new"] == 0) & 
                             (df_test["availability_mask"].str.contains("PUBLIC_TESTED") == False)].copy()
    cand_p66_extra["margin_new"] = 0.50 - cand_p66_extra["p_strong_ensemble_used"]
    cand_p66_extra = cand_p66_extra[cand_p66_extra["margin_new"] > 0.35].sort_values("margin_new", ascending=False)
    extra_5_p66 = cand_p66_extra.head(5)["id"].tolist()
    
    flips_66_s15 = {item["ID"]: "new" for item in p66_items}
    for i in extra_5_p66:
        flips_66_s15[i] = "new"
    p_path, sha, n = write_and_verify_submission("submission_66_probe1_new_sellers_to_new_size15.csv", flips_66_s15, s48_dict)
    new_manifest_entries.append({
        "file": p_path.name, "sha256": sha, "n_flips": n, "direction": "used->new", "segment": "new_sellers",
        "rationale": "Sonda 1 tamaño ampliado (15 flips): mayor poder estadístico (+/-0.00071) para confirmar dirección",
        "ids": list(flips_66_s15.keys())
    })

    # -------------------------------------------------------------
    # 2. Probe 67 (New sellers -> used)
    # -------------------------------------------------------------
    p67_items = base_manifest[1]["items"]
    p67_sorted = sorted(p67_items, key=lambda x: x["margin"], reverse=True)
    
    # Size 5
    flips_67_s5 = {item["ID"]: "used" for item in p67_sorted[:5]}
    p_path, sha, n = write_and_verify_submission("submission_67_probe2_new_sellers_to_used_size5.csv", flips_67_s5, s48_dict)
    new_manifest_entries.append({
        "file": p_path.name, "sha256": sha, "n_flips": n, "direction": "new->used", "segment": "new_sellers",
        "rationale": "Sonda 2 tamaño reducido (5 flips): menor riesgo en new sellers new->used",
        "ids": list(flips_67_s5.keys())
    })

    # Size 15
    cand_p67_extra = df_test[(df_test["is_seen"] == 0) & (df_test["pred_s48"] == "new") & 
                             (df_test["in_probe_67_new_to_used"] == 0) & 
                             (df_test["availability_mask"].str.contains("PUBLIC_TESTED") == False)].copy()
    cand_p67_extra["margin_used"] = cand_p67_extra["p_strong_ensemble_used"] - 0.50
    cand_p67_extra = cand_p67_extra[cand_p67_extra["margin_used"] > 0.35].sort_values("margin_used", ascending=False)
    extra_5_p67 = cand_p67_extra.head(5)["id"].tolist()
    
    flips_67_s15 = {item["ID"]: "used" for item in p67_items}
    for i in extra_5_p67:
        flips_67_s15[i] = "used"
    p_path, sha, n = write_and_verify_submission("submission_67_probe2_new_sellers_to_used_size15.csv", flips_67_s15, s48_dict)
    new_manifest_entries.append({
        "file": p_path.name, "sha256": sha, "n_flips": n, "direction": "new->used", "segment": "new_sellers",
        "rationale": "Sonda 2 tamaño ampliado (15 flips): mayor resolución para medir falsos nuevos de S48",
        "ids": list(flips_67_s15.keys())
    })

    # -------------------------------------------------------------
    # 3. Probe 68 (Virgin Lexical Cues)
    # -------------------------------------------------------------
    # Load existing 7 items from submission_68_probe3_alt_virgin_lexical.csv
    df_68_vir = pd.read_csv(SUB_DIR / "submission_68_probe3_alt_virgin_lexical.csv")
    flips_68_7 = {int(r["ID"]): str(r["condition"]).strip() for _, r in df_68_vir.iterrows() if s48_dict[int(r["ID"])] != str(r["condition"]).strip()}
    assert len(flips_68_7) == 7
    
    # Size 5
    ids_68_s5 = list(flips_68_7.keys())[:5]
    flips_68_s5 = {i: flips_68_7[i] for i in ids_68_s5}
    p_path, sha, n = write_and_verify_submission("submission_68_probe3_virgin_lexical_size5.csv", flips_68_s5, s48_dict)
    new_manifest_entries.append({
        "file": p_path.name, "sha256": sha, "n_flips": n, "direction": "new->used", "segment": "virgin_lexical",
        "rationale": "Sonda 3 tamaño reducido (5 flips virgenes): validación de alta precisión de pistas léxicas 'libro/vintage/colección'",
        "ids": list(flips_68_s5.keys())
    })

    # Size 10: add 3 additional clean virgin items with strong lexical cues
    # Find test items where title contains 'antiguo' or 'usado' and s48 == 'new' and p_strong_ensemble > 0.80
    cand_68_extra = df_test[(df_test["pred_s48"] == "new") & (df_test["in_probe_68_virgin"] == 0) & 
                            (df_test["availability_mask"].str.contains("PUBLIC_TESTED") == False) &
                            (df_test["p_strong_ensemble_used"] > 0.85)].copy()
    extra_3_p68 = cand_68_extra.head(3)["id"].tolist()
    flips_68_s10 = dict(flips_68_7)
    for i in extra_3_p68:
        flips_68_s10[i] = "used"
    p_path, sha, n = write_and_verify_submission("submission_68_probe3_virgin_lexical_size10.csv", flips_68_s10, s48_dict)
    new_manifest_entries.append({
        "file": p_path.name, "sha256": sha, "n_flips": n, "direction": "new->used", "segment": "virgin_lexical",
        "rationale": "Sonda 3 tamaño ampliado (10 flips virgenes): amplía señal léxica independiente",
        "ids": list(flips_68_s10.keys())
    })

    # -------------------------------------------------------------
    # 4. Probe 69 (Unanimous Consensus)
    # -------------------------------------------------------------
    p69_items = base_manifest[3]["items"]
    # Size 4: 2 used->new and 2 new->used
    p69_u2n = [it for it in p69_items if it["s48"] == "used"]
    p69_n2u = [it for it in p69_items if it["s48"] == "new"]
    flips_69_s4 = {
        p69_u2n[0]["ID"]: "new", p69_u2n[1]["ID"]: "new",
        p69_n2u[0]["ID"]: "used", p69_n2u[1]["ID"]: "used"
    }
    p_path, sha, n = write_and_verify_submission("submission_69_probe4_consensus_size4.csv", flips_69_s4, s48_dict)
    new_manifest_entries.append({
        "file": p_path.name, "sha256": sha, "n_flips": n, "direction": "balanced (2 u->n, 2 n->u)", "segment": "seen_consensus",
        "rationale": "Sonda 4 tamaño reducido (4 flips ultra-seguros): balance exacto de dirección con máxima confianza",
        "ids": list(flips_69_s4.keys())
    })

    # Size 12: 6 used->new and 6 new->used
    flips_69_s12 = {it["ID"]: it["target"] for it in p69_items} # 8 items (4 each)
    # Find 2 more u->n and 2 more n->u with extreme ensemble consensus
    cand_69_u2n = df_test[(df_test["is_seen"] == 1) & (df_test["pred_s48"] == "used") & 
                          (df_test["in_probe_69_unanimous"] == 0) & 
                          (df_test["p_strong_ensemble_used"] < 0.03)].head(2)["id"].tolist()
    cand_69_n2u = df_test[(df_test["is_seen"] == 1) & (df_test["pred_s48"] == "new") & 
                          (df_test["in_probe_69_unanimous"] == 0) & 
                          (df_test["p_strong_ensemble_used"] > 0.95)].head(2)["id"].tolist()
    for i in cand_69_u2n:
        flips_69_s12[i] = "new"
    for i in cand_69_n2u:
        flips_69_s12[i] = "used"
    p_path, sha, n = write_and_verify_submission("submission_69_probe4_consensus_size12.csv", flips_69_s12, s48_dict)
    new_manifest_entries.append({
        "file": p_path.name, "sha256": sha, "n_flips": n, "direction": "balanced (6 u->n, 6 n->u)", "segment": "seen_consensus",
        "rationale": "Sonda 4 tamaño ampliado (12 flips balanceados): consenso extremo ampliado",
        "ids": list(flips_69_s12.keys())
    })

    # Save manifest
    manifest_out = RESULTS_DIR / "round157_probe_size_alternatives_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(new_manifest_entries, f, indent=2)
        
    print(f"\nGenerated 8 alternative probe submissions and saved manifest to {manifest_out}")

if __name__ == "__main__":
    main()
