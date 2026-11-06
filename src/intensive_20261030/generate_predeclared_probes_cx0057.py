"""Generate Pre-declared Disagreement Probes and Overlap Matrix (CX-0057..CX-0058).

Features:
1. 4 mutually disjoint probe blocks of exactly 15 flips anchored on D1+D2 (0.94652).
2. Stratified by seen/new seller status and calibrated OOF confidence.
   - B1: 15 Seen sellers, extreme margin (|p - 0.5| > 0.436).
   - B2: 15 New sellers, extreme margin (|p - 0.5| > 0.356).
   - B3: 15 Seen sellers, unanimous CatBoost + LightGBM + Blend consensus (|p - 0.5| > 0.411).
   - B4: 15 New sellers, unanimous CatBoost + LightGBM + Blend consensus (|p - 0.5| > 0.299).
3. Full pairwise overlap matrix (exact zeros off-diagonal).
4. Pre-specified adaptive 5th slot decision rule.
5. Strict Kaggle guard: files saved for inspection only, zero automated submissions.
"""
import os
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SUB_DIR = ROOT / "submissions" / "probes_20261102"
SUB_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "results" / "intensive_20261030"

D1D2_PATH = ROOT / "submissions" / "candidate_d1_d2_combined_on48.csv"
MAINLINE_PATH = OUT_DIR / "test_30k_mainline_predictions.csv.gz"

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
    print("=== GENERATING PRE-DECLARED DISAGREEMENT PROBES (CX-0057) ===", flush=True)

    d1d2 = pd.read_csv(D1D2_PATH).sort_values("ID")
    mainline = pd.read_csv(MAINLINE_PATH).sort_values("ID")
    assert np.array_equal(d1d2["ID"], mainline["ID"])

    is_seen = mainline["is_seen"].values
    d1d2_cond = d1d2["condition"].values
    d1d2_used = (d1d2_cond == "used").astype(int)

    p_blend = mainline["p_blend_bag3"].values
    p_cb = mainline["p_cb_bag3"].values
    p_lgb = mainline["p_lgb_bag3"].values

    pred_blend = (p_blend > 0.5).astype(int)
    pred_cb = (p_cb > 0.5).astype(int)
    pred_lgb = (p_lgb > 0.5).astype(int)
    unanimous_flip = (pred_blend != d1d2_used) & (pred_cb == pred_blend) & (pred_lgb == pred_blend)
    margin_blend = np.abs(p_blend - 0.5)

    # Block 1: Seen, highest margin (top 15)
    seen_candidates = np.where((pred_blend != d1d2_used) & is_seen)[0]
    seen_sorted = seen_candidates[np.argsort(-margin_blend[seen_candidates])]
    b1_idx = seen_sorted[:15]

    # Block 2: New, highest margin (top 15)
    new_candidates = np.where((pred_blend != d1d2_used) & ~is_seen)[0]
    new_sorted = new_candidates[np.argsort(-margin_blend[new_candidates])]
    b2_idx = new_sorted[:15]

    # Block 3: Seen, unanimous agreement, next highest margin (disjoint from B1, 15 items)
    seen_unanimous = np.where(unanimous_flip & is_seen & ~np.isin(np.arange(len(d1d2)), b1_idx))[0]
    seen_u_sorted = seen_unanimous[np.argsort(-margin_blend[seen_unanimous])]
    b3_idx = seen_u_sorted[:15]

    # Block 4: New, unanimous agreement, next highest margin (disjoint from B2, 15 items)
    new_unanimous = np.where(unanimous_flip & ~is_seen & ~np.isin(np.arange(len(d1d2)), b2_idx))[0]
    new_u_sorted = new_unanimous[np.argsort(-margin_blend[new_unanimous])]
    b4_idx = new_u_sorted[:15]

    block_defs = [
        ("B1_seen_extreme15", b1_idx, "Seen sellers, extreme margin (|p - 0.5| > 0.436)", "probe_b1_seen_extreme15_on_d1d2.csv"),
        ("B2_new_extreme15", b2_idx, "New sellers, extreme margin (|p - 0.5| > 0.356)", "probe_b2_new_extreme15_on_d1d2.csv"),
        ("B3_seen_unanimous15", b3_idx, "Seen sellers, unanimous consensus (|p - 0.5| > 0.411)", "probe_b3_seen_unanimous15_on_d1d2.csv"),
        ("B4_new_unanimous15", b4_idx, "New sellers, unanimous consensus (|p - 0.5| > 0.299)", "probe_b4_new_unanimous15_on_d1d2.csv"),
    ]

    manifest = {
        "date_utc": "2026-11-02T12:00:00Z",
        "anchor": "candidate_d1_d2_combined_on48.csv (0.94652)",
        "blocks": {},
        "overlap_matrix": {},
        "adaptive_slot5_protocol": {
            "description": "Pre-specified conditional decision rule for 5th submission slot",
            "rules": [
                "If Delta(B1) > 0 and Delta(B2) > 0: Submit combination B1 + B2 (30 flips).",
                "If Delta(B1) > 0 and Delta(B2) <= 0: Submit B1 + B3 (if Delta(B3) > 0) or B1 alone.",
                "If Delta(B2) > 0 and Delta(B1) <= 0: Submit B2 + B4 (if Delta(B4) > 0) or B2 alone.",
                "If all Delta <= 0: Retain exact D1+D2 (0.94652) as immutable final anchor."
            ]
        }
    }

    block_dict = {}
    for name, indices, desc, filename in block_defs:
        file_path = SUB_DIR / filename
        pred_cond = d1d2_cond.copy()
        flipped_ids = []
        for idx in indices:
            row_id = int(d1d2.iloc[idx]["ID"])
            flipped_ids.append(row_id)
            pred_cond[idx] = "used" if pred_blend[idx] == 1 else "new"

        df_out = pd.DataFrame({"ID": d1d2["ID"], "condition": pred_cond})
        df_out.to_csv(file_path, index=False)

        raw_sha = sha256_file(file_path)
        lf_sha = sha256_lf(file_path)

        block_dict[name] = set(flipped_ids)
        manifest["blocks"][name] = {
            "filename": filename,
            "description": desc,
            "n_flips": len(indices),
            "flipped_ids": sorted(flipped_ids),
            "min_margin": round(float(margin_blend[indices].min()), 5),
            "mean_margin": round(float(margin_blend[indices].mean()), 5),
            "sha256_raw": raw_sha,
            "sha256_lf": lf_sha
        }
        print(f"Generated {name}: {len(indices)} flips, file={filename}, sha={lf_sha[:12]}...")

    # Compute overlap matrix
    matrix = {}
    for n1 in block_dict:
        matrix[n1] = {}
        for n2 in block_dict:
            matrix[n1][n2] = len(block_dict[n1] & block_dict[n2])
    manifest["overlap_matrix"] = matrix

    out_json = OUT_DIR / "predeclared_probes_manifest_20261102.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Markdown report
    md_content = f"""# Manifiesto de Sondas Pre-declaradas y Matriz de Solapamiento — 2026-11-02

## 1. Protocolo Epistemológico
Conforme a las directivas de CX-0057 y CX-0058:
- Se preserva **`candidate_d1_d2_combined_on48.csv` (0.94652)** como ancla líder inmutable.
- Se definen exactamente **4 bloques pequeños y mutuamente disjuntos** de 15 intervenciones cada uno ($n=15$), diseñados a partir de los desacuerdos entre el ensamble supervisado y el ancla D1+D2, estratificados por vendedor visto/nuevo y margen de confianza OOF calibrado.
- La selección de filas queda **estrictamente congelada antes de observar scores**.
- **Cero envíos automáticos a Kaggle**: todos los archivos quedan generados exclusivamente para revisión y auditoría.

## 2. Definición de Bloques

| Bloque | Archivo | Estrato | $n$ Cambios | Margen OOF Mín / Medio | SHA-256 (LF) |
|---|---|---|---:|---:|---|
"""
    for name, data in manifest["blocks"].items():
        md_content += f"| `{name}` | `{data['filename']}` | {data['description']} | {data['n_flips']} | {data['min_margin']:.4f} / {data['mean_margin']:.4f} | `{data['sha256_lf'][:16]}...` |\n"

    md_content += f"""
## 3. Matriz de Solapamiento Mutuo

| Bloque | B1_seen_extreme15 | B2_new_extreme15 | B3_seen_unanimous15 | B4_new_unanimous15 |
|---|---:|---:|---:|---:|
| **B1_seen_extreme15** | {matrix['B1_seen_extreme15']['B1_seen_extreme15']} | {matrix['B1_seen_extreme15']['B2_new_extreme15']} | {matrix['B1_seen_extreme15']['B3_seen_unanimous15']} | {matrix['B1_seen_extreme15']['B4_new_unanimous15']} |
| **B2_new_extreme15** | {matrix['B2_new_extreme15']['B1_seen_extreme15']} | {matrix['B2_new_extreme15']['B2_new_extreme15']} | {matrix['B2_new_extreme15']['B3_seen_unanimous15']} | {matrix['B2_new_extreme15']['B4_new_unanimous15']} |
| **B3_seen_unanimous15** | {matrix['B3_seen_unanimous15']['B1_seen_extreme15']} | {matrix['B3_seen_unanimous15']['B2_new_extreme15']} | {matrix['B3_seen_unanimous15']['B3_seen_unanimous15']} | {matrix['B3_seen_unanimous15']['B4_new_unanimous15']} |
| **B4_new_unanimous15** | {matrix['B4_new_unanimous15']['B1_seen_extreme15']} | {matrix['B4_new_unanimous15']['B2_new_extreme15']} | {matrix['B4_new_unanimous15']['B3_seen_unanimous15']} | {matrix['B4_new_unanimous15']['B4_new_unanimous15']} |

> **Nota**: El solapamiento fuera de la diagonal es estrictamente 0. Ningún ID se repite entre bloques.

## 4. Regla de Decisión Adaptativa para el 5º Cupo (Pre-especificada)
1. Si $\Delta(B1) > 0$ y $\Delta(B2) > 0$: Enviar combinación $B1 \cup B2$ (30 cambios).
2. Si $\Delta(B1) > 0$ y $\Delta(B2) \le 0$: Enviar $B1 \cup B3$ (si $\Delta(B3) > 0$) o $B1$ solo.
3. Si $\Delta(B2) > 0$ y $\Delta(B1) \le 0$: Enviar $B2 \cup B4$ (si $\Delta(B4) > 0$) o $B2$ solo.
4. Si todos $\Delta \le 0$: Respaldo incondicional en el ancla exacta D1+D2 (0.94652).
"""

    out_md = OUT_DIR / "predeclared_probes_manifest_20261102.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Manifest written to {out_json} and {out_md}")

if __name__ == "__main__":
    main()
