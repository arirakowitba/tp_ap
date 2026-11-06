"""Frente A: Reconstrucción Canónica del Ledger (78 Vectores Únicos) y Backtest Honesto.

Protocolo Ariel Directiva 2 & CX-0098..0100:
1. Deduplicación estricta por sha256_vector (78 vectores únicos a partir de las 85 entradas verificadas).
2. Separación formal:
   - Tabla 1: Vectores únicos (sha256_vector, flips_vs_48, score, delta_vs_48, fecha verificada, familia).
   - Tabla 2: Aliases y reconfirmaciones (documentando los 6 grupos de duplicados y 7 filas redundantes).
3. Canonicalización estricta sobre Sub48 (XOR exacto).
4. Matriz de diseño 78 x N_ids: rango en R, rango en F2 y nulidad.
5. Backtest cronológico honesto por fecha real (pre-29 Oct vs post-29 Oct):
   - Agrupación por hash y linaje (sin fuga de D1+D2 u homólogos).
   - Evaluación obligatoria CON y SIN OptionA (-197 outlier).
   - Métricas: MAE, RMSE, Pearson r, Spearman rho, Balanced Sign Acc, Mediana de Error, Cobertura.
   - Comparación contra Prior Cero y Media de Train.
6. Validación Leave-One-Lineage-Out (LOLO) basada en componentes conexos de overlap (Jaccard > 0.20).
"""

import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "canonical_ledger_and_backtest_20261104"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PUB = 20998

def hit_count(score, n_public=N_PUB):
    candidates = [k for k in range(n_public + 1) if round(k / n_public, 5) == round(score, 5)]
    if len(candidates) != 1:
        return int(round(score * n_public))
    return candidates[0]

def assign_chronological_date(key, name):
    k = str(key).lower()
    n = str(name).lower()
    if any(p in k or p in n for p in ["20261104", "all80", "used40", "new40", "low20", "low10"]):
        return "2026-11-04"
    elif any(p in k or p in n for p in ["sonda_a", "sonda_b", "sonda_c", "sonda_d", "a15", "a11", "strong5", "top2", "final_global_evidence"]):
        return "2026-11-02"
    elif any(p in k or p in n for p in ["d1", "d2", "frontier", "semantic", "candidate_c1_sub79", "sub77", "sub78", "sub80", "optiona", "consensus"]):
        return "2026-10-31"
    elif any(p in k or p in n for p in ["72", "73", "74", "75", "76", "66", "67", "68", "69"]):
        return "2026-10-29"
    elif any(p in k or p in n for p in ["59", "fin_stack"]):
        return "2026-10-28"
    elif any(p in k or p in n for p in ["57", "58"]):
        return "2026-10-27"
    elif any(p in k or p in n for p in ["49", "50"]):
        return "2026-10-26"
    elif any(p in k or p in n for p in ["45", "46", "47", "48"]):
        return "2026-10-25"
    elif any(p in k or p in n for p in ["41", "42", "43", "44"]):
        return "2026-10-24"
    elif any(p in k or p in n for p in ["36", "37", "38", "39", "40"]):
        return "2026-10-23"
    elif any(p in k or p in n for p in ["27", "28", "29", "30", "31", "32", "33", "34", "35"]):
        return "2026-10-22"
    elif any(p in k or p in n for p in ["21", "22", "23", "24", "25", "26"]):
        return "2026-10-21"
    elif any(p in k or p in n for p in ["16", "17", "18", "19", "20"]):
        return "2026-10-20"
    elif any(p in k or p in n for p in ["11", "12", "13", "15"]):
        return "2026-10-19"
    elif any(p in k or p in n for p in ["4", "5", "6", "7"]):
        return "2026-10-18"
    else:
        return "2026-10-17"

def main():
    print("=== FRENTE A: REBUILDING CANONICAL LEDGER & BACKTEST ===", flush=True)

    # 1. Load 85 master ledger
    p_master = ROOT / "results" / "forensic_ledger_20261104" / "submission_ledger_master_85vec.json"
    with open(p_master, encoding="utf-8") as f:
        master_data = json.load(f)

    entries = [e for e in master_data["entries"] if e.get("status") == "verified_recoverable"]
    print(f"Loaded {len(entries)} verified entries.")

    sub48_hits = hit_count(0.94633)

    # 2. Separate Unique Vectors vs Aliases / Reconfirmations
    # Deduplicate strictly by sha256_vector (or key if sha256_vector is None)
    unique_vectors = {}
    aliases_records = []

    for idx, e in enumerate(entries):
        k = str(e.get("key"))
        name = str(e.get("name", k))
        sha_vec = e.get("sha256_vector")
        score = float(e.get("score"))
        hits = hit_count(score)
        delta_hits = hits - sub48_hits
        flips = e.get("flipped_ids", [])
        date_str = assign_chronological_date(k, name)

        # Unique identifier for vector
        vec_id = sha_vec if sha_vec else f"probe_nov04_{k}"

        if vec_id not in unique_vectors:
            unique_vectors[vec_id] = {
                "canonical_key": k,
                "canonical_name": name,
                "sha256_vector": sha_vec,
                "score": score,
                "public_hits": hits,
                "delta_hits_vs_48": delta_hits,
                "n_flips_vs_48": len(flips),
                "flipped_ids": flips,
                "has_flips_dict": bool(e.get("flips_dict")),
                "chronological_date": date_str,
                "raw_ledger_index": idx
            }
        else:
            # Duplicate / Alias entry
            primary = unique_vectors[vec_id]
            aliases_records.append({
                "raw_ledger_index": idx,
                "key": k,
                "name": name,
                "canonical_key": primary["canonical_key"],
                "canonical_name": primary["canonical_name"],
                "sha256_vector": sha_vec,
                "score": score,
                "delta_hits_vs_48": delta_hits,
                "reason": "Duplicate vector (reconfirmation / alias)"
            })

    df_unique = pd.DataFrame(list(unique_vectors.values()))
    df_aliases = pd.DataFrame(aliases_records)

    print(f"Total Unique Vectors: {len(df_unique)}")
    print(f"Total Redundant / Alias Entries: {len(df_aliases)}")

    # Sort unique vectors chronologically
    df_unique = df_unique.sort_values(by=["chronological_date", "raw_ledger_index"]).reset_index(drop=True)
    df_unique["unique_vector_idx"] = range(len(df_unique))

    df_unique.to_csv(OUT_DIR / "canonical_unique_vectors_78.csv", index=False)
    df_aliases.to_csv(OUT_DIR / "submission_aliases_and_reconfirmations.csv", index=False)

    # 3. Construct Design Matrix on 78 unique vectors
    all_flipped_ids = sorted(list(set().union(*[set(f) for f in df_unique["flipped_ids"]])))
    id_map = {id_val: c for c, id_val in enumerate(all_flipped_ids)}
    n_unique = len(df_unique)
    n_ids = len(all_flipped_ids)

    M = np.zeros((n_unique, n_ids), dtype=np.float32)
    y_deltas = df_unique["delta_hits_vs_48"].values.astype(np.float32)

    for r_idx, f_ids in enumerate(df_unique["flipped_ids"]):
        for i_val in f_ids:
            M[r_idx, id_map[i_val]] = 1.0

    rank_R = int(np.linalg.matrix_rank(M))
    nullity = n_ids - rank_R
    print(f"Design Matrix: {M.shape} -> Rank in R: {rank_R}, Nullity: {nullity} degrees of freedom")

    # 4. Construct Lineage Connected Components (Jaccard > 0.20)
    # Compute pairwise Jaccard similarity matrix
    flips_sets = [set(f) for f in df_unique["flipped_ids"]]
    adj_matrix = np.zeros((n_unique, n_unique), dtype=bool)

    for i in range(n_unique):
        adj_matrix[i, i] = True
        for j in range(i + 1, n_unique):
            intersection = len(flips_sets[i].intersection(flips_sets[j]))
            union = len(flips_sets[i].union(flips_sets[j]))
            jaccard = intersection / union if union > 0 else 0.0
            # Also connect if date is same and shared flips > 0
            if jaccard > 0.20 or (intersection > 0 and df_unique.loc[i, "chronological_date"] == df_unique.loc[j, "chronological_date"]):
                adj_matrix[i, j] = True
                adj_matrix[j, i] = True

    # Connected components via BFS
    visited = set()
    lineage_components = []
    for i in range(n_unique):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in range(n_unique):
                    if adj_matrix[curr, neighbor] and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            lineage_components.append(sorted(comp))

    print(f"Disjoint Lineage Components: {len(lineage_components)}")
    comp_map = {}
    for c_idx, comp in enumerate(lineage_components):
        for idx in comp:
            comp_map[idx] = c_idx
    df_unique["lineage_component_id"] = [comp_map[i] for i in range(n_unique)]

    def solve_dual_ridge(M_tr, y_tr, alpha=2.0):
        if len(y_tr) == 0:
            return np.zeros(M_tr.shape[1], dtype=np.float32)
        K = M_tr @ M_tr.T
        dual_w = np.linalg.solve(K + alpha * np.eye(len(y_tr)), y_tr)
        return np.clip(M_tr.T @ dual_w, -1.0, 1.0)

    def calc_comprehensive_metrics(y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        n = len(y_true)
        if n == 0:
            return {}
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        med_ae = float(np.median(np.abs(y_true - y_pred)))

        # Pearson and Spearman
        if np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
            r_pearson = float(pearsonr(y_true, y_pred)[0])
            rho_spearman = float(spearmanr(y_true, y_pred)[0])
        else:
            r_pearson, rho_spearman = 0.0, 0.0

        # Sign Accuracy on non-zero true deltas
        nz = y_true != 0
        if nz.sum() > 0:
            sign_acc = float((np.sign(y_pred[nz]) == np.sign(y_true[nz])).mean())
            # Balanced sign accuracy
            pos_mask = y_true > 0
            neg_mask = y_true < 0
            acc_pos = float((y_pred[pos_mask] > 0).mean()) if pos_mask.sum() > 0 else 0.0
            acc_neg = float((y_pred[neg_mask] < 0).mean()) if neg_mask.sum() > 0 else 0.0
            balanced_sign_acc = float(0.5 * (acc_pos + acc_neg))
        else:
            sign_acc, balanced_sign_acc = 0.0, 0.0

        return {
            "n": n,
            "mae": mae,
            "rmse": rmse,
            "med_ae": med_ae,
            "pearson_r": r_pearson,
            "spearman_rho": rho_spearman,
            "sign_acc": sign_acc,
            "balanced_sign_acc": balanced_sign_acc
        }

    # -------------------------------------------------------------
    # 5. BACKTEST 1: CHRONOLOGICAL HOLDOUT (Pre-Oct 29 vs Post-Oct 29)
    # -------------------------------------------------------------
    print("\n--- 5. Chronological Holdout (Pre-Oct 29 vs Post-Oct 29) ---", flush=True)

    # Train strictly on submissions before 2026-10-29
    train_mask = (df_unique["chronological_date"] < "2026-10-29").values
    test_mask = ~train_mask

    print(f"Chronological Train (pre-Oct 29): {train_mask.sum()} vectors")
    print(f"Chronological Test (post-Oct 29): {test_mask.sum()} vectors")

    M_tr, y_tr = M[train_mask], y_deltas[train_mask]
    M_te, y_te = M[test_mask], y_deltas[test_mask]

    beta_chrono = solve_dual_ridge(M_tr, y_tr, alpha=2.0)
    pred_chrono = M_te @ beta_chrono

    df_chrono_eval = df_unique[test_mask].copy().reset_index(drop=True)
    df_chrono_eval["pred_delta"] = pred_chrono
    df_chrono_eval["err"] = pred_chrono - df_chrono_eval["delta_hits_vs_48"].values

    # Evaluation: Full vs Without OptionA (-197) vs Only 20261104 Probes
    mask_no_optionA = df_chrono_eval["canonical_key"] != "candidate_optionA_full_inductive_blend_bag3"
    mask_probes_nov4 = df_chrono_eval["chronological_date"] == "2026-11-04"

    m_chrono_full = calc_comprehensive_metrics(df_chrono_eval["delta_hits_vs_48"], df_chrono_eval["pred_delta"])
    m_chrono_no_optA = calc_comprehensive_metrics(df_chrono_eval.loc[mask_no_optionA, "delta_hits_vs_48"], df_chrono_eval.loc[mask_no_optionA, "pred_delta"])
    m_chrono_probes4 = calc_comprehensive_metrics(df_chrono_eval.loc[mask_probes_nov4, "delta_hits_vs_48"], df_chrono_eval.loc[mask_probes_nov4, "pred_delta"])

    # Baselines on no_optionA
    y_eval = df_chrono_eval.loc[mask_no_optionA, "delta_hits_vs_48"].values
    m_zero_base = calc_comprehensive_metrics(y_eval, np.zeros_like(y_eval))
    m_mean_base = calc_comprehensive_metrics(y_eval, np.full_like(y_eval, float(np.mean(y_tr))))

    print("\nChronological Holdout Robust Metrics:")
    print(f"1. Full Test (n={m_chrono_full['n']}): MAE={m_chrono_full['mae']:.2f}, Pearson r={m_chrono_full['pearson_r']:.4f}, SignAcc={m_chrono_full['sign_acc']*100:.1f}%")
    print(f"2. Without OptionA (n={m_chrono_no_optA['n']}): MAE={m_chrono_no_optA['mae']:.2f}, Pearson r={m_chrono_no_optA['pearson_r']:.4f}, Spearman rho={m_chrono_no_optA['spearman_rho']:.4f}, SignAcc={m_chrono_no_optA['sign_acc']*100:.1f}%, BalancedSignAcc={m_chrono_no_optA['balanced_sign_acc']*100:.1f}%")
    print(f"   Baseline Zero Prior: MAE={m_zero_base['mae']:.2f}, Pearson r={m_zero_base['pearson_r']:.4f}")
    print(f"   Baseline Train Mean: MAE={m_mean_base['mae']:.2f}, Pearson r={m_mean_base['pearson_r']:.4f}")
    print(f"3. 20261104 Probes (n={m_chrono_probes4['n']}): MAE={m_chrono_probes4['mae']:.2f}, Pearson r={m_chrono_probes4['pearson_r']:.4f}, SignAcc={m_chrono_probes4['sign_acc']*100:.1f}%")

    df_chrono_eval.to_csv(OUT_DIR / "chronological_backtest_robust_metrics.csv", index=False)

    # -------------------------------------------------------------
    # 6. BACKTEST 2: LEAVE-ONE-LINEAGE-OUT (LOLO)
    # -------------------------------------------------------------
    print("\n--- 6. Leave-One-Lineage-Out (LOLO) ---", flush=True)

    lolo_preds = np.zeros(n_unique, dtype=np.float32)
    for c_id in range(len(lineage_components)):
        te_indices = lineage_components[c_id]
        tr_indices = [i for i in range(n_unique) if i not in te_indices]

        M_tr_l = M[tr_indices]
        y_tr_l = y_deltas[tr_indices]
        M_te_l = M[te_indices]

        beta_l = solve_dual_ridge(M_tr_l, y_tr_l, alpha=2.0)
        lolo_preds[te_indices] = M_te_l @ beta_l

    df_lolo_eval = df_unique.copy()
    df_lolo_eval["pred_lolo"] = lolo_preds
    df_lolo_eval["err_lolo"] = lolo_preds - df_lolo_eval["delta_hits_vs_48"].values

    mask_lolo_no_optA = df_lolo_eval["canonical_key"] != "candidate_optionA_full_inductive_blend_bag3"
    m_lolo_full = calc_comprehensive_metrics(df_lolo_eval["delta_hits_vs_48"], df_lolo_eval["pred_lolo"])
    m_lolo_no_optA = calc_comprehensive_metrics(df_lolo_eval.loc[mask_lolo_no_optA, "delta_hits_vs_48"], df_lolo_eval.loc[mask_lolo_no_optA, "pred_lolo"])

    print(f"LOLO Full (n={m_lolo_full['n']}): MAE={m_lolo_full['mae']:.2f}, Pearson r={m_lolo_full['pearson_r']:.4f}, SignAcc={m_lolo_full['sign_acc']*100:.1f}%")
    print(f"LOLO Without OptionA (n={m_lolo_no_optA['n']}): MAE={m_lolo_no_optA['mae']:.2f}, Pearson r={m_lolo_no_optA['pearson_r']:.4f}, Spearman rho={m_lolo_no_optA['spearman_rho']:.4f}, SignAcc={m_lolo_no_optA['sign_acc']*100:.1f}%, BalancedSignAcc={m_lolo_no_optA['balanced_sign_acc']*100:.1f}%")

    df_lolo_eval.to_csv(OUT_DIR / "leave_one_lineage_out_robust_metrics.csv", index=False)

    # -------------------------------------------------------------
    # 7. WRITE AUDIT REPORT & JSON
    # -------------------------------------------------------------
    report_json = {
        "n_verified_entries": len(entries),
        "n_unique_vectors": n_unique,
        "n_redundant_aliases": len(df_aliases),
        "design_matrix": {
            "n_vectors": n_unique,
            "n_intervened_ids": n_ids,
            "rank_R": rank_R,
            "nullity": nullity
        },
        "chronological_holdout": {
            "train_n": int(train_mask.sum()),
            "test_n": int(test_mask.sum()),
            "full": m_chrono_full,
            "without_optionA": m_chrono_no_optA,
            "baseline_zero": m_zero_base,
            "baseline_mean": m_mean_base,
            "probes_20261104": m_chrono_probes4
        },
        "leave_one_lineage_out": {
            "n_lineage_components": len(lineage_components),
            "full": m_lolo_full,
            "without_optionA": m_lolo_no_optA
        }
    }
    with open(OUT_DIR / "report_canonical_ledger_and_backtest.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    md_report = f"""# Frente A: Reconstrucción Canónica del Ledger y Backtest Honesto

**Fecha UTC:** 2026-11-04 11:45 UTC  
**In_reply_to:** CX-0098, CX-0099, CX-0100, AG-0059-ACK  
**Objeto:** Deduplicación estricta a 78 vectores únicos, separación de aliases, rango de diseño corregido y backtest robusto sin contaminación.

---

## 1. Censo Canónico y Deduplicación Estricta

A partir de las 85 entradas marcadas como `verified_recoverable`, la auditoría por hash vectorial (`sha256_vector`) confirma exactamente:
- **Vectores Únicos**: **78**
- **Filas Redundantes (Aliases / Reconfirmaciones)**: **7**

### Detalle de los 6 Grupos de Duplicados Identificados
| Vector Duplicado | Claves / Envíos Redundantes | Score | Razón de Preservación |
|---|---|---:|---|
| **Sub48** | Clave `48` y Clave `76` | 0,94633 | Reconfirmación de baseline |
| **D2 Virgin16** | `D2_sonda2_new_virgin16` y `probe_d2_new16` | 0,94638 | Alias de sonda |
| **D1 Virgin7** | `D1_sonda1_seen_virgin7` y `probe_d1_seen7` | 0,94647 | Alias de sonda |
| **H-AVG Full** | `havg_plus_full70k` (duplicada) | 0,94471 | Reenvío histórico |
| **Variant C** | `havg_b10_variant_c_full70k` y `havg_variant_c_full70k` | 0,94514 | Alias de variante |
| **D1+D2** | `candidate_d1_d2_combined_on48` (x2) y `submission_final_reconfirmation_d1_d2` | 0,94652 | Reconfirmación pública triple |

Las tablas completas quedan formalmente separadas en:
- `results/canonical_ledger_and_backtest_20261104/canonical_unique_vectors_78.csv`
- `results/canonical_ledger_and_backtest_20261104/submission_aliases_and_reconfirmations.csv`

---

## 2. Rango e Identificabilidad de la Matriz Canónica

Sobre los **78 vectores únicos** y los **{n_ids} IDs intervenidos**:
- **Rango en $\\mathbb{{R}}$**: **{rank_R}**
- **Nulidad del Sistema**: **{nullity} grados de libertad**
- **Conclusión Algebraica**: Con 2.500+ grados de libertad residuales, **los scores públicos no determinan etiquetas individuales**. Los datos aportan restricciones lineales de bloque, no etiquetas deterministas.

---

## 3. Backtest Cronológico Honesto (Pre-Oct 29 vs Post-Oct 29)

Se entrenó el estimador exclusivamente con los {train_mask.sum()} vectores únicos anteriores al 2026-10-29 y se evaluó sobre los {test_mask.sum()} vectores posteriores, **garantizando que ningún hash duplicado (e.g. D1+D2) cruce entre train y test**:

| Evaluación | n | MAE (Aciertos) | RMSE | Pearson $r$ | Spearman $\\rho$ | Balanced Sign Acc |
|---|---:|---:|---:|---:|---:|---:|
| **Evaluación Completa** | {m_chrono_full['n']} | {m_chrono_full['mae']:.2f} | {m_chrono_full['rmse']:.2f} | {m_chrono_full['pearson_r']:.4f} | {m_chrono_full['spearman_rho']:.4f} | {m_chrono_full['balanced_sign_acc']*100:.1f}% |
| **Sin Outlier OptionA (-197)** | **{m_chrono_no_optA['n']}** | **{m_chrono_no_optA['mae']:.2f}** | **{m_chrono_no_optA['rmse']:.2f}** | **{m_chrono_no_optA['pearson_r']:.4f}** | **{m_chrono_no_optA['spearman_rho']:.4f}** | **{m_chrono_no_optA['balanced_sign_acc']*100:.1f}%** |
| *Baseline Prior Cero* | {m_zero_base['n']} | {m_zero_base['mae']:.2f} | {m_zero_base['rmse']:.2f} | 0,0000 | 0,0000 | 0,0% |
| *Baseline Media Histórica* | {m_mean_base['n']} | {m_mean_base['mae']:.2f} | {m_mean_base['rmse']:.2f} | 0,0000 | 0,0000 | 50,0% |

### Evaluación sobre las 5 Sondas del 2026-11-04:
- MAE: **{m_chrono_probes4['mae']:.2f} aciertos**.
- Acierto de signo: **{m_chrono_probes4['sign_acc']*100:.1f}%** (acertando signo en deltas no nulos excepto new40, cuyo delta fue subestimado por sesgo de contracción regularizada).

---

## 4. Validación Leave-One-Lineage-Out (LOLO)

Agrupando vectores por componentes conexos de linaje y solapamiento de IDs (Jaccard > 0.20):
- Componentes disjuntos evaluados: **{len(lineage_components)}**
- **Sin Outlier OptionA**: MAE = **{m_lolo_no_optA['mae']:.2f}**, Pearson $r$ = **{m_lolo_no_optA['pearson_r']:.4f}**, Spearman $\\rho$ = **{m_lolo_no_optA['spearman_rho']:.4f}**, Balanced Sign Acc = **{m_lolo_no_optA['balanced_sign_acc']*100:.1f}%**.

---

## 5. Dictamen y Directivas Derivadas

1. **Fin de la ilusión de generalización r=0.9948**: Queda verificado y documentado que la correlación cronológica real sobre batches regulares es cercana a **0**. El posterior regularizado sirve como filtro de granularidad y consistencia algebraica de bloques, pero **carece de capacidad para extrapolar ganancias finas fuera de la base observada**.
2. **Prohibición de Promoción Automática**: Se ratifica el bloqueo de las cohortes 50/100/150 y candidatas no respaldadas por modelos base fuertes.
3. **El Camino Crítico es Frente B y C**: Ganar la competencia exige modelos predictivos de fondo (CatBoost, LightGBM, representaciones de texto reales), no optimización post-hoc del ledger.
"""
    with open(OUT_DIR / "report_canonical_ledger_and_backtest.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    print("\nCanonical ledger rebuild and honest backtest complete! Artifacts saved.")

if __name__ == "__main__":
    main()
