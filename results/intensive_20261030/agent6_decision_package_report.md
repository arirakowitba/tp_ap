# Agente 6 — Paquete de Decisión y Auditoría Cruzada de Submissions Históricas (CX-0002)

**Fecha:** 2026-10-30 01:49:14  
**Referencia base (`submission_48`):** `.94633` | SHA256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`  
**Manifiesto JSON:** [`agent6_decision_menu.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_decision_menu.json)  
**Auditoría cruzada 01–76:** [`agent6_historical_submissions_cross_audit.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_historical_submissions_cross_audit.csv)  
**Tabla de auditoría por ID:** [`agent6_per_id_audit_table.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_per_id_audit_table.csv)  

---

## 1. Auditoría Completa de Submissions Públicas Históricas (`01–76`)

- **Submissions 30k válidas recuperadas en repositorio (`N = 33`):**
  `[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 48, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]`
- **Submissions históricas faltantes en repositorio (`N = 43`):**
  `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56]`

> [!IMPORTANT]
> **Corrección metodológica verificada:** Se prohíbe calificar a una candidata como 'unprobed' basándose únicamente en no tocar las sondas 72–75. Como se documenta en la matriz siguiente, los flips de `Option_B1`, `Option_B2`, `Option_B3` y `Option_C2` están contenidos en los envíos de modelo completo `Sub 64` (1,774 flips) y `Sub 71` (1,338 flips), aunque son 100% disjuntos de las sondas pequeñas aisladas `72–76`. Únicamente `Option_C1` (20 new sellers texto limpio) tiene 0% de solapamiento con las 33 submissions históricas.

### Matriz Resumida de Solapamiento con Submissions Clave

| Candidata | Flips vs `48` | `Seen` / `New` | Solape Sondas `72–75` | Solape `Sub 64` (1,774 flips) | Solape `Sub 71` (1,338 flips) | Solape `Sub 65` (141 flips) | Estado Epistemológico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Option_A_Full_Inductive_Blend_Bag3`** | **`1033`** | `664 / 369` | `13` | `658` (`63.7%`) | `784` (`75.9%`) | `61` | `FULL_MODEL_REPLACEMENT_DEVELOPMENT_GRADE` |
| **`Option_B1_Seen15_Consensus_Sub80`** | **`15`** | `15 / 0` | `0` | `15` (`100.0%`) | `13` (`86.7%`) | `0` | `EXPLORATORY_CANDIDATE_INTERVENTION` |
| **`Option_B2_Disjoint15_Consensus_Sub78`** | **`15`** | `10 / 5` | `0` | `15` (`100.0%`) | `15` (`100.0%`) | `5` | `EXPLORATORY_CANDIDATE_INTERVENTION` |
| **`Option_B3_Inductive_CrossPipeline_Consensus`** | **`18`** | `17 / 1` | `0` | `18` (`100.0%`) | `18` (`100.0%`) | `1` | `EXPLORATORY_CANDIDATE_INTERVENTION` |
| **`Option_C1_Exploratory_NewSellers20_Sub79`** | **`20`** | `0 / 20` | `0` | `0` (`0.0%`) | `0` (`0.0%`) | `0` | `EXPLORATORY_DIAGNOSTIC_PROBE` |
| **`Option_C2_Exploratory_Margin5_Sub77`** | **`5`** | `2 / 3` | `0` | `5` (`100.0%`) | `5` (`100.0%`) | `3` | `EXPLORATORY_DIAGNOSTIC_PROBE` |

---

## 2. Reclasificación y Aclaración de Reglas de Intervención (Opciones B1, B2, B3)

- **B3 vs. Regla algorítmica en `confirm`:**
  - En `confirm`, se evaluó la regla algorítmica estricta `(pred_blend != pred_base) & (pred_lgb == pred_blend) & (is_seen == 1) & (|p_blend - 0.50| >= 0.04)`. Esta regla obtiene ganancia neta en las 3 particiones etiquetadas (`+5`, `+5`, `+3` netos con precisión de flips entre 61.5% y 77.8%).
  - Sin embargo, el archivo de test `candidate_optionB_inductive_unanimous_consensus.csv` (`B3`) se construyó intersectando dos listas de test previas (B1 U B2) filtradas por coincidencia con el refit 70k, **contiene 1 vendedor nuevo** (17 seen / 1 new) y no impuso el umbral de margen exacto de confirm.
  - **Resolución:** Siguiendo `CX-0002`, se revoca la etiqueta `VALIDATED_UNANIMOUS_CROSS_PIPELINE_GATE`. Las opciones B1, B2 y B3 se clasifican formalmente como **Intervenciones Candidatas Exploratorias**.

---

## 3. Menú de Alternativas para la Decisión del Cupo

| Opción | Clase | Archivo | SHA256 | Flips vs `48` | `Seen` / `New` | `u->n` / `n->u` | Relación con Otros Bloques |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Option_A_Full_Inductive_Blend_Bag3`** | A (Full Model Prediction) | `submissions\20261030\candidate_optionA_full_inductive_blend_bag3.csv` | `d93af8e75bde...` | **`1033`** | `664 / 369` | `474 / 559` | `FULL_MODEL_REPLACEMENT_DEVELOPMENT_GRADE` |
| **`Option_B1_Seen15_Consensus_Sub80`** | B (Restricted Candidate Intervention — Core 15 Seen Sellers) | `submissions\20261030\candidate_sub80_seen_sellers_conservative_unprobed.csv` | `23415a6147e0...` | **`15`** | `15 / 0` | `9 / 6` | `EXPLORATORY_CANDIDATE_INTERVENTION` |
| **`Option_B2_Disjoint15_Consensus_Sub78`** | B (Restricted Candidate Intervention — Disjoint 15 Consensus) | `submissions\20261030\candidate_sub78_consensus_disjoint15_unprobed.csv` | `98e1913752df...` | **`15`** | `10 / 5` | `10 / 5` | `EXPLORATORY_CANDIDATE_INTERVENTION` |
| **`Option_B3_Inductive_CrossPipeline_Consensus`** | B (Restricted Candidate Intervention — Cross-Pipeline Consensus, 18 Flips) | `submissions\20261030\candidate_optionB_inductive_unanimous_consensus.csv` | `03d88c4aa23e...` | **`18`** | `17 / 1` | `10 / 8` | `EXPLORATORY_CANDIDATE_INTERVENTION` |
| **`Option_C1_Exploratory_NewSellers20_Sub79`** | C (Exploratory Diagnostic Probe — 20 New Sellers Clean Text) | `submissions\20261030\candidate_sub79_new_sellers_clean_text.csv` | `81ef2f8dc4c7...` | **`20`** | `0 / 20` | `18 / 2` | `EXPLORATORY_DIAGNOSTIC_PROBE` |
| **`Option_C2_Exploratory_Margin5_Sub77`** | C (Exploratory Diagnostic Probe — 5 Boundary Flips) | `submissions\20261030\candidate_sub77_nested_margin5_unprobed.csv` | `9b07f45caaf9...` | **`5`** | `2 / 3` | `3 / 2` | `EXPLORATORY_DIAGNOSTIC_PROBE` |
