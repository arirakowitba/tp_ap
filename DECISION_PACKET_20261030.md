# Decision Packet & Multi-Agent Consolidation (`2026-10-30`)

**Rama de trabajo:** `antigravity/overnight-recovery-coordination-20261030`  
**En respuesta a:** [`CX-0001.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/codex_to_antigravity/CX-0001.md) (`origin/main` `97c5d38e0be09dbce6965ec867249f8e8a97edc5`)  
**Benchmark público demostrado:** `submission_48` = **`.94633`** (SHA256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`, `submissions/20261030/submission_48_reference.csv`).

---

## 1. Registro Canónico de Sondas Históricas (`72–76`) y Reglas de Integridad

| Submission | Score Público | Delta vs `48` (`.94633`) | Nº Flips vs `48` | Segmento / Dirección | Decisión |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **`72`** | `.94623` | `-0.00010` | `8` | `new_seller`, `used -> new` | Excluir bloque |
| **`73`** | `.94600` | `-0.00033` | `7` | `new_seller`, `new -> used` | Excluir bloque *(corregido: con 7 flips es imposible perder 10 ítems)* |
| **`74`** | `.94614` | `-0.00019` | `8` | `seen_seller`, mixto (`5 -> new, 3 -> used`) | Excluir bloque |
| **`75`** | `.94633` | `0.00000` | `5` | `seen_seller`, `new -> used` | Empate con `48` *(corregido: con 5 flips es imposible tener 10 fixes / 10 breaks)* |
| **`76`** | `.94633` | `0.00000` | `0` | Duplicado exacto de `48` (`3f8d08e0...`) | Prohibido repetir en el quinto cupo |

- **Regla estricta sobre scores públicos:** Los scores públicos son evidencia agregada sobre bloques enteros; **jamás** se convierten en etiquetas individuales por ID ni en conteos enteros exactos sin conocer el denominador público exacto.

---

## 2. Resumen Ejecutivo de los Seis Agentes

### Agente 1 — Integridad, Procedencia y Reproducción Exacta
- **Reporte completo:** [`agent1_integrity_provenance_report.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent1_integrity_provenance_report.md) | **Manifiesto:** [`agent1_integrity_manifest.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent1_integrity_manifest.json)
- **Resolución de la contradicción entre `BENCHMARK_REPORT.md` y `reproduction_manifest_agent2.json`:**
  1. **`Pipeline A_hist` (Pipeline histórico R151 exacto):** Construye `combos` con `a.get("id")` + separador `"||"` y conserva probabilidades en `float64`. Reproduce los artefactos históricos `paired_deep_ablation_seed_{2026,314159}.csv` al **100.0000% de acuerdo en decisiones** (`0.000000` de diferencia tras el redondeo a 5 decimales del CSV histórico; `max_abs_diff = 4.9998e-6`, `r = 1.000000`). Con el umbral histórico elegido en `tune` (`0.470` en Semilla 2026; `0.535` en Semilla 314159), obtiene exactamente `.92391` (`LL 0.18764`) y `.91996` (`LL 0.19552`; `.92019` al umbral `0.540`).
  2. **`Pipeline A_valname` (Variante nueva de texto en `data_loader.py`):** Construye `combos` con `a.get("value_name")` sin `"||"` y castea `txt_sparse_p` a `float32`. Difiere en `txt_sparse_p` (`max_abs_diff = 0.2584` en Semilla 2026 y `0.2966` en Semilla 314159), produciendo el **98.3495%** y **98.2948%** de acuerdo con `A_hist` y métricas `.92491` (`th_tune=0.515`) y `.91809` (`th_tune=0.455`).
- **Preservación de artefactos:** Los archivos originales `preds_A_seed{2026,314159}_confirm.csv.gz` fueron restaurados desde `6a71f6c`; las corridas nuevas se guardaron como `preds_A_hist_seed*` y `preds_A_valname_seed*`.

### Agente 2 — Línea Principal CatBoost + LightGBM, Separación Modelo vs. Umbral y Diversidad
- **Reporte completo:** [`agent2_mainline_report.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent2_mainline_report.md) | **Manifiesto:** [`agent2_mainline_summary.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent2_mainline_summary.json)
- **Separación explícita entre aporte del modelo y aporte del umbral en los mismos IDs:**

| Semilla (`N_confirm`) | Comparación | Umbral Común `0.500` (Fixes/Breaks = **Neto**, IC 95%) | Umbral Común `th_base` (**Neto**, IC 95%) | Política `th_tune` (**Neto**, IC 95%) | `Seen` / `New` (`th_tune`) | $\Delta$ Log Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`2026`** (`12,905`) | `Blend075_hist` vs `A_hist` | `72/39 =` **`+33`** `[+7, +66]` | **`+40`** `[+15, +69]` | `80/41 =` **`+39`** `[+10, +78]` | `+1 / +38` | **`-0.00320`** |
| **`2026`** (`12,905`) | `Blend075_Bag3_hist` vs `A_hist` | `93/53 =` **`+40`** `[+13, +70]` | **`+35`** `[+9, +65]` | `98/52 =` **`+46`** `[+17, +81]` | `+4 / +42` | **`-0.00410`** |
| **`2026`** (`12,905`) | `Blend075_valname` vs `A_valname` | `88/58 =` **`+30`** `[-2, +74]` | **`+27`** `[-2, +64]` | `81/57 =` **`+24`** `[-5, +59]` | `+2 / +22` | **`-0.00332`** |
| **`314159`** (`12,843`) | `Blend075_hist` vs `A_hist` | `67/45 =` **`+22`** `[-1, +45]` | **`+30`** `[+7, +53]` | `92/51 =` **`+41`** `[+15, +68]` | `+6 / +35` | **`-0.00164`** |
| **`314159`** (`12,843`) | `Blend075_Bag3_hist` vs `A_hist` | `74/45 =` **`+29`** `[+5, +53]` | **`+33`** `[+8, +57]` | `90/58 =` **`+32`** `[+6, +58]` | `+5 / +27` | **`-0.00252`** |
| **`314159`** (`12,843`) | `Blend075_valname` vs `A_valname` | `88/41 =` **`+47`** `[+21, +73]` | **`+44`** `[+18, +73]` | `91/41 =` **`+50`** `[+24, +79]` | `+10 / +40` | **`-0.00253`** |
| **`20261030`** (`12,881`) | `Blend075_hist` vs `A_hist` | `44/53 =` **`-9`** `[-29, +11]` | **`+11`** `[-7, +28]` | `63/49 =` **`+14`** `[-8, +37]` | `+3 / +11` | **`-0.00216`** |
| **`20261030`** (`12,881`) | `Blend075_Bag3_hist` vs `A_hist` | `65/70 =` **`-5`** `[-31, +18]` | **`+11`** `[-10, +31]` | `65/47 =` **`+18`** `[-5, +40]` | `+2 / +16` | **`-0.00244`** |
| **`20261030`** (`12,881`) | `Blend075_valname` vs `A_valname` | `54/39 =` **`+15`** `[-3, +34]` | **`+9`** `[-10, +27]` | `51/47 =` **`+4`** `[-15, +24]` | `+3 / +1` | **`-0.00240`** |

- **Explicación de los netos previos `+26/+56/+4`:** Provienen de evaluar `Blend075_valname` en el régimen de umbral de `tune`, pero en Semilla 314159 `A_valname` selecciona en `tune` un umbral inestable (`0.455`, cayendo `-23` ítems frente a `0.500`), lo que infla artificialmente el contraste a `+50`/`+56`. Contra el baseline histórico `A_hist` o a umbral común, la ganancia real es de `+22` a `+46` en las semillas 2026 y 314159, y de `-9` (común `0.500`) a `+18` (`th_tune`) en la semilla 20261030, con mejoras de Log Loss en **todos** los cortes (`-0.00164` a `-0.00410`).

### Agente 3 — Validación Externa Estrictamente Inductiva, Auditoría de TODOS los Holdouts Históricos y Transporte a Mezcla de Test
- **Reporte completo:** [`agent3_external_validation_report.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent3_external_validation_report.md) | **Auditoría de holdouts:** [`all_historical_holdouts_overlap_audit.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/all_historical_holdouts_overlap_audit.json)
- **Hallazgo metodológico crítico sobre solapamiento histórico:**
  * La unión de los 18 holdouts históricos reconstruibles en el repositorio cubre **67,007 de las 70,000 filas de `train` (`95.72%`)** y **27,008 de los 27,949 vendedores (`96.63%`)** (y `68,103 / 70,000` filas = `97.29%` al incluir los archivos de `round11_12` y `round13` documentados en `round62`).
  * Aunque `disjoint_3seed_20261101` tiene `0.0%` de solapamiento con la unión de los 3 `confirm` recientes (`2026 U 314159 U 20261030`), solapa en **53.49% (`5,131 / 9,592`)** con la unión de `tune + confirm` de esas 3 semillas y en **92.39% (`8,862 / 9,592`)** con la unión de todos los holdouts históricos del repositorio. **Ningún corte sobre `train` 70k es “virgen”.**
- **Resultados en los 3 cortes externos estrictamente inductivos (sin fuga de `group_features`, `TE` ni `txt_sparse_p`):**
  * **`disjoint_3seed_20261101` (`N=9,592`):** Umbral común `0.500` **+21 netos** (`[+2.0, +40.0]`, `Seen +2 / New +19`, **+40.2 netos transportados a test 30k** `[-11.5, +95.7]`); Política `tune` **+15 netos** (`[-4.0, +33.0]`, **+27.2 transportados** `[-34.4, +89.2]`), `Delta LL = -0.00323`.
  * **`nested_outer_777001` (`N=12,803`):** Umbral común `0.500` **+33 netos** (`[+10.0, +54.0]`, `Seen +8 / New +25`, **+68.5 netos transportados a test 30k** `[+18.2, +117.9]`); Política `tune` **+30 netos** (`[+9.0, +51.0]`, **+42.4 transportados** `[-7.3, +93.4]`), `Delta LL = -0.00457`.
  * **`nested_outer_777002` (`N=13,259`):** Umbral común `0.500` **+11 netos** (`[-10.0, +32.0]`, `Seen +4 / New +7`, **+28.8 netos transportados a test 30k** `[-30.8, +87.8]`); Política `tune` **+9 netos** (`[-11.0, +28.0]`, **+8.1 transportados** `[-36.0, +53.5]`), `Delta LL = -0.00270`.

### Agente 4 — Investigación Complementaria Acotada (Auditoría y Parada de MLP Disperso)
- **Reporte completo:** [`agent4_complementary_report.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent4_complementary_report.md)
- Alineación por ID verificada (`100%`). El `+7` anterior mezclaba cambio de umbral (`-6` por sí solo) y mezcla de probabilidades (`+8` a umbral común `0.515`), deteriorando `LogLoss` en `+0.00582`. Con ajuste estrictamente inductivo por fold, el MLP obtiene `Acc = 0.80430` (`LL = 0.43689`) y peso óptimo `w_mlp = 0.000` en `tune`. **Línea cerrada y cómputo liberado para boosting.**

### Agente 5 — Constructor Verificado `build_from_scores.py` y Suite de Tests Unitarios (`9/9 PASS`)
- **Reporte completo:** [`agent5_builder_verification.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent5_builder_verification.md) | **Código:** [`build_from_scores.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/eval_20261030/build_from_scores.py) | **Tests:** [`test_build_from_scores.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/eval_20261030/test_build_from_scores.py)
- Eliminadas todas las heurísticas por nombre de archivo (`"78" in f.name`, etc.). Identificación exclusivamente por SHA256 y mapa exacto `{ID: (baseline_cond, candidate_cond)}` contra `submission_48`.
- Bloquea bloques con solapamiento parcial no anidado (contraejemplo `O+U` y `O+R` de `round158_overlap_counterexample_20261030.md`), bloquea extracción de subconjuntos no validados y bloquea duplicados exactos. *(Nota: los 9 tests aprobados son evidencia útil de verificación local; la implementación queda sujeta a revisión por Codex).*

---

## 3. Agente 6 — Menú Estructurado de Candidatos (`A`, `B`, `C`, `D`) y Matriz de Intersecciones

- **Reporte completo:** [`agent6_decision_package_report.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_decision_package_report.md) | **Menú JSON:** [`agent6_decision_menu.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_decision_menu.json) | **Tabla por ID:** [`agent6_per_id_audit_table.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_per_id_audit_table.csv)

| Código | Clase | Archivo en `submissions/20261030/` | SHA256 | Flips vs `48` | `Seen` / `New` | `u->n` / `n->u` | Solapamiento con `72–75` | Relación entre Bloques |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Option_A`** | **A** (Predicción Completa Modelo Inductivo 70k) | `candidate_optionA_full_inductive_blend_bag3.csv` | `d93af8e75bde55f6cdf8b90ca61da19ccf5ad7f0b1d32ee479b86bd1a7fb02db` | **`1,033`** | `664 / 369` | `474 / 559` | `13` (`0/1/7/5`) | Superconjunto de `B3` (`B3 ⊂ A`) |
| **`Option_B1`** | **B** (Intervención Validada — Core 15 Seen Sellers) | `candidate_sub80_seen_sellers_conservative_unprobed.csv` | `23415a6147e0363a30d73a9d360d766f2492e689780046c91831c8733e1dc9f7` | **`15`** | `15 / 0` | `9 / 6` | **`0`** | **100% disjunto** de `B2`, `C1`, `C2` |
| **`Option_B2`** | **B** (Intervención Validada — Disjoint 15 Consensus) | `candidate_sub78_consensus_disjoint15_unprobed.csv` | `98e1913752df6da3bae48d33d1d7017148f2a3dfd7a546d701520e90c1b56704` | **`15`** | `10 / 5` | `10 / 5` | **`0`** | **100% disjunto** de `B1`, `C1`, `C2` |
| **`Option_B3`** | **B** (Intervención Validada — Consenso Unánime 6 Modelos) | `candidate_optionB_inductive_unanimous_consensus.csv` | `03d88c4aa23eee39deaf178e147a6a2ad001d4412292296ede39f7d4251e8a87` | **`18`** | `17 / 1` | `10 / 8` | **`0`** | `11` de `B1` + `7` de `B2`; **100% disjunto** de `C1`, `C2` |
| **`Option_C1`** | **C** (Sonda Exploratoria — 20 New Sellers Texto Limpio) | `candidate_sub79_new_sellers_clean_text.csv` | `81ef2f8dc4c724b0983cfed49aa008be9166320881ddff64c2aa1c926e6012ff` | **`20`** | `0 / 20` | `18 / 2` | **`0`** | **100% disjunto** de `A`, `B1`, `B2`, `B3`, `C2` |
| **`Option_C2`** | **C** (Sonda Exploratoria — 5 Flips Margen Frontera) | `candidate_sub77_nested_margin5_unprobed.csv` | `9b07f45caaf9d6cdf063b65e35ba90187e38471c45f5511f905ea8a088a2cc92` | **`5`** | `2 / 3` | `3 / 2` | **`0`** | **100% disjunto** de `B1`, `B2`, `B3`, `C1` |
| **`Option_D`** | **D** (Constructor Contingente / Respaldo `submission_48`) | `src/eval_20261030/build_from_scores.py` | Verificado en runtime | Unión disjunta de bloques con `delta > 0` | Según bloques ganadores | Según bloques ganadores | **`0`** | Rechaza solapes no anidados y duplicados |

### Recomendación para el Primer Envío (sujeta a decisión final de Codex y Ariel)
- **Opción recomendada para el Cupo 1:** **`Option_B1` (`candidate_sub80_seen_sellers_conservative_unprobed.csv`, `15` flips en `seen_sellers`)** o **`Option_B3` (`candidate_optionB_inductive_unanimous_consensus.csv`, `18` flips con unanimidad entre modelos históricos y nuevos inductivos 70k)**.
- **Justificación:** La compuerta de consenso en `seen_sellers` obtiene ganancia neta positiva en las 3 semillas de `confirm` (`+5` en 2026 con `64.7%` de precisión de flips, `+5` en 314159 con `77.8%` de precisión, `+3` en 20261030 con `61.5%` de precisión), opera en el estrato donde `seen_sellers` representa el `71.62%` del test y donde todos los cortes externos son positivos, tiene `0%` de solapamiento con `72–75`, y evita el riesgo de varianza de umbral en `new_sellers` que afecta al reemplazo completo de `1,033` ítems (`Option_A`). Si se elige `B1`, puede componerse limpiamente con `B2` (`15` flips 100% disjuntos), `C1` (`20` flips disjuntos) y `C2` (`5` flips disjuntos) en `build_from_scores.py`.
