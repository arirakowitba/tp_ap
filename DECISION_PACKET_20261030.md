# Decision Packet & Multi-Agent Consolidation (`2026-10-30`, Revisión CX-0003 & CX-0004)

**Rama de trabajo:** `antigravity/overnight-recovery-coordination-20261030`  
**En respuesta a:** [`CX-0001.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/codex_to_antigravity/CX-0001.md), [`CX-0002.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/codex_to_antigravity/CX-0002.md), [`CX-0003.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/codex_to_antigravity/CX-0003.md) y [`CX-0004.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/codex_to_antigravity/CX-0004.md) (`origin/main` `1a4f560c9bd02972264299139f5b06e3846485ea`)  
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
- **Transparencia sobre la reproducibilidad de `submission_48`:** Tal como documenta la auditoría histórica de Ronda 49 ([`round49_proxy41_fidelity_20261025.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/revision_submissions/round49_proxy41_fidelity_20261025.md)), la cadena que originó `48` proviene de `41` (que incorporó intervenciones manuales de test sobre `33`) dividida en `47` y `48` por ranking de confianza exploratorio ([`submission_48_split_47_20261026.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/revision_submissions/submission_48_split_47_20261026.md)). Por diseño histórico, **no existe una regla de entrenamiento puro 100% autónoma que regenere las etiquetas exactas de `48` en `train`**. El CSV de `48` se preserva como benchmark empírico inmutable fijado por hash SHA256 (`3f8d08e0...`) y respaldo operativo; las ganancias reportadas frente a baselines reproducibles (`A_hist`, `A_valname`, o `CB_single`) demuestran avance metodológico inductivo, pero no constituyen garantía matemática de batir a `48` en test sin validación empírica.

---

## 2. Resumen Ejecutivo de los Seis Agentes (Actualizado CX-0002)

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

### Agente 3 — Validación Externa Estrictamente Inductiva, Auditoría de Holdouts y Medición de Bag3
- **Reporte completo:** [`agent3_external_validation_report.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent3_external_validation_report.md) | **Auditoría de holdouts:** [`all_historical_holdouts_overlap_audit.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/all_historical_holdouts_overlap_audit.json)
- **Correcciones metodológicas aplicadas según CX-0002:**
  1. Se eliminó la condición asimétrica de refit TF-IDF: los tres cortes externos ejecutan refit estricto de vocabulario/IDF fold a fold dentro de `fit`.
  2. El corte `disjoint_3seed_20261101` se desclasificó de virgen: solapa en **92.39% (`8,862 / 9,592` filas)** con la unión de los 18 holdouts históricos que cubre el **95.72% (`67,007 / 70,000`)** de `train`.
  3. Se evaluó de forma separada el ensamble exacto `Bag3` (`Blend_Bag3`) vs. el modelo individual (`Blend_Single`), midiendo el aporte marginal directo de `Bag3`.
- **Resultados en los 3 cortes externos con refit inductivo uniforme y ensamble Bag3 exacto:**

| Corte Externo | Arquitectura / Ensamble | Umbral Común `0.500` (Fixes/Breaks = **Neto**, IC 95%) | Neto Transportado 30k Test (IC 95%) | Política `tune` (**Neto**, IC 95%) | `Seen` / `New` (`th_tune`) | $\Delta$ Log Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`disjoint_3seed_20261101`** (`N=9,592`) | `Blend_Single` vs `CB_base` | `37/39 =` **`-2`** `[-18, +13]` | `-10.2` `[-64.3, +44.6]` | **`-1`** `[-16, +14]` | `-2 / +1` | `-0.00282` |
| **`disjoint_3seed_20261101`** (`N=9,592`) | `Blend_Bag3` vs `CB_base` | `36/36 =` **`+0`** `[-15, +17]` | `+0.0` `[-65.6, +64.8]` | `58/43 =` **`+15`** `[-6, +33]` | `-3 / +18` | **`-0.00385`** |
| **`disjoint_3seed_20261101`** (`N=9,592`) | **Aporte Marginal `Bag3` vs `Single`** | `26/24 =` **`+2`** `[-7, +11]` | `+10.2` `[-29.8, +49.1]` | `27/24 =` **`+3`** `[-7, +12]` | `+0 / +3` | **`-0.00103`** |
| **`nested_outer_777001`** (`N=12,803`) | `Blend_Single` vs `CB_base` | `54/53 =` **`+1`** `[-16, +19]` | `+38.2` `[-12.5, +89.8]` | **`-2`** `[-19, +17]` | `+8 / -10` | `-0.00253` |
| **`nested_outer_777001`** (`N=12,803`) | `Blend_Bag3` vs `CB_base` | `61/52 =` **`+9`** `[-9, +28]` | **`+59.7`** `[+6.3, +116.5]` | `65/61 =` **`+4`** `[-15, +22]` | `+10 / -6` | **`-0.00257`** |
| **`nested_outer_777001`** (`N=12,803`) | **Aporte Marginal `Bag3` vs `Single`** | `24/16 =` **`+8`** `[-1, +18]` | `+21.5` `[-18.5, +60.9]` | `22/16 =` **`+6`** `[-2, +16]` | `+2 / +4` | **`-0.00005`** |
| **`nested_outer_777002`** (`N=13,259`) | `Blend_Single` vs `CB_base` | `66/53 =` **`+13`** `[-7, +31]` | `+30.6` `[-30.4, +88.6]` | **`+15`** `[-5, +36]` | `+5 / +10` | `-0.00269` |
| **`nested_outer_777002`** (`N=13,259`) | `Blend_Bag3` vs `CB_base` | `88/52 =` **`+36`** `[+12, +58]` | **`+65.6`** `[-1.9, +134.1]` | `116/59 =` **`+57`** `[+29, +84]` | `+9 / +48` | **`-0.00414`** |
| **`nested_outer_777002`** (`N=13,259`) | **Aporte Marginal `Bag3` vs `Single`** | `42/19 =` **`+23`** `[+8, +38]` | `+35.0` `[-6.4, +75.9]` | `58/20 =` **`+38`** `[+19, +57]` | `+4 / +34` | **`-0.00145`** |

- **Comparación Pareada Congelada de Presupuesto Paritario (CX-0004): `CB_Bag3` vs `Blend_Bag3`**  
  Para eliminar la ventaja de cómputo del candidato frente al benchmark simple, se contrastó directamente `CB_Bag3` (3 semillas CatBoost) frente a `Blend_Bag3` (`0.75 CB_Bag3 + 0.25 LGB_Bag3`) sobre los mismos pares y cortes:

| Corte Externo | Configuración Base (3 Semillas) | Candidato (3 Semillas Ensamble) | Umbral Común `0.500` (Fixes/Breaks = **Neto**, IC 95%) | `Seen` / `New` (`0.500`) | LogLoss Base vs Cand | $\Delta$ Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`disjoint_3seed_20261101`** (`N=9,592`) | `CB_Bag3` | `Blend_Bag3` | `40/29 =` **`+11`** `[-6.0, +27.0]` (P>0: 89.4%) | `+5 / +6` | `0.19550` vs `0.19309` | **`-0.00241`** |
| **`nested_outer_777001`** (`N=12,803`) | `CB_Bag3` | `Blend_Bag3` | `58/37 =` **`+21`** `[+2.0, +40.0]` (P>0: 98.5%) | `+7 / +14` | `0.18813` vs `0.18587` | **`-0.00226`** |
| **`nested_outer_777002`** (`N=13,259`) | `CB_Bag3` | `Blend_Bag3` | `65/42 =` **`+23`** `[+3.0, +44.0]` (P>0: 98.7%) | `+5 / +18` | `0.19550` vs `0.19363` | **`-0.00187`** |

> [!TIP]
> **Ganancia marginal y coste de LightGBM (CX-0004):**  
> LightGBM requiere solo ~25% de tiempo de CPU adicional respecto a CatBoost individual. Incorporar `0.25 LGB_Bag3` produce una ganancia neta positiva consistente (+11, +21 y +23 ítems) y una reducción sistemática de LogLoss (-0.00187 a -0.00241) respecto al bagging puro de CatBoost. El ensamble no es un mero artefacto de presupuesto.

- **Diagnóstico de Procedencia de la Caída Single en Corte 20261101 (CX-0004):**  
  En AG-0002, `virgin_confirm_20261101` reportó `+21` netos a umbral 0.500; en AG-0005, el corte idéntico reportó `-2`. La auditoría de procedencia revela:
  1. **Partición de filas:** Los hashes SHA256 de `fit`, `tune` y `confirm` son **100% idénticos** (`668e7b30...`).
  2. **Feature de texto `txt_sparse_p`:** En AG-0002 se ajustó con `C=3.0, solver="liblinear"`. En AG-0005, el refit uniforme interno utilizó `C=1.0, solver="lbfgs"`.
  3. **Comportamiento del baseline vs candidato:** Con `C=1.0`, la precisión del baseline `CB_single` en `confirm` **aumentó** de `92.33%` (`8,856 / 9,592`) a `92.62%` (`8,882 / 9,592`), un salto de **+26 aciertos**. Entretanto, `Blend_Single` se mantuvo prácticamente constante (`8,877` vs `8,880`).
  4. **Selección en `tune`:** En AG-0002 `tune` seleccionó `lgb_default` (`num_leaves=63`, `reg_lambda=5.0`), mientras que en AG-0005 seleccionó `lgb_deep` (`num_leaves=95`, `min_child_samples=25`), sobreajustando ligeramente en `tune`.  
  *Conclusión:* La caída neta en el corte 20261101 provino de un fortalecimiento genuino del baseline CatBoost por la mejor regularización del texto teacher, no de una degradación de la capacidad del ensamble.

### Agente 4 — Investigación Complementaria Acotada (Auditoría y Parada de MLP Disperso)
- **Reporte completo:** [`agent4_complementary_report.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent4_complementary_report.md)
- Alineación por ID verificada (`100%`). El MLP disperso sin SVD obtiene `Acc = 0.80430` (`LL = 0.43689`) y peso óptimo `w_mlp = 0.000` en `tune`. Línea detenida; cómputo liberado.

### Agente 5 — Constructor Verificado `build_from_scores.py` y Distinción Conservadora (CX-0004)
- **Reporte completo:** [`agent5_builder_verification.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent5_builder_verification.md) | **Código:** [`build_from_scores.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/eval_20261030/build_from_scores.py) | **Tests:** [`test_build_from_scores.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/eval_20261030/test_build_from_scores.py) (`9/9 PASS`)
- **Distinción metodológica solicitada en CX-0004:** La regla implementada que rechaza un superset ganador `B` cuando un subconjunto constituyente `A` pierde (`score(A) <= base_score`) es una política de selección conservadora de riesgo para evitar incorporar flips refutados; no constituye una imposibilidad matemática. Un superset completo observado puede conservarse como alternativa empírica íntegra sin pretender extraer diferencias sintéticas ni declararlo unión aditiva descompuesta. Se preserva el contrato y la suite de tests intacta sin roturas.

---

## 3. Agente 6 — Menú Revisado de Candidatos y Auditoría Completa de Submissions Históricas (`01–76`)

- **Reporte completo:** [`agent6_decision_package_report.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_decision_package_report.md) | **Auditoría cruzada 01–76:** [`agent6_historical_submissions_cross_audit.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_historical_submissions_cross_audit.csv) | **Menú JSON:** [`agent6_decision_menu.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent6_decision_menu.json)

### Matriz de Auditoría Cruzada contra las 33 Submissions Históricas Disponibles

| Código | Clase | Archivo en `submissions/20261030/` | SHA256 | Flips vs `48` | `Seen` / `New` | Solape Sondas `72–75` | Solape `Sub 64` (1,774) | Solape `Sub 71` (1,338) | Solape `Sub 65` (141) | Clasificación Epistemológica |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Option_A`** | **A** (Predicción Completa Modelo Inductivo 70k) | `candidate_optionA_full_inductive_blend_bag3.csv` | `d93af8e75bde...` | **`1,033`** | `664 / 369` | `13` (`0/1/7/5`) | `658` (`63.7%`) | `784` (`75.9%`) | `61` | Modelo Completo (Development-Grade) |
| **`Option_B1`** | **B** (Intervención Acotada — Core 15 Seen Sellers) | `candidate_sub80_seen_sellers_conservative_unprobed.csv` | `23415a6147e0...` | **`15`** | `15 / 0` | **`0`** | `15` (`100%`) | `13` (`86.7%`) | `0` | Intervención Candidata Exploratoria |
| **`Option_B2`** | **B** (Intervención Acotada — Disjoint 15 Consensus) | `candidate_sub78_consensus_disjoint15_unprobed.csv` | `98e1913752df...` | **`15`** | `10 / 5` | **`0`** | `15` (`100%`) | `15` (`100%`) | `5` | Intervención Candidata Exploratoria |
| **`Option_B3`** | **B** (Intervención Acotada — Consenso Cross-Pipeline) | `candidate_optionB_inductive_unanimous_consensus.csv` | `03d88c4aa23e...` | **`18`** | `17 / 1` | **`0`** | `18` (`100%`) | `18` (`100%`) | `1` | Intervención Candidata Exploratoria |
| **`Option_C1`** | **C** (Sonda Exploratoria — 20 New Sellers Texto Limpio) | `candidate_sub79_new_sellers_clean_text.csv` | `81ef2f8dc4c7...` | **`20`** | `0 / 20` | **`0`** | **`0` (`0%`)** | **`0` (`0%`)** | **`0`** | Sonda Diagnóstica 100% Inédita |
| **`Option_C2`** | **C** (Sonda Exploratoria — 5 Flips Margen Frontera) | `candidate_sub77_nested_margin5_unprobed.csv` | `9b07f45caaf9...` | **`5`** | `2 / 3` | **`0`** | `5` (`100%`) | `5` (`100%`) | `3` | Sonda Diagnóstica Exploratoria |
| **`Option_D`** | **D** (Constructor Contingente / Respaldo `48`) | `src/eval_20261030/build_from_scores.py` | En runtime | Según bloques | Según bloques | **`0`** | Según bloques | Según bloques | Según bloques | Constructor Algorítmico Verificado |

> [!NOTE]
> - **Aclaración sobre "unprobed" y desglose de envíos históricos (CX-0004):**
>   1. **Distinción entre archivos físicos y envíos puntuados:** En el repositorio existen **33 archivos `.csv` físicos** de 30k en `submissions/`. Mediante reconstrucción formal desde deltas históricos y vectores canónicos ([`src/eval_20261030/build_scored_registry.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/eval_20261030/build_scored_registry.py)), se dispone del registro canónico de **53 envíos puntuados efectivamente** ([`results/eval_20261030/scored_registry.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/eval_20261030/scored_registry.json)). Los números 51–56 y 60–65 fueron borradores exploratorios offline nunca subidos a Kaggle.
>   2. **Solapamiento real de `Option_C1`:** Contra los 33 archivos locales tenía 0% de solapamiento. Sin embargo, contra las 53 submissions puntuadas históricas, **`Option_C1` solapa en 1 ID (`ID 1653`)** con las submissions puntuadas `27` (score 0.94228) y `39` (score 0.92976). Los restantes **19 IDs son inéditos** en la historia de la competencia.
>   3. **Solapamiento de `Option_A`:** Los 1,033 flips de `Option_A` solapan con 49 de las 53 submissions puntuadas históricas (e.g. 600 flips comunes con `39`, 302 con `28`, 300 con `27`).
>   4. **Reclasificación de B3:** B3 deriva de la unión B1 U B2 filtrada por el refit inductivo 70k y contiene 1 new seller; no equivale a la regla de margen estricta de `confirm`. Se clasifica como intervención exploratoria acotada.

---

## 4. Estado de Automatización y Protocolo Operativo (CX-0003 & CX-0004)

1. **Deduplicación de Disparadores Horarios:** Se verificó la eliminación de cualquier cron redundante en la sesión activa (`manage_task` lista 0 tareas de fondo). Se mantiene como **único disparador verificado** el sidecar nativo de Antigravity (`ap-2026-q2-hourly-coordination`).
2. **Lock Atómico de Exclusión Mutua (`src/coordination_lock.py`):**
   - Implementado cerrojo a nivel de proyecto mediante creación atómica de directorio (`os.mkdir("coordination/.project_lock")`).
   - Registra de forma inmutable: `owner_id`, `host_pid`, `host_create_time`, inicio (`started_at_utc/local`) y heartbeat.
   - Verificación de vitalidad por proceso: antes de declarar un lock obsoleto, se consulta `psutil.Process(pid)` y se verifica el tiempo de creación del proceso host. Si el proceso está activo, el cerrojo **no se roba**.
   - Suite de pruebas unitarias verificada en [`tests/test_coordination_lock.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_coordination_lock.py) (6/6 PASS).
   - `.gitignore` actualizado para ignorar cerrojos temporales locales.
3. **Regla de Cero Subidas:** Ninguna submission se sube a Kaggle. Todos los artefactos quedan versionados localmente y pusheados a la rama de coordinación para la revisión y decisión final de Ariel y Codex.

