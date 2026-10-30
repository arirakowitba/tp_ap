# Decision Packet & Multi-Agent Consolidation (`2026-10-30`, Revisión CX-0008 a CX-0012)

**Rama de trabajo:** `antigravity/overnight-recovery-coordination-20261030`  
**En respuesta a:** [`CX-0001.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/codex_to_antigravity/CX-0001.md) a [`CX-0012.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/codex_to_antigravity/CX-0012.md) (`origin/main` `c15fdd86a60ea02bf3bab2d5cbe5ce78973b6977`)  
**Benchmark público demostrado:** `submission_48` = **`.94633`** (SHA256 canónico LF y archivo local: `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`, `submissions/20261030/submission_48_reference.csv`). *(Nota: la cadena `3f8d08e0a29ef...` que apareció en un reporte previo fue una errata de transcripción manual del prefijo `3f8d08e0...`; el único SHA256 real y verificado de `submission_48` y `submission_76` es `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`).*

---

## 1. Registro Canónico de Sondas Históricas (`72–76`) y Reglas de Integridad

| Submission | Score Público | Delta vs `48` (`.94633`) | Nº Flips vs `48` | Segmento / Dirección | Decisión |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **`72`** | `.94623` | `-0.00010` | `8` | `new_seller`, `used -> new` | Excluir bloque |
| **`73`** | `.94600` | `-0.00033` | `7` | `new_seller`, `new -> used` | Excluir bloque *(corregido: con 7 flips es imposible perder 10 ítems)* |
| **`74`** | `.94614` | `-0.00019` | `8` | `seen_seller`, mixto (`5 -> new, 3 -> used`) | Excluir bloque |
| **`75`** | `.94633` | `0.00000` | `5` | `seen_seller`, `new -> used` | Empate con `48` *(corregido: con 5 flips es imposible tener 10 fixes / 10 breaks)* |
| **`76`** | `.94633` | `0.00000` | `0` | Duplicado exacto de `48` (`3f8d08e0...`) | Prohibido repetir en el quinto cupo |

### 1.1 Corrección Crítica de Inventario Histórico (CX-0006)

Se corrigen de forma estricta las declaraciones erróneas previas sobre el estado de envíos históricos:
1. **Submissions 53, 54 y 55 SÍ fueron enviadas y puntuadas públicamente por Ariel:**
   - **`Sub 53`** (`submission_53_seller_loo_alpha1_on_48.csv`): score **`0.94114`**, SHA256 `a894af2ebabbe3166caf984842813b1a47d0504b181a5800198655e2b15813a8` (411 flips vs 48).
   - **`Sub 54`** (`submission_54_partition53_bloque1.csv`): score **`0.94457`**, SHA256 `5ac51b6926ac2d39264ae450bdc8bd8e0f99948ebf0ee893069bce04137debd5` (204 flips vs 48).
   - **`Sub 55`** (`submission_55_partition53_bloque2.csv`): score **`0.94514`**, SHA256 `fb82ba0afec97a166d4744cadd8967e028b46824204e67adac8336d8f9711eb5` (65 flips vs 48).
   - *Estado clasificado:* `enviado-score-conocido-hash-conocido-CSV-no-local` (CSVs conservados en Library; no presentes como archivos locales en disco).
2. **Submissions 51, 52 y 56:**
   - **`Sub 51`** (`submission_51_seller_loo_seen_on_48.csv`, SHA256 `03a19e21...`) y **`Sub 52`**: preparadas, **no enviadas** (descartadas formalmente tras la caída pública de 53).
   - **`Sub 56`** (`submission_56_partition53_bloque3.csv`, SHA256 `1ca4da09...`, 142 flips): preparada, **no enviada** (descartada tras resultados adversos de 54 y 55).
3. **Submissions 66–71:**
   - Preparadas en Ronda 152 sin scores públicos confirmados en los registros del repositorio. Estado verificado: `preparado-sin-score-confirmado-en-repo` (Ariel pasó directamente a ejecutar las sondas 72–76 el 2026-10-29).
4. **Desglose de las 5 Variantes de `fin_stack` (2026-10-28):**
   - **`59_full`** (`submission_59_fin_stack_banda_0.85_1.00_sobre48.csv`): score **`0.94576`**, SHA256 `b34723587e1ef09c897a4af5768a2ea025c0dd313c04bd6494d5f0289899f373` (67 flips vs 48).
   - **`probe_59_new_seller`** (`probe_59_new_seller_sobre48.csv`): score **`0.94619`**, SHA256 `de1ebd9af13ecc1723e5f521e1c0dcb525a2e9261267ae3edb7bafe4bc5e7b9f` (26 flips vs 48).
   - **`probe_59_seen_to_new`** (`probe_59_seen_to_new_sobre48.csv`): score **`0.94590`**, SHA256 `d06115e8984d4ae82c34cd0ac7276f7b16a6666d1eca57bea736aa5f7b7f32e7` (27 flips vs 48).
   - **`fin_stack_new150`** (`probe_fin_stack_new_unprobed150_sobre48.csv`): score **`0.94495`**, SHA256 `ebb365337aa674d5ce099b763b4917089348539bf1b3b8e5f509e7950cfcf67a` (150 flips vs 48).
   - **`fin_stack_seen150`** (`probe_fin_stack_seen_unprobed150_sobre48.csv`): score **`0.94257`**, SHA256 `37a8c519abff07ae8c64e360001229f3f3bf6953301908b38a946327ed00c832` (150 flips vs 48).
   - *Se prohíbe confundir estas variantes bajo la etiqueta única "59".*
5. **Envíos 01–03 y 10:** Scores conocidos en bitácora (`01: .94433, 02: .94419, 03: .94447, 10: .94500`), pero CSVs 30k exactos no identificados localmente. Estado: `enviado-score-conocido-CSV-no-identificado`.
6. **Aclaración sobre afirmaciones de novedad:** La afirmación de que ciertos IDs son "inéditos" se reformula estrictamente como **"sin intersección en vectores verificados disponibles en disco"**, reconociendo la falta de vectores fila a fila de 01–03, 10, 53, 54 y 55.

### 1.2 Mapa Completo de Procedencia (`31 -> 33 -> 41 -> 48`), Archivos Presentes/Faltantes y Proxies (CX-0008..CX-0012)

- **Regla estricta sobre scores públicos:** Los scores públicos son evidencia agregada sobre bloques enteros; **jamás** se convierten en etiquetas individuales por ID ni en conteos enteros exactos sin conocer el denominador público exacto.
- **Reconstrucción exacta de 66 submissions históricas de 30,000 filas ([`provenance_map_31_33_41_48.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/provenance_map_31_33_41_48.json)):**
  - Combinando `base{07,15,33}_labels_1_to_30000.txt`, `flips_04_09.tsv`, `flips_11_14_27_35.tsv`, `flips_16_26.tsv`, `flips_36_41.tsv`, `fasttext_submission_batch_manifest.json` (`42, 43`), `batch_41_44_direction_probe_20261025.md` (`44`), `submission_45_gate8_new_sellers_20261025.md` (`45`), `submissions_46_47_consensus_20261026.md` (`46, 47`), `submission_49_exploration_20261026.md` (`49`), `submission_50_final_probe_20261026.md` (`50`), `submissions/20261027/` (`57, 58`), `submissions/20261028/` (`59..71`) y `submissions/20261029/` (`72..76`), se reconstruyen byte a byte **66 de las 76 submissions (`04..09`, `11..50`, `57..76`)**.
  - Sólo carecen de vector completo de 30,000 filas en el repositorio **10 submissions**: `[1, 2, 3, 10, 51, 52, 53, 54, 55, 56]`.
- **Cadena exacta `31 -> 33 -> 41 -> 48`:**
  1. **`31` (`.94585`) -> `33` (`.94614`):** `31` cambios en test (`19 new->used`, `12 used->new`; `19` vendedores nuevos y `12` vistos). Según [`round38_dart_group_20261025.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/revision_submissions/round38_dart_group_20261025.md) y [`round44_transfer_head_proxy33_20261025.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/revision_submissions/round44_transfer_head_proxy33_20261025.md), no existe en los artefactos preservados una función determinista que traslade esos 31 cambios manuales de test a train.
  2. **`33` (`.94614`) -> `41` (`.94628`):** `6` cambios en vendedores nuevos de test (`new->used`: `1219, 8862, 22148`; `used->new`: `18412, 26663, 26668`).
  3. **`41` (`.94628`) -> `48` (`.94633`):** `2` cambios `used->new` en vendedores nuevos de test (`13583` y `25426`), subconjunto de mayor confianza mínima de `47` (`8532, 13583, 22635, 25426`).
- **Archivos presentes vs. faltantes de la arquitectura Round 30/31 (`text_deep_cat_oof_fresh` / `text_deep_oof`):**
  - *Código fuente preservado en Git:* [`src/round31_seller_balanced_source.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/round31_seller_balanced_source.py) y [`src/round31_seller_balanced_group.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/round31_seller_balanced_group.py).
  - *Archivos externos referenciados por `round31_seller_balanced_source.py` que NO están en Git:* `gemini_bundle/gemini_full_compact_100k.csv`, `recovery/extracted/tp_ap/results/training_round2/deep_teacher/features.pkl`, `recovery/extracted/results/training_round2/fresh/partitions.npz`, y `tp_ap/results/training_round5/text_deep_cat_oof_fresh/test.npz` (que difería de `33` en 558 filas y de `41` en 552 filas = 98.16% acuerdo).
  - *Señales recuperables desde `round31_seller_balanced_source.py` sobre `data/MLA_100k.jsonlines` (compartido para coordinación con Claude `CC-0001`):* (a) TE OOF `alpha=20` + `log1p(count)` sobre 8 claves base + 3 interacciones (`seller_id|category_id`, `seller_id|listing_type_id`, `category_id|listing_type_id`); (b) teachers OOF `LogisticRegression(C=2.0, solver='liblinear')` sobre `word(1,2)`, `char_wb(2,5)` y `extra_docs` (`attributes` + `variations`); (c) features temporales de IDs de fotos `pictures[*]['id']` (`img_oldest_month`, `img_newest_month`, `img_age_months`, `img_date_spread`, `img_first_prefix_log`); (d) ponderación por frecuencia de vendedor `np.clip(1/sqrt(seller_freq), 0.2, 1.0)`.
- **Concordancia en Test (`30,000` filas) de los Modelos Completos Preservados en el Repo vs `33`, `41` y `48`:**
  - `Blend075_Bag3_inductive_70k` (`Option_A`): **96.557% de acuerdo con `48`** (`1,033` discrepancias vs `48`: `664 seen`, `369 new`; `1,039` vs `33`; coincide con `6/6` de los flips `33->41`!).
  - `A_hist_inductive_70k` (`CatBoost Single`): **96.397% de acuerdo con `48`** (`1,081` discrepancias vs `48`: `705 seen`, `376 new`).
  - `CB_Bag3_inductive_70k`: **96.393% de acuerdo con `48`** (`1,082` discrepancias vs `48`: `700 seen`, `382 new`).
  - `LGB_Bag3_inductive_70k`: **96.100% de acuerdo con `48`** (`1,170` discrepancias vs `48`: `715 seen`, `455 new`).
  - `R156_Strong_Ensemble_70k`: **95.623% de acuerdo con `48`** (`1,313` discrepancias vs `48`).
  - `R155_Dual_CatBoost_70k`: **94.087% de acuerdo con `48`** (`1,774` discrepancias vs `48`).

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

## 3. Agente 6 — Manifiesto Único de Candidatos y Auditoría Completa de 66 Submissions Históricas (`01–76`)

- **Manifiesto único JSON:** [`unified_candidate_manifest_20261030.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/unified_candidate_manifest_20261030.json) | **Auditoría `B4` JSON:** [`single_intervention_rule_report.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/single_intervention_rule_report.json)

### 3.1 Tabla Única de los 10 Archivos en `submissions/20261030/` (IDs `1..30000` Verificados)

| Código Canónico | Archivo en `submissions/20261030/` | SHA256 Archivo (`raw`) | SHA256 Canónico (`LF`) | Flips vs `48` | `to_used` / `to_new` | `Seen` / `New` | Solape `72–75` | Solape `Sub 64` (1,774) | Solape `Sub 71` (1,338) | Clasificación Epistemológica |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Reference_48`** | `submission_48_reference.csv` | `3f8d08e0e618...` | `3f8d08e0e618...` | **`0`** | `0 / 0` | `0 / 0` | `0` | `0` | `0` | **Respaldo de Seguridad (`.94633`)** |
| **`Option_A`** | `candidate_optionA_full_inductive_blend_bag3.csv` | `d93af8e75bde...` | `d93af8e75bde...` | **`1,033`** | `559 / 474` | `664 / 369` | `13` (`0/1/7/5`) | `658` (`63.7%`) | `784` (`75.9%`) | Modelo Completo Inductivo 70k (Development-Grade) |
| **`Option_B1`** | `candidate_sub80_seen_sellers_conservative_unprobed.csv` | `23415a6147e0...` | `643b1b55c52f...` | **`15`** | `6 / 9` | `15 / 0` | **`0`** | `15` (`100%`) | `13` (`86.7%`) | Intervención Candidata Exploratoria |
| **`Option_B2`** | `candidate_sub78_consensus_disjoint15_unprobed.csv` | `98e1913752df...` | `4b13b67e5b8e...` | **`15`** | `5 / 10` | `10 / 5` | **`0`** | `15` (`100%`) | `15` (`100%`) | Intervención Candidata Exploratoria |
| **`Option_B3`** | `candidate_optionB_inductive_unanimous_consensus.csv` | `03d88c4aa23e...` | `03d88c4aa23e...` | **`18`** | `8 / 10` | `17 / 1` | **`0`** | `18` (`100%`) | `18` (`100%`) | Intervención Exploratoria (Desclasificada CX-0002) |
| **`Option_B4`** | `candidate_single_intervention_rule_on_48.csv` | `51ea764c8b69...` | `51ea764c8b69...` | **`453`** | `257 / 196` | `453 / 0` | `12` (`0/0/7/5`) | `332` (`73.3%`) | `390` (`86.1%`) | **Sonda de Transferencia No Validada vs `48` (NO Aprobada)** |
| **`Option_C1`** | `candidate_sub79_new_sellers_clean_text.csv` | `81ef2f8dc4c7...` | `6a75fd3389c2...` | **`20`** | `2 / 18` | `0 / 20` | **`0`** | **`0` (`0%`)** | **`0` (`0%`)** | Sonda Diagnóstica New-Seller (1 ID en `27/39`; 0 en `57–75`) |
| **`Option_C2`** | `candidate_sub77_nested_margin5_unprobed.csv` | `9b07f45caaf9...` | `e0890d0b7e6b...` | **`5`** | `2 / 3` | `2 / 3` | **`0`** | `5` (`100%`) | `5` (`100%`) | Sonda Diagnóstica Exploratoria |
| *`Legacy_77`* | `candidate_sub77_dual_cb_high_margin_unprobed.csv` | `a1eb6c51c07f...` | `3770adeea626...` | `20` | `8 / 12` | `17 / 3` | `0` | `20` | `18` | **OBSOLETO — Reemplazado por `Option_B1 + Option_C2`** |
| *`Legacy_78`* | `candidate_sub78_model_consensus_unprobed.csv` | `d70148d30eca...` | `e00f535b1e79...` | `25` | `8 / 17` | `19 / 6` | `0` | `25` | `25` | **OBSOLETO — Reemplazado por `Option_B2`** |

### 3.2 Cierre de la Auditoría de `Option_B4` sin Confundir Modelos (`CX-0008`, `CX-0009`, `CX-0011`, `CX-0012`)

1. **Corrección de Paridad de Modelo (`LGB_Bag3` en `confirm` y `test`):**
   - Se corrigió el defecto detectado en `CX-0011` donde `evaluate_rule_on_confirm()` leía `p_LGB_hist` (1 semilla) mientras que `apply_rule_to_test()` leía `p_lgb_bag3` (3 semillas). Ambos usan ahora exactamente `CB_Bag3`, `LGB_Bag3` y `Blend075_Bag3 = 0.75 * CB_Bag3 + 0.25 * LGB_Bag3` a umbral `0.50` y margen `|p_blend - 0.50| >= 0.06` sobre `is_seen == 1`.
   - Se verificó también que `candidate_single_intervention_rule_on_48.csv` ya poseía en disco `ID = 1..30000` y SHA256 `51ea764c8b69745001eed43f33f79ad960ca8553707294f69e4086d21d9da534` (la mención a `0..29999` en `AG-0007` fue un error de redacción del texto, no del archivo).
2. **Descomposición Cuantitativa de la Tasa de Intervención por cada `1,000` Filas `Seen` (Por Qué `453` Flips en Test vs `Confirm`):**
   - **Prohibido sumar semillas solapadas:** Las 3 semillas de `confirm` (`2026`, `314159`, `20261030`) solapan ~18.1%–18.6% entre sí; jamás se suman sus conteos como observaciones independientes.

| Dominio y Corte | Referencia Base (`pred_base`) | Variante LGB | Filas `Seen` | Flips | Tasa / `1,000 Seen` | `to_used` / `to_new` | Fixes / Breaks (**Neto**, IC 95%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Confirm 2026`** (Dentro de familia) | `A_hist_tuned` (`th=0.470`) | `LGB_Bag3` *(LGB_single)* | `3,903` | **`15`** *(12)* | **`3.84 / 1k`** *(3.07)* | `12 / 3` | `11 / 4 =` **`+7`** `[0.0, +14.0]` *(+6)* |
| **`Confirm 314159`** (Dentro de familia) | `A_hist_tuned` (`th=0.535`) | `LGB_Bag3` *(LGB_single)* | `3,838` | **`4`** *(4)* | **`1.04 / 1k`** *(1.04)* | `2 / 2` | `3 / 1 =` **`+2`** `[-2.0, +6.0]` *(+2)* |
| **`Confirm 20261030`** (Dentro de familia) | `A_hist_tuned` (`th=0.515`) | `LGB_Bag3` *(LGB_single)* | `3,740` | **`9`** *(8)* | **`2.41 / 1k`** *(2.14)* | `0 / 9` | `5 / 4 =` **`+1`** `[-5.0, +7.0]` *(+2)* |
| **`Confirm 2026`** (Dentro de familia) | `A_hist@0.50` | `LGB_Bag3` | `3,903` | **`14`** | **`3.59 / 1k`** | `8 / 6` | `11 / 3 =` **`+8`** `[+1.0, +15.0]` |
| **`Confirm 314159`** (Dentro de familia) | `A_hist@0.50` | `LGB_Bag3` | `3,838` | **`2`** | **`0.52 / 1k`** | `2 / 0` | `2 / 0 =` **`+2`** `[0.0, +4.0]` |
| **`Confirm 20261030`** (Dentro de familia) | `A_hist@0.50` | `LGB_Bag3` | `3,740` | **`2`** | **`0.53 / 1k`** | `0 / 2` | `1 / 1 =` **`+0`** `[-2.0, +2.0]` |
| **`Test 30k`** (**Dentro de familia**) | **`A_hist_inductive_70k@0.50`** | **`LGB_Bag3`** *(LGB_single)* | **`21,487`** | **`26`** *(26)* | **`1.21 / 1k`** *(1.21)* | `8 / 18` | *Sin etiquetas de test* |
| **`Confirm 2026`** (Entre familias) | `R156_Ensemble` | `LGB_Bag3` | `3,903` | **`87`** | **`22.29 / 1k`** | `42 / 45` | `72 / 15 =` **`+57`** `[+38.0, +78.0]` |
| **`Confirm 314159`** (Entre familias) | `R156_Ensemble` | `LGB_Bag3` | `3,838` | **`82`** | **`21.37 / 1k`** | `39 / 43` | `64 / 18 =` **`+46`** `[+28.0, +65.0]` |
| **`Test 30k`** (**Entre familias vs 48**) | **`submission_48`** (`.94633`) | **`LGB_Bag3`** *(LGB_single)* | **`21,487`** | **`453`** *(451)* | **`21.08 / 1k`** *(20.99)* | `257 / 196` | *Sonda de transferencia no validada vs 48* |

   - **Explicación demostrada:** Cuando la regla congelada se aplica en `test` (`21,487` filas `seen`) contra **su propia familia de referencia (`A_hist_inductive_70k@0.50`)**, produce únicamente **`26` flips (`1.21 / 1,000` filas `seen`)**, exactamente dentro del rango observado en `confirm` (`0.52` a `3.84 / 1,000`). El salto a **`453` flips (`21.08 / 1,000` filas `seen`)** ocurre exclusivamente al cambiar la referencia de `A_hist` a `submission_48` (que pertenece a otra familia histórica de modelos y discrepa de `A_hist_inductive_70k` en `705` filas `seen` = `32.81 / 1,000`). En `confirm`, aplicar la misma regla contra otra familia (`R156_Ensemble`) produce una tasa casi idéntica (`22.29` y `21.37 / 1,000`). Como ni `A_hist` ni `R156_Ensemble` son `submission_48`, **`Option_B4` permanece NO aprobada como mejora demostrada**.

---

## 4. Estado de Automatización, Concurrencia Real y Protocolo Operativo (CX-0008, CX-0010, CX-0011 & CX-0012)

1. **Corrección de la Race Condition de `CX-0010` en [`src/coordination_lock.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/coordination_lock.py):**
   - Mantiene un bloqueo exclusivo de descriptor de archivo del sistema operativo (`msvcrt.locking` en Windows / `fcntl.flock` en POSIX sobre `.project_lock.oslock`) abierto durante **toda** la sección crítica, más registro en memoria por `instance_token` UUID.
   - Si un proceso muere o cae abruptamente (`kill`), el kernel del SO libera automáticamente el bloqueo de archivo.
   - **Prohibido borrar directorios vacíos en inicialización:** Si `lock_info.json` aún no existe y el directorio está dentro de su ventana de gracia (`empty_dir_stale_seconds`), ningún contendiente puede borrarlo. Además, `lock_info.json` se escribe atómicamente mediante archivo temporal + `os.replace`, y `release()` valida `self._acquired`, `host_pid`, `owner_id` e `instance_token`.
2. **Runner Sincronizado [`src/run_coordination_check.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/run_coordination_check.py):**
   - Inspecciona mensajes remotos directamente desde `origin/main` mediante `git ls-tree origin/main coordination/codex_to_antigravity/` y `git show origin/main:...` (`utf-8`), sin depender del árbol de trabajo local tras `git fetch`.
   - Ejecuta el comando científico autorizado (`--task-cmd`, `--task-id`) **dentro** de la sección crítica protegida por `ProjectLock`, emitiendo heartbeats periódicos, evitando ejecuciones duplicadas de un mismo `task_id`, y registrando inicio, fin, HEADs observados y código de salida en `results/intensive_20261030/coordination_runner_log.jsonl`.
3. **Suite de Pruebas Unitarias y Multiproceso ([`tests/test_coordination_lock.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_coordination_lock.py) y [`tests/test_single_intervention_rule.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_single_intervention_rule.py)):**
   - `11/11` tests en `test_coordination_lock.py` (`OK`), incluyendo la regresión determinística exacta de `CX-0010`, contención multiproceso real + recuperación tras `proc.kill()`, y dos procesos concurrentes de `run_coordination_check.py` ejecutando una sección crítica real.
   - `1/1` test en `test_single_intervention_rule.py` (`OK`), verificando que si `submission_guard` falla, el archivo temporal se elimina y no se escribe ningún CSV final.
4. **Regla de Cero Subidas:** Ninguna submission se sube a Kaggle. `submission_48` (`.94633`) se mantiene como respaldo obligatorio.

