# Agente 3 — Validación Externa Estrictamente Inductiva, Auditoría Exhaustiva de Holdouts Históricos y Transporte a Mezcla de Test

**Fecha:** 2026-10-30 01:19:00  
**Tiempo total de ejecución:** 775.66s (entrenamientos inductivos) + 60.1s (auditoría exhaustiva de holdouts)  
**Clasificación Epistemológica:** `DEVELOPMENT_GRADE_REUSED_HOLDOUT_POOL` (Evidencia de desarrollo inductiva con mejora consistente en Log Loss y ganancia moderada en Accuracy; **no** constituye confirmación virgen independiente).  
**Artefactos verificados:**
- [`agent3_external_summary.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent3_external_summary.json)
- [`all_historical_holdouts_overlap_audit.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/all_historical_holdouts_overlap_audit.json)
- [`all_historical_holdouts_overlap_audit.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/all_historical_holdouts_overlap_audit.csv)

> [!WARNING]
> **Ningún corte sobre las 70,000 filas de `train` es “virgen”.** Aunque el corte `disjoint_3seed_20261101` (`paired_external_virgin_confirm_20261101.csv.gz`) fue construido para tener **0.0% de solapamiento de filas** con la unión de los tres conjuntos `confirm` recientes (`R151_2026_confirm U R151_314159_confirm U R151_20261030_confirm`), la auditoría exhaustiva contra **todos** los holdouts históricos del repositorio demuestra que:
> 1. Solapa en **53.49% de sus filas (`5,131 / 9,592`)** con la unión de `tune + confirm` de las tres semillas R151 (`2026`, `314159`, `20261030`).
> 2. Solapa en **92.39% de sus filas (`8,862 / 9,592`)** y **95.30% de sus vendedores (`6,143 / 6,446`)** con la unión de los 18 holdouts históricos reconstruibles en el repositorio (`67,007 / 70,000` filas totales = `95.72%` del train).
> 3. Según las auditorías históricas [`round61_validation_coverage_20261026.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/revision_submissions/round61_validation_coverage_20261026.md) y [`round62_historical_validation_coverage_20261026.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/revision_submissions/round62_historical_validation_coverage_20261026.md), al incluir además los archivos históricos de `round11_12` y `round13`, **sólo quedan 1,897 filas de 1,647 vendedores** fuera de los holdouts anteriores.
> Por lo tanto, **se prohíbe llamar “virgen” a cualquier corte de ~10k–13k filas sobre `train`**: todos estos cortes evalúan generalización inductiva libre de fuga (*zero leakage* entre `fit`, `tune` y `confirm` dentro de cada realización), pero sobre un pool histórico reutilizado.

---

## 1. Auditoría Exhaustiva de Solapamiento contra TODOS los Holdouts Históricos del Repositorio

En [`audit_all_historical_holdouts.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/intensive_20261030/audit_all_historical_holdouts.py) cruzamos los 6 cortes evaluados por Agente 2 y Agente 3 contra las **18 particiones históricas** de `tune`, `confirm`, `selection` y `holdout` preservadas o reconstruibles en el repositorio:

| Holdout Histórico en el Repositorio | `N_hist` Filas (`Vendedores`) | `R151_2026_confirm` (`12,905`) | `R151_314159_confirm` (`12,843`) | `R151_20261030_confirm` (`12,881`) | `disjoint_3seed_20261101` (`9,592`) | `nested_outer_777001` (`12,803`) | `nested_outer_777002` (`13,259`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R151_seed2026_tune`** | `11,765` (`6,533`) | `0` (`0.0%`) | `2,179` (`17.0%`) | `2,122` (`16.5%`) | `2,103` (`21.9%`) | `2,203` (`17.2%`) | `2,195` (`16.6%`) |
| **`R151_seed2026_confirm`** | `12,905` (`6,649`) | `12,905` (`100%`) | `2,337` (`18.2%`) | `2,393` (`18.6%`) | **`0` (`0.0%`)** | `2,413` (`18.9%`) | `2,625` (`19.8%`) |
| **`R151_seed314159_tune`** | `12,473` (`6,573`) | `2,507` (`19.4%`) | `0` (`0.0%`) | `2,147` (`16.7%`) | `2,169` (`22.6%`) | `2,187` (`17.1%`) | `2,455` (`18.5%`) |
| **`R151_seed314159_confirm`** | `12,843` (`6,642`) | `2,337` (`18.1%`) | `12,843` (`100%`) | `2,249` (`17.5%`) | **`0` (`0.0%`)** | `2,264` (`17.7%`) | `2,360` (`17.8%`) |
| **`R151_seed20261030_tune`** | `13,451` (`6,573`) | `2,257` (`17.5%`) | `2,526` (`19.7%`) | `0` (`0.0%`) | `2,250` (`23.5%`) | `2,376` (`18.6%`) | `2,737` (`20.6%`) |
| **`R151_seed20261030_confirm`** | `12,881` (`6,586`) | `2,393` (`18.5%`) | `2,249` (`17.5%`) | `12,881` (`100%`) | **`0` (`0.0%`)** | `2,105` (`16.4%`) | `2,745` (`20.7%`) |
| **`R157_seed2026_tune`** | `11,765` (`6,547`) | `462` (`3.6%`) | `2,163` (`16.8%`) | `2,105` (`16.3%`) | `2,005` (`20.9%`) | `2,186` (`17.1%`) | `2,159` (`16.3%`) |
| **`R157_seed2026_confirm`** | `12,906` (`6,656`) | `8,850` (`68.6%`) | `2,300` (`17.9%`) | `2,382` (`18.5%`) | `685` (`7.1%`) | `2,410` (`18.8%`) | `2,597` (`19.6%`) |
| **`R157_seed314159_tune`** | `12,473` (`6,602`) | `2,487` (`19.3%`) | `688` (`5.4%`) | `2,137` (`16.6%`) | `2,100` (`21.9%`) | `2,207` (`17.2%`) | `2,489` (`18.8%`) |
| **`R157_seed314159_confirm`** | `12,844` (`6,614`) | `2,238` (`17.3%`) | `9,328` (`72.6%`) | `2,293` (`17.8%`) | `625` (`6.5%`) | `2,309` (`18.0%`) | `2,332` (`17.6%`) |
| **`SellerSplit_seed46077_tune` (R47)** | `12,534` (`6,637`) | `2,486` (`19.3%`) | `2,256` (`17.6%`) | `2,210` (`17.2%`) | `1,698` (`17.7%`) | `2,262` (`17.7%`) | `2,312` (`17.4%`) |
| **`SellerSplit_seed46077_confirm` (R47)** | `12,327` (`6,575`) | `2,143` (`16.6%`) | `2,363` (`18.4%`) | `2,168` (`16.8%`) | `1,729` (`18.0%`) | `2,224` (`17.4%`) | `2,365` (`17.8%`) |
| **`SellerSplit_seed42_tune`** | `12,639` (`6,599`) | `2,457` (`19.0%`) | `2,276` (`17.7%`) | `2,121` (`16.5%`) | `1,730` (`18.0%`) | `2,235` (`17.5%`) | `2,649` (`20.0%`) |
| **`SellerSplit_seed42_confirm`** | `12,977` (`6,619`) | `2,296` (`17.8%`) | `2,375` (`18.5%`) | `2,278` (`17.7%`) | `1,744` (`18.2%`) | `2,195` (`17.1%`) | `2,368` (`17.9%`) |
| **`round_v3_confirm_seed731` (CSV)** | `14,000` (`8,357`) | `2,581` (`20.0%`) | `2,549` (`19.9%`) | `2,588` (`20.1%`) | `1,912` (`19.9%`) | `2,625` (`20.5%`) | `2,607` (`19.7%`) |
| **`round_v5_selection_seed42` (CSV)** | `14,000` (`8,368`) | `2,651` (`20.5%`) | `2,613` (`20.4%`) | `2,605` (`20.2%`) | `1,888` (`19.7%`) | `2,583` (`20.2%`) | `2,711` (`20.5%`) |
| **`optimize_leaderboard_seed42_h3`** | `21,000` (`11,458`) | `3,994` (`31.0%`) | `3,907` (`30.4%`) | `3,894` (`30.2%`) | `2,866` (`29.9%`) | `3,855` (`30.1%`) | `4,035` (`30.4%`) |
| **`UNION_R151_3Seeds_Confirm`** | `32,060` (`14,618`) | `12,905` (`100%`) | `12,843` (`100%`) | `12,881` (`100%`) | **`0` (`0.0%`)** | `5,665` (`44.3%`) | `6,443` (`48.6%`) |
| **`UNION_R151_3Seeds_Tune+Confirm`** | `52,035` (`22,069`) | `12,905` (`100%`) | `12,843` (`100%`) | `12,881` (`100%`) | **`5,131` (`53.5%`)** | `9,408` (`73.5%`) | `10,234` (`77.2%`) |
| **`UNION_All_Reconstructed_Holdouts`** | `67,007` (`27,008`) | `12,905` (`100%`) | `12,843` (`100%`) | `12,881` (`100%`) | **`8,862` (`92.4%`)** | `12,243` (`95.6%`) | `12,762` (`96.3%`) |

---

## 2. Tamaños, Hashes y Proporciones `Seen / New` de los Cortes Externos Estrictamente Inductivos

Dentro de cada uno de los tres cortes externos, **toda** transformación supervisada o agregada por grupo (`compute_inductive_group_features`, `compute_oof_te`, `compute_oof_text_sparse_nb`, selección de variante LightGBM y selección de umbral en `tune`) se ajustó **exclusivamente dentro de `fit` / `tune`**, sin acceder jamás a `confirm`:

| Corte Externo (No-Virgen Global) | Semilla | `N_fit` | `N_tune` | `N_confirm` | `Seen` Confirm | `New` Confirm | SHA256 Split | SHA256 Paired CSV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`disjoint_3seed_20261101`** *(antes `virgin_confirm_20261101`)* | `20261101` | 47,157 | 13,251 | 9,592 | 4,202 (43.81%) | 5,390 (56.19%) | `668e7b306bf2...` | `20590fd534c0...` |
| **`nested_outer_777001`** | `777001` | 44,415 | 12,782 | 12,803 | 3,836 (29.96%) | 8,967 (70.04%) | `b5915cf53418...` | `9a5754225988...` |
| **`nested_outer_777002`** | `777002` | 44,158 | 12,583 | 13,259 | 3,815 (28.77%) | 9,444 (71.23%) | `fe7d0ea5ee77...` | `9aaa7adbd7ef...` |

---

## 3. Resultados Observados vs. Estimación Transportada a la Mezcla `Seen/New` de Test (`71.623% Seen / 28.377% New`)

| Corte Externo | Régimen Umbral | Acc Base | Acc Blend | Fixes | Breaks | Neto Obs | IC 95% Vendedor Obs | Seen Net | New Net | $\Delta$ Acc Transportado | Neto Esperado 30k Test | IC 95% Transportado 30k | $\Delta$ Log Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`disjoint_3seed_20261101`** | Común 0.500 | 0.92327 | 0.92546 | 55 | 34 | **+21** | [+2.0, +40.0] | +2 | +19 | `+0.00134` | **+40.2** | [-11.5, +95.7] | `-0.00323` |
| **`disjoint_3seed_20261101`** | Política tune | 0.92389 | 0.92546 | 51 | 36 | **+15** | [-4.0, +33.0] | +1 | +14 | `+0.00091` | **+27.2** | [-34.4, +89.2] | `-0.00323` |
| **`nested_outer_777001`** | Común 0.500 | 0.92463 | 0.92720 | 74 | 41 | **+33** | [+10.0, +54.0] | +8 | +25 | `+0.00228` | **+68.5** | [+18.2, +117.9] | `-0.00457` |
| **`nested_outer_777001`** | Política tune | 0.92463 | 0.92697 | 70 | 40 | **+30** | [+9.0, +51.0] | +3 | +27 | `+0.00141` | **+42.4** | [-7.3, +93.4] | `-0.00457` |
| **`nested_outer_777002`** | Común 0.500 | 0.92601 | 0.92684 | 61 | 50 | **+11** | [-10.0, +32.0] | +4 | +7 | `+0.00096` | **+28.8** | [-30.8, +87.8] | `-0.00270` |
| **`nested_outer_777002`** | Política tune | 0.92647 | 0.92714 | 53 | 44 | **+9** | [-11.0, +28.0] | +0 | +9 | `+0.00027` | **+8.1** | [-36.0, +53.5] | `-0.00270` |

---

## 4. Supuestos Explícitos de Transportabilidad y Análisis de Regresiones

### Supuestos del estimador transportado a la mezcla de test:
1. **Invarianza condicional por estrato:** $\mathbb{E}[\Delta \text{Acc} \mid \text{estrato}, \text{test}] = \mathbb{E}[\Delta \text{Acc} \mid \text{estrato}, \text{confirm}]$ para $\text{estrato} \in \{\text{seen}, \text{new}\}$.
2. **Reponderación de mezcla:** En los cortes de validación `confirm`, la mezcla es `~30% seen / ~70% new` (`43.8% / 56.2%` en `disjoint_3seed_20261101`), mientras que en los `30,000` ítems de test la mezcla observada sin usar etiquetas es **`71.623% seen (21,487) / 28.377% new (8,513)`**.
3. **Sensibilidad del umbral al cambio de mezcla:** Un umbral escalar único optimizado sobre `tune` (`30% seen / 70% new`) está sesgado hacia la distribución de `new_sellers` y pierde eficiencia al transportarse a una mezcla `71.6% seen` (por ejemplo, en `nested_outer_777002` el neto transportado cae de `+28.8` con umbral común `0.500` a `+8.1` con umbral de `tune`).

### Análisis de regresiones por banda de margen (`|p_base - 0.50|`):

| Corte Externo | Banda de Margen `|p_base - 0.50|` | Filas | Fixes | Breaks | Neto |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `disjoint_3seed_20261101` | `[0.00, 0.03)` | 119 | 31 | 23 | **+8** |
| `disjoint_3seed_20261101` | `[0.03, 0.08)` | 183 | 19 | 7 | **+12** |
| `disjoint_3seed_20261101` | `[0.08, 0.15)` | 300 | 5 | 4 | **+1** |
| `disjoint_3seed_20261101` | `[0.15, 1.00)` | 8,990 | 0 | 0 | **+0** |
| `nested_outer_777001` | `[0.00, 0.03)` | 151 | 38 | 28 | **+10** |
| `nested_outer_777001` | `[0.03, 0.08)` | 248 | 30 | 11 | **+19** |
| `nested_outer_777001` | `[0.08, 0.15)` | 412 | 6 | 2 | **+4** |
| `nested_outer_777001` | `[0.15, 1.00)` | 11,992 | 0 | 0 | **+0** |
| `nested_outer_777002` | `[0.00, 0.03)` | 162 | 41 | 27 | **+14** |
| `nested_outer_777002` | `[0.03, 0.08)` | 266 | 20 | 20 | **+0** |
| `nested_outer_777002` | `[0.08, 0.15)` | 398 | 0 | 3 | **-3** |
| `nested_outer_777002` | `[0.15, 1.00)` | 12,433 | 0 | 0 | **+0** |
