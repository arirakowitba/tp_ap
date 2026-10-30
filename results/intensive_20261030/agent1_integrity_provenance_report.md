# Agente 1 — Reporte de Integridad, Procedencia y Reproducción Exacta vs. Variante Nueva

**Fecha:** 2026-10-30  
**Rama Activa:** `antigravity/overnight-recovery-coordination-20261030`  
**Manifiesto Verificado:** [`results/intensive_20261030/agent1_integrity_manifest.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/agent1_integrity_manifest.json) / [`results/benchmark_20261030/reproduction_manifest_agent2.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/benchmark_20261030/reproduction_manifest_agent2.json)

---

## 1. Resolución Definitiva de la Contradicción entre `BENCHMARK_REPORT.md` y `reproduction_manifest_agent2.json`

### 1.1 Diagnóstico de la Contradicción
Existía una discrepancia aparente entre dos documentos del repositorio:
- **`BENCHMARK_REPORT.md`** afirmaba una reproducción 100.00% exacta del baseline histórico R151 (`Confirm Acc = 0.92391` en Semilla `2026` y `0.92019` en Semilla `314159`, correlación `1.000000`).
- **`reproduction_manifest_agent2.json`** registraba ~`98.35%` de acuerdo de predicción (`max_abs_diff = 0.531`, `Confirm Acc = 0.92491` en Semilla `2026` y `0.91809` en Semilla `314159`).

### 1.2 Causa Raíz Exacta Verificada en Código
Al auditar línea por línea los scripts que generaron ambos artefactos, demostramos que **no se trataba de no-determinismo sino de dos definiciones distintas del texto de entrada (`combos`) en la feature `txt_sparse_p`**, junto con el tipo de punto flotante (`float64` vs `float32`) y `thread_count`:

1. **`Pipeline A_hist` (Reproducción Exacta del Pipeline Histórico R151):**
   - Definido en [`src/revision_submissions/deep_catboost_nbsvm_ablation_20261029.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/revision_submissions/deep_catboost_nbsvm_ablation_20261029.py#L161-L165) y ejecutado originalmente por [`src/benchmark_20261030/audit_attribution_gap.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/benchmark_20261030/audit_attribution_gap.py).
   - Construye el texto combinado concatenando los **IDs de atributo** (`a.get("id")`) separados por `||`:
     ```python
     attrs = " ".join(str(a.get("id", "")) for a in lst(r.get("attributes")))
     combos.append(f"{t} || {w} || {attrs}")
     ```
   - Al evaluar contra los archivos históricos [`paired_deep_ablation_seed_2026.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/revision_submissions/paired_deep_ablation_seed_2026.csv) y [`paired_deep_ablation_seed_314159.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/revision_submissions/paired_deep_ablation_seed_314159.csv) (guardados a 5 decimales):
     - **Semilla 2026:** `max_abs_diff = 4.9998e-6` (**`0.000000` tras redondear a 5 decimales**), acuerdo **`100.0000%`**, `Confirm Acc = 0.92391`, `LL = 0.18764` (`th_tune = 0.470`).
     - **Semilla 314159 (`preds_A_seed314159_confirm.csv.gz` de `6a71f6c`, `thread_count=8`):** `max_abs_diff = 4.99998e-6` (**`0.000000` tras redondear a 5 decimales**), acuerdo **`100.0000%`**, `Confirm Acc = 0.92019`, `LL = 0.19558` (`th_tune = 0.535`). Con `thread_count=4` (limitado por presupuesto multiproceso), `A_hist` obtiene `99.5095%` de acuerdo (`r = 0.999625`, `Acc = 0.91996`, `LL = 0.19552`, mismo `th_tune = 0.535`) debido al particionamiento interno por bloques de hilos de las estadísticas online CTR de CatBoost.

2. **`Pipeline A_valname` (Variante Nueva con `value_name` de Atributos y Umbral en `tune`):**
   - Definido en [`src/benchmark_20261030/data_loader.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/benchmark_20261030/data_loader.py#L36-L48) y ejecutado posteriormente por `reproduce_baseline_agent2.py`.
   - Construye el texto combinado usando los **nombres legibles de los valores de atributo** (`a.get("value_name")`):
     ```python
     v = a.get("value_name")
     ```
   - Esto cambia los bigramas del vectorizador TF-IDF (`vec_combo`), modificando la columna `txt_sparse_p` que entra a CatBoost.
   - Al comparar `A_valname` contra el histórico `A_hist`, las probabilidades tienen correlación de Pearson `r = 0.99637` (Semilla 2026) y `r = 0.99633` (Semilla 314159), con diferencias puntuales de hasta `0.531` en ítems donde el valor textual del atributo (ej. `"Usado"`, `"Reacondicionado"`, `"Nuevo"`) aporta señal léxica distinta del ID del atributo (`"ITEM_CONDITION"`), produciendo un acuerdo de **`98.3495%`** (Semilla 2026) y **`98.2948%`** (Semilla 314159).

### 1.3 Tabla Comparativa Directa en los Mismos IDs (`A_hist` vs `A_valname`)

| Semilla | N Confirm | Vista / Variante | Definición `combos` en `txt_sparse_p` | Umbral `tune` | Acc `tune` | Acc `confirm` (`th_tune`) | Acc `confirm` (`0.500`) | Log Loss `confirm` | Acuerdo vs R151 Histórico | Corr. Pearson vs R151 |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2026** | 12,905 | **`A_hist` (R151 Exacto)** | `title \|\| warranty \|\| attr.id` (`float64`) | `0.470` | 0.92809 | **0.92391** | 0.92437 | **0.18764** | **100.0000%** (`diff_5dec=0.0`) | **1.000000** |
| **2026** | 12,905 | **`A_valname` (Variante Nueva)** | `title warranty attr.value_name` (`float32`) | `0.515` | 0.92971 | **0.92491** | 0.92445 | 0.18773 | 98.3495% (`max_diff=0.531`) | 0.996371 |
| **314159** | 12,843 | **`A_hist` (`6a71f6c`, `tc=8`)** | `title \|\| warranty \|\| attr.id` (`float64`) | `0.535` | 0.93290 | **0.92019** | 0.92089 | 0.19558 | **100.0000%** (`diff_5dec=0.0`) | **1.000000** |
| **314159** | 12,843 | **`A_hist` (`tc=4` acotado)** | `title \|\| warranty \|\| attr.id` (`float64`) | `0.535` | 0.93217 | **0.91996** | 0.92089 | **0.19552** | 99.5095% (`max_diff=0.115`) | 0.999625 |
| **314159** | 12,843 | **`A_valname` (Variante Nueva)** | `title warranty attr.value_name` (`float32`) | `0.455` | 0.93209 | **0.91809** | 0.91910 | 0.19701 | 98.2948% (`max_diff=0.485`) | 0.996329 |
| **20261030** | 12,881 | **`A_hist` (R151 Exacto)** | `title \|\| warranty \|\| attr.id` (`float64`) | `0.440` | 0.93339 | **0.92920** | **0.93137** | **0.17039** | *(Semilla nueva)* | *(Referencia)* |
| **20261030** | 12,881 | **`A_valname` (Variante Nueva)** | `title warranty attr.value_name` (`float32`) | `0.470` | 0.93435 | **0.93028** | 0.92997 | 0.17042 | 98.7579% vs `A_hist` | 0.996864 |

---

## 2. Verificación de Etiquetas, Polaridad, Particiones, Orden de Features y Entorno

1. **Datos Crudos y Polaridad (`used=1, new=0`):**
   - `data/ap_2026_q_2/raw/train_data.jsonlines`: SHA256 `aa49b8710d06f666eeb3cf65c256c8c4f90bf4dbb22c697e313a814ebf5e01cf` (`70,000` filas; `32,412` con `condition == "used"` -> `46.3029%` tasa base `used=1`).
   - `data/ap_2026_q_2/raw/test_data.jsonlines`: SHA256 `05921bd73deba1c4ed5a4d564e0351729eae8f2ef197bc851e10f9099d277460` (`30,000` filas, IDs `1..30000`).
2. **Alineación Exacta por ID:**
   - En las semillas `2026` y `314159`, `ids[confirm]`, `y_confirm` y `is_seen_confirm` coinciden en el **100.00% de las posiciones** (`id_order_exact_match = True`, `y_true_exact_match = True`, `is_seen_exact_match = True`) con los CSV históricos `paired_deep_ablation_seed_{2026,314159}.csv`.
3. **Orden Exacto de las 105 Features:**
   - 96 features numéricas/booleanas + 5 categóricas (`category_id`, `listing_type_id`, `buying_mode`, `state`, `ship_mode`) + 3 Target Encodings OOF (`te_seller_id`, `te_category_id`, `te_city` con $\alpha=5.0$) + 1 probabilidad OOF de texto (`txt_sparse_p`). El listado completo de las 105 columnas en orden exacto está registrado en `agent1_integrity_manifest.json`.
4. **Versiones de Librerías Verificadas:**
   - Python `3.12.7`, NumPy `2.4.2`, Pandas `3.0.1`, SciPy `1.17.0`, Scikit-Learn `1.8.0`, CatBoost `1.2.10`, LightGBM `4.6.0`.

---

## 3. Preservación de Artefactos Históricos sin Sobrescritura

Cumpliendo la regla de no sobrescribir predicciones anteriores bajo el mismo nombre:
- Los archivos originales del commit `6a71f6c` fueron restaurados y verificados por SHA256:
  - `results/benchmark_20261030/preds_A_seed2026_confirm.csv.gz`: SHA256 `0db12d52610bbff70b8992b7e24171b6cf85a67bca7dcaeb7aa87798766ab1bd` (359,699 bytes).
  - `results/benchmark_20261030/preds_A_seed314159_confirm.csv.gz`: SHA256 `9289a5be3822cd8e2e389d665e69f8d6bede7825a7bff49660fdaac06a5049f1` (349,757 bytes).
- Las corridas de `A_hist` y `A_valname` (tanto `tune` como `confirm`) se guardaron bajo nombres explícitos y separados:
  - `results/benchmark_20261030/preds_A_hist_seed{2026,314159,20261030}_{tune,confirm}.csv.gz`
  - `results/benchmark_20261030/preds_A_valname_seed{2026,314159,20261030}_{tune,confirm}.csv.gz`

---

## 4. Auditoría de Solapamiento Histórico de `confirm` (Prohibido Llamar "Virgen" a un Conjunto Reutilizado)

Auditamos la intersección exacta de filas e IDs de vendedor entre los tres cortes `confirm` de desarrollo (`2026`, `314159`, `20261030`):

| Par de Particiones `confirm` | Filas Compartidas | % del Primer `confirm` (Filas) | Vendedores Compartidos | % del Primer `confirm` (Vendedores) |
| :--- | :---: | :---: | :---: | :---: |
| `r151_seed_2026` vs `r151_seed_314159` | **2,337** | **18.11%** | **2,089** | **31.42%** |
| `r151_seed_2026` vs `r151_seed_20261030` | **2,393** | **18.54%** | **2,073** | **31.18%** |
| `r151_seed_314159` vs `r151_seed_20261030` | **2,249** | **17.51%** | **2,020** | **30.41%** |
| **`seed_20261030` en unión `(2026 ∪ 314159)`** | **4,232** | **32.85%** | **3,170** | **48.13%** |
| **Intersección triple `(2026 ∩ 314159 ∩ 20261030)`** | **410** | **3.18%** | — | — |

- **Conclusión metodológica:** `seed_20261030` reutiliza el **32.85% de sus filas** (`4,232 / 12,881`) y el **48.13% de sus vendedores** respecto de la unión de `2026` y `314159`. Por lo tanto, **ninguno de los tres cortes `confirm` (`2026`, `314159`, `20261030`) es virgen ni independiente**, y jamás deben sumarse sus ganancias netas como si fueran ensayos independientes.

---

## 5. Auditoría de Ramas Git (`main`, `claude`, `antigravity`, `codex`)

| Rama / Referencia | HEAD | Fecha último commit | Resumen / Estado frente a `main` (`f681466`) |
| :--- | :---: | :---: | :--- |
| `main` / `origin/main` | `f681466` | 2026-10-29 23:53 -0300 | Contiene la integración previa (`6a71f6c` + `797e197` + `f681466`). |
| `antigravity/overnight-recovery-coordination-20261030` | `f681466` | 2026-10-29 23:53 -0300 | Rama activa de coordinación para esta entrega. |
| `antigravity/benchmark-recovery-20261030` | `82c36f6` | 2026-10-29 22:39 -0300 | Integrada completamente en `main`. |
| `antigravity/decision-packet-submissions-20261029` | `c45ecdc` | 2026-10-29 19:26 -0300 | Integrada en `main`; aportó el registro canónico 01–76. |
| `antigravity/round156-baseline-audit-and-ensemble-20261029` | `af873e6` | 2026-10-29 17:58 -0300 | Integrada en `main`; auditoría R156/R157. |
| `origin/claude/posterior-y-modelos-20261027` | `72c790c` | 2026-10-29 01:48 +0000 | Integrada en `main`; aportó `s48.json` y diagnóstico de `fin_stack` (5 pruebas fallidas). |
| `origin/experiment-ledger-20261025` | `8cd47f6` | 2026-10-28 20:18 -0300 | Integrada en `main`; ledger histórico de experimentos. |
| `origin/codex/multimodal-2h-20261010` | `d39266d` | 2026-10-10 22:53 -0300 | Histórica (R39/R52 FastText + HistGB). |
| `origin/codex/nuevas-representaciones-20261010` | `8cbaf45` | 2026-10-10 22:17 -0300 | Histórica (notebooks y representaciones iniciales). |
