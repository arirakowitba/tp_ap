# AG-0195: Reproducción Rigurosa de Dos Baselines Fuertes (Line3 vs Texto OOF con Atributos)

**Fecha:** 2026-11-14T05:15:00Z  
**Emisor:** Antigravity (AG)  
**Receptor:** Codex / Ariel  
**Estado:** INMUTABLE / AUDITADO  
**Split Canónico:** `data/splits/canonical_3way_splits.csv` (`SHA-256: 6afe07abc3fa2a93437dc544af5e709d7e84d3b6e8eaf9bcadac29912ef5b133`)  
**Pares Privados:** `results/baselines_paired_tune_pairs_private.csv` (`SHA-256: 99e40a38e8b19bb98b8c7ec99f9e6f26089dfc0d2687c2aa5c1d3acd90392975`)  
**Reporte JSON:** `results/baselines_reproduction_report.json`

---

## 0. Resumen negativo primero

1. **228 regresiones (breaks) introducidas por atributos:**  
   Al pasar de Line 3 a LightGBM Enriched con atributos sobre las mismas 15.044 filas de `tune`, se introdujeron **228 errores nuevos** (filas donde Line 3 acertaba y el modelo con atributos falló). De estos, 207 cayeron en el régimen `cold_new` y 21 en `seen_bracketed`.
2. **Line 3 aislado tiene techo bajo en transferibilidad general:**  
   Line 3 (texto OOF SAGA LR + SVD(64) + seller TE causal) alcanza 90.7338% de accuracy (13.650/15.044 aciertos en tune; logloss 0.226416). Esto confirma cuantitativamente que basarse exclusivamente en texto y TE causal deja escapar 462 aciertos que los atributos estructurados (garantía, envío, reputación, cantidad y TEs de marca/modelo/ubicación) corrigen.
3. **Concentración moderada en categorías y vendedores top:**  
   - Las 5 categorías con mayor aporte positivo (`MLA3530`, `MLA42435`, `MLA5599`, `MLA1227`, `MLA15171`) concentran **61 aciertos netos** (13.2% del delta total). Al removerlas, la ganancia se mantiene sólida en **+401 aciertos**.
   - Los 5 vendedores con mayor aporte neto concentran **62 aciertos netos** (13.4% del delta total). Al removerlos, la ganancia se mantiene en **+400 aciertos**.
4. **Asimetría de flips de clase:**  
   El modelo con atributos realizó **498 flips hacia NEW** y **420 flips hacia USED** con respecto a Line 3. El sesgo hacia NEW proviene de señales como garantía de fábrica, tienda oficial y envío gratis que rescatan publicaciones nuevas con títulos ambiguos.
5. **Declaración Gate C:**  
   Todas las probabilidades y cuantiles de transferencia pública no condicionada se declaran estrictamente como `"NO_IDENTIFIABLE"`. No se proyectan scores públicos sin calibración prequencial.

---

## 1. Verificación de Entorno y HEADs

- `origin/main`: `a778dbad7c69330e359f586c30b67071377fc5e1`
- `origin/codex/ap-refit-candidates-20261111`: `3952aee9417671302a7c78b2e5fa22fae8c83730`
- `origin/claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
- `local AG (HEAD)`: `bc7f5f4` (`antigravity/overnight-recovery-coordination-20261030`)

---

## 2. Protocolo de Ejecución y Garantías Anti-Fuga

1. **Mismas filas exactas:**  
   Evaluación realizada sobre las 15.044 filas del partition `tune` (`data/splits/canonical_3way_splits.csv`), compuesto por 8.758 filas `tune_cold_new` y 6.286 filas `tune_seen_bracketed`.
2. **Verificación de polaridad (`used=1`, `new=0`):**  
   - Base de entrenamiento `fit` (40.881 filas): 18.389 `used` (44.98%), 22.492 `new` (55.02%).
   - Presencia de palabras clave usadas (`usado`, `antiguo`, `impecable`): $P(\text{used}=1) = 0.9176$.
   - Presencia de palabras clave nuevas (`nuevo`, `0km`, `sellado`): $P(\text{used}=1) = 0.2111$.
   - Polaridad formalmente verificada: probabilidades modelan $P(y = 1) = P(\text{used})$.
3. **Joins 1:1 por ID alfanumérico:**  
   - 15.044 IDs coincidentes exactamente con el split (`MLA...`), 0 nulos, 0 duplicados, orden canónico idéntico.
4. **Ajuste de hiperparámetros y early stopping estrictamente interno (Gate B & E):**  
   - Partición de `fit` (40.881) en 4 folds internos mediante `StratifiedGroupKFold` agrupado por `seller_id` (0 solapamiento de vendedores entre inner train e inner validation).
   - Early stopping con `stopping_rounds=50` monitoreando `binary_logloss` sobre la validación interna.
   - Búsqueda en grilla de hiperparámetros ejecutada 100% dentro de `fit`, sin tocar `tune`.

---

## 3. Resultados de la Sintonización Interna (4-Fold Inner CV)

### Baseline 1: Line 3 (M0, 78 features)
- **Candidatos evaluados:**
  - Cand 1 (`num_leaves=48, min_child_samples=30, colsample=0.75`): OOF Logloss = **0.335423**, Acc = **0.856046** (34.996 aciertos), Median Best Iter = **258** (folds: [210, 307, 205, 382]).
  - Cand 2 (`num_leaves=63, min_child_samples=30, colsample=0.75`): OOF Logloss = 0.335698, Acc = 0.855972 (34.993 aciertos), Median Best Iter = 198.
  - Cand 3 (`num_leaves=95, min_child_samples=40, colsample=0.70`): OOF Logloss = 0.335435, Acc = 0.855361 (34.968 aciertos), Median Best Iter = 208.
- **Seleccionado:** Candidate 1 (`num_leaves=48`, `min_child_samples=30`, `colsample_bytree=0.75`, `n_estimators=258`).
- **Costo de tuning interno:** 34.04 segundos.

### Baseline 2: Texto OOF seller-group con atributos (M1 / LightGBM Enriched, 144 features)
- **Candidatos evaluados:**
  - Cand 1 (`num_leaves=48, min_child_samples=35, colsample=0.75`): OOF Logloss = **0.230934**, Acc = **0.906265** (37.049 aciertos), Median Best Iter = **307** (folds: [319, 296, 210, 321]).
  - Cand 2 (`num_leaves=63, min_child_samples=30, colsample=0.75`): OOF Logloss = 0.230963, Acc = 0.904993 (36.997 aciertos), Median Best Iter = 216.
  - Cand 3 (`num_leaves=95, min_child_samples=40, colsample=0.80`): OOF Logloss = 0.231095, Acc = 0.906044 (37.040 aciertos), Median Best Iter = 242.
- **Seleccionado:** Candidate 1 (`num_leaves=48`, `min_child_samples=35`, `colsample_bytree=0.75`, `n_estimators=307`).
- **Costo de tuning interno:** 48.92 segundos.

---

## 4. Métricas Pareadas en `tune` (15.044 filas)

Entrenamiento final con ensamble embolsado de 3 semillas (`[42, 2026, 777]`) sobre las 40.881 filas de `fit`, ruteando por régimen (`unseen` para `cold_new`, `seen_causal` para `seen_bracketed`).

| Métrica / Partición | Baseline 1: Line 3 | Baseline 2: LGB Enriched (Atributos) | Delta Neto (B2 − B1) | IC 90% Cluster Bootstrap (Seller, Cat) |
|---|:---:|:---:|:---:|:---:|
| **Tune Total (15.044 filas)** | | | | |
| Aciertos (Hits) | 13.650 | **14.112** | **+462** | [+402.0, +524.0] |
| Accuracy | 0.907338 | **0.938048** | **+0.030710** | — |
| Logloss | 0.226416 | **0.158633** | **−0.067783** | [−0.074481, −0.061693] |
| Brier Score | 0.067402 | **0.045935** | **−0.021467** | — |
| Correcciones (Fixes) | — | — | **690** | — |
| Regresiones (Breaks) | — | — | **228** | (Ratio Fix:Break = 3.03 : 1) |
| Flips de predicción | — | — | 498 a NEW, 420 a USED | — |
| **Tune Cold New (8.758 filas)** | | | | |
| Aciertos (Hits) | 7.667 | **8.029** | **+362** | [+306.0, +418.1] |
| Accuracy | 0.875428 | **0.916762** | **+0.041334** | — |
| Logloss | 0.291589 | **0.203549** | **−0.088040** | [−0.097977, −0.077852] |
| Brier Score | 0.089204 | **0.060238** | **−0.028966** | — |
| Fixes / Breaks | — | — | 569 / 207 | (Ratio Fix:Break = 2.75 : 1) |
| **Tune Seen Bracketed (6.286 filas)** | | | | |
| Aciertos (Hits) | 5.983 | **6.083** | **+100** | [+78.0, +122.0] |
| Accuracy | 0.951798 | **0.967706** | **+0.015908** | — |
| Logloss | 0.135613 | **0.096054** | **−0.039559** | [−0.045880, −0.033313] |
| Brier Score | 0.037027 | **0.026007** | **−0.011020** | — |
| Fixes / Breaks | — | — | 121 / 21 | (Ratio Fix:Break = 5.76 : 1) |

---

## 5. Costo Computacional y Recursos de Ejecución

| Componente | Baseline 1 (Line 3) | Baseline 2 (LGB Enriched) | Pipeline Total |
|---|:---:|:---:|:---:|
| Tuning interno (4 folds $\times$ 3 candidatos) | 34.04 s | 48.92 s | 82.96 s |
| Ajuste final (3 semillas $\times$ 2 regímenes) | 13.50 s | 22.05 s | 35.55 s |
| **Tiempo total de corrida** | **47.54 s** | **70.97 s** | **129.34 s** (~2.15 min) |
| Árboles óptimos seleccionados | 258 | 307 | — |
| Features en matriz | 78 | 144 | — |
| Cores utilizados | 6 | 6 | 6 |
| Memoria RAM pico | ~1.4 GB | ~1.8 GB | < 2.0 GB |

---

## 6. Archivos y Artefactos Entregados

1. **Pares Privados:**  
   `results/baselines_paired_tune_pairs_private.csv` (gitignored, SHA-256: `99e40a38e8b19bb98b8c7ec99f9e6f26089dfc0d2687c2aa5c1d3acd90392975`)  
   Columnas: `id, y, p_line3, p_lgb_attrs, regime, seller_id, category_id`.
2. **Reporte JSON Completo:**  
   `results/baselines_reproduction_report.json` (incluye toda la auditoría fold por fold, importancias, ablación y bootstrap).
3. **Script Ejecutable y Reproducible:**  
   `scripts/reproduce_two_strong_baselines.py`.

Ambos baselines quedan fijados como referencias inmutables del split canónico para la comparación pareada de cualquier candidato posterior.
