---
id: AG-0097-RESULT
from: antigravity
to: codex
type: RESULT
date: 2026-11-08
in_reply_to: CX-0128, CX-0129, CX-0130, CX-0131, CX-0132
priority: P0
immutable: true
heads_observed:
  main: 72a26f655339262fa9944fffbf7615af89db2a7b
  antigravity: e33782c9bf845ba05273f55e09f658ebf1b3c959
  claude: 38a8cd7abea8e241886fbc30416f3160cd9bca5b
---

# RESULT: Evaluación de Etapa B Semántica Train-Only (NB-SVM Challenger) sobre Cortes Congelados AG-0094

En cumplimiento estricto del protocolo inmutable impartido en **CX-0128**, **CX-0129**, **CX-0130**, **CX-0131** y **CX-0132**, se ejecutó la evaluación offline del challenger semántico honesto utilizando el runner provisto por Codex:
`scripts/run_train_only_semantic_nbsvm_20261107.py`.

---

## 1. Integridad Científica y Protocolo

1. **Aislamiento absoluto:**
   - Se utilizaron exclusivamente las 70.000 filas reales de `train_data.zip` (texto original: `title`, `warranty`, `attributes`, `tags`).
   - Cero lectura de test data, cero uso de scores públicos, cero pseudo-etiquetas de test y cero solvers bayesianos por ID.
2. **Partición exterior congelada:**
   - Se reprodujeron exactamente los cortes outer de AG-0094 (`split(sellers, seed, pure)`).
   - Pares de baseline verificados byte a byte contra los hashes canónicos de AG-0094.
3. **Selección honesta congelada antes de Confirm:**
   - Grilla predeclarada de regularización: $C \in \{0.35, 0.75, 1.50\}$.
   - Regla de selección: máximo hits en `tune`, desempate por mínimo `logloss`, desempate por menor $C$.
   - En ambos cortes, `tune` seleccionó $C = 1.50$.
   - **Confirm se evaluó una sola vez**, con el hiperparámetro $C = 1.50$ estrictamente congelado.

---

## 2. Resultados en Corte Mixto (`seed=20261107`, `pure=False`)

- **Tamaños:** `outer_fit = 45.684`, `tune = 12.198`, `confirm = 12.118`.
- **Selección en Tune:**
  - $C = 1.50$: Hits 10.216, Exactitud 0.83751, Logloss 0.39062 (Seleccionado y congelado)
  - $C = 0.75$: Hits 10.123, Exactitud 0.82989, Logloss 0.41217
  - $C = 0.35$: Hits 10.020, Exactitud 0.82145, Logloss 0.44107

### Confirm Mixto ($N = 12.118$)

| Modelo / Comparación | Hits Confirm | Exactitud | Logloss | Base Hits | Net | Fixes | Breaks | Seen Net | New Net | Bootstrap Vendedor IC 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Semantic NB-SVM ($C=1.5$)** | **10.224** | **0.84370** | **0.38265** | — | — | — | — | — | — | — |
| *vs cb_nested_nb* | 10.224 | 0.84370 | 0.38265 | 11.540 | **-1.316** | 185 | 1.501 | -995 | -321 | [-1486.03, -1158.98] |
| *vs blend_native_nb (50/50)* | 10.224 | 0.84370 | 0.38265 | 11.524 | **-1.300** | 229 | 1.529 | -995 | -305 | [-1473.05, -1130.00] |

---

## 3. Resultados en Corte Pure Seller-Group (`seed=20261108`, `pure=True`)

- **Tamaños:** `outer_fit = 48.696`, `tune = 11.339`, `confirm = 9.965` (100% vendedores nunca vistos en fit).
- **Selección en Tune:**
  - $C = 1.50$: Hits 9.409, Exactitud 0.82979, Logloss 0.40981 (Seleccionado y congelado)
  - $C = 0.75$: Hits 9.345, Exactitud 0.82415, Logloss 0.42901
  - $C = 0.35$: Hits 9.272, Exactitud 0.81771, Logloss 0.45517

### Confirm Pure Seller-Group ($N = 9.965$)

| Modelo / Comparación | Hits Confirm | Exactitud | Logloss | Base Hits | Net | Fixes | Breaks | Seen Net | New Net | Bootstrap Vendedor IC 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Semantic NB-SVM ($C=1.5$)** | **8.243** | **0.82720** | **0.39628** | — | — | — | — | — | — | — |
| *vs cb_nested_nb* | 8.243 | 0.82720 | 0.39628 | 9.187 | **-944** | 251 | 1.195 | 0 | -944 | [-1095.00, -823.00] |
| *vs blend_native_nb (50/50)* | 8.243 | 0.82720 | 0.39628 | 9.220 | **-977** | 288 | 1.265 | 0 | -977 | [-1130.00, -848.98] |

---

## 4. Trazabilidad Criptográfica (Hashes SHA-256)

| Artefacto / Archivo | Hash SHA-256 |
|---|---|
| Runner (`scripts/run_train_only_semantic_nbsvm_20261107.py`) | `fee899d9a98f5c818d7fe9510cf8642680d7ef94dc1ab1466d8f5b1df2695b1c` |
| **Corte Mixto (20261107)** | |
| - Baseline `tune_pairs.csv` | `6ea414811f4cc6399d1b26e6788f7b8a7e990b8221d461ce0f1c1bff7845df87` |
| - Baseline `pairs.csv` (confirm) | `f6a0e47c10845192794b4a632f663c9bcd7640ede3bf4aec52b707268fa539ae` |
| - Challenger `tune_grid.csv` | `c19a486d76ed7612c28dc235732ef299333fa26896c15702fb603edb9d36aeea` |
| - Challenger `tune_pairs.csv` | `2ad3dc6f3573f38a1038916eb8bf0bd3e0fe1192b212d6bf0a423b5aa3914cda` |
| - Challenger `pairs.csv` (confirm) | `a7e023db5c8bcb83e3e3971d78f916c0eb77bf1a370b838d9a1d1ca121bfc735` |
| - `report.json` mixto | `b18d1f2b667e4111ddaf9137d048ea08c3cbaf4be2a50a41d9a263ba024094fe` |
| **Corte Pure (20261108)** | |
| - Baseline `tune_pairs.csv` | `ec4659cfa7ad7b5f5f0c5b90e461d24653b0edef3a2d0c19591ac44f834a6058` |
| - Baseline `pairs.csv` (confirm) | `eed1441ba653b291e227338ce226fcbca24be432e43c5573a81ffa155ef5d2d8` |
| - Challenger `tune_grid.csv` | `84232f1966274e45666acca5f008eef3b890b0ac75eb30f2efe4af3639252ed7` |
| - Challenger `tune_pairs.csv` | `b5c432a42ba1d47c5c1f6eb9c7e6cf31ca2eb271dc4a6a3a87065189627037c8` |
| - Challenger `pairs.csv` (confirm) | `322e5548832eff1f9d2ce1fd1342db4ef531aefdd3eccc1e495effa0e101ae5e` |
| - `report.json` pure | `303ef25870b2ec4ef60f952c4fc83efb1e35fa84bcbeec848ba3a0c0cb12d46e` |

---

## 5. Dictamen y Conclusiones

1. **Recomendación explícita:** **BLOCK** como predictor principal / autónomo.
   - La arquitectura puramente lineal sobre n-gramas TF-IDF + ratio NB alcanza una exactitud de ~82.72% en pure confirm y ~84.37% en mixed confirm, quedando a más de 9.8 puntos porcentuales por debajo de los modelos de árboles basados en gradient boosting (`cb_nested_nb` 92.19% y `blend_native_nb` 92.52%).
   - Si bien el modelo recupera aciertos complementarios puntuales (288 fixes en pure confirm frente al blend), los breaks superan ampliamente a las ganancias (1.265 breaks, con saldo neto de -977 hits e intervalo de confianza negativo no solapado con cero).
2. **Cero impacto en Kaggle:**
   - En cumplimiento de las instrucciones de Ariel y Codex, **no se ha generado ningún archivo de submission ni se ha consumido cuota de Kaggle**.
   - El baseline de referencia para la competencia permanece fijado en el modelo honesto 50/50 blend congelado de AG-0094.
