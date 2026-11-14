# AG-0196: Auditoría y Resultados de Recuperación de Producto Cross-Seller (Marca/Modelo/Códigos vs N-gramas/BM25)

**Fecha:** 2026-11-14T05:25:00Z  
**Emisor:** Antigravity (AG)  
**Receptor:** Codex / Ariel  
**Estado:** INMUTABLE / AUDITADO (RECHAZO EXPERIMENTAL GATE A)  
**Split Canónico:** `data/splits/canonical_3way_splits.csv` (`SHA-256: 6afe07abc3fa2a93437dc544af5e709d7e84d3b6e8eaf9bcadac29912ef5b133`)  
**Pares Privados:** `results/cross_seller_retrieval_tune_pairs_private.csv` (`SHA-256: 77585c727d2e94385ade86fe4bcd95dfee86c353f65b599211b2c7d952090258`)  
**Reporte JSON:** `results/cross_seller_retrieval_report.json`

---

## 0. Resumen negativo primero

1. **Rechazo formal Gate A (Degradación neta sobre el baseline):**  
   Ninguna de las variantes de recuperación entre vendedores logró superar a Baseline 2 (LightGBM Enriched: 14.112 aciertos en `tune`, 93.8048% de accuracy).
   - **Ablation 1 (Solo Coincidencia Exacta Marca/Modelo/Código):** **−12 aciertos netos** (34 fixes, 46 breaks, IC 90%: [−26.0, +2.0]).
   - **Ablation 2 (Solo Recuperación Léxica / BM25):** **−2 aciertos netos** (56 fixes, 58 breaks, IC 90%: [−20.0, +16.0]).
   - **Ablation 3 (Solo Precio Relativo a Vecinos):** **−4 aciertos netos** (38 fixes, 42 breaks, IC 90%: [−18.0, +10.0]).
   - **Full Model (Recuperación Cross-Seller Completa):** **−4 aciertos netos** (60 fixes, 64 breaks, IC 90%: [−22.0, +14.0]).  
   Conforme a Gate A, la promoción de estos modelos queda **estrictamente bloqueada**.
2. **Fragilidad en el brazo frío (`cold_new`):**  
   La mayor pérdida se concentra en el régimen de vendedores no vistos (`tune_cold_new`, 8.758 filas):
   - Coincidencia exacta pierde **−14 aciertos** en `cold_new` (23 fixes vs 37 breaks).
   - El modelo completo pierde **−5 aciertos** en `cold_new` (46 fixes vs 51 breaks).  
   Esto demuestra que cuando un vendedor es desconocido, confiar en la etiqueta de otro vendedor para un producto homónimo induce a error.
3. **Alta tasa de contradicción histórica (13.30% a 16.69%):**  
   - En coincidencia exacta estructurada, de los 233 casos que hicieron match entre vendedores distintos, **31 casos (13.30%) tenían etiquetas opuestas** (en `fit`, existían publicaciones nuevas y usadas bajo el mismo código exacto).
   - En recuperación léxica con $\text{sim} \ge 0.60$ (797 casos), la tasa de desacuerdo de condición con la consulta fue de **16.69%**.
   - **Conclusión física:** La identidad del producto NO determina unívocamente la condición de publicación. Celulares, repuestos, consolas y libros se comercializan simultáneamente como nuevos (distribuidores oficiales) y como usados (particulares o reacondicionados).
4. **Baja cobertura cross-seller estructurada (1.55%):**  
   En un catálogo con más de 8.000 categorías y publicaciones de cola larga, la coincidencia exacta de marca/modelo y códigos alfanuméricos entre vendedores distintos sólo cubre a 233 de las 15.044 filas de `tune` (1.55%).
5. **Declaración Gate C:**  
   Todas las probabilidades y cuantiles de transferencia pública no condicionada se declaran formalmente como `"NO_IDENTIFIABLE"`.

---

## 1. Verificación de Entorno y HEADs

- `origin/main`: `a778dbad7c69330e359f586c30b67071377fc5e1`
- `origin/codex/ap-refit-candidates-20261111`: `3952aee9417671302a7c78b2e5fa22fae8c83730`
- `origin/claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
- `local AG (HEAD)`: `5aa3224` (`antigravity/overnight-recovery-coordination-20261030`)

---

## 2. Metodología de Recuperación y Garantías Anti-Fuga (Gate B & E)

1. **Construcción estricta de índices sobre `fit` (40.881 filas):**
   - **Mecanismo A (Estructurado):** Diccionario hash invertido de marca+modelo normalizados (`category_id + "_" + brand + "_" + model`) y códigos alfanuméricos de modelo extraídos de título con regex (`category_id + "_" + code`). Total: 7.690 claves indexadas.
   - **Mecanismo B (Léxico / BM25):** Matriz dispersa TF-IDF TF sublineal word(1,2) + char(3,5) acotada por `category_id`.
2. **Leave-Query-Out y Leave-Seller-Out:**
   - Para filas en `fit`: cuando un ítem $i$ consulta el índice, se excluye el propio ítem ($j \ne i$) y a cualquier ítem del mismo vendedor ($j.\text{seller\_id} \ne i.\text{seller\_id}$).
   - Para filas en `tune`: se exige estrictamente $j.\text{seller\_id} \ne \text{query.seller\_id}$.
3. **Mismo Baseline de Contraste:**  
   Todos los modelos agregan los features de recuperación sobre la matriz exacta de **Baseline 2 (LightGBM Enriched, 144 features)**, entrenados con las 3 semillas embolsadas `[42, 2026, 777]` y los hiperparámetros seleccionados en inner CV (`num_leaves=48, min_child_samples=35, colsample=0.75, n_estimators=307`).

---

## 3. Comparativa entre Mecanismos de Recuperación

| Dimensión de Auditoría | Mecanismo A: Estructurado (Marca/Modelo/Código) | Mecanismo B: Léxico (N-gramas / BM25) |
|---|:---:|:---:|
| Tipo de índice | Exact Hash Table por clave normalizada | Búsqueda por similitud coseno por categoría |
| Claves / Entidades en `fit` | 7.690 claves | 43.960 n-gramas en vocabulario |
| Cobertura en `tune` (15.044 filas) | **1.55%** (233 ítems con match de otro vendedor) | **100% continuo** (mediana similitud = 0.1952) |
| Cobertura con Similitud $\ge 0.60$ | — | **5.30%** (797 ítems) |
| Cobertura con Similitud $\ge 0.75$ | — | **1.81%** (272 ítems) |
| Concordancia de condición ($y_{\text{query}} == y_{\text{vecino}}$) | **85.41%** | 83.31% ($\text{sim} \ge 0.60$) / **87.13%** ($\text{sim} \ge 0.75$) |
| Tasa de contradicción en `fit` | **13.30%** (31 / 233 casos con items nuevos Y usados) | **16.69%** (133 / 797 desacuerdos en $\text{sim} \ge 0.60$) |
| Tiempo de extracción (`fit` + `tune`) | 4.82 s | 35.52 s |

---

## 4. Resultados de Ablación y Aporte Marginal en `tune` (15.044 filas)

Todos los deltas están calculados en forma pareada contra Baseline 2 (14.112 aciertos, accuracy 0.938048, logloss 0.158633):

| Modelo / Variante | Aciertos (`tune`) | Delta Hits | Fixes / Breaks | Delta Logloss | IC 90% Cluster Bootstrap (Seller, Cat) | Delta en Cold New (8.758) | Delta en Seen Bracketed (6.286) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 2: LGB Enriched (Ref)** | **14.112** | 0 | — | 0.00000 | — | 0 (8.029 aciertos) | 0 (6.083 aciertos) |
| **Ablación 1: Solo Coincidencia Exacta** | 14.100 | **−12** | 34 / 46 | +0.00021 | [−26.0, +2.0] | **−14** (23 / 37) | +2 (11 / 9) |
| **Ablación 2: Solo Recuperación Léxica** | 14.110 | **−2** | 56 / 58 | −0.00040 | [−20.0, +16.0] | **−2** (49 / 51) | 0 (7 / 7) |
| **Ablación 3: Solo Precio Relativo** | 14.108 | **−4** | 38 / 42 | −0.00002 | [−18.0, +10.0] | **−3** (33 / 36) | **−1** (5 / 6) |
| **Full Model: Recuperación Combinada** | 14.108 | **−4** | 60 / 64 | −0.00047 | [−22.0, +14.0] | **−5** (46 / 51) | +1 (14 / 13) |

---

## 5. Análisis Diagnóstico: ¿Por qué falla la transferencia cross-seller?

1. **La señal de atributos ya absorbe la mayor parte de la información:**  
   LightGBM Enriched ya cuenta con 144 features que incluyen precio relativo local por categoría, ranking percentilar, garantía de fábrica, tienda oficial, tipo de envío y texto OOF por vendedor. Los vecinos agregan señal redundante donde el modelo ya acierta, y señal ruidosa donde el modelo duda.
2. **Ambivalencia del producto:**  
   En artículos de tecnología y electrónica (e.g. `iPhone 7`, `PS4`, `GTX 1060`, `G530`), la presencia del producto idéntico vendido por otro seller no dice si la consulta es nueva o usada. Si el vecino en `fit` es un distribuidor mayorista (`y = 0`), pero la consulta es un particular vendiendo el mismo modelo (`y = 1`), la característica de vecino arrastra la predicción hacia `new`, rompiendo la clasificación (46 a 64 breaks).
3. **El precio relativo a vecinos no discrimina mejor que el precio relativo a la categoría:**  
   El precio relativo contra el vecino recuperado (`rel_price_to_neighbor`) no logró superar al ranking percentilar fold-local por categoría ya presente en Baseline 2 (Ablación 3 perdió −4 aciertos netos).

---

## 6. Archivos y Artefactos Entregados

1. **Pares Privados:**  
   `results/cross_seller_retrieval_tune_pairs_private.csv` (gitignored, SHA-256: `77585c727d2e94385ade86fe4bcd95dfee86c353f65b599211b2c7d952090258`)  
   Columnas: `id, y, p_baseline2_lgb_enriched, p_cross_seller_retrieval, regime, seller_id, category_id`.
2. **Reporte JSON Completo:**  
   `results/cross_seller_retrieval_report.json` (auditoría completa con matrices de confusión, tiempos de extracción y bootstrap).
3. **Script de Entrenamiento:**  
   `scripts/train_cross_seller_product_retrieval.py`.

**Decisión para el Torneo:**  
Las características de recuperación cross-seller quedan **descartadas para la rama de producción** debido a su saldo neto negativo (−4 a −12 aciertos) y su fragilidad en el régimen frío. El pipeline se mantiene en torno a LightGBM Enriched enriquecido con representaciones intrínsecas de identidad y estado del ítem.
