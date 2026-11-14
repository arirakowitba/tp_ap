# AG-0197: Señal Textual Complementaria, Encoders Regularizados y Modelado de Negaciones

**Fecha:** 2026-11-14T05:40:00Z  
**Emisor:** Antigravity (AG)  
**Receptor:** Codex / Ariel  
**Estado:** INMUTABLE / AUDITADO (RESULTADO POSITIVO MODESTO / GATE A APROBADO)  
**Split Canónico:** `data/splits/canonical_3way_splits.csv` (`SHA-256: 6afe07abc3fa2a93437dc544af5e709d7e84d3b6e8eaf9bcadac29912ef5b133`)  
**Pares Privados:** `results/complementary_text_tune_pairs_private.csv` (`SHA-256: a4f64fbfe1686cb13fa3f9ac5f3c7a626313c16191156a21343227ff6dbc5dc8`)  
**Reporte JSON:** `results/complementary_text_report.json`

---

## 0. Resumen negativo primero

1. **Margen estrecho sobre el baseline (+4 aciertos, IC incluye cero):**  
   La integración de la señal textual complementaria con modelado de negaciones sobre Baseline 2 (LightGBM Enriched: 14.112 aciertos en `tune`) produjo una mejora modesta de **+4 aciertos netos** (14.116 aciertos, 70 fixes vs 66 breaks). El intervalo de confianza del 90% (cluster bootstrap de dos vías vendedor $\times$ categoría a 2.000 iteraciones) es **[−16.0, +24.0]** con mediana +4.0, confirmando el comportamiento observado previamente por Codex (+4 aciertos con intervalo que contiene el cero).
2. **Asimetría de regímenes (ganancia en `cold_new` vs regresión en `seen_bracketed`):**  
   - En el régimen frío (`tune_cold_new`, 8.758 filas), el especialista textual aporta **+9 aciertos netos** (62 fixes vs 53 breaks) y una reducción de logloss de **−0.002264** ($P(\Delta > 0) = 0.771$).
   - En el régimen conocido (`tune_seen_bracketed`, 6.286 filas), el modelo integrado sufre **−5 aciertos netos** (8 fixes vs 13 breaks). En este régimen, la historia causal del vendedor (`seller_te_causal`) es extremadamente precisa y la señal textual introduce perturbaciones marginales.
3. **Ausencia de frameworks de encoders profundos:**  
   La auditoría del entorno confirmó que `torch`, `transformers`, `sentence_transformers` y `fasttext` no están instalados, y no existe caché local en `~/.cache/huggingface/hub`. Conforme al protocolo del proyecto, se siguió la mejor vía local disponible: regresión regularizada de alta capacidad sobre n-gramas de palabras y caracteres con modelado de ámbito de negación.
4. **Fracaso del blending lineal directo:**  
   Intentar ensamblar el modelo textual con Baseline 2 mediante combinaciones convexas lineales ($\alpha \cdot p_{\text{base}} + (1-\alpha) \cdot p_{\text{text}}$) resultó en deltas neutros o negativos (−2 a −10 aciertos). Solo la integración de la probabilidad out-of-fold (`p_text_negation_oof`) y sus diferencias dentro de las matrices de LightGBM permitió a los árboles aprender cuándo atender al texto y cuándo suprimirlo ante señales tabulares concluyentes.
5. **Declaración Gate C:**  
   Todas las probabilidades y cuantiles de transferencia pública no condicionada se declaran formalmente como `"NO_IDENTIFIABLE"`.

---

## 1. Verificación de Entorno y Auditoría de Encoders

- `origin/main`: `a778dbad7c69330e359f586c30b67071377fc5e1`
- `origin/codex/ap-refit-candidates-20261111`: `3952aee9417671302a7c78b2e5fa22fae8c83730`
- `origin/claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
- `local AG (HEAD)`: `443948b` (`antigravity/overnight-recovery-coordination-20261030`)

**Estado de encoders en el entorno:**
- `torch`: No disponible.
- `transformers`: No disponible.
- `sentence_transformers`: No disponible.
- `fasttext`: No disponible.
- `huggingface hub cache`: Inexistente (`hf_cache_exists = False`).
- **Vía ejecutada:** Modelos lineales regularizados (L-BFGS $L_2$, SAGA ElasticNet, Ridge calibrado) sobre TF-IDF word(1,2) + char_wb(3,5) con transformación de ámbito de negaciones (`NOT_` prefixing).

---

## 2. Ablación de Fuentes Textuales y Modelado de Negaciones

Se evaluaron cuatro configuraciones de corpus textual sobre las 40.881 filas de `fit` y se evaluaron en modo standalone sobre las 15.044 filas de `tune`:

| Configuración de Corpus | Dimensiones | Hits Standalone | Accuracy | Logloss | Fixes de Errores de Baseline 2 (sobre 932) | Desacuerdo con B2 | Yule's Q |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **T1: Título + Tags** | 100.000 | 12.482 | 0.829700 | 0.394045 | 336 (36.05%) | 15.30% | 0.8328 |
| **T2: Título + Tags + Atributos** | 100.000 | 12.533 | 0.833090 | 0.387525 | 351 (37.66%) | 15.16% | 0.8253 |
| **T3: Título + Attrs + Garantía/Sub** | 100.000 | **12.770** | **0.848843** | **0.357516** | 346 (37.12%) | 13.52% | 0.8515 |
| **T4: Título + Attrs + Garantía + Negaciones** | 100.000 | 12.731 | 0.846251 | 0.359821 | **360 (38.63%)** | 13.97% | 0.8373 |

### Hallazgos de la ablación:
1. **La garantía y subtítulo son la mayor fuente de ganancia léxica:** Pasar de T1 a T3 mejora el accuracy standalone de 82.97% a 84.88% (+288 aciertos) y reduce el logloss en −0.0365.
2. **El modelado de negaciones maximiza la ortogonalidad:** La configuración T4 (que transforma frases como `sin uso` $\to$ `sin NOT_uso` y `no funciona` $\to$ `no NOT_funciona`) corrige **360 de los 932 errores de Baseline 2 (38.63%)**, la tasa de rescate más alta de todos los modelos textuales.

---

## 3. Comparativa de Clasificadores Regularizados (sobre Corpus T4)

| Familia de Clasificador | Hiperparámetros / Regularización | Hits Standalone | Accuracy | Logloss | Fixes de B2 | Tiempo de Ajuste |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression L-BFGS** | $C=1.5, \text{penalty}=L_2, \text{tol}=10^{-4}$ | **12.731** | **0.846251** | **0.359821** | **360** | **2.97 s** |
| **Logistic Regression SAGA** | $C=1.0, \text{penalty}=\text{ElasticNet}, l_1=0.3$ | 12.660 | 0.841532 | 0.373608 | 344 | 88.92 s |
| **RidgeClassifier (Calibrado)** | $\alpha=1.0, \text{calibrated sigmoid}$ | 12.641 | 0.840269 | 0.492850 | 367 | 3.92 s |

**Conclusión:** L-BFGS con regularización $L_2$ ($C=1.5$) ofrece el mejor balance entre logloss (0.3598), corrección de errores (360 fixes) y eficiencia computacional (<3 segundos).

---

## 4. Resultados del Modelo Integrado en `tune` (15.044 filas)

Se integró la probabilidad OOF del especialista textual T4 (`p_neg_fit_oof`, calculada estrictamente en 4 folds internos de `fit` agrupados por `seller_id`) y su señal de contraste (`p_neg - p_text_baseline`) dentro de las matrices de LightGBM Enriched, entrenado con 3 semillas embolsadas (`[42, 2026, 777]`):

| Métrica / Partición | Baseline 2: LGB Enriched | Modelo Integrado (+Texto Negaciones) | Delta Neto | IC 90% Cluster Bootstrap (Seller, Cat) |
|---|:---:|:---:|:---:|:---:|
| **Tune Total (15.044 filas)** | | | | |
| Aciertos (Hits) | 14.112 | **14.116** | **+4** | [−16.0, +24.0] (mediana +4.0) |
| Accuracy | 0.938048 | **0.938314** | **+0.000266** | — |
| Logloss | 0.158633 | **0.157494** | **−0.001139** | [−0.002089, −0.000177] |
| Brier Score | 0.045935 | **0.045592** | **−0.000343** | — |
| Correcciones (Fixes) | — | — | **70** | — |
| Regresiones (Breaks) | — | — | **66** | (Ratio Fix:Break = 1.06 : 1) |
| Flips de clase | — | — | 65 a NEW, 71 a USED | — |
| **Tune Cold New (8.758 filas)** | | | | |
| Aciertos (Hits) | 8.029 | **8.038** | **+9** | [−8.0, +28.0] ($P(\Delta > 0) = 0.771$) |
| Accuracy | 0.916762 | **0.917789** | **+0.001028** | — |
| Logloss | 0.203549 | **0.201285** | **−0.002264** | [−0.003817, −0.000714] |
| Fixes / Breaks | — | — | 62 / 53 | (Ratio Fix:Break = 1.17 : 1) |
| **Tune Seen Bracketed (6.286 filas)** | | | | |
| Aciertos (Hits) | **6.083** | 6.078 | **−5** | [−12.0, +2.0] |
| Accuracy | **0.967706** | 0.966911 | **−0.000795** | — |
| Logloss | **0.096054** | 0.096481 | **+0.000428** | — |
| Fixes / Breaks | — | — | 8 / 13 | (Ratio Fix:Break = 0.62 : 1) |

---

## 5. Recomendación y Síntesis para el Torneo Final

1. **La señal con negaciones es genuinamente complementaria en el régimen frío:**  
   En `cold_new`, donde el vendedor no aporta historia causal, la representación con negaciones (`NOT_uso`, `NOT_funciona`) aporta **+9 aciertos netos** y reduce el logloss de forma estadísticamente significativa ([−0.0038, −0.0007]).
2. **Ruteo condicional por régimen:**  
   Para maximizar la transferencia sin degradar `seen_bracketed`, se recomienda activar las features del especialista textual **únicamente en el brazo frío** o atenuar su peso cuando `seller_hist_cnt > 0`.
3. **Pares privados guardados:**  
   `results/complementary_text_tune_pairs_private.csv` (gitignored, SHA-256: `a4f64fbfe1686cb13fa3f9ac5f3c7a626313c16191156a21343227ff6dbc5dc8`).
4. **Auditoría completa:**  
   `results/complementary_text_report.json` y script [train_complementary_text_specialist.py](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_complementary_text_specialist.py).
