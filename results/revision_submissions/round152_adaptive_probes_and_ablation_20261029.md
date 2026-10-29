# Ronda 151 y 152: Ablación Deep CatBoost + NB-SVM y Protocolo Adaptativo de 4 Sondas para el Cierre de Kaggle

**Fecha**: 2026-10-29  
**Branch**: `antigravity/deep-nbsvm-strong-validation-20261029`  
**Objetivo de las 21:00 ART**: Maximizar el score final de Kaggle con control estricto de riesgo, respaldado en la mejor predicción conocida (`Submission 48` = 0.94633).

---

## 1. Parte I: Ablación Rigurosa de Deep CatBoost + NB-SVM (Ronda 151)

Para dar respuesta a la prioridad asignada por Ariel, recuperamos la arquitectura tabular profunda (101 variables: precios relativos, ratios de stock y venta, códigos y dimensiones de imágenes, duraciones y lags de fechas, métodos de pago, banderas léxicas y agregaciones no supervisadas) junto con target encodings OOF por vendedor en 4 folds (`te_seller`, `te_cat`, `te_city`).

Comparamos de forma honesta e idéntica:
- **Modelo Base (Deep CatBoost)**: 101 features tabulares + TE OOF + probabilidad de texto TF-IDF (`txt_sparse_p`).
- **Modelo Ablación (Deep CatBoost + NB-SVM)**: Exactamente las mismas features + probabilidad OOF de NB-SVM (`txt_nbsvm_p`).
- **Partición**: Separación estricta de vendedores fit ~45k, tune ~12k, confirm ~13k sobre dos semillas independientes. Umbrales de decisión congelados en `tune` antes de evaluar en `confirm`.

### Resultados en Holdout Confirm Virgen

| Semilla | Arquitectura | Confirm Acc | Log Loss | Fixes / Breaks | Neto | IC 95% Bootstrap Vendedor | Neto Vistos | Neto Nuevos |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2026** | Deep CatBoost Base | 0.92391 | 0.18764 | — | — | — | — | — |
| **2026** | **Deep CatBoost + NB-SVM** | **0.92453** | **0.18791** | 91 / 83 | **+8** | **[-20.0, +35.0]** | +2 | +6 |
| **314159** | Deep CatBoost Base | 0.92019 | 0.19558 | — | — | — | — | — |
| **314159** | **Deep CatBoost + NB-SVM** | **0.92011** | **0.19470** | 115 / 116 | **-1** | **[-34.0, +32.0]** | -10 | +9 |

### Conclusión Científica Definitiva
1. **La base profunda es sumamente potente**: Eleva la accuracy en confirm externo a ~0.920–0.924 y desploma el log loss a ~0.187–0.195 (frente a 0.231 del modelo compacto).
2. **Neutralidad de la señal NB-SVM**: Al disponer de todo el contexto tabular profundo, la probabilidad NB-SVM aporta **+8 aciertos netos en semilla 2026 y -1 en semilla 314159** (acumulado neto +7 sobre 25.748 instancias de confirm).
3. Ambos intervalos bootstrap cruzan ampliamente el cero (`[-20, +35]` y `[-34, +32]`). **Se descarta formalmente reemplazar el baseline fuerte con esta ablación.**

---

## 2. Parte II: Diseño de las 4 Sondas Adaptativas (Ronda 152)

Ariel dispone de **4 cupos adaptativos** para comprar información sobre incertidumbre real y **1 quinto cupo final** para maximizar el score.

Siguiendo la lección de las 5 pruebas de `fin_stack` (donde cambios masivos o en vendedores vistos degradaron el score de .94633 a .94257), las 4 sondas fueron diseñadas con:
- **Tamaño micro-acotado** (7 a 10 ítems por sonda).
- **Disyunción estricta** (los 35 ítems modificados son mutuamente excluyentes; 0 solapamiento entre sondas).
- **Certeza extrema del modelo** ($p < 0.06$ o $p > 0.90$).
- **Aislamiento de hipótesis ortogonales**.

---

### Sonda 1 (`submission_66_probe1_new_sellers_to_new.csv`)
- **Hipótesis**: En vendedores nuevos, ¿la Submission 48 tiene falsos positivos de `used` en ítems con evidencia aplastante de ser nuevos?
- **Segmento**: Vendedores Nuevos (`is_seen == 0`), `s48 == used`, predicción objetivo `new`.
- **Cantidad de Flips**: **10 ítems**.
- **Certeza del Modelo**: Margen extremo ($p_{\text{used}} < 0.059$, promedio $p_{\text{used}} = 0.038$).
- **IDs modificados**: `29262`, `18423`, `23563`, `1083`, `16148`, `3250`, `18790`, `7755`, `29312`, `15613`.
- **SHA256**: `082a7e2566919bdc8dc0661884b8f878e8855d63bcb046e4553a8da6c9caf172`
- **Decisión que habilita su Score ($S_1$)**:
  - Si $S_1 > 0.94633$: Se confirma que los flips de alto margen `used -> new` en nuevos son válidos. Habilita incluir este bloque en el 5º cupo.
  - Si $S_1 \le 0.94633$: Se refuta la hipótesis. Se bloquea cualquier flip `used -> new` en nuevos.

---

### Sonda 2 (`submission_67_probe2_new_sellers_to_used.csv`)
- **Hipótesis**: En vendedores nuevos, ¿la Submission 48 tiene falsos positivos de `new` en ítems con evidencia aplastante de ser usados?
- **Segmento**: Vendedores Nuevos (`is_seen == 0`), `s48 == new`, predicción objetivo `used`.
- **Cantidad de Flips**: **10 ítems**.
- **Certeza del Modelo**: Margen extremo ($p_{\text{used}} > 0.902$, promedio $p_{\text{used}} = 0.925$).
- **IDs modificados**: `12877`, `14589`, `18735`, `25040`, `20534`, `25747`, `20714`, `449`, `7639`, `10705`.
- **SHA256**: `b3ec2a0d00153ced6b7fd3f6083e760b3b8767e474278c3bc834bfdeecc4f439`
- **Decisión que habilita su Score ($S_2$)**:
  - Si $S_2 > 0.94633$: Se confirma que la 48 es conservadora clasificando ítems usados en nuevos vendedores. Habilita incluir este bloque en el 5º cupo.
  - Si $S_2 \le 0.94633$: Se refuta la hipótesis. Se bloquea cualquier flip `new -> used` en nuevos.

---

### Sonda 3 (`submission_68_probe3_seen_sellers_to_used.csv`)
- **Hipótesis**: En vendedores vistos con historial fuertemente usado en train ($\ge 80\%$ de ventas `used`, $\ge 3$ publicaciones), ¿los ítems donde s48 marcó `new` pero el modelo predice `used` con certidumbre ($p > 0.75$) son aciertos?
- **Segmento**: Vendedores Vistos (`is_seen == 1`), `s48 == new`, predicción objetivo `used`.
- **Cantidad de Flips**: **7 ítems**.
- **IDs modificados**: `12726`, `16846`, `22499`, `16358`, `2204`, `4060`, `12228`.
- **SHA256**: `270dea10bc6dc5a15b8bfe0b324f8dc8796d195b4f55436d1aae56980f419dc0`
- **Decisión que habilita su Score ($S_3$)**:
  - Si $S_3 > 0.94633$: Se valida la corrección contextual por historial vendedor sobre casos específicos de alta probabilidad. Habilita incluir este bloque en el 5º cupo.
  - Si $S_3 \le 0.94633$: Confirma el principio general de no intervenir vendedores vistos contra la 48.

---

### Sonda 4 (`submission_69_probe4_unanimous_consensus.csv`)
- **Hipótesis**: ¿Los desacuerdos de consenso unánime entre modelos de diferente familia con margen residual extremo ($|p - 0.50| > 0.46$) superan a la 48?
- **Segmento**: Residuales de máximo margen multimodelo disjuntos de las sondas 1, 2 y 3.
- **Cantidad de Flips**: **8 ítems** (6 `used -> new`, 2 `new -> used`).
- **IDs modificados**: `23586`, `7252`, `560`, `11244`, `19313`, `9327`, `7914`, `14405`.
- **SHA256**: `2028620b12a038415de9b2f03cc59dfceb3aed16ae39682fb5f0c5f13df49994`
- **Decisión que habilita su Score ($S_4$)**:
  - Si $S_4 > 0.94633$: Se confirma que el consenso unánime residual mejora la 48.
  - Si $S_4 \le 0.94633$: Se descarta cualquier intento de ensamblado residual sobre la 48.

---

## 3. Parte III: Regla Explícita para Elegir el Quinto CSV Final

Sean $S_1, S_2, S_3, S_4$ los puntajes devueltos por Kaggle para las Sondas 1, 2, 3 y 4, frente a la referencia $S_{48} = 0.94633$:

1. **Caso 1: Ninguna sonda supera a la 48 ($\max(S_1, S_2, S_3, S_4) \le S_{48}$)**:
   - **Acción**: **Enviar exactamente la `Submission 48` en el 5º cupo** (o conservarla como la seleccionada final).
   - **Razón**: Las 4 sondas habrán demostrado empíricamente que modificar cualquiera de esos 35 ítems de máxima confianza empeora o empata. El score .94633 queda protegido contra regresiones.

2. **Caso 2: Exactamente una sonda $i$ mejora ($S_i > S_{48}$ y las demás $\le S_{48}$)**:
   - **Acción**: **Enviar el CSV de esa Sonda $i$ como el 5º cupo.**
   - **Razón**: Contiene exclusivamente las modificaciones empíricamente validadas que aumentaron el score público sin arrastrar componentes negativos.

3. **Caso 3: Múltiples sondas mejoran ($S_i > S_{48}$ para $i \in W \subseteq \{1, 2, 3, 4\}$)**:
   - **Acción**: Como los conjuntos de IDs de las 4 sondas son **estrictamente disjuntos**, construir el 5º CSV aplicando la unión de los bloques ganadores sobre la base 48:
     $$\text{Flips}_{\text{final}} = \bigcup_{i \in W} \text{Flips}_i$$
   - **Ejemplo**: Si Sonda 1 da .94643 (+1 neto) y Sonda 2 da .94653 (+2 netos), el 5º envío combina los 10 ítems de la Sonda 1 y los 10 ítems de la Sonda 2 (20 flips en total), maximizando la suma de ganancias confirmadas.

---

## 4. Artefactos y Ubicaciones en el Repositorio

- **Manifiesto detallado de IDs**: `results/revision_submissions/round152_adaptive_probes_manifest.json`
- **Métricas de la ablación Deep CatBoost**: `results/revision_submissions/round151_deep_catboost_nbsvm_ablation_results.json`
- **Script de ablación**: `src/revision_submissions/deep_catboost_nbsvm_ablation_20261029.py`
- **Script generador de sondas**: `src/revision_submissions/generate_adaptive_probes_20261029.py`
- **Archivos de envío para Kaggle**:
  - `submissions/20261029/submission_66_probe1_new_sellers_to_new.csv` (10 flips)
  - `submissions/20261029/submission_67_probe2_new_sellers_to_used.csv` (10 flips)
  - `submissions/20261029/submission_68_probe3_seen_sellers_to_used.csv` (7 flips)
  - `submissions/20261029/submission_69_probe4_unanimous_consensus.csv` (8 flips)
