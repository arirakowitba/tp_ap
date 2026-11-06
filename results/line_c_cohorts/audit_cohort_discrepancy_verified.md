# Auditoría de Discrepancias y Cohortes en Confirm 20261143

**Archivo auditado**: `results\line_a_text\paired_20261143.csv`  
**SHA-256**: `39e551da3db630a45d397bc1f729ddd00c10209cc5a3143984b557a197fb5783`  
**Filas confirm**: `12360` (9.000 vendedores vistos, 3.360 vendedores nuevos).

## 1. Identificación Exacta de los Modelos Comparados

1. **Línea A Standalone**:
   - Clasificador puramente textual: Regresión logística sobre TF-IDF multicanal sparse (171.459 variables: título n-gramas 1-3 palabra y 3-6 char, garantía y atributos).
   - **No incluye seller_id, precio, categoría, ciudad ni atributos tabulares**.
   - Accuracy en confirm: **0,85647** (10.586 correctos de 12.360).
2. **CatBoost Bag3**:
   - Ensamble multi-modal completo: Promedio de 3 semillas CatBoost (depth=6, lr=0.07) entrenadas sobre 101 features profundas + `seller_id` como categoría + TE + texto OOF.
   - Accuracy en confirm: **0,94733** (11.709 correctos de 12.360).
3. **CatBoost Single (Histórico Fuerte)**:
   - Modelo individual semilla 20261143. Accuracy en confirm: **0,94709** (11.706 correctos).
4. **Blend Predeclarado en Tune**:
   - Ponderación fija: $0,10 \times \text{Línea A} + 0,90 \times \text{Bag3}$, umbral $0,505$. Accuracy en confirm: **0,94757** (11.712 correctos).

## 2. Aclaración de las Cifras: 227 Fixes y 1.350 Breaks

La tabla de contingencia $2 \times 2$ entre **Línea A Standalone** y **CatBoost Bag3** sobre los 12.360 ítems de confirm es:

| | Bag3 Correcto (11.709) | Bag3 Incorrecto (651) | Total Línea A |
|---|---|---|---|
| **Línea A Correcto** | 10.359 (Ambos aciertan) | **227 (Fixes de Línea A)** | 10.586 |
| **Línea A Incorrecto** | **1.350 (Breaks de Línea A)** | 424 (Ambos fallan) | 1.774 |
| **Total Bag3** | 11.709 | 651 | 12.360 |

- **Origen del saldo negativo de -1.123**: No proviene de un comparador invertido ni de un bug de polaridad. Proviene de comparar un **modelo de texto puro** (accuracy 85,6%) contra un **ensamble multimodal completo de árboles** (accuracy 94,7%). Los 1.350 casos corresponden a ítems donde el árbol acierta apoyándose en el vendedor o metadatos y el texto solo no alcanza.
- **Valor de los 227 Fixes**: Demuestran que en 227 casos donde el ensamble de árboles falla, el modelo de texto tiene la clasificación correcta.
- **Razón del descarte del Confidence Router**: Cuando se intentó usar la certidumbre del texto ($|p_A - 0.5| > 0.2$) para intervenir en los casos donde el árbol dudaba ($|p_{\text{bag3}} - 0.5| < 0.1$), el ratio resultante fue de **28 fixes vs 29 breaks (neto -1)**. El router por margen no logra aislar los fixes de forma segura.

## 3. Desglose Pareado por Tipo de Vendedor

| Comparación | Fixes | Breaks | Net Total | Net Vistos (9.000) | Net Nuevos (3.360) |
|---|---|---|---|---|---|
| **Bag3 vs Single** | 35 | 32 | **+3** | +4 | -1 |
| **Línea A Standalone vs Bag3** | 227 | 1.350 | **-1.123** | -855 | -268 |
| **Blend Predeclarado vs Bag3** | 35 | 32 | **+3** | +4 | -1 |
