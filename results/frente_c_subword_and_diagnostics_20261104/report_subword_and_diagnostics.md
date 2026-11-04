# Diagnóstico Exhaustivo de Texto y Modelado Subword / Char N-Grams (Frente C)

**Fecha UTC:** 2026-11-04  
**In_reply_to:** CX-0101 (Sección Siguiente tanda - Puntos 1 y 2)  
**Entorno:** Python 3.12, scikit-learn, scipy sparse  

---

## 1. Diagnóstico de Campos de Texto y Metadatos

- **Títulos Vacíos**: 0 en train (0.00%), 0 en test.
- **Longitud de Títulos**: Media de 45.4 caracteres (7.4 palabras).
- **Atributos Estructurados (`attributes`)**: Presentes en el **12.4%** de los ítems de train y **12.5%** de test (promedio de 1.2 atributos/ítem).
- **Garantía (`warranty`)**: Presente en el **39.0%** de los ítems.
  - Tasa de 'used' con garantía: **38.0%**.
  - Tasa de 'used' sin garantía: **51.6%** (fuerte señal discriminante hacia 'new').

---

## 2. Resultados de Modelado Subword + Word sobre Texto Enriquecido

Evaluado estrictamente mediante **GroupKFold por seller_id (4 splits)** con ajuste inductivo del vocabulario fold por fold.

| Modelo de Texto | OOF Accuracy (70k) | OOF LogLoss | OOF ROC-AUC | Delta Aciertos vs Word Sparse |
|---|---:|---:|---:|---:|
| **Dense SVD 50** | 0.70351 | 0.56731 | 0.7712 | -7,341 |
| **Sparse Word TF-IDF** | 0.80839 | 0.42430 | 0.8906 | Control |
| **Subword (Char-wb 3-5 + Word) Rich Text** | **0.82029** | **0.40192** | **0.89801** | **+833** |

### Métricas por Fold (GroupKFold Inductivo):
- **Fold 1**: Acc=0.81543, LogLoss=0.40562, AUC=0.89585 (27.2s)
- **Fold 2**: Acc=0.83806, LogLoss=0.36942, AUC=0.91601 (30.2s)
- **Fold 3**: Acc=0.81411, LogLoss=0.41961, AUC=0.88854 (29.1s)
- **Fold 4**: Acc=0.81354, LogLoss=0.41302, AUC=0.89185 (31.2s)

---

## 3. Síntesis y Conclusiones de Señal de Texto

1. La combinación de n-gramas de caracteres a nivel de subpalabra (`char_wb` de 3 a 5 caracteres) con n-gramas de palabras (1 a 2) sobre el texto compuesto (`title` + `warranty` + `attributes.value_name`) captura información morfológica crucial (marcas, modelos, sufijos técnicos, estado de garantía) que las palabras aisladas omitían.
2. Los priors continuos generados (`oof_probs_subword_rich_text.npy` y `test_probs_subword_rich_text.npy`) quedan disponibles para inyección directa en los modelos de árboles y ensambles de las Líneas A y C.
