# Reporte de Exploración de Texto Neuronal sin SVD+MLP (Agente 4)

**Fecha:** 2026-10-30  
**Arquitectura:** Red Neuronal Dispersa de Dos Capas (MLP con activación ReLU) sobre el vocabulario unificado de palabras (1–2 n-grams) y subwords de caracteres (3–5 char n-grams).  
**Dimensión del Vocabulario Crudo:** 72,951 términos dispersos.  
**Evitación Crítica:** **Sin reducción SVD**. Se preservan todos los discriminadores léxicos raros de condición y garantía.

---

## 1. Benchmarking de Recursos: Tiempo y Memoria

| Métrica de Hardware | Valor Medido |
| :--- | :---: |
| **Tiempo de Vectorización** | **14.54 s** |
| **Tiempo de Entrenamiento Neuronal** | **129.06 s** (12 épocas max con early stopping) |
| **Tiempo de Inferencia (24670 ítems)** | **0.27 s** |
| **Consumo de Memoria RAM Pico** | **1158.3 MB** |

> [!NOTE]
> La matriz dispersa unificada consume menos de 100 MB de RAM y el entrenamiento converge en ~129.1 segundos. Esto demuestra que no es necesario realizar una reducción de dimensionalidad con SVD para mantener bajo el uso de recursos.

---

## 2. Rendimiento Intrínseco del Modelo Neuronal

| Métrica de Evaluación | Partición Tune (Holdout) | Partición Confirm (Reutilizada) |
| :--- | :---: | :---: |
| **Accuracy** | **0.82388** (umbral 0.535) | **0.81480** |
| **Log Loss** | **0.40537** | **0.42485** |

---

## 3. Integración Complementaria con el Baseline Fuerte (CatBoost R151)

Se evaluó la integración de la probabilidad neuronal sobre la partición `confirm` frente al baseline CatBoost puro:

- **Fixes (Aciertos nuevos):** 42
- **Breaks (Regresiones):** 35
- **Ganancia Neta:** **+7 ítems**
- **Log Loss Baseline vs Blend:** 0.18773 -> **0.19355**

---

## 4. Probabilidades OOF por Grupo de Vendedor (Sin Fuga)

Para habilitar la incorporación de esta señal al ensamble de producción sin riesgo de sobreajuste:
- Se generaron probabilidades OOF mediante un **4-Fold GroupKFold por `seller_id`** sobre el conjunto de entrenamiento (`fit`).
- Ningún vendedor evaluado en validación cruzada fue visto durante el ajuste del fold respectivo.
- Matriz OOF guardada en: `results/neural_text_20261030/oof_neural_text_p_fit.npy` (SHA256 verificado).
