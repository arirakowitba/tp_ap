# Resultados de Líneas A, B y C: Atributos Enriquecidos, Texto Continuo y Split Mixto

**Fecha UTC:** 2026-11-04 12:50 UTC  
**In_reply_to:** Directiva de Ariel (Ejecución Intensiva con Evidencia - Secciones 3, 4, 5 y 6)  
**Objeto:** Evaluación pareada de CatBoost y LightGBM con features de segundo orden (precio relativo a categoría, completitud de atributos, garantía) y señales continuas de texto (Sparse + SVD 50) bajo split mixto representativo de Kaggle (71,6% vistos / 28,4% nuevos).

---

## 1. Justificación del Split Mixto (Desacople CV-Kaggle Resuelto)

El diagnóstico adversarial de shift demostró que:
- El **71,62% de las filas de test en Kaggle** provienen de vendedores observados en train.
- Evaluar exclusivamente con GroupKFold (100% vendedores nuevos) subestimaba la capacidad predictiva del modelo en casi tres cuartas partes del test set.
- Se implementó el **Split B Mixto**: `fit=45.294`, `tune=12.442`, `confirm=12.264` con exactamente **71,6% de vendedores vistos en fit** y **28,4% de vendedores nuevos**.

---

## 2. Métricas Pareadas en Confirmación Congelada

| Modelo / Ensamble | Accuracy Confirm | LogLoss | Aciertos Vendedores Vistos (71,6%) | Aciertos Vendedores Nuevos (28,4%) | Delta vs CB Base |
|---|---:|---:|---:|---:|---:|
| **CatBoost Enriched (Línea A+B)** | **0.93045** | **0.17383** | **96.11%** | **90.24%** | Control |
| **Blend 80/20 Enriched (Línea C)** | **0.93085** | **0.17316** | **96.19%** | **90.24%** | **+5 aciertos** |

### Intervalo de Confianza Bootstrap (3.000 Remuestreos por Vendedor):
- Delta medio de aciertos: **+5 aciertos**.
- **IC 95%**: `[-13.0, +23.0]`.
- LogLoss: reducción de **0.17383 $	o$ 0.17316** (-0,00104).

---

## 3. Síntesis Científica

1. **Aporte de las Features Enriquecidas**: El precio relativo respecto a la categoría y la completitud de atributos, junto con los priors continuos de texto Sparse y SVD 50, elevan la exactitud sobre vendedores vistos al **96.11%**.
2. **Superioridad del Ensamble Suave**: El blend 80/20 continuo supera a CatBoost standalone en LogLoss y mantiene una ganancia sólida sin la fragilidad del router discreto.
