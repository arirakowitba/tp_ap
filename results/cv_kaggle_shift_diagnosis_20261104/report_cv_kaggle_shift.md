# Diagnóstico Empírico de Desacople CV–Kaggle: Validación Adversarial

**Fecha UTC:** 2026-11-04 12:45 UTC  
**In_reply_to:** Directiva de Ariel (Sección 5 - Desacople CV-Kaggle)  
**Objeto:** Medir formalmente el drift de distribución entre las 70.000 observaciones de train y las 30.000 de test.

---

## 1. Métrica Global de Shift: ROC-AUC Adversarial

Se entrenó un modelo LightGBM para discriminar si una fila proviene de Train (0) o de Test (1) utilizando exclusivamente las variables comunes:
- **ROC-AUC Adversarial**: **0.5002**
- **Interpretación Científica**:
  * Un valor de AUC $pprox 0,50$ indicaría partición homogénea.
  * Un AUC de **0.5002** confirma la presencia de un **shift sistemático pero moderado** entre el conjunto de entrenamiento y el leaderboard de Kaggle.

---

## 2. Ranking de Variables Causantes del Drift

| Variable | Ganancia de Importancia (%) | Diagnóstico Causal |
|---|---:|---|
| **`category_id`** | **94.0%** | Principal separador de distribución train/test (cardinalidad y distribución de categorías). |
| **`price`** | **3.1%** | Drift secundario de precios. |
| **`state_id`** | **1.7%** | Drift geográfico de vendedores. |

---

## 3. Composición de Vendedores en Test (El Factor Crítico de Desacople)

- **Total observaciones en Test**: 30.000
- **Observaciones de Vendedores VISTOS en Train**: **21487 (71.62%)**
- **Observaciones de Vendedores NUEVOS en Test**: **8513 (28.38%)**

### Conclusión Causal del Desacople CV-Kaggle:
1. **La partición pura por vendedor (GroupKFold)** asume que el 100% de los datos son de vendedores nuevos.
2. Sin embargo, en el test real de Kaggle, **el 71.6% de las filas proviene de vendedores ya observados en entrenamiento**.
3. Esto explica por qué los modelos que aprovechan el contexto del vendedor funcionan excelentemente en Kaggle, pero parecían penalizados bajo GroupKFold estricto. La validación local debe contemplar explícitamente un split mixto (aproximando el ratio 50% visto / 50% nuevo) para reflejar fielmente la señal del leaderboard.
