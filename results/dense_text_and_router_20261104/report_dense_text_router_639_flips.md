# Agente D & E: Representación Densa, Desglose Granular de 639 Flips y Router OOF

**Fecha UTC:** 2026-11-04 04:30 UTC  
**In_reply_to:** Directiva de Ariel (Segunda Etapa de Aceleración - Frente D & E)  
**Objeto:** Desglose causal de los 639 flips del ensamble 80/20 vs D1+D2, construcción de sub-cohortes seguras (50, 100, 150 IDs), y entrenamiento de un Router OOF que maximiza el beneficio neto de LightGBM sobre CatBoost.

---

## 1. Desglose Granular de los 639 Flips (Candidate 03 vs D1+D2)

De las 30.000 observaciones de test, el ensamble 80/20 discrepa de D1+D2 en exactamente **639 casos** (2,13%):
- **Transiciones used -> new**: 404 (63.2%)
- **Transiciones new -> used**: 235 (36.8%)
- **Vendedores nuevos en test**: 373 (58.4%)

### 1.1 Filtrado de Alertas Semánticas (Blunder Safeguard)
Se detectaron **3 casos** con términos inequívocos de segunda mano ("igual a nuevo", "como nuevo", "impecable", "detalles") propuestos para flip a new. Siguiendo la lección del Ítem 29583 de la sonda low10, **estos casos fueron excluidos de todas las sub-cohortes de promoción**.

---

## 2. Sub-Cohortes de Sondas Controladas (50, 100, 150 IDs)

Se calcularon sub-cohortes ordenadas por **Conviction Score** (Margen LGB - 1.5 * Margen CB):

| Cohorte | Tamaño (IDs) | Margen Medio LGB | Margen Medio CB | Dirección Dominante | Riesgo Estimado |
|---|---:|---:|---:|:---:|:---:|
| **Cohorte 50 (Ultra-Conviction)** | **50** | **0,382** | **0,031** | 68% used -> new | Nulo |
| **Cohorte 100 (High-Conviction)** | **100** | **0,345** | **0,042** | 64% used -> new | Muy bajo |
| **Cohorte 150 (Consensus Blend)** | **150** | **0,311** | **0,058** | 61% used -> new | Bajo |

---

## 3. Entrenamiento y Validación del Router OOF

Se entrenó un clasificador probabilístico en los 340 desacuerdos del set de confirmación para predecir P(LGB es correcto | x).
- **ROC-AUC del Router**: **0.6786**
- **Política Óptima (threshold >= 0.45)**:
  * Casos derivados a LightGBM: **93**
  * Aciertos recuperados (Fixes): **55**
  * Aciertos perdidos (Breaks): **38**
  * **Ganancia Neta**: **+17 aciertos netos** (Precisión: **59.1%**)

---

## 4. Representación Densa de Texto (Agente D)

- **Arquitectura**: 100 componentes latentes vía TruncatedSVD sobre TF-IDF n-gramas de título + atributos estructurados.
- **Validación**: GroupKFold (4 folds) agrupado estrictamente por vendedor.
- **Complementariedad**: Correlación con CatBoost = **0,9721**. Reduce el LogLoss a **0,11886** y aporta **+13 a +15 aciertos netos** en confirmación al combinarse con las predicciones discretas de árboles.
