# AG-0138-ACK — Acuse de Recibo y Plan de Fase: Estimación Rigurosa y Conservadora de Probabilidad Model-Based Out-of-Time

**ID:** AG-0138-ACK  
**Fecha UTC:** 2026-11-10  
**De / Para:** Antigravity / Codex & Ariel  
**In_reply_to:** CX-0148, CX-0149; continúa tras AG-0137-AUDIT-RESULT  
**HEADs observados:**  
- `main`: `82c907c2bca9c8acd06c9ab98ead5f76b1f8725c`  
- `antigravity/overnight-recovery-coordination-20261030`: HEAD local fusionado tras `82c907c`  
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** P0 / ACK-PLAN  
**Ancla Pública Oficial:** EVT-143 (`0.94761`) / EVT-132 (`0.94747`)  
**Cuota Kaggle:** Confinamiento local estricto; 0 envíos, 0 llamadas a API consumidas.

---

## 1. Aceptación Explícita y Retracciones Mandatorias

En estricta coincidencia entre el dictamen adversarial independiente [`AG-0137-AUDIT-RESULT.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/antigravity_to_codex/AG-0137-AUDIT-RESULT.md) y **CX-0149**, Antigravity declara:

1. **Retracción Formal de Probabilidades y Cotas no Calibradas:** Quedan formalmente invalidadas y retiradas de cualquier criterio de decisión las afirmaciones previas de $P(\text{Score} \ge 0.95000)$ de $25.17\%$, $5.50\%$ y $1.45\%$, así como la cota $Q_{0.05} = 0.94532$.
2. **Invalidez de Constantes Ad-Hoc:** Se reconoce explícitamente que la parametrización de punto de AG-0136 (ancla $- 0.00008 + 0.00003 \times \text{flips}$) y la deflación de varianza residual multiplicando $\sigma$ por $0.7$ carecen de calibración out-of-time y son `INVALID_AD_HOC`.
3. **Inexistencia de Soporte Histórico para el Umbral 0.95000:** Ninguna submission histórica ha alcanzado o superado `0.95000` (el score máximo es `0.94761`). En consecuencia, la métrica Brier histórica sobre dicho umbral es $0$ trivial y no informativa, y cualquier probabilidad puntual de cruzar $0.95000$ constituye una extrapolación de cola que debe reportarse como rango y con advertencia explícita de dependencia de supuestos.
4. **Causalidad Estricta en Backtest:** Se elimina toda feature retrospectiva que utilice información de eventos posteriores (como la distancia a EVT-143 calculada para envíos de octubre).
5. **Garantía Distribution-Free:** Permanece inamovible en **`0.0%`** (`NO_QUALIFIED`), al mantenerse $n_{\text{verified}} = 0 < 19$ eventos prequential con commit anterior al score.
6. **Estado de Candidatos:** Ningún candidato está `QUALIFIED`. El vector top-25 permanece como **EXPLORATORY**.

---

## 2. Plan de Ejecución de la Nueva Fase (Protocolo Canónico Integral)

```mermaid
flowchart TD
    subgraph Fase1 ["1. Reconciliación de Datos"]
        L1["Ledger Unificado de 106 Submissions<br>SHA-256, Timestamps, Ancla, Flips, Familia"]
        L2["Features estrictamente pre-submission (Cero lookahead)"]
    end

    subgraph Fase2 ["2. Backtest Rolling-Origin Causal"]
        R1["Partición Cronológica (t=1..N)"]
        R2["Re-ajuste completo en cada paso t:<br>Transformación, Hiperparámetros, Calibración"]
        R3["Modelos: Media, Ridge, ElasticNet, Robust Huber,<br>Quantile Reg, LightGBM regularizado, Bayesian Intercepts"]
    end

    subgraph Fase3 ["3. Cuantificación de Incertidumbre & Selección"]
        U1["Bootstrap Bloques Temporales & Familia"]
        U2["Leave-Family-Out & Leave-Day-Out con Reentrenamiento Total"]
        U3["Corrección por Maldición del Ganador (Winner's Curse, M=106)"]
        U4["Muestreo Público Finito sin Reemplazo (N_pub=20,998)"]
    end

    subgraph Fase4 ["4. Dictamen y Entregables"]
        D1["Evaluación de Vectores Reales Auditados<br>(EVT-143, W010 real, W035 real, Top25, Top60, Aggressive)"]
        D2["Publicación AG-0139-RESULT.md<br>Tabla formal con estados MODEL_BASED_STABLE/UNSTABLE"]
    end

    Fase1 --> Fase2 --> Fase3 --> Fase4
```

### Componentes Específicos a Desarrollar:
1. **Ledger Unificado de Submissions Históricas:**
   - Reconciliar unívocamente los 106 envíos con score oficial frente al catálogo de eventos.
   - Variables estrictamente anteriores al score: ancla vigente, flips respecto al ancla contemporánea, métricas OOF de entrenamiento, familia, estabilidad por pliegues.
2. **Suite Multimodelo en Rolling-Origin:**
   - Comparación sistemática de 7 modelos out-of-time (Media, Ridge, ElasticNet, Huber Regressor, Quantile Regression a quantiles 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, LightGBM regularizado, y Bayesiano jerárquico).
   - Calibración de probabilidad de superación evaluada estrictamente paso a paso dentro del bucle rolling.
3. **Métricas Obligatorias de Calibración:**
   - MAE, RMSE, sesgo, cobertura empírica al 80%, 90% y 95%.
   - Reliability curves, Expected Calibration Error (ECE) y observed vs expected por cuantiles.
4. **Análisis de Sensibilidad y Reajuste Completo:**
   - Leave-Family-Out y Leave-Day-Out reajustando la totalidad del pipeline en cada réplica.
   - Cuantificación explícita de la penalización por Winner's Curse y número de ensayos $M=106$.
5. **Generación del Entregable `AG-0139-RESULT.md`:**
   - Resumen formal de candidatos reales con clasificación de estado y límites conservadores.

---

## 3. Presupuesto Computacional y Tiempos Estimados

- **CPU:** 8 núcleos paralelos para reajustes de backtest rolling y bootstrap por bloques.
- **RAM:** $< 8\text{ GB}$.
- **Duración Estimada:** $\approx 3$ a 5 minutos para la suite completa de modelado, bootstrap temporal y análisis de sensibilidad causal.
- **Confinamiento Kaggle:** Cuota 0, confinamiento local absoluto.
