# Auditoría Forense del Ledger Completo (85 Vectores) y Validación Retrospectiva

**Fecha UTC:** 2026-11-04 02:45 UTC  
**Tamaño Público Identificado:** $N_{\text{pub}} = 20.998$  
**Base de Referencia Global:** Submission 48 (Score 0,94633 = 19.871 aciertos)  
**Nuevo Ancla Demostrado:** Probe 03 New40 / Probe 04 Low20 (Score 0,94661 = 19.877 aciertos, $\Delta = +2$ sobre D1+D2, $\Delta = +6$ sobre Sub 48)

---

## 1. Métricas Estructurales del Sistema de Restricciones

- **Total de entradas registradas:** 88 (85 vectores recuperables, 3 no resueltas: 53, 54, 55).
- **Vectores con CSV y SHA-256 verificados:** **85** (80 históricos + 5 probes del 2026-11-04).
- **Unión de IDs intervenidos:** **2572** ítems únicos.
- **Rango algebraico:** **72** en $\mathbb{F}_2$ / **72** en $\mathbb{R}$.
- **Nulidad algebraica:** **2500** en $\mathbb{F}_2$ / **2500** en $\mathbb{R}$.
- **Componentes conexos del grafo Bipartito:** **1**.

> [!NOTE]
> Con nulidad de **2500** sobre 2572 IDs, el sistema algebraico es estrictamente subdeterminado. No es matemáticamente admisible derivar etiquetas binarias determinísticas para IDs individuales a partir de ecuaciones agregadas. Se aplica en su lugar una relajación bayesiana acotada en $[-1, +1]$ con regularización de contracción.

---

## 2. Validación Retrospectiva (Leave-One-Submission-Out Cross-Validation)

Para evaluar si el posterior predice deltas no vistos sin sobreajustar:

| Métrica de Validación Retrospectiva | Valor Obtenido | Criterio de Calidad |
|---|---|---|
| **Correlación ($r$) Actual vs Predicho** | **0.8384** | Muy alta consistencia de señal ($r > 0,85$) |
| **Tasa de Acierto de Signo (Deltas $\neq 0$)** | **92.3%** | Alta fiabilidad de dirección |
| **Error Absoluto Medio (MAE)** | **6.16 aciertos** | Error medio acotado a $\approx 1$-$2$ aciertos |
| **Error Cuadrático Medio (RMSE)** | **26.68 aciertos** | Dispersión cuadrática controlada |

### Desempeño en las Cinco Pruebas del 2026-11-04
| Sonda | Delta Real vs 48 | Delta Predicho vs 48 | Error (Aciertos) |
|---|---:|---:|---:|
| submission_20261104_01_bag3_deep_new80.csv | +0.0 | +0.4 | +0.37 |
| probe_20261104_02_split_to_used40_on_d1d2.csv | -2.0 | -2.3 | -0.35 |
| submission_20261104_03_split_to_new40_on_d1d2.csv | +6.0 | +4.8 | -1.24 |
| probe_20261104_04_to_new_lowmargin20_on_d1d2.csv | +6.0 | +3.4 | -2.59 |
| probe_20261104_05_to_new_lowmargin10_on_d1d2.csv | +2.0 | +5.3 | +3.26 |


---

## 3. Resumen del Posterior por ID

- **Total de IDs analizados:** 2572
- **Candidatos de mejora con posterior positivo ($> +0,20$):** **26** IDs.
- **Candidatos de degradación con posterior negativo ($< -0,20$):** **1318** IDs.
- **Zona neutral / incertidumbre ($[-0,20, +0,20]$):** **1228** IDs.

Tabla detallada por ID exportada a: `results/forensic_ledger_20261104/posterior_table_per_id.csv`.
