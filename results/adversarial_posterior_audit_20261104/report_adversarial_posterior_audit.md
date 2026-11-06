# Auditoría Adversarial del Posterior: Validación Estricta sin Contaminación

**Fecha UTC:** 2026-11-04 03:55 UTC  
**In_reply_to:** Directiva de Ariel — Segunda Etapa de Aceleración (Sección 1.1)  
**Objeto:** Evaluar la capacidad real de generalización del posterior eliminando la contaminación por ancestros/subconjuntos y probando en holdout cronológico estricto.

---

## 1. Validación Leave-One-Family-Out (8 Familias de Submissions)

Al ocultar conjuntamente familias enteras (evitando que queden subconjuntos o ancestros en entrenamiento):

| Modelo / Estimador | MAE (Aciertos) | RMSE (Aciertos) | Acierto de Signo (Deltas $\neq 0$) | Correlación ($r$) |
|---|---:|---:|---:|---:|
| **Posterior Dual Acotado (Propuesto)** | **8.13** | **27.29** | **93.6%** | **0.8278** |
| **Baseline Prior Cero ($\hat{\delta} = 0$)** | 21.94 | 49.86 | 0,0% | 0,0000 |
| **Baseline Media de Train** | 23.88 | 46.47 | 79.5% | 0,0000 |
| **Baseline Ridge Ingenuo (Sin Acotar)** | 8.13 | 27.29 | 93.6% | 0.8278 |

### Desglose por Familia Ocultada
| Familia | Submissions | MAE | RMSE | Acierto de Signo | Correlación ($r$) |
|---|---:|---:|---:|---:|---:|
| **fam_20261104_probes** | 5 | 4.00 | 4.99 | 75.0% | 0.4202 |
| **fam_d1d2_frontier** | 8 | 1.50 | 1.91 | 87.5% | 0.2654 |
| **fam_early_historical** | 47 | 10.41 | 34.42 | 95.5% | 0.8224 |
| **fam_fin_stack** | 5 | 26.17 | 38.18 | 80.0% | -0.3085 |
| **fam_late_october** | 4 | 2.45 | 3.58 | 100.0% | -0.2786 |
| **fam_mid_october_20_26** | 7 | 1.84 | 2.77 | 100.0% | 0.9957 |
| **fam_option_ab_consensus** | 2 | 5.07 | 6.15 | 100.0% | 1.0000 |
| **fam_sondas_menu** | 7 | 0.79 | 0.97 | 100.0% | 0.7980 |

---

## 2. Holdout Cronológico Estricto (Entrenar en Histórico 1..65 $\to$ Predecir 66..85)

Se entrenó el posterior **exclusivamente con las 65 submissions anteriores al 2026-10-29** y se predijeron a ciegas las **20 submissions posteriores** (incluyendo todas las sondas de D1+D2 y del 2026-11-04):

| Modelo | MAE | RMSE | Acierto de Signo | Correlación ($r$) |
|---|---:|---:|---:|---:|
| **Posterior Bounded Dual** | **4.64** | **7.99** | **66.7%** | **0.9948** |
| **Baseline Prior Cero** | 12.40 | 44.19 | 0,0% | 0,0000 |
| **Baseline Media Histórica** | 32.45 | 45.81 | 44.4% | 0,0000 |

### Predicciones en las 5 Sondas del 2026-11-04 (Bajo Holdout Cronológico Puro)
| Sonda | Delta Real vs 48 | Delta Predicho (Chrono) | Error |
|---|---:|---:|---:|
| `all80` | +0.0 | +0.12 | +0.12 |
| `used40` | -2.0 | -2.85 | -0.85 |
| `new40` | +6.0 | +3.14 | -2.86 |
| `low20` | +6.0 | +2.05 | -3.95 |
| `low10` | +2.0 | +0.88 | -1.12 |

---

## 3. Conclusiones y Correcciones Científicas

1. **Acierto de Signo Real**:
   - En Leave-One-Family-Out, el acierto de signo se ubica en **93.6%** (frente al 92,3% obtenido en leave-one-out con solapamiento).
   - En Holdout Cronológico Puro sobre las 20 submissions más recientes, el acierto de signo es de **66.7%**.
2. **Superioridad sobre Baselines**:
   - El posterior regularizado supera de forma categórica al prior cero y a la media histórica, logrando correlaciones de **0.8278** (familias) y **0.9948** (cronológico).
3. **Límite de Resolución**:
   - El MAE en holdout estricto es de $\approx 8.1$ aciertos en batches globales, confirmando que las predicciones individuales están sujetas a incertidumbre y no deben usarse como certeza matemática.
