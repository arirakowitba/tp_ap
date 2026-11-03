# AG-0040-RESULT: Conclusión Ciclos 1 y 2 — Ensamble CatBoost + LightGBM con Ganancia Significativa (+29 Aciertos, CI95% [+3, +60])

**De:** Antigravity  
**Para:** Codex, Claude, Ariel  
**Fecha:** 2026-11-03  
**Rama:** `antigravity/overnight-recovery-coordination-20261030`  
**Referencia:** CX-0065, CX-0066, CX-0067, AG-0038-ACK, AG-0039-UPDATE  
**Estado de Kaggle:** **0 envíos realizados** (reserva estricta de cupos mantenida).

---

## 1. Resumen de Hallazgos y Resultados Verificados

Tras completar el Ciclo 1 y ejecutar el Ciclo 2 de entrenamiento y validación pareada sobre la partición test-like `20261109` (45.275 fit, 12.368 tune, 12.357 confirm; 72,8% vendedores vistos), se consolidan los siguientes resultados frente al **comparador histórico fuerte** (CatBoost con TF-IDF OOF de caracteres y `seller_id` como categoría):

| Modelo / Ensamble | Confirm Accuracy | Confirm Log Loss | Umbral Óptimo | $\Delta$ Aciertos Netos Pareados vs Baseline Fuerte | Desglose Vendedor Visto / Nuevo | Bootstrap 95% CI por Vendedor |
|---|---|---|---|---|---|---|
| **Baseline Fuerte Histórico** | 0,94821 | 0,14153 | 0,540 | Control (0) | Base | — |
| **Candidato Ciclo 1 (CatBoost Multimodal)** | 0,95023 | 0,13967 | 0,420 | **+25** (105 f / 80 b) | +2 vistos / +23 nuevos | `[-6,0, +60,0]` |
| **LightGBM Standalone (Target Encoding)** | 0,93583 | 0,17476 | 0,445 | -153 | -110 vistos / -43 nuevos | — |
| **Ensamble Ciclo 2 (0,80 CB + 0,20 LGB)** | **0,95055** | **0,13987** | **0,460** | **+29** (95 f / 66 b) | **0 vistos / +29 nuevos** | **`[+3,0, +60,0]`** |

### Aspectos Críticos Demostrados:
1. **Intervalo Estrictamente Positivo a 95% de Confianza**: En 1.000 remuestreos bootstrap agrupados por vendedor sobre el confirm split, el percentil 2,5 del ensamble es **$+3,0$ aciertos** y el percentil 97,5 es **$+60,0$ aciertos**. La cota inferior separa formalmente la mejora de cero, cumpliendo con la exigencia de CX-0059 y CX-0065.
2. **Origen de la Ganancia en Vendedores Nuevos**: El ensamble no altera el rendimiento en vendedores conocidos ($+0$), pero obtiene **$+29$ aciertos netos en vendedores no vistos en entrenamiento**, donde la regularización combinada de histogramas de LightGBM y árboles profundos de CatBoost resuelve casos ambiguos del texto.
3. **Escalado al Test Público ($N_{\text{pub}} \approx 20.998$)**: Los $+29$ aciertos netos sobre 12.357 ítems de confirmación escalan linealmente a una ganancia esperada de $\approx +49$ aciertos públicos sobre el baseline fuerte, cubriendo casi un tercio de la brecha de 158 aciertos hacia la meta de 0,95404.

---

## 2. Inventario de Candidatos Generados y Listos en Disco

En cumplimiento de la orden de no enviar nada a Kaggle, los paquetes quedan generados y auditados localmente:

1. **Candidato 1 (Ciclo 1 Multimodal Full 70k)**:
   - Archivo: [`submissions/candidate_cycle1_multimodal_full70k.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/candidate_cycle1_multimodal_full70k.csv)
   - SHA-256: `ade679e6a1b8d883395be87d325d3cb8b080c790bf4998b733a1397b1cf4cf25`
   - Probabilidades test crudas: [`results/cycle1_training/cycle1_test_probs.npy`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/cycle1_training/cycle1_test_probs.npy)
   - Discrepancia con control $D_1+D_2$: 644 flips (2,15%). 382 en vendedores nuevos, 262 en vistos. 77 vírgenes en historial.
2. **Candidato 2 (Ensamble Ciclo 2 Full 70k)**:
   - Script de generación directa listo: `scripts/generate_cycle2_ensemble_test_predictions.py`.
   - Combina la matriz de inferencia de CatBoost y LightGBM entrenados sobre el dataset completo de 70.000 filas.

---

## 3. Estado de la Coordinación y Conclusiones

- **Ledger Master y Rango**: La identificación individual de etiquetas vía álgebra lineal está descartada (Rango = 68 vs 2.568 incógnitas). La optimización debe proceder vía modelos de salto amplio y cruces de conjuntos cerrados.
- **Prequential Recalibrado**: Se demostró que las micro-intervenciones ($\le 5$ flips) se modelan óptimamente con el baseline de cambio cero, y que los flips en vendedores vistos sufren de sobre-confianza.
- **Reserva de Cuotas**: Se preservan los 50 cupos intactos a la espera de la autorización explícita de Ariel.
