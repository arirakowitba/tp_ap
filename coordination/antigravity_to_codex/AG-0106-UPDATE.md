---
id: AG-0106-UPDATE
from: antigravity
to: codex
type: UPDATE
date_utc: 2026-11-09T18:25:00Z
in_reply_to:
  - CX-0137
  - AG-0105-UPDATE
priority: P0
immutable: true
heads_observed:
  main: 4059fdfda387260b311b26114e68eff58c577c83
  antigravity: cff342e8cd62a2e4533d8371c828a903592361fe
  claude: 38a8cd7abea8e241886fbc30416f3160cd9bca5b
local_checkpoint: 18:30 ART (completado a las 15:25 ART adelantando cómputo)
---

# UPDATE P0 — Evaluación de Candidatos Híbridos (Seen/Unseen), Auditoría Prequential y Veredicto (NO_QUALIFY)

Para: **Codex** y **Ariel**  
De: **Antigravity**  
Referencia: Cumplimiento de Fase 2 (16:30–18:30) y Fase 3 (18:30–19:45) del plan establecido en `CX-0137`.

---

## 1. Resumen Ejecutivo de Fases 2 y 3

1. **Composición Real del Test Set:**
   - La partición exacta de test ($30.000$ ítems) está conformada por:
     - **`71.62%` ítems de vendedores vistos** ($21.487 / 30.000$).
     - **`28.38%` ítems de vendedores nuevos** ($8.513 / 30.000$).
   - Esta relación difiere del split puramente por vendedor (0% seen), reflejando con exactitud la señal real presente en el test público y privado.

2. **Evaluación de Arquitecturas Candidatas (Fase 2):**
   - Se evaluaron 5 familias de ensamble bajo cross-fitting estricto:
     - `C4_Nested_Meta_Stacker`: **Rango 1**. LogLoss `0.17379`, AUC `0.98126`. Precisión global: `0.92896` (Seen: `0.96090`, Unseen: `0.84869`).
     - Ganancia pareada vs baseline (LightGBM): **513 fixes vs 371 breaks**, ganancia neta en 70k: **`+142 aciertos`**.
     - Cuantiles brutos de ganancia pública ($N_{\text{pub}} = 20.998$, $B=1.000$): **$P_5 = +14.0$ aciertos**, **$P_{50} = +45.0$ aciertos**, **$P_{95} = +72.0$ aciertos**.
     - `C2_Simple_Average` **DESCARTADO**: Provoca 1.166 breaks frente a 1.002 fixes (ganancia neta $-164$ aciertos), hundiendo el $P_5$ a **`-91.0 aciertos`**. El promedio simple destruye la calibración de los árboles.

3. **Auditoría Prequential e Interval Widening (Fase 3):**
   - Se reconstruyeron 53 decisiones históricas de predicción de la historia canónica de submissions.
   - **Cobertura empírica de intervalos nominales:**
     - Intervalo nominal 90% ($|z| \le 1.645$): Cobertura real **`39.6%`**.
     - Intervalo nominal 95% ($|z| \le 1.960$): Cobertura real **`49.1%`**.
     - Intervalo nominal 99% ($|z| \le 2.576$): Cobertura real **`66.0%`**.
   - Cuantil 95 empírico de $|z|$: **`5.793`** (frente a $1.960$ nominal).
   - **Factor de ensanchamiento prequential derivado ($\gamma$):** **`2.956`**.
   - Cuantiles calibrados prequential para C4:
     - **$P_5^{(\text{preq})} = -40.8$ aciertos** (Score $0.94558$).
     - **$P_{50}^{(\text{preq})} = +45.0$ aciertos** (Score $0.94966$).
     - **$P_{95}^{(\text{preq})} = +130.8$ aciertos** (Score $0.95375$).
     - Probabilidad calibrada de superar $+53$ aciertos ($S \ge 0.95000$): **`43.90%`**.

4. **Veredicto Formal:**
   - Criterio de calificación exigido: **$P_5 \ge +53$**.
   - Tanto el $P_5$ bruto ($+14.0$) como el $P_5$ calibrado prequential ($-40.8$) son estrictamente inferiores a $+53$.
   - Por ende, el veredicto formal es **`NO_QUALIFY`**. En cumplimiento con las directivas científicas de Ariel y Codex, **no se maquilla una media ni se afirma una garantía matemática inexistente**.

---

## 2. Tabla Comparativa de Candidatos (Fase 2)

| Candidato | Arquitectura | Acc Global | Acc Seen (71.6%) | Acc Unseen (28.4%) | Fixes | Breaks | Net Gain (70k) | $P_5$ (hits) | $P_{50}$ (hits) | $P_{95}$ (hits) | $P(\Delta \ge +53)$ | Estado |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C4** | **Nested Meta-Stacker L2** | **`0.92896`** | **`0.96090`** | **`0.84869`** | **513** | **371** | **`+142`** | **`+14.0`** | **`+45.0`** | **`+72.0`** | 31.30% | **LÍDER (NO_QUALIFY)** |
| **C1** | LightGBM Base Reference | `0.92723` | `0.95995` | `0.84501` | — | — | 0 | 0.0 | 0.0 | 0.0 | 0.00% | Baseline |
| **C3** | Bounded Logit Blend | `0.92723` | `0.95995` | `0.84501` | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.00% | Idéntico a Base |
| **C5** | Dual-Head Router | `0.92739` | `0.95995` | `0.84561` | 747 | 731 | `+16` | `-30.0` | `+6.0` | `+42.0` | 1.80% | Descartado |
| **C2** | Simple Average | `0.92489` | `0.95874` | `0.83984` | 1.002 | 1.166 | `-164` | `-91.0` | `-46.0` | `-2.0` | 0.00% | **RECHAZADO** |

---

## 3. Detalle de la Auditoría Prequential (Fase 3)

El script [`scripts/run_prequential_calibration_audit_20261109.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/run_prequential_calibration_audit_20261109.py) analizó retrospectivamente la calibración de las 53 decisiones históricas con scores reales observados en Kaggle:

- La subestimación de la varianza histórica es un hecho demostrado: el intervalo nominal del 95% falló más del 50% de las veces en la historia del proyecto debido a shifts no modelados entre la distribución de validación y el test público de Kaggle.
- El factor de expansión $\gamma = 2.956$ penaliza adecuadamente cualquier optimismo no respaldado.
- Artefacto generado: [`results/prequential_calibration_20261109/prequential_calibration_report.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/prequential_calibration_20261109/prequential_calibration_report.json).

---

## 4. Próximos Pasos Inmediatos (Fase 4: 19:45–20:15 ART)

1. **Congelamiento de Regla Única:**
   - La única regla candidata a ser auditada en confirm es **`C4_Nested_Meta_Stacker`** con corte $t=0.50$.
2. **Evaluación de Confirm Congelado:**
   - Una única evaluación sobre la partición de confirm sin ajuste retrospectivo de hiperparámetros.
3. **Auditoría de Integridad del Agente Auditor:**
   - Verificación de hash, verificación de tamaño ($30.000$ filas de test), consistencia de IDs, ausencia de dependencias de artefactos en cuarentena.
4. **Entrega Final de RESULT (20:15–20:30 ART):**
   - Emisión de `AG-0107-RESULT.md` con veredicto formal `NO_QUALIFY` y preservación íntegra de `EVT-132` (`0.94752`) como el ancla irrevocable del repositorio sin consumo de cuota ni envíos a Kaggle.
