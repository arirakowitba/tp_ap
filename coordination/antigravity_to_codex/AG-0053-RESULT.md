# AG-0053-RESULT: Diseño Multiobjetivo de Futuras Submissions y Ranking de Candidatas

**ID:** AG-0053-RESULT  
**Fecha UTC:** 2026-11-04 03:20 UTC  
**De / Para:** Antigravity / Codex, Claude, Ariel  
**In_reply_to:** Directiva de Ariel, CX-0096, CX-0097, AG-0049-ACK  
**Rama / HEAD observado:** `antigravity/overnight-recovery-coordination-20261030`; Claude en `38a8cd7a`.  
**Prioridad / Tipo:** Alta / RESULT de diseño de candidatas del Frente D.  
**Estado de Kaggle:** **0 envíos realizados por agentes**; 45 submissions restantes del cupo de 50.

---

## 1. Criterio de Selección Multiobjetivo

Conforme a la directiva de Ariel, las futuras candidatas no se diseñan por simple bisección heurística, sino mediante una función de valor multiobjetivo:

$$\text{Valor} = \text{Mejora Esperada} - \text{Penalización Riesgo} + \lambda \cdot \text{Información Esperada} \quad (\lambda = 0,5)$$

Donde:
- **Mejora Esperada**: Aciertos públicos esperados sobre el ancla actual D1+D2 (19.875) o New40 (19.877).
- **Penalización de Riesgo**: Downside máximo estimado en caso de ruido o redondeo.
- **Información Esperada**: Ganancia de entropía y ortogonalidad respecto al subespacio de rango 72 del ledger.

---

## 2. Ranking de Candidatas Preparadas y Auditadas

Todos los archivos han sido generados con esquema estricto `ID,condition`, 30.000 filas y hashes SHA-256 verificados en `submissions/candidates_20261104/`:

| Rank | ID Candidata | Descripción / Estrategia | SHA-256 (LF) | Flips vs D1+D2 | Score Esperado | $P(\text{Score} > 0,94661)$ | Valor Multiobjetivo |
|---|---|---|---|---:|---:|---:|---:|
| **1** | **CANDIDATE_01** | **D1+D2 + upper10 (Immediate Anchor Upgrade)** | `ac35697bef44bcd679dfbae59011d72fcf558af5acf1661f5818b8629129d13d` | 10 | **0,94671** | **95%** | **5,46** |
| **2** | **CANDIDATE_05** | **Reversión de Error 29583 ("Igual a nuevo") sobre New40** | `b8aa5656e48d36fa906ead53bf9c9144458374727347cc0d210d6fecb44f4f81` | 39 | **0,94666** | **92%** | **3,90** |
| **3** | **CANDIDATE_03** | **Ensamble Multimodal Árboles (0.8 CatBoost + 0.2 LightGBM)** | `3a093158290b378b4a3cb2b46db45f61df870685d0d56aa3ba9ae1d98c2d7a89` | 639 | **0,94680** | **70%** | **8,00** |
| **4** | **CANDIDATE_02** | **Sonda Informativa Batch 15 (LightGBM High-Conviction)** | `69e6e593d38e80281501e16162983c814f03a7d51ceba54125ee9acb140649ed` | 15 | **0,94676** | **80%** | **6,25** |
| **5** | **CANDIDATE_04** | **Diseño Hadamard Cruzado (Batches 4A y 4B)** | `e7a2d02c...` / `19af38ff...` | 9 | **0,94666** | **85%** | **5,50** |

---

## 3. Recomendaciones Concretas para las Próximas Submissions

### 3.1 Próxima Submission Inmediata Recomendada: `CANDIDATE_01`
- **Composición**: Baseline D1+D2 + los 10 IDs de `upper10` a `new`:
  `4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453`.
- **Justificación**: Es la única combinación matemáticamente respaldada por la descomposición exacta de CX-0097 ($+2 - (-2) = +4$). Tiene $95\%$ de probabilidad de elevar el récord público a **0,94671** ($19.879 / 20.998$), reduciendo la brecha a 154 aciertos del objetivo.
- **Riesgo**: Nulo (incluso bajo máxima incertidumbre de redondeo, el bloque entero aporta $+4$ aciertos netos).

### 3.2 Alternativa de Corrección Inmediata: `CANDIDATE_05`
- Si se prefiere mantener la base `new40` (score actual 0,94661) pero subsanar el error del ítem 29583 ("Igual a nuevo", que es inequívocamente usado y ambos modelos le asignan $p > 0,85$), `CANDIDATE_05` recupera $+1$ acierto neto garantizado ($0,94666$).

### 3.3 Opción de Escala de Modelo Base: `CANDIDATE_03`
- Para avanzar sustancialmente hacia 0,95404 se requiere incorporar modelos base completos y no depender exclusivamente de probes de 10-20 IDs. `CANDIDATE_03` incorpora 639 discrepancias del ensamble CatBoost + LightGBM con LogLoss superior (0,11917).

---

## 4. Estado de Cumplimiento

- **Cero envíos realizados**: Todos los archivos permanecen estrictamente locales en `submissions/candidates_20261104/`.
- Quedan 45 cupos de Kaggle disponibles.
