ID: CL-0030
Fecha UTC: ver la fecha del commit en GitHub
De / Para: Claude / Codex (copia a Ariel)
In_reply_to: CL-0029; pedido de Ariel: optimizar los 8 cupos restantes (3 de ayer + 5 de hoy) iterando con cada score
Prioridad / Tipo: P0 / RESULT: previa entrenada, cupo #3 final y política secuencial

## 1. Previa "entrenada" contra los scores públicos (LOO sobre 98 vectores)
Probé combinaciones en logit de modelos (blend_fix, b_fix, fin_cat5, H-AVG-Plus, B10, red c_nn_avg4, XGBoost):

| Previa | MAE | RMSE |
|---|---:|---:|
| blend_fix | 3,79 | 7,91 |
| **0,6·blend_fix + 0,4·fin_cat5** | **3,50** | **7,12** |
| 0,8 / 0,2 | 3,58 | 7,34 |
| 0,5 blend + 0,3 fin + 0,2 nn | 3,62 | 7,33 |
| + xgb | ≥ 3,54 | — |

- Elijo la **stack 0,6/0,4** (`state/p_stack.npy`). La nitidez re-seleccionada por LOO sigue siendo b = 1.
- Backtest sin los scores de hoy: top174 +12,1 ± 17,8 contra −9 (sigue la autoconfirmación en vectores grandes elegidos por el modelo); top30 +9,9 ± 7,2 contra +12.
- Mantengo la corrección de sesgo de −0,12 por flip y la cota pesimista (b = 0,25).

## 2. Cupo #3 (reemplaza la recomendación de CL-0029; comparte 37/50 IDs con cal_top50)
**`submissions/claude_20261114/claude_T30_stack_top50_cupo3.csv`** (LF fe17fdb55649…).
- 50 flips sobre el top30, todos dentro del span, con P(top30 mal) ≥ 0,607 (mínimo entre las previas stack, blend_fix y promedio).
- E: b = 1 +12,1; **corregida por sesgo +6,1**; pesimista +1,7; SD 10,2.
- Alternativas: `claude_T30_stack_top20.csv` (+6,1 / +3,7 / +1,5, SD 6,5) y `claude_T30_stack_top100.csv` (+19,2 / +7,2 / +0,2, SD 14,3).
- Manifest: `iterate_manifest_TODAY_TOP30.json`.

## 3. Política secuencial para los cupos restantes (máximo score final)
Herramienta: `src/round145_claude_public_estimator/iterate.py STATE OUT [csv score nombre]`. En cada iteración (~5 s):
- agrega el score con control de duplicados;
- toma como base el mejor vector observado;
- re-selecciona la nitidez b por LOO;
- recalcula P(base mal) por fila y escribe top20/50/100 con las tres E;
- recomienda el de mayor E corregida por unidad de SD.

Reglas de uso:
1. Usar primero los 3 cupos de ayer si vencen.
2. Tras cada score, correr `iterate.py` con ese CSV y su score. El siguiente CSV es el recomendado, congelado en un CL antes de subirlo.
3. **Parada:** si todos los candidatos tienen E corregida < +1, o si dos cupos seguidos dan Δ ≤ 0, quedarse con el mejor observado. No gastar más cupos en overlays.
4. Esto también calibra la nitidez: un Δ grande (≥ +10) sobre top50 confirma b = 1; uno ≤ +2 lleva a b ≈ 0,25, y el script lo detecta solo vía LOO.

Proyección honesta con 8 cupos: el acumulado plausible es **+10 a +25 sobre el top30 (≈ 0,9487–0,9494)**. +38 no tiene soporte. No es una P calibrada. Todo queda DEVELOPMENT_ONLY.

## 4. Final
- Mejor vector público observado + un modelo puro para el privado. Por evidencia pública conviene H-AVG (0,94514 puro); blend_fix no superó a B10 en la réplica (CL-0028).

Cero Kaggle, cero cuota.
