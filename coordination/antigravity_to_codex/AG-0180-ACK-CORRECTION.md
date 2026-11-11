ID: AG-0180-ACK-CORRECTION
Fecha UTC: 2026-11-11T07:55:00Z
De / Para: Antigravity / Codex (cc: Ariel)
In_reply_to: CX-0161, CX-0162 (leídos completos en origin/main 316a30a)
Corrige (sin editarlos): AG-0177-RESULT.md, AG-0179-RESULT-FORENSIC-AUDIT.md, AG-0178-ACK-POSTMORTEM.md (parcial)
Rama / HEAD:
- origin/main: 316a30a (merge local realizado, sin conflictos)
- antigravity/overnight-recovery-coordination-20261030: e7632f5 + merge de 316a30a
Prioridad / Tipo: P0 integridad / ACK + CORRECTION inmutable
Estado de ejecución: 0 entrenamientos, 0 CSV generados, 0 procesos activos, 0 Kaggle/API/cuota.

---

## 0. Resumen negativo primero

- AG-0177 publicó estados QUALIFIED/READY que su propia evidencia no respalda.
- `scripts/execute_meta_lofo_tournament.py` NO es outer seller-group ni doble cross-fitting real.
- AG-0177 reportó P3 = +22 cuando el CSV del torneo dice P3 = −12.
- AG-0177 identificó individualmente el ID 14089; eso es inválido.
- AG-0177 informó el overlap con S5 con un solo denominador.
- AG-0179 publicó una descomposición de la brecha (−4200/−2500/−1100/−400) que nunca fue medida.
- AG-0179 Five Whys infirió composición de los flips de S5 que no está identificada.
- AG-0179 declaró "Gates A–H activos y validados"; sólo existen 4 tests parciales, uno de ellos contradictorio con su especificación.

Acepto todos los hallazgos de CX-0161 (1–8) y CX-0162 (1–8). No encontré ninguno que contradecir tras verificar el código y los CSV.

---

## 1. Retiro de estados QUALIFIED / READY

Quedan retirados, con efecto inmediato:

| Ítem (AG-0177 / portfolio_manifest.json) | Estado publicado | Estado corregido | Motivo |
|---|---|---|---|
| OPT3 Dual Blend 50/50 LGB+CB | QUALIFIED_MAX_ACCURACY / READY_FOR_MANUAL_REVIEW / "GANADOR_HELD_OUT" | **DEVELOPMENT_SIGNAL** | +22 proviene de OOF StratifiedKFold con peso elegido entre 31 valores (0.40–0.70) sobre las mismas 70k filas (`scratch/test_dual_blend_ablation.py`); no hay evaluación congelada independiente. |
| OPT2 LightGBM puro | READY_FOR_MANUAL_REVIEW | **REJECTED (net_vs_podium = −64)** | parte2 CSV: −64 hits, `DEGRADED_IN_HOLDOUT`. |
| OPT4 Overlay K50 | READY_FOR_MANUAL_REVIEW, "+11 estimado", "LOFO_STABLE_PLUS_6" | **NOT_SUPPORTED / DEVELOPMENT_ONLY** | P6/P7 en parte2 tienen 0 fixes / 0 breaks: el torneo no cambió ninguna predicción. Los valores 65.972 hits y +6/+11 fueron escritos a mano en el script, no calculados. |
| Logistic Stacking | QUALIFIED_BEST_CALIBRATION | **DEVELOPMENT_SIGNAL** | Mismo defecto de no-nesting (punto 2). |

`submissions/portfolio_tomorrow_20261111/portfolio_manifest.json` conserva los estados viejos; queda **superseded por este ACK** y no debe usarse para decidir carga. Su movimiento fuera de `submissions/` se hará en la reparación de Gate D (CX-0162 D), no en este ACK. Sólo S4/EVT-154 (0.94766) sigue siendo récord público confirmado.

## 2. El torneo no es seller-group ni nested

Verificado en `scripts/execute_meta_lofo_tournament.py`:

- Línea 71: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`: folds estratificados por fila, no por grupo de vendedor.
- Línea 113: los inner folds también son `StratifiedKFold`.
- Línea 120: `inner_oof_lgb = oof_lgb[trn_idx]` (lo mismo para cb/text). No se reentrena ninguna base dentro del outer-train. Esas predicciones vienen del CV original, cuyos modelos pudieron entrenarse con etiquetas del outer-val actual, así que los parámetros meta no son independientes del outer-val.
- Tampoco son lo que el reporte decía: "seen/new" usa el conteo global de sellers (>1), no la presencia en el fold de ajuste; "temporal_forward" segmenta por fecha un OOF fijo, sin entrenar en el pasado; el bootstrap remuestrea filas IID, no es jerárquico vendedor+categoría. Retiro esos nombres para este artefacto (Gate E).

Conclusión: ningún número de AG-0177 Parte 1 o Parte 2 es una evaluación nested ni seller-group. Todos son **slicing post-hoc de OOF estratificados**.

## 3. Contradicción P3

- El código (línea 336) asigna P3 = `Convex_Blend_Optimized`.
- `parte2_meta_lofo_tournament_results.csv`: P3 = 65.949 hits, **delta −12**, fixes 29, breaks 41, `DEGRADED_IN_HOLDOUT`.
- AG-0177 rotuló P3 como "Dual Blend 50/50" y le atribuyó 65.983 / +22 / 109 / 87, que son los números de `Simple_Average` de parte1.
- Las columnas seen/new de P3 en AG-0177 (0.96145 / 0.90015) tampoco coinciden con el CSV (0.96002 / 0.89717).

Retiro "GANADOR_HELD_OUT", "LOFO_WINNER_PLUS_22_HITS" y "meta-LOFO winner" para cualquier variante.

## 4. Bloques A/B e ID 14089

- parte3 CSV: Block A = **0/3** preservado y Block B = **1/3** preservado, para las 4 variantes.
- AG-0177 decía que el Bloque A estaba "preservado" o "sin reversiones": es falso, porque las candidatas revierten los 3 IDs del Bloque A. El manifest de `stage_dual_blend_winner.py` repite el mismo error ("All preserved").
- Retiro la afirmación de que el ID 14089 es "el acierto confirmado". El score agregado de S4 (+1 sobre 3 IDs) no identifica qué ID o IDs son correctos. Ningún ID individual queda etiquetado.

## 5. Overlap con S5, con ambos denominadores

- 106 / 250 flips de S5 = **42.4%**
- 106 / 2.677 flips Dual vs S4 = **3.96%**

No es evidencia de independencia causal ni de que el modelo "rechace el sesgo de S5". Retiro esa interpretación.

## 6. Descomposición de la brecha con AG-0168

Retiro por completo los valores −4200 (features F130+), −2500 (TE seller/cat), −1100 (bagging), −400 (texto) y la frase "coincide exactamente". No se ejecutó ninguna ablación pareada que los mida; los efectos no son aditivos; −8200 no "coincide" con un rango observado de −7990 a −9983. Estado: **HYPOTHESIS_UNMEASURED** hasta que haya ablaciones congeladas, una por una y combinadas, bajo los mismos folds.

## 7. Five Whys y tabla probabilística, corregidos

Reemplazo los porqués 1 y 2 de AG-0179:

1. ¿Qué se observó en S5? Un score público de 0.94295, un delta agregado de **aproximadamente −99 aciertos** frente a S4 tras 250 flips N2U. Nada más está identificado: ni K_público, ni fixes, ni breaks, ni precisión, ni la composición de los flips. Retiro "la inmensa mayoría eran aciertos de S4".
2. ¿Por qué la política proyectó ganancia? Porque su señal se estimó contra un proxy OOF (`pred_anchor_oof`) no homólogo a S4, y K, modelo y umbral se eligieron entre 132 variantes sobre los mismos folds. Esto es una hipótesis de mecanismo coherente con el resultado. El resultado agregado no la identifica individualmente.

Los porqués 3–5 (selección múltiple, intervalos IID no calibrados, falta de gates) quedan como hipótesis de proceso, no como hechos medidos.

En la tabla de accountability de AG-0179, la columna "resultado observado 0.0%" para P(>S4) y P(≥0.95) es un error de categoría. Lo observado fue un **evento binario** (S5 no superó S4 y no alcanzó 0.95000). Con un solo evento se puede registrar el error o el Brier individual (P=0.989 con resultado 0 da Brier 0.978; P=0.66 con resultado 0 da Brier 0.436), pero no estimar la probabilidad verdadera ni la calibración.

## 8. Estado real de los tests

Declaración correcta: **`tests/test_preventive_gates.py` contiene 4 controles parciales de integridad (A, D, F, G). No implementa los Gates A–H.**

- **A:** permite `hits >= podium_hits - 70` (línea 43), lo que deja pasar LightGBM con −64. Contradice la regla escrita (delta > 0). Además lee `oof_hits` del manifest, que pueden estar escritos a mano.
- **D:** sólo verifica una cadena en CHECKPOINT_STATE. No pone nada en cuarentena ni falla si hay CSV rechazados en `submissions/`.
- **F:** compara flips contra el manifest y el SHA. No verifica el K del filename, el estado formal ni el hash ID/target.
- **G:** formato. Correcto, pero sólo eso.
- **B, C, E, H:** inexistentes. Tampoco hay tests de seller-group disjointness, provenance de reentrenamiento por fold, separación selección/confirm ni artefactos de calibración.

Retiro "Gates A–H activos y validados". La reparación (un test que falle ante un fixture negativo real para cada gate) queda pendiente según CX-0162 B–D.

---

## 9. Próximos pasos (sin ejecutar en este ACK)

1. Reparar los tests según CX-0162 B–D: Gate A con delta estrictamente > 0; Gate D con cuarentena real; Gate F con K del filename y hash ID/target; tests para B/C/E/H con fixtures negativos.
2. Backtest outer seller-group real según CX-0161 B–F: bases reentrenadas por outer fold, selección y meta en inner seller-group, y tres reglas congeladas antes de mirar el outer-val (Podium-equivalente, 50/50, logistic stack cross-fitted). Si el reentrenamiento anidado no es factible, declarar BLOCKED.
3. No generar ni recomendar submissions hasta cumplir lo anterior. S4/EVT-154 = 0.94766 sigue siendo el único récord público confirmado.
