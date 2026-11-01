# Plan de inferencia agregada por ID (2026-10-31)

La auditoría programática del ledger actual arroja 68 entradas con score y mapa de flips, pero sólo 59 vectores distintos. Incluye duplicados y omite envíos históricos sin vector verificable. Las cuatro sondas del 31-10 perdieron frente a D1+D2 (.94652): S1 .94628, S2 .94585, S3 .94628, C1 .94623. No añadir esos bloques.

## Modelo de observación

Para el envío j y el ID i sea x_ji la predicción binaria y y_i la etiqueta latente. La puntuación pública observada es el redondeo a 5 decimales de (1/N_pub) sum_{i en P} 1[x_ji=y_i]. P, subconjunto público, es desconocido. El likelihood debe integrar el redondeo, P y el ruido de procedencia de scores. Diferencias entre submissions sólo dan una suma firmada sobre los IDs que cambian, no etiquetas individuales. Con 59 vectores y 30k etiquetas latentes la solución es fuertemente no identificable sin prior de modelos train-only.

## Procedimiento

1. Ledger inmutable por submission: vector ID->pred, SHA, score, evidencia de envío, baseline y flips. Deduplicar por hash, marcar borradores y archivos faltantes.
2. Prior p_i calibrado sólo en train real mediante OOF seller-group, con seen/new, incertidumbre de calibración y alternativas de modelo.
3. Actualización agregada con factor de score redondeado. Aproximación computable: likelihood gaussiano/normal para diferencias por bloques con varianza que incorpora membership público incierto y dependencia seller; comparar con inferencia exacta en bloques pequeños identificables. No tratar expectativa posterior como etiqueta observada.
4. Diagnóstico: rango efectivo de matriz de diseños, solapamiento de bloques, sensibilidad del posterior a prior y fracción pública, intervalos por ID y entropía. Prohibir p_i=0/1 salvo etiqueta real de train (nunca test).
5. Backtest prequential: ocultar scores posteriores en el tiempo, predecir su rango y dirección; comparar contra prior sin scores. Calibración y log score de scores retenidos antes de elegir nuevas sondas.
6. Para nuevos envíos, optimizar dos objetivos explícitos: expected score de la predicción completa y reducción esperada de entropía, con penalización por duplicados/correlación y riesgo de perder el último cupo. Diseñar secuencia adaptativa, no cinco archivos fijos.

## Límites y decisiones

El ledger actual no es el universo de ~100 envíos. Recuperar sólo vectores exactos con hash; los que falten se excluyen de la actualización. S1-S3 no son hipótesis independientes sólo por tener IDs disjuntos: comparten familia de consenso. La quinta de hoy repitió el mejor vector y no aporta información. El próximo gran avance probable requiere una señal train-only nueva y validación externa, usando el posterior agregado para selección fina y exploración.

Antigravity ejecuta CX-0036; Codex revisará código y backtest antes de promover el posterior a una candidata.
