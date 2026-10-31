# Plan de inferencia y diseño de sondas — 30-10-2026

## Estado verificable
Mejor público observado: combinación D1+D2 sobre48, 0.94652 (23 cambios disjuntos, SHA 57172f209ff5018d4ceef8c57172cc731828cc78eb9cb02dedccff4a51292502). Los modelos completos H-AVG y H-AVG-Plus obtuvieron 0.94514 y 0.94471. No transformamos diferencias agregadas en etiquetas por ID.

## Modelo de observación
Para dos submissions A/B que sólo difieren en un conjunto D, la diferencia de aciertos en el slice público es la suma de contribuciones por fila *si esa fila pertenece al público*. Si y_i es la etiqueta real, b_i la predicción de A y c_i la de B, la contribución es I(c_i=y_i)-I(b_i=y_i). El score publicado está redondeado; por tanto la observación es un intervalo para la suma dividida por N_public, no un valor exacto por ID. La pertenencia al slice público es desconocida y común a todas las submissions. Los scores son evidencia de restricciones agregadas, no ground truth individual.

## Inferencia probabilística permitida
1. Construir prior p_i sólo con train real: OOF por vendedor, calibración congelada, evaluación de log loss y calibración seller seen/new. Conservar múltiples modelos para sensibilidad.
2. Construir matriz de diseño de todas las submissions reales: ID, predicción, hash, score, fecha y baseline. Verificar CSV originales, tamaños, polaridad, duplicados y dirección de cambio. No completar entradas faltantes con supuestos de nombre o distancia de predicciones.
3. Posterior p(y_i=used | scores agregados, modelos, hipótesis sobre pertenencia pública). Integrar/redondear scores y pertenencia pública latente. Reportar sensibilidad a prior/modelo y a N_public; intervalos, entropía, effective sample size, no etiquetas duras.
4. Validación retrospectiva *precuencial*: ocultar una submission pasada, predecir distribución del delta de score y evaluar cobertura/calibración contra el score observado; repetir por familias de bloques. Evitar entrenamiento/selección usando la misma submission que se evalúa. Un posterior que sólo ajusta scores históricos no basta.
5. Diseño de siguientes sondas por utilidad esperada de información y potencial de mejora, condicionado en historia *anterior* a la sonda, con bloques disjuntos o factoriales identificables, tamaños suficientes para superar resolución/redondeo, y presupuesto de cupos. Optimizar utilidad para score final, no sólo entropía. No considerar los 23 flips positivos como labels individuales.
6. Validar una política propuesta en labels de train con seller-group fit/tune/confirm y contra mejor baseline fuerte; reportar paired accuracy/log loss, seen/new, bootstrap vendedor. Confirm reutilizado es desarrollo. Test sin etiquetas sólo diseño/diagnóstico.
7. Promoción: CSV 30k hash, mapa ID/target, diferencias exactas vs mejor0.94652 y vs48, guard que parsea 30000 registros e IDs consecutivos, y reporte de incertidumbre. Ariel sube Kaggle.

## Magnitud del objetivo
0.96000 − 0.94652 = 0.01348 de accuracy pública. Si el slice público tuviera 21.000 filas, serían ~283 aciertos netos adicionales; N_public debe verificarse. Las microsondas de 7/16 filas prueban señales pero no cierran esa brecha por sí solas. Priorizar mejora de modelo generalizable y uso de scores como restricciones probabilísticas auditadas, no pseudoetiquetado.
