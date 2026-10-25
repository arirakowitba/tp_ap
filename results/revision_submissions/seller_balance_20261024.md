# Ponderación inversa por frecuencia de vendedor

Hipótesis: una segunda etapa HistGB entrenada con pesos proporcionales a 1/√(n de avisos del vendedor) puede transferir mejor a vendedores nuevos. Se comparó contra la misma etapa HistGB sin ponderar, con docentes de texto y variables OOF idénticos, en tres cortes agrupados por vendedor; umbral fijo 0.475. El peso se calculó solo en el ajuste. No se usó leaderboard ni etiquetas de test.

| Corte | N | HistGB | Balanceado | Blend 25% | Fixes / breaks blend | Log loss HistGB → blend |
|---|---:|---:|---:|---:|---:|---:|
| 883 | 14335 | .928776 | .929473 | .929264 | 17 / 10 | .183874 → .183329 |
| 2026 | 13493 | .918254 | .917735 | .917587 | 25 / 34 | .213349 → .211518 |
| 314159 | 13898 | .917974 | .917038 | .918693 | 31 / 21 | .209811 → .209170 |

El blend 25% ganó +7, perdió −9 y ganó +10 aciertos. La mejora en log loss es consistente, pero la accuracy no: no generar submission aún. El 50% se eligió explorando el primer corte y no confirmó, por lo que no debe declararse estable. Los cortes agrupados se solapan en etiquetas de train y no son tres confirmaciones independientes. El checkpoint contiene predicciones pareadas y scripts para volver a generar la variante; requiere `data.zip` original, `original_docs.pkl` del Sheet original y código `optimized/src` del checkpoint 30. Siguiente paso: evaluar una regla no lineal o uso de incertidumbre basado en probabilidades, congelando selección antes de un corte nuevo, y contrastar vendedor visto/nuevo cuando el split lo permita.
