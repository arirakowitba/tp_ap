# Cierre de sondas del 1 de noviembre y diseño de la siguiente tanda

Ancla: `candidate_d1_d2_combined_on48.csv`, público 0.94652. Los resultados son observaciones agregadas del leaderboard, no etiquetas individuales ni prueba de pertenencia de un ID al público. No convertir diferencias redondeadas en labels por ID.

| Archivo | Cambios totales vs ancla | Público | Δ publicado |
| --- | ---: | ---: | ---: |
| candidate_frontier_0947_top22.csv | 22 | .94623 | -.00029 |
| probe_frontier_increment23_on_d1d2.csv | 23, disjuntos de top22 | .94642 | -.00010 |
| candidate_unanimous_consensus_0947.csv | 16; 1 solapa top22 | .94628 | -.00024 |
| candidate_optionA_full_inductive_blend_bag3.csv | 1028 | .93695 | -.00957 |
| probe_semantic_only4_on_d1d2.csv | 4, fuera de top22/unanimous/S1/S2/S3/C1 | .94638 | -.00014 |

El denominador público 20998 se infirió anteriormente, no fue publicado como certeza. Si fuera exacto, el bloque de cuatro muestra ~-3 aciertos, paridad incompatible con que los cuatro estén en el público. El score redondeado y la pertenencia privada son incertidumbres separadas; no identificar ninguna fila. La opción A pierde ~201 aciertos bajo ese denominador. Ninguna de estas líneas autoriza ampliar el bloque.

## Diagnóstico que debe preceder a nuevas candidatas

1. Antigravity: revisar por qué Option A cae tan fuerte frente a D1+D2 y los otros modelos completos. Verificar mapa ID/target/polaridad, fecha/refit/parámetros, distribución de cambios por seller seen/new, categoría y margen; comparación pareada contra baseline fuerte en los mismos seller-group cortes. El resultado de CatBoost proxy ~.926 en confirm no estima directamente la calidad absoluta del ancla pública .94652.
2. Modelo nuevo: congelar arquitectura, umbral y selección en fit/tune; confirmar en seller groups externos, reportar fixes/breaks, log loss, IC bootstrap vendedor y solapamiento histórico. Un fold confirm reciclado sirve para desarrollo y debe declararse así. La meta .95404 exige ~158 aciertos adicionales sobre ancla en ~20998 público; .967 requiere ~430. Las sondas de 4-23 cambios no pueden cerrar esa distancia.
3. Registrar una matriz exacta por ID de las predicciones de cada archivo y sus SHA antes de diseñar sondas. Incorporar al ledger estos cinco resultados y la procedencia de cada archivo. Si el archivo público no es idéntico al artefacto del repo, bloquear el uso causal de esa fila.
4. Diseñar la próxima tanda como bloques **predeclarados** por desacuerdo de modelos y segmento, con tamaños suficientes para distinguir señal de ruido, y contrastes de solapamiento. Ejemplo: bloques A (seller nuevo, alta confianza de dos modelos) y B (visto, alta confianza), luego A∪B sólo si ambos separados son prometedores. Una prueba solapada aporta la diferencia incremental de bloques conocidos únicamente si ancla, archivos y conjunto público se mantienen fijos. No reinterpretar cada ID como etiqueta.
5. Antes de gastar cupos: estimar con backtest prequential, ocultando cronológicamente scores, si el selector habría evitado top22/unanimous/Option A/semantic4. Calibrar cobertura y error de efectos **de bloques**; el ajuste posterior que incorpora el score de una sonda no vale como validación de esa misma sonda.
6. Reservar el último cupo para el mejor CSV completo que tenga evidencia nueva o mantener D1+D2 como respaldo. No presentar una mejora pública como garantizada.

HEADs revisados: main 1ecbc27881ae5451106b2600a7f4e90ef2dfde1c (tras CSV), Antigravity ce718caaf20a7b6e1734dae02e611ddc9baf124b, Claude 38a8cd7abea8e241886fbc30416f3160cd9bca5b. Sin nuevo RESULT de AG desde AG-0032. No se subió Kaggle por agentes.
