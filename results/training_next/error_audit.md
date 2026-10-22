# Auditoría de errores y flips para el siguiente entrenamiento

## Alcance

Se inspeccionaron las cohortes históricas A–E y sus scores agregados, y las predicciones de confirmación de dos CatBoost entrenados el 21/10. El diagnóstico de generalización usa exclusivamente las etiquetas del train de 70.000 filas. No usa etiquetas recuperadas del test ni toma los flips inferidos como etiquetas individuales. El holdout de 14.000 filas ya fue inspeccionado; las conclusiones siguientes sirven para formular hipótesis, no para hacer otra selección y anunciar una mejora confirmada en ese mismo holdout.

Resultados reproducibles: `scripts/audit_errors_next.py`; archivos CSV en este directorio. Features de vendedor calculadas solo con las 56.000 filas de desarrollo correspondientes al modelo.

## Dónde están los errores del CatBoost nativo

El modelo tiene 958 errores sobre 14.000 casos (93,157% accuracy).

| Segmento | Filas | Errores | Accuracy |
|---|---:|---:|---:|
| Vendedor nuevo | 4.277 | 554 | 87,047% |
| Vendedor visto | 9.723 | 404 | 95,845% |
| Vendedor con un solo ejemplo previo | 1.454 | 119 | 91,816% |
| Vendedor mixto: proporción usada previa entre 10% y 90% | 1.458 | 185 | 87,311% |
| Vendedor previamente solo nuevo | 4.493 | 74 | 98,353% |
| Vendedor previamente solo usado | 2.658 | 115 | 95,673% |
| Cantidad inicial 1 | 9.802 | 876 | 91,063% |
| Cantidad inicial 3–10 | 2.178 | 41 | 98,118% |
| Cantidad inicial 11–100 | 1.010 | 8 | 99,208% |
| Sin texto de garantía | 8.455 | 706 | 91,650% |
| Con texto de garantía | 5.545 | 252 | 95,455% |

Los segmentos se superponen: no sumar sus errores. Cantidad 1 concentra 91,4% de los errores; vendedores nuevos concentran 57,8%. Esta estructura favorece un experto para cantidad 1 y vendedor nuevo/mixto, preservando el experto actual en inventarios grandes y vendedores puros.

Quitar seller_id globalmente no resuelve el problema: el modelo sin vendedor corrige 202 errores del nativo pero rompe 524 aciertos. En vendedores nuevos corrige 46 y rompe 57: hay complementariedad, pero no una mejora automática. Vendedores mixtos siguen beneficiándose del seller_id (87,31% contra82,72%); conviene agregar contexto de catálogo/categoría y regularizar el historial, no eliminarlo.

## Confianza y ejemplos difíciles

| Confianza máxima del modelo | Filas | Accuracy real |
|---|---:|---:|
| 50–60% | 524 | 55,34% |
| 60–70% | 581 | 63,51% |
| 70–80% | 706 | 77,62% |
| 80–90% | 1.370 | 85,91% |
| 90–95% | 1.415 | 93,92% |
| 95–99% | 4.115 | 98,35% |
| 99–100% | 5.289 | 99,87% |

La mala calidad predictiva de nuevos no implica sobreconfianza global: varios bins son ligeramente conservadores. La calibración debe medirse por segmento, clase y familia; no aplicar una temperatura general esperando elevar accuracy. Vendedores singleton sí presentan confianza media93,49% frente91,82% de accuracy.

Los errores de mayor confianza incluyen: reacondicionados/vintage/restaurados, repuestos y autopartes, coleccionables/libros, productos en cantidades grandes etiquetados usados, animales y contenidos digitales. Algunos casos parecen semánticamente ambiguos o potencialmente ruidosos. No relabelarlos manualmente usando intuición. Extraer representación semántica y modelar excepciones con evidencia de train.

## Qué aprendimos de los flips

Saldos públicos inferidos bajo máscara fija de21.000 filas y truncamiento: A+2; B−7; C−18; D−17; E−15. E se descompuso en E1−6, E2=0, E3−7, E4−2. En E2, top16 por CatBoost histórico−3 y complemento+3. Los saldos no identifican etiquetas ni pertenencia pública de las filas.

La única señal positiva amplia, A, tenía6,5/7 votos medios de semillas refit y apoyo alternativo histórico medio60,5%; B tenía0,31/7 y35,7%, aunque otros modelos nuevos apoyaban el flip. Esto respalda conservar robustez del refit y apoyo histórico como features del ranking. No demuestra que contar semillas equivalga a independencia ni que A transfiera a más casos.

E1 tiene apoyo alto de dos modelos textuales (76,6% y77,9%) pero bajo de árboles (32,6–37,3%) y perdió6 aciertos. C también se apoya principalmente en texto (63–64%) y perdió18. Es evidencia contra usar desacuerdo textual por sí solo como permiso para flip. D tuvo4,23/7 votos pero solo34% de apoyo CatBoost histórico y perdió17: consenso de semillas tampoco basta solo.

El top16 E2 por CatBoost histórico fue peor que el complemento. Es una muestra pequeña elegida adaptativamente, no evidencia de que deba invertirse ese ranking en casos nuevos. El saldo positivo del complemento queda como corrección pública localizada hasta que se valide una regla transferible en train.

## Features y experimentos prioritarios

1. **Precio relativo categoría×moneda**: logprecio menos mediana, percentil, z-score robusto, distancias a cuantiles. Añadir versión categoría×título similar cuando exista soporte suficiente. Ajustar estadísticas con fit únicamente; fallback a categoría/moneda global en categorías raras. Un mismo precio absoluto no representa igual nuevo/usado entre teléfonos, muebles y libros.
2. **Historial de vendedor suavizado y contextual**: proporción usada Bayesiana, logconteo, entropía, número de categorías, especialización, seller×category. Generar target encodings out-of-fold en fit; mapear tune/confirm con fit correspondiente. Nunca usar el target de la propia fila ni labels del test. Para nuevos, agregar características de catálogo sin target (frecuencias/dispersiones) con protocolo transductivo explícito y comparación inductiva.
3. **Texto más rico para casos difíciles**: caracteres n-gram para errores/abreviaturas, palabras uni/bigram con NB weighting o linear classifier; campos separados título/garantía/atributos y cross-fitted probabilidades por campo. Semántica: restaurado, antiguo, vintage, repuesto, original, reacondicionado, roto/para reparar, sellado/en caja, sin uso. Tratar negaciones y contexto (nuevo modelo, caja original) en vez de regex ingenua.
4. **Nested estructurado**: brand/model cuando existan; nombres/valores de atributos por separado; variaciones con combinación de atributos y stock; shipping tags, tags, payment IDs; dimensiones/relación aspecto de imágenes y su dispersión. En muchos registros attributes está vacío, por lo que su ausencia es parte de la señal. URLs e IDs pueden codificar artefactos: probar aparte y documentar; no mezclar identidad con semántica sin auditoría.
5. **Cantidades y fechas**: sold/initial, available/initial, inconsistencia stock, stock unitario, bins; duración entre created/start/update/stop y renovaciones como señal de exposición. Tratar censura y fechas del snapshot: no confundir venta observada posterior con variable prepublicación si el objetivo académico exige predicción prelisting.
6. **Ensamble condicionado**: probar pocos pesos predefinidos por seen/new y cantidad1; stacker regularizado sobre predicciones OOF + incertidumbre de vendedor. Entrenar la compuerta OOF; medir errores corregidos y rotos, no solo accuracy de cada experto.

## Protocolo para no confundir mejora con búsqueda repetida

- Conservar las particiones actuales para diagnóstico. Hacer búsqueda dentro de desarrollo usando CV de filas para la mezcla real y una evaluación seller-disjoint adicional para el problema de nuevos.
- Reservar una confirmación final nueva o un outer CV con predicciones completamente fuera de entrenamiento. Los folds históricos con IDs todos únicos no separaban vendedores.
- Comparar todos los candidatos sobre las mismas filas y métricas vistas/nuevas ponderadas según test (71,6233% vistos). Informar diferencia pareada, fixes/breaks y distribución por categoría.
- No entrenar ni calibrar con etiquetas de test recuperadas ni con saldos de probes convertidos en etiquetas. Los modelos base deben poder evaluarse sin ajustes de leaderboard.
- Hacer un registro cerrado de selección de hiperparámetros/pesos antes de confirmar; checkpoint con seed, filas, parámetros, tiempos, predicciones y hashes.
