# Plan de entrenamiento y pruebas — 30 octubre 2026

## Estado verificado al cierre
Mejor pública conocida48=.94633.72=.94623;73=.94600;74=.94614;75=.94633;76=.94633 (idéntica48). Registro canónico: registry_72_76_20261029.json; generador probado: src/revision_submissions/rebuild_registry_72_76.py. Los cuatro bloques son disjuntos.72–74 rechazados para combinación;75 empató y no demuestra mejora oculta.76 fue un duplicado sin información nueva: no repetir duplicados.

## Prioridad1 — benchmark común y restauración real
Propietario: agente de baseline.
1. Comparar scripts/artefactos de round13, deep151/153,156 y157, verificar polaridad, IDs, fit/tune/confirm, features, texto OOF, TE, árboles y umbrales.
2. Round156 omitió txt_sparse_p; la comparación directa contra deep anterior dio -189/-194 en idénticos confirm. Round157 incorpora texto pero obtiene .915/.911, no .924/.920. Su intersección con deep anterior pierde -33/-71 con43/52 discrepancias seen/new; ésa es sólo comparación diagnóstica por cambios de fit.
3. Reentrenar ambas referencias en EXACTAMENTE los mismos IDs y misma procedencia. Persistir matrices/cachés por hash y modelos. No volver a llamar restaurado al baseline sin reproducir métricas o explicar cada diferencia.
4. Benchmark de desarrollo: tune fijo para selección. Auditoría seller-group para nuevos; protocolo mixto explícito para vistos (vendedores pueden compartir fit sólo en segmento visto). Reportar ambos por separado y mezcla con pesos predeclarados. Ningún vendedor de confirm nuevo aparece en fit/tune.
5. Muchos confirm históricos ya fueron inspeccionados: llamarlos desarrollo/reutilizados, no vírgenes. Congelar configuración antes de una evaluación externa adicional; auditar solapamiento en filas y vendedores. Si no hay holdout no inspeccionado, reconocerlo y usar outer CV anidada predeclarada, sin venderla como datos totalmente nuevos.

## Prioridad2 — entrenar señales realmente complementarias
Propietario: agente de modelos.
Sólo iniciar búsqueda amplia después de recuperar el baseline. Usar70k etiquetas reales/original100k. Historial público guía la elección de familias y bloques agregados, nunca labels individuales ni targets de entrenamiento.
Elegir una hipótesis principal y una alternativa: modelo de texto entrenado sobre tokens/subwords originales con capacidad no reducida por SVD, o boosting sobre matriz profunda completa con interacciones nuevas justificadas por error residual. Evaluar un piloto y rendimiento/tiempo antes de escalar. Reutilizar features/cachés sin reutilizar predicciones con fuga.
No repetir NB-SVM profundo(+8/-1),SVD+MLP(+16/-10,LL peor),gate historial,fin_stack,NMF ni lexical aislado sin una nueva intervención identificable.
Optimización de hiperparámetros con presupuesto acotado, inner folds de fit y tune; selección de iteración fuera de confirm. Para stack: bases cross-fitted dentro de cada outer fit; TE y stats etiquetadas recalculadas dentro de cada fold.
Comparar pares vs CatBoost profundo,HistGB+FastText y otro boosting; accuracy,LL,fixes/breaks,seen/new,grupos por categoría/frecuencia,bootstrap vendedor,semillas y estabilidad de decisiones. Entrenar con todas las etiquetas sólo tras congelar configuración.

## Prioridad3 — preparar candidatos públicos para aprender y mejorar
Propietario: Codex/integración; otros agentes entregan evidencia.
Las tres pérdidas72–74 refutan esos bloques, no cada ID ni todo segmento. El empate75 no valida sus cinco targets. No expandir por margen bruto, nombre de consenso ni precisión léxica global. Para seguir investigando esos cambios: reproducir reglas en desacuerdos de train frente a baseline equivalente, distinguir correcciones de regresiones y cambiar la hipótesis antes de nuevas sondas.
Crear matriz por ID de predicciones test completas con procedencia,modelo,polaridad,seen/new y pertenencia a cada bloque público; catálogo distinto de borradores. Los archivos01–03/10 y scores26–30 siguen pendientes salvo verificación documental nueva.
Diseñar un menú adaptativo con:
- comparación de MODELO nuevo si existe ganancia etiquetada controlada;
- bloque complementario con desacuerdos estables entre fits;
- una prueba que discrimine una hipótesis y tenga decisión posterior predeclarada;
- cupo de reserva para adaptar tras scores;
- candidato final nuevo si hay evidencia; nunca enviar copia de48/ganador para llenar cupo.
No fijar cinco microsondas sólo para gastar cuota. Si se usa cuota exploratoria sin ganancia confirmada, identificar su incertidumbre y valor de información esperado. No prometer mejora ni óptimo global.

## Entregables antes de uploads
Código,config,seeds,versiones,hashes de datos/particiones/features,tiempo/memoria,modelos o checkpoints,predicciones pareadas con IDs,JSON métricas,manifest test con direcciones/segmentos y README de reproducción. Cada CSV30000 IDs únicos,en orden,labels new/used,hash,base exacta,lista de cambios y comparación con TODO envío público registrado. Detector de hashes duplicados y solapamientos; versionar candidatos, no sobrescribir.
Constructor final acepta mapa explícito archivo/hash/score; no indexar scores por posición de un manifiesto variable ni usar scores ficticios por defecto. Validar disyunción y conflictos. Empate no se promueve automáticamente. Con ganancias disjuntas, suma pública es válida sólo sobre igual base/subset/métrica y sujeto a precisión; no garantiza oculto.
Sólo Ariel sube Kaggle. Otros agentes pushean ramas propias; integrar después de revisar. Actualizar registro al recibir cada score.

## Decisiones de evaluación
Promover a refit si mejora accuracy pareada de forma consistente,LL sin deterioro material y sin regresión grave seen/new; IC y tamaño de muestra se interpretan junto con solapamiento y búsqueda múltiple. Un IC positivo contra base débil no reemplaza comparación con la mejor base. Ganancia pequeña/no estable sigue como investigación.
