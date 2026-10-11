# Cierre de la ronda complementaria

La sesión se interrumpió. El último respaldo durable conserva **57 configuraciones con métricas**:5 de normalización,25 especialistas,11 bases/meta-modelos de stacking y16 búsquedas conjuntas. No hay evidencia de una confirmación final del blend ni un refit nuevo. Se conserva la submission cuyo **94,428% público** informó el usuario.

| Rama | Mejor selección local | Confirmación |
|---|---:|---|
| Normalización y tabular |94,0357%|Pendiente|
| Especialistas |93,9643%|Pendiente|
| Stacking aprendido,3 bases |93,8643%|94,0143%|
| Normalización + OOF + tabular |94,0571%|Pendiente|
| Referencia previa |94,0214% sobre estas filas|94,2143%|

La selección de la referencia proviene de un modelo entrenado con49.000 filas; los candidatos usan56.000. La comparación en confirmación utiliza56.000 para ambos. Las semillas42 y731 son particiones agrupadas del mismo dataset ya explorado, no un test externo intacto. El objetivo95% no quedó demostrado.

Los especialistas exactos corrigen17 errores e introducen14 con su umbral elegido; esa diferencia de3 incluye cambiar el umbral. Al fijar .495, empatan al modelo general. Los clusters sin normalizar se colapsaron; los ensayos con centroides L2 no mejoraron. El blend intermedio de94,0714% no es una selección final válida ni una mejora confirmada.

Se informó inferencia real de70.000 títulos con384 dimensiones y regresiones logísticas de71,5%,72,3643% y72,4286%. Sus predicciones no llegaron al respaldo. Los vectores no eran constantes, pero el backend cuantizado mostró pequeñas diferencias entre títulos duplicados según el batch. Los ensayos incrementales y el stacking ampliado quedaron sin resultados finales recuperables. No se fabrican métricas para completarlos.

Las14 solicitudes a imágenes devolvieron403: esta hipótesis quedó bloqueada por acceso y no se entrenó un modelo visual. ONNX Runtime fue rechazado por revisión automática debido a telemetría; se cambió a OpenVINO CPU, con inferencia local y pesos preentrenados congelados.

El notebook maestro ofrece full y report, presupuesto global y confirmación antes del refit. En esta entrega se verificó report, no se volvió a ejecutar full. El ilustrativo conserva18 figuras históricas y agrega la comparación de esta ronda. Caches/modelos se regeneran con los datos oficiales; no deben confundirse los estados pendientes con ejecuciones completas.
