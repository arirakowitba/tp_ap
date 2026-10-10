# Cierre de la ronda

Se completaron 49 configuraciones nuevas, además de las 44 conservadas de la ronda previa. No se alcanzó el objetivo del 95%. La submission recomendada sigue siendo `submissions/submission_optimized_ensemble.csv`, con 94.428% público informado por el usuario.

El mejor modelo nuevo obtuvo 94.0357% en selección y 94.0429% en la confirmación congelada (seed 731), frente a 94.2143% del candidato previo en esa confirmación. No se promovió la nueva ronda. `submission_v2.csv` es experimental. `final_ensemble.json` conserva una propuesta sin confirmación y no debe interpretarse como resultado validado.

La búsqueda está detenida. Los intentos incompletos no cuentan entre las 49 configuraciones: la representación sparse terminó por memoria (exit 137); los demás procesos pendientes se interrumpieron. Se excedió el límite de tiempo solicitado.

Los notebooks incluyen auditoría de 70.000 filas de entrenamiento y 30.000 de test, separación por familia, prevención de fuga, comparaciones, errores y exportación. El maestro permite reproducir el pipeline y limita sus subprocesses al presupuesto global configurado. Los notebooks entregados se ejecutaron en modo de reporte sobre los resultados existentes; no equivale a volver a entrenar todo desde cero.
