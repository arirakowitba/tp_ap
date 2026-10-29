# Ronda 144: texto original condicionado por categoría (29-10-2026)

Fuente: Sheet original íntegro, 70.000 train. Split histórico de ronda 13: 56.000 fit, 14.000 valid, índices/etiquetas alineados con el checkpoint CatBoost. Se entrenó regresión logística `liblinear` con TF-IDF word 1–2 global (52.280 términos) y unigramas prefijados por categoría (44.839 términos), vocabularios ajustados sólo en fit. No se usaron scores públicos ni etiquetas de test.

Texto solo: accuracy .857786 y log loss .339644; CatBoost fuerte, mismas filas: .944214/.147403. Seen .881728 vs .962673; new .803812 vs .902603.

Valid histórico se dividió por vendedor con `GroupShuffleSplit(seed=144)` en tune 7.130 y confirm 6.870, sin vendedores compartidos. Pesos texto predeclarados 0/.025/.05/.10/.20. En tune, peso 0 y .025 empataron accuracy .945863, pero .025 empeoró log loss .145031→.146801; mayor peso bajó accuracy. El criterio accuracy y desempate log loss eligió **peso 0** antes de abrir confirm. Confirm queda idéntico a CatBoost: .942504/.149865, cero flips. No hay señal para submission.

Este confirm interno pertenece a un corte histórico ya explorado; no es validación virgen. Se descarta esta variante como complemento lineal. Código, pares por ID y resultados: Library `libfile_edc44d39c87c819188d4d4ae3e6cb5ba`, SHA256 ZIP `512ea50d073f352c8b54334695078b49b4d824298fcfdfe4663b57de6e686865`.
