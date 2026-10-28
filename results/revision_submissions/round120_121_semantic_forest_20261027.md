# Rondas 120–121: modelos base alternativos (2026-10-27)

Se entrenaron solo con las 70k etiquetas de train; los vectores FastText provienen de los 100k textos originales sin etiquetas test. Tres particiones seller-group preexistentes a/b/c con tune/confirm, etiquetas del holdout nunca usadas en fit. Todos los holdouts son vendedores nuevos por construcción. Comparador común: HistGB+FastText con textos word/char OOF, threshold .475. Selección de peso usando exclusivamente tune: mejora estricta tanto en accuracy neta como log loss; confirm leído después.

R120: k-means 24 sobre FastText original 100k sin etiquetas, 24 regresiones logísticas locales sobre FastText+precio+cantidad, mezcla con base. a/b seleccionaron peso 0; c seleccionó 5%, confirm 20 fixes/16 breaks = +4/14.648, log loss .215678→.213095. Bootstrap por vendedor 95% del neto [−7,+16], 5.590 vendedores. No promover.

R121: bosque aleatorio de segunda etapa (450 árboles, min leaf8, max features .5, max samples .85) sobre las variables exactas HistGB+FastText con teachers word/char OOF agrupados por vendedor. Tres tunes eligieron peso 0; ninguna ganancia confirm seleccionada. No promover.

Los confirms se solapan a/b 2.868, a/c 3.218, b/c 3.185 filas: no son réplicas independientes. El ensayo no aporta un modelo de vendedor visto que reemplace la política pública 48. Tampoco prueba superioridad frente a CatBoost por ausencia de pares CatBoost de esos cortes. No se generó CSV ni gastó cuota. La caída 53=.94114 sigue siendo razón para no seleccionar cambios basados en el corrector seller LOO contra la base más débil.

Siguiente: nueva hipótesis con representaciones complementarias al texto y validación seller-group y temporal, con comparación pareada contra las bases comunes; exigir ganancia clara antes de gastar uno de los cuatro cupos restantes. Conservamos la submission 48 exacta como mejor pública observada .94633.

Checkpoint Library: `libfile_65b135f89cb48191bda39966afb34d87`.
