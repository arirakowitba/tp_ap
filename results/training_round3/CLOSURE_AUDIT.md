# Cierre de tanda — auditoría

Verificados los cuatro modelos base: vectores de confirmación alineados exactamente con las 14.000 filas de la partición y 30.000 probabilidades test finitas en [0,1]. Existen locks de selección. Los test de bases sin ID explícito dependen del orden canónico del productor; los dos exports finales sí guardan ID y lo verifican.

Ambos CSV finales tienen 30.000 IDs únicos, secuenciales 1–30.000, clases new/used, y etiquetas exactamente iguales al umbral 0,5 de sus NPZ. Hashes coinciden con los informes. La confirmación histórica contiene 4.147 filas únicas del conjunto de confirmación.

Inspección del compositor: ambas selecciones C se escriben antes de cargar confirmación; no hay sustitución de entradas faltantes, flips ni etiquetas públicas en las entradas. Esta inspección del código no reconstruye por sí sola la cronología de ejecución.

Límites: se reutilizaron holdouts e hiperparámetros sobre tune; el CV interno es condicional. No corresponde elegir retrospectivamente una familia por su mejor confirmación ni interpretar diferencias pequeñas como ganancia garantizada. No hay score público observado de estos candidatos.
