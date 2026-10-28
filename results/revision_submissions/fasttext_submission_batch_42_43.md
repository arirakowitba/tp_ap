# Tanda FastText 42–43 — protocolo congelado

Base exacta: submission 33 (score público 0.94614 reportado). Fuentes: checkpoint de ronda 39 y auditoría de ronda 40; la candidata original de 34 cambios conserva SHA256 8df48f148afcb9f432c9e8bf51f967031b358e8b84ff33f74e7f9929255980c6. No hay score de la 41 informado. La usuaria subió 42 y 43 y reportó 0.94614 para ambas; la 41 permanece sin score.

| N.º | CSV | IDs cambiados vs 33 | Dirección | SHA256 |
|---|---|---:|---|---|
| 42 | submission_42_fasttext_new_margin_ge005_13_sobre33.csv | 13 | used→new, vendedor nuevo | ab1cdfd90f6a3704f2a730a8a9f71f639141b5a12712634423163679bd0e8bae |
| 43 | submission_43_fasttext_new_full34_sobre33.csv | 34 | used→new, vendedor nuevo | 8df48f148afcb9f432c9e8bf51f967031b358e8b84ff33f74e7f9929255980c6 |

42 es un subconjunto estricto de 43. El bloque adicional de 43 tiene 21 cambios. Los CSV tienen 30.000 IDs únicos, orden 1..30000 y columnas ID,condition; todos los demás valores son idénticos a 33. 42 exige `.475-p_fasttext_used >= .05` en los 34 candidatos de revisión. El manifiesto exacto de IDs está en `fasttext_submission_batch_manifest.json`.

En dos particiones seller-group con tune/confirm disjuntos por vendedor, la regla used→new contra HistGB produjo netos +20/+26 y +20/+42 para el grupo completo. Al separar retrospectivamente por margen FastText .05, el segmento >=.05 dio +13/+26 y +16/+22; el <.05 dio +7/0 y +4/+20. Este umbral se examinó después de esos resultados, por lo que la 42 es una **prueba nueva de transferencia**, no una confirmación independiente. La validación compara con HistGB y no reproduce la política completa de 33: ningún efecto público está garantizado. Los 34 son demasiado pocos para cerrar por sí solos la brecha al líder.

**Feedback 42 (reportado por la usuaria, 2026-10-25):** 0.94614, empate a cinco decimales con 33; delta público redondeado 0.00000. Los 13 cambios no lograron ganancia neta observable. No se deducen aciertos por ID; el empate y el redondeo tampoco prueban que cada caso esté calibrado o que haya exactamente igual número de aciertos y errores sin conocer el tamaño y la evaluación públicos. No se verificó la página de Kaggle de forma independiente.

**Feedback 43 (reportado por la usuaria, 2026-10-25):** 0.94614, empate a cinco decimales con 42 y 33; delta público redondeado del bloque adicional de 21 igual a 0.00000. La familia FastText de cambios directos sobre 33 no mostró ganancia neta observable en ninguno de sus dos bloques. No se deducen etiquetas individuales ni una cantidad exacta de aciertos y errores; puede haber redondeo. No se verificó la página de Kaggle de forma independiente.

**Siguiente decisión:** conservar 33 como mejor base medida. Reservar 41 (seis cambios disjuntos) mientras se prioriza una nueva hipótesis evaluada contra la política de 33 en train con cortes de vendedores y segmentos. La 41 permanece una prueba independiente posible si hay cuota suficiente, pero seis cambios ofrecen alcance muy limitado para cerrar la brecha. Registrar cualquier score posterior, hash e IDs sin inferir etiquetas individuales.
