# Auditoría de cobertura del ledger y rango de las pruebas — 2026-11-03

Fecha UTC: 2026-11-03 18:48. Fuente: rama Antigravity `1c9d5d72c67ab0ab3b0eb79643b5b9753f80fad3`, `submission_ledger_master_83.json`, `inventory_coverage_and_unresolved.json`; CSV recientes locales de 30.000 filas con hash LF cotejado con el ledger; baseline D1+D2 con hash LF `57172f209ff5018d4ceef8c57172cc731828cc78eb9cb02dedccff4a51292502`. Ningún score público se trató como etiqueta individual.

## Resultado verificable

- El ledger contiene **83 filas con score**, no 83 vectores completos con hash. Las rondas 53, 54 y 55 declaran respectivamente 411, 204 y 65 cambios, pero sus `flipped_ids` están vacíos y su `sha256_vector` es nulo. Esos tres experimentos no aportan ecuaciones por ID hasta recuperar los CSV exactos.
- Las cinco filas recientes A15, A11, strong5, top2 y final_global_evidence tienen `sha256_lf` y diferencias frente a D1+D2, pero carecen de `flipped_ids` frente a submission48 y de `sha256_vector`. Se reconstruyeron **sólo para esta auditoría** desde los cinco CSV completos, sus hashes LF y el baseline D1+D2. Los cinco hashes LF coinciden con el ledger.
- Los `n_flips_vs_48` declarados para esas cinco filas son **28, 24, 24, 23, 27**; los CSV exactos implican **36, 32, 28, 25, 25**. Sus distancias frente a D1+D2 sí son 15, 11, 5, 2, 10 como declara el ledger. Hay que corregir los cinco contadores y persistir los vectores completos, sin alterar la evidencia histórica de scores.
- Sobre **80 vectores recuperables** (75 filas históricas con vector y cinco CSV recientes), hay **2.568 IDs** intervenidos, **rango real 68** y **nulidad 2.500**. El rango se reprodujo con eliminación binaria GF(2) y álgebra lineal NumPy real; ambos dieron 68. La cifra algebraica de AG-0041 es correcta **sólo después** de recuperar los cinco CSV recientes. Su dimensión `78×2568` y la afirmación «83 entradas con vectores y hashes» no describen la matriz realmente usada. Los scores de 53–55 quedan fuera del cálculo de rango. Rango bajo y nulidad alta no autorizan inferencias puntuales de etiquetas.
- El inventario `inventory_coverage_and_unresolved.json` contiene **39** registros `real_submissions` con archivo y hash y **83** filas scored en el ledger. Los 39 hashes aparecen en el ledger. Sus contadores adicionales `56 candidates_unsubmitted`, `13 duplicates` y `6 inconsistent` son categorías de inventario; no existe allí una correspondencia fila por fila de los **106 uploads** informados por Ariel. La igualdad 83+17+6=106 de AG-0041 sigue siendo una hipótesis sin manifiesto de Kaggle: no demuestra cuáles 17 uploads faltan ni si los seis rechazados/duplicados están ya contados en las 83 filas.

## Impacto en la decisión

El ancla pública sigue siendo D1+D2 = 0.94652. El ledger sirve para contrastes agregados de vectores exactos, pero no para asignar certeza por ID a 106 uploads. Bloquear cualquier promoción basada en los 23 uploads sin conciliación y cualquier supuesto label individual inferido por scores.

## Acciones solicitadas

1. Exportar o reconstruir un manifiesto de Kaggle de los 106 intentos, con timestamp, nombre de archivo, estado (completed/error), score y hash del CSV **efectivamente enviado**; vincular cada intento a un vector canónico o marcarlo `unresolved`. Un mismo vector puede tener varios uploads. No fabricar 17/6 registros para cuadrar la suma.
2. Recalcular y persistir `flipped_ids`, `n_flips_vs_48` y `sha256_vector` de las cinco filas recientes desde los CSV completos verificados; mantener un changelog inmutable de la corrección. Recuperar 53–55 sólo si existen sus CSV originales y hashes.
3. Dar prioridad de cómputo al nuevo modelo train-only contra el baseline fuerte word+char OOF+seller. Esta auditoría no constituye mejora predictiva ni justifica gastar cuota.
