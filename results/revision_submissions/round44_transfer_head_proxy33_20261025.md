# Ronda 44 — límite del proxy 33 y cabeza de transferencia para vendedores nuevos

Fecha: 2026-10-25. Sin submission a Kaggle.

## Auditoría de la referencia pública

Se revisó el inventario 01–41 y los CSV exactos locales de 31 y 33. La 33 agrega 31 decisiones sobre 31: **12 `used→new` y 19 `new→used`**. El modelo completo más cercano disponible (`text_deep_cat_oof_fresh/test.npz`) todavía difiere de la 33 en **558/30.000** filas al umbral 0,5; el HistGB del mismo programa difiere en 582. No se encontró en el código y artefactos preservados una función determinista que extienda los 31 cambios de test de la 33 a filas de train. Por ello, evaluar un modelo contra estos proxies no estima el efecto público exacto de cambiar la 33. Tampoco se usaron sus decisiones como etiquetas de verdad.

## Nueva hipótesis

Se entrenó una **cabeza supervisada de transferencia** solo con etiquetas reales de vendedores reservados. En cada una de dos particiones disjuntas por vendedor, la base HistGB + FastText se había ajustado en ~42.000 filas de fit y ya tenía probabilidades para tune y confirm, ambos compuestos por vendedores ajenos a fit y entre sí. La cabeza HistGB regularizada usa esas probabilidades, predicciones HistGB previas, 40 vectores FastText y señales numéricas originales; no recibe identificadores ni etiquetas del vendedor de confirm. Produce OOF de tune mediante tres folds agrupados por vendedor, se reajusta en tune completo y predice confirm. Regla congelada antes de evaluar: dentro de `0,2 <= p_base <= 0,8`, mezclar 25% de cabeza y 75% de base; fuera de ese intervalo conservar la base. Umbral de clase fijo `p(used) >= 0,475`.

## Resultados pareados contra FastText + HistGB

| Partición | Segmento | n | Casos en banda | Corrige | Rompe | Neto | Log loss base → regla | IC 95% neto por vendedor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39002 | tune OOF | 13.787 | 1.743 | 20 | 26 | **−6** | 0,204014 → 0,203612 | `[−19,+6]` |
| 39002 | confirm | 14.533 | 1.806 | 18 | 19 | **−1** | 0,210628 → 0,209954 | `[−13,+10]` |
| 39011 | tune OOF | 14.335 | 1.858 | 44 | 30 | **+14** | 0,203370 → 0,202156 | `[−3,+31]` |
| 39011 | confirm | 14.490 | 1.957 | 36 | 44 | **−8** | 0,205913 → 0,205511 | `[−26,+11]` |

La reducción de log loss se sostuvo, pero la métrica de Kaggle (accuracy) no: ambos confirm quedaron negativos y todos los intervalos incluyen cero. Entrenar la cabeza con etiquetas de tune significa que los resultados OOF de tune se usan para diagnóstico, mientras confirm sigue siendo la comparación decisiva. Los dos cortes comparten datos y la búsqueda acumulada de modelos reduce la interpretación de los intervalos.

## Decisión

No generar ni enviar submission con esta cabeza. Conservarla como evidencia de una mejora probabilística limitada a vendedores nuevos. La candidata FastText de 34 flips permanece solo para revisión; su validación sigue siendo contra HistGB y no contra la política completa de la 33. Para una nueva base amplia se necesita una arquitectura que cambie aciertos de forma consistente en confirm y una prueba pública autorizada de transferencia; no se debe construir un proxy de 33 entrenando contra scores agregados como si fueran etiquetas por ID.

## Reproducción

`round44_new_seller_transfer_head.py` produce `round44_transfer_pairs_{first,second}.npz` y reporte JSON; `round44_bootstrap.py` ejecuta 2.000 remuestreos por vendedor con semilla 44044. El checkpoint incluye `data.zip`, textos originales, vectores FastText exactos, pares base de ronda 39, código, fuentes compartidas, resultados y hashes. `data.zip` SHA256 `f8e78574061cbd75197896fa1cd08a0c78af3cb5d73e7e2597df4ab43b82d574`. No se usaron etiquetas de test ni cuota de Kaggle.
