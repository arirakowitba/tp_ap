# Rondas 90–93: memoria del vendedor sin fuga propia (27-10-2026)

**Estado:** mejora fuerte en validación etiquetada para vendedores vistos; sin submission. Checkpoint reproducible en Library: `libfile_193b11fec1d08191a498dd513e4d53d7` (SHA-256 `fe70f17697ddbc3bc8c25664ade377d7732063f5384439fa2f6958ba390c02d2`). Incluye código, pares de predicciones, diagnósticos, dependencias con hashes y resultados.

## Hipótesis y método

La segunda etapa HistGB+FastText previa recibía estadísticas del vendedor al predecir avisos, pero el entrenamiento por grupos veía una prior para la mayoría de vendedores y no aprendía bien esa señal. Agregué probabilidad de clase del vendedor con suavizado 20 y conteo. En cada fila de **fit** se restan su propia etiqueta y su propia observación: `(suma_y_vendedor - y_fila + 20*prior)/(n_vendedor - 1 + 20)`. En tune/confirm se usan exclusivamente etiquetas de fit. Se mantienen las dos señales de texto OOF y el mismo umbral .475.

Primero entrené sin restricción en una partición por filas y una partición por vendedor. La variante sin restricción perdió 66 aciertos en confirm de vendedores nuevos del corte por grupo (log loss .210628 → .220856). Fijé entonces la regla **usar la variante LOO solo si el vendedor aparece en fit; en otros casos conservar HistGB+FastText**. La regla precedió a las particiones de confirm 91 y 92; no fue elegida mirando sus etiquetas.

## Confirmaciones pareadas con etiquetas reales

Cada partición tiene 42.000 fit, 14.000 tune y 14.000 confirm. `neto` = arreglos menos roturas frente a HistGB+FastText; la compuerta deja intactos los avisos de vendedores nuevos.

| Partición | Vistos en confirm | Arreglos / roturas | Neto vistos | Bootstrap 95% por vendedor | Log loss vistos, base → compuerta | Nuevos |
|---|---:|---:|---:|---:|---:|---:|
| R90, semilla inicial | 9.195 | 250 / 111 | +139 | +98 a +180 | .147741 → .103972 | 4.805, sin cambio con compuerta |
| R91, 20261030 | 9.150 | 269 / 116 | +153 | +110 a +198 | .152500 → .111568 | 4.850, sin cambio |
| R92, 20261032 | 9.240 | 269 / 109 | +160 | +115 a +205 | .145636 → .104522 | 4.760, sin cambio |

Los tres cortes comparten filas y vendedores: **no son tres réplicas independientes**. Los incrementos aparecen desde vendedores con un solo aviso en fit hasta vendedores con 21 o más; el detalle por frecuencia y los pares están en el checkpoint. La comparación etiquetada con CatBoost para el segmento de vendedores vistos no está disponible en los artefactos históricos; sus pares publicados cubren sobre todo desacuerdos entre vendedores nuevos. No atribuyo a esta prueba una victoria sobre CatBoost.

## Reentrenamiento completo y diagnóstico de test

Con las 70.000 etiquetas de train y los textos originales de 100.000 avisos, el modelo completo cambia 907 de 30.000 decisiones respecto de HistGB+FastText; las 907 corresponden a vendedores vistos. La política resultante difiere de la submission 33 en 736 avisos, y el CatBoost histórico discrepa en 608 de los 907 cambios. **Esto mide desacuerdo, no aciertos.** No hay etiquetas de test ni comparación validada con la política exacta de la submission 48, mejor pública informada .94633; el líder informado es .95404. Por volumen y desacuerdo, no recomiendo enviar estos 907 cambios todavía.

## Límites y próximo control

La ganancia es para vendedores con historia etiquetada; la generalización a vendedores nuevos sigue sin mejorar. Hay que reconstruir la política exacta 48 en validación, comparar pares contra ella y contra CatBoost en avisos vistos, y repetir con separación de vendedores que controle el número de observaciones disponibles por vendedor. Los scores públicos se mantienen como resultados agregados y nunca se interpretan como etiquetas individuales. No se gastó cuota Kaggle.
