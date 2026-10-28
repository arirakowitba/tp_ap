# Ronda 75 — predictor OOF de título aislado

Hipótesis distinta de términos seleccionados: el texto compuesto repite título y agrega metadatos; un predictor supervisado independiente entrenado **solo con título** puede aportar una señal menos contaminada para vendedores nuevos. Se añadió su probabilidad word 1–2 como una columna OOF por vendedor al HistGB+FastText de ronda 39. Se conservaron la arquitectura, las demás columnas, el umbral 0.475 y los tres cortes seller-group a/b/c. Selección de vocabulario y aprendizaje de etiquetas solo dentro de fit; no se utilizaron etiquetas de test ni scores públicos por ID. Los tres cortes ya se habían utilizado y se solapan.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC 95% bootstrap vendedor | Log loss confirm base → título |
|---|---:|---:|---:|---:|---:|
| a | −13 | 86/68 | +18 | [−7,+42] | .210628→.209709 |
| b | +15 | 104/89 | +15 | [−15,+47] | .205913→.205880 |
| c, configuración congelada | +5 | 72/77 | −5 | [−29,+20] | .215678→.216789 |

En confirm, los cambios hacia **new** dieron netos +11,+23,−14; los cambios hacia **used** +7,−8,+9. Es una dirección inestable y los tres IC incluyen cero. En el subconjunto previamente seleccionado de desacuerdos con CatBoost (254/293/273 filas), el título frente a CatBoost dio +5/+22/+18, pero este subconjunto no representa el confirm completo y no respalda una regla de test.

Auditoría: cada confirm tiene 5.590 vendedores; las intersecciones de a/b, a/c y b/c tienen 1.119, 1.119 y 1.118 vendedores. También comparten 2.868, 3.218 y 3.185 filas, respectivamente. No se pueden tratar como tres replicaciones independientes. Esta variante no es una mejora robusta y no se propone submission.

Referencia pública informada por Ariel: 48=.94633; líder=.95404. El historial 41–50 y la relación entre bloques quedan en los reportes de submissions del PR; no se deduce ninguna etiqueta por ID de esas diferencias agregadas.

Checkpoint reproducible: `round75_title_teacher_checkpoint_20261027.zip`, Library ID `libfile_bd50991c8c688191bd264661088e08de`, SHA256 `0766e7eaa399643526e83ad490bd9096c0018f8baa51cf4e516d82e76a92d9af`. Incluye código, pares completos por corte, bootstrap, métricas, fuentes auxiliares y manifiesto con hashes de dependencias. Requiere data.zip y checkpoint ronda39/50 identificados en el manifiesto. No hubo submission ni consumo de cuota Kaggle.
