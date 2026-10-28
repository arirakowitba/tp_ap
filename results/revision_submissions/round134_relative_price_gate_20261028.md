# Ronda 134 — precio relativo sin etiqueta en compuerta de desacuerdos

HEAD Claude comprobado `612303c91d73645ccba5aa97504af43a13a64490`, sin cambio respecto a la revisión anterior. Hipótesis nueva de D: precio relativo al vendedor y a la categoría, sin usar condition. Se calculó log(precio), desviación respecto de mediana por vendedor y categoría, y conteo de avisos del vendedor sobre los 100k textos/avisos originales (train+test sin etiquetas test). Compromiso: estos agregados transductivos usan test sin label, permitido para diagnóstico pero deben evaluarse con cuidado frente a distribución real.

Sobre los pares seller-group de ronda39, se entrenó una compuerta HistGB pequeña con probabilidades HistGB+FastText y los cuatro contextos de precio. Cada tune se dividió por hash fijo de seller en fit de compuerta y selección; confirm seller-disjoint no se tocó para seleccionar peso. Mezclas .05/.1/.2/.35; regla: neto de aciertos >0 y log loss menor en selección de tune.

| Corte | Peso elegido | Confirm vs FastText | Log loss base→nuevo |
|---|---:|---:|---:|
| a | base | 0 | .210628→.210628 |
| b | base | 0 | .205913→.205913 |
| c | .2 | +3/14648 (18 fixes, 15 breaks) | .215678→.214872 |

Bootstrap por vendedor del neto de c, 2000 muestras, IC95 [−9,+15]. El beneficio es demasiado débil para promover el modelo o generar CSV. Los tres cortes históricos solapan y no son réplicas independientes. No hay pares CatBoost equivalentes para estos cortes. Código, índices, probabilidades pareadas y reporte: Library `libfile_b41020e6aae88191b633a3770842995d`, SHA256 `cdab4975d6471ca09665f00c9805fe54841b95ef0f101a9868b83d610235a44b`. No hubo etiquetas de test ni scores públicos convertidos a labels.
