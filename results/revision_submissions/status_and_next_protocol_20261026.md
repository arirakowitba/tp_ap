# Estado y dirección tras rondas 41–56 — 2026-10-26

Fuente del leaderboard: tabla suministrada directamente por Ariel el 2026-10-26; no es una consulta nueva a Kaggle. Ariel figura quinto con .94628 y 66 entradas; último .94623. Líder Maximo Lladhon .95404 (11 entradas), segundo Nicolás Caveda .94985, tercero Gonzalo Mariano Bühler .94895 y cuarto Agustín Elías Sánchez .94671.

Brechas desde .94628: cuarto .00043, tercero .00267, segundo .00357, líder .00776. Si hubiera exactamente 21,000 filas públicas, equivaldrían aproximadamente a 9, 56, 75 y 163 aciertos netos. Esa cantidad de filas sigue sin verificarse. Los scores solo informan resultados agregados.

Este documento actualiza el estado pendiente de la estrategia del 2026-10-25: 41 obtuvo .94628; 42/43 .94614; 44 .94619; 45 .94623. Conservar 41 como mejor pública. No extender los bloques de 42/43/45 por consenso o margen.

## Decisiones acumuladas

- Compuerta logística HistGB/FastText: descartar selección de flips; perdió frente a CatBoost en tres confirmaciones y mostró inestabilidad entre fits en test.
- Blend CatBoost/FastText y pesos por dominio: no promover por accuracy inconsistente.
- Regla used→new FastText y consenso CatBoost/FastText: el cuarto corte aporta evidencia débil frente a CatBoost y amplios intervalos por vendedor; no ampliar ni ajustar márgenes sobre el mismo confirm. Usar la corrección del umbral CatBoost .5 documentada en round54_threshold_correction_20261026.md.
- RBF text specialist de round56: descartar por pérdidas de 1,078/1,236 aciertos tune/confirm y peor log loss. La comparación carecía de variables estructuradas presentes en el baseline; no atribuir la pérdida solo al kernel.

## Próximo experimento predeclarado

La siguiente arquitectura debe conservar las variables estructuradas y los predictores OOF comunes para evitar una comparación confundida por pérdida de información. Evaluar una representación supervisada complementaria con esos mismos inputs, comparada con HistGB+FastText y CatBoost en el mismo fit. Fijar umbrales y reglas usando solo tune; abrir confirm una vez. Conservar predicciones y bootstrap por vendedor, segmentos por dirección, margen y categoría. No convertir test en conjunto de evaluación etiquetado. No gastar cuota Kaggle sin instrucción de Ariel.

No hay garantía de alcanzar .95404 o .960. Los últimos resultados justifican cambiar la familia de representación, no seguir refinando las mismas pocas filas.
