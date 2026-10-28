# Ronda 127: vecinos semánticos FastText

Se probó KNN por coseno (k=5,20,100) sobre embeddings FastText de los textos originales. Fit y holdout son disjuntos por vendedor; cada holdout se divide en tune y confirm sin solapamiento. Se mezclaron probabilidades vecinas con el baseline HistGB+FastText de ronda 39 en pesos .05,.10,.20,.35. Regla congelada: seleccionar en tune solo un peso con neto de accuracy positivo y menor log loss.

En los tres cortes a/b/c, ninguna combinación pasó esa regla. Por ello, el modelo seleccionado fue el baseline en cada corte: neto 0 y log loss confirm .210628/.205913/.215678. Las pruebas diagnósticas de combinaciones no seleccionadas muestran que los pequeños netos de accuracy positivos en confirm (hasta +7/+4/+12) venían con log loss mucho peor (.230356/.216870/.234405 para las respectivas mejores por accuracy). No se promoverá KNN ni se generará submission.

Código, probabilidades pareadas por fila, índices de vendedor fit/tune/confirm y resultados: checkpoint Library libfile_38db05cc25dc8191a9151b20ab2283aa, SHA256 a9390e607bde628f0eff40a0ee869521a7e4aeae29a1f8bde0b4c9e7c9c6f018.

Los tres cortes históricos pueden solaparse en filas; no se interpretan como tres réplicas independientes. No se usaron etiquetas de test ni scores públicos para ajustar el modelo.
