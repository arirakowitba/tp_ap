# Ronda 49 — consenso e incertidumbre sin etiquetas inferidas

Checkpoint reproducible en Library: [round49_uncertainty_consensus_20261025.zip](https://chatgpt.com/library?file_id=libfile_4a128759e8888191af86e1ede299b385&account_id=personal). Incluye código, JSON de métricas y tabla de siete candidatos como **predicciones no verificadas**. Ninguna submission preparada ni enviada.

Con etiquetas de train, acuerdo HistGB/FastText a probabilidad mínima 0,70: 12.543/13.218 (94,89%), 12.442/13.077 (95,14%) y 12.463/13.155 (94,74%) en tres confirm de vendedores reservados. HistGB/CatBoost en nuevos ronda 13: 3.519/3.734 (94,24%). Son grupos amplios, no desacuerdos con la política exacta 41, que no puede reproducirse actualmente en train.

En test, la probabilidad mínima de los tres modelos para la clase propuesta tuvo mediana 0,827 en el bloque 41 (seis cambios, score 0,94628) y 0,504 en el bloque 45 (ocho cambios, score 0,94623). Comparación **post hoc descriptiva**: ni score ni consenso identifican aciertos individuales. Al umbral 0,70 aparecen siete vendedores nuevos no aislados, 4 cambios hacia new y 3 hacia used, disjuntos de 41/43/45. Los siete estaban dentro de la enorme 39 que perdió; su contribución no se puede aislar de ese score. No recomendar subirlos sin confirm con etiquetas de train de la política relevante.

Próximo paso: diseñar una referencia congelada semejante a la 41 que pueda aplicarse en train, o reconstruir su política; medir fixes/breaks de los tres modelos cuando contradicen esa referencia en tune y confirm seller-group, segmentar dirección y tipo de vendedor, bootstrap por vendedor. El análisis no entrenó con test ni gastó cuota Kaggle.
