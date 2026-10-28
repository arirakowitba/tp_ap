# Ronda 39 — FastText subword y candidato pequeño para revisión

## Representación y validación

Se entrenó FastText subword de 40 dimensiones de modo no supervisado sobre los
`docs` **originales** de los 100.000 avisos del Google Sheet. Los textos de
test se usaron solo para aprender vocabulario y contexto; ninguna etiqueta de
test ni score público entró al entrenamiento. El promedio de vectores de
palabras, ponderado por frecuencia inversa sin supervisión, se añadió a la
segunda etapa HistGB de ronda 30. Se mantuvieron las predicciones de texto
word/char OOF y las variables profundas existentes. Umbral fijo: 0,475.

| Corte seller-group | Filas | Arreglos | Roturas | Neto | Log loss HistGB → FastText |
|---|---:|---:|---:|---:|---:|
| 883 | 14.335 | 109 | 103 | +6 | 0,183874 → 0,183327 |
| 2026 | 13.493 | 97 | 87 | +10 | 0,213349 → 0,208920 |
| 314159 | 13.898 | 126 | 111 | +15 | 0,209811 → 0,205204 |

Los intervalos bootstrap 95% por vendedor del neto incluyen cero en esos
tres cortes: [−23,+35], [−20,+41] y [−17,+47]. Comparten train y fueron
explorados en rondas previas, por lo que no representan tres confirmaciones
independientes.

Se creó un split adicional con un solo fit y grupos de vendedores disjuntos
para tune y confirm. Con 41.680 avisos de fit, el modelo ganó **+44/13.787**
en tune y **+12/14.533** en confirm; el log loss mejoró de 0,208811 a
0,204014 y de 0,212552 a 0,210628. El intervalo bootstrap por vendedor del
neto combinado (+56/28.320) fue [12,103], pero el de confirm por separado
fue [−22,46]. Esto apoya la representación, no una candidata amplia.

## Regla direccional y comprobación posterior

Al inspeccionar los primeros cortes, se observó que el cambio de HistGB
`used` hacia FastText `new` era más prometedor que el inverso. Esa dirección
se **congeló** antes de un segundo split seller-group (fit 41.175; tune
14.335; confirm 14.490). En el primer split disjunto, la regla había dado
+20/116 y +26/114 flips. En el segundo dio **+20/132** en tune y
**+42/136** en confirm (89 arreglos, 47 roturas). El bootstrap por vendedor
de ese último confirm es [17,65]. Como los datasets se reutilizan entre
experimentos, no se trata de evidencia completamente independiente de toda
la exploración histórica; sí es una comprobación adicional con la regla
fijada antes de ver sus etiquetas de validación.

## Refit full70k y candidata de revisión

Se guardaron probabilidades full70k para 30.000 filas, alineadas con
`kaggle_ID=1..30000`. FastText cambia 404 decisiones contra HistGB de ronda
30, 176 en vendedores nuevos. Difiere de la submission exacta 33 en 1.357
decisiones; **esas diferencias no son ganancias conocidas**.

La regla congelada sobre test produce 37 avisos nuevos donde HistGB y la 33
dicen `used` y FastText dice `new`. Se excluyeron los IDs 7089, 24522 y
26141 porque ya estaban en bloques públicos 36–41. Quedó una candidata de
**34 cambios `used→new`, todos vendedores nuevos**, con base 33 exacta y
ningún otro cambio. El archivo
`candidate_fasttext_34_new_seller_toward_new_vs33_REVIEW_ONLY.csv` tiene
30.000 filas y SHA256
`8df48f148afcb9f432c9e8bf51f967031b358e8b84ff33f74e7f9929255980c6`.
IDs y regla completa constan en `round39_candidate_review_manifest.json`.

**Estado:** solo para revisión. No se generó una submission numerada ni se
subió a Kaggle. La validación pareada mide una corrección contra HistGB,
mientras que la 33 tiene una política distinta no reproducida en train. Por
eso el +42/136 de confirm no es una estimación válida del efecto de estos
34 flips sobre la score pública de la 33. Además, una prueba pequeña no
acerca por sí sola la brecha grande al líder; sirve para medir transferencia
de la nueva señal sin gastar un cupo en cambios masivos.

## Reproducción

Python 3.12, gensim 4.4.0, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.2.3.
El checkpoint incluye `original_docs.pkl`, el embedding exacto de 100.000
filas, fuentes, vectores pareados, predicciones de test y candidata. Para
repetir la extracción desde JSONL, utilizar `data.zip` con SHA256
`f8e78574061cbd75197896fa1cd08a0c78af3cb5d73e7e2597df4ab43b82d574`
del checkpoint de ronda 37. Ejecutar, en orden:

```
python work20/round39_fasttext_embeddings.py
python work20/round39_group_fasttext_hist.py 883
python work20/round39_group_fasttext_hist.py 2026
python work20/round39_group_fasttext_hist.py 314159
python work20/round39_disjoint_holdout.py
AP_HOLDOUT_BASE_SEED=39011 AP_HOLDOUT_NAME=confirm2 python work20/round39_disjoint_holdout.py
python work20/round39_full_fasttext_hist.py
python work20/round39_prepare_review_candidate.py
```

FastText se entrenó con dos workers: diferencias mínimas de redondeo pueden
ocurrir al repetirlo. El checkpoint conserva los embeddings y las
predicciones exactas auditadas, además de un manifiesto SHA256.
