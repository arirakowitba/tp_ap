# Señal de historial de vendedor frente a baseline dispersa enriquecida

28-10-2026. Código [agent_signal_sparse_seller_20261028.py](../../src/revision_submissions/agent_signal_sparse_seller_20261028.py), checkpoint Library `libfile_e7be123c4f6c8191882442047cc98a05` (incluye `seller_diagnostic.py` dependencia), hashes SHA256 código `68274b8f59550d45de05b7188c5c67c8f53afaef99908ede7c4070f953ab8329`, resultados `62050f11bdb07def5497b27e30c983fd0c365d1964eb4081be8b197484a00fa4`. 70k etiquetas reales train, sin test ni score público como etiquetas.

Hipótesis: comprobar si la señal de condición dominante del vendedor complementa un modelo más fuerte que el título solo de [round139_seller_diagnostic_20261028.md](round139_seller_diagnostic_20261028.md). Baseline: TF-IDF palabras 1–2 y caracteres 3–5 del título, más one-hot categoría/listing/estado/shipping y atributos numéricos precio/cantidades/longitud, regresión logística C=3. Ajuste de vectorizadores y modelo sólo en fit. Misma regla fijada *antes* de confirmar: historial >=3, prior suavizado alpha=3 a distancia >=.30 de .5 y contradicción con baseline; combinación .6 prior vendedor + .4 probabilidad modelo. Dos seeds con splits seller-group nuevos y holdout de filas de vistos, tune/confirm disjuntos dentro de cada seed. No se eligió hiperparámetro con confirm.

| Seed | Parte | Base acc | Regla acc | Gate | Fix/break | Logloss base→regla |
|---:|---|---:|---:|---:|---:|---:|
| 2026 | tune 11.765 | .883383 | .888653 | 127 | 93/31 | .283090→.276804 |
| 2026 | confirm 12.906 | .889199 | .895475 | 135 | 107/26 | .280074→.273654 |
| 314159 | tune 12.473 | .884791 | .891766 | 133 | 109/22 | .278916→.272413 |
| 314159 | confirm 12.844 | .884070 | .890688 | 142 | 113/28 | .292827→.285759 |

Confirm vistos: .90810→.92861 (seed 2026, n=3950) y .90812→.93031 (314159, n=3831); nuevos sin cambio porque carecen de historial en fit. Bootstrap 2.000 remuestreos por vendedor visto, ganancia neta en confirm 95% percentil: seed 2026 +53 a +109; seed 314159 +55 a +119. Los folds comparten filas y vendedores entre semillas; no son evidencia independiente ni se suman. La fracción de nuevos en confirm ronda 69–70%, distinta del test mencionado (~28%). CSV pareados de confirm con IDs, etiquetas reales, probabilidades y gate: SHA256 2026 `b2b36dfbb9f3998eeeff414feb04a7b297a033d238d1c5a02c2ddfdd83f948e4`, 314159 `e86e4e9cede4a90ccd77989abd8b64c8211324b769e762e821f5bbdff7369b6a`.

**Decisión:** confirma el mecanismo frente a un baseline de .88–.89, pero no prueba ganancia sobre HistGB+FastText, CatBoost o la política 48. La selección condicional sobre un baseline débil produce un gate especialmente corregible; su 79–80% accuracy interno no se extrapola a desacuerdos entre modelos fuertes. No generar submission por esta señal aislada. El siguiente experimento debe calcular la misma regla, o una variante congelada en tune, sobre probabilidades fuertes OOF estrictamente externas, comparar pares y logloss en los mismos confirm, y auditar dependencia del prior con el modelo base. No usar datos externos etiquetados.
