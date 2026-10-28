# Ronda 64 — vecinos FastText para vendedores nuevos (26/10/2026)

Hipótesis: las etiquetas de avisos de entrenamiento cercanos en el espacio FastText original de 40 dimensiones agregan una señal local a HistGB+FastText en vendedores no vistos. No se usan etiquetas de test ni scores públicos como etiquetas.

## Insumos y protocolo

- Datos reales: `data.zip` con 70.000 train etiquetados, `original_docs.pkl` con 100.000 textos y embeddings `round39_fasttext_100k.npy` de `round39_fasttext_checkpoint_20261025.zip`.
- Se conservan los dos cortes seller-group disjuntos dentro de cada partición de ronda 39 (`a` y `b`). En cada corte, los vecinos provienen solo de `fit`; `tune` y `confirm` no contribuyen etiquetas al posterior. Los sellers de fit, tune y confirm son disjuntos dentro de cada corte. Los dos cortes se solapan entre sí y ya habían sido utilizados; estos confirm no son independientes del historial.
- Posterior de vecinos coseno: k=20 o 100, peso similitud positiva a la cuarta, con suavizado de dos etiquetas según prevalencia en fit. Se probaron pesos logit 0,10/0,25/0,50 sobre HistGB+FastText y el umbral fijo 0,475. La combinación k=20/peso=0,10 se seleccionó por saldo de desacuerdos en tune del corte a y se congeló para los diagnósticos confirm.
- Referencia: `p_fasttext` de la ronda 39. Esta no reproduce la política pública exacta 41; no se infiere rendimiento contra 41.

| Corte | Segmento | Filas | Cambios | Correcciones | Roturas | Neto | IC 95% neto por bootstrap de vendedores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a | tune total | 13.787 | 45 | — | — | +7 | — |
| a | confirm total | 14.533 | 42 | 19 | 23 | −4 | [−17, +8] |
| b | tune total | 14.335 | 54 | — | — | 0 | — |
| b | confirm total | 14.490 | 57 | 26 | 31 | −5 | [−23, +11] |

En los desacuerdos HistGB/FastText, el neto confirm fue +6 en 14 cambios del corte a y 0 en 24 cambios del b. Por dirección, a: used→new 18 cambios/neto 0, new→used 24/neto −4; b: used→new 27/neto −1, new→used 30/neto −4. El log loss confirm de la mezcla cambia −0,00036 en a y +0,00094 en b frente a la referencia. Pesos 0,25 y 0,50 fueron claramente peores en b confirm (−31 y −135 aciertos con k=20).

Conclusión: la señal local no aporta accuracy estable, ni incluso en los desacuerdos. No preparar submission con ella. Los intervalos por vendedor incluyen cero y los cortes ya fueron reutilizados; no constituyen confirmación nueva. La submission 46 de siete casos se preparó por separado y sigue con score no informado en este registro. Tampoco se deducen etiquetas individuales de ella.

Checkpoint reproducible: `round64_fasttext_neighbors_checkpoint_20261026.zip`, Library ID `libfile_d2ab8d23a8d08191a3b4e4b734c335d4`, SHA256 `17074f23922a43eb06bdc04b354a84bb12eb6afa240e10ed8a800467a49ef334`. Contiene código, versiones, predicciones pareadas por fila, posterior k=20/100, métricas y hashes. Requiere los archivos originales ya conservados en el checkpoint de ronda 39 y `data.zip`.
